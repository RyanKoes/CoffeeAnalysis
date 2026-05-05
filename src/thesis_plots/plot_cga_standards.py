from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
	import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
except ImportError as exc:  # pragma: no cover
	raise ImportError(
		"ThesisPlotGeneration.plot_cga_standards requires 'matplotlib'. "
		"Install it into your active environment (e.g., pip install matplotlib)."
	) from exc

# Allow imports from repo root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ThesisPlotGeneration.plot_generator import (  # noqa: E402
	FIGSIZE_SINGLE_COLUMN,
	apply_publication_style,
	savefig_pdf,
)


@dataclass(frozen=True)
class StandardTrace:
	ppm: float
	voltage_v: np.ndarray
	current: np.ndarray


_PPM_RE = re.compile(r"(?P<ppm>\d+(?:\.\d+)?)\s*ppm", re.IGNORECASE)


def _parse_ppm_from_filename(path: Path) -> float:
	"""Parse concentration (ppm) from a standards filename.

	Supports patterns like:
	- CGA100.txt -> 100
	- 100ppm.txt -> 100
	- CGA_50ppm_run1.txt -> 50
	"""
	match = _PPM_RE.search(path.stem)
	if match:
		return float(match.group("ppm"))
	# Fallback: extract first number anywhere in the stem (e.g., CGA100 -> 100)
	fallback = re.search(r"(\d+(?:\.\d+)?)", path.stem)
	if fallback:
		return float(fallback.group(1))

	raise ValueError(f"Could not parse ppm from filename: {path.name}")


def _read_standard_file(path: Path) -> StandardTrace:
	"""Read a standards trace file.

	Expected format: 3 columns (time, voltage, current) comma-separated.
	We plot current vs voltage.
	"""
	arr = np.genfromtxt(path, delimiter=",")
	if arr.ndim != 2 or arr.shape[1] < 3:
		raise ValueError(f"Unexpected file format for {path}: expected 3 columns")

	voltage_v = arr[:, 1].astype(float)
	current = arr[:, 2].astype(float)

	# Ensure a monotonic x-axis for plotting (some exports can jitter)
	order = np.argsort(voltage_v)
	voltage_v = voltage_v[order]
	current = current[order]

	return StandardTrace(ppm=_parse_ppm_from_filename(path), voltage_v=voltage_v, current=current)


def _moving_average_centered(y: np.ndarray, window: int) -> tuple[np.ndarray, int, int]:
	"""Centered moving average via convolution.

	Returns:
		y_ma: smoothed values (length n-window+1)
		start_idx, end_idx: slice indices for aligning x[start_idx:end_idx]
	"""
	if window <= 1:
		return y.copy(), 0, y.size
	if window > y.size:
		raise ValueError(f"Moving average window {window} > data length {y.size}")

	kernel = np.ones(int(window), dtype=float) / float(window)
	y_ma = np.convolve(y, kernel, mode="valid")

	start_idx = int(window) // 2
	end_idx = start_idx + y_ma.size
	return y_ma, start_idx, end_idx


def plot_cga_standards(
	*, standards_dir: Path, out_path: Path, window: int = 20
) -> None:
	files = sorted(standards_dir.glob("*.txt"))
	if not files:
		raise FileNotFoundError(f"No .txt files found in {standards_dir}")

	traces: list[StandardTrace] = [_read_standard_file(p) for p in files]
	traces.sort(key=lambda t: t.ppm)

	# Taller-than-default single-column figure (but not a perfect square).
	w0, h0 = FIGSIZE_SINGLE_COLUMN
	apply_publication_style(figsize=(float(w0), float(h0) * 1.25))
	fig, ax = plt.subplots()

	# Use a qualitative palette for more distinct colors.
	cmap = plt.get_cmap("tab10")
	colors = [cmap(i % 10) for i in range(len(traces))]
	all_y: list[np.ndarray] = []

	for color, trace in zip(colors, traces, strict=False):
		y_ma, start, end = _moving_average_centered(trace.current, window)
		x = trace.voltage_v[start:end]
		ax.plot(x, y_ma, linewidth=1.25, color=color, label=f"{trace.ppm:g}")
		all_y.append(y_ma)

	ax.set_xlabel("Potential (V)")
	ax.set_ylabel("Current (µA)")
	ax.set_title("CGA Standards")
	ax.grid(True, alpha=0.25)
	ax.set_xlim(0.0, 2.0)

	# Cap the y-axis top at 200 (matching caffeine plot) while keeping the lower limit data-driven.
	if all_y:
		all_y_concat = np.concatenate(all_y)
		y_min = float(np.nanmin(all_y_concat))
		y_max = 200.0
		pad = 0.05 * (y_max - y_min) if np.isfinite(y_min) and y_max > y_min else 0.0
		ax.set_ylim(y_min - pad, y_max)
	else:
		ax.set_ylim(top=200.0)

	ax.legend(title="CGA (ppm)", frameon=False, ncol=2)
	plt.tight_layout()

	out_path.parent.mkdir(parents=True, exist_ok=True)
	savefig_pdf(str(out_path))
	plt.close(fig)
	print(f"Saved plot to {out_path}")


def _build_argparser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Plot CGA standards from ThesisPlotGeneration/cga_standards and save as PDF."
	)
	parser.add_argument(
		"--standards-dir",
		type=Path,
		default=Path(__file__).resolve().parent / "cga_standards",
		help="Directory containing CGA standards files (default: ThesisPlotGeneration/cga_standards)",
	)
	parser.add_argument(
		"--window",
		type=int,
		default=50,
		help="Moving average window size in points (default: 20)",
	)
	parser.add_argument(
		"--out",
		type=Path,
		default=Path(__file__).resolve().parent / "plots" / "cga_standards.pdf",
		help="Output PDF path (default: ThesisPlotGeneration/plots/cga_standards.pdf)",
	)
	return parser


def main() -> None:
	args = _build_argparser().parse_args()
	plot_cga_standards(standards_dir=args.standards_dir, out_path=args.out, window=args.window)


if __name__ == "__main__":
	main()
