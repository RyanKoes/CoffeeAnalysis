from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
	import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
except ImportError as exc:  # pragma: no cover
	raise ImportError(
		"ThesisPlotGeneration.sample_repeats requires 'matplotlib'. "
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
class SampleTrace:
	name: str
	voltage_v: np.ndarray
	current: np.ndarray


def _moving_window_mean(values: list[float], window: int) -> list[float]:
	"""Centered moving average with boundary-aware shrinking windows.

	This matches the helper used in ThesisPlotGeneration/plotting_notebook.ipynb.
	It keeps the output length unchanged.
	"""
	if window <= 1:
		return list(values)

	n = len(values)
	if n == 0:
		return []

	left = int(window) // 2
	right = int(window) - left

	out: list[float] = []
	for i in range(n):
		start = max(0, i - left)
		end = min(n, i + right)  # exclusive
		segment = values[start:end]
		out.append(sum(segment) / float(len(segment)))

	return out


def _oxidation_segment(potential_v: np.ndarray, current_ua: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Return oxidation sweep (initial up-scan): start -> first max potential.

	Matches the logic used in the plotting notebook.
	"""
	if potential_v.size == 0:
		return potential_v, current_ua
	if current_ua.size != potential_v.size:
		raise ValueError(f"Potential/current length mismatch: {potential_v.size} vs {current_ua.size}")

	i_max = int(np.argmax(potential_v))
	if i_max == 0 and potential_v.size > 1:
		# likely reversed ordering
		potential_v = potential_v[::-1]
		current_ua = current_ua[::-1]
		i_max = int(np.argmax(potential_v))

	return potential_v[: i_max + 1], current_ua[: i_max + 1]


def _nearest_index(values: np.ndarray, target: float) -> int:
	return int(np.argmin(np.abs(values - float(target))))


def _feature_plateau_lowest_slope(
	potential_ox: np.ndarray,
	current_ox: np.ndarray,
	*,
	vmin: float = 1.28,
	vmax: float = 1.6,
) -> tuple[float, float] | None:
	"""Plateau feature: lowest |dI/dV| point in [vmin, vmax] on oxidation sweep.

	Returns (v_pick, i_pick) or None if insufficient points.
	"""
	mask = (potential_ox >= float(vmin)) & (potential_ox <= float(vmax))
	if int(np.sum(mask)) < 3:
		return None
	v_win = potential_ox[mask]
	i_win = current_ox[mask]
	slope = np.gradient(i_win, v_win)
	pick = int(np.argmin(np.abs(slope)))
	return float(v_win[pick]), float(i_win[pick])


def _interp_y_at_x(x: np.ndarray, y: np.ndarray, x0: float) -> float:
	"""Linear interpolation helper for monotonic x."""
	x_arr = np.asarray(x, dtype=float)
	y_arr = np.asarray(y, dtype=float)
	if x_arr.size == 0:
		return float("nan")
	if x0 < float(x_arr[0]) or x0 > float(x_arr[-1]):
		return float("nan")
	return float(np.interp(float(x0), x_arr, y_arr))


def _percent_rsd(values: np.ndarray) -> float:
	"""Percent relative standard deviation (%RSD).

	Uses sample std dev (ddof=1) when n>=2.
	"""
	arr = np.asarray(values, dtype=float)
	arr = arr[np.isfinite(arr)]
	if arr.size < 2:
		return float("nan")
	mean = float(np.mean(arr))
	std = float(np.std(arr, ddof=1))
	denom = abs(mean)
	if denom == 0.0 or not np.isfinite(denom):
		return float("nan")
	return 100.0 * std / denom


def _read_sample_file(path: Path) -> SampleTrace:
	"""Read a repeat-sample trace file.

	This follows the conventions used in the standards plotting scripts:
	- Expected common format: 3 columns (time, voltage, current)
	- Plot current vs voltage

	We allow a couple of variations to be robust to exports:
	- 2 columns: (voltage, current)
	- >=3 columns: (time, voltage, current, ...)
	"""

	# First try comma-delimited (Metrohm export), then fall back to whitespace.
	arr = np.genfromtxt(path, delimiter=",")
	if not (isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] >= 2):
		arr = np.genfromtxt(path, delimiter=None)

	if not (isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] >= 2):
		raise ValueError(
			f"Unexpected file format for {path}: expected at least 2 numeric columns; got shape {getattr(arr, 'shape', None)}"
		)

	# Handle common export shapes.
	if arr.shape[1] >= 3:
		voltage_v = arr[:, 1].astype(float)
		current = arr[:, 2].astype(float)
	else:
		voltage_v = arr[:, 0].astype(float)
		current = arr[:, 1].astype(float)

	return SampleTrace(name=path.stem, voltage_v=voltage_v, current=current)


def plot_sample_repeats(
	*,
	samples_dir: Path,
	out_path: Path,
	window: int = 20,
	label_prefix: str = "Starbucks",
) -> None:
	# Common extensions; if none match we fall back to "any file".
	files_set: set[Path] = set()
	for pattern in ("*.txt", "*.csv", "*.dat"):
		files_set.update(samples_dir.glob(pattern))
	files = sorted(files_set)
	if not files:
		files = sorted([p for p in samples_dir.iterdir() if p.is_file() and not p.name.startswith(".")])

	if not files:
		raise FileNotFoundError(
			f"No sample files found in {samples_dir}. Put your 5 repeat trace files in ThesisPlotGeneration/same_samples/ "
			"(e.g., .txt or .csv exports) and re-run."
		)

	traces: list[SampleTrace] = [_read_sample_file(p) for p in files]

	# Taller-than-default single-column figure (matches standards plots).
	w0, h0 = FIGSIZE_SINGLE_COLUMN
	apply_publication_style(figsize=(float(w0), float(h0) * 1.25))
	fig, ax = plt.subplots()

	# Use a qualitative palette for more distinct colors.
	cmap = plt.get_cmap("tab10")
	all_y: list[np.ndarray] = []

	# %RSD targets (in volts) for "current-at-voltage" reporting
	target_voltages = (0.8, 1.3)
	values_by_voltage: dict[float, list[tuple[str, float]]] = {v: [] for v in target_voltages}

	# Plateau extraction (%RSD across extracted plateau currents)
	plateau_points: list[tuple[str, float, float]] = []  # (label, v_pick, i_pick)

	for i, trace in enumerate(traces):
		color = cmap(i % 10)
		label = f"{label_prefix} {i + 1}"

		# Restrict to thesis voltage window: 0–2 V (keep scan order).
		mask_0_2 = (trace.voltage_v >= 0.0) & (trace.voltage_v <= 2.0)
		voltage_0_2 = trace.voltage_v[mask_0_2]
		current_0_2 = trace.current[mask_0_2]
		if voltage_0_2.size == 0:
			print(f"Warning: {trace.name} has no points in 0–2 V; skipping {label}.")
			continue

		# Smooth using the same moving-window mean as the plotting notebook.
		current_sm = np.asarray(_moving_window_mean(list(map(float, current_0_2)), window), dtype=float)
		ax.plot(voltage_0_2, current_sm, linewidth=1.25, color=color, alpha=0.65, label=label)
		all_y.append(current_sm)

		# Notebook-style feature extraction operates on the oxidation sweep only.
		potential_arr = np.asarray(voltage_0_2, dtype=float)
		potential_ox, current_ox = _oxidation_segment(potential_arr, current_sm)

		# Current-at-voltage features (nearest point on oxidation sweep)
		for v in target_voltages:
			idx = _nearest_index(potential_ox, float(v))
			values_by_voltage[float(v)].append((label, float(current_ox[idx])))

		# Plateau extraction (lowest |dI/dV| between 1.28–1.6 V on oxidation sweep)
		plateau = _feature_plateau_lowest_slope(potential_ox, current_ox, vmin=1.28, vmax=1.6)
		if plateau is None:
			print(f"Warning: {trace.name} has insufficient points in plateau window; skipping plateau for {label}.")
		else:
			v_pick, i_pick = plateau
			plateau_points.append((label, v_pick, i_pick))

	ax.set_xlabel("Potential (V)")
	ax.set_ylabel("Current (µA)")
	ax.set_title("Sample repeats")
	ax.grid(True, alpha=0.25)
	ax.set_xlim(0.0, 2.0)

	# Keep y-limits data-driven but consistent.
	if all_y:
		all_y_concat = np.concatenate(all_y)
		y_min = float(np.nanmin(all_y_concat))
		y_max = float(np.nanmax(all_y_concat))
		if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
			pad = 0.05 * (y_max - y_min)
			ax.set_ylim(y_min - pad, y_max + pad)

	ax.legend(title=label_prefix, frameon=False)
	plt.tight_layout()

	out_path.parent.mkdir(parents=True, exist_ok=True)
	savefig_pdf(str(out_path))
	plt.close(fig)
	print(f"Saved plot to {out_path}")

	# Print %RSDs across repeats at the requested voltages.
	print("\n%RSD across repeats (oxidation sweep, notebook-style smoothing)")
	for v in target_voltages:
		pairs = values_by_voltage.get(float(v), [])
		vals = np.array([p[1] for p in pairs], dtype=float)
		labels = [p[0] for p in pairs]
		if len(labels) != 3:
			print(f"Note: computed using {len(labels)} traces (expected 3).")
		rsd = _percent_rsd(vals)
		mean = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
		std = float(np.nanstd(vals, ddof=1)) if np.sum(np.isfinite(vals)) >= 2 else float("nan")
		print(f"- {v:.1f} V: mean={mean:.6g} µA, std={std:.6g} µA, %RSD={rsd:.4g}")
		for lbl, val in zip(labels, vals, strict=False):
			print(f"  - {lbl}: {val:.6g} µA")

	# Print plateau %RSD across files
	print("\nPlateau extraction (lowest |dI/dV| in 1.28–1.6 V on oxidation sweep)")
	if plateau_points:
		p_labels = [p[0] for p in plateau_points]
		p_vs = np.array([p[1] for p in plateau_points], dtype=float)
		p_is = np.array([p[2] for p in plateau_points], dtype=float)
		if len(p_labels) != 5:
			print(f"Note: plateau %RSD computed using {len(p_labels)} files (expected 5).")
		rsd_p = _percent_rsd(p_is)
		mean_p = float(np.nanmean(p_is)) if np.isfinite(p_is).any() else float("nan")
		std_p = float(np.nanstd(p_is, ddof=1)) if np.sum(np.isfinite(p_is)) >= 2 else float("nan")
		print(f"- Plateau current: mean={mean_p:.6g} µA, std={std_p:.6g} µA, %RSD={rsd_p:.4g}")
		for lbl, v_pick, i_pick in zip(p_labels, p_vs, p_is, strict=False):
			print(f"  - {lbl}: V={v_pick:.6g} V, I={i_pick:.6g} µA")
	else:
		print("No usable plateau points extracted.")


def _build_argparser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Plot repeated measurements from ThesisPlotGeneration/same_samples and save as a thesis-style PDF."
	)
	parser.add_argument(
		"--samples-dir",
		type=Path,
		default=Path(__file__).resolve().parent / "same_samples",
		help="Directory containing repeat trace files (default: ThesisPlotGeneration/same_samples)",
	)
	parser.add_argument(
		"--window",
		type=int,
		default=20,
		help="Moving average window size in points (default: 20)",
	)
	parser.add_argument(
		"--label-prefix",
		type=str,
		default="Starbucks",
		help="Legend label prefix (default: Starbucks -> Starbucks 1, Starbucks 2, ...)",
	)
	parser.add_argument(
		"--out",
		type=Path,
		default=Path(__file__).resolve().parent / "plots" / "sample_repeats.pdf",
		help="Output PDF path (default: ThesisPlotGeneration/plots/sample_repeats.pdf)",
	)
	return parser


def main() -> None:
	args = _build_argparser().parse_args()
	plot_sample_repeats(
		samples_dir=args.samples_dir,
		out_path=args.out,
		window=args.window,
		label_prefix=args.label_prefix,
	)


if __name__ == "__main__":
	main()
