from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score

try:
	import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
	from matplotlib.ticker import MultipleLocator  # type: ignore[reportMissingImports]
except ImportError as exc:  # pragma: no cover
	raise ImportError(
		"ThesisPlotGeneration.plot_best_hplc_caff requires 'matplotlib'. "
		"Install it into your active environment (e.g., pip install matplotlib)."
	) from exc

# Allow imports from repo root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from thesis_plots.plot_generator import (  # noqa: E402
	FIGSIZE_SINGLE_COLUMN,
	apply_publication_style,
	savefig_pdf,
)


DEFAULT_PKL = Path(__file__).resolve().parent / "data" / "BEST_HPLC_Caff_fixed.pkl"


def _as_1d(x) -> np.ndarray:
	arr = np.asarray(x)
	if arr.ndim == 2 and arr.shape[1] == 1:
		arr = arr[:, 0]
	return arr.astype(float).reshape(-1)


def _nice_tick_step(data_range: float, *, n_ticks: int = 5) -> float:
	"""Return a "nice" major tick step for a given axis range.

	The step is chosen from {1, 2, 2.5, 5, 10} * 10^k so plots land on
	clean increments (e.g., 0.25, 2.5, 25, 250, ...).
	"""
	if not np.isfinite(data_range) or data_range <= 0:
		return 1.0
	if n_ticks < 2:
		n_ticks = 2

	rough = float(data_range) / float(n_ticks - 1)
	exponent = 10.0 ** np.floor(np.log10(rough))
	fraction = rough / exponent

	if fraction <= 1.0:
		nice_fraction = 1.0
	elif fraction <= 2.0:
		nice_fraction = 2.0
	elif fraction <= 2.5:
		nice_fraction = 2.5
	elif fraction <= 5.0:
		nice_fraction = 5.0
	else:
		nice_fraction = 10.0

	step = nice_fraction * exponent
	# Snap tiny floating error (e.g. 249.999999999 -> 250.0)
	if step >= 1:
		step = float(np.round(step, decimals=10))
	return float(step)


def _infer_tick_step_from_ticks(ticks: np.ndarray) -> float | None:
	"""Infer a major tick step from an array of tick locations."""
	arr = np.asarray(ticks, dtype=float)
	if arr.size < 2:
		return None

	diffs = np.diff(arr)
	# Keep positive finite diffs only (ignore any weird ordering or duplicates)
	diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
	if diffs.size == 0:
		return None

	# Typical locators produce constant diffs; median is robust to edge cases.
	step = float(np.median(diffs))
	if not np.isfinite(step) or step <= 0:
		return None
	return float(np.round(step, decimals=12))


def _plot_parity(*, actual: np.ndarray, pred: np.ndarray, title: str, out_path: Path) -> tuple[float, float]:
	w, h = FIGSIZE_SINGLE_COLUMN
	figsize = (w, h * 1.3)
	fig, ax = plt.subplots(figsize=figsize)

	ax.scatter(actual, pred, s=18, alpha=0.8, zorder=2)

	# Fit axis limits to the data but keep major ticks on a 400-ppm grid.
	# We *do not* snap the axis max to the next 400 multiple, because that can
	# unnecessarily jump the max to 1600 when the data is only slightly >1200.
	step = 400.0
	cap_max = 1200.0
	max_headroom = 0.25 * step  # small extra space above 1200 without adding a 1600 tick

	min_val = float(min(np.min(actual), np.min(pred)))
	max_val = float(max(np.max(actual), np.max(pred)))
	data_range = max(max_val - min_val, 1.0)
	pad = 0.05 * data_range

	# PPM should not go negative; keep the view anchored at 0.
	plot_min = 0.0
	plot_max = max_val + pad

	# If everything is within 0–1200, keep the top near 1200 with a little headroom.
	if plot_max <= cap_max:
		plot_max = min(cap_max + max_headroom, cap_max + (cap_max - plot_min) * 0.08)
		plot_max = max(plot_max, cap_max)  # ensure the 1200 tick is visible

	# Ensure there's at least one major tick interval visible.
	if plot_max - plot_min < step:
		plot_max = plot_min + step

	ax.set_xlim(plot_min, plot_max)
	ax.set_ylim(plot_min, plot_max)

	ax.plot(
		[plot_min, plot_max],
		[plot_min, plot_max],
		color="black",
		linestyle="--",
		linewidth=1.6,
		zorder=3,
		alpha=0.5,
	)

	r2 = float(r2_score(actual, pred))
	mae = float(mean_absolute_error(actual, pred))

	ax.set_xlabel("Actual (ppm)")
	ax.set_ylabel("Predicted (ppm)")
	ax.set_title(title)

	# Major ticks: every 400 on both axes.
	ax.xaxis.set_major_locator(MultipleLocator(step))
	ax.yaxis.set_major_locator(MultipleLocator(step))

	plt.tight_layout()
	savefig_pdf(str(out_path))
	plt.close(fig)
	return r2, mae


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate thesis-style HPLC Caff parity plot from saved CV points.")
	parser.add_argument(
		"--pkl",
		type=Path,
		default=DEFAULT_PKL,
		help="Path to BEST_HPLC_Caff_fixed.pkl (defaults to ThesisPlotGeneration/data/BEST_HPLC_Caff_fixed.pkl)",
	)
	args = parser.parse_args()

	apply_publication_style(figsize=(3.35, 3.35))

	pkl_path: Path = args.pkl
	if not pkl_path.exists():
		raise FileNotFoundError(f"Could not find {pkl_path}")

	data = torch.load(pkl_path, map_location=torch.device("cpu"), weights_only=False)
	metadata = data.get("metadata")
	if not isinstance(metadata, dict):
		metadata = {}

	if "cv_test_actual" not in data or "cv_test_pred" not in data:
		raise ValueError(
			"Expected keys 'cv_test_actual' and 'cv_test_pred' in the PKL. "
			"This script plots the saved CV parity points."
		)

	actual = _as_1d(data["cv_test_actual"])
	pred = _as_1d(data["cv_test_pred"])
	if actual.shape != pred.shape:
		raise ValueError(f"Shape mismatch: cv_test_actual {actual.shape} vs cv_test_pred {pred.shape}")

	title = "Caffeine"

	plots_dir = Path(__file__).resolve().parent / "plots"
	plots_dir.mkdir(exist_ok=True)

	out_pdf = plots_dir / "scatter_Caffeine.pdf"
	r2, mae = _plot_parity(actual=actual, pred=pred, title=title, out_path=out_pdf)

	stats_path = plots_dir / "caffeine_model_stats.txt"
	with stats_path.open("w", encoding="utf-8") as f:
		f.write(f"pkl: {pkl_path.name}\n")
		f.write(f"title: {title}\n")
		if "architecture" in metadata:
			f.write(f"architecture: {metadata.get('architecture')}\n")
		if "window" in metadata:
			f.write(f"window: {metadata.get('window')}\n")
		f.write(f"n_samples: {len(actual)}\n")
		f.write(f"r2: {r2:.6f}\n")
		f.write(f"mae: {mae:.6f}\n")
		f.write(f"plot_pdf: {out_pdf.name}\n")

	print(f"Saved: {out_pdf}")
	print(f"Wrote stats: {stats_path}")


if __name__ == "__main__":
	main()
