from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

try:
	import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
except ImportError as exc:  # pragma: no cover
	raise ImportError(
		"ThesisPlotGeneration.print_best_tds_results requires 'matplotlib'. "
		"Install it into your active environment (e.g., pip install matplotlib)."
	) from exc

# Allow imports from repo root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from thesis_plots.plot_generator import (  # noqa: E402
	FIGSIZE_SINGLE_COLUMN,
	apply_publication_style,
	savefig_pdf,
)


DEFAULT_PKL = Path(__file__).resolve().parent / "data" / "BEST_TDS_fixed.pkl"


def _as_1d(x) -> np.ndarray:
	arr = np.asarray(x)
	if arr.ndim == 2 and arr.shape[1] == 1:
		arr = arr[:, 0]
	return arr.astype(float).reshape(-1)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	y_true = _as_1d(y_true)
	y_pred = _as_1d(y_pred)
	ss_res = float(np.sum((y_true - y_pred) ** 2))
	ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
	if ss_tot == 0.0:
		return float("nan")
	return 1.0 - ss_res / ss_tot


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	y_true = _as_1d(y_true)
	y_pred = _as_1d(y_pred)
	return float(np.mean(np.abs(y_true - y_pred)))


def _plot_parity(*, actual: np.ndarray, pred: np.ndarray, title: str, out_path: Path) -> None:
	w, h = FIGSIZE_SINGLE_COLUMN
	figsize = (w, h * 1.3)
	fig, ax = plt.subplots(figsize=figsize)

	ax.scatter(actual, pred, s=18, alpha=0.8, zorder=2)

	min_val = float(min(np.min(actual), np.min(pred)))
	max_val = float(max(np.max(actual), np.max(pred)))
	rng = max_val - min_val
	plot_min = min_val - 0.05 * rng
	plot_max = max_val + 0.05 * rng

	ax.plot(
		[plot_min, plot_max],
		[plot_min, plot_max],
		color="black",
		linestyle="--",
		linewidth=1.6,
		zorder=3,
		alpha=0.5,
	)

	ax.set_xlabel("Actual (%)")
	ax.set_ylabel("Predicted (%)")
	ax.set_title(title)

	plt.tight_layout()
	savefig_pdf(str(out_path))
	plt.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Plot and print a summary of the saved BEST_TDS_fixed.pkl checkpoint (metadata + CV parity metrics)."
		)
	)
	parser.add_argument(
		"--pkl",
		type=Path,
		default=DEFAULT_PKL,
		help="Path to BEST_TDS_fixed.pkl (default: ThesisPlotGeneration/data/BEST_TDS_fixed.pkl)",
	)
	args = parser.parse_args()

	apply_publication_style(figsize=(3.35, 3.35))

	pkl_path: Path = args.pkl
	if not pkl_path.exists():
		raise FileNotFoundError(f"Could not find {pkl_path}")

	data = torch.load(pkl_path, map_location=torch.device("cpu"), weights_only=False)
	if not isinstance(data, dict):
		raise TypeError(f"Expected dict from torch.load({pkl_path}), got {type(data)}")

	metadata = data.get("metadata")
	if not isinstance(metadata, dict):
		metadata = {}

	if "cv_test_actual" not in data or "cv_test_pred" not in data:
		raise ValueError(
			"Expected keys 'cv_test_actual' and 'cv_test_pred' in the PKL. "
			"This script summarizes the saved CV parity points."
		)

	actual = _as_1d(data["cv_test_actual"])
	pred = _as_1d(data["cv_test_pred"])
	if actual.shape != pred.shape:
		raise ValueError(f"Shape mismatch: cv_test_actual {actual.shape} vs cv_test_pred {pred.shape}")

	err = pred - actual

	target = metadata.get("target", "TDS")
	arch = metadata.get("architecture", "<unknown>")
	window = metadata.get("window", "<unknown>")
	stored_r2 = metadata.get("r2")

	computed_r2 = _r2_score(actual, pred)
	computed_mae = _mae(actual, pred)
	rmse = float(np.sqrt(np.mean(err**2)))
	bias = float(np.mean(err))
	err_std = float(np.std(err))

	plots_dir = Path(__file__).resolve().parent / "plots"
	plots_dir.mkdir(exist_ok=True)

	out_pdf = plots_dir / "scatter_TDS.pdf"
	_plot_parity(actual=actual, pred=pred, title="TDS", out_path=out_pdf)

	stats_path = plots_dir / "tds_model_stats.txt"
	with stats_path.open("w", encoding="utf-8") as f:
		f.write(f"pkl: {pkl_path.name}\n")
		f.write("title: TDS\n")
		if arch != "<unknown>":
			f.write(f"architecture: {arch}\n")
		f.write(f"window: {window}\n")
		f.write(f"n_samples: {len(actual)}\n")
		if stored_r2 is not None:
			f.write(f"stored_r2: {float(stored_r2):.6f}\n")
		f.write(f"r2: {computed_r2:.6f}\n")
		f.write(f"mae: {computed_mae:.6f}\n")
		f.write(f"rmse: {rmse:.6f}\n")
		f.write(f"bias(pred-actual): {bias:.6f}\n")
		f.write(f"residual_std: {err_std:.6f}\n")
		f.write(f"actual_range: [{float(np.min(actual)):.6f}, {float(np.max(actual)):.6f}]\n")
		f.write(f"pred_range:   [{float(np.min(pred)):.6f}, {float(np.max(pred)):.6f}]\n")
		f.write(f"plot_pdf: {out_pdf.name}\n")

	print("BEST_TDS_fixed summary")
	print("---------------------")
	print(f"pkl: {pkl_path}")
	print(f"target: {target}")
	print(f"architecture: {arch}")
	print(f"window: {window}")
	print(f"n_samples: {len(actual)}")
	if stored_r2 is not None:
		print(f"stored_r2: {float(stored_r2):.6f}")
	print(f"computed_r2: {computed_r2:.6f}")
	print(f"mae: {computed_mae:.6f}")
	print(f"rmse: {rmse:.6f}")
	print(f"bias(pred-actual): {bias:.6f}")
	print(f"residual_std: {err_std:.6f}")
	print(f"actual_range: [{float(np.min(actual)):.6f}, {float(np.max(actual)):.6f}]")
	print(f"pred_range:   [{float(np.min(pred)):.6f}, {float(np.max(pred)):.6f}]")
	print(f"Saved: {out_pdf}")
	print(f"Wrote stats: {stats_path}")


if __name__ == "__main__":
	main()
