from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
	import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
except ImportError as exc:  # pragma: no cover
	raise ImportError(
		"ThesisPlotGeneration.plot_best_roast requires 'matplotlib'. "
		"Install it into your active environment (e.g., pip install matplotlib)."
	) from exc

# Allow imports from repo root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ThesisPlotGeneration.plot_generator import (  # noqa: E402
	FIGSIZE_SINGLE_COLUMN,
	apply_publication_style,
	savefig_pdf,
)


DEFAULT_PKL = Path(__file__).resolve().parent / "data" / "BEST_Roast_fixed.pkl"


def _as_1d(x) -> np.ndarray:
	arr = np.asarray(x)
	if arr.ndim == 2 and arr.shape[1] == 1:
		arr = arr[:, 0]
	return arr.astype(float).reshape(-1)


def _plot_parity(*, actual: np.ndarray, pred: np.ndarray, title: str, out_path: Path) -> tuple[float, float]:
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

	r2 = float(r2_score(actual, pred))
	mae = float(mean_absolute_error(actual, pred))

	ax.set_xlabel("Actual Roast (%)")
	ax.set_ylabel("Predicted Roast (%)")
	ax.set_title(title)

	plt.tight_layout()
	savefig_pdf(str(out_path))
	plt.close(fig)
	return r2, mae


def _find_architecture_builder(architecture_name: str):
	"""Return the network builder for a given architecture name."""
	from nn_roast_window_search import get_network_architectures  # local import

	for arch in get_network_architectures():
		if arch.get("network_name") == architecture_name:
			return arch.get("network")
	raise KeyError(
		f"Unknown architecture '{architecture_name}'. "
		"Expected one of nn_roast_window_search.get_network_architectures()."
	)


def _run_loco_cv(*, target_name: str, window: tuple[float, float], architecture_name: str, epochs: int,
			 seed: int | None, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
	"""Re-run LOCO CV and return concatenated (actual, pred) in original units."""
	from nn_0_synthetic_data_gen import build_model_data
	from nn_1_train_model import CoffeeNetBase, evaluate_model, train_coffeenet

	if seed is not None:
		np.random.seed(seed)
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)

	df_all = build_model_data(
		NORMALIZE=False,
		REDOX=False,
		USE_BINS=False,
		test_train_split=False,
	)

	if target_name not in df_all.columns:
		raise ValueError(f"Target column '{target_name}' not found in build_model_data output.")

	# Roast values can be missing for some samples.
	df_all = df_all.dropna(subset=[target_name])
	if len(df_all) == 0:
		raise RuntimeError(f"No rows with non-missing '{target_name}' values.")

	n_points = len(df_all["cv_raw"].iloc[0])
	full_voltage_array = np.linspace(0.0, 2.0, n_points)
	vmin, vmax = window
	voltage_mask = (full_voltage_array >= vmin) & (full_voltage_array <= vmax)
	voltage_indices = np.where(voltage_mask)[0]
	if len(voltage_indices) == 0:
		raise ValueError(f"Voltage window {window} contains no points for n_points={n_points}.")

	build_network = _find_architecture_builder(architecture_name)
	coffees = df_all["Coffee Name"].unique()

	all_test_actual: list[float] = []
	all_test_pred: list[float] = []

	for test_coffee in coffees:
		test_mask = df_all["Coffee Name"] == test_coffee
		df_train = df_all[~test_mask]
		df_test = df_all[test_mask]

		X_train = np.array([
			np.asarray(cv)[voltage_indices]
			for cv in df_train["cv_raw"]
		])
		X_test = np.array([
			np.asarray(cv)[voltage_indices]
			for cv in df_test["cv_raw"]
		])
		if X_train.shape[1] == 0:
			continue

		y_train = df_train[target_name].values.reshape(-1, 1)
		y_test = df_test[target_name].values.reshape(-1, 1)

		X_scaler = StandardScaler().fit(X_train)
		y_scaler = StandardScaler().fit(y_train)

		X_train_s = X_scaler.transform(X_train)
		y_train_s = y_scaler.transform(y_train)
		X_test_s = X_scaler.transform(X_test)

		model = CoffeeNetBase()
		model.network = build_network(X_train.shape[1])
		model.to(device)

		model = train_coffeenet(model, X_train_s, y_train_s, None, None, num_epochs=epochs)

		test_pred_s = evaluate_model(model, X_test_s, y_train_s)
		test_pred = y_scaler.inverse_transform(test_pred_s)

		all_test_actual.extend([float(x) for x in y_test.flatten()])
		all_test_pred.extend([float(x) for x in test_pred.flatten()])

	actual = _as_1d(all_test_actual)
	pred = _as_1d(all_test_pred)
	if actual.shape != pred.shape:
		raise ValueError(f"Shape mismatch: actual {actual.shape} vs pred {pred.shape}")
	return actual, pred


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Generate thesis-style Roast parity plot. "
			"This uses BEST_Roast_fixed.pkl metadata (window + architecture) and recomputes LOCO CV points."
		)
	)
	parser.add_argument(
		"--pkl",
		type=Path,
		default=DEFAULT_PKL,
		help="Path to BEST_Roast_fixed.pkl (defaults to ThesisPlotGeneration/data/BEST_Roast_fixed.pkl)",
	)
	parser.add_argument(
		"--epochs",
		type=int,
		default=2000,
		help="Epochs per LOCO fold training run (default: 2000).",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=0,
		help="Random seed (default: 0). Use -1 for no seeding.",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="auto",
		choices=["auto", "cpu", "cuda"],
		help="Device: auto|cpu|cuda (default: auto).",
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

	target_name = str(metadata.get("target", "Roast"))
	window_val = metadata.get("window")
	arch_name = metadata.get("architecture")
	if not isinstance(window_val, (tuple, list)) or len(window_val) != 2:
		raise ValueError("metadata['window'] must be a 2-tuple (vmin, vmax).")
	if not isinstance(arch_name, str) or not arch_name:
		raise ValueError("metadata['architecture'] must be a non-empty string.")

	window = (float(window_val[0]), float(window_val[1]))
	seed = None if int(args.seed) < 0 else int(args.seed)

	if args.device == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("--device=cuda requested but CUDA is not available.")
	device = (
		torch.device("cuda")
		if args.device == "cuda"
		else torch.device("cpu")
		if args.device == "cpu"
		else torch.device("cuda" if torch.cuda.is_available() else "cpu")
	)

	print(
		f"Roast plot: target={target_name}, arch={arch_name}, window={window}, "
		f"epochs={int(args.epochs)}, device={device.type}"
	)

	actual, pred = _run_loco_cv(
		target_name=target_name,
		window=window,
		architecture_name=arch_name,
		epochs=int(args.epochs),
		seed=seed,
		device=device,
	)

	plots_dir = Path(__file__).resolve().parent / "plots"
	plots_dir.mkdir(exist_ok=True)

	out_pdf = plots_dir / "scatter_Roast.pdf"
	r2, mae = _plot_parity(actual=actual, pred=pred, title="Roast", out_path=out_pdf)

	stats_path = plots_dir / "roast_model_stats.txt"
	with stats_path.open("w", encoding="utf-8") as f:
		f.write(f"pkl: {pkl_path.name}\n")
		f.write(f"title: Roast\n")
		f.write(f"target: {target_name}\n")
		f.write(f"architecture: {arch_name}\n")
		f.write(f"window: {window}\n")
		f.write(f"epochs: {int(args.epochs)}\n")
		f.write(f"device: {device.type}\n")
		f.write(f"n_samples: {len(actual)}\n")
		f.write(f"r2: {r2:.6f}\n")
		f.write(f"mae: {mae:.6f}\n")
		f.write(f"plot_pdf: {out_pdf.name}\n")

	print(f"Saved: {out_pdf}")
	print(f"Wrote stats: {stats_path}")


if __name__ == "__main__":
	main()
