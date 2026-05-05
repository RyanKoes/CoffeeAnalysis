from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
	from tqdm import trange  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover
	trange = None

try:
	import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
except ImportError as exc:  # pragma: no cover
	raise ImportError(
		"nn_brightness_timeseries requires 'matplotlib'. Install it into your active environment (e.g., pip install matplotlib)."
	) from exc

# Allow imports from src/ so sibling packages (e.g. thesis_plots) resolve
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))

from nn_0_synthetic_data_gen import build_model_data  # noqa: E402
from nn_1_train_model import CoffeeNetBase, evaluate_model  # noqa: E402
from nn_timeseries_model_window_search import TCNRegressor  # noqa: E402

# Thesis plotting helpers (optional but preferred for consistency)
try:
	from thesis_plots.plot_generator import apply_publication_style, savefig_pdf  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover
	apply_publication_style = None
	savefig_pdf = None


TARGET_COL = "Brightness"


def _train_with_progress(
	model: torch.nn.Module,
	*,
	X: np.ndarray,
	y: np.ndarray,
	epochs: int,
	fold_desc: str,
) -> torch.nn.Module:
	"""Train with an epoch progress bar (full-batch) to match existing utilities.

	This is intentionally similar to `train_coffeenet` but adds a per-epoch
	progress bar for LOCO folds.
	"""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
	y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

	criterion = nn.HuberLoss(delta=1.0)
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	it = trange(int(epochs), desc=fold_desc, leave=False) if trange is not None else range(int(epochs))
	for epoch in it:
		model.train()
		optimizer.zero_grad()
		outputs = model(X_tensor)
		loss = criterion(outputs, y_tensor)
		loss.backward()
		optimizer.step()
		if trange is not None:
			it.set_postfix(loss=float(loss.detach().cpu().item()))
		elif (epoch + 1) in (1, 5, 10) or (epoch + 1) % 250 == 0 or (epoch + 1) == int(epochs):
			print(f"{fold_desc}: epoch {epoch + 1}/{int(epochs)} loss={float(loss.detach().cpu().item()):.6g}")

	return model


def _plot_parity(*, actual: np.ndarray, pred: np.ndarray, out_path: Path) -> None:
	if apply_publication_style is not None:
		apply_publication_style(figsize=(3.35, 3.35))

	fig, ax = plt.subplots()
	ax.scatter(actual, pred, s=18, alpha=0.8, zorder=2)

	# Attribute scale is typically ~7.5–10; keep it data-driven but add margin.
	min_val = float(min(np.min(actual), np.min(pred)))
	max_val = float(max(np.max(actual), np.max(pred)))
	rng = max(max_val - min_val, 1e-9)
	pad = 0.06 * rng
	plot_min = min_val - pad
	plot_max = max_val + pad
	ax.set_xlim(plot_min, plot_max)
	ax.set_ylim(plot_min, plot_max)

	ax.plot([plot_min, plot_max], [plot_min, plot_max], linestyle="--", linewidth=1.6, color="black", alpha=0.5)
	ax.set_xlabel("Actual")
	ax.set_ylabel("Predicted")
	ax.set_title("Brightness")
	ax.grid(True, alpha=0.25)
	plt.tight_layout()

	out_path.parent.mkdir(parents=True, exist_ok=True)
	if savefig_pdf is not None:
		savefig_pdf(str(out_path))
	else:
		fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
	plt.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser(description="Train a sequence model (TCN) to predict the sensory attribute Brightness.")
	parser.add_argument(
		"--mode",
		choices=["loco"],
		default="loco",
		help="Evaluation mode (currently only loco = leave-one-coffee-out CV).",
	)
	parser.add_argument("--epochs", type=int, default=2500, help="Training epochs per LOCO fold (default: 2500).")
	parser.add_argument(
		"--out",
		type=Path,
		default=Path(__file__).resolve().parents[2] / "src" / "thesis_plots" / "plots" / "scatter_Brightness_timeseries.pdf",
		help="Output parity PDF path.",
	)
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device.type == "cpu":
		print("Warning: CUDA not available; training will be slower on CPU.")

	# Load dataset
	df_all = build_model_data(
		NORMALIZE=False,
		REDOX=False,
		USE_BINS=False,
		test_train_split=False,
	)

	if TARGET_COL not in df_all.columns:
		raise KeyError(f"Missing target column '{TARGET_COL}' in dataset. Available cols: {sorted(df_all.columns)[:40]} ...")

	# Use only labeled rows
	df_lbl = df_all.dropna(subset=[TARGET_COL]).copy()
	if df_lbl.empty:
		raise ValueError(f"No labeled samples found for '{TARGET_COL}'")

	coffees = df_lbl["Coffee Name"].astype(str).unique()
	print(f"Loaded labeled Brightness dataset: {len(df_lbl)} samples, {len(coffees)} coffees")

	# Use full 0–2V trace (no windowing)
	first_cv = df_lbl["cv_raw"].iloc[0]
	voltage_indices = np.arange(len(first_cv), dtype=int)
	print(f"Using full 0–2V trace (n={len(voltage_indices)})")

	all_actual: list[float] = []
	all_pred: list[float] = []

	# LOCO-CV
	for fold_i, test_coffee in enumerate(coffees, start=1):
		test_mask = df_lbl["Coffee Name"].astype(str) == str(test_coffee)
		df_train = df_lbl[~test_mask]
		df_test = df_lbl[test_mask]
		if df_test.empty or df_train.empty:
			continue

		X_train = np.array([np.asarray(cv, dtype=float)[voltage_indices] for cv in df_train["cv_raw"]])
		X_test = np.array([np.asarray(cv, dtype=float)[voltage_indices] for cv in df_test["cv_raw"]])
		y_train = df_train[TARGET_COL].values.reshape(-1, 1).astype(float)
		y_test = df_test[TARGET_COL].values.reshape(-1, 1).astype(float)

		X_scaler = StandardScaler().fit(X_train)
		y_scaler = StandardScaler().fit(y_train)
		X_train_s = X_scaler.transform(X_train)
		X_test_s = X_scaler.transform(X_test)
		y_train_s = y_scaler.transform(y_train)
		y_test_s = y_scaler.transform(y_test)

		# Build a strong, generally reliable sequence model
		model = CoffeeNetBase()
		model.network = TCNRegressor(channels=96, kernel_size=5, dilations=(1, 2, 4, 8), dropout=0.15)
		model.to(device)
		fold_desc = f"Fold {fold_i}/{len(coffees)} ({test_coffee})"
		model = _train_with_progress(model, X=X_train_s, y=y_train_s, epochs=int(args.epochs), fold_desc=fold_desc)

		pred_s = evaluate_model(model, X_test_s, y_test_s)
		pred_s_np = np.asarray(pred_s, dtype=float).reshape(-1, 1)
		pred = y_scaler.inverse_transform(pred_s_np).reshape(-1)

		all_actual.extend(y_test.reshape(-1).tolist())
		all_pred.extend(pred.tolist())

	actual = np.asarray(all_actual, dtype=float)
	pred = np.asarray(all_pred, dtype=float)
	if actual.size == 0:
		raise ValueError("No predictions generated (unexpected).")

	r2 = float(r2_score(actual, pred))
	mae = float(mean_absolute_error(actual, pred))
	print(f"\nBrightness (TCN) LOCO-CV results")
	print(f"- n_samples: {len(actual)}")
	print(f"- r2: {r2:.4f}")
	print(f"- mae: {mae:.4f}")

	_plot_parity(actual=actual, pred=pred, out_path=args.out)
	print(f"Saved parity plot: {args.out}")

	stats_path = args.out.with_suffix(".txt")
	stats_path.write_text(
		"\n".join(
			[
				"target: Brightness",
				"model: TCNRegressor(channels=96, kernel_size=5, dilations=(1,2,4,8), dropout=0.15)",
				f"mode: LOCO",
				f"epochs: {int(args.epochs)}",
				"voltage_mode: full_0p0_2p0",
				f"n_samples: {int(actual.size)}",
				f"r2: {r2:.6f}",
				f"mae: {mae:.6f}",
			],
		),
		encoding="utf-8",
	)
	print(f"Wrote stats: {stats_path}")


if __name__ == "__main__":
	main()
