from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score

try:
	from tqdm import tqdm  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover
	tqdm = None

try:
	import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
except ImportError as exc:  # pragma: no cover
	raise ImportError(
		"ThesisPlotGeneration.plot_best_sensory_models requires 'matplotlib'. "
		"Install it into your active environment (e.g., pip install matplotlib)."
	) from exc

# Allow imports from repo root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ThesisPlotGeneration.plot_generator import (  # noqa: E402
	FIGSIZE_SINGLE_COLUMN,
	apply_publication_style,
	savefig_pdf,
)
from nn_0_synthetic_data_gen import build_model_data  # noqa: E402
from nn_1_train_model import CoffeeNetBase, evaluate_model, train_coffeenet  # noqa: E402
from nn_attributes_model_window_search import (  # noqa: E402
	get_network_architectures as get_attribute_architectures,
)
from nn_flavors_model_window_search import (  # noqa: E402
	get_network_architectures as get_flavor_architectures,
)


SENSORY_BEST_PKLS: tuple[str, ...] = (
	"BEST_Attribute_Brightness_fixed.pkl",
	"BEST_Attribute_Clean_Cup_fixed.pkl",
	"BEST_Attribute_Finish_fixed.pkl",
	"BEST_Attribute_Uniformity_fixed.pkl",
	"BEST_Flavor_Caramel_fixed.pkl",
	"BEST_Flavor_Citrus_fixed.pkl",
	"BEST_Flavor_Rustic_fixed.pkl",
)


# ------------------------------------------------------------
# GLOBAL SWITCHES
# ------------------------------------------------------------
# If True, uses the voltage window indices saved in each BEST_*.pkl.
# If False, ignores the saved window and uses the full 0.0–2.0 CV trace.
USE_SAVED_VOLTAGE_WINDOW: bool = True

# If set (e.g. 0.3), overrides *existing* Dropout layers in the chosen
# architecture during LOCO retraining. This does NOT add new layers.
#
# Notes:
# - Applies only to --mode loco (training). Dropout is disabled in eval anyway.
# - If an architecture has no nn.Dropout layers, this has no effect.
DROPOUT_P_OVERRIDE: float | None = 0.5

# Plot padding: add a little room beyond the requested axis limits.
# Example: for (7.5, 10.0) and pad=0.04 -> axes become (7.4, 10.1).
AXIS_PAD_FRAC: float = 0.04

# Cache LOCO predictions so we don't have to retrain every time.
# Cache is keyed by: target, architecture, voltage window indices, epochs,
# dropout override, and a hash of the underlying labeled dataset.
CACHE_LOCO_RESULTS: bool = True


def _apply_dropout_override(module: nn.Module, *, p: float) -> None:
	"""Recursively override nn.Dropout probabilities in-place."""
	for child in module.modules():
		if isinstance(child, nn.Dropout):
			child.p = float(p)


def _safe_slug(text: str) -> str:
	return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in text).strip("_")


def _load_best_artifact(pkl_path: Path) -> dict:
	data = torch.load(pkl_path, map_location=torch.device("cpu"), weights_only=False)
	metadata = data.get("metadata")
	if not isinstance(metadata, dict):
		raise ValueError(f"Missing/invalid metadata in {pkl_path.name}")

	final_state_dict = data.get("final_model_state_dict")
	search_state_dict = data.get("model_state_dict")

	x_scaler = data.get("x_scaler")
	y_scaler = data.get("y_scaler")
	voltage_indices = data.get("voltage_window_indices")

	if x_scaler is None or y_scaler is None or voltage_indices is None:
		raise ValueError(
			f"Missing final-model artifacts (x_scaler/y_scaler/voltage_window_indices) in {pkl_path.name}. "
			"These plots are generated from the final trained model saved in the PKL."
		)

	target_columns = metadata.get("target_columns")
	if not isinstance(target_columns, list) or not target_columns:
		raise ValueError(f"Missing/invalid metadata['target_columns'] in {pkl_path.name}")
	if len(target_columns) != 1:
		raise ValueError(
			f"Expected single-target artifact but got {len(target_columns)} targets in {pkl_path.name}: {target_columns}"
		)

	target_display = str(metadata.get("target", pkl_path.stem))
	architecture = str(metadata.get("architecture", ""))
	window = metadata.get("window")
	mean_r2 = metadata.get("mean_r2")
	architecture_source = _infer_architecture_source(
		target_display=target_display,
		pkl_name=pkl_path.name,
		metadata_target=str(metadata.get("target", "")),
	)

	return {
		"target_display": target_display,
		"target_col": str(target_columns[0]),
		"architecture": architecture,
		"architecture_source": architecture_source,
		"saved_mean_r2": mean_r2,
		"window": window,
		"final_state_dict": final_state_dict,
		"search_state_dict": search_state_dict,
		"x_scaler": x_scaler,
		"y_scaler": y_scaler,
		"voltage_indices": np.asarray(voltage_indices, dtype=int),
	}

def _infer_architecture_source(*, target_display: str, pkl_name: str, metadata_target: str) -> str:
	needle = " ".join([target_display, pkl_name, metadata_target]).lower()
	if "flavor" in needle:
		return "flavor"
	if "attribute" in needle:
		return "attribute"
	return "unknown"


def _get_architecture_catalog(architecture_source: str) -> list[dict]:
	if architecture_source == "flavor":
		return get_flavor_architectures()
	if architecture_source == "attribute":
		return get_attribute_architectures()
	# We intentionally do not fall back silently to a generic catalog.
	raise ValueError(
		"Could not determine which architecture catalog to use for this artifact. "
		"Expected a Flavor_* or Attribute_* target." 
	)


def _get_architecture_def(architecture: str, *, catalog: list[dict]) -> dict:
	arch_def = next((a for a in catalog if a.get("network_name") == architecture), None)
	if arch_def is None:
		known = ", ".join(sorted({str(a.get("network_name", "")) for a in catalog}))
		raise ValueError(
			f"Architecture '{architecture}' not found in selected catalog. "
			f"Known architectures: {known}"
		)
	return arch_def


def _build_network(arch_def: dict, *, input_size: int, output_size: int) -> nn.Module:
	"""Build a network from an architecture definition.

	Flavor/Attribute window search catalogs define `network(input_size, output_size)`.
	Legacy single-target catalogs define `network(input_size)`.
	"""
	ctor = arch_def.get("network")
	if ctor is None:
		raise ValueError("Architecture definition is missing 'network' constructor")
	try:
		return ctor(input_size, output_size)
	except TypeError:
		# Fallback: single-target constructor signature
		if output_size != 1:
			raise
		return ctor(input_size)


def _validate_state_dict_matches_architecture(
	*,
	architecture: str,
	architecture_source: str,
	input_size: int,
	output_size: int,
	state_dict,
	context: str,
) -> None:
	if state_dict is None:
		return
	if not isinstance(state_dict, dict):
		raise ValueError(f"Invalid state_dict type for {context}: {type(state_dict)}")

	catalog = _get_architecture_catalog(architecture_source)
	arch_def = _get_architecture_def(architecture, catalog=catalog)

	model = CoffeeNetBase()
	model.network = _build_network(arch_def, input_size=input_size, output_size=output_size)

	model_state = model.state_dict()
	model_keys = set(model_state.keys())
	loaded_keys = set(state_dict.keys())
	if model_keys != loaded_keys:
		extra = sorted(loaded_keys - model_keys)
		missing = sorted(model_keys - loaded_keys)
		raise ValueError(
			f"Saved weights do not match architecture '{architecture}' ({architecture_source}) for {context}. "
			f"Extra keys: {extra[:12]}{'...' if len(extra) > 12 else ''}; "
			f"Missing keys: {missing[:12]}{'...' if len(missing) > 12 else ''}"
		)

	for key in sorted(model_keys):
		expected_shape = tuple(model_state[key].shape)
		loaded_shape = tuple(state_dict[key].shape)
		if expected_shape != loaded_shape:
			raise ValueError(
				f"Shape mismatch for '{key}' in {context}: expected {expected_shape}, got {loaded_shape}. "
				f"Architecture='{architecture}' source='{architecture_source}'"
			)


def _predict_final_model_in_sample(
	df_all,
	*,
	target_col: str,
	voltage_indices: np.ndarray,
	x_scaler,
	y_scaler,
	architecture: str,
	architecture_source: str,
	state_dict,
):
	# Filter to rows with labels
	df = df_all.dropna(subset=[target_col]).copy()
	if df.empty:
		raise ValueError(f"No rows have non-NaN values for target column '{target_col}'")

	X = np.array([np.asarray(cv)[voltage_indices] for cv in df["cv_raw"]])
	y = df[target_col].values.reshape(-1, 1)

	X_s = x_scaler.transform(X)
	input_size = int(X.shape[1])
	output_size = 1
	_validate_state_dict_matches_architecture(
		architecture=architecture,
		architecture_source=architecture_source,
		input_size=input_size,
		output_size=output_size,
		state_dict=state_dict,
		context=f"final-eval target='{target_col}'",
	)
	catalog = _get_architecture_catalog(architecture_source)
	arch_def = _get_architecture_def(architecture, catalog=catalog)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = CoffeeNetBase()
	model.network = _build_network(arch_def, input_size=input_size, output_size=output_size)
	model.load_state_dict(state_dict)
	model.to(device)

	y_pred_s = evaluate_model(model, X_s, y)
	if isinstance(y_pred_s, torch.Tensor):
		y_pred_s = y_pred_s.detach().cpu().numpy()

	y_pred = y_scaler.inverse_transform(y_pred_s)
	return y.flatten(), y_pred.flatten()


def _predict_loco_cv(
	df_all,
	*,
	target_col: str,
	voltage_indices: np.ndarray,
	architecture: str,
	architecture_source: str,
	epochs: int,
) -> tuple[np.ndarray, np.ndarray]:
	"""Leave-one-coffee-out CV.

	This retrains the network from scratch for each held-out coffee.
	"""
	# Filter to labeled rows only
	df = df_all.dropna(subset=[target_col]).copy()
	if df.empty:
		raise ValueError(f"No rows have non-NaN values for target column '{target_col}'")

	# Build design matrix once (faster than rebuilding per fold)
	coffee_names = df["Coffee Name"].astype(str).to_numpy()
	X_all = np.array([np.asarray(cv)[voltage_indices] for cv in df["cv_raw"]])
	y_all = df[target_col].values.reshape(-1, 1)

	coffees = np.unique(coffee_names)
	catalog = _get_architecture_catalog(architecture_source)
	arch_def = _get_architecture_def(architecture, catalog=catalog)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	all_actual: list[float] = []
	all_pred: list[float] = []

	iterator = coffees
	if tqdm is not None:
		iterator = tqdm(coffees, desc=f"LOCO {target_col}")

	for test_coffee in iterator:
		test_mask = coffee_names == str(test_coffee)
		X_train = X_all[~test_mask]
		X_test = X_all[test_mask]
		y_train = y_all[~test_mask]
		y_test = y_all[test_mask]

		# Scale per fold
		from sklearn.preprocessing import StandardScaler

		x_scaler = StandardScaler().fit(X_train)
		y_scaler = StandardScaler().fit(y_train)
		X_train_s = x_scaler.transform(X_train)
		y_train_s = y_scaler.transform(y_train)
		X_test_s = x_scaler.transform(X_test)
		y_test_s = y_scaler.transform(y_test)

		model = CoffeeNetBase()
		model.network = _build_network(
			arch_def,
			input_size=int(X_train.shape[1]),
			output_size=1,
		)
		if DROPOUT_P_OVERRIDE is not None:
			_override_p = float(DROPOUT_P_OVERRIDE)
			if not (0.0 <= _override_p < 1.0):
				raise ValueError(
					f"Invalid DROPOUT_P_OVERRIDE={DROPOUT_P_OVERRIDE}. Expected 0.0 <= p < 1.0."
				)
			_apply_dropout_override(model.network, p=_override_p)
		model.to(device)
		model = train_coffeenet(
			model,
			X_train_s,
			y_train_s,
			X_test=None,
			y_test=None,
			num_epochs=int(epochs),
		)

		y_pred_s = evaluate_model(model, X_test_s, y_test_s)
		if isinstance(y_pred_s, torch.Tensor):
			y_pred_s = y_pred_s.detach().cpu().numpy()
		y_pred = y_scaler.inverse_transform(y_pred_s)

		all_actual.extend(y_test.flatten().tolist())
		all_pred.extend(y_pred.flatten().tolist())

	return np.asarray(all_actual, dtype=float), np.asarray(all_pred, dtype=float)


def _hash_loco_inputs(
	*,
	coffee_names: np.ndarray,
	X_all: np.ndarray,
	y_all: np.ndarray,
	target_col: str,
	voltage_indices: np.ndarray,
	architecture: str,
	architecture_source: str,
	epochs: int,
	dropout_p_override: float | None,
	use_saved_voltage_window: bool,
) -> str:
	"""Compute a stable hash for LOCO prediction caching."""
	h = hashlib.sha256()

	meta = {
		"target_col": str(target_col),
		"architecture": str(architecture),
		"architecture_source": str(architecture_source),
		"epochs": int(epochs),
		"dropout_p_override": dropout_p_override,
		"use_saved_voltage_window": bool(use_saved_voltage_window),
		"voltage_indices": voltage_indices.astype(int).tolist(),
		"n_samples": int(len(y_all)),
		"n_features": int(X_all.shape[1]),
	}
	h.update(json.dumps(meta, sort_keys=True).encode("utf-8"))

	# Include the labeled dataset itself.
	# - Coffee names (as UTF-8 with separators)
	# - X values (float array bytes)
	# - y values (float array bytes)
	name_blob = "\n".join(coffee_names.astype(str).tolist()).encode("utf-8")
	h.update(name_blob)
	h.update(np.ascontiguousarray(X_all).tobytes())
	h.update(np.ascontiguousarray(y_all).tobytes())

	return h.hexdigest()


def _plot_parity(
	*,
	actual: np.ndarray,
	pred: np.ndarray,
	title: str,
	xlabel: str,
	ylabel: str,
	xlim: tuple[float, float] | None = None,
	ylim: tuple[float, float] | None = None,
	tick_step: float | None = None,
	axis_pad_frac: float = AXIS_PAD_FRAC,
	out_path: Path,
) -> tuple[float, float]:
	# Match style from plot_best_models.py
	w, h = FIGSIZE_SINGLE_COLUMN
	figsize = (w, h * 1.3)
	fig, ax = plt.subplots(figsize=figsize)

	ax.scatter(actual, pred, s=18, alpha=0.8, zorder=2)

	# Default: auto-scale from data, with a small margin.
	min_val = float(min(np.min(actual), np.min(pred)))
	max_val = float(max(np.max(actual), np.max(pred)))
	rng = max_val - min_val
	plot_min = min_val - 0.05 * rng
	plot_max = max_val + 0.05 * rng

	# If limits are provided, use them (with a small padding) for axis bounds
	# and for the parity line.
	if xlim is not None and ylim is not None:
		x0, x1 = float(xlim[0]), float(xlim[1])
		y0, y1 = float(ylim[0]), float(ylim[1])
		pad_x = (x1 - x0) * float(axis_pad_frac)
		pad_y = (y1 - y0) * float(axis_pad_frac)
		ax.set_xlim(x0 - pad_x, x1 + pad_x)
		ax.set_ylim(y0 - pad_y, y1 + pad_y)
		plot_min = float(min(x0 - pad_x, y0 - pad_y))
		plot_max = float(max(x1 + pad_x, y1 + pad_y))

		if tick_step is not None:
			step = float(tick_step)
			if step <= 0:
				raise ValueError(f"tick_step must be positive, got {tick_step}")
			eps = step * 1e-6
			xticks = np.arange(x0, x1 + eps, step)
			yticks = np.arange(y0, y1 + eps, step)
			ax.set_xticks(xticks)
			ax.set_yticks(yticks)

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
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_title(title)

	plt.tight_layout()
	savefig_pdf(str(out_path))
	plt.close(fig)
	return r2, mae



def main() -> None:
	parser = argparse.ArgumentParser(description="Generate thesis-style sensory parity plots.")
	parser.add_argument(
		"--mode",
		choices=["loco", "final"],
		default="loco",
		help=(
			"loco=retrain per held-out coffee (proper CV parity); "
			"final=use saved final model and evaluate on same labeled set (will look overly optimistic)."
		),
	)
	parser.add_argument(
		"--epochs",
		type=int,
		default=5000,
		help="Training epochs per LOCO fold (only used in --mode loco).",
	)
	args = parser.parse_args()

	apply_publication_style(figsize=(3.35, 3.35))

	base_dir = Path(__file__).resolve().parent
	data_dir = base_dir / "data"
	plots_dir = base_dir / "plots"
	plots_dir.mkdir(exist_ok=True)

	# Load dataset once
	df_all = build_model_data(
		NORMALIZE=False,
		REDOX=False,
		USE_BINS=False,
		test_train_split=False,
	)

	print(f"Loaded dataset: {len(df_all)} samples, {df_all['Coffee Name'].nunique()} coffees")

	if not USE_SAVED_VOLTAGE_WINDOW and args.mode == "final":
		raise ValueError(
			"USE_SAVED_VOLTAGE_WINDOW=False is incompatible with --mode final. "
			"The saved model weights/scalers were trained on the best voltage window, "
			"so they cannot be evaluated on a different input dimensionality (full 0.0–2.0 trace). "
			"Use --mode loco (retrain per fold) or regenerate BEST_*.pkl artifacts trained on full voltage."
		)

	# Precompute full-voltage indices once (0.0–2.0 inclusive)
	first_cv = df_all["cv_raw"].iloc[0]
	full_voltage_indices = np.arange(len(first_cv), dtype=int)

	stats_rows: list[dict] = []

	for filename in SENSORY_BEST_PKLS:
		pkl_path = data_dir / filename
		if not pkl_path.exists():
			raise FileNotFoundError(f"Could not find {pkl_path}")

		artifact = _load_best_artifact(pkl_path)
		target_display = artifact["target_display"]
		target_col = artifact["target_col"]
		architecture = artifact["architecture"]
		architecture_source = artifact["architecture_source"]
		saved_mean_r2 = artifact.get("saved_mean_r2")

		print(f"\nPlotting {target_display} (col='{target_col}')")
		print(f"  Architecture: {architecture}")
		print(f"  Architecture source: {architecture_source}")
		if USE_SAVED_VOLTAGE_WINDOW:
			print(f"  Window: {artifact.get('window')}")
		else:
			print("  Window override: FULL (0.0–2.0)")
		if saved_mean_r2 is not None:
			print(f"  Saved search CV mean_r2: {saved_mean_r2:.3f}")

		# Strict validation: ensure the artifact architecture name is known in the
		# appropriate catalog and that the saved weights are compatible.
		input_size = int(len(artifact["voltage_indices"]))
		_validate_state_dict_matches_architecture(
			architecture=architecture,
			architecture_source=architecture_source,
			input_size=input_size,
			output_size=1,
			state_dict=artifact["final_state_dict"] or artifact["search_state_dict"],
			context=f"artifact '{pkl_path.name}'",
		)

		# Select which voltage indices to use for this run.
		# - For normal operation, we use the per-target best window saved in the PKL.
		# - If USE_SAVED_VOLTAGE_WINDOW=False, we force full-voltage (0.0–2.0).
		voltage_indices = artifact["voltage_indices"] if USE_SAVED_VOLTAGE_WINDOW else full_voltage_indices

		if args.mode == "final":
			state_dict = artifact["final_state_dict"] or artifact["search_state_dict"]
			if state_dict is None:
				raise ValueError(f"No state_dict found in {pkl_path.name}")
			actual, pred = _predict_final_model_in_sample(
				df_all,
				target_col=target_col,
				voltage_indices=voltage_indices,
				x_scaler=artifact["x_scaler"],
				y_scaler=artifact["y_scaler"],
				architecture=architecture,
				architecture_source=architecture_source,
				state_dict=state_dict,
			)
			print(f"  Mode: final (in-sample)")
		else:
			# LOCO caching: reuse saved predictions when inputs haven't changed.
			cache_hit = False
			cache_path: Path | None = None
			if CACHE_LOCO_RESULTS:
				cache_dir = plots_dir / "loco_cache"
				cache_dir.mkdir(exist_ok=True)
				# Build labeled subset arrays once for hashing (and to avoid retraining when cached).
				df_lbl = df_all.dropna(subset=[target_col]).copy()
				coffee_names = df_lbl["Coffee Name"].astype(str).to_numpy()
				X_all = np.array([np.asarray(cv)[voltage_indices] for cv in df_lbl["cv_raw"]])
				y_all = df_lbl[target_col].values.reshape(-1, 1)
				cache_key = _hash_loco_inputs(
					coffee_names=coffee_names,
					X_all=X_all,
					y_all=y_all,
					target_col=target_col,
					voltage_indices=voltage_indices,
					architecture=architecture,
					architecture_source=architecture_source,
					epochs=int(args.epochs),
					dropout_p_override=DROPOUT_P_OVERRIDE,
					use_saved_voltage_window=USE_SAVED_VOLTAGE_WINDOW,
				)
				cache_path = cache_dir / f"loco_{_safe_slug(target_display)}_{cache_key[:16]}.npz"
				if cache_path.exists():
					loaded = np.load(cache_path, allow_pickle=False)
					actual = loaded["actual"].astype(float)
					pred = loaded["pred"].astype(float)
					cache_hit = True
					print(f"  LOCO cache hit: {cache_path.name}")

			if not cache_hit:
				actual, pred = _predict_loco_cv(
					df_all,
					target_col=target_col,
					voltage_indices=voltage_indices,
					architecture=architecture,
					architecture_source=architecture_source,
					epochs=int(args.epochs),
				)
				if CACHE_LOCO_RESULTS and cache_path is not None:
					npz_meta = {
						"target_display": target_display,
						"target_col": target_col,
						"architecture": architecture,
						"architecture_source": architecture_source,
						"epochs": int(args.epochs),
						"dropout_p_override": DROPOUT_P_OVERRIDE,
						"use_saved_voltage_window": USE_SAVED_VOLTAGE_WINDOW,
					}
					np.savez_compressed(
						cache_path,
						actual=np.asarray(actual, dtype=float),
						pred=np.asarray(pred, dtype=float),
						meta=json.dumps(npz_meta, sort_keys=True),
					)
					print(f"  LOCO cache saved: {cache_path.name}")
			print(f"  Mode: LOCO-CV (epochs={int(args.epochs)})")

		print(f"  Samples: {len(actual)}")

		safe_name = _safe_slug(target_display)
		out_path = plots_dir / f"scatter_{safe_name}.pdf"
		# Axis limits requested for thesis plots.
		# - Attributes: x/y both 7.5–10
		# - Flavors: x 0–5, y -1–5
		if architecture_source == "attribute":
			xlim = (7.5, 10.0)
			ylim = (7.5, 10.0)
		elif architecture_source == "flavor":
			xlim = (0.0, 5.0)
			ylim = (-1.0, 5.0)
		else:
			xlim = None
			ylim = None
		r2, mae = _plot_parity(
			actual=actual,
			pred=pred,
			title=str(target_col),
			xlabel="Actual",
			ylabel="Predicted",
			xlim=xlim,
			ylim=ylim,
			tick_step=0.5 if architecture_source == "attribute" else None,
			axis_pad_frac=AXIS_PAD_FRAC,
			out_path=out_path,
		)
		print(f"  R2: {r2:.3f}")
		print(f"  MAE: {mae:.3g}")
		print(f"  Saved: {out_path}")

		stats_rows.append(
			{
				"pkl": pkl_path.name,
				"target_display": target_display,
				"target_col": target_col,
				"architecture": architecture,
				"architecture_source": architecture_source,
				"window": (0.0, 2.0) if not USE_SAVED_VOLTAGE_WINDOW else artifact.get("window"),
				"voltage_mode": "saved_window" if USE_SAVED_VOLTAGE_WINDOW else "full_0p0_2p0",
				"dropout_p_override": DROPOUT_P_OVERRIDE,
				"saved_search_cv_mean_r2": saved_mean_r2,
				"mode": args.mode,
				"epochs": int(args.epochs) if args.mode == "loco" else None,
				"n_samples": int(len(actual)),
				"r2": float(r2),
				"mae": float(mae),
				"plot_pdf": out_path.name,
			}
		)

	# Write a single summary file for easy reference
	if args.mode == "loco":
		summary_name = f"sensory_model_stats_loco_epochs{int(args.epochs)}.txt"
	else:
		summary_name = "sensory_model_stats_final.txt"

	summary_path = plots_dir / summary_name
	with summary_path.open("w", encoding="utf-8") as f:
		f.write(f"mode: {args.mode}\n")
		if args.mode == "loco":
			f.write(f"epochs: {int(args.epochs)}\n")
		f.write(f"voltage_mode: {'saved_window' if USE_SAVED_VOLTAGE_WINDOW else 'full_0p0_2p0'}\n")
		f.write(f"dropout_p_override: {DROPOUT_P_OVERRIDE}\n")
		f.write(f"n_models: {len(stats_rows)}\n")
		f.write("\n")
		for row in stats_rows:
			f.write(f"{row['target_col']}\n")
			f.write(f"  pkl: {row['pkl']}\n")
			f.write(f"  target_display: {row['target_display']}\n")
			f.write(f"  architecture: {row['architecture']} ({row['architecture_source']})\n")
			f.write(f"  window: {row['window']}\n")
			if row["saved_search_cv_mean_r2"] is not None:
				f.write(f"  saved_search_cv_mean_r2: {float(row['saved_search_cv_mean_r2']):.6f}\n")
			f.write(f"  n_samples: {row['n_samples']}\n")
			f.write(f"  r2: {row['r2']:.6f}\n")
			f.write(f"  mae: {row['mae']:.6f}\n")
			f.write(f"  plot_pdf: {row['plot_pdf']}\n")
			f.write("\n")

	print(f"\nWrote stats summary: {summary_path}")


if __name__ == "__main__":
	main()
