from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from util import DATADIR, PLOTDIR, setup_mplt
from nn_1_train_model import CoffeeNetBase, train_coffeenet, evaluate_model


def load_results(target: str):
	"""Load full window search results produced by nn_model_search.py, if present.

	Tries target-specific files first (full_window_search_results_<target>.pkl),
	then falls back to the legacy name full_window_search_results.pkl.
	Returns a DataFrame or None if nothing is found.
	"""

	# Prefer target-specific file
	results_path = DATADIR / f"full_window_search_results_{target}.pkl"
	if results_path.exists():
		return pd.read_pickle(results_path)

	# Fallback to legacy filename if it exists
	legacy_path = DATADIR / "full_window_search_results.pkl"
	if legacy_path.exists():
		return pd.read_pickle(legacy_path)

	return None


def get_architecture_factories():
	"""Return a mapping from network_name to a factory that builds the network.

	These definitions mirror the architectures used in nn_model_search.py so
	that we can re-train the same models from the saved *_all.pkl data.
	"""
	architectures = [
		{
			"network": lambda input_size: nn.Sequential(
				nn.Linear(input_size, 64),
				nn.ReLU(),
				nn.Linear(64, 1),
			),
			"network_name": "Tiny-64-1",
		},
		{
			"network": lambda input_size: nn.Sequential(
				nn.Linear(input_size, 32),
				nn.Tanh(),
				nn.Linear(32, 1),
			),
			"network_name": "Tiny-32-1-Tanh",
		},
		{
			"network": lambda input_size: nn.Sequential(
				nn.Linear(input_size, 64),
				nn.ReLU(),
				nn.Linear(64, 32),
				nn.ReLU(),
				nn.Linear(32, 1),
			),
			"network_name": "Small-64-32-1",
		},
		{
			"network": lambda input_size: nn.Sequential(
				nn.Linear(input_size, 32),
				nn.LeakyReLU(0.1),
				nn.Linear(32, 16),
				nn.LeakyReLU(0.1),
				nn.Linear(16, 1),
			),
			"network_name": "Small-32-16-1-Leaky",
		},
		{
			"network": lambda input_size: nn.Sequential(
				nn.Linear(input_size, 16),
				nn.ReLU(),
				nn.Linear(16, 1),
			),
			"network_name": "Minimal-16-1",
		},
		{
			"network": lambda input_size: nn.Sequential(
				nn.Linear(input_size, 128),
				nn.BatchNorm1d(128),
				nn.ReLU(),
				nn.Dropout(0.1),
				nn.Linear(128, 1),
			),
			"network_name": "Simple-128-1",
		},
		{
			"network": lambda input_size: nn.Sequential(
				nn.Linear(input_size, 256),
				nn.BatchNorm1d(256),
				nn.ReLU(),
				nn.Dropout(0.1),
				nn.Linear(256, 1),
			),
			"network_name": "Simple-256-1",
		},
		{
			"network": lambda input_size: nn.Sequential(
				nn.Linear(input_size, 512),
				nn.BatchNorm1d(512),
				nn.ReLU(),
				nn.Dropout(0.1),
				nn.Linear(512, 1),
			),
			"network_name": "Simple-512-1",
		},
		{
			"network": lambda input_size: nn.Sequential(
				nn.Linear(input_size, 128),
				nn.BatchNorm1d(128),
				nn.ReLU(),
				nn.Dropout(0.1),
				nn.Linear(128, 64),
				nn.BatchNorm1d(64),
				nn.ReLU(),
				nn.Dropout(0.1),
				nn.Linear(64, 1),
			),
			"network_name": "TwoLayer-128-64-1",
		},
		{
			"network": lambda input_size: nn.Sequential(
				nn.Linear(input_size, 256),
				nn.BatchNorm1d(256),
				nn.ReLU(),
				nn.Dropout(0.1),
				nn.Linear(256, 64),
				nn.BatchNorm1d(64),
				nn.ReLU(),
				nn.Dropout(0.1),
				nn.Linear(64, 1),
			),
			"network_name": "TwoLayer-256-64-1",
		},
		{
			"network": lambda input_size: nn.Sequential(
				nn.Linear(input_size, 256),
				nn.BatchNorm1d(256),
				nn.ReLU(),
				nn.Dropout(0.1),
				nn.Linear(256, 128),
				nn.BatchNorm1d(128),
				nn.ReLU(),
				nn.Dropout(0.1),
				nn.Linear(128, 1),
			),
			"network_name": "TwoLayer-256-128-1",
		},
		{
			"network": lambda input_size: nn.Sequential(
				nn.Linear(input_size, 512),
				nn.BatchNorm1d(512),
				nn.ReLU(),
				nn.Dropout(0.1),
				nn.Linear(512, 128),
				nn.BatchNorm1d(128),
				nn.ReLU(),
				nn.Dropout(0.1),
				nn.Linear(128, 1),
			),
			"network_name": "TwoLayer-512-128-1",
		},
		{
			"network": lambda input_size: nn.Sequential(
				nn.Linear(input_size, 256),
				nn.BatchNorm1d(256),
				nn.ReLU(),
				nn.Dropout(0.15),
				nn.Linear(256, 128),
				nn.BatchNorm1d(128),
				nn.ReLU(),
				nn.Dropout(0.15),
				nn.Linear(128, 64),
				nn.BatchNorm1d(64),
				nn.ReLU(),
				nn.Dropout(0.15),
				nn.Linear(64, 1),
			),
			"network_name": "ThreeLayer-256-128-64-1",
		},
		{
			"network": lambda input_size: nn.Sequential(
				nn.Linear(input_size, 512),
				nn.BatchNorm1d(512),
				nn.ReLU(),
				nn.Dropout(0.15),
				nn.Linear(512, 256),
				nn.BatchNorm1d(256),
				nn.ReLU(),
				nn.Dropout(0.15),
				nn.Linear(256, 128),
				nn.BatchNorm1d(128),
				nn.ReLU(),
				nn.Dropout(0.15),
				nn.Linear(128, 1),
			),
			"network_name": "ThreeLayer-512-256-128-1",
		},
		{
			"network": lambda input_size: nn.Sequential(
				nn.Linear(input_size, 256),
				nn.BatchNorm1d(256),
				nn.ReLU(),
				nn.Dropout(0.3),
				nn.Linear(256, 128),
				nn.BatchNorm1d(128),
				nn.ReLU(),
				nn.Dropout(0.3),
				nn.Linear(128, 1),
			),
			"network_name": "HighDrop-256-128-1",
		},
		{
			"network": lambda input_size: nn.Sequential(
				nn.Linear(input_size, 512),
				nn.BatchNorm1d(512),
				nn.ReLU(),
				nn.Dropout(0.25),
				nn.Linear(512, 256),
				nn.BatchNorm1d(256),
				nn.ReLU(),
				nn.Dropout(0.25),
				nn.Linear(256, 1),
			),
			"network_name": "HighDrop-512-256-1",
		},
		{
			"network": lambda input_size: nn.Sequential(
				nn.Linear(input_size, 1024),
				nn.BatchNorm1d(1024),
				nn.ReLU(),
				nn.Dropout(0.2),
				nn.Linear(1024, 1),
			),
			"network_name": "Wide-1024-1",
		},
		{
			"network": lambda input_size: nn.Sequential(
				nn.Linear(input_size, 512),
				nn.BatchNorm1d(512),
				nn.ReLU(),
				nn.Dropout(0.1),
				nn.Linear(512, 64),
				nn.BatchNorm1d(64),
				nn.ReLU(),
				nn.Dropout(0.1),
				nn.Linear(64, 1),
			),
			"network_name": "Bottleneck-512-64-1",
		},
		{
			"network": lambda input_size: nn.Sequential(
				nn.Linear(input_size, 512),
				nn.BatchNorm1d(512),
				nn.ReLU(),
				nn.Dropout(0.2),
				nn.Linear(512, 256),
				nn.BatchNorm1d(256),
				nn.ReLU(),
				nn.Dropout(0.2),
				nn.Linear(256, 128),
				nn.BatchNorm1d(128),
				nn.ReLU(),
				nn.Dropout(0.2),
				nn.Linear(128, 64),
				nn.BatchNorm1d(64),
				nn.ReLU(),
				nn.Dropout(0.2),
				nn.Linear(64, 1),
			),
			"network_name": "FourLayer-512-256-128-64-1",
		},
		{
			"network": lambda input_size: nn.Sequential(
				nn.Linear(input_size, 384),
				nn.BatchNorm1d(384),
				nn.ReLU(),
				nn.Dropout(0.15),
				nn.Linear(384, 192),
				nn.BatchNorm1d(192),
				nn.ReLU(),
				nn.Dropout(0.15),
				nn.Linear(192, 96),
				nn.BatchNorm1d(96),
				nn.ReLU(),
				nn.Dropout(0.15),
				nn.Linear(96, 1),
			),
			"network_name": "Pyramid-384-192-96-1",
		},
	]

	return {a["network_name"]: a["network"] for a in architectures}


def choose_architecture(target: str, df_results: pd.DataFrame, network_name: str | None):
	"""Return the network_name to use, preferring an explicit argument, then the best architecture.

	If network_name is None, this will look for architecture_search_summary.csv
	(created by nn_model_search.py) and choose the best architecture for the
	specified target. If that summary is missing, it falls back to the
	architecture with the highest mean test R^2 in df_results.
	"""

	if network_name is not None:
		return network_name

	# Prefer target-specific summary file
	summary_path = DATADIR / f"architecture_search_summary_{target}.csv"
	if summary_path.exists():
		df_summary = pd.read_csv(summary_path)
		row = df_summary.loc[df_summary["Target"] == target]
		if not row.empty:
			return row.iloc[0]["Best Architecture"]

	# Fallback to legacy summary if present
	legacy_summary = DATADIR / "architecture_search_summary.csv"
	if legacy_summary.exists():
		df_summary = pd.read_csv(legacy_summary)
		row = df_summary.loc[df_summary["Target"] == target]
		if not row.empty:
			return row.iloc[0]["Best Architecture"]

	# Fallback: compute best architecture directly from results
	df_target = df_results[df_results["target"] == target]
	if df_target.empty:
		raise ValueError(f"No results found for target '{target}'.")

	test_col = f"test_{target}_r2"
	if test_col not in df_target.columns:
		raise KeyError(f"Expected column '{test_col}' not found in results.")

	df_avg = (
		df_target.groupby(["experiment_name", "network_name"])[test_col]
		.mean()
		.reset_index()
	)
	best_idx = df_avg[test_col].idxmax()
	return df_avg.loc[best_idx]["network_name"]


def parse_all_pkl_filename(path: Path):
	"""Parse a *_all.pkl filename into (target, network_name, num_epochs).

	Expected pattern: SingleTarget-<TARGET>-OX-<NETWORK_NAME>-<EPOCHS>_all.pkl
	"""
	stem = path.stem  # e.g., SingleTarget-HPLC_Caff-OX-Tiny-64-1-3000_all
	parts = stem.split("-")
	if len(parts) < 5 or parts[0] != "SingleTarget":
		raise ValueError(f"Unexpected _all.pkl filename format: {path.name}")

	target = parts[1]
	# parts[2] is 'OX' or 'REDOX'; network_name spans parts[3:-1]
	network_name = "-".join(parts[3:-1])
	epoch_part = parts[-1]  # '3000_all'
	num_epochs = int(epoch_part.split("_")[0])
	return target, network_name, num_epochs


def recompute_from_all_pkls(target: str, network_name: str | None, show: bool = False):
	"""Re-train models from *_all.pkl files and generate plots.

	If network_name is None, this processes all architectures for the given
	target. Otherwise, it only processes the specified architecture.
	"""
	arch_factories = get_architecture_factories()
	data_dir = DATADIR
	pattern = f"SingleTarget-{target}-*_all.pkl"
	paths = sorted(data_dir.glob(pattern))
	if not paths:
		raise FileNotFoundError(
			f"No *_all.pkl files found for target '{target}' in {data_dir}"
		)

	for path in paths:
		file_target, net_name, num_epochs = parse_all_pkl_filename(path)
		if file_target != target:
			continue
		if network_name is not None and net_name != network_name:
			continue
		if net_name not in arch_factories:
			print(f"Skipping {path.name}: unknown architecture '{net_name}'")
			continue

		print(f"Processing {path.name} (target={file_target}, network={net_name}, epochs={num_epochs})")
		df_all = pd.read_pickle(path)
		coffees = df_all["Coffee Name"].unique()

		train_actual_list = []
		train_pred_list = []
		test_actual_list = []
		test_pred_list = []

		for fold, test_coffee in enumerate(coffees):
			test_mask = df_all["Coffee Name"] == test_coffee
			df_train = df_all[~test_mask]
			df_test = df_all[test_mask]

			# Setup train data
			cv_raw_list = df_train["cv_raw"].to_list()
			X_train = np.array(cv_raw_list)
			y_train = np.array(df_train[target].values).reshape(-1, 1)

			input_size = X_train.shape[1]

			X_scaler = StandardScaler().fit(X_train)
			y_scaler = StandardScaler().fit(y_train)

			X_train_standard = X_scaler.transform(X_train)
			y_train_standard = y_scaler.transform(y_train)

			# Setup test data
			cv_raw_test_list = df_test["cv_raw"].to_list()
			X_test = np.array(cv_raw_test_list)
			y_test = np.array(df_test[target].values).reshape(-1, 1)

			X_test_standard = X_scaler.transform(X_test)
			y_test_standard = y_scaler.transform(y_test)

			model = CoffeeNetBase()
			model.network = arch_factories[net_name](input_size)

			model = train_coffeenet(
				model,
				X_train_standard,
				y_train_standard,
				X_test_standard,
				y_test_standard,
				num_epochs=num_epochs,
			)

			# Evaluate on training data
			train_predictions = evaluate_model(model, X_train_standard, y_train_standard)
			train_predictions_original = y_scaler.inverse_transform(train_predictions)

			train_actual_list.append(y_train.flatten())
			train_pred_list.append(train_predictions_original.flatten())

			# Evaluate on test data
			test_predictions = evaluate_model(model, X_test_standard, y_test_standard)
			test_predictions_original = y_scaler.inverse_transform(test_predictions)

			test_actual_list.append(y_test.flatten())
			test_pred_list.append(test_predictions_original.flatten())

		train_actual_all = np.concatenate(train_actual_list)
		train_pred_all = np.concatenate(train_pred_list)
		test_actual_all = np.concatenate(test_actual_list)
		test_pred_all = np.concatenate(test_pred_list)

		print(f"Generating plot for target={target}, network={net_name}")
		plot_predictions(
			target=target,
			network_name=net_name,
			train_actual=train_actual_all,
			train_pred=train_pred_all,
			test_actual=test_actual_all,
			test_pred=test_pred_all,
			show=show,
		)

		# If the user requested a single architecture, stop after the first one
		if network_name is not None:
			break


def collect_predictions(df_results: pd.DataFrame, target: str, network_name: str):
	"""Collect concatenated train and test predictions/actuals for an architecture."""

	df_sel = df_results[
		(df_results["target"] == target)
		& (df_results["network_name"] == network_name)
	]

	if df_sel.empty:
		raise ValueError(
			f"No rows found for target '{target}' and network '{network_name}'."
		)

	train_actual_list = []
	train_pred_list = []
	test_actual_list = []
	test_pred_list = []

	train_actual_key = f"train_{target}_actual"
	train_pred_key = f"train_{target}_predictions"
	test_actual_key = f"test_{target}_actual"
	test_pred_key = f"test_{target}_predictions"

	for _, row in df_sel.iterrows():
		train_actual = row[train_actual_key]
		train_pred = row[train_pred_key]
		test_actual = row[test_actual_key]
		test_pred = row[test_pred_key]

		if isinstance(train_actual, np.ndarray) and isinstance(train_pred, np.ndarray):
			train_actual_list.append(train_actual.reshape(-1))
			train_pred_list.append(train_pred.reshape(-1))
		if isinstance(test_actual, np.ndarray) and isinstance(test_pred, np.ndarray):
			test_actual_list.append(test_actual.reshape(-1))
			test_pred_list.append(test_pred.reshape(-1))

	if not train_actual_list or not test_actual_list:
		raise ValueError(
			"Results do not contain prediction arrays for the requested target/architecture."
		)

	train_actual_all = np.concatenate(train_actual_list)
	train_pred_all = np.concatenate(train_pred_list)
	test_actual_all = np.concatenate(test_actual_list)
	test_pred_all = np.concatenate(test_pred_list)

	return train_actual_all, train_pred_all, test_actual_all, test_pred_all


def plot_predictions(
	target: str,
	network_name: str,
	train_actual: np.ndarray,
	train_pred: np.ndarray,
	test_actual: np.ndarray,
	test_pred: np.ndarray,
	show: bool = False,
):
	"""Generate a scatter plot of predicted vs actual with train and test points."""

	setup_mplt()

	# Compute overall metrics
	train_r2 = r2_score(train_actual, train_pred)
	train_mae = mean_absolute_error(train_actual, train_pred)
	test_r2 = r2_score(test_actual, test_pred)
	test_mae = mean_absolute_error(test_actual, test_pred)

	fig, ax = plt.subplots(1, 1)

	ax.scatter(
		train_actual,
		train_pred,
		s=40,
		alpha=0.6,
		edgecolor="k",
		linewidth=0.5,
		label=f"Train (n={len(train_actual)})",
	)
	ax.scatter(
		test_actual,
		test_pred,
		s=60,
		alpha=0.85,
		edgecolor="k",
		linewidth=0.6,
		label=f"Test (n={len(test_actual)})",
	)

	# Identity line
	all_actual = np.concatenate([train_actual, test_actual])
	line_min = all_actual.min()
	line_max = all_actual.max()
	ax.plot([line_min, line_max], [line_min, line_max], "k--", lw=1, label="Ideal")

	ax.set_xlabel(f"Actual {target}")
	ax.set_ylabel(f"Predicted {target}")
	ax.set_title(
		f"{target} – {network_name}\n"
		f"Train R²={train_r2:.3f}, MAE={train_mae:.2f}; "
		f"Test R²={test_r2:.3f}, MAE={test_mae:.2f}"
	)
	ax.legend(frameon=True)

	plot_dir = PLOTDIR / "model_search"
	plot_dir.mkdir(parents=True, exist_ok=True)
	safe_target = target.replace("/", "-")
	safe_network = network_name.replace("/", "-")
	out_path = plot_dir / f"{safe_target}_{safe_network}_pred_vs_actual.pdf"
	fig.tight_layout()
	fig.savefig(out_path)

	if show:
		plt.show()
	else:
		plt.close(fig)

	print(f"Saved plot to {out_path}")


def main():
	import argparse

	parser = argparse.ArgumentParser(
		description="Plot train and test predictions from nn_model_search results."
	)
	parser.add_argument(
		"--target",
		type=str,
		choices=["HPLC_Caff", "HPLC_CGA", "TDS"],
		required=True,
		help="Target variable to evaluate.",
	)
	parser.add_argument(
		"--network",
		type=str,
		default=None,
		help=(
			"Optional network_name to evaluate. If omitted, the best "
			"architecture from architecture_search_summary.csv (or the "
			"highest test R² in the results) will be used."
		),
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Display the plot interactively in addition to saving it.",
	)

	args = parser.parse_args()

	# Try to use aggregated results if available; otherwise, fall back to
	# recomputing from the individual *_all.pkl files.
	df_results = load_results(args.target)
	if df_results is not None:
		print("Using aggregated full_window_search_results.pkl")
		network_name = choose_architecture(args.target, df_results, args.network)

		print(f"Using architecture: {network_name}")

		train_actual, train_pred, test_actual, test_pred = collect_predictions(
			df_results, args.target, network_name
		)

		plot_predictions(
			target=args.target,
			network_name=network_name,
			train_actual=train_actual,
			train_pred=train_pred,
			test_actual=test_actual,
			test_pred=test_pred,
			show=args.show,
		)
	else:
		print(
			"full_window_search_results.pkl not found; "
			"recomputing metrics directly from *_all.pkl files."
		)
		recompute_from_all_pkls(target=args.target, network_name=args.network, show=args.show)


if __name__ == "__main__":
	main()

