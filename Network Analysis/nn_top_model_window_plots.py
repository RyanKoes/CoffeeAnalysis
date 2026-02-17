import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score, mean_absolute_error

from util import DATADIR, PLOTDIR, setup_mplt
from nn_1_train_model import CoffeeNetBase
from nn_top_model_window_search import get_top_model_configs


def _load_window_results_for_target(target: str) -> tuple[pd.DataFrame, Path]:
    """Load detailed window-search results for a given target.

    Tries per-target files first, then falls back to the all-targets file.
    Handles both PKL and CSV formats.
    """

    candidates = [
        DATADIR / f"topmodel_window_search_results_{target}.pkl",
        DATADIR / f"topmodel_window_search_results_{target}.csv",
        DATADIR / "topmodel_window_search_results_all_targets.pkl",
        DATADIR / "topmodel_window_search_results_all_targets.csv",
    ]

    last_error = None

    for path in candidates:
        if not path.exists():
            continue

        try:
            if path.suffix == ".pkl":
                df = pd.read_pickle(path)
            else:
                df = pd.read_csv(path)

            if "target" in df.columns:
                df = df[df["target"] == target]

            if df.empty:
                continue

            return df, path
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
            continue

    if last_error is not None:
        raise RuntimeError(
            f"Failed to load window-search results for target '{target}' from candidates: "
            f"{[p.name for p in candidates if p.exists()]}. Last error: {last_error}"
        )

    raise FileNotFoundError(
        f"No non-empty window-search results found for target '{target}'. "
        f"Looked for: {[p.name for p in candidates]} in {DATADIR}."
    )



def _get_experiment_base_from_model_path(model_path: Path) -> str:
    """Infer the base experiment name used for the cached *_all.pkl file.

    Model paths look like:
        TopWindow-<target>-OX-<network>-<epochs>-V<start>-<end>-fold-<k>.pth
    The cached data is stored as:
        <experiment_base>_all.pkl
    where experiment_base is the part before "-V...".
    """

    stem = model_path.stem  # strip .pth
    if "-fold-" in stem:
        stem, _ = stem.rsplit("-fold-", 1)
    if "-V" in stem:
        stem, _ = stem.rsplit("-V", 1)
    return stem


def _load_base_data_for_window(df_window: pd.DataFrame) -> pd.DataFrame:
    """Load the cached full-voltage data for the given window.

    Uses the first row's model_path to reconstruct the *_all.pkl path.
    """

    first_row = df_window.iloc[0]
    model_path = first_row["model_path"]
    if isinstance(model_path, str):
        model_path = Path(model_path)
    elif isinstance(model_path, Path):
        pass
    else:
        model_path = Path(str(model_path))

    experiment_base = _get_experiment_base_from_model_path(model_path)
    all_data_path = DATADIR / f"{experiment_base}_all.pkl"

    if not all_data_path.exists():
        raise FileNotFoundError(
            f"Cached base data file not found for experiment '{experiment_base}'. "
            f"Expected at {all_data_path}"
        )

    return pd.read_pickle(all_data_path)


def _collect_predictions_for_window(
    df_results: pd.DataFrame,
    target: str,
    window_row: pd.Series,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """Recompute train/test predictions for a single (network, window).

    Returns concatenated (train_actual, train_pred, test_actual, test_pred, train_r2,
    train_mae, test_r2, test_mae).
    """

    network_name = window_row["network_name"]
    v_start = float(window_row["v_start"])
    v_end = float(window_row["v_end"])

    df_window = df_results[
        (df_results["network_name"] == network_name)
        & (df_results["v_start"] == v_start)
        & (df_results["v_end"] == v_end)
        & (df_results["target"] == target)
    ]

    if df_window.empty:
        raise ValueError(
            f"No detailed rows found for target={target}, network={network_name}, "
            f"window=({v_start}, {v_end})."
        )

    df_all_full = _load_base_data_for_window(df_window)

    n_points = len(df_all_full["cv_raw"].iloc[0])
    full_voltage_array = np.linspace(0, 2, n_points)
    voltage_mask = (full_voltage_array >= v_start) & (full_voltage_array <= v_end)
    voltage_indices = np.where(voltage_mask)[0]

    if voltage_indices.size == 0:
        raise ValueError(
            f"Window {v_start:.2f}-{v_end:.2f} V has no voltage points after masking."
        )

    df_all = df_all_full.copy()
    df_all["cv_raw"] = df_all["cv_raw"].apply(lambda x: np.asarray(x)[voltage_indices])

    train_actual_list: list[np.ndarray] = []
    train_pred_list: list[np.ndarray] = []
    test_actual_list: list[np.ndarray] = []
    test_pred_list: list[np.ndarray] = []

    top_configs = get_top_model_configs()
    if target not in top_configs:
        raise KeyError(f"Target '{target}' not found in top model configs.")

    configs_for_target = {cfg["network_name"]: cfg for cfg in top_configs[target]}
    if network_name not in configs_for_target:
        raise KeyError(
            f"Network '{network_name}' for target '{target}' not found in top model configs."
        )

    net_cfg = configs_for_target[network_name]

    for _, row in df_window.iterrows():
        test_coffee = row["test_coffee"]
        model_path = row["model_path"]
        if isinstance(model_path, str):
            model_path = Path(model_path)
        elif isinstance(model_path, Path):
            pass
        else:
            model_path = Path(str(model_path))

        if not model_path.exists():
            # Skip folds without a saved model
            continue

        checkpoint = torch.load(model_path, map_location=device)

        X_mean = np.asarray(checkpoint["X_mean"], dtype=float)
        X_std = np.asarray(checkpoint["X_std"], dtype=float)
        y_mean = np.asarray(checkpoint["y_mean"], dtype=float)
        y_std = np.asarray(checkpoint["y_std"], dtype=float)
        input_size = int(checkpoint["input_size"])

        eps = 1e-8
        X_std_safe = np.where(X_std == 0, eps, X_std)

        model = CoffeeNetBase()
        model.network = net_cfg["network"](input_size)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        df_train = df_all[df_all["Coffee Name"] != test_coffee]
        df_test = df_all[df_all["Coffee Name"] == test_coffee]

        if df_test.empty or df_train.empty:
            continue

        X_train_raw = np.vstack(df_train["cv_raw"].apply(np.asarray))
        y_train_raw = df_train[target].to_numpy(dtype=float).reshape(-1, 1)
        X_test_raw = np.vstack(df_test["cv_raw"].apply(np.asarray))
        y_test_raw = df_test[target].to_numpy(dtype=float).reshape(-1, 1)

        if X_train_raw.shape[1] != input_size or X_test_raw.shape[1] != input_size:
            raise ValueError(
                f"Input size mismatch for model {model_path.name}: "
                f"expected {input_size}, got train {X_train_raw.shape[1]}, "
                f"test {X_test_raw.shape[1]}"
            )

        X_train_std = (X_train_raw - X_mean) / X_std_safe
        X_test_std = (X_test_raw - X_mean) / X_std_safe

        with torch.no_grad():
            y_train_pred_std = (
                model(torch.tensor(X_train_std, dtype=torch.float32, device=device))
                .cpu()
                .numpy()
            )
            y_test_pred_std = (
                model(torch.tensor(X_test_std, dtype=torch.float32, device=device))
                .cpu()
                .numpy()
            )

        y_train_pred = y_train_pred_std * y_std + y_mean
        y_test_pred = y_test_pred_std * y_std + y_mean

        train_actual_list.append(y_train_raw.reshape(-1))
        train_pred_list.append(y_train_pred.reshape(-1))
        test_actual_list.append(y_test_raw.reshape(-1))
        test_pred_list.append(y_test_pred.reshape(-1))

    if not train_actual_list or not test_actual_list:
        raise ValueError(
            f"No valid folds with predictions found for target={target}, "
            f"network={network_name}, window=({v_start}, {v_end})."
        )

    train_actual_all = np.concatenate(train_actual_list)
    train_pred_all = np.concatenate(train_pred_list)
    test_actual_all = np.concatenate(test_actual_list)
    test_pred_all = np.concatenate(test_pred_list)

    train_r2 = r2_score(train_actual_all, train_pred_all)
    train_mae = mean_absolute_error(train_actual_all, train_pred_all)
    test_r2 = r2_score(test_actual_all, test_pred_all)
    test_mae = mean_absolute_error(test_actual_all, test_pred_all)

    return (
        train_actual_all,
        train_pred_all,
        test_actual_all,
        test_pred_all,
        train_r2,
        train_mae,
        test_r2,
        test_mae,
    )


def _plot_window(
    target: str,
    window_row: pd.Series,
    train_actual: np.ndarray,
    train_pred: np.ndarray,
    test_actual: np.ndarray,
    test_pred: np.ndarray,
    train_r2: float,
    train_mae: float,
    test_r2: float,
    test_mae: float,
    out_dir: Path,
) -> None:
    """Generate and save a prediction-vs-actual scatter plot for one window."""

    setup_mplt()

    network_name = window_row["network_name"]
    v_start = float(window_row["v_start"])
    v_end = float(window_row["v_end"])
    rank = int(window_row.get("rank", -1))

    fig, ax = plt.subplots(1, 1)

    ax.scatter(
        train_actual,
        train_pred,
        s=30,
        alpha=0.6,
        edgecolor="k",
        linewidth=0.4,
        label=f"Train (n={len(train_actual)})",
    )
    ax.scatter(
        test_actual,
        test_pred,
        s=50,
        alpha=0.85,
        edgecolor="k",
        linewidth=0.6,
        label=f"Test (n={len(test_actual)})",
    )

    all_actual = np.concatenate([train_actual, test_actual])
    line_min = all_actual.min()
    line_max = all_actual.max()
    ax.plot([line_min, line_max], [line_min, line_max], "k--", lw=1, label="Ideal")

    title_prefix = f"Rank {rank} – " if rank > 0 else ""
    ax.set_xlabel(f"Actual {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title(
        f"{title_prefix}{target} – {network_name}\n"
        f"Window {v_start:.2f}–{v_end:.2f} V | "
        f"Train R²={train_r2:.3f}, MAE={train_mae:.2f}; "
        f"Test R²={test_r2:.3f}, MAE={test_mae:.2f}"
    )
    ax.legend(frameon=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    safe_target = str(target).replace("/", "-")
    safe_network = str(network_name).replace("/", "-")
    fname_parts = []
    if rank > 0:
        fname_parts.append(f"{rank:02d}")
    fname_parts.append(safe_target)
    fname_parts.append(safe_network)
    fname_parts.append(f"V{v_start:.2f}-{v_end:.2f}")
    filename = "_".join(fname_parts) + "_pred_vs_actual.png"

    out_path = out_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved window plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot top windows from nn_top_model_window_search results by recomputing "
            "R²/MAE from stored models and cached data."
        )
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["HPLC_Caff", "HPLC_CGA", "TDS", "all"],
        default="all",
        help="Target variable to evaluate (default: all).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top windows (by mean test R²) to plot per target.",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.target == "all":
        targets = ["HPLC_Caff", "HPLC_CGA", "TDS"]
    else:
        targets = [args.target]

    for target in targets:
        print(f"\n=== Processing target: {target} ===")
        try:
            df_results, path_used = _load_window_results_for_target(target)
        except Exception as exc:
            print(f"Skipping target {target}: {exc}")
            continue

        print(f"Loaded window-search results for {target} from {path_used}")
        # Identify unique (network, window) combinations for this target
        window_keys = ["network_name", "v_start", "v_end", "v_window_size", "n_voltage_points"]
        if not set(window_keys).issubset(df_results.columns):
            print(f"Results for target {target} are missing required window keys; skipping.")
            continue

        df_target = df_results[df_results["target"] == target]
        df_windows = df_target[window_keys].drop_duplicates().reset_index(drop=True)

        candidates = []
        print("Computing recomputed R² for all windows (this may take a while)...")
        for _, win_row in df_windows.iterrows():
            try:
                (
                    _train_actual,
                    _train_pred,
                    _test_actual,
                    _test_pred,
                    train_r2,
                    train_mae,
                    test_r2,
                    test_mae,
                ) = _collect_predictions_for_window(df_results, target, win_row, device)
            except Exception as exc:
                print(
                    f"Skipping window for target={target}, network={win_row['network_name']}, "
                    f"window=({win_row['v_start']}, {win_row['v_end']}): {exc}"
                )
                continue

            rec = win_row.to_dict()
            rec.update(
                {
                    "recomputed_train_r2": train_r2,
                    "recomputed_train_mae": train_mae,
                    "recomputed_test_r2": test_r2,
                    "recomputed_test_mae": test_mae,
                }
            )
            candidates.append(rec)

        if not candidates:
            print(f"No windows with valid recomputed metrics for target {target}; skipping.")
            continue

        df_top = pd.DataFrame(candidates)
        df_top.sort_values("recomputed_test_r2", ascending=False, inplace=True)
        df_top.reset_index(drop=True, inplace=True)

        top_n = max(1, min(args.top_n, len(df_top)))
        df_top = df_top.iloc[:top_n].copy()
        df_top["rank"] = np.arange(1, len(df_top) + 1)

        print("Top windows by recomputed test R²:")
        cols_to_show = [
            "rank",
            "network_name",
            "v_start",
            "v_end",
            "v_window_size",
            "n_voltage_points",
            "recomputed_test_r2",
            "recomputed_test_mae",
        ]
        print(df_top[cols_to_show].to_string(index=False))

        out_dir = PLOTDIR / "topmodel_window_search" / target
        plots_count = 0

        for _, row in df_top.iterrows():
            try:
                (
                    train_actual,
                    train_pred,
                    test_actual,
                    test_pred,
                    train_r2,
                    train_mae,
                    test_r2,
                    test_mae,
                ) = _collect_predictions_for_window(df_results, target, row, device)
            except Exception as exc:
                print(
                    f"Failed to compute predictions for target={target}, "
                    f"network={row['network_name']}, window=({row['v_start']}, {row['v_end']}): {exc}"
                )
                continue

            _plot_window(
                target,
                row,
                train_actual,
                train_pred,
                test_actual,
                test_pred,
                train_r2,
                train_mae,
                test_r2,
                test_mae,
                out_dir,
            )

            plots_count += 1

        print(
            f"Finished target {target}: saved {plots_count} plot(s) in {out_dir}"
        )


if __name__ == "__main__":
    main()
