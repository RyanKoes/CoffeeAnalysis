
# --- repo-root bootstrap (added by reorg) ---
import sys as _sys
from pathlib import Path as _Path
_repo_root = _Path(__file__).resolve().parents[2]
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))
# --- end bootstrap ---

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
from nn_0_synthetic_data_gen import build_model_data


def _parse_topwindow_checkpoint(path: Path) -> dict:
    """Parse metadata from a TopWindow-*.pth checkpoint filename.

    Expected pattern (stem):
        TopWindow-<target>-OX-<network_name>-<epochs>-V<start>-<end>-fold-<k>
    where <network_name> may itself contain hyphens.
    """

    stem = path.stem

    if "-fold-" not in stem or "-V" not in stem:
        raise ValueError(f"Not a TopWindow checkpoint name: {stem}")

    base, fold_str = stem.rsplit("-fold-", 1)
    fold = int(fold_str)

    base_before_v, window_str = base.rsplit("-V", 1)
    # window_str like "0.2-0.6"
    try:
        v_start_str, v_end_str = window_str.split("-", 1)
    except ValueError as exc:
        raise ValueError(f"Cannot parse voltage window from '{window_str}' in '{stem}'") from exc

    v_start = float(v_start_str)
    v_end = float(v_end_str)

    tokens = base_before_v.split("-")
    if len(tokens) < 5 or tokens[0] != "TopWindow" or tokens[2] != "OX":
        raise ValueError(f"Unexpected TopWindow name structure: {stem}")

    target = tokens[1]
    epochs = int(tokens[-1])
    network_name = "-".join(tokens[3:-1])

    experiment_base = base_before_v  # TopWindow-<target>-OX-<network>-<epochs>

    return {
        "target": target,
        "network_name": network_name,
        "epochs": epochs,
        "v_start": v_start,
        "v_end": v_end,
        "fold": fold,
        "experiment_base": experiment_base,
        "checkpoint_path": path,
    }


def _group_checkpoints_by_window(target_filter=None) -> dict:
    """Scan DATADIR for TopWindow-*.pth files and group them by window.

    Returns a dict mapping
        (target, network_name, v_start, v_end, experiment_base) -> list[meta_dict]
    where each meta_dict is the result of _parse_topwindow_checkpoint.
    """

    groups: dict[tuple, list[dict]] = {}

    for path in DATADIR.glob("TopWindow-*.pth"):
        try:
            meta = _parse_topwindow_checkpoint(path)
        except Exception:
            # Ignore unrelated or malformed files
            continue

        target = meta["target"]
        if target_filter is not None and target != target_filter:
            continue

        key = (
            meta["target"],
            meta["network_name"],
            meta["v_start"],
            meta["v_end"],
            meta["experiment_base"],
        )
        groups.setdefault(key, []).append(meta)

    return groups


def _collect_predictions_for_group(
    target: str,
    network_name: str,
    v_start: float,
    v_end: float,
    experiment_base: str,
    metas: list[dict],
    device: torch.device,
):
    """Recompute train/test predictions for one (target, network, window).

    Uses the *_all.pkl cache plus the list of fold checkpoints.
    """

    all_data_path = DATADIR / f"{experiment_base}_all.pkl"
    if all_data_path.exists():
        df_all_full = pd.read_pickle(all_data_path)
    else:
        # Fallback: rebuild base data using the original experiment config
        top_configs = get_top_model_configs()
        if target not in top_configs:
            raise KeyError(f"Target '{target}' not found in top model configs.")

        configs_for_target = {cfg["network_name"]: cfg for cfg in top_configs[target]}
        if network_name not in configs_for_target:
            raise KeyError(
                f"Network '{network_name}' for target '{target}' not found in top model configs."
            )

        net_cfg = configs_for_target[network_name]

        print(
            "Base cache not found for experiment '{}', building data and caching to '{}'".format(
                experiment_base, all_data_path
            )
        )
        df_all_full = build_model_data(test_train_split=False, **net_cfg)
        df_all_full.to_pickle(all_data_path)

    # Recreate voltage mask exactly as in the training script
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

    coffees = df_all_full["Coffee Name"].unique()

    top_configs = get_top_model_configs()
    if target not in top_configs:
        raise KeyError(f"Target '{target}' not found in top model configs.")

    configs_for_target = {cfg["network_name"]: cfg for cfg in top_configs[target]}
    if network_name not in configs_for_target:
        raise KeyError(
            f"Network '{network_name}' for target '{target}' not found in top model configs."
        )

    net_cfg = configs_for_target[network_name]

    train_actual_list: list[np.ndarray] = []
    train_pred_list: list[np.ndarray] = []
    test_actual_list: list[np.ndarray] = []
    test_pred_list: list[np.ndarray] = []

    for meta in metas:
        fold = meta["fold"]
        ckpt_path = meta["checkpoint_path"]

        if fold < 0 or fold >= len(coffees):
            # Inconsistent with how folds were created; skip
            continue

        test_coffee = coffees[fold]

        if not ckpt_path.exists():
            continue

        checkpoint = torch.load(ckpt_path, map_location=device)

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
                f"Input size mismatch for model {ckpt_path.name}: "
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


def _plot_window_all(
    target: str,
    network_name: str,
    v_start: float,
    v_end: float,
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
    """Generate and save a prediction-vs-actual plot for one window."""

    setup_mplt()

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

    ax.set_xlabel(f"Actual {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title(
        f"{target} – {network_name}\n"
        f"Window {v_start:.2f}–{v_end:.2f} V | "
        f"Train R²={train_r2:.3f}, MAE={train_mae:.2f}; "
        f"Test R²={test_r2:.3f}, MAE={test_mae:.2f}"
    )
    ax.legend(frameon=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    safe_target = target.replace("/", "-")
    safe_network = network_name.replace("/", "-")
    filename = (
        f"{safe_target}_{safe_network}_V{v_start:.2f}-{v_end:.2f}_pred_vs_actual.png"
    )

    out_path = out_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved window plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot every TopWindow checkpoint by recomputing R²/MAE from stored "
            "models and *_all.pkl cached data."
        )
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["HPLC_Caff", "HPLC_CGA", "TDS", "all"],
        default="all",
        help="Target variable to filter on (default: all).",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    target_filter = None if args.target == "all" else args.target

    groups = _group_checkpoints_by_window(target_filter=target_filter)
    if not groups:
        print("No TopWindow checkpoints found matching the requested target(s).")
        return

    total_plots = 0

    for key, metas in sorted(groups.items()):
        target, network_name, v_start, v_end, experiment_base = key
        print(
            f"\nProcessing window: target={target}, network={network_name}, "
            f"V={v_start:.2f}-{v_end:.2f}, experiment={experiment_base}"
        )

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
            ) = _collect_predictions_for_group(
                target,
                network_name,
                v_start,
                v_end,
                experiment_base,
                metas,
                device,
            )
        except Exception as exc:
            print(
                f"Skipping window target={target}, network={network_name}, "
                f"V={v_start:.2f}-{v_end:.2f}: {exc}"
            )
            continue

        out_dir = PLOTDIR / "topmodel_window_search_all" / target
        _plot_window_all(
            target,
            network_name,
            v_start,
            v_end,
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
        total_plots += 1

    print(
        f"\nFinished: saved {total_plots} plot(s) in "
        f"{PLOTDIR / 'topmodel_window_search_all'}"
    )


if __name__ == "__main__":
    main()
