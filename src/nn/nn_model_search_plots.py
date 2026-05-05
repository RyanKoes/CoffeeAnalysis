import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

from util import DATADIR, PLOTDIR, setup_mplt


def _parse_array_field(value):
    """Convert a stored prediction/actual field to a 1D numpy array.

    Handles both numpy arrays (from PKL) and string representations written to CSV,
    e.g. "[ 920.36 919.89 ... ]".
    Returns np.ndarray or None if parsing fails.
    """
    if isinstance(value, np.ndarray):
        return value.reshape(-1)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # Remove surrounding brackets if present
        if s[0] == "[" and s[-1] == "]":
            s = s[1:-1]
        try:
            arr = np.fromstring(s, sep=" ")
            return arr.reshape(-1) if arr.size > 0 else None
        except Exception:
            return None
    return None


def load_results_for_target(target: str) -> tuple[pd.DataFrame, Path]:
    """Load the full-window search results for a given target.

    Tries the per-target file first (full_window_search_results_<target>.pkl),
    then falls back to the legacy full_window_search_results.pkl.
    Returns (df_results, path_used).
    """

    candidates = [
        DATADIR / f"full_window_search_results_{target}.pkl",
        DATADIR / "full_window_search_results.pkl",
    ]
    for path in candidates:
        if path.exists():
            # First try PKL
            try:
                df = pd.read_pickle(path)
                if "target" in df.columns:
                    df = df[df["target"] == target]
                if df.empty:
                    raise ValueError(
                        f"Results file {path} does not contain rows for target '{target}'."
                    )
                return df, path
            except Exception:
                # If unpickling fails (e.g., pandas version mismatch), fall back to CSV
                try:
                    csv_path = path.with_suffix(".csv")
                    if not csv_path.exists():
                        raise FileNotFoundError
                    df = pd.read_csv(csv_path)
                    if "target" in df.columns:
                        df = df[df["target"] == target]
                    if df.empty:
                        raise ValueError(
                            f"Results CSV {csv_path} does not contain rows for target '{target}'."
                        )
                    print(
                        f"Warning: failed to read pickle {path.name}, "
                        f"using CSV {csv_path.name} instead."
                    )
                    return df, csv_path
                except Exception:
                    continue

    raise FileNotFoundError(
        f"No readable results file found for target '{target}'. "
        f"Tried PKL/CSV based on {candidates[0].name} and {candidates[1].name} in {DATADIR}."
    )


def aggregate_metrics(df_results: pd.DataFrame, target: str) -> pd.DataFrame:
    """Aggregate train/test metrics per architecture by recomputing from arrays."""

    records = []

    train_actual_key = f"train_{target}_actual"
    train_pred_key = f"train_{target}_predictions"
    test_actual_key = f"test_{target}_actual"
    test_pred_key = f"test_{target}_predictions"

    for network_name, df_net in df_results.groupby("network_name"):
        train_actual_list = []
        train_pred_list = []
        test_actual_list = []
        test_pred_list = []

        for _, row in df_net.iterrows():
            ta = _parse_array_field(row.get(train_actual_key))
            tp = _parse_array_field(row.get(train_pred_key))
            tsa = _parse_array_field(row.get(test_actual_key))
            tsp = _parse_array_field(row.get(test_pred_key))

            if ta is not None and tp is not None:
                train_actual_list.append(ta)
                train_pred_list.append(tp)
            if tsa is not None and tsp is not None:
                test_actual_list.append(tsa)
                test_pred_list.append(tsp)

        if not train_actual_list or not test_actual_list:
            # Skip architectures that for some reason have no stored predictions
            continue

        train_actual_all = np.concatenate(train_actual_list)
        train_pred_all = np.concatenate(train_pred_list)
        test_actual_all = np.concatenate(test_actual_list)
        test_pred_all = np.concatenate(test_pred_list)

        train_r2 = r2_score(train_actual_all, train_pred_all)
        train_mae = mean_absolute_error(train_actual_all, train_pred_all)
        test_r2 = r2_score(test_actual_all, test_pred_all)
        test_mae = mean_absolute_error(test_actual_all, test_pred_all)

        records.append(
            {
                "network_name": network_name,
                "train_r2": train_r2,
                "train_mae": train_mae,
                "test_r2": test_r2,
                "test_mae": test_mae,
                "n_train": len(train_actual_all),
                "n_test": len(test_actual_all),
            }
        )

    if not records:
        raise ValueError("No architectures with valid prediction arrays were found.")

    df_summary = pd.DataFrame(records)
    df_summary.sort_values(by="test_r2", ascending=False, inplace=True)
    df_summary.reset_index(drop=True, inplace=True)
    df_summary["rank"] = np.arange(1, len(df_summary) + 1)
    return df_summary


def plot_architecture(
    df_net: pd.DataFrame,
    target: str,
    network_name: str,
    out_dir: Path,
    is_top: bool,
) -> None:
    """Create a train vs test predicted-vs-actual scatter plot for one architecture."""

    setup_mplt()

    train_actual_key = f"train_{target}_actual"
    train_pred_key = f"train_{target}_predictions"
    test_actual_key = f"test_{target}_actual"
    test_pred_key = f"test_{target}_predictions"

    train_actual_list = []
    train_pred_list = []
    test_actual_list = []
    test_pred_list = []

    for _, row in df_net.iterrows():
        ta = _parse_array_field(row.get(train_actual_key))
        tp = _parse_array_field(row.get(train_pred_key))
        tsa = _parse_array_field(row.get(test_actual_key))
        tsp = _parse_array_field(row.get(test_pred_key))

        if ta is not None and tp is not None:
            train_actual_list.append(ta)
            train_pred_list.append(tp)
        if tsa is not None and tsp is not None:
            test_actual_list.append(tsa)
            test_pred_list.append(tsp)

    if not train_actual_list or not test_actual_list:
        return

    train_actual_all = np.concatenate(train_actual_list)
    train_pred_all = np.concatenate(train_pred_list)
    test_actual_all = np.concatenate(test_actual_list)
    test_pred_all = np.concatenate(test_pred_list)

    train_r2 = r2_score(train_actual_all, train_pred_all)
    train_mae = mean_absolute_error(train_actual_all, train_pred_all)
    test_r2 = r2_score(test_actual_all, test_pred_all)
    test_mae = mean_absolute_error(test_actual_all, test_pred_all)

    fig, ax = plt.subplots(1, 1)

    ax.scatter(
        train_actual_all,
        train_pred_all,
        s=40,
        alpha=0.6,
        edgecolor="k",
        linewidth=0.5,
        label=f"Train (n={len(train_actual_all)})",
    )
    ax.scatter(
        test_actual_all,
        test_pred_all,
        s=60,
        alpha=0.85,
        edgecolor="k",
        linewidth=0.6,
        label=f"Test (n={len(test_actual_all)})",
    )

    all_actual = np.concatenate([train_actual_all, test_actual_all])
    line_min = all_actual.min()
    line_max = all_actual.max()
    ax.plot([line_min, line_max], [line_min, line_max], "k--", lw=1, label="Ideal")

    top_tag = " [TOP]" if is_top else ""
    ax.set_xlabel(f"Actual {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title(
        f"{target} – {network_name}{top_tag}\n"
        f"Train R²={train_r2:.3f}, MAE={train_mae:.2f}; "
        f"Test R²={test_r2:.3f}, MAE={test_mae:.2f}"
    )
    ax.legend(frameon=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_target = target.replace("/", "-")
    safe_network = network_name.replace("/", "-")
    out_path = out_dir / f"{safe_target}_{safe_network}_pred_vs_actual.png"

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved plot for {network_name} to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate nn_model_search results: recompute R² from stored "
            "predictions and generate train/test scatter plots per model."
        )
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["HPLC_Caff", "HPLC_CGA", "TDS"],
        required=True,
        help="Target variable to evaluate.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top models (by test R²) to highlight.",
    )
    parser.add_argument(
        "--only-top",
        action="store_true",
        help="If set, only generate plots for the top-N models.",
    )

    args = parser.parse_args()

    df_results, path_used = load_results_for_target(args.target)
    print(f"Loaded results for target {args.target} from {path_used}")

    df_summary = aggregate_metrics(df_results, args.target)

    # Mark top-N models
    top_n = max(1, args.top_n)
    df_summary["is_top"] = df_summary["rank"] <= top_n

    # Save ranking summary
    ranking_path = DATADIR / f"model_search_ranking_{args.target}.csv"
    df_summary.to_csv(ranking_path, index=False)
    print(f"Saved ranking summary to {ranking_path}")

    # Print a compact view of the top models
    print("\nTop models by test R²:")
    cols_to_show = [
        "rank",
        "network_name",
        "train_r2",
        "test_r2",
        "train_mae",
        "test_mae",
        "n_train",
        "n_test",
    ]
    print(df_summary[df_summary["is_top"]][cols_to_show].to_string(index=False))

    # Generate plots
    out_dir = PLOTDIR / "model_search" / args.target
    for network_name, df_net in df_results.groupby("network_name"):
        is_top = bool(df_summary.loc[df_summary["network_name"] == network_name, "is_top"].any())
        if args.only_top and not is_top:
            continue
        plot_architecture(df_net, args.target, network_name, out_dir, is_top=is_top)


if __name__ == "__main__":
    main()
