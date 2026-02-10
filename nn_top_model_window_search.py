import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import tabulate
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from util import DATADIR
from nn_0_synthetic_data_gen import build_model_data
from nn_1_train_model import CoffeeNetBase, train_coffeenet, evaluate_model


# Voltage ranges (inclusive) for fixed 0.1 V windows per target
VOLTAGE_RANGES = {
    "HPLC_Caff": (0.8, 1.8),   # Caffeine
    "HPLC_CGA": (0.0, 1.2),    # CGA
    "TDS": (0.0, 1.8),         # TDS
}


def generate_fixed_step_windows(v_min: float, v_max: float, step: float = 0.1):
    """Generate **all** windows on a step grid within [v_min, v_max].

    Windows have start/end on the step grid and width >= step.

    Example (v_min=0.0, v_max=0.3, step=0.1) ->
        [(0.0, 0.1), (0.0, 0.2), (0.0, 0.3),
         (0.1, 0.2), (0.1, 0.3),
         (0.2, 0.3)]
    """
    # Create grid: v_min, v_min+step, ..., v_max
    n_steps = int(round((v_max - v_min) / step))
    grid = [round(v_min + i * step, 2) for i in range(n_steps + 1)]

    windows = []
    for i, v_start in enumerate(grid[:-1]):
        for v_end in grid[i + 1 :]:
            windows.append((v_start, v_end))

    return windows


def get_top_model_configs():
    """Return per-target configs for the top architectures from nn_model_search.

    Each entry mirrors the architecture definitions used there, with num_epochs
    chosen to match your architecture search (3000).
    """
    base = {
        "NORMALIZE": False,
        "REDOX": False,
        "USE_BINS": False,
        "num_epochs": 3000,
    }

    return {
        # Caffeine: TwoLayer-512-128-1, Simple-512-1, Bottleneck-512-64-1
        "HPLC_Caff": [
            dict(
                base,
                **{
                    "network_name": "TwoLayer-512-128-1",
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
                },
            ),
            dict(
                base,
                **{
                    "network_name": "Simple-512-1",
                    "network": lambda input_size: nn.Sequential(
                        nn.Linear(input_size, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(512, 1),
                    ),
                },
            ),
            dict(
                base,
                **{
                    "network_name": "Bottleneck-512-64-1",
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
                },
            ),
        ],
        # CGA: TwoLayer-256-64-1, Simple-256-1, Bottleneck-512-64-1
        "HPLC_CGA": [
            dict(
                base,
                **{
                    "network_name": "TwoLayer-256-64-1",
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
                },
            ),
            dict(
                base,
                **{
                    "network_name": "Simple-256-1",
                    "network": lambda input_size: nn.Sequential(
                        nn.Linear(input_size, 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(256, 1),
                    ),
                },
            ),
            dict(
                base,
                **{
                    "network_name": "Bottleneck-512-64-1",
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
                },
            ),
        ],
        # TDS: Minimal-16-1, Wide-1024-1, Simple-256-1
        "TDS": [
            dict(
                base,
                **{
                    "network_name": "Minimal-16-1",
                    "network": lambda input_size: nn.Sequential(
                        nn.Linear(input_size, 16),
                        nn.ReLU(),
                        nn.Linear(16, 1),
                    ),
                },
            ),
            dict(
                base,
                **{
                    "network_name": "Wide-1024-1",
                    "network": lambda input_size: nn.Sequential(
                        nn.Linear(input_size, 1024),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(1024, 1),
                    ),
                },
            ),
            dict(
                base,
                **{
                    "network_name": "Simple-256-1",
                    "network": lambda input_size: nn.Sequential(
                        nn.Linear(input_size, 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(256, 1),
                    ),
                },
            ),
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Window sweep for top NN architectures per target (0.1 V windows)"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=False,
        choices=["HPLC_Caff", "HPLC_CGA", "TDS", "all"],
        default="all",
        help="Target variable to train on (default: all)",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if device.type == "cpu":
        print("Warning: No GPU found, using CPU for training. Abort.")
        return
    print(f"Using device: {device}")

    top_model_configs = get_top_model_configs()

    # Determine which targets to run
    if args.target == "all":
        target_list = ["HPLC_Caff", "HPLC_CGA", "TDS"]
    else:
        target_list = [args.target]

    exp_results = []

    for target_name in target_list:
        if target_name not in VOLTAGE_RANGES:
            print(f"No voltage range configured for target {target_name}, skipping.")
            continue

        v_min, v_max = VOLTAGE_RANGES[target_name]
        voltage_windows = generate_fixed_step_windows(v_min, v_max, step=0.1)

        print("\n" + "=" * 80)
        print(
            f"Training top models for {target_name} over {len(voltage_windows)} windows "
            f"from {v_min:.2f} V to {v_max:.2f} V in 0.1 V steps"
        )
        print("=" * 80)

        experiments = top_model_configs[target_name]

        for experiment in experiments:
            experiment_name_base = (
                f"TopWindow-{target_name}-OX-"
                f"{experiment['network_name']}-{experiment['num_epochs']}"
            )

            all_data_path = DATADIR / f"{experiment_name_base}_all.pkl"

            # Build or load full-voltage data (no windowing yet)
            if all_data_path.exists():
                print(f"Loading base data for {target_name}, {experiment['network_name']}...")
                df_all_full = pd.read_pickle(all_data_path)
            else:
                print(
                    f"Building base data for {target_name}, {experiment['network_name']} and caching..."
                )
                df_all_full = build_model_data(test_train_split=False, **experiment)
                df_all_full.to_pickle(all_data_path)

            # Assume underlying CV spans 0 to 2 V with uniform spacing
            n_points = len(df_all_full["cv_raw"].iloc[0])
            full_voltage_array = np.linspace(0, 2, n_points)

            coffees = df_all_full["Coffee Name"].unique()

            for v_start, v_end in tqdm(
                voltage_windows,
                desc=f"{target_name} {experiment['network_name']} windows",
            ):
                # Create voltage mask and indices
                voltage_mask = (full_voltage_array >= v_start) & (full_voltage_array <= v_end)
                voltage_indices = np.where(voltage_mask)[0]

                # Skip if very small window in terms of points
                if len(voltage_indices) < 10:
                    print(
                        f"Skipping window {v_start:.2f}-{v_end:.2f} V: only "
                        f"{len(voltage_indices)} points."
                    )
                    continue

                # Windowed copy of the data
                df_all = df_all_full.copy()
                df_all["cv_raw"] = df_all["cv_raw"].apply(
                    lambda x: np.array(x)[voltage_indices]
                )

                experiment_name = (
                    f"{experiment_name_base}-V{v_start:.1f}-{v_end:.1f}"
                )

                # Leave-one-coffee-out cross-validation
                for fold, test_coffee in enumerate(coffees):
                    test_mask = df_all["Coffee Name"] == test_coffee
                    df_train = df_all[~test_mask]
                    df_test = df_all[test_mask]

                    row = {
                        "fold": fold,
                        "test_coffee": test_coffee,
                        "experiment_name": experiment_name,
                        "target": target_name,
                        "network_name": experiment["network_name"],
                        "v_start": v_start,
                        "v_end": v_end,
                        "v_window_size": v_end - v_start,
                        "n_voltage_points": len(voltage_indices),
                    }

                    # Train data
                    X_train = np.array(df_train["cv_raw"].to_list())
                    y_train = (
                        np.array(df_train[target_name].values).reshape(-1, 1)
                    )

                    input_size = X_train.shape[1]

                    X_scaler = StandardScaler().fit(X_train)
                    y_scaler = StandardScaler().fit(y_train)

                    X_train_standard = X_scaler.transform(X_train)
                    y_train_standard = y_scaler.transform(y_train)

                    # Test data
                    X_test = np.array(df_test["cv_raw"].to_list())
                    y_test = (
                        np.array(df_test[target_name].values).reshape(-1, 1)
                    )

                    X_test_standard = X_scaler.transform(X_test)
                    y_test_standard = y_scaler.transform(y_test)

                    model = CoffeeNetBase()
                    model.network = experiment["network"](input_size)

                    model_path = DATADIR / f"{experiment_name}-fold-{fold}.pth"
                    row["model_path"] = model_path

                    if model_path.exists():
                        checkpoint = torch.load(model_path, map_location=device)
                        model.load_state_dict(checkpoint["model_state_dict"])
                        model.to(device)
                    else:
                        model.to(device)
                        model = train_coffeenet(
                            model,
                            X_train_standard,
                            y_train_standard,
                            X_test_standard,
                            y_test_standard,
                            num_epochs=experiment["num_epochs"],
                        )

                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "X_mean": X_scaler.mean_.tolist(),
                                "X_std": X_scaler.scale_.tolist(),
                                "y_mean": y_scaler.mean_.tolist(),
                                "y_std": y_scaler.scale_.tolist(),
                                "input_size": input_size,
                                "v_start": v_start,
                                "v_end": v_end,
                            },
                            model_path,
                        )

                    # Evaluate on training data
                    train_predictions = evaluate_model(
                        model, X_train_standard, y_train_standard
                    )
                    train_predictions_original = y_scaler.inverse_transform(
                        train_predictions
                    )

                    row[f"train_{target_name}_r2"] = r2_score(
                        y_train, train_predictions_original
                    )
                    row[f"train_{target_name}_mae"] = mean_absolute_error(
                        y_train, train_predictions_original
                    )

                    # Evaluate on test data
                    test_predictions = evaluate_model(
                        model, X_test_standard, y_test_standard
                    )
                    test_predictions_original = y_scaler.inverse_transform(
                        test_predictions
                    )

                    row[f"test_{target_name}_r2"] = r2_score(
                        y_test, test_predictions_original
                    )
                    row[f"test_{target_name}_mae"] = mean_absolute_error(
                        y_test, test_predictions_original
                    )

                    exp_results.append(row)

    # Aggregate and save results
    if not exp_results:
        print("No experiments were run; exp_results is empty.")
        return

    df_results = pd.DataFrame(exp_results)

    # Compute mean percent errors if useful later
    # (kept minimal here; can be expanded similarly to nn_model_search)

    suffix = args.target if args.target != "all" else "all_targets"

    results_path = DATADIR / f"topmodel_window_search_results_{suffix}.pkl"
    df_results.to_pickle(results_path)

    csv_path = DATADIR / f"topmodel_window_search_results_{suffix}.csv"
    df_results.to_csv(csv_path, index=False)

    print(f"\nAll detailed results saved to {results_path}")
    print(f"CSV summary saved to {csv_path}")

    # Per-target, per-network best window summary (by mean test R^2)
    summary_rows = []
    for target_name in df_results["target"].unique():
        df_t = df_results[df_results["target"] == target_name]
        metric_r2 = f"test_{target_name}_r2"
        metric_mae = f"test_{target_name}_mae"

        grouped = (
            df_t.groupby(
                ["network_name", "v_start", "v_end", "v_window_size"],
                as_index=False,
            )[[metric_r2, metric_mae, "n_voltage_points"]]
            .mean()
        )

        for net in grouped["network_name"].unique():
            df_net = grouped[grouped["network_name"] == net]
            best_idx = df_net[metric_r2].idxmax()
            best = df_net.loc[best_idx]

            summary_rows.append(
                {
                    "Target": target_name,
                    "Network": net,
                    "Best_v_start": best["v_start"],
                    "Best_v_end": best["v_end"],
                    "Best_window_size": best["v_window_size"],
                    "n_voltage_points": best["n_voltage_points"],
                    "Best_test_r2": best[metric_r2],
                    "Best_test_mae": best[metric_mae],
                }
            )

    df_summary = pd.DataFrame(summary_rows)
    summary_path = DATADIR / f"topmodel_window_search_summary_{suffix}.csv"
    df_summary.to_csv(summary_path, index=False)

    print("\nBest window per (target, network):")
    if not df_summary.empty:
        print(
            tabulate.tabulate(
                df_summary,
                headers="keys",
                tablefmt="psql",
                floatfmt=".4f",
                showindex=False,
            )
        )
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
