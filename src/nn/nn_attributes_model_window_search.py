
# --- repo-root bootstrap (added by reorg) ---
import sys as _sys
from pathlib import Path as _Path
_repo_root = _Path(__file__).resolve().parents[2]
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))
# --- end bootstrap ---

from util import DATADIR, PLOTDIR, read_coffehub
from nn_0_synthetic_data_gen import build_model_data
from nn_1_train_model import CoffeeNetBase, train_coffeenet, evaluate_model

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import argparse

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
# Set to True to use test set for early stopping (optimistic bias).
# Set to False to train for fixed epochs (no cheating/leakage).
USE_EARLY_STOPPING = False


# ------------------------------------------------------------
# Voltage Window Generator (0.2 V resolution)
# ------------------------------------------------------------


def generate_voltage_windows():
    voltage_steps = np.round(np.arange(0.0, 2.0 + 0.0001, 0.2), 2)
    windows = []
    for vmin in voltage_steps:
        for vmax in voltage_steps:
            if vmax > vmin:
                windows.append((vmin, vmax))
    return windows


"""Utilities to generate and apply voltage windows.

The original implementation assumed cv_raw was a 2D array with an
explicit voltage column. In this project cv_raw is a 1D current trace
over an implicit 0–2 V grid, so windowing is done via precomputed
index masks instead (see main search loop).
"""


# ------------------------------------------------------------
# Architectures
# ------------------------------------------------------------


def get_network_architectures():
    """Return a list of candidate network architectures.

    Each entry is a dict with:
      - 'network': lambda input_size -> nn.Module
      - 'network_name': human-readable string
    """

    return [

        # =================================================
        # 1–8  : VERY SMALL / MINIMAL
        # =================================================

        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 8),
                nn.ReLU(),
                nn.Linear(8, output_size)
            ),
            'network_name': 'Tiny-8-1'
        },
        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 16),
                nn.ReLU(),
                nn.Linear(16, output_size)
            ),
            'network_name': 'Tiny-16-1'
        },
        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Linear(32, output_size)
            ),
            'network_name': 'Tiny-32-1'
        },
        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 16),
                nn.Tanh(),
                nn.Linear(16, output_size)
            ),
            'network_name': 'Tiny-16-1-Tanh'
        },
        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, output_size)
            ),
            'network_name': 'Tiny-32-1-Leaky'
        },
        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, output_size)
            ),
            'network_name': 'Small-32-16-1'
        },
        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, output_size)
            ),
            'network_name': 'Small-64-16-1'
        },
        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_size)
            ),
            'network_name': 'Small-64-32-1'
        },

        # =================================================
        # 9–14 : SMALL WITH BN / DROPOUT
        # =================================================

        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, output_size)
            ),
            'network_name': 'SmallBN-64-1'
        },
        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_size)
            ),
            'network_name': 'SmallBN-64-32-1'
        },
        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_size)
            ),
            'network_name': 'SmallBN-128-64-1'
        },
        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, output_size)
            ),
            'network_name': 'SmallDrop-128-32-1'
        },
        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_size)
            ),
            'network_name': 'Small-128-64-32-1'
        },
        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, output_size)
            ),
            'network_name': 'Medium-256-64-1'
        },

        # =================================================
        # 15–18 : DEEP BUT NARROW
        # =================================================

        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, output_size)
            ),
            'network_name': 'Deep-32x3'
        },
        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, output_size)
            ),
            'network_name': 'Deep-64x3'
        },
        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, output_size)
            ),
            'network_name': 'Deep-128x3'
        },
        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, output_size)
            ),
            'network_name': 'DeepBN-64x3'
        },

        # =================================================
        # 19–22 : MEDIUM / HIGH CAPACITY
        # =================================================

        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, output_size)
            ),
            'network_name': 'Medium-256-128-1'
        },
        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, output_size)
            ),
            'network_name': 'Wide-512-128-1'
        },
        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, output_size)
            ),
            'network_name': 'WideBN-512-256-1'
        },
        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 768),
                nn.ReLU(),
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Linear(256, output_size)
            ),
            'network_name': 'Wide-768-256-1'
        },

        # =================================================
        # 23–25 : LARGER MODELS
        # =================================================

        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, output_size)
            ),
            'network_name': 'Large-1024-256-1'
        },
        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, output_size)
            ),
            'network_name': 'LargeBN-1024-512-1'
        },
        {
            'network': lambda input_size, output_size: nn.Sequential(
                nn.Linear(input_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, output_size)
            ),
            'network_name': 'Large-1024-512-256-1'
        },
    ]


# ------------------------------------------------------------
# MAIN SEARCH + CV PLOTTING
# ------------------------------------------------------------


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Search best voltage window + network architecture for attribute prediction. "
            "By default this runs multi-target prediction over all configured attribute columns. "
            "Use --attribute-target to search for a single target column."
        )
    )
    parser.add_argument(
        "--attribute-target",
        type=str,
        default=None,
        help=(
            "Optional: run a single-target search for this attribute column (case-insensitive). "
            "Example: --attribute-target Brightness"
        ),
    )
    args = parser.parse_args()

    # Multi-target attribute prediction
    ATTRIBUTE_COLUMNS_REQUESTED = [
        "Brightness",
        "Flavor",
        "Body",
        "Finish",
        "Sweetness",
        "Clean Cup",
        "Complexity",
        "Uniformity",
        "Fragrance",
        "Wet Aroma",
    ]

    requested_target = args.attribute_target
    if requested_target is not None:
        requested_target = requested_target.strip()
        if requested_target == "":
            requested_target = None

    require_columns = ATTRIBUTE_COLUMNS_REQUESTED
    if requested_target is not None:
        lowered = {c.lower(): c for c in ATTRIBUTE_COLUMNS_REQUESTED}
        if requested_target.lower() not in lowered:
            raise ValueError(
                "Unknown --attribute-target. Choose one of: "
                + ", ".join(ATTRIBUTE_COLUMNS_REQUESTED)
            )
        requested_target = lowered[requested_target.lower()]
        require_columns = [requested_target]

    target_name = "Attributes" if requested_target is None else f"Attribute_{requested_target}"
    safe_target_name = target_name.replace(" ", "_")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: CUDA not available, using CPU. This search may be very slow.")

    # Load once: full-voltage CV data (no normalization, no bins)
    df_all = build_model_data(
        NORMALIZE=False,
        REDOX=False,
        USE_BINS=False,
        test_train_split=False,
        require_columns=require_columns,
    )

    if requested_target is None:
        missing_cols = [c for c in ATTRIBUTE_COLUMNS_REQUESTED if c not in df_all.columns]
        if missing_cols:
            raise KeyError(
                "Missing required attribute columns in dataset: " + ", ".join(missing_cols)
            )
        ATTRIBUTE_COLUMNS = ATTRIBUTE_COLUMNS_REQUESTED
    else:
        if requested_target not in df_all.columns:
            raise KeyError(
                f"Requested attribute target '{requested_target}' not found in dataset columns. "
                "Available columns include: " + ", ".join(map(str, df_all.columns))
            )
        ATTRIBUTE_COLUMNS = [requested_target]

    output_size = len(ATTRIBUTE_COLUMNS)

    print(
        f"\nSearching best window + architecture for {target_name} "
        f"({output_size} targets) on {device.type}\n"
    )

    # Drop samples missing any target attribute
    n_before = len(df_all)
    df_all = df_all.dropna(subset=ATTRIBUTE_COLUMNS)
    n_after = len(df_all)
    if n_after < n_before:
        print(f"Dropped {n_before - n_after} samples with missing attributes")

    # Confirm dataset size up-front
    n_coffees = df_all['Coffee Name'].nunique()
    print(f"Loaded {n_after} samples across {n_coffees} unique coffees")

    # Assume underlying CV spans 0 to 2 V with uniform spacing
    n_points = len(df_all['cv_raw'].iloc[0])
    full_voltage_array = np.linspace(0.0, 2.0, n_points)

    coffees = df_all['Coffee Name'].unique()
    windows = generate_voltage_windows()
    architectures = get_network_architectures()

    total_runs = len(windows) * len(architectures)
    pbar = tqdm(total=total_runs, desc=f"{target_name} Search")

    best_score = -np.inf
    best_model_state = None
    best_metadata = None
    # Store the CV test predictions/actuals for the best config
    best_test_actual = None
    best_test_pred = None

    # ------------------------------------------------------------
    # SEARCH LOOP (leave-one-coffee-out for each config)
    # ------------------------------------------------------------

    for vmin, vmax in windows:

        # Indices corresponding to this voltage window
        voltage_mask = (full_voltage_array >= vmin) & (full_voltage_array <= vmax)
        voltage_indices = np.where(voltage_mask)[0]

        # Skip windows that contain no points in the discretized CV
        if len(voltage_indices) == 0:
            continue

        for arch in architectures:

            all_test_actual = []
            all_test_pred = []

            for test_coffee in coffees:

                test_mask = df_all['Coffee Name'] == test_coffee
                df_train = df_all[~test_mask]
                df_test = df_all[test_mask]

                # Slice current curves to this voltage window using precomputed indices
                X_train = np.array([
                    np.asarray(cv)[voltage_indices]
                    for cv in df_train['cv_raw']
                ])

                X_test = np.array([
                    np.asarray(cv)[voltage_indices]
                    for cv in df_test['cv_raw']
                ])

                if X_train.shape[1] == 0:
                    continue

                y_train = df_train[ATTRIBUTE_COLUMNS].values
                y_test = df_test[ATTRIBUTE_COLUMNS].values

                # Scale
                X_scaler = StandardScaler().fit(X_train)
                y_scaler = StandardScaler().fit(y_train)

                X_train_s = X_scaler.transform(X_train)
                y_train_s = y_scaler.transform(y_train)

                X_test_s = X_scaler.transform(X_test)
                y_test_s = y_scaler.transform(y_test)

                # Build model
                model = CoffeeNetBase()
                model.network = arch['network'](X_train.shape[1], output_size)
                model.to(device)

                # Train
                # If USE_EARLY_STOPPING is True, we pass the test set to enable early stopping.
                # If False, we pass None, so the model trains for the full 'num_epochs' without peeking.
                val_X = X_test_s if USE_EARLY_STOPPING else None
                val_y = y_test_s if USE_EARLY_STOPPING else None

                model = train_coffeenet(
                    model,
                    X_train_s, y_train_s,
                    val_X, val_y,
                    num_epochs=2000
                )

                # Evaluate on held-out coffee (inverse-transform back to original scale)

                test_pred = evaluate_model(model, X_test_s, y_test_s)
                if isinstance(test_pred, torch.Tensor):
                    test_pred = test_pred.detach().cpu().numpy()
                test_pred_original = y_scaler.inverse_transform(test_pred)

                all_test_actual.append(y_test)
                all_test_pred.append(test_pred_original)

            if len(all_test_actual) == 0:
                pbar.update(1)
                continue

            actual_mat = np.vstack(all_test_actual)
            pred_mat = np.vstack(all_test_pred)

            per_target_r2 = {}
            for idx, col in enumerate(ATTRIBUTE_COLUMNS):
                try:
                    per_target_r2[col] = float(r2_score(actual_mat[:, idx], pred_mat[:, idx]))
                except ValueError:
                    per_target_r2[col] = float("nan")

            mean_r2 = float(np.nanmean(list(per_target_r2.values())))

            # Keep best only (based on mean CV test R² across targets)
            if mean_r2 > best_score:
                best_score = mean_r2
                best_model_state = model.state_dict()
                best_metadata = {
                    'target': target_name,
                    'target_columns': ATTRIBUTE_COLUMNS,
                    'window': (vmin, vmax),
                    'architecture': arch['network_name'],
                    'mean_r2': mean_r2,
                    'per_target_r2': per_target_r2,
                }
                # Capture the exact CV test points that produced best_score
                best_test_actual = actual_mat
                best_test_pred = pred_mat

            pbar.update(1)

    pbar.close()

    # ------------------------------------------------------------
    # SAVE BEST MODEL + METADATA
    # ------------------------------------------------------------

    suffix = "_earlystop" if USE_EARLY_STOPPING else "_fixed"
    save_path = DATADIR / f"BEST_{safe_target_name}{suffix}.pkl"

    # Optionally retrain a final model on all data (best window + architecture)
    final_artifacts = {}
    if best_metadata is not None:
        best_vmin, best_vmax = best_metadata['window']
        best_mask = (full_voltage_array >= best_vmin) & (full_voltage_array <= best_vmax)
        best_indices = np.where(best_mask)[0]

        X_all = np.array([
            np.asarray(cv)[best_indices]
            for cv in df_all['cv_raw']
        ])
        y_all = df_all[ATTRIBUTE_COLUMNS].values

        X_scaler_all = StandardScaler().fit(X_all)
        y_scaler_all = StandardScaler().fit(y_all)

        X_all_s = X_scaler_all.transform(X_all)
        y_all_s = y_scaler_all.transform(y_all)

        # Find the matching architecture constructor
        best_arch_ctor = None
        for arch in architectures:
            if arch['network_name'] == best_metadata['architecture']:
                best_arch_ctor = arch['network']
                break

        if best_arch_ctor is not None:
            final_model = CoffeeNetBase()
            final_model.network = best_arch_ctor(X_all.shape[1], output_size)
            final_model.to(device)
            final_model = train_coffeenet(
                final_model,
                X_all_s, y_all_s,
                None, None,
                num_epochs=2000,
            )
            final_artifacts = {
                'final_model_state_dict': final_model.state_dict(),
                'x_scaler': X_scaler_all,
                'y_scaler': y_scaler_all,
                'voltage_window_indices': best_indices,
                'voltage_window': (best_vmin, best_vmax),
            }

    torch.save({
        'model_state_dict': best_model_state,
        'metadata': best_metadata,
        **final_artifacts,
    }, save_path)

    print(f"\nBest Model Saved: {save_path}")
    print(best_metadata)

    # ------------------------------------------------------------
    # PLOT: CV TEST POINTS USED FOR best_score (one plot per target)
    # ------------------------------------------------------------

    if best_test_actual is None or best_test_pred is None:
        print("No CV test predictions stored; cannot generate CV plot.")
        print("Search complete.\n")
    else:
        vmin, vmax = best_metadata['window']
        arch_name = best_metadata['architecture']

        actual_mat = np.asarray(best_test_actual)
        pred_mat = np.asarray(best_test_pred)

        for idx, col in enumerate(ATTRIBUTE_COLUMNS):
            y_true = actual_mat[:, idx]
            y_hat = pred_mat[:, idx]
            try:
                test_r2 = float(r2_score(y_true, y_hat))
            except ValueError:
                test_r2 = float("nan")

            plt.figure(figsize=(6, 6))
            plt.scatter(y_true, y_hat, alpha=0.7, label="CV test (held-out coffees)")

            lim_min = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_hat)]))
            lim_max = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_hat)]))
            plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle='--', color='black')

            plt.xlabel(f"Actual {col}")
            plt.ylabel(f"Predicted {col}")
            plt.title(
                f"{col}\n"
                f"{arch_name}\n"
                f"Window: {vmin}-{vmax} V\n"
                + (
                    f"CV Test R² = {test_r2:.4f}"
                    if output_size == 1
                    else f"CV Test R² = {test_r2:.4f} (Mean across targets = {best_score:.4f})"
                )
            )
            plt.legend()

            safe_col = col.replace(" ", "_")
            plot_path = PLOTDIR / f"BEST_{safe_target_name}_{safe_col}{suffix}.png"
            plt.savefig(plot_path, dpi=300)
            plt.close()

            print(f"Plot Saved ({col}): {plot_path} (R²={test_r2:.4f})")

        print("\nSearch complete.\n")
