
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
import argparse
import matplotlib.pyplot as plt
import os

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
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 8),
                nn.ReLU(),
                nn.Linear(8, 1)
            ),
            'network_name': 'Tiny-8-1'
        },
        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            ),
            'network_name': 'Tiny-16-1'
        },
        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ),
            'network_name': 'Tiny-32-1'
        },
        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 16),
                nn.Tanh(),
                nn.Linear(16, 1)
            ),
            'network_name': 'Tiny-16-1-Tanh'
        },
        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 1)
            ),
            'network_name': 'Tiny-32-1-Leaky'
        },
        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            ),
            'network_name': 'Small-32-16-1'
        },
        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            ),
            'network_name': 'Small-64-16-1'
        },
        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ),
            'network_name': 'Small-64-32-1'
        },

        # =================================================
        # 9–14 : SMALL WITH BN / DROPOUT
        # =================================================

        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1)
            ),
            'network_name': 'SmallBN-64-1'
        },
        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ),
            'network_name': 'SmallBN-64-32-1'
        },
        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ),
            'network_name': 'SmallBN-128-64-1'
        },
        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ),
            'network_name': 'SmallDrop-128-32-1'
        },
        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ),
            'network_name': 'Small-128-64-32-1'
        },
        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ),
            'network_name': 'Medium-256-64-1'
        },

        # =================================================
        # 15–18 : DEEP BUT NARROW
        # =================================================

        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ),
            'network_name': 'Deep-32x3'
        },
        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ),
            'network_name': 'Deep-64x3'
        },
        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ),
            'network_name': 'Deep-128x3'
        },
        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ),
            'network_name': 'DeepBN-64x3'
        },

        # =================================================
        # 19–22 : MEDIUM / HIGH CAPACITY
        # =================================================

        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ),
            'network_name': 'Medium-256-128-1'
        },
        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ),
            'network_name': 'Wide-512-128-1'
        },
        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ),
            'network_name': 'WideBN-512-256-1'
        },
        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 768),
                nn.ReLU(),
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ),
            'network_name': 'Wide-768-256-1'
        },

        # =================================================
        # 23–25 : LARGER MODELS
        # =================================================

        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ),
            'network_name': 'Large-1024-256-1'
        },
        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            ),
            'network_name': 'LargeBN-1024-512-1'
        },
        {
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ),
            'network_name': 'Large-1024-512-256-1'
        },
    ]


# ------------------------------------------------------------
# MAIN SEARCH + CV PLOTTING
# ------------------------------------------------------------


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--target",
                        type=str,
                        choices=['HPLC_Caff', 'HPLC_CGA', 'TDS'],
                        required=True)
    args = parser.parse_args()

    target_name = args.target

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: CUDA not available, using CPU. This search may be very slow.")

    print(f"\nSearching best window + architecture for {target_name} on {device.type}\n")

    # Load once: full-voltage CV data (no normalization, no bins)
    df_all = build_model_data(
        NORMALIZE=False,
        REDOX=False,
        USE_BINS=False,
        test_train_split=False,
    )

    # Assume underlying CV spans 0 to 2 V with uniform spacing
    n_points = len(df_all['cv_raw'].iloc[0])
    full_voltage_array = np.linspace(0.0, 2.0, n_points)

    coffees = df_all['Coffee Name'].unique()
    print(f"Number of coffees (LOCO folds): {len(coffees)}")
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

                y_train = df_train[target_name].values.reshape(-1, 1)
                y_test = df_test[target_name].values.reshape(-1, 1)

                # Scale
                X_scaler = StandardScaler().fit(X_train)
                y_scaler = StandardScaler().fit(y_train)

                X_train_s = X_scaler.transform(X_train)
                y_train_s = y_scaler.transform(y_train)

                X_test_s = X_scaler.transform(X_test)
                y_test_s = y_scaler.transform(y_test)

                # Build model
                model = CoffeeNetBase()
                model.network = arch['network'](X_train.shape[1])
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
                test_pred_original = y_scaler.inverse_transform(test_pred)

                all_test_actual.extend(y_test.flatten())
                all_test_pred.extend(test_pred_original.flatten())

            if len(all_test_actual) == 0:
                pbar.update(1)
                continue

            cv_r2 = r2_score(all_test_actual, all_test_pred)

            # Keep best only (based purely on CV test R²)
            if cv_r2 > best_score:
                best_score = cv_r2
                best_model_state = model.state_dict()
                best_metadata = {
                    'target': target_name,
                    # Cast to Python floats so torch.load(weights_only=True) stays happy
                    # on newer PyTorch versions.
                    'window': (float(vmin), float(vmax)),
                    'architecture': arch['network_name'],
                    'r2': float(cv_r2)
                }
                # Capture the exact CV test points that produced best_score
                best_test_actual = [float(x) for x in all_test_actual]
                best_test_pred = [float(x) for x in all_test_pred]

            pbar.update(1)

    pbar.close()

    # ------------------------------------------------------------
    # SAVE BEST MODEL + METADATA
    # ------------------------------------------------------------

    suffix = "_earlystop" if USE_EARLY_STOPPING else "_fixed"
    save_path = DATADIR / f"BEST_{target_name}{suffix}.pkl"

    torch.save({
        'model_state_dict': best_model_state,
        'metadata': best_metadata,
        # Save the CV test points for the best config so you can re-plot later
        # without re-running the whole search.
        'cv_test_actual': best_test_actual,
        'cv_test_pred': best_test_pred,
    }, save_path)

    print(f"\nBest Model Saved: {save_path}")
    print(best_metadata)

    # ------------------------------------------------------------
    # PLOT: CV TEST POINTS USED FOR best_score
    # ------------------------------------------------------------

    if best_test_actual is None or best_test_pred is None:
        print("No CV test predictions stored; cannot generate CV plot.")
        print("Search complete.\n")
    else:
        vmin, vmax = best_metadata['window']
        arch_name = best_metadata['architecture']

        test_actual = best_test_actual
        test_pred = best_test_pred

        # R² here should match best_score (CV search R²)
        test_r2 = r2_score(test_actual, test_pred)

        plt.figure(figsize=(6, 6))

        plt.scatter(test_actual, test_pred, alpha=0.7, label="CV test (held-out coffees)")

        lims = [
            min(test_actual + test_pred),
            max(test_actual + test_pred)
        ]

        plt.plot(lims, lims, linestyle='--', color='black')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(
            f"{target_name}\n"
            f"{arch_name}\n"
            f"Window: {vmin}-{vmax} V\n"
            f"CV Test R² = {test_r2:.4f} (CV search R² = {best_score:.4f})"
        )

        plt.legend()
        plot_path = PLOTDIR / f"BEST_{target_name}{suffix}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"Plot Saved (CV test points): {plot_path}")
        print(f"CV Test R² (from stored predictions): {test_r2:.4f}")
        print("\nSearch complete.\n")
