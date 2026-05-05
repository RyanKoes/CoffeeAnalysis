
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


class _Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class TimeSeriesCNN1DRegressor(nn.Module):
    """1D CNN regressor for a single-channel sequence.

    Expects input as (batch, seq_len) and reshapes internally to
    (batch, channels=1, seq_len).
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, ...] = (16, 32, 64),
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        c_in = in_channels
        padding = kernel_size // 2

        for c_out in channels:
            layers.extend(
                [
                    nn.Conv1d(c_in, c_out, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(c_out),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            c_in = c_out

        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            _Flatten(),
            nn.Linear(channels[-1], max(16, channels[-1] // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, channels[-1] // 2), 1),
        )

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(f"Expected (batch, seq_len) input, got shape {tuple(x.shape)}")
        x = x.unsqueeze(1)  # (B, 1, T)
        x = self.backbone(x)
        x = self.pool(x)  # (B, C, 1)
        return self.head(x)


class TimeSeriesRNNRegressor(nn.Module):
    """LSTM/GRU regressor over a single-channel sequence.

    Expects input as (batch, seq_len) and reshapes internally to
    (batch, seq_len, features=1).
    """

    def __init__(
        self,
        rnn_type: str = "lstm",
        hidden_size: int = 64,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        rnn_type = rnn_type.lower()
        rnn_dropout = dropout if num_layers > 1 else 0.0

        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=bidirectional,
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        out_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_size),
            nn.Linear(out_size, max(32, out_size // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, out_size // 2), 1),
        )

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(f"Expected (batch, seq_len) input, got shape {tuple(x.shape)}")
        x = x.unsqueeze(-1)  # (B, T, 1)
        out, h = self.rnn(x)

        # Use last-layer hidden state.
        # LSTM returns (h_n, c_n); GRU returns h_n.
        h_n = h[0] if isinstance(h, tuple) else h

        num_directions = 2 if self.bidirectional else 1
        h_n = h_n.view(self.num_layers, num_directions, x.shape[0], self.hidden_size)
        last_layer = h_n[-1]  # (D, B, H)
        if self.bidirectional:
            last = torch.cat([last_layer[0], last_layer[1]], dim=1)  # (B, 2H)
        else:
            last = last_layer[0]
        return self.head(last)


class CNNLSTMRegressor(nn.Module):
    """CNN frontend + LSTM backend.

    CNN extracts local patterns; LSTM models longer-range dependencies.
    Expects input as (batch, seq_len).
    """

    def __init__(
        self,
        conv_channels: int = 32,
        kernel_size: int = 7,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional

        padding = kernel_size // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.pool = nn.MaxPool1d(kernel_size=2)

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=(dropout if lstm_layers > 1 else 0.0),
            bidirectional=bidirectional,
        )

        out_size = lstm_hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_size),
            nn.Linear(out_size, max(32, out_size // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, out_size // 2), 1),
        )

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(f"Expected (batch, seq_len) input, got shape {tuple(x.shape)}")
        x = x.unsqueeze(1)  # (B, 1, T)
        x = self.cnn(x)  # (B, C, T)
        if x.shape[-1] >= 2:
            x = self.pool(x)  # (B, C, T')
        x = x.transpose(1, 2)  # (B, T', C)
        out, (h_n, _) = self.lstm(x)
        num_directions = 2 if self.bidirectional else 1
        h_n = h_n.view(self.lstm_layers, num_directions, x.shape[0], self.lstm_hidden)
        last_layer = h_n[-1]  # (D, B, H)
        if self.bidirectional:
            last = torch.cat([last_layer[0], last_layer[1]], dim=1)
        else:
            last = last_layer[0]
        return self.head(last)


class _TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class TCNRegressor(nn.Module):
    """Temporal Convolutional Network (dilated residual conv blocks)."""

    def __init__(
        self,
        channels: int = 64,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[_TCNBlock(channels, kernel_size=kernel_size, dilation=d, dropout=dropout) for d in dilations]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            _Flatten(),
            nn.Linear(channels, max(32, channels // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, channels // 2), 1),
        )

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(f"Expected (batch, seq_len) input, got shape {tuple(x.shape)}")
        x = x.unsqueeze(1)  # (B, 1, T)
        x = self.in_proj(x)
        x = self.blocks(x)
        x = self.pool(x)
        return self.head(x)


def get_network_architectures():
    """Return a list of candidate network architectures.

    Each entry is a dict with:
      - 'network': lambda input_size -> nn.Module
      - 'network_name': human-readable string
    """

    # NOTE: input_size here is the *sequence length* of the trace (full 0–2 V in this script).
    # All models below accept flattened input shaped (batch, seq_len) and reshape internally.

    return [
        {
            'network': lambda input_size: TimeSeriesCNN1DRegressor(
                channels=(32, 64, 128), kernel_size=5, dropout=0.15
            ),
            'network_name': 'TS-CNN1D-32-64-128-k5'
        },
        {
            'network': lambda input_size: TimeSeriesCNN1DRegressor(
                channels=(32, 64, 64), kernel_size=9, dropout=0.15
            ),
            'network_name': 'TS-CNN1D-32-64-64-k9'
        },
        {
            'network': lambda input_size: TimeSeriesRNNRegressor(
                rnn_type='lstm', hidden_size=64, num_layers=2, bidirectional=True, dropout=0.20
            ),
            'network_name': 'TS-LSTM-2x64-bi'
        },
        {
            'network': lambda input_size: TimeSeriesRNNRegressor(
                rnn_type='gru', hidden_size=96, num_layers=2, bidirectional=True, dropout=0.20
            ),
            'network_name': 'TS-GRU-2x96-bi'
        },
        {
            'network': lambda input_size: CNNLSTMRegressor(
                conv_channels=32,
                kernel_size=7,
                lstm_hidden=64,
                lstm_layers=1,
                bidirectional=True,
                dropout=0.15,
            ),
            'network_name': 'TS-CNN32-k7+LSTM64-bi'
        },
        {
            'network': lambda input_size: CNNLSTMRegressor(
                conv_channels=64,
                kernel_size=7,
                lstm_hidden=64,
                lstm_layers=2,
                bidirectional=True,
                dropout=0.20,
            ),
            'network_name': 'TS-CNN64-k7+LSTM2x64-bi'
        },
        {
            'network': lambda input_size: TCNRegressor(
                channels=64, kernel_size=3, dilations=(1, 2, 4, 8), dropout=0.15
            ),
            'network_name': 'TS-TCN-64-d1248-k3'
        },
        {
            'network': lambda input_size: TCNRegressor(
                channels=96, kernel_size=5, dilations=(1, 2, 4, 8), dropout=0.20
            ),
            'network_name': 'TS-TCN-96-d1248-k5'
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
    architectures = get_network_architectures()

    # No windowing: always use the full 0–2 V trace.
    voltage_indices = np.arange(n_points, dtype=int)

    total_runs = len(architectures)
    pbar = tqdm(total=total_runs, desc=f"{target_name} Full-trace Search")

    best_score = -np.inf
    best_model_state = None
    best_metadata = None
    # Store the CV test predictions/actuals for the best config
    best_test_actual = None
    best_test_pred = None

    # ------------------------------------------------------------
    # SEARCH LOOP (leave-one-coffee-out for each config)
    # ------------------------------------------------------------

    for arch in architectures:

            all_test_actual = []
            all_test_pred = []

            for test_coffee in coffees:

                test_mask = df_all['Coffee Name'] == test_coffee
                df_train = df_all[~test_mask]
                df_test = df_all[test_mask]

                # Slice current curves using precomputed indices (full trace in this script)
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
                    'window': (0.0, 2.0),
                    'voltage_mode': 'full_0p0_2p0',
                    'architecture': arch['network_name'],
                    'r2': cv_r2
                }
                # Capture the exact CV test points that produced best_score
                best_test_actual = list(all_test_actual)
                best_test_pred = list(all_test_pred)

            pbar.update(1)

    pbar.close()

    # ------------------------------------------------------------
    # SAVE BEST MODEL + METADATA
    # ------------------------------------------------------------

    suffix = "_earlystop" if USE_EARLY_STOPPING else "_fixed"
    save_path = DATADIR / f"BEST_{target_name}_timeseries{suffix}.pkl"

    torch.save({
        'model_state_dict': best_model_state,
        'metadata': best_metadata
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
            f"Full trace: {vmin}-{vmax} V\n"
            f"CV Test R² = {test_r2:.4f} (CV search R² = {best_score:.4f})"
        )

        plt.legend()
        plot_path = PLOTDIR / f"BEST_{target_name}_timeseries{suffix}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"Plot Saved (CV test points): {plot_path}")
        print(f"CV Test R² (from stored predictions): {test_r2:.4f}")
        print("\nSearch complete.\n")
