
# --- repo-root bootstrap (added by reorg) ---
import sys as _sys
from pathlib import Path as _Path
_repo_root = _Path(__file__).resolve().parents[2]
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))
# --- end bootstrap ---

from util import DATADIR, PLOTDIR
from nn_0_synthetic_data_gen import build_model_data
from nn_1_train_model import CoffeeNetBase, train_coffeenet, evaluate_model

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
# Set to True to use the held-out coffee as a validation set for early stopping.
# Set to False to train for fixed epochs (no leakage).
USE_EARLY_STOPPING = False

# Fixed voltage window for all models (as requested)
WINDOW_VMIN = 0.1
WINDOW_VMAX = 1.8

NUM_EPOCHS = 5000


# ------------------------------------------------------------
# Sequence Model Building Blocks
# ------------------------------------------------------------


class SequenceRegressor(nn.Module):
    def __init__(
        self,
        cell_type: str,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        cell_type = cell_type.upper().strip()
        if cell_type not in {"RNN", "GRU", "LSTM"}:
            raise ValueError("cell_type must be one of: RNN, GRU, LSTM")

        # Each voltage point is treated as a timestep with a single feature (current).
        input_size = 1

        rnn_dropout = dropout if num_layers > 1 else 0.0

        if cell_type == "RNN":
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        elif cell_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )

        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last)


def get_sequence_architectures():
    """Candidate sequence models (RNN/GRU/LSTM only)."""

    def make(cell_type: str, hidden: int, layers: int, dropout: float = 0.0):
        name = f"{cell_type}-{hidden}-L{layers}"
        if dropout and layers > 1:
            name += f"-Drop{dropout:g}"
        return {
            "network": lambda output_size: SequenceRegressor(
                cell_type=cell_type,
                hidden_size=hidden,
                num_layers=layers,
                output_size=output_size,
                dropout=dropout,
            ),
            "network_name": name,
        }

    return [
        make("RNN", 16, 1),
        make("RNN", 32, 1),
        make("RNN", 64, 1),
        make("GRU", 16, 1),
        make("GRU", 32, 1),
        make("GRU", 64, 1),
        make("LSTM", 16, 1),
        make("LSTM", 32, 1),
        make("LSTM", 64, 1),
        make("GRU", 32, 2, dropout=0.1),
        make("LSTM", 32, 2, dropout=0.1),
    ]


# ------------------------------------------------------------
# MAIN: per-target search on fixed window
# ------------------------------------------------------------


if __name__ == "__main__":

    FLAVOR_COLUMNS_REQUESTED = [
        "Spice",
        "Body",
        "Floral",
        "Honey",
        "Sugars",
        "Caramel",
        "Fruits",
        "Citrus",
        "Berry",
        "Cocoa",
        "Nuts",
        "Rustic",
    ]

    df_all = build_model_data(
        NORMALIZE=False,
        REDOX=False,
        USE_BINS=False,
        test_train_split=False,
        require_columns=FLAVOR_COLUMNS_REQUESTED,
    )

    FLAVOR_COLUMNS = [c for c in FLAVOR_COLUMNS_REQUESTED if c in df_all.columns]
    missing_cols = [c for c in FLAVOR_COLUMNS_REQUESTED if c not in df_all.columns]
    if missing_cols:
        print(
            "Warning: missing some requested flavor columns in dataset: "
            + ", ".join(missing_cols)
        )
    if len(FLAVOR_COLUMNS) == 0:
        raise KeyError(
            "None of the requested flavor columns were found in the dataset. "
            "Available columns include: " + ", ".join(map(str, df_all.columns))
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: CUDA not available, using CPU. This may be slow.")

    # Voltage index slice for the fixed window
    n_points = len(df_all["cv_raw"].iloc[0])
    full_voltage_array = np.linspace(0.0, 2.0, n_points)
    voltage_mask = (full_voltage_array >= WINDOW_VMIN) & (full_voltage_array <= WINDOW_VMAX)
    voltage_indices = np.where(voltage_mask)[0]
    if len(voltage_indices) == 0:
        raise ValueError("Fixed voltage window contained no points in cv_raw discretization")

    print(
        f"\nSequence-model flavor search on fixed window {WINDOW_VMIN}-{WINDOW_VMAX} V "
        f"for {len(FLAVOR_COLUMNS)} targets on {device.type}\n"
    )

    architectures = get_sequence_architectures()

    # Outer loop: each target gets its own best model + plot
    for target_col in FLAVOR_COLUMNS:

        print(f"\n=== Target: {target_col} ===")

        # Drop samples missing this target
        n_before = len(df_all)
        df_t = df_all.dropna(subset=[target_col]).copy()
        n_after = len(df_t)
        if n_after < n_before:
            print(f"Dropped {n_before - n_after} samples missing '{target_col}'")
        if n_after == 0:
            print(f"No samples remain for target '{target_col}', skipping")
            continue

        coffees = df_t["Coffee Name"].unique()

        best_score = -np.inf
        best_metadata = None
        best_test_actual = None
        best_test_pred = None

        pbar = tqdm(total=len(architectures), desc=f"{target_col} (seq arch search)")

        for arch in architectures:

            all_test_actual = []
            all_test_pred = []

            for test_coffee in coffees:

                test_mask = df_t["Coffee Name"] == test_coffee
                df_train = df_t[~test_mask]
                df_test = df_t[test_mask]

                X_train = np.array([
                    np.asarray(cv)[voltage_indices]
                    for cv in df_train["cv_raw"]
                ])
                X_test = np.array([
                    np.asarray(cv)[voltage_indices]
                    for cv in df_test["cv_raw"]
                ])

                if X_train.size == 0 or X_test.size == 0 or X_train.shape[1] == 0:
                    continue

                # Single-target regression
                y_train = df_train[[target_col]].values
                y_test = df_test[[target_col]].values

                # Scale X per-timestep (features = timesteps), then reshape to sequence
                X_scaler = StandardScaler().fit(X_train)
                y_scaler = StandardScaler().fit(y_train)

                X_train_s = X_scaler.transform(X_train)
                X_test_s = X_scaler.transform(X_test)

                y_train_s = y_scaler.transform(y_train)
                y_test_s = y_scaler.transform(y_test)

                X_train_seq = X_train_s[:, :, None]
                X_test_seq = X_test_s[:, :, None]

                # Build model
                model = CoffeeNetBase()
                model.network = arch["network"](output_size=1)
                model.to(device)

                # Train
                val_X = X_test_seq if USE_EARLY_STOPPING else None
                val_y = y_test_s if USE_EARLY_STOPPING else None

                model = train_coffeenet(
                    model,
                    X_train_seq,
                    y_train_s,
                    val_X,
                    val_y,
                    num_epochs=NUM_EPOCHS,
                )

                # Evaluate (inverse-transform back to original scale)
                test_pred = evaluate_model(model, X_test_seq, y_test_s)
                if isinstance(test_pred, torch.Tensor):
                    test_pred = test_pred.detach().cpu().numpy()

                test_pred_original = y_scaler.inverse_transform(test_pred)

                all_test_actual.append(y_test)
                all_test_pred.append(test_pred_original)

            if len(all_test_actual) == 0:
                pbar.update(1)
                continue

            actual_vec = np.vstack(all_test_actual).reshape(-1)
            pred_vec = np.vstack(all_test_pred).reshape(-1)

            try:
                cv_r2 = float(r2_score(actual_vec, pred_vec))
            except ValueError:
                cv_r2 = float("nan")

            if np.isfinite(cv_r2) and cv_r2 > best_score:
                best_score = cv_r2
                best_metadata = {
                    "target": target_col,
                    "window": (WINDOW_VMIN, WINDOW_VMAX),
                    "architecture": arch["network_name"],
                    "cv_r2": cv_r2,
                    "n_samples": int(len(df_t)),
                    "n_coffees": int(len(coffees)),
                }
                best_test_actual = actual_vec
                best_test_pred = pred_vec

            pbar.update(1)

        pbar.close()

        if best_metadata is None:
            print(f"No valid CV results for target '{target_col}', skipping save")
            continue

        print(f"Best for {target_col}: {best_metadata['architecture']} (CV R²={best_metadata['cv_r2']:.4f})")

        # Retrain final model on all data for this target using best architecture
        best_arch_ctor = None
        for arch in architectures:
            if arch["network_name"] == best_metadata["architecture"]:
                best_arch_ctor = arch["network"]
                break

        final_artifacts = {}
        if best_arch_ctor is not None:
            X_all = np.array([
                np.asarray(cv)[voltage_indices]
                for cv in df_t["cv_raw"]
            ])
            y_all = df_t[[target_col]].values

            X_scaler_all = StandardScaler().fit(X_all)
            y_scaler_all = StandardScaler().fit(y_all)

            X_all_s = X_scaler_all.transform(X_all)
            y_all_s = y_scaler_all.transform(y_all)

            X_all_seq = X_all_s[:, :, None]

            final_model = CoffeeNetBase()
            final_model.network = best_arch_ctor(output_size=1)
            final_model.to(device)
            final_model = train_coffeenet(
                final_model,
                X_all_seq,
                y_all_s,
                None,
                None,
                num_epochs=NUM_EPOCHS,
            )

            final_artifacts = {
                "final_model_state_dict": final_model.state_dict(),
                "x_scaler": X_scaler_all,
                "y_scaler": y_scaler_all,
                "voltage_window_indices": voltage_indices,
                "voltage_window": (WINDOW_VMIN, WINDOW_VMAX),
            }

        suffix = "_earlystop" if USE_EARLY_STOPPING else "_fixed"
        safe_target = target_col.replace(" ", "_")

        save_path = DATADIR / f"BEST_FlavorSeq_{safe_target}{suffix}.pkl"
        torch.save(
            {
                "metadata": best_metadata,
                "cv_test_actual": best_test_actual,
                "cv_test_pred": best_test_pred,
                **final_artifacts,
            },
            save_path,
        )
        print(f"Saved model/artifacts: {save_path}")

        # Plot CV predictions for the best architecture
        y_true = np.asarray(best_test_actual)
        y_hat = np.asarray(best_test_pred)

        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_hat, alpha=0.7, label="CV test (held-out coffees)")

        lim_min = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_hat)]))
        lim_max = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_hat)]))
        plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", color="black")

        plt.xlabel(f"Actual {target_col}")
        plt.ylabel(f"Predicted {target_col}")
        plt.title(
            f"{target_col}\n"
            f"{best_metadata['architecture']}\n"
            f"Window: {WINDOW_VMIN}-{WINDOW_VMAX} V\n"
            f"CV Test R² = {best_metadata['cv_r2']:.4f}"
        )
        plt.legend()

        plot_path = PLOTDIR / f"BEST_FlavorSeq_{safe_target}{suffix}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved plot: {plot_path}")

    print("\nAll targets complete.\n")
