
# --- repo-root bootstrap (added by reorg) ---
import sys as _sys
from pathlib import Path as _Path
_repo_root = _Path(__file__).resolve().parents[2]
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))
# --- end bootstrap ---

import argparse
import random

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from util import DATADIR, PLOTDIR
from nn_0_synthetic_data_gen import build_model_data
from nn_1_train_model import CoffeeNetBase, evaluate_model, train_coffeenet


# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
# Set to True to use test set for early stopping (optimistic bias).
# Set to False to train for fixed epochs (no leakage/peeking).
USE_EARLY_STOPPING = False


def build_deepbn_64x3_model(input_size: int) -> nn.Module:
    """DeepBN-64x3 architecture.

    Matches the definition in nn_model_window_search.get_network_architectures().
    """

    return nn.Sequential(
        nn.Linear(input_size, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train caffeine model (HPLC_Caff) with DeepBN-64x3 "
            "on fixed 0.4–1.2 V window using leave-one-coffee-out CV."
        )
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Number of training epochs per fold (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional RNG seed for reproducibility. If provided, seeds Python, NumPy, "
            "and PyTorch so results are more stable run-to-run."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "If set (and --seed is set), requests deterministic PyTorch algorithms where possible. "
            "May reduce performance."
        ),
    )
    parser.add_argument(
        "--save-full-model",
        action="store_true",
        help=(
            "If set, trains one final model on ALL samples (using a single scaler) "
            "and saves it to data/."
        ),
    )
    parser.add_argument(
        "--save-preds",
        action="store_true",
        help=(
            "If set, saves per-sample actual/predicted values for BOTH train and test "
            "points across all LOCO folds (useful for plotting from the pickle alone)."
        ),
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        if args.deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                # Older PyTorch builds or some ops may not support strict determinism.
                pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    target_name = "HPLC_Caff"
    network_name = "DeepBN-64x3"
    v_start, v_end = 0.4, 1.2

    # ------------------------------------------------------------
    # Load full-voltage data (no normalization / bins)
    # ------------------------------------------------------------

    df_all = build_model_data(
        NORMALIZE=False,
        REDOX=False,
        USE_BINS=False,
        test_train_split=False,
    )

    # Assume underlying CV spans 0 to 2 V with uniform spacing
    n_points = len(df_all["cv_raw"].iloc[0])
    full_voltage_array = np.linspace(0.0, 2.0, n_points)

    voltage_mask = (full_voltage_array >= v_start) & (full_voltage_array <= v_end)
    voltage_indices = np.where(voltage_mask)[0]

    if len(voltage_indices) == 0:
        raise ValueError("Selected voltage window contains no points in CV trace.")

    print(
        f"Training {target_name} model with {network_name} on "
        f"window {v_start:.2f}-{v_end:.2f} V "
        f"({len(voltage_indices)} voltage points)."
    )

    coffees = df_all["Coffee Name"].unique()
    print(f"Number of coffees (LOCO folds): {len(coffees)}")

    all_train_actual = []
    all_train_pred = []
    all_test_actual = []
    all_test_pred = []
    exp_rows = []
    pred_rows = []

    for fold, test_coffee in enumerate(tqdm(coffees, desc="Leave-one-coffee-out folds")):
        test_mask = df_all["Coffee Name"] == test_coffee
        df_train = df_all[~test_mask]
        df_test = df_all[test_mask]

        if df_test.empty:
            continue

        row = {
            "fold": fold,
            "test_coffee": test_coffee,
            "target": target_name,
            "network_name": network_name,
            "v_start": v_start,
            "v_end": v_end,
            "v_window_size": v_end - v_start,
            "n_voltage_points": int(len(voltage_indices)),
            "num_epochs": int(args.epochs),
            "use_early_stopping": bool(USE_EARLY_STOPPING),
        }

        # Slice current curves to this voltage window using precomputed indices
        X_train = np.array([np.asarray(cv)[voltage_indices] for cv in df_train["cv_raw"]])
        X_test = np.array([np.asarray(cv)[voltage_indices] for cv in df_test["cv_raw"]])

        if X_train.shape[1] == 0:
            continue

        y_train = df_train[target_name].values.reshape(-1, 1)
        y_test = df_test[target_name].values.reshape(-1, 1)

        input_size = X_train.shape[1]

        # Standardize per fold using train data only
        X_scaler = StandardScaler().fit(X_train)
        y_scaler = StandardScaler().fit(y_train)

        X_train_s = X_scaler.transform(X_train)
        y_train_s = y_scaler.transform(y_train)

        X_test_s = X_scaler.transform(X_test)
        y_test_s = y_scaler.transform(y_test)

        # Build model
        model = CoffeeNetBase()
        model.network = build_deepbn_64x3_model(input_size)
        model.to(device)

        # Train
        val_X = X_test_s if USE_EARLY_STOPPING else None
        val_y = y_test_s if USE_EARLY_STOPPING else None

        model = train_coffeenet(
            model,
            X_train_s,
            y_train_s,
            val_X,
            val_y,
            num_epochs=args.epochs,
        )

        # Evaluate on train
        train_pred_s = evaluate_model(model, X_train_s, y_train_s)
        train_pred = y_scaler.inverse_transform(train_pred_s)

        row["train_r2"] = r2_score(y_train, train_pred)
        row["train_mae"] = mean_absolute_error(y_train, train_pred)

        all_train_actual.extend(y_train.flatten())
        all_train_pred.extend(train_pred.flatten())

        if args.save_preds:
            train_actual_flat = y_train.flatten().tolist()
            train_pred_flat = train_pred.flatten().tolist()
            for sample_name, coffee_name, a, p in zip(
                df_train["Sample Name"].tolist(),
                df_train["Coffee Name"].tolist(),
                train_actual_flat,
                train_pred_flat,
            ):
                pred_rows.append(
                    {
                        "fold": fold,
                        "split": "train",
                        "test_coffee": test_coffee,
                        "sample_name": sample_name,
                        "coffee_name": coffee_name,
                        "target": target_name,
                        "network_name": network_name,
                        "v_start": v_start,
                        "v_end": v_end,
                        "actual": float(a),
                        "pred": float(p),
                    }
                )

        # Evaluate on held-out coffee (inverse-transform back to original scale)
        test_pred_s = evaluate_model(model, X_test_s, y_test_s)
        test_pred = y_scaler.inverse_transform(test_pred_s)

        row["test_r2"] = r2_score(y_test, test_pred)
        row["test_mae"] = mean_absolute_error(y_test, test_pred)

        all_test_actual.extend(y_test.flatten())
        all_test_pred.extend(test_pred.flatten())

        if args.save_preds:
            test_actual_flat = y_test.flatten().tolist()
            test_pred_flat = test_pred.flatten().tolist()
            for sample_name, coffee_name, a, p in zip(
                df_test["Sample Name"].tolist(),
                df_test["Coffee Name"].tolist(),
                test_actual_flat,
                test_pred_flat,
            ):
                pred_rows.append(
                    {
                        "fold": fold,
                        "split": "test",
                        "test_coffee": test_coffee,
                        "sample_name": sample_name,
                        "coffee_name": coffee_name,
                        "target": target_name,
                        "network_name": network_name,
                        "v_start": v_start,
                        "v_end": v_end,
                        "actual": float(a),
                        "pred": float(p),
                    }
                )

        exp_rows.append(row)

    if not exp_rows:
        print("No folds were run; exp_rows is empty.")
        return

    df_results = pd.DataFrame(exp_rows)

    # ------------------------------------------------------------
    # Aggregate CV metrics
    # ------------------------------------------------------------

    overall_r2 = r2_score(all_test_actual, all_test_pred)
    overall_mae = mean_absolute_error(all_test_actual, all_test_pred)

    train_overall_r2 = r2_score(all_train_actual, all_train_pred)
    train_overall_mae = mean_absolute_error(all_train_actual, all_train_pred)

    print("\nPer-fold test performance (sorted by test R^2):")
    print(
        df_results.sort_values("test_r2", ascending=False)[
            ["fold", "test_coffee", "test_r2", "test_mae"]
        ].to_string(index=False)
    )

    print("\nOverall leave-one-coffee-out performance (test):")
    print(f"  Test R^2  = {overall_r2:.4f}")
    print(f"  Test MAE  = {overall_mae:.4f}")

    print("\nOverall training performance (pooled over folds):")
    print(f"  Train R^2 = {train_overall_r2:.4f}")
    print(f"  Train MAE = {train_overall_mae:.4f}")

    # Save detailed results
    base_name = f"caff_{network_name}_V{v_start:.2f}-{v_end:.2f}"
    base_name += "_earlystop" if USE_EARLY_STOPPING else "_fixed"

    pkl_path = DATADIR / f"{base_name}_loco_results.pkl"
    csv_path = DATADIR / f"{base_name}_loco_results.csv"

    df_results.to_pickle(pkl_path)
    df_results.to_csv(csv_path, index=False)

    print(f"\nDetailed LOCO results saved to: {pkl_path}")
    print(f"CSV version saved to: {csv_path}")

    if args.save_preds and pred_rows:
        df_preds = pd.DataFrame(pred_rows)
        preds_pkl_path = DATADIR / f"{base_name}_preds.pkl"
        preds_csv_path = DATADIR / f"{base_name}_preds.csv"
        df_preds.to_pickle(preds_pkl_path)
        df_preds.to_csv(preds_csv_path, index=False)
        print(f"Per-sample predictions saved to: {preds_pkl_path}")
        print(f"CSV version saved to: {preds_csv_path}")

    # --------------------------------------------------------
    # Plot: train vs test predictions (actual on x, predicted on y)
    # --------------------------------------------------------

    plt.figure(figsize=(7, 7))

    combined_actual = all_train_actual + all_test_actual
    combined_pred = all_train_pred + all_test_pred
    max_val = max(max(combined_actual), max(combined_pred))
    limit = max_val * 1.05

    plt.plot([0, limit], [0, limit], linestyle="--", color="black", linewidth=1, label="Ideal (y=x)")

    plt.scatter(
        all_train_actual,
        all_train_pred,
        color="#1f77b4",
        alpha=1.0,
        s=40,
        edgecolor="white",
        linewidth=0.5,
        label="Train",
        zorder=2,
    )

    plt.scatter(
        all_test_actual,
        all_test_pred,
        color="#ff7f0e",
        alpha=1.0,
        s=40,
        edgecolor="white",
        linewidth=0.5,
        label="Test",
        zorder=3,
    )

    plt.xlim(0, limit)
    plt.ylim(0, limit)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.xlabel("Actual HPLC_Caff", fontsize=11, fontweight="medium")
    plt.ylabel("Predicted HPLC_Caff", fontsize=11, fontweight="medium")
    plt.title(
        f"HPLC_Caff prediction\n"
        f"{network_name} | Window {v_start:.2f}-{v_end:.2f} V\n"
        f"Train R²={train_overall_r2:.3f} | Test R²={overall_r2:.3f}",
        fontsize=12,
    )

    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, linestyle="-", alpha=0.15, color="gray")

    plot_path = PLOTDIR / f"{base_name}_train_test.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Train/Test scatter plot saved to: {plot_path}")

    # --------------------------------------------------------
    # Optional: Train a single final model on ALL data and save
    # --------------------------------------------------------

    if args.save_full_model:
        X_all = np.array([np.asarray(cv)[voltage_indices] for cv in df_all["cv_raw"]])
        y_all = df_all[target_name].values.reshape(-1, 1)

        X_scaler = StandardScaler().fit(X_all)
        y_scaler = StandardScaler().fit(y_all)

        X_all_s = X_scaler.transform(X_all)
        y_all_s = y_scaler.transform(y_all)

        model = CoffeeNetBase()
        model.network = build_deepbn_64x3_model(X_all_s.shape[1])
        model.to(device)

        model = train_coffeenet(model, X_all_s, y_all_s, None, None, num_epochs=args.epochs)

        full_model_path = DATADIR / f"{base_name}_full_model.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "network_name": network_name,
                "target": target_name,
                "v_start": float(v_start),
                "v_end": float(v_end),
                "input_size": int(X_all_s.shape[1]),
                "X_mean": X_scaler.mean_.tolist(),
                "X_std": X_scaler.scale_.tolist(),
                "y_mean": y_scaler.mean_.tolist(),
                "y_std": y_scaler.scale_.tolist(),
                "num_epochs": int(args.epochs),
            },
            full_model_path,
        )

        print(f"Full-data trained model saved to: {full_model_path}")


if __name__ == "__main__":
    main()
