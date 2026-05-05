import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from util import DATADIR, PLOTDIR
from nn_0_synthetic_data_gen import build_model_data
from nn_1_train_model import CoffeeNetBase, train_coffeenet

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
USE_EARLY_STOPPING = False

# Options: 'HPLC_Caff', 'HPLC_CGA', 'TDS'
TARGET_NAME = 'TDS'

def build_visual_model(input_size: int) -> nn.Module:
    """
    1024 -> 256 -> 128 architecture for weight visualization.
    """
    return nn.Sequential(
        nn.Linear(input_size, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(1024, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(128, 1)
    )

def main():
    parser = argparse.ArgumentParser(
        description="Train 1024->256->128 model and visualize input weights."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200, # Start with reasonable epochs, user didn't specify
        help="Number of training epochs per fold"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    target_name = TARGET_NAME
    
    # Use full voltage window 0.0 - 2.0 V
    v_start, v_end = 0.0, 2.0

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------
    df_all = build_model_data(
        NORMALIZE=False,
        REDOX=False,
        USE_BINS=False,
        test_train_split=False,
    )

    n_points_total = len(df_all["cv_raw"].iloc[0])
    full_voltage_array = np.linspace(0.0, 2.0, n_points_total)
    
    # Mask if we wanted to restrict, but here we use full range or close to it.
    voltage_mask = (full_voltage_array >= v_start) & (full_voltage_array <= v_end)
    voltage_indices = np.where(voltage_mask)[0]
    voltage_axis = full_voltage_array[voltage_indices]
    
    input_size = len(voltage_indices)
    print(f"Input size: {input_size} points.")

    coffees = df_all["Coffee Name"].unique()
    print(f"Number of coffees (LOCO folds): {len(coffees)}")

    # Storage for weights
    # We will accumulate the SUM of weights for each input node across all models
    # Shape: [input_size]
    total_input_weights = np.zeros(input_size)
    total_folds_run = 0

    for fold, test_coffee in enumerate(tqdm(coffees, desc="LOCO folds")):
        test_mask = df_all["Coffee Name"] == test_coffee
        df_train = df_all[~test_mask]
        df_test = df_all[test_mask]

        if df_test.empty:
            continue

        # Prepare X data
        X_train = np.array([
            np.asarray(cv)[voltage_indices] for cv in df_train["cv_raw"]
        ])
        X_test = np.array([
            np.asarray(cv)[voltage_indices] for cv in df_test["cv_raw"]
        ])
        
        y_train = df_train[target_name].values.reshape(-1, 1)
        y_test = df_test[target_name].values.reshape(-1, 1)

        # Standardize train data (fit on train)
        X_scaler = StandardScaler().fit(X_train)
        y_scaler = StandardScaler().fit(y_train)

        X_train_s = X_scaler.transform(X_train)
        y_train_s = y_scaler.transform(y_train)
        
        # Build model
        model = CoffeeNetBase()
        model.network = build_visual_model(input_size)
        model.to(device)

        # Train
        model = train_coffeenet(
            model,
            X_train_s,
            y_train_s,
            X_test=None,
            y_test=None, 
            num_epochs=args.epochs,
        )

        # Extract weights from first linear layer
        # model.network[0] is nn.Linear(input_size, 1024)
        # Weights shape: [1024, input_size]
        first_layer = model.network[0]
        weights = first_layer.weight.detach().cpu().numpy()
        
        # Sum weights for every input node (sum across neurons)
        # Result shape: [input_size]
        summed_weights = np.sum(weights, axis=0)
        
        total_input_weights += summed_weights
        total_folds_run += 1

    # ------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------
    if total_folds_run > 0:
        avg_weights = total_input_weights / total_folds_run
        
        plt.figure(figsize=(12, 6))
        plt.bar(voltage_axis, total_input_weights, width=(2.0/input_size), align='center')
        plt.title('Summed Input Layer Weights (Across All LOCO Folds)')
        plt.xlabel('Voltage (V)')
        plt.ylabel('Summed Weight Value')
        plt.grid(True, alpha=0.3)
        
        plot_path = PLOTDIR / 'input_weights_summed.png'
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        
        # Also save data
        weight_df = pd.DataFrame({
            'Voltage': voltage_axis,
            'SummedWeight': total_input_weights
        })
        csv_path = PLOTDIR / 'input_weights_summed.csv'
        weight_df.to_csv(csv_path, index=False)
        print(f"Weights data saved to {csv_path}")

if __name__ == "__main__":
    main()
