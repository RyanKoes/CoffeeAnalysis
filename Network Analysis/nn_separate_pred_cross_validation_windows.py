from util import setup_mplt, DATADIR, PLOTDIR
from nn_0_synthetic_data_gen import build_model_data, generate_combined_data
from nn_1_train_model import CoffeeNetBase, train_coffeenet, evaluate_model

from collections import defaultdict
import tabulate
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools

import torch
import torch.nn as nn


def generate_voltage_windows(min_v=0.0, max_v=2.0, num_windows=100, min_window_size=0.2, max_window_size=1.0):
    """
    Generate a diverse set of voltage windows across the full range.

    Parameters:
    - min_v: Minimum voltage in the full range
    - max_v: Maximum voltage in the full range
    - num_windows: Number of different windows to generate
    - min_window_size: Minimum size of a voltage window
    - max_window_size: Maximum size of a voltage window

    Returns:
    - List of tuples (v_start, v_end)
    """
    windows = []

    # Strategy 1: Systematic scan with different window sizes (50 windows)
    window_sizes = np.linspace(min_window_size, max_window_size, 5)
    for window_size in window_sizes:
        n_positions = 10
        for i in range(n_positions):
            v_start = min_v + i * (max_v - min_v - window_size) / (n_positions - 1)
            v_end = v_start + window_size
            windows.append((round(v_start, 2), round(v_end, 2)))

    # Strategy 2: Random windows (30 windows)
    np.random.seed(42)
    for _ in range(30):
        window_size = np.random.uniform(min_window_size, max_window_size)
        v_start = np.random.uniform(min_v, max_v - window_size)
        v_end = v_start + window_size
        windows.append((round(v_start, 2), round(v_end, 2)))

    # Strategy 3: Key regions of interest (20 windows)
    # These are common regions where redox peaks often occur
    key_regions = [
        (0.0, 0.3), (0.1, 0.4), (0.2, 0.5), (0.3, 0.6),
        (0.4, 0.7), (0.5, 0.8), (0.6, 0.9), (0.7, 1.0),
        (0.8, 1.1), (0.9, 1.2), (1.0, 1.3), (1.1, 1.4),
        (1.2, 1.5), (1.3, 1.6), (1.4, 1.7), (1.5, 1.8),
        (1.6, 1.9), (1.7, 2.0), (0.0, 0.5), (1.5, 2.0)
    ]
    windows.extend(key_regions)

    # Remove duplicates and ensure we have exactly num_windows
    windows = list(set(windows))

    # If we have more than needed, randomly sample
    if len(windows) > num_windows:
        np.random.seed(42)
        windows = list(np.random.choice(len(windows), num_windows, replace=False))
        windows = [windows[i] for i in range(num_windows)]

    # If we need more, add some more random ones
    while len(windows) < num_windows:
        window_size = np.random.uniform(min_window_size, max_window_size)
        v_start = np.random.uniform(min_v, max_v - window_size)
        v_end = v_start + window_size
        new_window = (round(v_start, 2), round(v_end, 2))
        if new_window not in windows:
            windows.append(new_window)

    return sorted(windows[:num_windows])


if __name__ == "__main__":
    setup_mplt()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        print("Warning: No GPU found, using CPU for training. Abort.")
        exit()
    print(f"Using device: {device}")

    # Generate voltage windows to test
    NUM_VOLTAGE_WINDOWS = 100
    voltage_windows = generate_voltage_windows(min_v=0.0, max_v=2.0, num_windows=NUM_VOLTAGE_WINDOWS)

    print(f"\nGenerated {len(voltage_windows)} voltage windows to test")
    print(f"Sample windows: {voltage_windows[:5]}")

    # Define base network configurations for each target variable
    target_configs = {
        'HPLC_Caff': {
            'NORMALIZE': False,
            'REDOX': False,
            'USE_BINS': False,
            'num_epochs': 2000,
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 256),
                nn.BatchNorm1d(256),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.BatchNorm1d(64),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(64, 1)
            ),
            'network_name': 'Caff-256-64-1'
        },
        'HPLC_CGA': {
            'NORMALIZE': False,
            'REDOX': False,
            'USE_BINS': False,
            'num_epochs': 2000,
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 256),
                nn.BatchNorm1d(256),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.BatchNorm1d(64),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(64, 1)
            ),
            'network_name': 'CGA-256-64-1'
        },
        'TDS': {
            'NORMALIZE': False,
            'REDOX': False,
            'USE_BINS': False,
            'num_epochs': 2000,
            'network': lambda input_size: nn.Sequential(
                nn.Linear(input_size, 256),
                nn.BatchNorm1d(256),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.BatchNorm1d(64),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(64, 1)
            ),
            'network_name': 'TDS-256-64-1'
        }
    }

    exp_results = []

    # Loop through each target variable
    for target_name, base_config in target_configs.items():
        print(f"\n{'=' * 80}")
        print(f"Training models for {target_name} with {len(voltage_windows)} different voltage windows")
        print(f"{'=' * 80}")

        # Load the full voltage data once
        experiment_name_base = f'SingleTarget-{target_name}-OX-{base_config["network_name"]}-{base_config["num_epochs"]}'
        all_data_path = DATADIR / f'{experiment_name_base}_all.pkl'

        if all_data_path.exists():
            print(f"Loading base data for {target_name}...")
            df_all_full = pd.read_pickle(all_data_path)
        else:
            print(f"Building base data for {target_name}...")
            df_all_full = build_model_data(test_train_split=False, **base_config)
            df_all_full.to_pickle(all_data_path)

        # Assume voltage array from 0 to 2V with uniform spacing
        # Adjust this if your actual voltage array is different
        n_points = len(df_all_full['cv_raw'].iloc[0])
        full_voltage_array = np.linspace(0, 2, n_points)

        coffees = df_all_full['Coffee Name'].unique()

        # Loop through each voltage window
        for v_window in tqdm(voltage_windows, desc=f"{target_name} voltage windows"):
            v_start, v_end = v_window

            # Create voltage mask
            voltage_mask = (full_voltage_array >= v_start) & (full_voltage_array <= v_end)
            voltage_indices = np.where(voltage_mask)[0]

            # Skip if window is too small (less than 10 points)
            if len(voltage_indices) < 10:
                print(f"Skipping window {v_window} - too few points ({len(voltage_indices)})")
                continue

            # Apply voltage windowing to create a new dataframe
            df_all = df_all_full.copy()
            df_all['cv_raw'] = df_all['cv_raw'].apply(lambda x: np.array(x)[voltage_indices])

            experiment_name = f'{experiment_name_base}-V{v_start}-{v_end}'

            # Leave-one-coffee-out cross-validation
            for fold, test_coffee in enumerate(coffees):
                test_mask = df_all['Coffee Name'] == test_coffee
                df_train = df_all[~test_mask]
                df_test = df_all[test_mask]

                e = {
                    'fold': fold,
                    'test_coffee': test_coffee,
                    'experiment_name': experiment_name,
                    'target': target_name,
                    'v_start': v_start,
                    'v_end': v_end,
                    'v_window_size': v_end - v_start,
                    'n_voltage_points': len(voltage_indices)
                }

                # Setup train data
                X_train = np.array(df_train['cv_raw'].to_list())
                y_train = np.array(df_train[target_name].values).reshape(-1, 1)

                input_size = X_train.shape[1]

                X_scaler = StandardScaler().fit(X_train)
                y_scaler = StandardScaler().fit(y_train)

                X_train_standard = X_scaler.transform(X_train)
                y_train_standard = y_scaler.transform(y_train)

                # Setup test data
                X_test = np.array(df_test['cv_raw'].to_list())
                y_test = np.array(df_test[target_name].values).reshape(-1, 1)

                X_test_standard = X_scaler.transform(X_test)
                y_test_standard = y_scaler.transform(y_test)

                model = CoffeeNetBase()
                model.network = base_config['network'](input_size)

                # Train the model
                model_path = DATADIR / f'{experiment_name}-fold-{fold}.pth'
                e['model_path'] = model_path

                if model_path.exists():
                    checkpoint = torch.load(model_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(device)
                else:
                    model.to(device)
                    model = train_coffeenet(model,
                                            X_train_standard, y_train_standard,
                                            X_test_standard, y_test_standard,
                                            num_epochs=base_config['num_epochs'])

                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'X_mean': X_scaler.mean_.tolist(),
                        'X_std': X_scaler.scale_.tolist(),
                        'y_mean': y_scaler.mean_.tolist(),
                        'y_std': y_scaler.scale_.tolist(),
                        'input_size': input_size,
                        'v_start': v_start,
                        'v_end': v_end
                    }, model_path)

                # Evaluate on training data
                train_predictions = evaluate_model(model, X_train_standard, y_train_standard)
                train_predictions_original = y_scaler.inverse_transform(train_predictions)

                e[f'train_{target_name}_r2'] = r2_score(y_train, train_predictions_original)
                e[f'train_{target_name}_mae'] = mean_absolute_error(y_train, train_predictions_original)

                # Evaluate on test data
                test_predictions = evaluate_model(model, X_test_standard, y_test_standard)
                test_predictions_original = y_scaler.inverse_transform(test_predictions)

                e[f'test_{target_name}_r2'] = r2_score(y_test, test_predictions_original)
                e[f'test_{target_name}_mae'] = mean_absolute_error(y_test, test_predictions_original)

                exp_results.append(e)

    ### PRINT RESULTS ###

    df_results = pd.DataFrame(exp_results)

    # Print results for each target
    for target_name in target_configs.keys():
        print(f"\n{'=' * 80}")
        print(f"Results for {target_name}")
        print(f"{'=' * 80}")

        df_target = df_results[df_results['target'] == target_name]

        # Group by voltage window and compute average performance
        df_avg = df_target.groupby(['v_start', 'v_end', 'v_window_size']).agg({
            f'train_{target_name}_r2': 'mean',
            f'test_{target_name}_r2': 'mean',
            f'train_{target_name}_mae': 'mean',
            f'test_{target_name}_mae': 'mean'
        }).reset_index()

        df_avg.sort_values(by=f'test_{target_name}_r2', ascending=False, inplace=True)

        print(f"\nTop 20 voltage windows for {target_name} (sorted by test R²):")
        print(tabulate.tabulate(df_avg.head(20), floatfmt=".4f", headers='keys', tablefmt='psql', showindex=False))

        # Print best voltage window details
        best_window = df_avg.iloc[0]
        print(f"\nBest voltage window: {best_window['v_start']:.2f}V - {best_window['v_end']:.2f}V")
        print(f"Test R²: {best_window[f'test_{target_name}_r2']:.4f}")
        print(f"Test MAE: {best_window[f'test_{target_name}_mae']:.4f}")

    # Save results
    results_path = DATADIR / 'voltage_window_search_results.pkl'
    df_results.to_pickle(results_path)
    print(f"\nAll results saved to {results_path}")

    # Save summary CSV for easy viewing
    summary_path = DATADIR / 'voltage_window_search_summary.csv'
    df_summary = df_results.groupby(['target', 'v_start', 'v_end', 'v_window_size']).agg({
        f'train_HPLC_Caff_r2': 'mean',
        f'test_HPLC_Caff_r2': 'mean',
        f'train_HPLC_CGA_r2': 'mean',
        f'test_HPLC_CGA_r2': 'mean',
        f'train_TDS_r2': 'mean',
        f'test_TDS_r2': 'mean',
    }).reset_index()
    df_summary.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")