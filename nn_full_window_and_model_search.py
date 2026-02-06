from util import DATADIR, PLOTDIR
from nn_0_synthetic_data_gen import build_model_data, generate_combined_data
from nn_1_train_model import CoffeeNetBase, train_coffeenet, evaluate_model

from collections import defaultdict
import tabulate
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from tqdm import tqdm

import torch
import torch.nn as nn

import os
import argparse

cache_file = DATADIR / 'raw_data_cache.pkl'
if cache_file.exists():
    os.remove(cache_file)

# Now call normally; will reload from sheet and re-cache
from util import read_coffehub
df = read_coffehub()

# Generate all possible voltage windows with 0.1V increments
def generate_voltage_windows(min_v=0.0, max_v=2.0, step=0.1):
    """Generate all possible voltage windows from min_v to max_v with given step."""
    voltage_points = np.arange(min_v, max_v + step/2, step)  # Include max_v
    voltage_points = np.round(voltage_points, 1)  # Round to avoid floating point errors
    
    windows = []
    for i in range(len(voltage_points)):
        for j in range(i + 1, len(voltage_points)):
            windows.append((voltage_points[i], voltage_points[j]))
    
    return windows

# Generate all 210 voltage windows
all_voltage_windows = generate_voltage_windows()


def filter_cv_by_voltage(cv_data, voltage_range, voltages):
    """
    Filter CV data to only include measurements within the specified voltage range.

    Args:
        cv_data: Array of CV measurements
        voltage_range: Tuple of (min_voltage, max_voltage)
        voltages: Array of voltage values corresponding to cv_data

    Returns:
        Filtered CV data array
    """
    min_v, max_v = voltage_range
    mask = (voltages >= min_v) & (voltages <= max_v)
    return cv_data[mask]


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Neural network hyperparameter search for coffee analysis')
    parser.add_argument("--target", type=str, required=False, 
                        choices=['HPLC_Caff', 'HPLC_CGA', 'TDS', 'all'],
                        default='all',
                        help='Target variable to train on (default: all)')
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if device == 'cpu':
        exit()


    # Define 15 different network architectures to test
    def get_network_architectures():
        """Returns a list of 15 diverse network architectures"""
        architectures = [
            # Simple shallow networks
            # Add these to the architectures list in get_network_architectures()
            {
                'network': lambda input_size: nn.Sequential(
                    nn.Linear(input_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                ),
                'network_name': 'Tiny-64-1'
            },
            {
                'network': lambda input_size: nn.Sequential(
                    nn.Linear(input_size, 32),
                    nn.Tanh(),
                    nn.Linear(32, 1)
                ),
                'network_name': 'Tiny-32-1-Tanh'
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
            {
                'network': lambda input_size: nn.Sequential(
                    nn.Linear(input_size, 32),
                    nn.LeakyReLU(0.1),
                    nn.Linear(32, 16),
                    nn.LeakyReLU(0.1),
                    nn.Linear(16, 1)
                ),
                'network_name': 'Small-32-16-1-Leaky'
            },
            {
                'network': lambda input_size: nn.Sequential(
                    nn.Linear(input_size, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1)
                ),
                'network_name': 'Minimal-16-1'
            },
            {
                'network': lambda input_size: nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 1)
                ),
                'network_name': 'Simple-128-1'
            },
            {
                'network': lambda input_size: nn.Sequential(
                    nn.Linear(input_size, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 1)
                ),
                'network_name': 'Simple-256-1'
            },
            {
                'network': lambda input_size: nn.Sequential(
                    nn.Linear(input_size, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 1)
                ),
                'network_name': 'Simple-512-1'
            },
            # Two-layer networks with varying widths
            {
                'network': lambda input_size: nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 1)
                ),
                'network_name': 'TwoLayer-128-64-1'
            },
            {
                'network': lambda input_size: nn.Sequential(
                    nn.Linear(input_size, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 1)
                ),
                'network_name': 'TwoLayer-256-64-1'
            },
            {
                'network': lambda input_size: nn.Sequential(
                    nn.Linear(input_size, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 1)
                ),
                'network_name': 'TwoLayer-256-128-1'
            },
            {
                'network': lambda input_size: nn.Sequential(
                    nn.Linear(input_size, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 1)
                ),
                'network_name': 'TwoLayer-512-128-1'
            },
            # Three-layer networks
            {
                'network': lambda input_size: nn.Sequential(
                    nn.Linear(input_size, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                    nn.Linear(64, 1)
                ),
                'network_name': 'ThreeLayer-256-128-64-1'
            },
            {
                'network': lambda input_size: nn.Sequential(
                    nn.Linear(input_size, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                    nn.Linear(128, 1)
                ),
                'network_name': 'ThreeLayer-512-256-128-1'
            },
            # Networks with higher dropout
            {
                'network': lambda input_size: nn.Sequential(
                    nn.Linear(input_size, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 1)
                ),
                'network_name': 'HighDrop-256-128-1'
            },
            {
                'network': lambda input_size: nn.Sequential(
                    nn.Linear(input_size, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, 1)
                ),
                'network_name': 'HighDrop-512-256-1'
            },
            # Wide shallow network
            {
                'network': lambda input_size: nn.Sequential(
                    nn.Linear(input_size, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(1024, 1)
                ),
                'network_name': 'Wide-1024-1'
            },
            # Bottleneck architectures
            {
                'network': lambda input_size: nn.Sequential(
                    nn.Linear(input_size, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 1)
                ),
                'network_name': 'Bottleneck-512-64-1'
            },
            # Four-layer deep network
            {
                'network': lambda input_size: nn.Sequential(
                    nn.Linear(input_size, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1)
                ),
                'network_name': 'FourLayer-512-256-128-64-1'
            },
            # Pyramid architecture
            {
                'network': lambda input_size: nn.Sequential(
                    nn.Linear(input_size, 384),
                    nn.BatchNorm1d(384),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                    nn.Linear(384, 192),
                    nn.BatchNorm1d(192),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                    nn.Linear(192, 96),
                    nn.BatchNorm1d(96),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                    nn.Linear(96, 1)
                ),
                'network_name': 'Pyramid-384-192-96-1'
            },
        ]
        return architectures


    # Create experiment configurations for each target
    base_config = {
        'NORMALIZE': False,
        'REDOX': False,
        'USE_BINS': False,
        'num_epochs': 3000,
    }

    target_configs = {}
    
    # Determine which targets to run
    if args.target == 'all':
        target_list = ['HPLC_Caff', 'HPLC_CGA', 'TDS']
    else:
        target_list = [args.target]
    
    for target_name in target_list:
        experiments = []
        # For each voltage window
        for voltage_window in all_voltage_windows:
            # For each architecture
            for arch in get_network_architectures():
                exp = base_config.copy()
                exp.update(arch)
                exp['voltage_range'] = voltage_window  # Add voltage window to config
                experiments.append(exp)
        target_configs[target_name] = experiments

    exp_results = []

    # Loop through each target variable
    for target_name, experiments in target_configs.items():

        for experiment in tqdm(experiments, desc=f"{target_name} experiments"):
            # Get voltage range for this experiment
            voltage_range = experiment['voltage_range']
            v_min, v_max = voltage_range
            
            experiment_name = f'SingleTarget-{target_name}-'

            if experiment.get('bins') == True:
                experiment_name += 'NORM-' if experiment["NORMALIZE"] else "NONORM-"

            experiment_name += 'REDOX-' if experiment["REDOX"] else "OX-"

            if "add_noise" in experiment:
                if experiment["add_noise"]:
                    experiment_name += f"NOISE{experiment['add_noise']}-{experiment['noise_level']}-"
                else:
                    experiment_name += "NONOISE-"

            # Add voltage window to experiment name
            experiment_name += f"{experiment['network_name']}-V{v_min:.1f}-{v_max:.1f}-{experiment['num_epochs']}"

            all_data_path = DATADIR / f'{experiment_name}_all.pkl'

            # Build experiment data
            if all_data_path.exists():
                df_all = pd.read_pickle(all_data_path)
            else:
                df_all = build_model_data(test_train_split=False, **experiment)
                df_all.to_pickle(all_data_path)

            coffees = df_all['Coffee Name'].unique()

            # Get voltage array (assuming it's stored in the dataframe or can be accessed)
            # This assumes the voltage values are available - adjust based on your data structure
            # You may need to get this from build_model_data or store it separately
            if 'voltages' in df_all.columns:
                voltages = df_all['voltages'].iloc[0]
            else:
                # If voltages aren't stored, you'll need to reconstruct them
                # This is a placeholder - adjust based on your actual voltage array
                voltages = None

            # Leave-one-coffee-out cross-validation
            for fold, test_coffee in enumerate(coffees):

                test_mask = df_all['Coffee Name'] == test_coffee
                df_train = df_all[~test_mask]
                df_test = df_all[test_mask]

                e = {
                    'fold': fold,
                    'test_coffee': test_coffee,
                    'all_data_path': all_data_path,
                    'experiment_name': experiment_name,
                    'target': target_name,
                    'voltage_range': voltage_range,
                    'v_min': v_min,
                    'v_max': v_max,
                    'network_name': experiment['network_name']
                }

                # Setup train data with voltage filtering
                cv_raw_list = df_train['cv_raw'].to_list()

                # Filter CV data by voltage range for this experiment
                if voltages is not None:
                    X_train_filtered = []
                    for cv_data in cv_raw_list:
                        filtered_cv = filter_cv_by_voltage(cv_data, voltage_range, voltages)
                        X_train_filtered.append(filtered_cv)
                    X_train = np.array(X_train_filtered)
                else:
                    X_train = np.array(cv_raw_list)

                y_train = np.array(df_train[target_name].values).reshape(-1, 1)  # Single target

                input_size = X_train.shape[1]

                # Add noise if specified
                if 'add_noise' in experiment and experiment['add_noise']:
                    noise_num = experiment['add_noise']
                    noise_level = experiment['noise_level']

                    X_list = []
                    y_list = []

                    for _ in range(noise_num):
                        noise = np.random.normal(0, noise_level, X_train.shape)
                        X_list.append(X_train + noise)
                        y_list.append(y_train)

                    X_train = np.concatenate(X_list)
                    y_train = np.concatenate(y_list)

                X_scaler = StandardScaler().fit(X_train)
                y_scaler = StandardScaler().fit(y_train)

                X_train_standard = X_scaler.transform(X_train)
                y_train_standard = y_scaler.transform(y_train)

                # Setup test data with voltage filtering
                cv_raw_test_list = df_test['cv_raw'].to_list()

                # Filter CV data by voltage range for this experiment
                if voltages is not None:
                    X_test_filtered = []
                    for cv_data in cv_raw_test_list:
                        filtered_cv = filter_cv_by_voltage(cv_data, voltage_range, voltages)
                        X_test_filtered.append(filtered_cv)
                    X_test = np.array(X_test_filtered)
                else:
                    X_test = np.array(cv_raw_test_list)

                y_test = np.array(df_test[target_name].values).reshape(-1, 1)  # Single target

                X_test_standard = X_scaler.transform(X_test)
                y_test_standard = y_scaler.transform(y_test)

                model = CoffeeNetBase()
                model.network = experiment['network'](input_size)

                # Train the model (no checkpoint saving)
                model.to(device)
                model = train_coffeenet(model,
                                        X_train_standard, y_train_standard,
                                        X_test_standard, y_test_standard,
                                        num_epochs=experiment['num_epochs'])

                # Evaluate on training data
                train_predictions = evaluate_model(model, X_train_standard, y_train_standard)
                train_predictions_original = y_scaler.inverse_transform(train_predictions)

                e[f'train_{target_name}_r2'] = r2_score(y_train, train_predictions_original)
                e[f'train_{target_name}_mae'] = mean_absolute_error(y_train, train_predictions_original)
                e[f'train_{target_name}_predictions'] = train_predictions_original.flatten()
                e[f'train_{target_name}_actual'] = y_train.flatten()
                e[f'train_{target_name}_error_pct'] = 100.0 * (
                        train_predictions_original.flatten() - y_train.flatten()) / y_train.flatten()

                # Evaluate on test data
                test_predictions = evaluate_model(model, X_test_standard, y_test_standard)
                test_predictions_original = y_scaler.inverse_transform(test_predictions)

                e[f'test_{target_name}_r2'] = r2_score(y_test, test_predictions_original)
                e[f'test_{target_name}_mae'] = mean_absolute_error(y_test, test_predictions_original)
                e[f'test_{target_name}_predictions'] = test_predictions_original.flatten()
                e[f'test_{target_name}_actual'] = y_test.flatten()
                e[f'test_{target_name}_error_pct'] = 100.0 * (
                        test_predictions_original.flatten() - y_test.flatten()) / y_test.flatten()

                exp_results.append(e)

    ### SAVE RESULTS ###

    df_results = pd.DataFrame(exp_results)

    # Add percent error means
    for target_name in target_configs.keys():
        df_results[f'train_{target_name}_error_pct_mean'] = df_results[f'train_{target_name}_error_pct'].apply(
            lambda x: np.mean(x) if isinstance(x, np.ndarray) else np.nan
        )
        df_results[f'test_{target_name}_error_pct_mean'] = df_results[f'test_{target_name}_error_pct'].apply(
            lambda x: np.mean(x) if isinstance(x, np.ndarray) else np.nan
        )

    # Save results
    results_path = DATADIR / 'full_window_search_results.pkl'
    df_results.to_pickle(results_path)
    
    # Also save as CSV for easier inspection
    csv_path = DATADIR / 'full_window_search_results.csv'
    df_results.to_csv(csv_path, index=False)

    # Create summary comparison across all targets
    summary_data = []
    for target_name in target_configs.keys():
        df_target = df_results[df_results['target'] == target_name]
        df_avg = df_target.groupby(['experiment_name', 'v_min', 'v_max', 'network_name']).agg({
            f'test_{target_name}_r2': 'mean',
            f'test_{target_name}_mae': 'mean'
        }).reset_index()
        
        best_idx = df_avg[f'test_{target_name}_r2'].idxmax()
        best_row = df_avg.loc[best_idx]
        
        summary_data.append({
            'Target': target_name,
            'Voltage Range': f"{best_row['v_min']:.1f}-{best_row['v_max']:.1f}V",
            'Best Architecture': best_row['network_name'],
            'Test R²': best_row[f'test_{target_name}_r2'],
            'Test MAE': best_row[f'test_{target_name}_mae']
        })

    df_summary = pd.DataFrame(summary_data)