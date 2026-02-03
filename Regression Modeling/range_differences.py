"""
INTEGRATION INSTRUCTIONS:
========================
To use this script with your existing codebase:

1. Place this file in the same directory as your other scripts (where util.py is located)
2. Run: python regression_model.py
3. The script will:
   - Load your coffee data
   - Extract voltage range differences as features
   - Train separate linear regression models for CGA, Caffeine, and TDS
   - Generate Plots and performance metrics
   - Save results to DATADIR and PLOTDIR

KEY CHANGES FROM NEURAL NETWORKS:
==================================
- Feature: Single value per sample = CV_response(V_max) - CV_response(V_min)
- Model: Simple Linear Regression (y = mx + b)
- Much faster training (seconds instead of minutes)
- More interpretable (direct relationship between voltage difference and target)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nn_0_synthetic_data_gen import build_model_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

from util import setup_mplt, DATADIR, PLOTDIR, read_coffehub

# Clear cache to reload fresh data
cache_file = DATADIR / 'raw_data_cache.pkl'
if cache_file.exists():
    os.remove(cache_file)

df = read_coffehub()

# Define voltage ranges for each target
voltage_ranges = {
    'HPLC_CGA': (0.0, 0.8),  # CGA: 0 to 0.8 V
    'HPLC_Caff': (1.0, 1.3),  # Caffeine: 1 to 1.3 V
    'TDS': (1.07, 1.47)  # TDS: 1.07 to 1.47 V
}


def get_voltage_range_difference(cv_data, voltage_range, voltages):
    """
    Calculate the difference between response at max and min voltage.

    This is the KEY feature extraction:
    Feature = CV_response(V_max) - CV_response(V_min)

    Args:
        cv_data: Array of CV measurements (pandas Series or numpy array)
        voltage_range: Tuple of (min_voltage, max_voltage)
        voltages: Array of voltage values corresponding to cv_data

    Returns:
        Difference between response at max_v and min_v
    """
    min_v, max_v = voltage_range

    # Convert to numpy array if it's a pandas Series
    if hasattr(cv_data, 'values'):
        cv_data = cv_data.values
    if hasattr(voltages, 'values'):
        voltages = voltages.values

    # Find indices closest to target voltages
    min_idx = np.argmin(np.abs(voltages - min_v))
    max_idx = np.argmin(np.abs(voltages - max_v))

    # Return the difference using positional indexing
    return cv_data[max_idx] - cv_data[min_idx]


def train_and_evaluate_regression(df_all, target_name, voltage_range, voltages):
    """
    Train and evaluate linear regression with leave-one-coffee-out CV.

    Returns:
        Dictionary with all results and metrics
    """
    coffees = df_all['Coffee Name'].unique()

    results = {
        'train_actual': [],
        'train_pred': [],
        'test_actual': [],
        'test_pred': [],
        'test_coffees': [],
        'fold_metrics': []
    }

    # Leave-one-coffee-out cross-validation
    for fold, test_coffee in enumerate(coffees):
        # Split data
        test_mask = df_all['Coffee Name'] == test_coffee
        df_train = df_all[~test_mask]
        df_test = df_all[test_mask]

        # Extract features (voltage range differences)
        X_train = []
        for cv_data in df_train['cv_raw']:
            diff = get_voltage_range_difference(cv_data, voltage_range, voltages)
            X_train.append([diff])  # 2D array for sklearn
        X_train = np.array(X_train)
        y_train = df_train[target_name].values

        X_test = []
        for cv_data in df_test['cv_raw']:
            diff = get_voltage_range_difference(cv_data, voltage_range, voltages)
            X_test.append([diff])
        X_test = np.array(X_test)
        y_test = df_test[target_name].values

        # Train simple linear regression
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        # Store results
        results['train_actual'].extend(y_train)
        results['train_pred'].extend(y_train_pred)
        results['test_actual'].extend(y_test)
        results['test_pred'].extend(y_test_pred)
        results['test_coffees'].extend([test_coffee] * len(y_test))

        results['fold_metrics'].append({
            'fold': fold,
            'test_coffee': test_coffee,
            'train_r2': train_r2,
            'train_mae': train_mae,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'coef': model.coef_[0],
            'intercept': model.intercept_
        })

    # Convert to arrays
    results['train_actual'] = np.array(results['train_actual'])
    results['train_pred'] = np.array(results['train_pred'])
    results['test_actual'] = np.array(results['test_actual'])
    results['test_pred'] = np.array(results['test_pred'])

    # Calculate overall metrics
    results['overall_metrics'] = {
        'train_r2': r2_score(results['train_actual'], results['train_pred']),
        'train_mae': mean_absolute_error(results['train_actual'], results['train_pred']),
        'test_r2': r2_score(results['test_actual'], results['test_pred']),
        'test_mae': mean_absolute_error(results['test_actual'], results['test_pred'])
    }

    return results


if __name__ == "__main__":
    setup_mplt()

    print("\n" + "=" * 80)
    print("SIMPLE LINEAR REGRESSION FOR COFFEE ANALYSIS")
    print("Feature: Voltage Range Difference (V_max - V_min)")
    print("=" * 80)

    # Build or load data
    all_data_path = DATADIR / 'regression_simple_all.pkl'

    if all_data_path.exists():
        print(f"\nLoading cached data from {all_data_path}")
        df_all = pd.read_pickle(all_data_path)
    else:
        print("\nBuilding model data (this may take a moment)...")
        df_all = build_model_data(
            test_train_split=False,
            NORMALIZE=False,
            REDOX=False,
            USE_BINS=False
        )
        df_all.to_pickle(all_data_path)
        print(f"Data saved to {all_data_path}")

    # Get voltage array
    if 'voltages' in df_all.columns:
        voltages = df_all['voltages'].iloc[0]
        print(f"\nVoltage array loaded: {len(voltages)} points from {voltages.min():.2f}V to {voltages.max():.2f}V")
    else:
        # Fallback: reconstruct voltage array
        print("\nWARNING: Reconstructing voltage array - please verify this matches your data!")
        num_points = len(df_all['cv_raw'].iloc[0])
        voltages = np.linspace(-0.5, 1.5, num_points)
        print(f"Created voltage array: {num_points} points from {voltages.min():.2f}V to {voltages.max():.2f}V")

    # Get coffee names
    coffees = df_all['Coffee Name'].unique()
    print(f"\nNumber of unique coffees: {len(coffees)}")
    print(f"Total samples: {len(df_all)}")

    # Train models for all targets
    all_results = {}

    for target_name, voltage_range in voltage_ranges.items():
        print(f"\n{'=' * 80}")
        print(f"Training Regression Model for {target_name}")
        print(f"Voltage range: {voltage_range[0]:.2f}V to {voltage_range[1]:.2f}V")
        print(f"{'=' * 80}")

        results = train_and_evaluate_regression(
            df_all, target_name, voltage_range, voltages
        )
        all_results[target_name] = results

        # Print results for this target
        print(f"\n{target_name} Results:")
        print(f"  Training   → R² = {results['overall_metrics']['train_r2']:.4f}, "
              f"MAE = {results['overall_metrics']['train_mae']:.4f}")
        print(f"  Test (CV)  → R² = {results['overall_metrics']['test_r2']:.4f}, "
              f"MAE = {results['overall_metrics']['test_mae']:.4f}")

        # Fold statistics
        fold_df = pd.DataFrame(results['fold_metrics'])
        print(f"\n  Cross-validation stats (across {len(coffees)} folds):")
        print(f"    Test R²  = {fold_df['test_r2'].mean():.4f} ± {fold_df['test_r2'].std():.4f}")
        print(f"    Test MAE = {fold_df['test_mae'].mean():.4f} ± {fold_df['test_mae'].std():.4f}")

    # Create visualization
    print(f"\n{'=' * 80}")
    print("Creating visualizations...")
    print(f"{'=' * 80}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Linear Regression Results: Voltage Range Difference Method\n'
                 'Feature = CV_Response(V_max) - CV_Response(V_min)',
                 fontsize=16, fontweight='bold')

    target_names_display = {
        'HPLC_CGA': 'CGA',
        'HPLC_Caff': 'Caffeine',
        'TDS': 'TDS'
    }

    for idx, (target_name, results) in enumerate(all_results.items()):
        # Training plot (top row)
        ax_train = axes[0, idx]
        ax_train.scatter(results['train_actual'], results['train_pred'],
                         alpha=0.6, s=50, edgecolors='k', linewidth=0.5, color='blue')

        # Perfect prediction line
        min_val = min(results['train_actual'].min(), results['train_pred'].min())
        max_val = max(results['train_actual'].max(), results['train_pred'].max())
        ax_train.plot([min_val, max_val], [min_val, max_val],
                      'r--', linewidth=2, label='Perfect prediction')

        ax_train.set_xlabel('Actual Value', fontsize=12, fontweight='bold')
        ax_train.set_ylabel('Predicted Value', fontsize=12, fontweight='bold')
        ax_train.set_title(f'{target_names_display[target_name]} - Training Set\n'
                           f'R² = {results["overall_metrics"]["train_r2"]:.4f}, '
                           f'MAE = {results["overall_metrics"]["train_mae"]:.4f}',
                           fontsize=11, fontweight='bold')
        ax_train.legend(loc='best')
        ax_train.grid(True, alpha=0.3)

        # Test plot (bottom row)
        ax_test = axes[1, idx]
        ax_test.scatter(results['test_actual'], results['test_pred'],
                        alpha=0.6, s=50, edgecolors='k', linewidth=0.5, color='orange')

        min_val = min(results['test_actual'].min(), results['test_pred'].min())
        max_val = max(results['test_actual'].max(), results['test_pred'].max())
        ax_test.plot([min_val, max_val], [min_val, max_val],
                     'r--', linewidth=2, label='Perfect prediction')

        ax_test.set_xlabel('Actual Value', fontsize=12, fontweight='bold')
        ax_test.set_ylabel('Predicted Value', fontsize=12, fontweight='bold')

        v_range = voltage_ranges[target_name]
        ax_test.set_title(f'{target_names_display[target_name]} - Test Set (Leave-One-Out CV)\n'
                          f'Range: {v_range[0]:.2f}V to {v_range[1]:.2f}V | '
                          f'R² = {results["overall_metrics"]["test_r2"]:.4f}, '
                          f'MAE = {results["overall_metrics"]["test_mae"]:.4f}',
                          fontsize=11, fontweight='bold')
        ax_test.legend(loc='best')
        ax_test.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = PLOTDIR / 'regression_simple_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    plt.show()

    # Save results
    results_path = DATADIR / 'regression_simple_results.pkl'
    pd.to_pickle(all_results, results_path)
    print(f"Results saved to: {results_path}")

    # Print final summary
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print(f"{'=' * 80}")

    summary_data = []
    for target_name, results in all_results.items():
        summary_data.append({
            'Target': target_names_display[target_name],
            'Voltage Range': f"{voltage_ranges[target_name][0]:.2f}-{voltage_ranges[target_name][1]:.2f}V",
            'Train R²': f"{results['overall_metrics']['train_r2']:.4f}",
            'Train MAE': f"{results['overall_metrics']['train_mae']:.4f}",
            'Test R²': f"{results['overall_metrics']['test_r2']:.4f}",
            'Test MAE': f"{results['overall_metrics']['test_mae']:.4f}"
        })

    df_summary = pd.DataFrame(summary_data)
    print("\n" + df_summary.to_string(index=False))
    print(f"\n{'=' * 80}")

    print("\n✓ Training complete!")
    print(f"✓ Results saved to {results_path}")
    print(f"✓ Plot saved to {output_path}")