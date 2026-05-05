"""
GRID SEARCH REGRESSION INSTRUCTIONS:
====================================
1. Place in the same directory as util.py
2. Run: python regression_grid_search.py
3. The script will:
   - Apply Baseline Correction (Subtract current at 0.8V) if enabled
   - Scan voltage ranges (0.6-1.0V for CGA/TDS, 1.1-1.5V for Caff)
   - Train a LOOCV regression model for EVERY 0.01V step
   - Automatically select the voltage with the highest Test R²
   - Plot and save only the best models
"""

import os
import sys

UTIL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
if UTIL_DIR not in sys.path:
    sys.path.insert(0, UTIL_DIR)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nn_0_synthetic_data_gen import build_model_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

from util import setup_mplt, DATADIR, PLOTDIR

# Clear cache to ensure fresh data load
cache_file = DATADIR / 'raw_data_cache.pkl'
if cache_file.exists():
    os.remove(cache_file)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Baseline Correction: Subtract the response at ref_voltage from the entire curve
BASELINE_CORRECTION = True
BASELINE_REF_VOLTAGE = 0.8

# Search Ranges
search_ranges = {
    'HPLC_CGA': (0.0, 2.0),
    'HPLC_Caff': (0.0, 2.0),
    'TDS': (0.0, 2.0)
}

def get_voltage_index(voltages, target_v):
    """Finds the index in the voltage array closest to target_v"""
    if hasattr(voltages, 'values'):
        voltages = voltages.values
    return np.argmin(np.abs(voltages - target_v))

def apply_baseline_correction_to_df(df, voltages, ref_voltage):
    """
    Subtracts the current value at ref_voltage from the entire curve
    for every sample in the dataframe.
    """
    ref_idx = get_voltage_index(voltages, ref_voltage)

    # Create a copy to avoid SettingWithCopy warnings
    new_cv_data = []

    for cv_curve in df['cv_raw']:
        # FIX: Ensure we use the raw values, not a pandas Series with an index
        if hasattr(cv_curve, 'values'):
            curve_vals = cv_curve.values
        else:
            curve_vals = np.array(cv_curve)

        baseline_val = curve_vals[ref_idx]
        new_cv_data.append(curve_vals - baseline_val)

    df['cv_raw'] = new_cv_data
    return df

def get_voltage_response(cv_data, target_v, voltages):
    """
    Extract the response at a specific single voltage point.
    """
    # Ensure inputs are numpy arrays to prevent indexing errors
    if hasattr(cv_data, 'values'):
        cv_data = cv_data.values
    elif not isinstance(cv_data, np.ndarray):
        cv_data = np.array(cv_data)

    if hasattr(voltages, 'values'):
        voltages = voltages.values

    # Find index closest to the target voltage
    idx = np.argmin(np.abs(voltages - target_v))
    return cv_data[idx]


def train_and_evaluate_regression(df_all, target_name, target_v, voltages):
    """
    Train and evaluate linear regression with Leave-One-Coffee-Out CV.
    Returns the results dictionary including metrics.
    """
    coffees = df_all['Coffee Name'].unique()

    results = {
        'train_actual': [], 'train_pred': [],
        'test_actual': [], 'test_pred': [],
        'test_coffees': [], 'fold_metrics': []
    }

    # Leave-One-Out Cross-Validation Loop
    for fold, test_coffee in enumerate(coffees):
        test_mask = df_all['Coffee Name'] == test_coffee
        df_train = df_all[~test_mask]
        df_test = df_all[test_mask]

        # Extract feature: Response at target_v
        X_train = np.array([[get_voltage_response(cv, target_v, voltages)] for cv in df_train['cv_raw']])
        y_train = df_train[target_name].values

        X_test = np.array([[get_voltage_response(cv, target_v, voltages)] for cv in df_test['cv_raw']])
        y_test = df_test[target_name].values

        # Train Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Collect Results
        results['train_actual'].extend(y_train)
        results['train_pred'].extend(y_train_pred)
        results['test_actual'].extend(y_test)
        results['test_pred'].extend(y_test_pred)
        results['test_coffees'].extend([test_coffee] * len(y_test))

    # Calculate Overall Metrics (Aggregated across all folds)
    results['train_actual'] = np.array(results['train_actual'])
    results['train_pred'] = np.array(results['train_pred'])
    results['test_actual'] = np.array(results['test_actual'])
    results['test_pred'] = np.array(results['test_pred'])

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
    print("VOLTAGE GRID SEARCH REGRESSION")
    print(f"Baseline Correction: {BASELINE_CORRECTION} (Ref: {BASELINE_REF_VOLTAGE}V)")
    print("Scanning voltage ranges to find optimal correlation points...")
    print("=" * 80)

    # 1. Load Data
    all_data_path = DATADIR / 'regression_simple_all.pkl'
    if all_data_path.exists():
        print(f"Loading data from {all_data_path}")
        df_all = pd.read_pickle(all_data_path)
    else:
        print("Building model data...")
        df_all = build_model_data(test_train_split=False, NORMALIZE=False, REDOX=False, USE_BINS=False)
        df_all.to_pickle(all_data_path)

    # Get Voltage Array
    if 'voltages' in df_all.columns:
        voltages = df_all['voltages'].iloc[0]
    else:
        # Fallback: assume CV runs from 0 to 2 V
        voltages = np.linspace(0.0, 2.0, len(df_all['cv_raw'].iloc[0]))

    print(f"Voltage resolution: {(voltages[1] - voltages[0]):.4f} V")

    # --- APPLY BASELINE CORRECTION ---
    if BASELINE_CORRECTION:
        print(f"Applying baseline correction (subtracting current at {BASELINE_REF_VOLTAGE} V)...")
        df_all = apply_baseline_correction_to_df(df_all, voltages, BASELINE_REF_VOLTAGE)
    # ---------------------------------

    best_models = {}
    target_names_display = {'HPLC_CGA': 'CGA', 'HPLC_Caff': 'Caffeine', 'TDS': 'TDS'}

    # 2. Main Search Loop
    for target_name, (start_v, end_v) in search_ranges.items():
        print(f"\nScanning {target_names_display[target_name]} [{start_v}V -> {end_v}V]...")

        # Generate search grid (inclusive of end_v)
        search_grid = np.arange(start_v, end_v + 0.001, 0.01)

        best_r2 = -np.inf
        best_result = None
        best_v = None

        # Progress bar manually
        for v in search_grid:
            # Train model at this specific voltage
            res = train_and_evaluate_regression(df_all, target_name, v, voltages)

            # Check if this is the best model so far (based on Test R2)
            current_r2 = res['overall_metrics']['test_r2']

            if current_r2 > best_r2:
                best_r2 = current_r2
                best_result = res
                best_v = v

            # Optional: Print progress dots
            sys.stdout.write('')
            sys.stdout.flush()

        print(f"\n   -> Best Voltage found: {best_v:.2f} V (Test R²: {best_r2:.4f})")

        # Save the winner
        best_models[target_name] = {
            'voltage': best_v,
            'results': best_result
        }

    # 3. Visualization of BEST models
    print(f"\n{'=' * 80}")
    print("Creating visualizations for best models...")

    # --- Visualization: uA response vs. actual value (TEST ONLY, with fit line) ---

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    title_suffix = f"(Baseline Corrected @ {BASELINE_REF_VOLTAGE}V)" if BASELINE_CORRECTION else "(Raw Data)"
    fig.suptitle(f'uA Response vs. Actual Value {title_suffix}', fontsize=14, fontweight='bold')

    for idx, (target_name, data) in enumerate(best_models.items()):
        results = data['results']
        v_point = data['voltage']

        # Test set: mask and extract
        test_mask = df_all['Coffee Name'].isin(results['test_coffees'])
        X_test_uA = np.array([
            get_voltage_response(cv, v_point, voltages)
            for cv in df_all.loc[test_mask, 'cv_raw']
        ])
        y_test = df_all.loc[test_mask, target_name].values

        test_r2 = results['overall_metrics']['test_r2']

        ax_test = axes[idx]

        # Scatter of test points
        ax_test.scatter(X_test_uA, y_test, alpha=0.7, color='orange', edgecolors='k', s=30, label='Test data')

        # Fit regression line on test data for visualization
        if X_test_uA.size > 1:
            X_line = X_test_uA.reshape(-1, 1)
            reg = LinearRegression().fit(X_line, y_test)
            x_min, x_max = X_test_uA.min(), X_test_uA.max()
            x_fit = np.linspace(x_min, x_max, 100)
            y_fit = reg.predict(x_fit.reshape(-1, 1))
            ax_test.plot(x_fit, y_fit, color='red', linestyle='--', linewidth=2, label='Fit line')

        ax_test.set_xlabel(f'uA Response @ {v_point:.2f}V')
        ax_test.set_ylabel('Actual Value')
        ax_test.set_title(
            f"{target_names_display[target_name]} Test LOOCV\n"
            f"Best V: {v_point:.2f}V, R² = {test_r2:.3f}"
        )
        ax_test.grid(True, alpha=0.3)
        # Make each subplot visually square
        try:
            ax_test.set_box_aspect(1)
        except AttributeError:
            pass

        ax_test.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0.08, 1, 0.9])
    filename_suffix = "_baseline_corrected" if BASELINE_CORRECTION else "_raw"
    save_path = PLOTDIR / f'regression_uA_vs_actual{filename_suffix}.png'
    plt.savefig(save_path, dpi=300)
    print(f"uA vs. Actual plot saved to {save_path}")
    plt.show()

    # --- Visualization: All voltammograms with best-voltage markers ---

    print("\nCreating voltammogram plot with best-voltage markers for CGA, Caffeine, and TDS...")

    fig2, ax2 = plt.subplots(figsize=(12, 8))

    # Select lowest 5 and highest 5 caffeine samples
    df_sorted_caff = df_all.sort_values('HPLC_Caff')
    df_subset = pd.concat([df_sorted_caff.head(5), df_sorted_caff.tail(5)])

    # Plot each selected sample's voltammogram
    for _, row in df_subset.iterrows():
        cv_curve = row['cv_raw']
        if hasattr(cv_curve, 'values'):
            cv_vals = cv_curve.values
        else:
            cv_vals = np.array(cv_curve)

        ax2.plot(voltages, cv_vals, color='lightgray', alpha=0.4, linewidth=0.8)

        # Overlay marker only for Caffeine's best voltage with response & caffeine labels
        best_v = best_models['HPLC_Caff']['voltage']
        idx = get_voltage_index(voltages, best_v)
        x = voltages[idx]
        y = cv_vals[idx]

        caff_val = row['HPLC_Caff']
        response_val = y

        ax2.scatter(x, y, color='tab:red', marker='s', s=15, alpha=0.9)
        ax2.text(
            x,
            y,
            f"resp {response_val:.3f}, Caff {caff_val:.1f}",
            fontsize=3,
            color='tab:red',
            ha='left',
            va='bottom',
        )

    ax2.set_xlabel('Voltage (V)')
    ax2.set_ylabel('Current (uA)')
    title2 = 'Coffee Voltammograms with Best-Voltage Markers'
    if BASELINE_CORRECTION:
        title2 += f" (Baseline @ {BASELINE_REF_VOLTAGE} V)"
    ax2.set_title(title2)
    ax2.grid(True, alpha=0.3)

    # Zoom into the region of interest
    ax2.set_xlim(1.0, 1.5)

    # Legend using proxy artists
    from matplotlib.lines import Line2D

    legend_elements = [
         Line2D([0], [0], marker='s', color='w', label='Caffeine best V',
             markerfacecolor='tab:red', markersize=6),
        ]
    ax2.legend(handles=legend_elements, loc='best', fontsize=8)

    volt_plot_name = 'voltammograms_best_voltage_markers'
    if BASELINE_CORRECTION:
        volt_plot_name += '_baseline_corrected'
    volt_save_path = PLOTDIR / f'{volt_plot_name}.png'
    plt.tight_layout()
    plt.savefig(volt_save_path, dpi=300)
    print(f"Voltammogram plot with best-voltage markers saved to {volt_save_path}")
    plt.show()

    # 4. Final Summary Table
    summary_data = []
    for target_name, data in best_models.items():
        res = data['results']
        metrics = res['overall_metrics']
        summary_data.append({
            'Target': target_names_display[target_name],
            'Best Voltage': f"{data['voltage']:.2f} V",
            'Train R²': f"{metrics['train_r2']:.4f}",
            'Test R²': f"{metrics['test_r2']:.4f}",
            'Test MAE': f"{metrics['test_mae']:.4f}"
        })

    print("\n" + "=" * 60)
    print("FINAL OPTIMIZATION RESULTS")
    print(f"Baseline Corrected: {BASELINE_CORRECTION}")
    print("=" * 60)
    print(pd.DataFrame(summary_data).to_string(index=False))
    print("\n" + "=" * 60)