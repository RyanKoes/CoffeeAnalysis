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

# --- repo-root bootstrap (added by reorg) ---
import sys as _sys
from pathlib import Path as _Path
_repo_root = _Path(__file__).resolve().parents[2]
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))
# --- end bootstrap ---


import os
import sys

UTIL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
if UTIL_DIR not in sys.path:
    sys.path.insert(0, UTIL_DIR)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util import setup_mplt, build_model_data, DATADIR, PLOTDIR

from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_absolute_error


from model_search_evaluation import plot_predictions

# Clear cache to ensure fresh data load
cache_file = DATADIR / 'pls_data_cache.pkl'
#if cache_file.exists():
#    os.remove(cache_file)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Baseline Correction: Subtract the response at ref_voltage from the entire curve
BASELINE_CORRECTION = False
BASELINE_REF_VOLTAGE = 0.8


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


def train_and_evaluate_pls_regression(df_all, target_name, n_components):
    """
    Train and evaluate PLS Regression with Leave-One-Coffee-Out CV.
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
        #X_train = np.array([[get_voltage_response(cv, target_v, voltages)] for cv in df_train['cv_raw']])
        X_train = np.array([cv.values for cv in df_train['cv_raw']])
        y_train = df_train[target_name].values

        #X_test = np.array([[get_voltage_response(cv, target_v, voltages)] for cv in df_test['cv_raw']])
        X_test = np.array([cv.values for cv in df_test['cv_raw']])
        y_test = df_test[target_name].values

        # Train Model
        model = PLSRegression(n_components=n_components)
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
    print("Partial Least Squares REGRESSION")
    target_names_display = {'HPLC_CGA': 'CGA', 'HPLC_Caff': 'Caffeine', 'TDS': 'TDS'}

    if cache_file.exists():
        best_models = pd.read_pickle(cache_file)
        print(f"USING CACHED RESULTS from {cache_file}")
        print("=" * 80)
    else:

        #print(f"Baseline Correction: {BASELINE_CORRECTION} (Ref: {BASELINE_REF_VOLTAGE}V)")
        #print("Scanning voltage ranges to find optimal correlation points...")
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

        # 2. Compute PLS Regression with Leave-One-Coffee-Out CV (search n_components)
        for target_name in target_names_display.keys():
            best_r2 = -np.inf
            best_result = None
            best_comp = None
            for n_components in range(2,11):

                print(f"Running PLS Regression on {target_name} with {n_components} components...", end="")
                
                # Train model at this specific voltage
                res = train_and_evaluate_pls_regression(df_all, target_name, n_components)

                # Check if this is the best model so far (based on Test R2)
                current_r2 = res['overall_metrics']['test_r2']

                # print result for debuggging
                print(f" - Test R2: {current_r2:.4f}")

                # Save the result
                if current_r2 > best_r2:
                    best_r2 = current_r2
                    best_result = res
                    best_comp = n_components

            best_models[target_name] = {
                'r2': best_r2,
                'n_components': best_comp,
                'results': best_result
            }
        pd.to_pickle(best_models, cache_file)



    # 3. Visualization of BEST models
    print(f"\n{'=' * 80}")
    print("Creating visualizations for best models...")


    # --- Visualization: All voltammograms with best-voltage markers ---

    filename_suffix = "_baseline_corrected" if BASELINE_CORRECTION else "_raw"
    save_path = PLOTDIR / f'regression_uA_vs_actual{filename_suffix}.png'
    
    print("\nCreating voltammogram plot with best-voltage markers for CGA, Caffeine, and TDS...")

    volt_plot_name = 'pls_voltammograms_best_voltage_markers'
    if BASELINE_CORRECTION:
        volt_plot_name += '_baseline_corrected'
    volt_save_path = PLOTDIR / f'{volt_plot_name}.png'
    

    


    # 4. Final Summary Table
    summary_data = []
    for target_name, data in best_models.items():

        plot_predictions(target_name, "PLS", 
                         data['results']['train_actual'],
                         data['results']['train_pred'],
                         data['results']['test_actual'],
                         data['results']['test_pred'], show=True)


        res = data['results']
        metrics = res['overall_metrics']
        summary_data.append({
            'Target': target_names_display[target_name],
            'n_comp': f"{data['n_components']:2d}",
            'Train R²': f"{metrics['train_r2']:.4f}",
            'Test R²': f"{metrics['test_r2']:.4f}",
            'Test MAE': f"{metrics['test_mae']:.4f}"
        })

    print("\n" + "=" * 60)
    print("PLS FINAL OPTIMIZATION RESULTS")
    print(f"Baseline Corrected: {BASELINE_CORRECTION}")
    print("=" * 60)
    print(pd.DataFrame(summary_data).to_string(index=False))
    print("\n" + "=" * 60)