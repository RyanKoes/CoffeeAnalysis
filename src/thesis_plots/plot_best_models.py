
import sys
import os
from pathlib import Path
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tqdm import tqdm

# Add parent directory to sys.path to allow imports from root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ThesisPlotGeneration.plot_generator import apply_publication_style, savefig_pdf, FIGSIZE_SINGLE_COLUMN
from nn_0_synthetic_data_gen import build_model_data
from nn_1_train_model import CoffeeNetBase, train_coffeenet, evaluate_model
from nn_model_window_search import get_network_architectures

def load_best_params(target, data_dir):
    filename = f"BEST_{target}_fixed.pkl"
    filepath = data_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Could not find {filepath}")
    
    print(f"Loading best params for {target} from {filepath}...")
    try:
        data = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)
        metadata = data.get('metadata', {})
        return metadata['window'], metadata['architecture'], metadata.get('r2')
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None, None

def perform_loco_cv(target_name, window, architecture_name):
    print(f"\nrunning LOCO CV for {target_name}...")
    print(f"Window: {window}, Architecture: {architecture_name}")
    
    # Load data (similar to nn_model_window_search.py)
    # Ensure cache is fresh or ignored if needed. But build_model_data doesn't expose it.
    # We rely on previous step having deleted cache.
    df_all = build_model_data(
        NORMALIZE=False,
        REDOX=False,
        USE_BINS=False, 
        test_train_split=False,
    )
    
    # Drop rows where target is missing (e.g. incomplete data)
    df_all = df_all.dropna(subset=[target_name])
    
    unique_coffees = df_all['Coffee Name'].unique()
    print(f"Loaded {len(df_all)} samples from {len(unique_coffees)} unique coffees.")
    
    # Assume underlying CV spans 0 to 2 V with uniform spacing (as per nn_model_window_search.py)
    # The cv_raw column contains the time-series/voltage-series 
    # Check shape to determine n_points
    sample_cv = df_all['cv_raw'].iloc[0]
    n_points = len(sample_cv)
    full_voltage_array = np.linspace(0.0, 2.0, n_points)
    
    vmin, vmax = window
    
    # Create mask for the specific window
    voltage_mask = (full_voltage_array >= vmin) & (full_voltage_array <= vmax)
    voltage_indices = np.where(voltage_mask)[0]
    
    if len(voltage_indices) == 0:
        print("Error: No voltage points found in window!")
        return [], []

    # Get Architecture definition
    architectures = get_network_architectures()
    arch_def = next((a for a in architectures if a['network_name'] == architecture_name), None)
    
    if arch_def is None:
        print(f"Error: Architecture '{architecture_name}' not found!")
        # Fallback or exit?
        return [], []
    
    coffees = df_all['Coffee Name'].unique()
    
    all_test_actual = []
    all_test_pred = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for test_coffee in tqdm(coffees, desc=f"LOCO {target_name}"):
        test_mask = df_all['Coffee Name'] == test_coffee
        df_train = df_all[~test_mask]
        df_test = df_all[test_mask]
        
        # Prepare X (features)
        X_train = np.array([np.asarray(cv)[voltage_indices] for cv in df_train['cv_raw']])
        X_test = np.array([np.asarray(cv)[voltage_indices] for cv in df_test['cv_raw']])
        
        # Prepare y (targets)
        y_train = df_train[target_name].values.reshape(-1, 1)
        y_test = df_test[target_name].values.reshape(-1, 1)
        
        # Scale
        X_scaler = StandardScaler().fit(X_train)
        y_scaler = StandardScaler().fit(y_train)
        
        X_train_s = X_scaler.transform(X_train)
        y_train_s = y_scaler.transform(y_train)
        X_test_s = X_scaler.transform(X_test)
        
        # Initialize model
        model = CoffeeNetBase()
        model.network = arch_def['network'](X_train.shape[1])
        model.to(device)
        
        # Train (using full epochs as USE_EARLY_STOPPING is False in original script by default)
        # We'll use a fixed number of epochs or whatever consistent with nn_model_window_search default
        # nn_model_window_search uses 2000 epochs.
        model = train_coffeenet(
            model, 
            X_train_s, y_train_s, 
            X_test=None, y_test=None, # No early stopping peeking
            num_epochs=2000
        )
        
        # Predict
        test_pred_s = evaluate_model(model, X_test_s, y_train_s) # y argument in evaluate_model is unused for prediction returns
        # Inverse transform
        test_pred_original = y_scaler.inverse_transform(test_pred_s)
        
        all_test_actual.extend(y_test.flatten())
        all_test_pred.extend(test_pred_original.flatten())
        
    return all_test_actual, all_test_pred

def main():
    # Use square figure for parity plots to avoid "thin" look and match publication style
    apply_publication_style(figsize=(3.35, 3.35))
    
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    targets = ["HPLC_Caff", "HPLC_CGA", "TDS"]
    
    for target in targets:
        window, architecture, stored_r2 = load_best_params(target, data_dir)
        if window is None:
            continue
            
        print(f"Target: {target}")
        print(f"  Architecture: {architecture}")
        print(f"  Window: {window}")
        if stored_r2 is not None:
            print(f"  Stored Best R2: {stored_r2:.4f}")
        
        actuals, preds = perform_loco_cv(target, window, architecture)
        
        if not actuals:
            print(f"Skipping plot for {target} due to missing data/errors.")
            continue
            
        # Plotting
        # Match height scaling from plotting_notebook (e.g. Cell 14 uses h * 1.3)
        w, h = FIGSIZE_SINGLE_COLUMN
        figsize = (w, h * 1.3)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot
        # Matching Cell 12 style in plotting_notebook.ipynb: s=18, alpha=0.8, zorder=2
        ax.scatter(actuals, preds, s=18, alpha=0.8, zorder=2)
        
        # Identity line
        min_val = min(min(actuals), min(preds))
        max_val = max(max(actuals), max(preds))
        # Add some padding
        range_val = max_val - min_val
        plot_min = min_val - 0.05 * range_val
        plot_max = max_val + 0.05 * range_val
        
        # Consistent line style (dashed black or red, but clean)
        ax.plot([plot_min, plot_max], [plot_min, plot_max], color='black', linestyle='--', linewidth=1.6, zorder=3, alpha=0.5, label='1:1 Line')
        
        # Calculate R2
        r2 = r2_score(actuals, preds)
        
        print(f"  Calculated LOCO R2: {r2:.4f}")
        if stored_r2 is not None:
            print(f"  Difference: {r2 - stored_r2:.4f}")
        
        ax.set_xlabel(f"Actual {target}")
        ax.set_ylabel(f"Predicted {target}")
        
        # Match title format from notebook (Cell 12): "Title\nR2=..."
        ax.set_title(f"{target}\nArchitecture: {architecture}\n$R^2 = {r2:.3f}$")
        
        # Removed text box to match notebook style favoring title stats
        
        plt.tight_layout()

        
        filename = f"scatter_{target}.pdf"
        out_path = plots_dir / filename
        savefig_pdf(str(out_path))
        print(f"Saved plot to {out_path}")
        plt.close(fig)

if __name__ == "__main__":
    main()
