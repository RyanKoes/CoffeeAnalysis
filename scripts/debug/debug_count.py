

# --- repo-root bootstrap (added by reorg) ---
import sys as _sys
from pathlib import Path as _Path
_repo_root = _Path(__file__).resolve().parents[2]
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))
# --- end bootstrap ---

from nn_0_synthetic_data_gen import build_model_data
from util import read_coffehub
import pandas as pd

def debug():
    print("--- Debugging Data Count ---")
    
    # 1. Check raw dataframe from read_coffehub
    # Pass empty list for require_columns to avoid triggering specific checks if any
    try:
        raw_df = read_coffehub(use_cache=False) # Force reload from sheet
        print(f"Raw DataFrame from Sheet: {len(raw_df)} rows")
        print("Columns:", raw_df.columns.tolist())
        print("Unique Names in Raw:", len(raw_df['Name'].unique()))
    except Exception as e:
        print(f"Error reading sheet directly: {e}")

    # 2. Check build_model_data
    df_all = build_model_data(
        NORMALIZE=False,
        REDOX=False,
        USE_BINS=False, 
        test_train_split=False,
    )
    print(f"\nbuild_model_data Result: {len(df_all)} samples")
    unique = df_all['Coffee Name'].unique()
    print(f"Unique Coffees: {len(unique)}")
    print("Coffee Names:", unique)

    # 3. Check for specific targets
    for target in ["HPLC_Caff", "HPLC_CGA", "TDS"]:
        subset = df_all.dropna(subset=[target])
        print(f"Valid {target}: {len(subset)} samples ({len(subset['Coffee Name'].unique())} coffees)")

if __name__ == "__main__":
    debug()
