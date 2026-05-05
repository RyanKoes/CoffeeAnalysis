
import torch
import pandas as pd
from pathlib import Path

def inspect(path):
    print(f"--- Inspecting {path} ---")
    try:
        data = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        if isinstance(data, dict):
            print(f"Keys: {data.keys()}")
            if 'metadata' in data:
                print(f"Metadata: {data['metadata']}")
            if 'results' in data:
                res = data['results']
                print(f"Results type: {type(res)}")
                if isinstance(res, (list, pd.DataFrame)):
                    print(f"Length: {len(res)}")
                    
            # Check for potentially stored data in other keys?
            for k in data.keys():
                if k not in ['model_state_dict', 'metadata', 'results']:
                    print(f"Key '{k}': {type(data[k])}")
                    if isinstance(data[k], list):
                         print(f"  Length: {len(data[k])}")
    except Exception as e:
        # Pytorch load might fail if it's a pandas pickle
        try:
            df = pd.read_pickle(path)
            print(f"Pandas DataFrame/Object")
            if isinstance(df, pd.DataFrame):
                print(f"Shape: {df.shape}")
                print(f"Columns: {df.columns}")
                if 'Coffee Name' in df.columns:
                    print(f"Unique Coffees: {len(df['Coffee Name'].unique())}")
            elif isinstance(df, list):
                 print(f"List length: {len(df)}")
        except Exception as e2:
            print(f"Failed to load as torch or pandas: {e} / {e2}")

files = [
    "ThesisPlotGeneration/data/BEST_HPLC_Caff_fixed.pkl",
    "data/caff_Large-1024-512-256-1_V0.00-0.80_earlystop_loco_results.pkl",
    "data/full_window_search_results_HPLC_Caff.pkl"
]

for f in files:
    if Path(f).exists():
        inspect(f)
    else:
        print(f"File {f} not found.")
