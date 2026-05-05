
import pandas as pd
from pathlib import Path

def check_cache(path):
    p = Path(path)
    if not p.exists():
        print(f"{path} does not exist.")
        return
    try:
        df = pd.read_pickle(p)
        print(f"Cache {path}: {len(df)} rows")
        if 'Coffee Name' in df.columns:
            print(f"Unique Coffees: {len(df['Coffee Name'].unique())}")
        elif 'Name' in df.columns:
             print(f"Unique Names: {len(df['Name'].unique())}")
        else:
             print("Create DataFrame but columns unknown:", df.columns)
    except Exception as e:
        print(f"Error reading {path}: {e}")

check_cache("src/regression_modeling/data/raw_data_cache.pkl")
check_cache("data/raw_data_cache.pkl")
