
import pandas as pd
from pathlib import Path

p = Path("data/raw_data_cache.pkl")
try:
    df = pd.read_pickle(p)
    print(f"Columns: {list(df.columns)}")
    attribute_cols = [
        "Brightness", "Flavor", "Body", "Finish", "Sweetness", "Clean Cup", 
        "Complexity", "Uniformity", "Fragrance", "Wet Aroma"
    ]
    missing = [c for c in attribute_cols if c not in df.columns]
    if missing:
        print(f"Missing attributes: {missing}")
    else:
        print("All attributes present.")
except Exception as e:
    print(f"Error: {e}")
