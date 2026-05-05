
import pandas as pd
from pathlib import Path

p = Path("src/regression_modeling/data/raw_data_cache.pkl")  
df = pd.read_pickle(p)
cols = [
    "Brightness",
        "Flavor",
        "Body",
        "Finish",
        "Sweetness",
        "Clean Cup",
        "Complexity",
        "Uniformity",
        "Fragrance",
        "Wet Aroma",
]
found = [c for c in cols if c in df.columns]
print(f"Found {len(found)}/{len(cols)} attribute columns.")
missing = [c for c in cols if c not in df.columns]
print(f"Missing: {missing}")
