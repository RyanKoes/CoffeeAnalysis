
import pandas as pd
from pathlib import Path

p = Path("data/raw_data_cache.pkl")  
df = pd.read_pickle(p)
print(f"Total rows: {len(df)}")
if 'Roast' in df.columns:
    print(f"Roast column found.")
    print(f"Missing in Roast: {df['Roast'].isna().sum()}")
else:
    print("Roast column NOT found.")
