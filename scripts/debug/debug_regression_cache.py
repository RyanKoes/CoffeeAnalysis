
import pandas as pd
from pathlib import Path

p = Path("src/regression_modeling/data/raw_data_cache.pkl")  
df = pd.read_pickle(p)
print(f"Total rows: {len(df)}")
cols = [
    'Name',
    'Brew date',
    'Roast',
    'TDS_1',
    'HPLC_Caff_1',
    'HPLC_CGA',
    'cv_data1',
    'cv_data2', 
    'cv_data3'
]

for c in cols:
    if c in df.columns:
        missing = df[c].isna().sum()
        print(f"{c}: {missing} missing")
    else:
        print(f"{c}: not in columns")

required = [
        'Name',
        'Brew date',
        'Roast',
        'TDS_1',
        'HPLC_Caff_1',
        'HPLC_CGA',
        'cv_data1',
]
required = [c for c in required if c in df.columns]

df_dropped = df.dropna(subset=required)
print(f"After dropping required: {len(df_dropped)}")
