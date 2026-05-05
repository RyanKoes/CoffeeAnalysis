import pandas as pd
from util import DATADIR

df = pd.read_pickle(DATADIR / 'separate_model_results.pkl')
print("Columns:\n", df.columns)

print("\nExample row keys:")
for col in df.columns:
    first = df.iloc[0][col]
    if isinstance(first, list):
        print(f"{col}: list of length {len(first)}")
    else:
        print(f"{col}: {type(first)}")
