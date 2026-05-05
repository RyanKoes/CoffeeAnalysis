
from nn_0_synthetic_data_gen import build_model_data
import pandas as pd

df = build_model_data(
    NORMALIZE=False,
    REDOX=False,
    USE_BINS=False,
    test_train_split=False,
)

print(f"Total samples: {len(df)}")
print(f"Unique keys: {df.index.unique()}") # Assuming index might not be unique in some way or checking structure
if 'Coffee Name' in df.columns:
    unique_coffees = df['Coffee Name'].unique()
    print(f"Unique Coffees ({len(unique_coffees)}): {unique_coffees}")
else:
    print("Column 'Coffee Name' not found.")

targets = ['HPLC_Caff', 'HPLC_CGA', 'TDS']
for t in targets:
    if t in df.columns:
        valid_count = df[t].notna().sum()
        total_count = len(df)
        print(f"Target '{t}': {valid_count}/{total_count} non-NaN values.")
        
        # Check unique coffees that have valid values for this target
        df_valid = df.dropna(subset=[t])
        valid_coffees = df_valid['Coffee Name'].unique()
        print(f"  Unique Coffees with valid '{t}': ({len(valid_coffees)})")
    else:
        print(f"Target '{t}' not in dataframe.")
