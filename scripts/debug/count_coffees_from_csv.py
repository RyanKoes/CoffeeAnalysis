
import pandas as pd
from pathlib import Path

csv_path = Path("data/full_window_search_results_HPLC_Caff.csv")
if csv_path.exists():
    try:
        df = pd.read_csv(csv_path)
        print(f"Total rows: {len(df)}")
        if 'test_coffee' in df.columns:
            unique_coffees = df['test_coffee'].unique()
            print(f"Unique Coffees: {len(unique_coffees)}")
            print(unique_coffees)
        else:
            print("Column 'test_coffee' not found.")
    except Exception as e:
        print(f"Error reading csv: {e}")
else:
    print(f"{csv_path} does not exist.")
