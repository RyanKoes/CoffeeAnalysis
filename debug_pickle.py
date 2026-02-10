import pandas as pd
import pickle
import sys
import traceback

file_path = 'data/SingleTarget-HPLC_Caff-OX-Bottleneck-512-64-1-3000_all.pkl'

print(f"Pandas version: {pd.__version__}")
print(f"Python version: {sys.version}")

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print("Success loading with pickle.load!")
    print(f"Type: {type(data)}")
    if isinstance(data, pd.DataFrame):
        print(data.info())
        print(data.head())
    elif isinstance(data, dict):
        print(f"Keys: {data.keys()}")
except Exception:
    print("Failed with pickle.load")
    traceback.print_exc()

print("-" * 20)

try:
    df = pd.read_pickle(file_path)
    print("Success loading with pd.read_pickle!")
    print(df.info())
except Exception:
    print("Failed with pd.read_pickle")
    traceback.print_exc()
