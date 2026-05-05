import torch
import sys
from pathlib import Path

# Need to find the file first.
path = Path("/Users/ryankoes/PycharmProjects/CoffeeAnalysis/src/thesis_plots/data/BEST_HPLC_Caff_fixed.pkl")
try:
    print(f"Loading {path}...")
    data = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    print(f"Keys: {data.keys()}")
    if 'metadata' in data:
        print(f"Metadata: {data['metadata']}")
    else:
        print("No metadata found.")

    if 'results' in data:
        print(f"Results type: {type(data['results'])}")
        if isinstance(data['results'], dict):
             print(f"Results keys: {data['results'].keys()}")
        elif isinstance(data['results'], list):
             print(f"Results (list) length: {len(data['results'])}")
             if len(data['results']) > 0:
                 print(f"First result item type: {type(data['results'][0])}")
                 if isinstance(data['results'][0], dict):
                     print(f"First result item keys: {data['results'][0].keys()}")
    else:
        print("No results found.")

except Exception as e:
    print(f"Error: {e}")
