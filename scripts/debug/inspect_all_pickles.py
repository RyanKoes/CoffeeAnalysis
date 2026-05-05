import torch
from pathlib import Path

files = [
    "src/thesis_plots/data/BEST_HPLC_Caff_fixed.pkl",
    "src/thesis_plots/data/BEST_HPLC_CGA_fixed.pkl",
    "src/thesis_plots/data/BEST_TDS_fixed.pkl"
]

for f in files:
    path = Path("/Users/ryankoes/PycharmProjects/CoffeeAnalysis") / f
    if not path.exists():
        print(f"File not found: {f}")
        continue
        
    try:
        print(f"Loading {path}...")
        data = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        print(f"Keys: {data.keys()}")
        if 'metadata' in data:
            print(f"Metadata: {data['metadata']}")
        
        if 'results' in data:
             print("Found results key!")
    except Exception as e:
        print(f"Error loading {f}: {e}")
    print("-" * 20)
