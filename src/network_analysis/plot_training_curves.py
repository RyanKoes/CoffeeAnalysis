
# --- repo-root bootstrap (added by reorg) ---
import sys as _sys
from pathlib import Path as _Path
_repo_root = _Path(__file__).resolve().parents[2]
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))
# --- end bootstrap ---

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
