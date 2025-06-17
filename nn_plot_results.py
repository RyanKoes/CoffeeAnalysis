import pandas as pd
import matplotlib as plt
from pathlib	import Path

from util import DATADIR




if __name__ == "__main__":
    resutls_path = DATADIR / 'results.pkl'
    df = pd.read_pickle(resutls_path)


