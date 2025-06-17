import pandas as pd
import matplotlib as plt
from pathlib	import Path

from util import DATADIR




if __name__ == "__main__":
    resutls_path = DATADIR / 'results.pkl'
    df = pd.read_pickle(resutls_path)

    # Index(['fold', 'train_index', 'test_index', 'all_data_path', 'experiment_name',
    #    'model_path', 'train_HPLC_Caff_r2', 'train_HPLC_Caff_mae',
    #    'train_HPLC_Caff_predictions', 'train_HPLC_Caff_actual',
    #    'train_HPLC_Caff_error_pct', 'train_HPLC_CGA_r2', 'train_HPLC_CGA_mae',
    #    'train_HPLC_CGA_predictions', 'train_HPLC_CGA_actual',
    #    'train_HPLC_CGA_error_pct', 'train_TDS_r2', 'train_TDS_mae',
    #    'train_TDS_predictions', 'train_TDS_actual', 'train_TDS_error_pct',
    #    'test_HPLC_Caff_r2', 'test_HPLC_Caff_mae', 'test_HPLC_Caff_predictions',
    #    'test_HPLC_Caff_actual', 'test_HPLC_Caff_error_pct', 'test_HPLC_CGA_r2',
    #    'test_HPLC_CGA_mae', 'test_HPLC_CGA_predictions',
    #    'test_HPLC_CGA_actual', 'test_HPLC_CGA_error_pct', 'test_TDS_r2',
    #    'test_TDS_mae', 'test_TDS_predictions', 'test_TDS_actual',
    #    'test_TDS_error_pct', 'train_HPLC_Caff_error_pct_mean',
    #    'test_HPLC_Caff_error_pct_mean', 'train_HPLC_CGA_error_pct_mean',
    #    'test_HPLC_CGA_error_pct_mean', 'train_TDS_error_pct_mean',
    #    'test_TDS_error_pct_mean'],
    print(df.columns)