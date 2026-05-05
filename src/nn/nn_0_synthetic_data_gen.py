import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mplt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
from scipy.integrate import simpson


import tabulate
from pathlib import Path


from itertools import combinations

from util import read_coffehub, read_cv_data, read_cv_data_bins, setup_mplt, DATADIR



def plot_cv_curve(df, ax, label=None):
    """ Plots the CV curve from a DataFrame.
    """

    #plt.figure(figsize=(10, 6))

    ax.plot(df.index, df['i_ma'],
             label=label,  lw=2, alpha=0.7)

    # plt.axvspan(PEAK_DETECTION_MIN, PEAK_DETECTION_MAX, color='green', alpha=0.3)

    # plt.title(title if title else 'CV Curve')
    # plt.xlabel('Voltage (V)')
    # plt.ylabel('Current (uA)')
    # plt.grid()
    # plt.legend()


def build_model_data(NORMALIZE, BINS=64, REDOX=False , test_train_split = True, **kwargs):

    require_columns = kwargs.pop('require_columns', None)
    if require_columns is not None:
        # read_coffehub is memoized; ensure hashable for caching
        require_columns = tuple(require_columns)

    def _parse_roast_value(roast_value):
        """Parse the CoffeeHub 'Roast' field into a float.

        In the CoffeeHub sheet this is typically a percent-like string
        (e.g. '13.80%' or '14.8%'). We store the numeric value (e.g. 13.8).
        """
        if roast_value is None or (isinstance(roast_value, float) and np.isnan(roast_value)):
            return np.nan
        if isinstance(roast_value, (int, float, np.number)):
            return float(roast_value)
        roast_str = str(roast_value).strip()
        if roast_str.endswith('%'):
            roast_str = roast_str[:-1].strip()
        if roast_str == "":
            return np.nan
        return float(roast_str)

    def _parse_optional_float(value):
        """Best-effort float parsing.

        Handles duplicate column labels in CoffeeHub exports where
        `row.get(label)` may return a Series. In that case, returns the first
        non-NaN value that can be parsed as float.
        """
        if value is None:
            return np.nan
        if isinstance(value, float) and np.isnan(value):
            return np.nan

        # If the sheet contains duplicate column labels, pandas Series.get()
        # can return a Series instead of a scalar.
        if isinstance(value, pd.Series):
            for v in value.values.tolist():
                parsed = _parse_optional_float(v)
                if not (isinstance(parsed, float) and np.isnan(parsed)):
                    return parsed
            return np.nan

        if isinstance(value, (list, tuple, np.ndarray)):
            for v in list(value):
                parsed = _parse_optional_float(v)
                if not (isinstance(parsed, float) and np.isnan(parsed)):
                    return parsed
            return np.nan

        if isinstance(value, str):
            s = value.strip()
            if s == "" or s.lower() in {"na", "nan", "none", "null"}:
                return np.nan
            try:
                return float(s)
            except ValueError:
                return np.nan

        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    train = """Alabaster Colombian Decaf
Alabaster Colombian Decaf + 200 ppm Caf
Alabaster Colombian Decaf + 400 ppm Caf
Alabaster Colombian Decaf + 600 ppm Caf
Alabaster Colombian Decaf + 800 ppm Caf
FRC Decaf Colombian, med roast IH
FRC Swiss Water Decaf Colombian, med roast IH
FRC Sumatra medium roast
FRC Kenya AA, medium roast IH
FRC ROBUSTA Brazil, medium roast IH
FRC Brazil Cerrado, medium roast IH
FRC Brazil Cerrado, medium roast IH- High BR
FRC Brazil Cerrado, medium roast IH- High BR, 2x dilute""".split('\n')

    df = read_coffehub(require_columns=require_columns)

    attribute_cols = [
        "Brightness",
        "Flavor",
        "Body",
        "Finish",
        "Sweetness",
        "Clean Cup",
        "Complexity",
        "Uniformity",
        "Fragrance",
        "Wet Aroma",
    ]

    flavor_cols = [
        'Spice',
        'Body',
        'Floral',
        'Honey',
        'Sugars',
        'Caramel',
        'Fruits',
        'Citrus',
        'Berry',
        'Cocoa',
        'Nuts',
        'Rustic',
    ]

    results = []
    for i, row in df.iterrows():
        roast_level = _parse_roast_value(row.get('Roast'))

        # Pull attributes + flavor descriptor scores if present (store as float where possible)
        attrs = {}
        for col in dict.fromkeys(attribute_cols + flavor_cols):
            if col in row.index:
                attrs[col] = _parse_optional_float(row.get(col))

        for k in range(1, 4):
            bins, raw = read_cv_data_bins(row[f'cv_data{k}'],
                                                 redox= REDOX,
                                                 normalize=NORMALIZE,
                                                 num_bins=BINS)
            results.append(
                {
                    'Sample Name': f"{row['Name']} ({k})",
                    'Coffee Name': row['Name'],
                    'Roast': roast_level,
                    'HPLC_Caff': row[f'HPLC_Caff_{k}'],
                    'HPLC_CGA': row[f'HPLC_CGA_{k}'],
                    'TDS': row[f'TDS_{k}'],
                    'cv_bins': bins,
                    'cv_raw': raw,
                    **attrs,
                }
            )

    df = pd.DataFrame(results)

    df.sort_values(by='HPLC_Caff', inplace=True)

    if test_train_split:

        train = df['Coffee Name'].isin(train)
        return  df[train], df[~train]

    else:
        return df


def combine_samples(samples, weights):

    #print("Combining samples:")
    #print(tabulate.tabulate(samples, headers='keys', tablefmt='psql'))

    # +----+---------------------------------------------------+-----------------------------------------------+-------------+------------+-------+--------------------------------------+
    # |    | Sample Name                                       | Coffee Name                                   |   HPLC_Caff |   HPLC_CGA |   TDS | cv_bins                              |
    # |----+---------------------------------------------------+-----------------------------------------------+-------------+------------+-------+--------------------------------------|
    assert 1-sum(weights) < 1e-6, "Weights must sum to 1"
    assert all([w >= 0 for w in weights]), f"Weights must be non-negative: {weights}"
    # combthese columns using the given weights
    newrow = {
        'Sample Name': ' + '.join(samples['Sample Name']),
        'Coffee Name': ' + '.join(samples['Coffee Name']),
        'HPLC_Caff': np.average(samples['HPLC_Caff'], weights=weights),
        'HPLC_CGA': np.average(samples['HPLC_CGA'], weights=weights),
        'TDS': np.average(samples['TDS'], weights=weights),
        'cv_bins': np.average(
            [s['cv_bins'] for _, s in samples.iterrows()],
            axis=0, weights=weights),
        'cv_raw': np.average(
            [s['cv_raw'] for _, s in samples.iterrows()],
            axis=0, weights=weights)
    }

    return newrow


def generate_combined_data(df_train):
    # get unique names
    names = df_train['Sample Name'].unique()

    print(f"Creating data from {len(names)} unique samples")

    data = []

    # add original samples
    for _, row in df_train.iterrows():
        data.append(row.to_dict())

    # combine pairs of samples
    for x in combinations(names, 2):
        for weights in np.linspace(0, 1, 12)[1:-1]:  # skip 0 and 1

            newrow = combine_samples (
                df_train[ df_train['Sample Name'].isin(x)],
                weights=(weights, 1-weights))
            data.append(newrow)

    # combine triplets of samples
    # for x in combinations(names, 3):
    #     for w1 in np.linspace(0, 1, 11)[1:-1]:
    #         for w2 in np.linspace(0, 1, 11)[1:-1]:
    #             w3 = 1 - w1 - w2
    #             weights = (w1, w2, w3)
    #             if any (w <= 0 for w in weights):
    #                 continue
    #             if sum(weights) > 1:
    #                 continue

    #             newrow = combine_samples (
    #                 df_train[ df_train['Sample Name'].isin(x)],
    #                 weights=(w1, w2, w3))
    #             data.append(newrow)

    return pd.DataFrame(data)

if __name__ == "__main__":
    setup_mplt()
    bins = 64
    #bins = 'raw'
    #df_train, df_test = build_model_data(NORMALIZE=True, BINS=16)
    #df_train, df_test = build_model_data(NORMALIZE=True, BINS=32)
    #df_train, df_test = build_model_data(NORMALIZE=True, BINS=64)
    df_train, df_test = build_model_data(NORMALIZE=True, REDOX=False, BINS=bins)

    if 0:
        fig, ax = plt.subplots(1,1,figsize=(6, 4))

        for i, name in enumerate(['Alabaster Colombian Decaf (1)', 'FRC Sumatra medium roast (1)']):
            pdf =  df_train[df_train['Sample Name'] == name].iloc[0]['cv_bins']
            x = np.arange(len(pdf))
            ax.bar(
                x + i*1/2,
                pdf,
                width=1/2, alpha=0.7, label=name)
        ax.legend()
        plt.tight_layout()
        plt.show()
        exit()


    test_name = f'2comb-10_bin{bins}'


    # write test data
    df_test.to_pickle(DATADIR / f'test_{bins}.pkl')

    print(f"Test data saved to {DATADIR / f'test_{bins}.pkl'}")


    df_combined = generate_combined_data(df_train)

    print (df_combined.describe())

    #print(f"Combined data has {len(df_combined)} rows")
    #print(tabulate.tabulate(df_combined, headers='keys', tablefmt='psql'))
    # save to csv
    df_combined.to_pickle(DATADIR / f'train_{test_name}.pkl')

    print(f"Created data saved to {DATADIR / f'train_{test_name}.pkl'}")