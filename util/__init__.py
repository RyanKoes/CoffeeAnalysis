
from functools import cache
import pandas as pd
from pathlib import Path
from scipy.integrate import simpson
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Resolve project paths relative to this package, so scripts work
# regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATADIR = PROJECT_ROOT / 'data'
PLOTDIR = PROJECT_ROOT / 'normalized_plots'
VOLTAMMETRY_DIR = PROJECT_ROOT / 'voltammetry-files'
EXTRA_VOLTAMMETRY_DIR = PROJECT_ROOT / 'extra-voltammetry-files'

def setup_mplt():
    # increase matplotlib font size
    plt.rcParams.update({'font.size': 12})
    mplt.rcParams['axes.titlesize'] = 14
    mplt.rcParams['axes.labelsize'] = 14
    mplt.rcParams['xtick.labelsize'] = 12
    mplt.rcParams['ytick.labelsize'] = 12
    mplt.rcParams['legend.fontsize'] = 12
    mplt.rcParams['figure.titlesize'] = 16
    mplt.rcParams['figure.figsize'] = (10, 6)
    mplt.rcParams['figure.dpi'] = 100
    mplt.rcParams['savefig.dpi'] = 300
    mplt.rcParams['savefig.bbox'] = 'tight'
    mplt.rcParams['savefig.transparent'] = True
    mplt.rcParams['savefig.format'] = 'pdf'


@cache
def read_coffehub(
    sheet_id='1Pa8iQ0_WjuVjassjfxEF_wE13O-19WQbBGbVffETHRA',
    use_cache=True,
    require_columns=None,
):
    """ Returns:
                                                    Name Brew date  HPLC_Caff  HPLC_CGA     cv_data1    cv_data 2        cv_data 3
    0                           Alabaster Colombian Decaf   5/17/25       52.0     920.0  aladec1.txt  aladec2.txt  aladec3edge.txt
    1               Alabaster Colombian 1/4 Reg 3/4 Decaf   5/17/25      260.0     930.0  ala1QC1.txt  ala1QC2.txt      ala1QC3.txt
    2               Alabaster Colombian 1/2 Reg 1/2 Decaf   5/17/25      380.0     900.0   alaHC1.txt   alaHC2.txt       alaHC3.txt
    3               Alabaster Colombian 3/4 Reg 1/4 Decaf   5/17/25      650.0    1000.0  ala3QC1.txt  ala3QC2.txt      ala3QC3.txt
    4                         Alabaster Colombian Regular   5/17/25      820.0     990.0  alareg1.txt  alareg2.txt  alareg3edge.txt
    5             Alabaster Colombian Decaf + 200 ppm Caf   5/17/25      320.0     900.0  alaD2p1.txt  alaD2p2.txt      alaD2p3.txt
    6             Alabaster Colombian Decaf + 400 ppm Caf   5/17/25      580.0     920.0  alaD4p1.txt  alaD4p2.txt      alaD4p3.txt
    7             Alabaster Colombian Decaf + 600 ppm Caf   5/17/25      720.0     890.0  alaD6p1.txt  alaD6p2.txt      alaD6p3.txt
    8             Alabaster Colombian Decaf + 800 ppm Caf   5/17/25      920.0     880.0  alaD8p1.txt  alaD8p2.txt      alaD8p3.txt
    9                   FRC Decaf Colombian, med roast IH   5/16/25       75.0     600.0       A1.txt       A2.txt       A3edge.txt
    10        FRC Sugarcame Decaf Colombian, med roast IH   5/16/25       60.0     520.0       B1.txt       B2.txt       B3edge.txt
    11      FRC Swiss Water Decaf Colombian, med roast IH   5/16/25       40.0     420.0       C1.txt       C2.txt       C3edge.txt
    12                           FRC Mexican medium roast   5/16/25      820.0     490.0       D1.txt       D2.txt           D3.txt
    13                           FRC Sumatra medium roast   5/16/25      830.0     360.0       E1.txt       E2.txt           E3.txt
    14                          FRC Colombia medium roast   5/16/25      770.0     400.0       F1.txt       F2.txt           F3.txt
    15                      FRC Kenya AA, medium roast IH   5/17/25      840.0     830.0       G1.txt       G2.txt           G3.txt
    16           FRC Ethiopia Yirgacheffe, light roast IH   5/17/25      770.0    1100.0       H1.txt       H2.txt           H3.txt
    17                FRC ROBUSTA Brazil, medium roast IH   5/17/25     1220.0     530.0       I1.txt       I2.txt           I3.txt
    18                 FRC Brazil Cerrado, light roast IH   5/17/25      870.0     880.0       J1.txt       J2.txt           J3.txt
    19                FRC Brazil Cerrado, medium roast IH   5/17/25      850.0     440.0       K1.txt       K2.txt           K3.txt
    20                  FRC Brazil Cerrado, dark roast IH   5/17/25      860.0      94.0       L1.txt       L2.txt           L3.txt
    21       FRC Brazil Cerrado, medium roast IH- High BR   5/17/25     1100.0     630.0       M1.txt       M2.txt           M3.txt
    22  FRC Ethiopia Yirgacheffe, light roast IH- High BR   5/17/25     1060.0    1580.0       N1.txt       N2.txt           N3.txt
    23  FRC Brazil Cerrado, medium roast IH- High BR, ...   5/17/25      640.0     300.0       O1.txt       O2.txt           O3.txt
    24  FRC Ethiopia Yirgacheffe, light roast IH- High...   5/17/25      570.0     800.0       P1.txt       P2.txt           P3.txt
    25                    Dunkin Original Blend, 5/22 8am   5/22/25      530.0     360.0  dunkin1.txt  dunkin2.txt      dunkin3.txt
    26                McDonald's Regular Coffee, 5/22 8am   5/22/25      520.0     340.0  mccafe1.txt  mccafe2.txt      mccafe3.txt
    27  Sheetz Classic Coffee (100% arabica), 12oz ref...   5/22/25      420.0     270.0  sheetz1.txt  sheetz2.txt      sheetz3.txt
    28               Starbucks Pike Place Roast, 5/22 8am   5/22/25      680.0     130.0  sbpike1.txt  sbpike2.txt      sbpike3.txt
    """
    # Construct the URL to export the *first* sheet as CSV.
    url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv'

    cache_path = DATADIR / 'raw_data_cache.pkl'
    attribute_cols = [
        'Brightness',
        'Flavor',
        'Body',
        'Finish',
        'Sweetness',
        'Clean Cup',
        'Complexity',
        'Uniformity',
        'Fragrance',
        'Wet Aroma',
    ]

    # Optional per-flavor descriptor columns (0–5 scale) used by
    # nn_flavors_model_window_search.py. We keep them when present.
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

    if cache_path.exists() and use_cache:
        # load from cache
        try:
            df_cached = pd.read_pickle(cache_path)
            # If cache already contains required columns, use it.
            # We relax the check for attribute_cols to allow using caches that might miss them
            # (like the 44-coffee dataset from Regression Modeling)
            required_cols = [] 
            if require_columns:
                required_cols.extend(list(require_columns))

            print(f"DEBUG: Checking cache. required: {required_cols}")
            
            missing_cols = [c for c in required_cols if c not in df_cached.columns]
            if not missing_cols:
                print("DEBUG: Using cache.")
                return df_cached
            else:
                print(f"DEBUG: Cache missing columns: {missing_cols}")
        except Exception as e:
            print(f"DEBUG: Failed to read cache: {e}")
        # Otherwise, fall through to refresh from Google Sheets.

    #raise("Not using cache!")

    print("Reading CoffeeHub data from Google Sheets...")
    print(f"URL: {url}")

    df = pd.read_csv(url)

    # Normalize header whitespace from Google Sheets exports
    df.columns = [str(c).strip() for c in df.columns]

    # Stripping can create duplicate column names (e.g., 'Body ' and 'Body').
    # Keep the first occurrence to ensure downstream indexing returns scalars.
    df = df.loc[:, ~df.columns.duplicated()]

    # COLUMNS
    # Index(['ID #', 'Name ', 'Brand ', 'Type ', 'Roast ', 'Roa date ', 'Brew date ',
    #    'Reffractometer TDS 1', 'TDS 2', 'TDS 3', 'TDS 4', 'TDS 5',
    #    '(%) TDS Avg', 'St Dev', 'HPLC Caffeine ppm 1', 'Caffeine ppm 2',
    #    '(ppm) Caff Avg', 'St Dev.1', 'CGA ppm (799/1/200 eluent new)',
    #    'CGA ppm 2', '(ppm) CGA Avg', 'St Dev.2', 'Voltammetry data 1',
    #    'data 2', 'data 3', 'Caffeine ppm 1', 'Caffeine ppm 2.1',
    #    'Caffeine ppm 3', 'Caff Avg', 'St Dev.3', 'CGA ppm 1', 'CGA ppm 2.1',
    #    'CGA ppm 3', 'CGA Avg', 'St Dev.4'],

    # print (df)
    # print (df.columns)
    #exit()
    # Keep core measurement columns and, when present, cupping attribute + flavor columns.
    base_cols = [
        'Name',
        'Brew date',
        'Roast',
        'Reffractometer TDS 1',
        'TDS 2',
        'TDS 3',
        'HPLC Caffeine ppm 1',
        'Caffeine ppm 2',
        'CGA ppm (799/1/200 eluent new)',
        'Voltammetry data 1',
        'data 2',
        'data 3',
    ]

    # Build keep_cols without duplicates while preserving order.
    keep_cols_raw = [c for c in (base_cols + attribute_cols + flavor_cols) if c in df.columns]
    keep_cols = []
    seen = set()
    for c in keep_cols_raw:
        if c in seen:
            continue
        keep_cols.append(c)
        seen.add(c)
    df = df[keep_cols]

    # rename columns to canonical internal names
    rename_map = {
        'Reffractometer TDS 1': 'TDS_1',
        'TDS 2': 'TDS_2',
        'TDS 3': 'TDS_3',
        'HPLC Caffeine ppm 1': 'HPLC_Caff_1',
        'Caffeine ppm 2': 'HPLC_Caff_2',
        'CGA ppm (799/1/200 eluent new)': 'HPLC_CGA',
        'Voltammetry data 1': 'cv_data1',
        'data 2': 'cv_data2',
        'data 3': 'cv_data3',
    }
    df = df.rename(columns=rename_map)


    #print(df [df['Name'] == 'FRC Swiss Water Decaf Colombian, med roast IH'])
    #exit()

    # Drop NA rows required for modeling (do not require attribute columns)
    # Modified to only require the first replicate, allowing for partial data
    required = [
        'Name',
        'Brew date',
        #'Roast',
        #'TDS_1',
        #'HPLC_Caff_1',
        #'HPLC_CGA',
        'cv_data1',
    ]
    required = [c for c in required if c in df.columns]
    df = df.dropna(subset=required)

    df['HPLC_Caff'] = df[['HPLC_Caff_1', 'HPLC_Caff_2']].mean(axis=1)
    df['HPLC_Caff_3'] = df['HPLC_Caff']
    df['HPLC_CGA_1'] = df['HPLC_CGA']
    df['HPLC_CGA_2'] = df['HPLC_CGA']
    df['HPLC_CGA_3'] = df['HPLC_CGA']

    df['TDS'] = df[['TDS_1', 'TDS_2', 'TDS_3']].mean(axis=1)

    if use_cache:
        df.to_pickle(cache_path)

    return df

@cache
def read_cv_data(filename, normalize = None, datadir = None):
    """ Reads a CV data file and returns a DataFrame with the data.
    """

    if normalize is None:
        raise ValueError("normalize must be True or False")
    if datadir is None:
        datadir = VOLTAMMETRY_DIR
    df = pd.read_csv(datadir / filename, sep=',', header=None, names=['t', 'v', 'i'], index_col='t')


    # 45 seconds are preconditioning
    # then it takes a second for the current to stabilize
    # ignore first 46 seconds as preconditioning
    df = df.loc[46:]

    #find max applied voltage
    max_voltage = df['v'].max()

    # remove reduction part of the curve
    # find index of max
    max_index = df['v'].idxmax()
    # keep only data before max
    df = df.loc[:max_index]

    df['v_ma'] = df['v'].rolling(window=10, center=True).mean()
    df['i_ma'] = df['i'].rolling(window=10, center=True).mean()

    # remove rows where v_avg is NaN
    df = df.dropna(subset=['v_ma', 'i_ma'])

    # reindex to voltage
    df = df.set_index('v_ma')
    df.drop(columns=['v', 'i'], inplace=True)

    # normalize if needed
    if normalize:
        #print("Normalizing CV data...")

        area = simpson(y=df['i_ma'], x=df.index)

        #offset = df['i_ma'][NORMALIZE_MIN_VOLTAGE:NORMALIZE_MAX_VOLTAGE].mean()
        #df['i_ma'] = df['i_ma'] - offset

        df['i_ma'] = df['i_ma'] / area

    return df

@cache
def read_cv_data_bins(filename, normalize = None,
                      redox = False,
                      datadir = None,
                      num_bins=16):
    """ Reads a CV data file and returns a DataFrame with the data.
    
    if redox is true the entire curve is returned for one cycle.
    """

    if normalize is None:
        raise ValueError("normalize must be True or False")
    if datadir is None:
        datadir = VOLTAMMETRY_DIR
    df = pd.read_csv(datadir / filename, sep=',', header=None, names=['t', 'v', 'i'], index_col='t')

    # 45 seconds are preconditioning
    # then it takes a second for the current to stabilize
    # ignore first 46 seconds as preconditioning
    df = df.loc[46:]

    #find max applied voltage
    #max_voltage = df['v'].max()

    # remove reduction part of the curve
    # find index of max
    max_index = df['v'].idxmax()

    if not redox:
        # keep only data before max
        df = df.loc[:max_index]

    # remove rows where v is NaN
    df = df.dropna(subset=['v', 'i'])



    # create bins
    #bins = np.linspace(0, 2.0, num=num_bins + 1)
    bins = np.linspace(df.index.min(), df.index.max(), num=num_bins + 1)


    #print (f"Creating {num_bins} bins for voltage ranges: {bins}")

    times = []
    areas = []
    for i in range(num_bins):
        #print(f"{i:2d} {bins[i]:5.4f} : {bins[i+1]:5.4f}")
        #x_vals.append((bins[i] + bins[i+1]) / 2)
        #y_vals.append (df[bins[i]:bins[i+1]]['i'].mean())
        y = df[bins[i]:bins[i+1]]['i']
        x = df[bins[i]:bins[i+1]].index

        times.append(x.values.mean())
        if len(x) == 0:
            areas.append(0.0)
        else:
            areas.append(simpson(y=y, x=x))


    #df = pd.DataFrame(y_vals, index = x_vals, columns=['i'])
    s = pd.Series(areas, index= times, name='iv')

    
    if normalize:
        s /= sum(areas)

    return s, df['i']

# moved from NN to here for reorg purposes
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
            try:
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
            except Exception as e:
                pass
                #print(f"Skipping {row['Name']} ({k}) due to error processing {row.get(f'cv_data{k}')}: {e}")

    df = pd.DataFrame(results)

    df.sort_values(by='HPLC_Caff', inplace=True)

    if test_train_split:

        train = df['Coffee Name'].isin(train)
        return  df[train], df[~train]

    else:
        return df


if __name__ == "__main__":
    setup_mplt()
    bins, raw = read_cv_data_bins('A1.txt', normalize=False, redox=False, num_bins=64)

    #print(df)

    fig,ax= plt.subplots(1, 1, figsize=(6, 4))
    #ax.plot(ser)
    ax2 = ax.twinx()
    ax.plot(raw.index, raw, color = 'red', label='Raw Current')
    ax2.bar(bins.index, bins, width = (bins.index[1] - bins.index[0])/1.2, alpha = 0.5, label='Binned Response')
    ax.legend()
    plt.tight_layout()
    plt.show(   )
