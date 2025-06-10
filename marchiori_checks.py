import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mplt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
from scipy.integrate import simpson

from functools import partial

import tabulate
from pathlib import Path



PEAK_DETECTION_MIN = 1.2  # Minimum voltage for peak detection
PEAK_DETECTION_MAX = 1.35  # Maximum voltage for peak detection

CGA_MIN_VOLTAGE = 0.7
CGA_MAX_VOLTAGE = 0.75

CGA_NORMALIZE = False


def getData(sheet_id = '1Pa8iQ0_WjuVjassjfxEF_wE13O-19WQbBGbVffETHRA'):
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


    df = pd.read_csv(url)

    # keep columns
    df = df[['Name ', 'Brew date ', '(ppm) Caff Avg', '(ppm) CGA Avg', 'Voltammetry data 1', 'data 2', 'data 3']]

    # rename columns
    df.columns = ['Name', 'Brew date', 'HPLC_Caff', 'HPLC_CGA', 'cv_data1', 'cv_data2', 'cv_data3']

    # drop NA rows
    df = df.dropna()

    return df


def read_cv_data(filename, normalize = CGA_NORMALIZE, datadir = Path('voltammetry-files')):
    """ Reads a CV data file and returns a DataFrame with the data.
    """
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
        offset = df['i_ma'][CGA_MIN_VOLTAGE:CGA_MAX_VOLTAGE].mean()

        df['i_ma'] = df['i_ma'] - offset


    return df


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


def compute_caff_response(v, i, vmin=PEAK_DETECTION_MIN, vmax=PEAK_DETECTION_MAX):
    """ Computes the caffeine response from the CV data.
    The response is computed as the area under the curve of the current vs voltage plot.
    """

    # find bound indices for the voltage range

    start = np.where(v >= vmin)[0][0]  # Get the first index where v >= vmin
    end = np.where(v >= vmax)[0][0] # Get the first index where v >= vmax

    # Compute the area under the curve using the trapezoidal rule
    area = simpson(y=i[start:end], x=v[start:end])

    return area

if __name__ == "__main__":
    df = getData()
    df.sort_values(by='HPLC_Caff', inplace=True)
    df_train = df[df['Name'].str.contains('Alabaster Colombian Decaf') ]
    df_test = df[~df['Name'].str.contains('Alabaster Colombian Decaf')]
    df = None

    if 0:
        fig, ax = plt.subplots(1,1,figsize=(10, 6))

        data = []
        for name in ["Alabaster Colombian Decaf",
                    "Alabaster Colombian Decaf + 400 ppm Caf",
                    'Alabaster Colombian Decaf + 800 ppm Caf']:

            sample = df[ df['Name'] == name]

            for i in range(1, 4):
                cv = read_cv_data(sample[f'cv_data{i}'].iloc[0])

                data.append({
                    'Name': f'{name}',
                    'x' : cv.index,
                    'i_ma': cv['i_ma'].to_numpy()
                })

        cmap = mplt.colormaps['copper']
        num_lines = len(data)
        colors = [cmap(i / num_lines) for i in range(num_lines)]

        for i, cv in enumerate(data):
            ax.plot(cv['x'], cv['i_ma'], color = colors[i], label=cv['Name'], lw=2, alpha=0.7)

        plt.axvspan(PEAK_DETECTION_MIN, PEAK_DETECTION_MAX, color='green', alpha=0.3)

        # plt.title(title if title else 'CV Curve')
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current (uA)')
        plt.grid()
        plt.legend()


        plt.show()

        exit()

    def mk_response(row, col='cv_data1', normalize=CGA_NORMALIZE):
        cv = read_cv_data(row[col], normalize=normalize)
        return compute_caff_response(cv.index, cv['i_ma'].to_numpy())

    def mk_r2(row, col='cv_data1', normalize=CGA_NORMALIZE):
        cv = read_cv_data(row[col], normalize=normalize)
        from Regression import find_peak_response
        return find_peak_response(cv.index, cv['i_ma'].to_numpy())[1]   

    #df['SPE_area_method'] = df.apply(partial(mk_response, normalize=False), axis=1)
    #df['SPE_peak_method'] = df.apply(partial(mk_r2, normalize=False), axis=1)
    df_train['SPE_area_norm'] = df_train.apply(partial(mk_response, normalize=True), axis=1)
    #df['SPE_peak_norm'] = df.apply(partial(mk_r2, normalize=True), axis=1)

    results = []
    for i, row in df_train.iterrows():
        for k in range(1, 4):
            results.append(
                {
                    'Name': row['Name'],
                    'HPLC_Caff': row['HPLC_Caff'],
                    #'HPLC_CGA': row['HPLC_CGA'],
                    'SPE_area_norm': mk_response(row, col=f'cv_data{k}', normalize=True),                    
                }
            )

    df_train = pd.DataFrame(results)
    
    results = []
    for i, row in df_test.iterrows():
        for k in range(1, 4):
            results.append(
                {
                    'Name': row['Name'],
                    'HPLC_Caff': row['HPLC_Caff'],
                    #'HPLC_CGA': row['HPLC_CGA'],
                    'SPE_area_norm': mk_response(row, col=f'cv_data{k}', normalize=True),                    
                }
            )

    df_test = pd.DataFrame(results)
    

    def lr(x, y):

        """ Returns a linear regression model for the given x and y data.
        """
        model = LinearRegression()
        model.fit(x.values.reshape(-1, 1), y.values)       
        #pred_y = model.predict(x.values.reshape(-1, 1))

        # predict from zero to double max x
        x_range = np.linspace(0, x.max() * 2, num=100)
        pred_y = model.predict(x_range.reshape(-1, 1))
        return x_range, pred_y, r2_score(y, model.predict(x.values.reshape(-1,1))), model.intercept_, model.coef_[0]
           
    def add_regression(ax, x, y):
        """ Adds a linear regression line to the given axes.
        """
        pred_x, pred_y, r2, intercept, slope = lr(x, y)
        ax.plot(pred_x, pred_y, color='red', label='Linear Regression', lw=2)

        # slope from model
        ax.annotate(f'y = {slope:.6f}x + {intercept:.3f} (R² = {r2:.2f})',
                    xy=(0.05, 0.90), xycoords='axes fraction', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    pred_x, pred_y, r2, intercept, slope = lr(df_train['SPE_area_norm'], df_train['HPLC_Caff'   ])
    df_train['SPE_Caff'] = df_train['SPE_area_norm'] * slope + intercept
    df_train['SPE_Caff_err'] = df_train['SPE_Caff'] - df_train['HPLC_Caff']
    df_train['SPE_Caff_err_pct'] = (df_train['SPE_Caff'] - df_train['HPLC_Caff']) / df_train['HPLC_Caff'] * 100
    
    df_test['SPE_Caff'] = df_test['SPE_area_norm'] * slope + intercept
    df_test['SPE_Caff_err'] = df_test['SPE_Caff'] - df_test['HPLC_Caff']
    df_test['SPE_Caff_err_pct'] = (df_test['SPE_Caff'] - df_test['HPLC_Caff']) / df_test['HPLC_Caff'] * 100

    print(tabulate.tabulate(df_train, headers='keys', tablefmt='pipe', showindex=False, floatfmt=".3f",
                           numalign="right", stralign="left"))
    
    print(tabulate.tabulate(df_test, headers='keys', tablefmt='pipe', showindex=False, floatfmt=".3f",
                           numalign="right", stralign="left"))
    
    # print mean std min max for the SPE_Caff_err_pct column
    print(f"Train SPE Caff Error (%): mean={df_train['SPE_Caff_err_pct'].mean():.2f}, std={df_train['SPE_Caff_err_pct'].std():.2f}, min={df_train['SPE_Caff_err_pct'].min():.2f}, max={df_train['SPE_Caff_err_pct'].max():.2f}")
    print(f"Test SPE Caff Error (%): mean={df_test['SPE_Caff_err_pct'].mean():.2f}, std={df_test['SPE_Caff_err_pct'].std():.2f}, min={df_test['SPE_Caff_err_pct'].min():.2f}, max={df_test['SPE_Caff_err_pct'].max():.2f}")

    print(f"Train SPE Caff Error (ppm): mean={df_train['SPE_Caff_err'].mean():.2f}, std={df_train['SPE_Caff_err'].std():.2f}, min={df_train['SPE_Caff_err'].min():.2f}, max={df_train['SPE_Caff_err'].max():.2f}")
    print(f"Test SPE Caff Error (ppm): mean={df_test['SPE_Caff_err'].mean():.2f}, std={df_test['SPE_Caff_err'].std():.2f}, min={df_test['SPE_Caff_err'].min():.2f}, max={df_test['SPE_Caff_err'].max():.2f}")

    if 1:
        #fig, axes = plt.subplots(2,2,figsize=(12, 12))
        fig, axes = plt.subplots(1,1,figsize=(12, 5))
        def make_plot(ax, x, y, label):
            ax.scatter(x, y,  color='blue', label=label)
            add_regression(ax, x, y)

            ax.set_xlabel('SPE Caffeine Response (uA.V)')
            ax.set_ylabel('HPLC Caffeine (ppm)')
            ax.set_title(f'{label} vs HPLC Caffeine')
            ax.set_ylim(-500, 1400)
            ax.set_xlim(0, x.max() * 1.2)
            ax.grid()
            #ax.legend(loc='lower left')

        #make_plot(axes[0,0], df['SPE_area_method'], 'SPE (Integral Method)')
        #make_plot(axes[0,1], df['SPE_area_norm'], 'SPE (Integral Method Normalized)')
        make_plot(axes, df_train['SPE_area_norm'], df_train['HPLC_Caff'], 'SPE (training)')
        #make_plot(axes[1,0], df['SPE_peak_method'], 'SPE (Peak Method)')
        #make_plot(axes[1,1], df['SPE_peak_norm'], 'SPE (Peak Method Normalized)')

        #make_plot(axes[1], df_test['SPE_area_norm'], df_test['HPLC_Caff'], 'SPE (test)')


    if 0:
        fig, ax = plt.subplots(1, 1, figsize=(4, 5))
        ax.violinplot([df_train['SPE_Caff_err_pct'], df_test['SPE_Caff_err_pct']], showmeans=True, widths=0.5)
        #ax.set_xticklabels(['Train', 'Test'])
        ax.set_xticks([1, 2], labels=['Train', 'Test'])
        ax.set_title('SPE Caffeine vs HPLC Error')
        #ax.set_ylim(-250,250)
        ax.set_ylabel('Error Percentage (%)')
        

        plt.tight_layout()
        #plt.savefig('caff_response.pdf', dpi=300)
    plt.show()

    # df.drop(columns=['cv_data1', 'cv_data 2', 'cv_data 3'], inplace=True)

    # print(tabulate.tabulate(df, headers='keys', tablefmt='pipe', showindex=False, floatfmt=".2f",
    #                        numalign="right", stralign="left", disable_numparse=True))