import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mplt

import numpy as np
from scipy.integrate import simps


import tabulate
from pathlib import Path



PEAK_DETECTION_MIN = 1.15  # Minimum voltage for peak detection
PEAK_DETECTION_MAX = 1.5  # Maximum voltage for peak detection


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


def read_cv_data(filename, datadir = Path('voltammetry-files')):
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
    area = simps(i[start:end], v[start:end])

    return area



if __name__ == "__main__":

    #this did not work for me.
    #plt.rcParams['text.usetex'] = True
    #plt.rcParams['font.family'] = 'serif'
    #plt.rcParams['font.serif'] = ['Computer Modern Roman'] # To use the classic LaTeX font

    df = getData()

    df.sort_values(by='HPLC_Caff', inplace=True)

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

    def mk_response(row):
        cv = read_cv_data(row['cv_data1'])
        return compute_caff_response(cv.index, cv['i_ma'].to_numpy())

    def mk_r2(row):
        cv = read_cv_data(row['cv_data1'])
        from Regression import find_peak_response
        return find_peak_response(cv.index, cv['i_ma'].to_numpy())[1]


    df['SPE_area_method'] = df.apply(mk_response, axis=1)
    df['SPE_peak_method'] = df.apply(mk_r2, axis=1)





    if 0:

        fig, axes = plt.subplots(2,1,figsize=(8, 6))

        ax = axes[0]
        ax.scatter(df['HPLC_Caff'], df['SPE_area_method'], color='blue', label='SPE Response')
        ax.set_xlabel('HPLC Caffeine (ppm)')
        ax.set_ylabel('SPE Caffeine Response (uA.V)')
        ax.set_title('HPLC Caffeine vs SPE Caffeine (Integral Method)')
        ax.grid()
        ax.legend()

        ax = axes[1]
        ax.scatter(df['HPLC_Caff'], df['SPE_peak_method'], color='brown', label='SPE Response')
        ax.set_xlabel('HPLC Caffeine (ppm)')
        ax.set_ylabel('SPE Caffeine Response (uA)')
        ax.set_title('HPLC Caffeine vs SPE Caffeine (Peak Method)')
        ax.grid()
        ax.legend()

        plt.tight_layout()
        plt.savefig('caff_response.pdf', dpi=300)
    plt.show()

    df.drop(columns=['cv_data1', 'cv_data 2', 'cv_data 3'], inplace=True)

    print(tabulate.tabulate(df, headers='keys', tablefmt='pipe', showindex=False, floatfmt=".2f",
                           numalign="right", stralign="left", disable_numparse=True))