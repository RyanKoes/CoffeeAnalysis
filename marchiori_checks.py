import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
    df.columns = ['Name', 'Brew date', 'HPLC_Caff', 'HPLC_CGA', 'cv_data1', 'cv_data 2', 'cv_data 3']

    # drop NA rows
    df = df.dropna()

    return df


def read_cv_data(filename, datadir = Path('voltammetry-files')):
    """ Reads a CV data file and returns a DataFrame with the data.
    """
    df = pd.read_csv(datadir / filename, sep=',', header=None, names=['t', 'v', 'i'], index_col='t')

    print(f"Reading {filename} .")
    # ignore first 45 seconds as preconditioning
    df = df.loc[45:]

    #find max applied voltage
    max_voltage = df['v'].max()

    # remove reduction part of the curve
    # find index of max
    max_index = df['v'].idxmax()
    # keep only data before max
    df = df.loc[:max_index]

    return df


def plot_cv_curve(df, title=None):
    """ Plots the CV curve from a DataFrame.
    """
    

    plt.figure(figsize=(10, 6))
    plt.plot(df['v'], df['i'], label='CV Curve')
    plt.title(title if title else 'CV Curve')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    df = getData()

    df.sort_values(by='HPLC_Caff', inplace=True)

    

    cv = read_cv_data(df.iloc[0]['cv_data1'])






    print(cv)


    plot_cv_curve(cv, title=df.iloc[0]['Name'])


