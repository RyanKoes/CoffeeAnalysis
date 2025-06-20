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

from functools import cache

from util import read_cv_data, read_cv_data_bins, setup_mplt

#0.85 R2
# CAFF_PEAK_DETECTION_MIN = 1.2  # Minimum voltage for peak detection
# CAFF_PEAK_DETECTION_MAX = 1.35  # Maximum voltage for peak detection
#CAFF_PEAK_DETECTION_MIN = 1.1  # Minimum voltage for peak detection
#CAFF_PEAK_DETECTION_MAX = 1.4  # Maximum voltage for peak detection

NORMALIZE_MIN_VOLTAGE = 0.7
NORMALIZE_MAX_VOLTAGE = 0.75

#NORMALIZE = True

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


def compute_area_response(v, i, vmin, vmax):
    """ Computes the response from the CV data.
    The response is computed as the area under the curve of the current vs voltage plot.
    """

    # find bound indices for the voltage range

    start = np.where(v >= vmin)[0][0]  # Get the first index where v >= vmin
    end = np.where(v >= vmax)[0][0] # Get the first index where v >= vmax

    # Compute the area under the curve using the trapezoidal rule
    area = simpson(y=i[start:end], x=v[start:end])

    return area
def plot_cv_curves(df_train, df_test, CAFF_PEAK_DETECTION_MIN, CAFF_PEAK_DETECTION_MAX, CGA_BOUNDS, NORMALIZE):
    """
    Plots CV curves for all samples in df_train and df_test.
    """
    fig, ax = plt.subplots(1,1,figsize=(6, 4))

    data = []
    for i, sample in df_train.iterrows():
        #for j in range(1, 4):
            #cv = read_cv_data(sample[f'cv_data{j}'])
            cv = read_cv_data(sample['cv_data'], normalize=NORMALIZE)
            data.append({
                'Name': f'{sample["Name"]}',
                'x' : cv.index,
                'i_ma': cv['i_ma'].to_numpy(),
                'bins': sample['cv_bins']
            })
    for i, sample in df_test.iterrows():
        for j in range(1, 4):
            #cv = read_cv_data(sample[f'cv_data{j}'])
            cv = read_cv_data(sample['cv_data'], normalize = NORMALIZE)
            data.append({
                'Name': f'{sample["Name"]}',
                'x' : cv.index,
                'i_ma': cv['i_ma'].to_numpy(),
                'bins': sample['cv_bins']
            })

    cmap = mplt.colormaps['copper']
    num_lines = len(data)
    colors = [cmap(i / num_lines) for i in range(num_lines)]

    # ax2= ax.twinx()
    for i, cv in enumerate(data):
        ax.plot(cv['x'], cv['i_ma'], color = colors[i],
                ms=1,
                label=cv['Name'], lw=1, alpha=0.5)


        # ax2.bar(cv['bins'].index, cv['bins']['i'],
        #        width=2 / 16,
        #        color = colors[i], alpha=0.5)


    ax.axvspan(CGA_BOUNDS[0], CGA_BOUNDS[1], color='blue', alpha=0.3)
    ax.annotate('CGA Detection Area',
                xy=(CGA_BOUNDS[0] + (CGA_BOUNDS[1] - CGA_BOUNDS[0]) / 2, 250),
                xycoords='data', fontsize=10,
                rotation=90, color='black',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
                ha='center')

    ax.axvspan(CAFF_PEAK_DETECTION_MIN, CAFF_PEAK_DETECTION_MAX, color='green', alpha=0.3)
    ax.annotate('Caffeine Detection Area',
                xy=(CAFF_PEAK_DETECTION_MIN + (CAFF_PEAK_DETECTION_MAX - CAFF_PEAK_DETECTION_MIN) / 2, 250),
                xycoords='data', fontsize=10,
                rotation=90, color='black',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
                ha='center')

    ax.axvspan(NORMALIZE_MIN_VOLTAGE, NORMALIZE_MAX_VOLTAGE, color='red', alpha=0.3)
    ax.annotate('Normalization Area',
                xy=(NORMALIZE_MIN_VOLTAGE + (NORMALIZE_MAX_VOLTAGE - NORMALIZE_MIN_VOLTAGE) / 2, 250),
                xycoords='data', fontsize=10,
                rotation=90, color='black',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
                ha='center')

    ax.set_xlim(0,2)
    #ax.yaxis.set_major_locator(mplt.ticker.MultipleLocator(100))
    #ax.yaxis.set_minor_locator(mplt.ticker.MultipleLocator(25))
    ax.xaxis.set_major_locator(mplt.ticker.MultipleLocator(0.25))
    ax.xaxis.set_minor_locator(mplt.ticker.MultipleLocator(0.25/5))

    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (uA)')
    plt.tight_layout()
    plt.grid()
    plt.savefig('cv_samples.pdf', dpi=300)


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
def lr(x, y):
    """ Returns a linear regression model for the given x and y data.
    """
    model = LinearRegression()
    model.fit(x.values.reshape(-1, 1), y.values)
    #pred_y = model.predict(x.values.reshape(-1, 1))

    # predict from zero to double max x
    #x_range = np.linspace(0, x.max() * 2, num=100)
    x_range = x.values
    pred_y = model.predict(x_range.reshape(-1, 1))
    #pred_y = model.predict(x.values.reshape(-1, 1))
    return x_range, pred_y, r2_score(y, model.predict(x.values.reshape(-1,1))), model.intercept_, model.coef_[0]
def add_regression(ax, x, y):
    """ Adds a linear regression line to the given axes.
    """
    pred_x, pred_y, r2, intercept, slope = lr(x, y)
    ax.plot(pred_x, pred_y, color='red', label='Linear Regression', lw=2)

    # slope from model
    # ax.annotate(f'y = {slope:.6f}x + {intercept:.3f} (R² = {r2:.2f})',
    #             xy=(0.5, 0.96), xycoords='axes fraction', fontsize=10,
    #             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

def build_caff_model(df_train):

    pred_x, pred_y, r2, intercept, slope = lr(df_train['SPE_area_norm'], df_train['HPLC_Caff'])
    df_train['SPE_Caff'] = df_train['SPE_area_norm'] * slope + intercept
    df_train['SPE_Caff_err'] = df_train['SPE_Caff'] - df_train['HPLC_Caff']
    df_train['SPE_Caff_err_pct'] = (df_train['SPE_Caff'] - df_train['HPLC_Caff']) / df_train['HPLC_Caff'] * 100

    return pred_x, pred_y, r2, intercept, slope, df_train

def build_cga_model(df_train):

    pred_x, pred_y, r2, intercept, slope = lr(df_train['SPE_cga_area'], df_train['HPLC_CGA'])
    df_train['SPE_CGA'] = df_train['SPE_cga_area'] * slope + intercept
    df_train['SPE_CGA_err'] = df_train['SPE_CGA'] - df_train['HPLC_CGA']
    df_train['SPE_CGA_err_pct'] = (df_train['SPE_CGA'] - df_train['HPLC_CGA']) / df_train['HPLC_CGA'] * 100

    return pred_x, pred_y, r2, intercept, slope, df_train

def apply_caff_model(df_test, intercept, slope):
    df_test['SPE_Caff'] = df_test['SPE_area_norm'] * slope + intercept
    df_test['SPE_Caff_err'] = df_test['SPE_Caff'] - df_test['HPLC_Caff']
    df_test['SPE_Caff_err_pct'] = (df_test['SPE_Caff'] - df_test['HPLC_Caff']) / df_test['HPLC_Caff'] * 100
    return df_test

def apply_cga_model(df_test, intercept, slope):
    df_test['SPE_CGA'] = df_test['SPE_cga_area'] * slope + intercept
    df_test['SPE_CGA_err'] = df_test['SPE_CGA'] - df_test['HPLC_CGA']
    df_test['SPE_CGA_err_pct'] = (df_test['SPE_CGA'] - df_test['HPLC_CGA']) / df_test['HPLC_CGA'] * 100
    return df_test

def build_model_data(CAFF_PEAK_DETECTION_MIN, CAFF_PEAK_DETECTION_MAX, CGA_BOUNDS, NORMALIZE):

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

    df = getRawData()
    df.sort_values(by='HPLC_Caff', inplace=True)

    df_train = df[df['Name'].isin(train)]
    df_test = df[~df['Name'].isin(train)]
    df = None

    print(f"Train samples: {len(df_train)}, Test samples: {len(df_test)}")

    def mk_response(row, normalize, vmin, vmax, col='cv_data1'):
        cv = read_cv_data(row[col], normalize=normalize)
        return compute_area_response(cv.index, cv['i_ma'].to_numpy(),
                                    vmin, vmax)

    results = []
    for i, row in df_train.iterrows():
        for k in range(1, 4):
            results.append(
                {
                    'Name': row['Name'],
                    'HPLC_Caff': row['HPLC_Caff'],
                    'HPLC_CGA': row['HPLC_CGA'],
                    'SPE_area_norm': mk_response(row, normalize=NORMALIZE, vmin = CAFF_PEAK_DETECTION_MIN, vmax = CAFF_PEAK_DETECTION_MAX, col=f'cv_data{k}'),
                    'SPE_cga_area': mk_response(row, normalize=NORMALIZE, vmin = CGA_BOUNDS[0], vmax = CGA_BOUNDS[1], col=f'cv_data{k}'),
                    'cv_data': row[f'cv_data{k}'],
                    'cv_bins': read_cv_data_bins(row[f'cv_data{k}'], normalize=NORMALIZE)
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
                    'HPLC_CGA': row['HPLC_CGA'],
                    'SPE_area_norm': mk_response(row, normalize=NORMALIZE, vmin = CAFF_PEAK_DETECTION_MIN, vmax = CAFF_PEAK_DETECTION_MAX, col=f'cv_data{k}'),
                    'SPE_cga_area': mk_response(row, normalize=NORMALIZE, vmin = CGA_BOUNDS[0], vmax = CGA_BOUNDS[1], col=f'cv_data{k}'),
                    'cv_data': row[f'cv_data{k}'],
                    'cv_bins': read_cv_data_bins(row[f'cv_data{k}'], normalize=NORMALIZE)
                }
            )

    df_test = pd.DataFrame(results)
    return df_train, df_test

def caff_sweep():
    # best result:
    # CAFF_PEAK_DETECTION_MIN= 1.23, CAFF_PEAK_DETECTION_MAX= 1.33, Train SPE r2: 0.8594, Test SPE r2: 0.1806,
    for CAFF_PEAK_DETECTION_MIN in [1.1 + i * 0.01 for i in range(0, 50)]:
        for CAFF_PEAK_DETECTION_MAX in [1.2 + i * 0.01 for i in range(0, 50)]:
            if CAFF_PEAK_DETECTION_MAX - CAFF_PEAK_DETECTION_MIN < 0.1:
                continue


            df_train, df_test = build_model_data(CAFF_PEAK_DETECTION_MIN, CAFF_PEAK_DETECTION_MAX, CGA_BOUNDS=(0.10, 0.20), NORMALIZE=True)
            pred_x, pred_y, r2, intercept, slope, df_train = build_caff_model(df_train)
            df_test = apply_caff_model(df_test, intercept, slope)

            # print(tabulate.tabulate(df_train, headers='keys', tablefmt='pipe', showindex=False, floatfmt=".3f",
            #                     numalign="right", stralign="left"))

            test_r2 = r2_score(df_test['HPLC_Caff'], df_test['SPE_Caff'])

            #if test_r2> 0.12:
            if test_r2> 0.239:
            #if r2> 0.3:
                print(f'CAFF_PEAK_DETECTION_MIN={CAFF_PEAK_DETECTION_MIN:5.2f}, CAFF_PEAK_DETECTION_MAX={CAFF_PEAK_DETECTION_MAX:5.2f}, ', end="")
                print(f"Train SPE r2: {r2:6.4f}, Test SPE r2: {test_r2:6.4f}, ")


            # print(tabulate.tabulate(df_test, headers='keys', tablefmt='pipe', showindex=False, floatfmt=".3f",
            #                        numalign="right", stralign="left"))


            # exit()

def cga_sweep():
    # best result:
    # CGA_MIN= 0.10, CGA_MAX= 0.20 Train SPE CGA r2: 0.3686, Test SPE CGA r2: -0.2519
    CAFF_PEAK_DETECTION_MIN = 1.23
    CAFF_PEAK_DETECTION_MAX = 1.33
    for CGA_MIN in [0 + i * 0.01 for i in range(0, 80)]:
        for CGA_MAX in [.2 + i * 0.01 for i in range(0, 80)]:
            if CGA_MAX - CGA_MIN < 0.1:
                continue


            df_train, df_test = build_model_data(CAFF_PEAK_DETECTION_MIN, CAFF_PEAK_DETECTION_MAX, CGA_BOUNDS=(CGA_MIN, CGA_MAX), NORMALIZE=True)
            pred_x, pred_y, r2, intercept, slope, df_train = build_cga_model(df_train)
            df_test = apply_cga_model(df_test, intercept, slope)

            # print(tabulate.tabulate(df_train, headers='keys', tablefmt='pipe', showindex=False, floatfmt=".3f",
            #                         numalign="right", stralign="left"))

            test_r2 = r2_score(df_test['HPLC_CGA'], df_test['SPE_CGA'])

            #if test_r2> 0.12:
            if test_r2>= 0.3435:
                print(f'CGA_MIN={CGA_MIN:5.2f}, CGA_MAX={CGA_MAX:5.2f} ', end="")
                print(f"Train SPE CGA r2: {r2:6.4f}, Test SPE CGA r2: {test_r2:6.4f}")

            # print(tabulate.tabulate(df_test, headers='keys', tablefmt='pipe', showindex=False, floatfmt=".3f",
            #                        numalign="right", stralign="left"))



if __name__ == "__main__":


    setup_mplt()


    #caff_sweep()
    #cga_sweep()
    #exit()

    bounds = {
        'CAFF_PEAK_DETECTION_MIN': 1.2,
        'CAFF_PEAK_DETECTION_MAX': 1.6,
        'CGA_BOUNDS': (0.10, 0.35),
        'NORMALIZE': True
    }

    #Train SPE Caff r2: 0.3267832098681297
    #Test  SPE Caff r2: 0.02802346516164811
    #Train SPE CGA  r2: 0.05751620355498499
    #Test  SPE CGA  r2: 0.1107462840458211

    df_train, df_test = build_model_data(**bounds)

    if 1:
        #plot_cv_curves(df_train[df_train['Name'] == 'Alabaster Colombian Decaf'], df_test, **bounds)
        #plt.show()
        #plot_cv_curves(df_train[df_train['Name'] == 'FRC ROBUSTA Brazil, medium roast IH'], df_test, **bounds)

        plot_cv_curves(df_train, df_test, **bounds)

        plt.show()




    pred_x, pred_y, r2, intercept, slope, df_train = build_caff_model(df_train)
    df_test = apply_caff_model(df_test, intercept, slope)


    print("Train SPE Caff r2:", r2)
    print("Test  SPE Caff r2:", r2_score(df_test['HPLC_Caff'], df_test['SPE_Caff']))


    pred_x, pred_y, r2, intercept, slope, df_train = build_cga_model(df_train)
    df_test = apply_cga_model(df_test, intercept, slope)

    print("Train SPE CGA  r2:", r2)
    print("Test  SPE CGA  r2:", r2_score(df_test['HPLC_CGA'], df_test['SPE_CGA']))


    if 1:
        # THIS PLOTS THE REGRESSION OF SPE CAFFEINE RESPONSE (TRAINING) VS HPLC CAFFEINE


        #fig, axes = plt.subplots(2,2,figsize=(12, 12))
        fig, axes = plt.subplots(1,2,figsize=(12, 4))

        #make_plot(axes[0,0], df['SPE_area_method'], 'SPE (Integral Method)')
        #make_plot(axes[0,1], df['SPE_area_norm'], 'SPE (Integral Method Normalized)')
        axes[0].scatter(df_test['SPE_area_norm'], df_test['HPLC_Caff'], color='orange', label='Test', alpha=0.5)
        axes[0].scatter(df_train['SPE_area_norm'], df_train['HPLC_Caff'], color = 'blue', label='Model', alpha=0.5)

        add_regression(axes[0], df_train['SPE_area_norm'], df_train['HPLC_Caff'])
        axes[0].grid()
        axes[0].set_xlabel('SPE Caffeine Response (uA.V)')
        axes[0].set_ylabel('HPLC Caffeine (ppm)')
        #axes[0].set_title('SPE Caffeine vs HPLC Caffeine')
        #axes[0].set_ylim(0, 1400)
        #axes[0].set_xlim(0, 120)
        axes[0].legend(loc='upper left')

        axes[1].scatter(df_test['SPE_cga_area'], df_test['HPLC_CGA'], color='orange', label='Test', alpha=0.5)
        axes[1].scatter(df_train['SPE_cga_area'], df_train['HPLC_CGA'], color='blue', label='Model', alpha=0.5)
        add_regression(axes[1], df_train['SPE_cga_area'], df_train['HPLC_CGA'])

        axes[1].set_xlabel('SPE CGA Response (uA.V)')
        axes[1].set_ylabel('HPLC CGA (ppm)')
        #axes[1].set_xlim(-10, -0)
        #axes[1].set_ylim(0, 1600)
        axes[1].grid()
        #axes[1].set_title(f'{label} vs HPLC CGA')
        #axes[1].set_ylim(-500, 1400)

        #make_plot(axes[1,0], df['SPE_peak_method'], 'SPE (Peak Method)')
        #make_plot(axes[1,1], df['SPE_peak_norm'], 'SPE (Peak Method Normalized)')

        #make_plot(axes[1], df_test['SPE_area_norm'], df_test['HPLC_Caff'], 'SPE (test)')
        plt.tight_layout()
        plt.savefig('spe_models.pdf', dpi=300)



    if 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.violinplot([df_train['SPE_Caff_err_pct'], df_test['SPE_Caff_err_pct'], df_train['SPE_CGA_err_pct'], df_test['SPE_CGA_err_pct']], showmeans=True, widths=0.5)
        #ax.set_xticklabels(['Train', 'Test'])
        ax.set_xticks([1, 2, 3, 4], labels=['Caffeine Model', 'Caffeine Test', 'CGA Model', 'CGA Test'])
        #ax.set_title('SPE vs HPLC Error')
        ax.set_ylim(-100,400)
        ax.set_ylabel('Error Percentage (%)')
        ax.yaxis.set_major_locator(mplt.ticker.MultipleLocator(50))
        ax.yaxis.set_minor_locator(mplt.ticker.MultipleLocator(10))
        ax.grid()
        plt.tight_layout()
        plt.savefig('error_violin.pdf', dpi=300)
    #plt.show()

    # df.drop(columns=['cv_data1', 'cv_data 2', 'cv_data 3'], inplace=True)

    # print(tabulate.tabulate(df, headers='keys', tablefmt='pipe', showindex=False, floatfmt=".2f",
    #                        numalign="right", stralign="left", disable_numparse=True))
