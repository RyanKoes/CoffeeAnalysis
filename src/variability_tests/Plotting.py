import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sc
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import os
import glob

'''
Important Globals
'''
SMOOTHING_WINDOW_SIZE = 25  # Size of the moving average window for smoothing
MSLOPE_DETECTION_MIN = 0.25  # Minimum voltage for slope detection
MSLOPE_DETECTION_MAX = 0.31  # Maximum voltage for slope detection
OBSERVED_RANGES = [(0.8, 1), (1.5, 1.6)]  # Observed ranges for spline fitting and other fits
PREDICTION_RANGE = (OBSERVED_RANGES[0][1], OBSERVED_RANGES[1][0])  # Prediction range for the fits

# <<< Specify the folder containing your files >>>
FOLDER_PATH = "CGA Standards"

# Load all .txt files in folder
file_paths = sorted(glob.glob(os.path.join(FOLDER_PATH, "*.txt")))

# Dynamic color scheme: uses matplotlib colormap, scales with number of files
cmap = plt.cm.get_cmap("tab10", len(file_paths))  # tab10, Set2, plasma, etc. are good options
colors = [cmap(i) for i in range(len(file_paths))]

data = []
rsd = []
V_responses = []


def cga_normalization(df):
    voltage = df['Applied Voltage'].values
    response = df['Detected Response'].values
    # Find the response between 0.9V and 1.1V in the up curve
    up_curve_indices = (voltage >= 0.75) & (voltage <= 1) & (np.diff(voltage, prepend=voltage[0]) > 0)
    response_at_1v = np.mean(response[up_curve_indices])
    V_responses.append(response_at_1v)


def plot_data(i, file_path, df, norm = False):
    voltage = df['Applied Voltage'].values
    response = df['Detected Response'].values

    # Normalize
    if norm and len(V_responses) == len(file_paths):
        response -= V_responses[i]

    y_smoothed = moving_average(response, window_size=SMOOTHING_WINDOW_SIZE)
    x_smoothed = voltage[len(voltage) - len(y_smoothed):]

    # Lowest slope point
    x_min_slope, y_min_slope, min_slope = find_lowest_slope(
        x_smoothed, y_smoothed, MSLOPE_DETECTION_MIN, MSLOPE_DETECTION_MAX
    )
    if x_min_slope is not None:
        print(f"File: {os.path.basename(file_path)}, Lowest Slope: {min_slope:.3f} at Voltage: {x_min_slope:.3f}")
        print(f"Detected Response at this point: {y_min_slope:.3f} uA")
        rsd.append(y_min_slope)

        plt.plot(x_smoothed[: len(x_smoothed) // 2], y_smoothed[: len(y_smoothed) // 2],
                 label=os.path.basename(file_path), color=colors[i], alpha=0.8)
        plt.scatter(x_min_slope, y_min_slope, color=colors[i], s=50, marker="o", edgecolors="k")

    return x_smoothed[: len(x_smoothed) // 2], y_smoothed[: len(y_smoothed) // 2]


def find_lowest_slope(voltage, response, v_min=1.490, v_max=1.500):
    valid_indices = (voltage >= v_min) & (voltage <= v_max)
    filtered_voltage = voltage[valid_indices]
    filtered_response = response[valid_indices]

    delta_voltage = np.diff(filtered_voltage)
    delta_response = np.diff(filtered_response)
    increasing_indices = delta_voltage > 0

    slopes = np.zeros_like(delta_voltage)
    # Designed to find the plateau with the lowest positive slope
    slopes[increasing_indices] = abs(delta_response[increasing_indices] / delta_voltage[increasing_indices])

    if np.any(increasing_indices):
        min_slope_index = np.argmin(slopes[increasing_indices])
        x_min_slope = filtered_voltage[:-1][increasing_indices][min_slope_index]
        y_min_slope = filtered_response[:-1][increasing_indices][min_slope_index]
        return x_min_slope, y_min_slope, slopes[increasing_indices][min_slope_index]
    else:
        return None, None, None


def read_data(file_path):
    try:
        df = pd.read_csv(file_path, delimiter=',', header=None,
                         names=['Time', 'Applied Voltage', 'Detected Response'])
        df = df[(df['Applied Voltage'] >= 0.0)]
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def moving_average(y, window_size=5):
    return np.convolve(y, np.ones(window_size) / window_size, mode='valid')


# (keeping your spline, exponential, polynomial, power law, and error functions as-is...)


if __name__ == '__main__':
    plt.figure(figsize=(10, 6))
    smoothed_responses = []

    for i, file_path in enumerate(file_paths):
        df = read_data(file_path)
        # Use this section if CGA normalization is needed
        # if df is not None:
        #     cga_normalization(df)

    for i, file_path in enumerate(file_paths):
        df = read_data(file_path)
        if df is not None:
            smoothed_responses.append(plot_data(i, file_path, df))

    # (rest of prediction/fit sections unchanged...)

    plt.xlabel('Applied Voltage')
    plt.ylabel('Response uA')
    plt.title('Cyclic Voltammetry - Coffee')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.ylim(-20, 100)
    plt.tight_layout()
    plt.show()

    # Linear Regression Analysis
    y_all = np.array([200] * 5 + [500] * 5 + [800] * 5)

    # Ensure rsd matches same order/length as concentrations
    rsd = np.array(rsd)

    # Perform linear regression
    slope, intercept = np.polyfit(y_all, rsd, 1)
    r2 = np.corrcoef(y_all, rsd)[0, 1] ** 2

    # Predicted line for plotting
    x_fit = np.linspace(min(y_all), max(y_all), 100)
    y_fit = slope * x_fit + intercept

    # Plot calibration curve
    plt.figure(figsize=(7, 5))
    plt.scatter(y_all, rsd, color='blue', label='Measured Points')
    plt.plot(x_fit, y_fit, color='red', label=f'Linear Fit: y = {slope:.3f}x + {intercept:.2f}')
    plt.text(
        0.05, 0.95,
        f'$R^2$ = {r2:.4f}',
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top'
    )

    plt.xlabel('CGA Concentration (ppm)')
    plt.ylabel('Detected Response (µA)')
    plt.title('Calibration Curve for Caffeine Standards')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
