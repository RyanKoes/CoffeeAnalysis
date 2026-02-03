import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sc
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.optimize import approx_fprime
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

'''
Important Globals
'''
SMOOTHING_WINDOW_SIZE = 50  # Size of the moving average window for smoothing
MSLOPE_DETECTION_MIN = 1.45  # Minimum voltage for slope detection
MSLOPE_DETECTION_MAX = 1.55  # Maximum voltage for slope detection
OBSERVED_RANGES = [(0.8, 1), (1.5, 1.6)]  # Observed ranges for spline fitting and other fits
PREDICTION_RANGE = (OBSERVED_RANGES[0][1], OBSERVED_RANGES[1][0])  # Prediction range for the fits
TEST_SIZE = 0.3  # Fraction of data to use for testing
RANDOM_STATE = 42  # For reproducible results

# Actual moisture loss percentages corresponding to each file
MOISTURE_LOSS_PERCENTAGES = [13.8, 13.8, 13.8,
                             14.1, 14.1, 14.1,
                             14.4, 14.4, 14.4,
                             13.4, 13.4, 13.4,
                             12.3, 12.3, 12.3,
                             15.5, 15.5, 15.5,
                             13.5, 13.5, 13.5,
                             15.3, 15.3, 15.3,
                             18.3, 18.3, 18.3,
                             15.3, 15.3, 15.3,
                             12.3, 12.3, 12.3,
                             15.3, 15.3, 15.3,
                             12.3, 12.3, 12.3]

file_paths = ['voltammetry-files/A1.txt',
              'voltammetry-files/A2.txt',
              'voltammetry-files/A3edge.txt',
              'voltammetry-files/B1.txt',
              'voltammetry-files/B2.txt',
              'voltammetry-files/B3edge.txt',
              'voltammetry-files/C1.txt',
              'voltammetry-files/C2.txt',
              'voltammetry-files/C3edge.txt',
              'voltammetry-files/G1.txt',
              'voltammetry-files/G2.txt',
              'voltammetry-files/G3.txt',
              'voltammetry-files/H1.txt',
              'voltammetry-files/H2.txt',
              'voltammetry-files/H3.txt',
              'voltammetry-files/I1.txt',
              'voltammetry-files/I2.txt',
              'voltammetry-files/I3.txt',
              'voltammetry-files/J1.txt',
              'voltammetry-files/J2.txt',
              'voltammetry-files/J3.txt',
              'voltammetry-files/K1.txt',
              'voltammetry-files/K2.txt',
              'voltammetry-files/K3.txt',
              'voltammetry-files/L1.txt',
              'voltammetry-files/L2.txt',
              'voltammetry-files/L3.txt',
              'voltammetry-files/M1.txt',
              'voltammetry-files/M2.txt',
              'voltammetry-files/M3.txt',
              'voltammetry-files/N1.txt',
              'voltammetry-files/N2.txt',
              'voltammetry-files/N3.txt'
              ]

colors = [
    "#1f77b4", "#1f77b4", "#1f77b4",  # blue
    "#ff7f0e", "#ff7f0e", "#ff7f0e",  # orange
    "#2ca02c", "#2ca02c", "#2ca02c",  # green
    "#d62728", "#d62728", "#d62728",  # red
    "#9467bd", "#9467bd", "#9467bd",  # purple
    "#8c564b", "#8c564b", "#8c564b",  # brown
    "#e377c2", "#e377c2", "#e377c2",  # pink
    "#7f7f7f", "#7f7f7f", "#7f7f7f",  # gray
    "#bcbd22", "#bcbd22", "#bcbd22",  # yellow-green
    "#17becf", "#17becf", "#17becf",  # cyan
    "#aec7e8", "#aec7e8", "#aec7e8",  # light blue
    "#ffbb78", "#ffbb78", "#ffbb78",  # light orange
    "#98df8a", "#98df8a", "#98df8a"  # light green
]
data = []
rsd = []
V_responses = []

'''
Normalizes the data by finding the response at 1V for each curve.
'''


def cga_normalization(df):
    voltage = df['Applied Voltage'].values
    response = df['Detected Response'].values
    # Find the response between 0.9V and 1.1V in the up curve
    up_curve_indices = (voltage >= 0.75) & (voltage <= 1) & (np.diff(voltage, prepend=voltage[0]) > 0)
    # Average response between 0.9V and 1.1V in the up curve
    response_at_1v = np.mean(response[up_curve_indices])
    V_responses.append(response_at_1v)


'''
Plot each curve and plot dot for point of lowest slope in certain range
'''


def plot_data(i, file_path, df):
    voltage = df['Applied Voltage'].values  # Apply the shift to the voltage
    response = df['Detected Response'].values

    # Subtract the corresponding response at 1V for each curve
    # response -= V_responses[i]
    # response += np.mean(V_responses)

    y_smoothed = moving_average(response, window_size=SMOOTHING_WINDOW_SIZE)
    x_smoothed = voltage[len(voltage) - len(y_smoothed):]

    # Find the point with the lowest slope within the 1.45V to 1.55V range
    x_min_slope, y_min_slope, min_slope = find_lowest_slope(x_smoothed, y_smoothed, MSLOPE_DETECTION_MIN,
                                                            MSLOPE_DETECTION_MAX)
    if x_min_slope is not None:
        print(f"File: {file_path}, Lowest Slope: {min_slope:.3f} at Voltage: {x_min_slope:.3f}")
        print(f"Detected Response at this point: {y_min_slope:.3f} uA")
        rsd.append(y_min_slope)

        # Plot the curve and the point of the lowest slope
        plt.plot(x_smoothed[0: len(x_smoothed) // 2], y_smoothed[0: len(y_smoothed) // 2], label=file_path,
                 color=colors[i], alpha=0.8)
        plt.scatter(x_min_slope, y_min_slope, color=colors[i], label=f"Min Slope {file_path}", zorder=5)

    return (x_smoothed[0: len(x_smoothed) // 2], y_smoothed[0: len(y_smoothed) // 2])


'''
Finds the lowest slope in the given voltage and response arrays within the specified range.
'''


def find_lowest_slope(voltage, response, v_min=1.490, v_max=1.500):
    # Filter the voltage and response arrays to only include values between 1.25V and 1.6V
    valid_indices = (voltage >= v_min) & (voltage <= v_max)
    filtered_voltage = voltage[valid_indices]
    filtered_response = response[valid_indices]

    # Calculate the differences between consecutive voltage and response values
    delta_voltage = np.diff(filtered_voltage)
    delta_response = np.diff(filtered_response)

    # Only consider slopes where the voltage is increasing
    increasing_indices = delta_voltage > 0

    # Calculate slopes only for increasing voltage values
    slopes = np.zeros_like(delta_voltage)
    slopes[increasing_indices] = delta_response[increasing_indices] / delta_voltage[increasing_indices]

    # Check that there are valid increasing voltage indices to calculate slopes
    if np.any(increasing_indices):
        # Find the index of the minimum slope among the increasing voltage section
        min_slope_index = np.argmin(slopes[increasing_indices])  # Index of the smallest slope among increasing voltages

        # Adjust indices to match the original filtered arrays
        x_min_slope = filtered_voltage[:-1][increasing_indices][min_slope_index]
        y_min_slope = filtered_response[:-1][increasing_indices][min_slope_index]

        return x_min_slope, y_min_slope, slopes[increasing_indices][min_slope_index]
    else:
        return None, None, None


'''
Reads the data from the specified file path and returns a DataFrame.
'''


def read_data(file_path):
    try:
        df = pd.read_csv(file_path, delimiter=',', header=None, names=['Time', 'Applied Voltage', 'Detected Response'])
        df = df[(df['Applied Voltage'] >= 0.0)]
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


'''
Calculates the moving average of a given array y with a specified window size.
'''


def moving_average(y, window_size=5):
    return np.convolve(y, np.ones(window_size) / window_size, mode='valid')


def average_percentages_by_groups(percentages, group_size=3):
    """
    Average percentages in groups of specified size.

    Args:
        percentages: List of percentage values
        group_size: Size of each group (default 3)

    Returns:
        List of averaged percentages and corresponding moisture loss values
    """
    averaged_percentages = []
    averaged_moisture_loss = []

    for i in range(0, len(percentages), group_size):
        group = percentages[i:i + group_size]
        if len(group) == group_size:  # Only process complete groups
            avg_percentage = np.mean(group)
            averaged_percentages.append(avg_percentage)

            # Get the corresponding moisture loss (should be the same for all 3 in the group)
            avg_moisture_loss = MOISTURE_LOSS_PERCENTAGES[i]  # Take the first one since they're all the same
            averaged_moisture_loss.append(avg_moisture_loss)

            print(f"Group {i // group_size + 1}: {group} -> Average: {avg_percentage:.4f}")

    return averaged_percentages, averaged_moisture_loss


def roast_prediction(percentages):
    # Average the percentages by groups of 3
    averaged_percentages, averaged_moisture_loss = average_percentages_by_groups(percentages, group_size=3)

    known_concentrations = np.array(averaged_moisture_loss)
    known_peak_values = np.array(averaged_percentages)

    print(f"\nAfter averaging:")
    print(f"Number of samples: {len(known_peak_values)}")

    # Fit a simple linear regression to calculate R²
    model = LinearRegression()
    model.fit(known_peak_values.reshape(-1, 1), known_concentrations)

    # Calculate R² for the entire dataset
    y_pred = model.predict(known_peak_values.reshape(-1, 1))
    r2 = r2_score(known_concentrations, y_pred)

    print(f"Overall R²: {r2:.4f}")
    print(f"Model Coefficient: {model.coef_[0]:.4f}")
    print(f"Model Intercept: {model.intercept_:.4f}")

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Moisture Loss Percentage
    ax1.scatter(known_peak_values, known_concentrations, color='blue', s=80, alpha=0.7, label='Data Points')

    # Plot regression line
    x_range = np.linspace(min(known_peak_values), max(known_peak_values), 100)
    y_range = model.predict(x_range.reshape(-1, 1))
    ax1.plot(x_range, y_range, color='red', linestyle='--', linewidth=2, label='Linear Fit')

    ax1.set_xlabel('Peak Current Ratio', fontsize=12)
    ax1.set_ylabel('Moisture Loss %', fontsize=12)
    ax1.set_title('Peak Current Ratio vs Moisture Loss Percentage', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add R² value as text on the plot
    ax1.text(0.5, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes,
             fontsize=14, verticalalignment='top', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Add equation text
    equation = f'y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}'
    ax1.text(0.5, 0.85, equation, transform=ax1.transAxes,
             fontsize=12, verticalalignment='top',  horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Plot 2: Roast Categories
    def categorize_roast(moisture_loss):
        if 11 <= moisture_loss < 14:
            return 'Light'
        elif 14 <= moisture_loss < 16:
            return 'Medium'
        elif 16 <= moisture_loss <= 20:
            return 'Dark'
        else:
            return 'Unknown'

    # Categorize the roast levels
    roast_categories = [categorize_roast(ml) for ml in known_concentrations]

    # Define colors for each category
    color_map = {'Light': '#D2B48C', 'Medium': '#8B4513', 'Dark': '#2F1B14'}  # tan, saddle brown, dark brown
    colors_for_plot = [color_map[cat] for cat in roast_categories]

    # Create scatter plot with categories
    for category in ['Light', 'Medium', 'Dark']:
        mask = np.array(roast_categories) == category
        if np.any(mask):
            ax2.scatter(known_peak_values[mask], [category] * np.sum(mask),
                        color=color_map[category], s=80, alpha=0.7, label=f'{category} Roast')

    ax2.set_xlabel('Peak Current Ratio', fontsize=12)
    ax2.set_ylabel('Roast Level', fontsize=12)
    ax2.set_title('Peak Current Ratio vs Roast Level', fontsize=14)
    ax2.legend(loc = 'upper center', fontsize=10, bbox_to_anchor=(0.5, 1.15), ncol=3)
    ax2.grid(True, alpha=0.3)

    # Add category ranges as text
    category_text = "Light: 11-14%\nMedium: 14-16%\nDark: 16-20%"
    ax2.text(0.5, 0.95, category_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Print category breakdown
    print(f"\nRoast Level Breakdown:")
    for category in ['Light', 'Medium', 'Dark']:
        count = roast_categories.count(category)
        print(f"{category}: {count} samples")

    plt.tight_layout()
    plt.show()

    return r2


if __name__ == '__main__':

    plt.figure(figsize=(10, 6))
    smoothed_responses = []

    for i, file_path in enumerate(file_paths):
        df = read_data(file_path)
        if df is not None:
            cga_normalization(df)

    # Plot each shifted curve and find the point with the lowest slope in the 1.45V - 1.55V range
    for i, file_path in enumerate(file_paths):
        df = read_data(file_path)
        if df is not None:
            smoothed_responses.append(plot_data(i, file_path, df))

    plt.xlabel('Applied Voltage')
    plt.ylabel('Response uA')
    plt.title('Cyclic Voltammetry - LMD analysis')
    plt.legend(bbox_to_anchor=(1.1, 1))
    plt.grid(True)
    plt.ylim(-20, 200)
    plt.show()

    percentages = []
    for i, (x_data, y_data) in enumerate(smoothed_responses):
        target = 0.9
        index = np.argmin(np.abs(x_data - target))
        max_val = y_data[index]

        target = 0.45
        index = np.argmin(np.abs(x_data - target))
        mid = y_data[index]
        percent = mid / max_val
        percentages.append(percent)
        print(
            f"Curve {i + 1}: Max Response at 0.9V: {max_val:.3f} uA, Mid Response at 0.45V: {mid:.3f} uA, Percent: {percent:.2%}")

    print()
    r2_value = roast_prediction(percentages)

    plt.show()