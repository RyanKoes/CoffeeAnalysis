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

'''
Important Globals
'''
SMOOTHING_WINDOW_SIZE = 50  # Size of the moving average window for smoothing
MSLOPE_DETECTION_MIN = 1.45  # Minimum voltage for slope detection
MSLOPE_DETECTION_MAX = 1.55  # Maximum voltage for slope detection
OBSERVED_RANGES = [(0.8, 1), (1.5, 1.6)]  # Observed ranges for spline fitting and other fits
PREDICTION_RANGE = (OBSERVED_RANGES[0][1], OBSERVED_RANGES[1][0])  # Prediction range for the fits

file_paths = [
                'voltammetry-files/BrazilCerado2.txt',
                'voltammetry-files/EthiopianDry1.txt',
                'voltammetry-files/GuatemalaDark1.txt',
              'voltammetry-files/GuatemalaLight1.txt',
              'voltammetry-files/GuatemalaMedium1.txt',
              'voltammetry-files/Java1.txt']
colors = ['#00ffc5',
        '#00ffc5',
        '#00ffc5',
          '#00ffc5',
          '#00ffc5',
          '#00ffc5']
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
    x_min_slope, y_min_slope, min_slope = find_lowest_slope(x_smoothed, y_smoothed, MSLOPE_DETECTION_MIN, MSLOPE_DETECTION_MAX)
    if x_min_slope is not None:
        print(f"File: {file_path}, Lowest Slope: {min_slope:.3f} at Voltage: {x_min_slope:.3f}")
        print(f"Detected Response at this point: {y_min_slope:.3f} uA")
        rsd.append(y_min_slope)

        # Plot the curve and the point of the lowest slope
        plt.plot(x_smoothed[0: len(x_smoothed) // 2], y_smoothed[0: len(y_smoothed) // 2], label=file_path,
                 color=colors[i], alpha=0.8)
        plt.scatter(x_min_slope, y_min_slope, color=colors[i], label=f"Min Slope {file_path}", zorder=5)

    return(x_smoothed[0: len(x_smoothed) // 2], y_smoothed[0: len(y_smoothed) // 2])


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

def roast_prediction(percentages):
    plt.figure(figsize=(10, 6))
    known_concentrations = np.array([14.6, 14.5, 17.1, 13.1, 15.1, 12.8])
    known_peak_values = np.array(percentages)
    print(known_peak_values)
    print(known_concentrations.shape, known_peak_values.shape)

    model = LinearRegression()
    #known_concentrations = known_concentrations.reshape(-1, 1)  # Reshape for sklearn
    model.fit(known_peak_values.reshape(-1, 1), known_concentrations)

    # Function to predict caffeine concentration from peak values
    def predict_caffeine_concentration(peak_values):
        peak_values = np.array(peak_values).reshape(-1, 1)
        predicted_concentrations = model.predict(peak_values)
        return predicted_concentrations

    # Example unknown peak values (collected peaks)
    unknown_peak_values = [0.80]  # Example collected peaks in uA

    # Predict caffeine concentration for unknown samples
    predicted_concentrations = predict_caffeine_concentration(unknown_peak_values)

    print("Predicted Caffeine Concentrations:")
    for peak, concentration in zip(unknown_peak_values, predicted_concentrations.flatten()):  # Convert to scalar
        print(f"Peak Value: {peak} uA -> Predicted Concentration: {concentration:.2f} ppm")

    # Plot known data and regression line
    plt.scatter(known_peak_values, known_concentrations, color='blue', label='Known Data')
    plt.plot(known_peak_values, model.predict(known_peak_values.reshape(-1, 1)), color='red', linestyle='--', label='Regression Line')

    # Plot unknown peak values and predictions
    plt.scatter(unknown_peak_values, predicted_concentrations, color='green', label='Predictions', marker='x')

    plt.xlabel('Peak Current (uA)')
    plt.ylabel('CGA moisture loss %')
    plt.title('CGA moisture loss Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()

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
    plt.legend(bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.ylim(-20, 200)
    plt.show()

    percentages = []
    for i, (x_data, y_data) in enumerate(smoothed_responses):
        target = 0.9
        index = np.argmin(np.abs(x_data - target))
        max = y_data[index]

        target = 0.45
        index = np.argmin(np.abs(x_data - target))
        mid = y_data[index]
        percent = mid / max
        percentages.append(percent)
        print(f"Curve {i+1}: Max Response at 1V: {max:.3f} uA, Mid Response at 0.5V: {mid:.3f} uA, Percent: {percent:.2%}")

    print()
    roast_prediction(percentages)

    plt.show()
