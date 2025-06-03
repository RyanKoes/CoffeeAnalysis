import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sc
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

'''
Important Globals
'''
SMOOTHING_WINDOW_SIZE = 30  # Size of the moving average window for smoothing
MSLOPE_DETECTION_MIN = 1.45  # Minimum voltage for slope detection
MSLOPE_DETECTION_MAX = 1.55  # Maximum voltage for slope detection
OBSERVED_RANGES = [(0.8, 1), (1.5, 1.6)]  # Observed ranges for spline fitting and other fits
PREDICTION_RANGE = (OBSERVED_RANGES[0][1], OBSERVED_RANGES[1][0])  # Prediction range for the fits

file_paths = ['voltammetry-files/AlabasterColumbiaDecaf1.txt',
                'voltammetry-files/AlabasterColumbiaDecaf2.txt',
                'voltammetry-files/AlabasterColumbiaDecaf3.txt',
                'voltammetry-files/AlabasterColumbiaDecaf4.txt',
                'voltammetry-files/AlabasterColumbiaDecaf5.txt',]
colors = ['#00ffc5',
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
    response -= V_responses[i]
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

'''
Train a spline on the observed data with gaps in the voltage range.
'''
def train_spline(voltage, response, observed_ranges, prediction_range, num_points=100):
    # Select only the points that fall within the observed ranges
    # Select only the points that fall within the observed ranges
    observed_indices = np.zeros_like(voltage, dtype=bool)
    for v_min, v_max in observed_ranges:
        observed_indices |= (voltage >= v_min) & (voltage <= v_max)

    voltage_observed = voltage[observed_indices]
    response_observed = response[observed_indices]

    if len(voltage_observed) < 5:
        print("Not enough data points for spline training.")
        return None, None

    # Fit the spline on the observed data
    spline = UnivariateSpline(voltage_observed, response_observed, k=3, s=1e-2)

    # Generate test points within the missing range
    voltage_pred = np.linspace(prediction_range[0], prediction_range[1], num_points)

    # Predict response values
    spline_prediction = spline(voltage_pred)

    return voltage_pred, spline_prediction

def power_law_fit_predict(voltage, response, observed_ranges, prediction_range, num_points=100):
    """Fits a power law and predicts within the prediction range."""

    observed_indices = np.zeros_like(voltage, dtype=bool)
    for v_min, v_max in observed_ranges:
        observed_indices |= (voltage >= v_min) & (voltage <= v_max)

    voltage_observed = voltage[observed_indices]
    response_observed = response[observed_indices]

    if len(voltage_observed) < 2:
        print("Not enough data points for power law fit.")
        return None, None

    def power_law_func(x, a, b):
        return a * x**b

    try:
        params, covariance = curve_fit(power_law_func, voltage_observed, response_observed, p0=[1, 1], maxfev=5000)
        voltage_pred = np.linspace(prediction_range[0], prediction_range[1], num_points)
        power_law_prediction = power_law_func(voltage_pred, *params)
        return voltage_pred, power_law_prediction

    except RuntimeError as e:
        print(f"Power law fit failed: {e}")
        return None, None
    except TypeError as e:
        print(f"Power law fit failed: {e}")
        return None, None


def exponential_fit_predict(voltage, response, observed_ranges, prediction_range, num_points=100):
    # Select only the points that fall within the observed ranges
    observed_indices = np.zeros_like(voltage, dtype=bool)
    for v_min, v_max in observed_ranges:
        observed_indices |= (voltage >= v_min) & (voltage <= v_max)

    voltage_observed = voltage[observed_indices]
    response_observed = response[observed_indices]

    if len(voltage_observed) < 3:  # Need at least 3 points for exponential fit
        print("Not enough data points for exponential fit.")
        return None, None

    # Define the exponential function
    def exponential_func(x, a, b, c):
        return a * np.exp(b * x) + c

    try:
        # Fit the exponential function
        params, covariance = curve_fit(exponential_func, voltage_observed, response_observed, p0=[1, 1, 1], maxfev=5000)

        # Generate test points within the prediction range
        voltage_pred = np.linspace(prediction_range[0], prediction_range[1], num_points)

        # Predict response values
        exponential_prediction = exponential_func(voltage_pred, *params)

        return voltage_pred, exponential_prediction

    except RuntimeError as e:
        print(f"Exponential fit failed: {e}")
        return None, None
    except TypeError as e:
        print(f"Exponential fit failed: {e}")
        return None,None

def polynomial_fit_predict(voltage, response, observed_ranges, prediction_range, degree=2, num_points=100):
    """Fits a polynomial and predicts within the prediction range."""

    observed_indices = np.zeros_like(voltage, dtype=bool)
    for v_min, v_max in observed_ranges:
        observed_indices |= (voltage >= v_min) & (voltage <= v_max)

    voltage_observed = voltage[observed_indices]
    response_observed = response[observed_indices]

    if len(voltage_observed) < degree + 1:
        print(f"Not enough data points for polynomial fit of degree {degree}.")
        return None, None

    try:
        # Fit polynomial
        coeffs = np.polyfit(voltage_observed, response_observed, degree)
        poly = np.poly1d(coeffs)

        # Generate prediction points
        voltage_pred = np.linspace(prediction_range[0], prediction_range[1], num_points)
        polynomial_prediction = poly(voltage_pred)

        return voltage_pred, polynomial_prediction

    except Exception as e:
        print(f"Polynomial fit failed: {e}")
        return None, None

'''
Main function for calculating error of spline
'''
def calculate_error(voltage_pred, spline_pred, voltage_actual, response_actual, prediction_range):
    """Calculates the error between the predicted and actual data, including MAPE."""

    # Extract actual data in the prediction range
    actual_indices = (voltage_actual >= prediction_range[0]) & (voltage_actual <= prediction_range[1])
    voltage_actual_in_range = voltage_actual[actual_indices]
    response_actual_in_range = response_actual[actual_indices]

    if len(voltage_actual_in_range) == 0:
        print("No actual data points in prediction range.")
        return None, None, None, None

    # Interpolate actual data at voltage_pred points
    interpolated_actual = np.interp(voltage_pred, voltage_actual_in_range, response_actual_in_range)

    # Calculate error metrics
    mse = np.mean((spline_pred - interpolated_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(spline_pred - interpolated_actual))

    # Calculate MAPE
    mape = np.mean(np.abs((interpolated_actual - spline_pred) / interpolated_actual)) * 100

    return mse, rmse, mae, mape


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

    # Handles spline interpolation
    for data in smoothed_responses:
        x_smoothed, y_smoothed = data
        observed_ranges = OBSERVED_RANGES
        v_pred, spline_pred = train_spline(x_smoothed, y_smoothed, observed_ranges, PREDICTION_RANGE)

        if v_pred is not None and spline_pred is not None:
            plt.plot(v_pred, spline_pred, label="Spline Prediction")

            # Calculate and print error
            mse, rmse, mae, mape = calculate_error(v_pred, spline_pred, x_smoothed, y_smoothed, PREDICTION_RANGE)
            if mse is not None:
                print(f"Spline - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}%")
        else:
            print("Spline prediction failed for this data.")

    # Handles exponential fit interpolation
    for data in smoothed_responses:
        x_smoothed, y_smoothed = data
        observed_ranges = OBSERVED_RANGES
        v_pred_exp, exp_pred = exponential_fit_predict(x_smoothed, y_smoothed, observed_ranges, PREDICTION_RANGE)

        if v_pred_exp is not None and exp_pred is not None:
            plt.plot(v_pred_exp, exp_pred, label="Exponential Prediction")

            # Calculate error (similar to spline example)
            mse_exp, rmse_exp, mae_exp, mape_exp = calculate_error(v_pred_exp, exp_pred, x_smoothed, y_smoothed,
                                                                   PREDICTION_RANGE)
            if mse_exp is not None:
                print(
                    f"Exponential - MSE: {mse_exp:.4f}, RMSE: {rmse_exp:.4f}, MAE: {mae_exp:.4f}, MAPE: {mape_exp:.4f}%")
        else:
            print("Exponential prediction failed for this data.")

    for data in smoothed_responses:
        x_smoothed, y_smoothed = data
        observed_ranges = OBSERVED_RANGES
        v_pred_poly, poly_pred = polynomial_fit_predict(x_smoothed, y_smoothed, observed_ranges, PREDICTION_RANGE,
                                                        degree=2)  # try degree 2

        if v_pred_poly is not None and poly_pred is not None:
            plt.plot(v_pred_poly, poly_pred, label="Polynomial Prediction")

            # Calculate error (similar to spline example)
            mse_poly, rmse_poly, mae_poly, mape_poly = calculate_error(v_pred_poly, poly_pred, x_smoothed, y_smoothed,
                                                                       PREDICTION_RANGE)
            if mse_poly is not None:
                print(
                    f"Polynomial - MSE: {mse_poly:.4f}, RMSE: {rmse_poly:.4f}, MAE: {mae_poly:.4f}, MAPE: {mape_poly:.4f}%")
        else:
            print("Polynomial prediction failed for this data.")

    for data in smoothed_responses:
        x_smoothed, y_smoothed = data
        observed_ranges = OBSERVED_RANGES

        # ... (Spline, Exponential, Polynomial predictions) ...

        v_pred_power, power_pred = power_law_fit_predict(x_smoothed, y_smoothed, observed_ranges, PREDICTION_RANGE)

        if v_pred_power is not None and power_pred is not None:
            plt.plot(v_pred_power, power_pred, label="Power Law Prediction")

            # Calculate error (similar to spline example)
            mse_power, rmse_power, mae_power, mape_power = calculate_error(v_pred_power, power_pred, x_smoothed,
                                                                           y_smoothed, PREDICTION_RANGE)
            if mse_power is not None:
                print(
                    f"Power Law - MSE: {mse_power:.4f}, RMSE: {rmse_power:.4f}, MAE: {mae_power:.4f}, MAPE: {mape_power:.4f}%")
        else:
            print("Power law prediction failed for this data.")

    plt.xlabel('Applied Voltage')
    plt.ylabel('Response uA')
    plt.title('Cyclic Voltammetry - LMD analysis')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.ylim(-20, 200)
    plt.show()