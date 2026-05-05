import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error
from itertools import product

# Original constants
SMOOTHING_WINDOW_SIZE = 30  # Size of the moving average window for smoothing
MSLOPE_DETECTION_MIN = 1.25  # Minimum voltage for slope detection
MSLOPE_DETECTION_MAX = 1.55  # Maximum voltage for slope detection
CGA_NORMALIZE = True

# Define ranges for grid search
CGA_MIN_RANGES = np.linspace(0.0, 1.9, 150)  # From 0.5 to 1.0 in 6 steps
CGA_MAX_RANGES = np.linspace(0.0, 1.9, 150)  # From 0.6 to 1.1 in 6 steps

file_paths = ['voltammetry-files/AlabasterColumbiaReg1.txt',
              'voltammetry-files/AlabasterColumbiaReg2.txt',
              'voltammetry-files/AlabasterColumbiaReg3.txt',
              'voltammetry-files/AlabasterColumbiaReg4.txt',
              'voltammetry-files/AlabasterColumbiaReg5.txt',
              'voltammetry-files/BrazilCerado1.txt',
              'voltammetry-files/BrazilCerado2.txt',
              'voltammetry-files/BrazilCerado3.txt',
              'voltammetry-files/BrazilCerado4.txt',
              'voltammetry-files/BrazilCerado5.txt',
              'voltammetry-files/EthiopianDry1.txt',
              'voltammetry-files/EthiopianDry2.txt',
              'voltammetry-files/EthiopianDry3.txt',
              'voltammetry-files/EthiopianDry4.txt',
              'voltammetry-files/EthiopianDry5.txt',
              'voltammetry-files/Java1.txt',
              'voltammetry-files/Java2.txt',
              'voltammetry-files/Java3.txt',
              'voltammetry-files/Java4.txt',
              'voltammetry-files/Java5.txt',
              'voltammetry-files/GuatemalaLight1.txt',
              'voltammetry-files/GuatemalaLight2.txt',
              'voltammetry-files/GuatemalaLight3.txt',
              'voltammetry-files/GuatemalaLight4.txt',
              'voltammetry-files/GuatemalaLight5.txt',
              'voltammetry-files/GuatemalaMedium1.txt',
              'voltammetry-files/GuatemalaMedium2.txt',
              'voltammetry-files/GuatemalaMedium3.txt',
              'voltammetry-files/GuatemalaMedium4.txt',
              'voltammetry-files/GuatemalaMedium5.txt',
              'voltammetry-files/GuatemalaDark1.txt',
              'voltammetry-files/GuatemalaDark2.txt',
              'voltammetry-files/GuatemalaDark3.txt',
              'voltammetry-files/GuatemalaDark4.txt',
              'voltammetry-files/GuatemalaDark5.txt']

caffeine_ppm = [803.1, 883.8, 846.8, 834.4, 835.7, 787.1, 854.0]

colors = [
    '#ff4d4d', '#ff4d4d', '#ff4d4d', '#ff4d4d', '#ff4d4d',
    '#ffb84d', '#ffb84d', '#ffb84d', '#ffb84d', '#ffb84d',
    '#ffff4d', '#ffff4d', '#ffff4d', '#ffff4d', '#ffff4d',
    '#80ff4d', '#80ff4d', '#80ff4d', '#80ff4d', '#80ff4d',
    '#4dffdb', '#4dffdb', '#4dffdb', '#4dffdb', '#4dffdb',
    '#4da6ff', '#4da6ff', '#4da6ff', '#4da6ff', '#4da6ff',
    '#b84dff', '#b84dff', '#b84dff', '#b84dff', '#b84dff'
]


def read_data(file_path):
    """
    Reads the data from the specified file path and returns a DataFrame.
    """
    try:
        df = pd.read_csv(file_path, delimiter=',', header=None, names=['Time', 'Applied Voltage', 'Detected Response'])
        df = df[(df['Applied Voltage'] >= 0.0)]
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def moving_average(y, window_size=5):
    """
    Calculates the moving average of a given array y with a specified window size.
    """
    return np.convolve(y, np.ones(window_size) / window_size, mode='valid')


def find_lowest_slope(voltage, response, v_min=1.490, v_max=1.500):
    """
    Finds the lowest slope in the given voltage and response arrays within the specified range.
    """
    # Filter the voltage and response arrays to only include values between v_min and v_max
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


def process_chunks(data):
    """
    Processes the data in chunks of 5 and calculates the averages and standard deviations.
    """
    averages = []
    std_devs = []

    # Process data in chunks of 5
    for i in range(0, len(data), 5):
        chunk = data[i:i + 5]
        if len(chunk) == 5:
            avg = statistics.mean(chunk)
            std = statistics.stdev(chunk)
            averages.append(avg)
            std_devs.append(std)

    return averages, std_devs


def train_and_evaluate_model(averages, ground_truth):
    """
    Regression model training and evaluation
    """
    # Reshape the input for sklearn (expects 2D array for features)
    X = np.array(averages).reshape(-1, 1)
    y = np.array(ground_truth)

    # Create and train the model
    model = LinearRegression()
    model.fit(X, y)

    # Predict using the model
    predictions = model.predict(X)

    # Calculate metrics
    r2 = r2_score(y, predictions)
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    percent_error = (mae / np.std(ground_truth)) * 100

    return model, predictions, r2, mse, mae, percent_error


def run_grid_search():
    """
    Run grid search to find optimal CGA_MIN_VOLTAGE and CGA_MAX_VOLTAGE values.
    """
    # Dictionary to store results
    results = {}

    # For storing best parameters
    best_r2 = -float('inf')
    best_mae = float('inf')
    best_params = None

    # Cache the data frames to avoid reading them multiple times
    dfs = {}
    for file_path in file_paths:
        df = read_data(file_path)
        if df is not None:
            dfs[file_path] = df

    # Ensure we only test valid parameter combinations (min < max)
    valid_combinations = []
    for min_val, max_val in product(CGA_MIN_RANGES, CGA_MAX_RANGES):
        if min_val < max_val:  # Ensure min is less than max
            valid_combinations.append((min_val, max_val))

    #print(f"Testing {len(valid_combinations)} parameter combinations...")

    # Loop through all combinations of parameters
    for CGA_MIN_VOLTAGE, CGA_MAX_VOLTAGE in valid_combinations:
        # Format values for cleaner output
        min_v_str = f"{CGA_MIN_VOLTAGE:.2f}"
        max_v_str = f"{CGA_MAX_VOLTAGE:.2f}"

        #print(f"\nTesting CGA_MIN_VOLTAGE={min_v_str}, CGA_MAX_VOLTAGE={max_v_str}")

        # Reset responses for each iteration
        responses = []
        V_responses = []

        # First pass: normalization
        for file_path, df in dfs.items():
            voltage = df['Applied Voltage'].values
            response = df['Detected Response'].values

            # Find the response between min and max in the up curve
            up_curve_indices = (voltage >= CGA_MIN_VOLTAGE) & (voltage <= CGA_MAX_VOLTAGE) & (
                        np.diff(voltage, prepend=voltage[0]) > 0)

            # Average response between min and max in the up curve
            if np.any(up_curve_indices):
                response_at_v = np.mean(response[up_curve_indices])
                V_responses.append(response_at_v)
            else:
                V_responses.append(0)  # Fallback if no points in range

        # Second pass: calculate responses
        for i, (file_path, df) in enumerate(dfs.items()):
            voltage = df['Applied Voltage'].values
            response = df['Detected Response'].values

            # Apply normalization if enabled
            if CGA_NORMALIZE:
                response -= V_responses[i]

            # Apply smoothing
            y_smoothed = moving_average(response, window_size=SMOOTHING_WINDOW_SIZE)
            x_smoothed = voltage[len(voltage) - len(y_smoothed):]

            # Find the point with the lowest slope
            x_min_slope, y_min_slope, min_slope = find_lowest_slope(x_smoothed, y_smoothed, MSLOPE_DETECTION_MIN,
                                                                    MSLOPE_DETECTION_MAX)

            if x_min_slope is not None:
                responses.append(y_min_slope)

        # Only proceed if we have enough responses
        if len(responses) >= 30:  # Expected 35 files
            # Calculate averages and standard deviations
            averages, std_devs = process_chunks(responses)

            # Only proceed if we have the right number of averages
            if len(averages) == len(caffeine_ppm):
                # Train and evaluate model
                _, _, r2, mse, mae, percent_error = train_and_evaluate_model(averages, caffeine_ppm)

                # Store results
                result_key = (min_v_str, max_v_str)
                results[result_key] = {
                    'r2': r2,
                    'mse': mse,
                    'mae': mae,
                    'percent_error': percent_error
                }

                # Check if this is the best result
                if r2 > best_r2:
                    best_r2 = r2
                    best_mae = mae
                    best_params = result_key

                #print(f"  R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, Percent Error: {percent_error:.2f}%")
            else:
                print(f"  Warning: Insufficient averages. Got {len(averages)}, expected {len(caffeine_ppm)}")
        else:
            print(f"  Warning: Insufficient responses. Got {len(responses)}, expected at least 30")

    # Report best results
    if best_params:
        print("\n===== BEST PARAMETERS =====")
        print(f"CGA_MIN_VOLTAGE = {best_params[0]}")
        print(f"CGA_MAX_VOLTAGE = {best_params[1]}")
        print(f"R² Score: {best_r2:.4f}")
        print(f"MAE: {best_mae:.4f}")

        # Create a dataframe for easy sorting and visualization
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.index.names = ['CGA_MIN', 'CGA_MAX']
        results_df = results_df.reset_index()

        # Sort by R2 (descending)
        top_results = results_df.sort_values('r2', ascending=False).head(10)
        print("\n===== TOP 10 PARAMETER COMBINATIONS =====")
        print(top_results)

        # Optionally visualize results
        plt.figure(figsize=(12, 8))

        # Prepare data for heatmap
        min_values = sorted(list(set([float(x[0]) for x in results.keys()])))
        max_values = sorted(list(set([float(x[1]) for x in results.keys()])))

        heatmap_data = np.zeros((len(min_values), len(max_values)))
        heatmap_data.fill(np.nan)  # Fill with NaN for invalid combinations

        for (min_str, max_str), result in results.items():
            i = min_values.index(float(min_str))
            j = max_values.index(float(max_str))
            heatmap_data[i, j] = result['r2']

        plt.imshow(heatmap_data, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='R² Score')
        plt.xlabel('CGA_MAX_VOLTAGE Index')
        plt.ylabel('CGA_MIN_VOLTAGE Index')
        plt.title('R² Score Heatmap for Different CGA Voltage Ranges')

        # Add x and y axis labels
        plt.xticks(range(len(max_values)), [f"{v:.2f}" for v in max_values], rotation=45, fontsize=5)
        plt.yticks(range(len(min_values)), [f"{v:.2f}" for v in min_values], fontsize=5)

        plt.tight_layout()
        plt.show()

        return best_params, results_df
    else:
        print("No valid parameter combinations found!")
        return None, None


if __name__ == '__main__':
    best_params, results_df = run_grid_search()

    # If you want to see detailed results for the best parameters
    if best_params:
        CGA_MIN_VOLTAGE = float(best_params[0])
        CGA_MAX_VOLTAGE = float(best_params[1])

        print(
            f"\nRunning detailed analysis with best parameters: CGA_MIN_VOLTAGE={CGA_MIN_VOLTAGE}, CGA_MAX_VOLTAGE={CGA_MAX_VOLTAGE}")

        # Here you could add code to run a detailed analysis with the best parameters
        # This would be similar to your original main code but with the optimized parameters