import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error

SMOOTHING_WINDOW_SIZE = 30  # Size of the moving average window for smoothing
MSLOPE_DETECTION_MIN = 1.25  # Minimum voltage for slope detection
MSLOPE_DETECTION_MAX = 1.55  # Maximum voltage for slope detection
CGA_MIN_VOLTAGE = 0.7
CGA_MAX_VOLTAGE = 0.75
CGA_NORMALIZE = True

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

# Coffee group names and their corresponding colors
coffee_groups = [
    ('Alabaster Columbia Reg', '#ff4d4d'),
    ('Brazil Cerado', '#ffb84d'),
    ('Ethiopian Dry', '#ffff4d'),
    ('Java', '#80ff4d'),
    ('Guatemala Light', '#4dffdb'),
    ('Guatemala Medium', '#4da6ff'),
    ('Guatemala Dark', '#b84dff')
]

data = []
responses = []
V_responses = []
legend_handles = []  # To store legend handles for unique groups

'''
Normalizes the data by finding the response at 1V for each curve.
'''


def cga_normalization(df):
    voltage = df['Applied Voltage'].values
    response = df['Detected Response'].values
    # Find the response between 0.9V and 1.1V in the up curve
    up_curve_indices = (voltage >= CGA_MIN_VOLTAGE) & (voltage <= CGA_MAX_VOLTAGE) & (
                np.diff(voltage, prepend=voltage[0]) > 0)
    # Average response between 0.9V and 1.1V in the up curve
    response_at_1v = np.mean(response[up_curve_indices])
    V_responses.append(response_at_1v)


def plot_data(i, file_path, df):
    voltage = df['Applied Voltage'].values  # Apply the shift to the voltage
    response = df['Detected Response'].values

    # Subtract the corresponding response at 1V for each curve
    if (CGA_NORMALIZE):
        response -= V_responses[i]
    # response += np.mean(V_responses)

    y_smoothed = moving_average(response, window_size=SMOOTHING_WINDOW_SIZE)
    x_smoothed = voltage[len(voltage) - len(y_smoothed):]

    # Find the point with the lowest slope within the 1.45V to 1.55V range
    x_min_slope, y_min_slope, min_slope = find_lowest_slope(x_smoothed, y_smoothed, MSLOPE_DETECTION_MIN,
                                                            MSLOPE_DETECTION_MAX)
    if x_min_slope is not None:
        print(f"File: {file_path}, Lowest Slope: {min_slope:.3f} at Voltage: {x_min_slope:.3f}")
        print(f"Detected Response at this point: {y_min_slope:.3f} uA")
        responses.append(y_min_slope)

        # Determine which coffee group this file belongs to
        group_index = i // 5  # Each group has 5 files
        group_name, group_color = coffee_groups[group_index]

        # Plot the curve without scatter points
        line, = plt.plot(x_smoothed[0: len(x_smoothed) // 2], y_smoothed[0: len(y_smoothed) // 2],
                         color=colors[i], alpha=0.8)

        # Add to legend handles only for the first file of each group
        if i % 5 == 0:  # First file of each group
            legend_handles.append((line, group_name))

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


'''
Regression model
'''


def train_and_evaluate_model(averages, ground_truth):
    # Reshape the input for sklearn (expects 2D array for features)
    X = np.array(averages).reshape(-1, 1)
    y = np.array(ground_truth)

    # Create and train the model
    model = LinearRegression()
    model.fit(X, y)

    # Predict using the model
    predictions = model.predict(X)

    r2 = r2_score(y, predictions)
    mse = mean_squared_error(y, predictions)

    # Print model evaluation metrics
    print(f"R² Score (Accuracy): {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")

    # Mean Absolute Error
    mae = mean_absolute_error(ground_truth, predictions)

    return model, predictions


'''
Processes the data in chunks of 5 and calculates the averages and standard deviations.
'''


def process_chunks(data):
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
    plt.title('Cyclic Voltammetry - Normalized')

    # Create legend with only unique coffee groups
    if legend_handles:
        lines, labels = zip(*legend_handles)
        plt.legend(lines, labels, bbox_to_anchor=(1, 1))

    plt.grid(True)
    plt.ylim(-20, 200)
    plt.show()

    averages, std_devs = process_chunks(responses)
    print(f"Average of Columbia Reg: {averages[0]:.2f} uA, Std Dev: {std_devs[0]:.2f} uA")
    print(f"Average of Brazil Cerado: {averages[1]:.2f} uA, Std Dev: {std_devs[1]:.2f} uA")
    print(f"Average of Ethiopian Dry: {averages[2]:.2f} uA, Std Dev: {std_devs[2]:.2f} uA")
    print(f"Average of Guatemala Light: {averages[3]:.2f} uA, Std Dev: {std_devs[3]:.2f} uA")
    print(f"Average of Guatemala Medium: {averages[4]:.2f} uA, Std Dev: {std_devs[4]:.2f} uA")
    print(f"Average of Guatemala Dark: {averages[5]:.2f} uA, Std Dev: {std_devs[5]:.2f} uA")

    model, predictions = train_and_evaluate_model(averages, caffeine_ppm)

    mae = mean_absolute_error(caffeine_ppm, predictions)
    percent_error = (mae / np.std(caffeine_ppm)) * 100
    print(f"Percent Error: {percent_error:.2f}%")

    plt.errorbar(averages, caffeine_ppm, yerr=std_devs, fmt='o', color='blue', label='Data', capsize=5)

    plt.plot(averages, predictions, color='red', label='Regression Line')
    mse = mean_squared_error(caffeine_ppm, predictions)
    standard_error = np.sqrt(mse)

    prediction_interval = 1.96 * standard_error

    plt.fill_between(averages, predictions - prediction_interval, predictions + prediction_interval, color='red',
                     alpha=0.2, label='Prediction Interval (95%)')

    plt.xlabel('Averages (uA)')
    plt.ylabel('Ground Truth (Caffeine ppm)')
    plt.title('Regression Line with Error Bars')

    # Show legend
    plt.legend()
    plt.show()