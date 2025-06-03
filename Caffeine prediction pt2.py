import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error

SMOOTHING_WINDOW_SIZE = 30  # Size of the moving average window for smoothing
CGA_MIN_VOLTAGE = 0.7
CGA_MAX_VOLTAGE = 0.75
CGA_NORMALIZE = True

# New parameters for the reference line and peak detection
REFERENCE_START_V = 1.15  # Voltage where reference line starts
REFERENCE_END_V = 1.5  # Voltage where reference line ends
PEAK_DETECTION_MIN = 1.15  # Minimum voltage for peak detection
PEAK_DETECTION_MAX = 1.5  # Maximum voltage for peak detection

file_paths = ['voltammetry-files/ColumbiaDecaf1.txt',
                'voltammetry-files/ColumbiaDecaf2.txt',
                'voltammetry-files/ColumbiaDecaf3.txt',
                'voltammetry-files/ColumbiaDecaf100ppm1.txt',
                'voltammetry-files/ColumbiaDecaf100ppm2.txt',
                'voltammetry-files/ColumbiaDecaf100ppm3.txt',
                'voltammetry-files/ColumbiaDecaf250ppm1.txt',
                'voltammetry-files/ColumbiaDecaf250ppm2.txt',
                'voltammetry-files/ColumbiaDecaf250ppm3.txt',
                'voltammetry-files/ColumbiaDecaf500ppm1.txt',
                'voltammetry-files/ColumbiaDecaf500ppm2.txt',
                'voltammetry-files/ColumbiaDecaf500ppm3.txt',
                'voltammetry-files/ColumbiaDecaf1000ppmnew1.txt',
                'voltammetry-files/ColumbiaDecaf1000ppmnew2.txt',
                'voltammetry-files/ColumbiaDecaf1000ppmnew3.txt',
              'voltammetry-files/ColumbiaReg1.txt',
                'voltammetry-files/ColumbiaReg2.txt',
                'voltammetry-files/ColumbiaReg3.txt']

# Coffee names for labeling
coffee_names = [
    "Decaf",
    "100ppm",
    "250ppm",
    "500ppm",
    "1000ppm",
    "Columbia Reg"
]

caffeine_ppm = [80, 180, 330, 580, 1080, 787.1]

colors = [
    '#ff4d4d', '#ff4d4d', '#ff4d4d',
    '#ffb84d', '#ffb84d', '#ffb84d',
    '#ffff4d', '#ffff4d', '#ffff4d',
    '#80ff4d', '#80ff4d', '#80ff4d',
    '#4dffdb', '#4dffdb', '#4dffdb',
    '#4da6ff', '#4da6ff', '#4da6ff'
]
data = []
responses = []
V_responses = []

'''
Calculate the y-value on a line between two points for a given x-value
'''


def calculate_line_y(x, x1, y1, x2, y2):
    # Equation of line: y = mx + b
    m = (y2 - y1) / (x2 - x1)  # slope
    b = y1 - m * x1  # y-intercept
    return m * x + b


'''
Find the nearest point in the data for a specific voltage
'''


def find_nearest_point(voltage_array, response_array, target_voltage):
    # Find the index of the value closest to the target voltage in the up curve
    up_curve_indices = np.diff(voltage_array, prepend=voltage_array[0]) > 0
    if not any(up_curve_indices):
        # If no up curve points are found, use all points
        idx = (np.abs(voltage_array - target_voltage)).argmin()
    else:
        # Only consider points in the up curve
        filtered_voltage = voltage_array[up_curve_indices]
        idx = (np.abs(filtered_voltage - target_voltage)).argmin()
        # Convert back to original array index
        idx = np.where(up_curve_indices)[0][idx]

    return voltage_array[idx], response_array[idx]


'''
Generate the reference line by finding actual data points at 1.1V and 1.6V
'''


def generate_dynamic_reference_line(voltage, response):
    # Find points closest to the reference voltages
    v_start, r_start = find_nearest_point(voltage, response, REFERENCE_START_V)
    v_end, r_end = find_nearest_point(voltage, response, REFERENCE_END_V)

    # Generate reference line for all voltage points
    reference_y = np.array([calculate_line_y(v, v_start, r_start, v_end, r_end) for v in voltage])

    return reference_y, (v_start, r_start), (v_end, r_end)


'''
Find the voltage with the highest response after subtracting the reference line
'''


def find_peak_response(voltage, response):
    # Generate dynamic reference line based on the curve's actual points
    reference_y, start_point, end_point = generate_dynamic_reference_line(voltage, response)

    # Subtract reference line from response
    difference = response - reference_y

    # Find the peak in the specified voltage range
    valid_indices = (voltage >= PEAK_DETECTION_MIN) & (voltage <= PEAK_DETECTION_MAX)
    filtered_voltage = voltage[valid_indices]
    filtered_difference = difference[valid_indices]

    if len(filtered_difference) > 0:
        # Find the index of maximum absolute difference
        max_diff_index = np.argmax(np.abs(filtered_difference))
        peak_voltage = filtered_voltage[max_diff_index]

        # Find the corresponding index in the original arrays
        original_index = np.where(voltage == peak_voltage)[0][0]
        original_response = response[original_index]

        return peak_voltage, original_response, filtered_difference[max_diff_index], reference_y, start_point, end_point
    else:
        return None, None, None, reference_y, start_point, end_point



'''
Normalizes the data by finding the response at 1V for each curve.
'''


def cga_normalization(df):
    voltage = df['Applied Voltage'].values
    response = df['Detected Response'].values
    # Find the response between CGA_MIN_VOLTAGE and CGA_MAX_VOLTAGE in the up curve
    up_curve_indices = (voltage >= CGA_MIN_VOLTAGE) & (voltage <= CGA_MAX_VOLTAGE) & (
                np.diff(voltage, prepend=voltage[0]) > 0)
    # Average response between CGA_MIN_VOLTAGE and CGA_MAX_VOLTAGE in the up curve
    response_at_1v = np.mean(response[up_curve_indices])
    V_responses.append(response_at_1v)


def plot_data(i, file_path, df):
    voltage = df['Applied Voltage'].values
    response = df['Detected Response'].values

    # Subtract the corresponding response at 1V for each curve
    if (CGA_NORMALIZE):
        response -= V_responses[i]

    y_smoothed = moving_average(response, window_size=SMOOTHING_WINDOW_SIZE)
    x_smoothed = voltage[len(voltage) - len(y_smoothed):]

    # Find the peak response after subtracting the dynamic reference line
    result = find_peak_response(x_smoothed, y_smoothed)

    if result[0] is not None:
        peak_voltage, peak_response, peak_diff, reference_y, start_point, end_point = result

        print(f"File: {file_path}, Peak Response: {peak_response:.3f} uA at Voltage: {peak_voltage:.3f}")
        responses.append(peak_response)

        # Plot the curve and the peak point
        plt.plot(x_smoothed[0: len(x_smoothed) // 2], y_smoothed[0: len(y_smoothed) // 2], label=file_path,
                 color=colors[i], alpha=0.8)
        plt.scatter(peak_voltage, peak_response, color=colors[i], label=f"Peak Response {file_path}", zorder=5)

        # Plot the dynamic reference line and its endpoints
        if i < 5:  # Only plot reference lines for the first few curves to avoid clutter
            first_half_idx = len(x_smoothed) // 2
            #plt.plot(x_smoothed[0:first_half_idx], reference_y[0:first_half_idx], '--', color=colors[i], alpha=0.3)
            plt.scatter([start_point[0], end_point[0]], [start_point[1], end_point[1]], color=colors[i], marker='x',
                        alpha=0.5)

    return (x_smoothed[0: len(x_smoothed) // 2], y_smoothed[0: len(y_smoothed) // 2])


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
    for i in range(0, len(data), 3):
        chunk = data[i:i + 3]
        if len(chunk) == 3:
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

    # Plot each shifted curve and find the peak response
    for i, file_path in enumerate(file_paths):
        df = read_data(file_path)
        if df is not None:
            smoothed_responses.append(plot_data(i, file_path, df))

    plt.xlabel('Applied Voltage')
    plt.ylabel('Response uA')
    plt.title('Cyclic Voltammetry - Dynamic Baseline Peak Detection')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.ylim(-20, 200)
    plt.show()

    averages, std_devs = process_chunks(responses)
    for i, coffee in enumerate(coffee_names[:len(averages)]):
        print(f"Average of {coffee}: {averages[i]:.2f} uA, Std Dev: {std_devs[i]:.2f} uA")

    model, predictions = train_and_evaluate_model(averages, caffeine_ppm)

    mae = mean_absolute_error(caffeine_ppm, predictions)
    percent_error = (mae / np.std(caffeine_ppm)) * 100
    print(f"Percent Error: {percent_error:.2f}%")

    # Create labeled regression plot
    plt.figure(figsize=(12, 8))

    # Plot error bars
    plt.errorbar(averages, caffeine_ppm, yerr=std_devs, fmt='o', color='blue', capsize=5)

    # Add labels to each data point
    for i, coffee in enumerate(coffee_names[:len(averages)]):
        plt.annotate(coffee,
                     (averages[i], caffeine_ppm[i]),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Plot regression line
    plt.plot(averages, predictions, color='red', label='Regression Line')

    # Add prediction interval
    mse = mean_squared_error(caffeine_ppm, predictions)
    standard_error = np.sqrt(mse)
    prediction_interval = 1.96 * standard_error

    plt.fill_between(averages, predictions - prediction_interval, predictions + prediction_interval, color='red',
                     alpha=0.2, label='Prediction Interval (95%)')

    plt.xlabel('Peak Response (uA)')
    plt.ylabel('Caffeine Content (ppm)')
    plt.title('Caffeine Content vs Peak Response with Dynamic Baseline Method')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()