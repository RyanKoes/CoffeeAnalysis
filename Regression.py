import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error

SMOOTHING_WINDOW_SIZE = 25  # Size of the moving average window for smoothing
CGA_MIN_VOLTAGE = 0.7
CGA_MAX_VOLTAGE = 0.75
CGA_NORMALIZE = True

# New parameters for the reference line and peak detection
REFERENCE_START_V = 1.15  # Voltage where reference Line starts
REFERENCE_END_V = 1.5  # Voltage where reference line ends
PEAK_DETECTION_MIN = 1.15  # Minimum voltage for peak detection
PEAK_DETECTION_MAX = 1.5  # Maximum voltage for peak detection

# Training data files
file_paths = [
    'voltammetry-files/aladec1.txt',
    'voltammetry-files/aladec2.txt',
    'voltammetry-files/aladec3edge.txt',
    'voltammetry-files/alaD2p1.txt',
    'voltammetry-files/alaD2p2.txt',
    'voltammetry-files/alaD2p3.txt',
    'voltammetry-files/alaD4p1.txt',
    'voltammetry-files/alaD4p2.txt',
    'voltammetry-files/alaD4p3.txt',
    'voltammetry-files/alaD6p1.txt',
    'voltammetry-files/alaD6p2.txt',
    'voltammetry-files/alaD6p3.txt',
    'voltammetry-files/alaD8p1.txt',
    'voltammetry-files/alaD8p2.txt',
    'voltammetry-files/alaD8p3.txt',
    'voltammetry-files/alareg1.txt',
    'voltammetry-files/alareg2.txt',
    'voltammetry-files/alareg3edge.txt'
]

# Coffee names for labeling
coffee_names = [
    "Decaf",
    "200 ppm",
    "400 ppm",
    "600 ppm",
    "800 ppm",
    "Columbia Reg"
]

caffeine_ppm = [52, 320, 580, 720, 920, 820]

colors = [
    '#ff4d4d', '#ff4d4d', '#ff4d4d',
    '#ffb84d', '#ffb84d', '#ffb84d',
    '#ffff4d', '#ffff4d', '#ffff4d',
    '#80ff4d', '#80ff4d', '#80ff4d',
    '#4dffdb', '#4dffdb', '#4dffdb',
    '#4da6ff', '#4da6ff', '#4da6ff'
]

# New files to predict caffeine content (empty by default, to be filled by user)
predict_file_paths = ['voltammetry-files/A1.txt',
                      'voltammetry-files/A2.txt',
                      'voltammetry-files/A3.txt',
                      'voltammetry-files/B1.txt',
                      'voltammetry-files/B2.txt',
                      'voltammetry-files/B3.txt',
                      'voltammetry-files/C1.txt',
                      'voltammetry-files/C2.txt',
                      'voltammetry-files/C3.txt',
                      'voltammetry-files/D1.txt',
                      'voltammetry-files/D2.txt',
                      'voltammetry-files/D3.txt',
                      'voltammetry-files/E1.txt',
                      'voltammetry-files/E2.txt',
                      'voltammetry-files/E3.txt',
                      'voltammetry-files/F1.txt',
                      'voltammetry-files/F2.txt',
                      'voltammetry-files/F3.txt',
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
                      'voltammetry-files/N3.txt',
                      'voltammetry-files/dunkin1.txt',
                      'voltammetry-files/dunkin2.txt',
                      'voltammetry-files/dunkin3.txt',
                      'voltammetry-files/mccafe1.txt',
                      'voltammetry-files/mccafe2.txt',
                      'voltammetry-files/mccafe3.txt',
                      'voltammetry-files/sheetz1.txt',
                      'voltammetry-files/sheetz2.txt',
                      'voltammetry-files/sheetz3.txt',
                      'voltammetry-files/sbpike1.txt',
                      'voltammetry-files/sbpike2.txt',
                      'voltammetry-files/sbpike3.txt'
                      ]

predict_names = ["FRC Decaf Colombian, med roast IH",
                 "FRC Sugarcame Decaf Colombian, med roast IH",
                 "FRC Swiss Water Decaf Colombian, med roast IH",
                 "FRC Mexican medium roast",
                 "FRC Sumatra medium roast",
                 "FRC Colombia medium roast",
                 "FRC Kenya AA, medium roast IH",
                 "FRC Ethiopia Yirgacheffe, light roast IH",
                 "FRC ROBUSTA Brazil, medium roast IH",
                 "FRC Brazil Cerrado, light roast IH",
                 "FRC Brazil Cerrado, medium roast IH",
                 "FRC Brazil Cerrado, dark roast IH",
                 "FRC Brazil Cerrado, medium roast IH- High BR",
                 "FRC Ethiopia Yirgacheffe, light roast IH- High BR",
                 "Dunkin Original Blend, 5/22 8am",
                 "McDonald's Regular Coffee, 5/22 8am",
                 "Sheetz Classic Coffee (100% arabica), 12oz refill  5/22 8am",
                 "Starbucks Pike Place Roast, 5/22 8am"]

# ACTUAL CAFFEINE VALUES FOR PREDICTION SAMPLES
# Add your actual caffeine values here (in ppm) - one value per group of 3 files
# These should correspond to the predict_names groups
actual_caffeine_ppm = [
    75,  # FRC Decaf Colombian
    60,  # FRC Sugarcame Decaf Colombian
    40,  # FRC Swiss Water Decaf Colombian
    820,  # FRC Mexican medium roast
    830,  # FRC Sumatra medium roast
    770,  # FRC Colombia medium roast
    840,  # FRC Kenya AA
    770,  # FRC Ethiopia Yirgacheffe
    1220,  # FRC ROBUSTA Brazil
    870,  # FRC Brazil Cerrado light
    850,  # FRC Brazil Cerrado medium
    860,  # FRC Brazil Cerrado dark
    1100,  # FRC Brazil Cerrado medium High BR
    1060,  # FRC Ethiopia Yirgacheffe High BR
    530,  # Dunkin Original Blend
    520,  # McDonald's Regular Coffee
    420,  # Sheetz Classic Coffee
    680,  # Starbucks Pike Place Roast
]

predict_colors = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # yellow-green
    "#17becf",  # cyan
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
    "#98df8a",  # light green
    "#ff9896",  # light red
    "#c5b0d5",  # light purple
    "#c49c94",  # light brown
    "#f7b6d2",  # light pink
    "#c7c7c7"  # light gray
]

data = []
responses = []
V_responses = []
prediction_responses = []
prediction_V_responses = []

# Store normalized data frames
normalized_training_dfs = []
normalized_prediction_dfs = []

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
    print(
        f"    Up curve filtering: {np.sum(up_curve_indices)} data points found from {len(voltage_array)} total points")

    if not any(up_curve_indices):
        # If no up curve points are found, use all points
        idx = (np.abs(voltage_array - target_voltage)).argmin()
        print("    No up curve points found, using all data points for nearest point search")
    else:
        # Only consider points in the up curve
        filtered_voltage = voltage_array[up_curve_indices]
        idx = (np.abs(filtered_voltage - target_voltage)).argmin()
        # Convert back to original array index
        idx = np.where(up_curve_indices)[0][idx]
        print(f"    Nearest point search completed on {len(filtered_voltage)} up curve points")

    return voltage_array[idx], response_array[idx]


'''
Generate the reference line by finding actual data points at specific voltages
'''


def generate_dynamic_reference_line(voltage, response):
    print(f"  Generating reference line from {len(voltage)} voltage points")
    # Find points closest to the reference voltages
    v_start, r_start = find_nearest_point(voltage, response, REFERENCE_START_V)
    v_end, r_end = find_nearest_point(voltage, response, REFERENCE_END_V)

    # Generate reference line for all voltage points
    reference_y = np.array([calculate_line_y(v, v_start, r_start, v_end, r_end) for v in voltage])
    print(f"  Reference line generated for {len(reference_y)} points")

    return reference_y, (v_start, r_start), (v_end, r_end)


'''
Find the voltage with the highest response after subtracting the reference line
'''


def find_peak_response(voltage, response):
    print(f"  Peak detection starting with {len(voltage)} data points")
    # Generate dynamic reference line based on the curve's actual points
    reference_y, start_point, end_point = generate_dynamic_reference_line(voltage, response)

    # Subtract reference line from response
    difference = response - reference_y
    # print(f"  Calculated difference array with {len(difference)} points")

    # # Find the peak in the specified voltage range
    # valid_indices = (voltage >= PEAK_DETECTION_MIN) & (voltage <= PEAK_DETECTION_MAX)
    # filtered_voltage = voltage[valid_indices]
    # filtered_difference = difference[valid_indices]
    # print(f"  Peak detection voltage range filtering: {len(filtered_voltage)} points from {len(voltage)} total points")



    x_start = np.where(voltage >= PEAK_DETECTION_MIN)[0][0]
    x_end = x_start + np.where(voltage[x_start:] > PEAK_DETECTION_MAX)[0][0]

    print(f"Start index: {x_start}, End index: {x_end}")

    filtered_voltage = voltage[x_start:x_end]
    filtered_difference = difference[x_start:x_end]


    if len(filtered_difference) > 0:
        # Find the index of maximum absolute difference
        max_diff_index = np.argmax(np.abs(filtered_difference))
        peak_voltage = filtered_voltage[max_diff_index]

        # Find the corresponding index in the original arrays
        original_index = np.where(voltage == peak_voltage)[0][0]
        original_response = response[original_index]
        print(f"  Peak found at index {original_index} in original array")

        return peak_voltage, original_response, filtered_difference[max_diff_index], reference_y, start_point, end_point
    else:
        print("  No valid points found in peak detection range")
        return None, None, None, reference_y, start_point, end_point


'''
Reads the data from the specified file path and returns a DataFrame.
'''


def read_data(file_path):
    try:
        df = pd.read_csv(file_path, delimiter=',', header=None, names=['Time', 'Applied Voltage', 'Detected Response'])
        original_length = len(df)
        print(f"Read {original_length} data points from {file_path}")

        df = df[(df['Applied Voltage'] >= 0.0)]
        filtered_length = len(df)
        print(
            f"After voltage filtering (>= 0.0V): {filtered_length} data points remaining (removed {original_length - filtered_length} points)")

        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


'''
Performs CGA normalization and returns normalized response values
'''


def cga_normalization(df, is_prediction=False):
    print(f"CGA normalization starting with {len(df)} data points")
    voltage = df['Applied Voltage'].values
    response = df['Detected Response'].values

    # Find the response between CGA_MIN_VOLTAGE and CGA_MAX_VOLTAGE in the up curve
    up_curve_indices = (voltage >= CGA_MIN_VOLTAGE) & (voltage <= CGA_MAX_VOLTAGE) & (
            np.diff(voltage, prepend=voltage[0]) > 0)
    cga_points = np.sum(up_curve_indices)
    print(f"CGA normalization range ({CGA_MIN_VOLTAGE}V to {CGA_MAX_VOLTAGE}V): {cga_points} points found")

    # Average response between CGA_MIN_VOLTAGE and CGA_MAX_VOLTAGE in the up curve
    response_at_1v = np.mean(response[up_curve_indices])

    if is_prediction:
        prediction_V_responses.append(response_at_1v)
    else:
        V_responses.append(response_at_1v)

    # Create a normalized dataframe with the response adjusted
    df_normalized = df.copy()
    if CGA_NORMALIZE:
        df_normalized['Detected Response'] = df['Detected Response'] - response_at_1v

    print(f"CGA normalization completed, output dataframe has {len(df_normalized)} data points")
    return df_normalized, response_at_1v


'''
Plots the data and finds peak responses
'''


def plot_data(i, file_path, df, is_prediction=False, color_index=0):
    print(f"\nPlotting data for {file_path}:")
    print(f"Input dataframe has {len(df)} data points")

    voltage = df['Applied Voltage'].values
    response = df['Detected Response'].values  # Response is already normalized

    y_smoothed = moving_average(response, window_size=SMOOTHING_WINDOW_SIZE)
    x_smoothed = voltage[len(voltage) - len(y_smoothed):]
    print(
        f"After moving average smoothing (window={SMOOTHING_WINDOW_SIZE}): {len(y_smoothed)} points from {len(response)} original points")

    # Find the peak response after subtracting the dynamic reference line
    result = find_peak_response(x_smoothed, y_smoothed)

    peak_response = None

    if result[0] is not None:
        peak_voltage, peak_response, peak_diff, reference_y, start_point, end_point = result

        # Choose color based on whether this is a prediction or training data
        if is_prediction:
            color = predict_colors[color_index % len(predict_colors)]
        else:
            color = colors[i]
            print(f"File: {file_path}, Peak Response: {peak_response:.3f} uA at Voltage: {peak_voltage:.3f}")
            responses.append(peak_response)

        # Plot the curve and the peak point
        first_half_points = len(x_smoothed) // 2
        x_plot = x_smoothed[0:first_half_points]
        y_plot = y_smoothed[0:first_half_points]
        print(f"Plotting first half of data: {len(x_plot)} points")

        plt.plot(x_plot, y_plot, label=file_path, color=color, alpha=0.8)
        plt.scatter(peak_voltage, peak_response, color=color,
                    label=f"{'Prediction' if is_prediction else 'Peak Response'} {file_path}", zorder=3)

        # Plot the dynamic reference line and its endpoints
        if (not is_prediction and i < 3) or (
                is_prediction and color_index < 2):  # Only plot reference lines for a few curves to avoid clutter
            plt.scatter([start_point[0], end_point[0]], [start_point[1], end_point[1]], color=color, marker='x',
                        alpha=0.5)

    if is_prediction and peak_response is not None:
        prediction_responses.append(peak_response)

    return (x_smoothed[0: len(x_smoothed) // 2], y_smoothed[0: len(y_smoothed) // 2])


'''
Calculates the moving average of a given array y with a specified window size.
'''


def moving_average(y, window_size=5):
    result = np.convolve(y, np.ones(window_size) / window_size, mode='valid')
    print(f"Moving average: input {len(y)} points, output {len(result)} points (window size={window_size})")
    return result


'''
Regression model
'''


def train_and_evaluate_model(averages, ground_truth):
    print(
        f"\nTraining regression model with {len(averages)} average response values and {len(ground_truth)} ground truth values")

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
    print(f"Mean Absolute Error: {mae:.4f}")

    return model, predictions


'''
Processes the data in chunks of 3 and calculates the averages and standard deviations.
'''


def process_chunks(data):
    print(f"\nProcessing {len(data)} response values in chunks of 3")
    averages = []
    std_devs = []

    # Process data in chunks of 3
    for i in range(0, len(data), 3):
        chunk = data[i:i + 3]
        if len(chunk) == 3:
            avg = statistics.mean(chunk)
            std = statistics.stdev(chunk)
            averages.append(avg)
            std_devs.append(std)

    print(f"Created {len(averages)} averaged data points from {len(data)} individual responses")
    return averages, std_devs


'''
Processes prediction data in groups of 3 files and makes caffeine content predictions
'''


def process_predictions(model, prediction_data):
    print(f"\nProcessing {len(prediction_data)} prediction responses in groups of 3")
    prediction_averages = []
    prediction_std_devs = []

    # Process prediction data in chunks of 3
    for i in range(0, len(prediction_data), 3):
        chunk = prediction_data[i:i + 3]
        if len(chunk) == 3:
            avg = statistics.mean(chunk)
            std = statistics.stdev(chunk)
            prediction_averages.append(avg)
            prediction_std_devs.append(std)

    print(
        f"Created {len(prediction_averages)} averaged prediction points from {len(prediction_data)} individual responses")

    # Make predictions using the trained model
    if prediction_averages:
        X_pred = np.array(prediction_averages).reshape(-1, 1)
        predicted_caffeine = model.predict(X_pred)
        print(f"Generated {len(predicted_caffeine)} caffeine predictions")

        # Print prediction results
        print("\nPrediction Results:")
        for i, (name, avg, std, pred) in enumerate(zip(predict_names, prediction_averages,
                                                       prediction_std_devs, predicted_caffeine)):
            print(
                f"Sample {name}: Avg Response: {avg:.2f} uA, Std Dev: {std:.2f} uA, Predicted Caffeine: {pred:.2f} ppm")

        return prediction_averages, prediction_std_devs, predicted_caffeine

    return [], [], []


'''
Evaluate prediction accuracy against actual values
'''


def evaluate_prediction_accuracy(predicted_values, actual_values, sample_names):
    if len(predicted_values) == 0 or len(actual_values) == 0:
        print("No prediction data available for accuracy evaluation.")
        return

    if len(predicted_values) != len(actual_values):
        print(
            f"Warning: Number of predictions ({len(predicted_values)}) doesn't match actual values ({len(actual_values)})")
        return

    print(
        f"\nEvaluating prediction accuracy with {len(predicted_values)} predictions vs {len(actual_values)} actual values")

    print("\n" + "=" * 80)
    print("PREDICTION ACCURACY EVALUATION")
    print("=" * 80)

    # Calculate metrics
    mae_pred = mean_absolute_error(actual_values, predicted_values)
    mse_pred = mean_squared_error(actual_values, predicted_values)
    r2_pred = r2_score(actual_values, predicted_values)

    # Calculate percent errors
    percent_errors = []
    for actual, predicted in zip(actual_values, predicted_values):
        if actual != 0:
            pe = abs((predicted - actual) / actual) * 100
            percent_errors.append(pe)

    mean_percent_error = np.mean(percent_errors) if percent_errors else 0

    print(f"Prediction Accuracy Metrics:")
    print(f"  R² Score: {r2_pred:.4f}")
    print(f"  Mean Absolute Error: {mae_pred:.2f} ppm")
    print(f"  Mean Squared Error: {mse_pred:.2f} ppm²")
    print(f"  Root Mean Squared Error: {np.sqrt(mse_pred):.2f} ppm")
    print(f"  Mean Percent Error: {mean_percent_error:.2f}%")

    print(f"\nDetailed Comparison:")
    print(f"{'Sample':<50} {'Actual':<10} {'Predicted':<10} {'Error':<10} {'% Error':<10}")
    print("-" * 90)

    for i, (name, actual, predicted) in enumerate(zip(sample_names, actual_values, predicted_values)):
        error = predicted - actual
        percent_error = abs(error / actual) * 100 if actual != 0 else 0
        print(f"{name:<50} {actual:<10.1f} {predicted:<10.1f} {error:<10.1f} {percent_error:<10.1f}")

    return mae_pred, mse_pred, r2_pred, mean_percent_error


'''
Create comprehensive accuracy visualization
'''


def plot_prediction_accuracy(predicted_values, actual_values, sample_names, pred_averages, pred_std_devs):
    if len(predicted_values) == 0 or len(actual_values) == 0:
        return

    print(f"\nCreating prediction accuracy plots with {len(predicted_values)} predictions")

    # Ensure all arrays have the same length
    min_length = min(len(predicted_values), len(actual_values), len(sample_names))
    predicted_values = predicted_values[:min_length]
    actual_values = actual_values[:min_length]
    sample_names = sample_names[:min_length]
    print(f"Plot data trimmed to {min_length} points for consistency")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Predicted vs Actual scatter plot
    ax1.scatter(actual_values, predicted_values, color='blue', alpha=0.7, s=100)

    # Add perfect prediction line
    min_val = min(min(actual_values), min(predicted_values))
    max_val = max(max(actual_values), max(predicted_values))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')

    # Add sample labels
    for i, name in enumerate(sample_names):
        ax1.annotate(f"{i + 1}", (actual_values[i], predicted_values[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax1.set_xlabel('Actual Caffeine Content (ppm)')
    ax1.set_ylabel('Predicted Caffeine Content (ppm)')
    ax1.set_title('Predicted vs Actual Caffeine Content')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Calculate R²
    r2_pred = r2_score(actual_values, predicted_values)
    ax1.text(0.05, 0.95, f'R² = {r2_pred:.4f}', transform=ax1.transAxes,
             bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))

    # 2. Residuals plot
    residuals = np.array(predicted_values) - np.array(actual_values)
    ax2.scatter(actual_values, residuals, color='red', alpha=0.7, s=100)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.8)
    ax2.set_xlabel('Actual Caffeine Content (ppm)')
    ax2.set_ylabel('Residuals (Predicted - Actual)')
    ax2.set_title('Residuals Plot')
    ax2.grid(True, alpha=0.3)

    # 3. Error bar plot - Fixed to ensure matching dimensions
    sample_indices = range(len(sample_names))  # This will now match the length
    width = 0.35

    bars1 = ax3.bar([i - width / 2 for i in sample_indices], actual_values, width,
                    label='Actual', alpha=0.7, color='green')
    bars2 = ax3.bar([i + width / 2 for i in sample_indices], predicted_values, width,
                    label='Predicted', alpha=0.7, color='blue')

    ax3.set_xlabel('Sample Number')
    ax3.set_ylabel('Caffeine Content (ppm)')
    ax3.set_title('Actual vs Predicted Caffeine Content by Sample')
    ax3.set_xticks(sample_indices)
    ax3.set_xticklabels([f"{i + 1}" for i in sample_indices])
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Percent error plot
    percent_errors = []
    for actual, predicted in zip(actual_values, predicted_values):
        if actual != 0:
            pe = abs((predicted - actual) / actual) * 100
            percent_errors.append(pe)
        else:
            percent_errors.append(0)

    bars = ax4.bar(sample_indices, percent_errors, color='orange', alpha=0.7)
    ax4.set_xlabel('Sample Number')
    ax4.set_ylabel('Absolute Percent Error (%)')
    ax4.set_title('Prediction Accuracy by Sample')
    ax4.set_xticks(sample_indices)
    ax4.set_xticklabels([f"{i + 1}" for i in sample_indices])
    ax4.grid(True, alpha=0.3)

    # Add mean error line
    mean_error = np.mean(percent_errors)
    ax4.axhline(y=mean_error, color='red', linestyle='--', alpha=0.8,
                label=f'Mean Error: {mean_error:.1f}%')
    ax4.legend()

    plt.tight_layout()
    plt.show()

    # Print sample legend
    print(f"\nSample Legend:")
    for i, name in enumerate(sample_names):
        print(f"{i + 1:2d}. {name}")


'''
Function to set up and add prediction file paths
'''


def add_prediction_files(file_groups, names):
    global predict_file_paths, predict_names
    predict_file_paths = file_groups
    predict_names = names


'''
Function to set actual caffeine values for predictions
'''


def set_actual_caffeine_values(values):
    global actual_caffeine_ppm
    actual_caffeine_ppm = values


if __name__ == '__main__':
    # Example of how to add prediction files (can be modified by user)
    # Uncomment and modify these lines to add your prediction files

    plt.figure(figsize=(10, 6))

    # First pass: Read all data and perform CGA normalization before plotting
    # Process training data for normalization
    for i, file_path in enumerate(file_paths):
        df = read_data(file_path)
        if df is not None:
            df_normalized, _ = cga_normalization(df)
            normalized_training_dfs.append((i, file_path, df_normalized))

    # Process prediction data for normalization
    for i, file_path in enumerate(predict_file_paths):
        df = read_data(file_path)
        if df is not None:
            df_normalized, _ = cga_normalization(df, is_prediction=True)
            normalized_prediction_dfs.append((i, file_path, df_normalized))

    # Second pass: Plot each normalized curve and find the peak response for training data
    smoothed_responses = []
    for i, file_path, df_normalized in normalized_training_dfs:
        smoothed_responses.append(plot_data(i, file_path, df_normalized))

    # Plot each normalized curve and find the peak response for prediction data
    prediction_smoothed = []
    for i, file_path, df_normalized in normalized_prediction_dfs:
        group_idx = i // 3  # Determine which group this file belongs to
        prediction_smoothed.append(plot_data(i, file_path, df_normalized, is_prediction=True, color_index=group_idx))

    plt.xlabel('Applied Voltage')
    plt.ylabel('Response μA')
    plt.title('Cyclic Voltammetry - Dynamic Baseline Peak Detection')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.ylim(-20, 200)
    plt.show()

    # Process training data
    averages, std_devs = process_chunks(responses)
    for i, coffee in enumerate(coffee_names[:len(averages)]):
        print(f"Average of {coffee}: {averages[i]:.2f} μA, Std Dev: {std_devs[i]:.2f} μA")

    # Train the model and make predictions on training data
    model, predictions = train_and_evaluate_model(averages, caffeine_ppm)

    # Calculate percent error
    mae = mean_absolute_error(caffeine_ppm, predictions)
    percent_error = (mae / np.mean(caffeine_ppm)) * 100
    print(f"Percent Error: {percent_error:.2f}%")

    # Process prediction data and make predictions
    pred_averages, pred_std_devs, pred_caffeine = process_predictions(model, prediction_responses)

    # Evaluate prediction accuracy if actual values are provided
    if actual_caffeine_ppm and len(actual_caffeine_ppm) > 0:
        # Group predict_names for display (every 3rd name represents a group)
        grouped_names = []
        for i in range(0, len(predict_names), 3):
            if i // 3 < len(actual_caffeine_ppm):
                grouped_names.append(predict_names[i // 3])

        # Evaluate accuracy
        mae_pred, mse_pred, r2_pred, mean_pe = evaluate_prediction_accuracy(
            pred_caffeine[:len(actual_caffeine_ppm)],
            actual_caffeine_ppm[:len(pred_caffeine)],
            grouped_names[:len(pred_caffeine)]
        )

        # Create visualization
        plot_prediction_accuracy(
            pred_caffeine[:len(actual_caffeine_ppm)],
            actual_caffeine_ppm[:len(pred_caffeine)],
            grouped_names[:len(pred_caffeine)],
            pred_averages,
            pred_std_devs
        )
    else:
        print("\nNo actual caffeine values provided. To evaluate prediction accuracy,")
        print("please add actual caffeine values to the 'actual_caffeine_ppm' list.")
        print("Example:")
        print("actual_caffeine_ppm = [15, 12, 18, 285, 310, ...]  # One value per sample group")

    # Create labeled regression plot
    plt.figure(figsize=(12, 8))

    # Plot training data points with error bars
    plt.errorbar(averages, caffeine_ppm, yerr=std_devs, fmt='o', color='blue', capsize=5,
                 label='Training Data')

    # Add labels to training data points
    for i, coffee in enumerate(coffee_names[:len(averages)]):
        plt.annotate(coffee,
                     (averages[i], caffeine_ppm[i]),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Plot regression line
    min_x = min(averages) - 3 if averages else 0
    max_x = max(averages) + 3 if averages else 100
    x_range = np.linspace(min_x, max_x, 100)
    x_range_reshaped = x_range.reshape(-1, 1)
    regression_line = model.predict(x_range_reshaped)
    plt.plot(x_range, regression_line, color='red', label='Regression Line')

    # Add prediction interval
    mse = mean_squared_error(caffeine_ppm, predictions)
    standard_error = np.sqrt(mse)
    prediction_interval = 1.96 * standard_error

    plt.fill_between(x_range, regression_line - prediction_interval, regression_line + prediction_interval,
                     color='red', alpha=0.2, label='Prediction Interval (95%)')

    # Plot prediction data points if available
    if pred_averages:
        plt.errorbar(pred_averages, pred_caffeine, yerr=pred_std_devs, fmt='s', color='green',
                     capsize=5, label='Predictions')

        # Add labels to prediction data points (use grouped names)
        grouped_predict_names = []
        for i in range(0, len(predict_names), 3):
            if i // 3 < len(pred_averages):
                grouped_predict_names.append(predict_names[i // 3])

        for i, name in enumerate(grouped_predict_names):
            if i < len(pred_averages):
                plt.annotate(name,
                             (pred_averages[i], pred_caffeine[i]),
                             xytext=(5, 5),
                             textcoords='offset points',
                             fontsize=9,
                             bbox=dict(boxstyle="round4,pad=0.3", fc="#e6ffe6", ec="green", alpha=0.8))

    # Plot actual values if available
    if actual_caffeine_ppm and len(actual_caffeine_ppm) > 0 and pred_averages:
        actual_to_plot = actual_caffeine_ppm[:len(pred_averages)]
        plt.scatter(pred_averages[:len(actual_to_plot)], actual_to_plot,
                    color='orange', marker='x', s=100, label='Actual Values', zorder=5)

    plt.xlabel('Peak Response (μA)')
    plt.ylabel('Caffeine Content (ppm)')
    plt.title('Caffeine Content vs Peak Response with Dynamic Baseline Method')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Summary statistics
    if actual_caffeine_ppm and len(actual_caffeine_ppm) > 0 and pred_caffeine:
        print(f"\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Training Data Performance:")
        print(f"  - R² Score: {r2_score(caffeine_ppm, predictions):.4f}")
        print(f"  - MAE: {mean_absolute_error(caffeine_ppm, predictions):.2f} ppm")

        if len(pred_caffeine) > 0 and len(actual_caffeine_ppm) > 0:
            actual_subset = actual_caffeine_ppm[:len(pred_caffeine)]
            print(f"\nPrediction Performance:")
            print(f"  - R² Score: {r2_score(actual_subset, pred_caffeine):.4f}")
            print(f"  - MAE: {mean_absolute_error(actual_subset, pred_caffeine):.2f} ppm")
            print(
                f"  - Mean % Error: {np.mean([abs((p - a) / a) * 100 for p, a in zip(pred_caffeine, actual_subset) if a != 0]):.2f}%")