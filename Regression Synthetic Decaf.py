import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

SMOOTHING_WINDOW_SIZE = 25  # Size of the moving average window for smoothing
CGA_MIN_VOLTAGE = 0.7
CGA_MAX_VOLTAGE = 0.75
CGA_NORMALIZE = True

# New parameters for the reference line and peak detection
REFERENCE_START_V = 1.15  # Voltage where reference Line starts
REFERENCE_END_V = 1.5  # Voltage where reference line ends
PEAK_DETECTION_MIN = 1.15  # Minimum voltage for peak detection
PEAK_DETECTION_MAX = 1.5  # Maximum voltage for peak detection

# NEW: Synthetic curve generation parameters
SYNTHETIC_CURVE_START = 1.15  # Start voltage for synthetic curve generation
SYNTHETIC_CURVE_END = 1.5  # End voltage for synthetic curve generation
LOW_DATA_START = 1.0  # Start voltage for low-end data fitting
LOW_DATA_END = 1.15  # End voltage for low-end data fitting
HIGH_DATA_START = 1.5  # Start voltage for high-end data fitting
HIGH_DATA_END = 1.65  # End voltage for high-end data fitting

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

# New files to predict caffeine content
predict_file_paths = ['voltammetry-files/A1.txt',
                      'voltammetry-files/A2.txt',
                      'voltammetry-files/A3edge.txt',
                      'voltammetry-files/B1.txt',
                      'voltammetry-files/B2.txt',
                      'voltammetry-files/B3edge.txt',
                      'voltammetry-files/C1.txt',
                      'voltammetry-files/C2.txt',
                      'voltammetry-files/C3edge.txt',
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

# Global variables for storing data
data = []
responses = []
V_responses = []
prediction_responses = []
prediction_V_responses = []
normalized_training_dfs = []
normalized_prediction_dfs = []


def calculate_line_y(x, x1, y1, x2, y2):
    """Calculate the y-value on a line between two points for a given x-value"""
    m = (y2 - y1) / (x2 - x1)  # slope
    b = y1 - m * x1  # y-intercept
    return m * x + b


def find_nearest_point(voltage_array, response_array, target_voltage):
    """Find the nearest point in the data for a specific voltage"""
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


def generate_synthetic_exponential_curve(voltage, response):
    """Generate synthetic exponential curve using data from low and high voltage ranges"""
    # Extract data from low voltage range (0V to SYNTHETIC_CURVE_START)
    low_mask = (voltage >= LOW_DATA_START) & (voltage <= LOW_DATA_END)
    low_v = voltage[low_mask]
    low_r = response[low_mask]

    # Extract data from high voltage range (SYNTHETIC_CURVE_END to 2V)
    high_mask = (voltage >= HIGH_DATA_START) & (voltage <= HIGH_DATA_END)
    high_v = voltage[high_mask]
    high_r = response[high_mask]

    # Combine the data for exponential fitting
    combined_v = np.concatenate([low_v, high_v])
    combined_r = np.concatenate([low_r, high_r])

    if len(combined_v) < 3:  # Need at least 3 points for fitting
        print("Warning: Not enough data points for exponential fitting")
        return np.zeros_like(voltage)

    try:
        # Fit exponential curve: y = a * exp(b * x) + c
        def exponential_func(x, a, b, c):
            return a * np.exp(b * x) + c

        # Initial guess for parameters
        p0 = [1.0, 0.1, np.min(combined_r)]

        # Fit the exponential function
        popt, _ = curve_fit(
            exponential_func,
            combined_v,
            combined_r,
            p0=p0,
            bounds=([0, -5, -np.inf], [np.inf, 5, np.inf]),  # limit b to between -5 and 5
            maxfev=5000
        )

        # Generate synthetic curve for the entire voltage range
        synthetic_response = exponential_func(voltage, *popt)

        return synthetic_response

    except Exception as e:
        print(f"Warning: Exponential fitting failed: {e}")
        # Fallback to linear interpolation between boundary points
        start_v, start_r = find_nearest_point(voltage, response, SYNTHETIC_CURVE_START)
        end_v, end_r = find_nearest_point(voltage, response, SYNTHETIC_CURVE_END)

        synthetic_response = np.array([calculate_line_y(v, start_v, start_r, end_v, end_r) for v in voltage])
        return synthetic_response


def generate_dynamic_reference_line(voltage, response):
    """Generate the reference line by finding actual data points at specific voltages"""
    # Find points closest to the reference voltages
    v_start, r_start = find_nearest_point(voltage, response, REFERENCE_START_V)
    v_end, r_end = find_nearest_point(voltage, response, REFERENCE_END_V)

    # Generate reference line for all voltage points
    reference_y = np.array([calculate_line_y(v, v_start, r_start, v_end, r_end) for v in voltage])

    return reference_y, (v_start, r_start), (v_end, r_end)


def find_peak_response(voltage, response):
    """Find the voltage with the highest response after subtracting the synthetic curve"""
    # Generate synthetic exponential curve
    synthetic_curve = generate_synthetic_exponential_curve(voltage, response)

    # In the synthetic range, use synthetic curve; outside use original dynamic reference line
    reference_y, start_point, end_point = generate_dynamic_reference_line(voltage, response)

    # Create mask for synthetic curve region
    synthetic_mask = (voltage >= SYNTHETIC_CURVE_START) & (voltage <= SYNTHETIC_CURVE_END)

    # Use synthetic curve in the specified range, reference line elsewhere
    baseline = reference_y.copy()
    baseline[synthetic_mask] = synthetic_curve[synthetic_mask]

    # Subtract baseline from response
    difference = response - baseline

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

        return peak_voltage, original_response, filtered_difference[
            max_diff_index], baseline, start_point, end_point, synthetic_curve
    else:
        return None, None, None, baseline, start_point, end_point, synthetic_curve


def read_data(file_path):
    """Reads the data from the specified file path and returns a DataFrame."""
    try:
        df = pd.read_csv(file_path, delimiter=',', header=None, names=['Time', 'Applied Voltage', 'Detected Response'])
        df = df[(df['Applied Voltage'] >= 0.0)]
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def cga_normalization(df, is_prediction=False):
    """Performs CGA normalization and returns normalized response values"""
    voltage = df['Applied Voltage'].values
    response = df['Detected Response'].values
    # Find the response between CGA_MIN_VOLTAGE and CGA_MAX_VOLTAGE in the up curve
    up_curve_indices = (voltage >= CGA_MIN_VOLTAGE) & (voltage <= CGA_MAX_VOLTAGE) & (
            np.diff(voltage, prepend=voltage[0]) > 0)
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

    return df_normalized, response_at_1v


def plot_data(i, file_path, df, is_prediction=False, color_index=0):
    """Plots the data and finds peak responses using synthetic curve"""
    voltage = df['Applied Voltage'].values
    response = df['Detected Response'].values  # Response is already normalized

    y_smoothed = moving_average(response, window_size=SMOOTHING_WINDOW_SIZE)
    x_smoothed = voltage[len(voltage) - len(y_smoothed):]

    # Find the peak response after subtracting the synthetic curve
    result = find_peak_response(x_smoothed, y_smoothed)

    peak_response = None

    if result[0] is not None:
        peak_voltage, peak_response, peak_diff, baseline, start_point, end_point, synthetic_curve = result

        # Choose color based on whether this is a prediction or training data
        if is_prediction:
            color = predict_colors[color_index % len(predict_colors)]
        else:
            color = colors[i]
            print(f"File: {file_path}, Peak Response: {peak_response:.3f} uA at Voltage: {peak_voltage:.3f}")
            # Use the difference (peak_diff) as the response for regression
            responses.append(peak_diff)

        # Plot the curve and the peak point
        plt.plot(x_smoothed[0: len(x_smoothed) // 2], y_smoothed[0: len(y_smoothed) // 2],
                 label=file_path, color=color, alpha=0.8)
        plt.scatter(peak_voltage, peak_response, color=color,
                    label=f"{'Prediction' if is_prediction else 'Peak Response'} {file_path}", zorder=3)

        # Plot the synthetic curve and baseline for visualization (first few curves only)
        if (not is_prediction and i < 2) or (is_prediction and color_index < 1):
            first_half_idx = len(x_smoothed) // 2
            synthetic_range_mask = (x_smoothed[:first_half_idx] >= SYNTHETIC_CURVE_START) & \
                                   (x_smoothed[:first_half_idx] <= SYNTHETIC_CURVE_END)

            if np.any(synthetic_range_mask):
                plt.plot(x_smoothed[:first_half_idx][synthetic_range_mask],
                         synthetic_curve[:first_half_idx][synthetic_range_mask],
                         color=color, linestyle='--', alpha=0.7, linewidth=2,
                         label=f'Synthetic Curve {file_path}')

            # Plot reference line endpoints
            plt.scatter([start_point[0], end_point[0]], [start_point[1], end_point[1]],
                        color=color, marker='x', alpha=0.5)

    if is_prediction and result[0] is not None:
        # Use the difference for predictions as well
        prediction_responses.append(peak_diff)

    return (x_smoothed[0: len(x_smoothed) // 2], y_smoothed[0: len(y_smoothed) // 2])


def moving_average(y, window_size=5):
    """Calculates the moving average of a given array y with a specified window size."""
    return np.convolve(y, np.ones(window_size) / window_size, mode='valid')


def train_and_evaluate_model(averages, ground_truth):
    """Trains and evaluates the regression model"""
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


def process_chunks(data):
    """Processes the data in chunks of 3 and calculates the averages and standard deviations."""
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

    return averages, std_devs


def process_predictions(model, prediction_data):
    """Processes prediction data in groups of 3 files and makes caffeine content predictions"""
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

    # Make predictions using the trained model
    if prediction_averages:
        X_pred = np.array(prediction_averages).reshape(-1, 1)
        predicted_caffeine = model.predict(X_pred)

        # Print prediction results
        print("\nPrediction Results:")
        for i, (name, avg, std, pred) in enumerate(zip(predict_names, prediction_averages,
                                                       prediction_std_devs, predicted_caffeine)):
            print(
                f"Sample {name}: Avg Response: {avg:.2f} uA, Std Dev: {std:.2f} uA, Predicted Caffeine: {pred:.2f} ppm")

        return prediction_averages, prediction_std_devs, predicted_caffeine

    return [], [], []


def evaluate_prediction_accuracy(predicted_values, actual_values, sample_names):
    """Evaluate prediction accuracy against actual values"""
    if len(predicted_values) == 0 or len(actual_values) == 0:
        print("No prediction data available for accuracy evaluation.")
        return

    if len(predicted_values) != len(actual_values):
        print(
            f"Warning: Number of predictions ({len(predicted_values)}) doesn't match actual values ({len(actual_values)})")
        return

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


def plot_regression_line(model, training_averages, caffeine_ppm, pred_averages, predicted_caffeine, coffee_names,
                         predict_names):
    """Plot regression line with training and prediction points, including prediction intervals"""
    plt.figure(figsize=(12, 8))

    # Create arrays for plotting
    X_train = np.array(training_averages)
    y_train = np.array(caffeine_ppm)

    # Plot training points
    plt.scatter(X_train, y_train, color='blue', s=100, alpha=0.7, label='Training Data', zorder=5)

    # Add labels for training points
    for i, (x, y, name) in enumerate(zip(X_train, y_train, coffee_names)):
        plt.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points',
                     fontsize=9, alpha=0.8, ha='left')

    # Plot prediction points if available
    if len(pred_averages) > 0 and len(predicted_caffeine) > 0:
        X_pred = np.array(pred_averages)
        y_pred = np.array(predicted_caffeine)
        plt.scatter(X_pred, y_pred, color='red', s=100, alpha=0.7, label='Predictions', zorder=5)

        # Add labels for prediction points (abbreviated names to avoid clutter)
        for i, (x, y, name) in enumerate(zip(X_pred, y_pred, predict_names)):
            # Shorten long names for better readability
            short_name = name[:20] + "..." if len(name) > 20 else name
            plt.annotate(short_name, (x, y), xytext=(5, 5), textcoords='offset points',
                         fontsize=8, alpha=0.8, ha='left', color='red')

    # Create regression line
    all_responses = np.concatenate([X_train, X_pred]) if len(pred_averages) > 0 else X_train
    x_line = np.linspace(min(all_responses) * 0.9, max(all_responses) * 1.1, 100)
    y_line = model.predict(x_line.reshape(-1, 1))

    # Calculate prediction intervals
    # Get predictions for training data to calculate MSE
    y_train_pred = model.predict(X_train.reshape(-1, 1))
    mse = np.mean((y_train - y_train_pred) ** 2)
    standard_error = np.sqrt(mse)
    prediction_interval = 1.96 * standard_error

    # Plot regression line with prediction interval
    plt.plot(x_line, y_line, color='red', linewidth=2, label='Regression Line', alpha=0.8)

    # Add prediction interval (confidence bands)
    plt.fill_between(x_line, y_line - prediction_interval, y_line + prediction_interval,
                     color='red', alpha=0.2, label='Prediction Interval (95%)')

    # Add equation and R² to the plot
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X_train.reshape(-1, 1), y_train)

    equation_text = f'y = {slope:.2f}x + {intercept:.2f}\nR² = {r2:.4f}\nMSE = {mse:.2f}'
    plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             verticalalignment='top', fontsize=10)

    plt.xlabel('Averages (µA)')
    plt.ylabel('Ground Truth (Caffeine ppm)')
    plt.title('Regression Line with Error Bars')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    print("Starting voltammetry analysis...")

    # Initialize plot
    plt.figure(figsize=(14, 10))

    # Process training data
    print("\nProcessing training data...")
    for i, file_path in enumerate(file_paths):
        df = read_data(file_path)
        if df is not None:
            df_normalized, response_at_1v = cga_normalization(df, is_prediction=False)
            normalized_training_dfs.append(df_normalized)
            data.append(df_normalized)
            plot_data(i, file_path, df_normalized, is_prediction=False)

    # Calculate training averages
    training_averages, training_std_devs = process_chunks(responses)
    print(f"\nTraining averages: {training_averages}")
    print(f"Training std devs: {training_std_devs}")

    # Train the model
    print("\nTraining model...")
    model, training_predictions = train_and_evaluate_model(training_averages, caffeine_ppm)

    # Process prediction data
    print("\nProcessing prediction data...")
    for i, file_path in enumerate(predict_file_paths):
        df = read_data(file_path)
        if df is not None:
            df_normalized, response_at_1v = cga_normalization(df, is_prediction=True)
            normalized_prediction_dfs.append(df_normalized)
            plot_data(i, file_path, df_normalized, is_prediction=True, color_index=i)

    # Make predictions
    print("\nMaking predictions...")
    pred_averages, pred_std_devs, predicted_caffeine = process_predictions(model, prediction_responses)

    # Evaluate prediction accuracy
    if len(predicted_caffeine) > 0:
        evaluate_prediction_accuracy(predicted_caffeine, actual_caffeine_ppm, predict_names)

    # ADD THIS NEW LINE - Plot regression analysis
    plot_regression_line(model, training_averages, caffeine_ppm, pred_averages, predicted_caffeine, coffee_names, predict_names)

    # Create a single comprehensive plot
    plt.xlabel('Applied Voltage (V)')
    plt.ylabel('Detected Response (µA)')
    plt.title('Voltammetry Analysis: Training Data and Predictions')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 2)

    # Add legend only for first few entries to avoid clutter
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(handles) > 10:
        plt.legend(handles[:10], labels[:10], bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()