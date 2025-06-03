import pandas as pd
import numpy as np

# Constants from your original code
SMOOTHING_WINDOW_SIZE = 25


def read_data(file_path):
    """Reads the data from the specified file path and returns a DataFrame."""
    try:
        df = pd.read_csv(file_path, delimiter=',', header=None, names=['Time', 'Applied Voltage', 'Detected Response'])
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def moving_average(y, window_size=5):
    """Calculates the moving average of a given array y with a specified window size."""
    return np.convolve(y, np.ones(window_size) / window_size, mode='valid')


def count_data_points(file_path):
    """Count data points at each processing step"""
    print(f"\nAnalyzing file: {file_path}")
    print("=" * 50)

    # Step 1: Read raw data
    df = read_data(file_path)
    if df is None:
        return

    raw_count = len(df)
    print(f"1. Raw data points: {raw_count}")

    # Step 2: Filter voltage >= 0.0 (like in your original code)
    df_filtered = df[(df['Applied Voltage'] >= 0.0)]
    filtered_count = len(df_filtered)
    print(f"2. After voltage filtering (>= 0.0V): {filtered_count}")

    # Step 3: Extract voltage and response arrays
    voltage = df_filtered['Applied Voltage'].values
    response = df_filtered['Detected Response'].values

    # Step 4: Apply smoothing (moving average)
    y_smoothed = moving_average(response, window_size=SMOOTHING_WINDOW_SIZE)
    smoothed_count = len(y_smoothed)
    print(f"3. After smoothing (window size {SMOOTHING_WINDOW_SIZE}): {smoothed_count}")

    # Step 5: Take first half (like in your plot_data function)
    first_half_count = smoothed_count // 2
    print(f"4. First half for analysis: {first_half_count}")

    # Summary
    print(f"\nData reduction summary:")
    print(f"  Original → Filtered: {raw_count} → {filtered_count} ({filtered_count / raw_count * 100:.1f}%)")
    print(f"  Filtered → Smoothed: {filtered_count} → {smoothed_count} ({smoothed_count / filtered_count * 100:.1f}%)")
    print(f"  Smoothed → Final: {smoothed_count} → {first_half_count} ({first_half_count / smoothed_count * 100:.1f}%)")
    print(f"  Overall reduction: {raw_count} → {first_half_count} ({first_half_count / raw_count * 100:.1f}%)")


# Example usage - replace with your actual file path
if __name__ == "__main__":
    # Test with one of your files
    test_file = '../voltammetry-files/aladec1.txt'  # Change this to your actual file path
    count_data_points(test_file)