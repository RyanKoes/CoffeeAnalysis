import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Moving window constant (adjustable)
MOVING_WINDOW = 25


def moving_average(data, window):
    """Apply moving average smoothing to data"""
    if window <= 1:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def load_and_filter_cv_data(filepath):
    """Load CV data and filter for voltage > 0 and increasing up to 2.0V"""
    data = np.loadtxt(filepath, delimiter=',')
    time = data[:, 0]
    voltage = data[:, 1]
    current = data[:, 2]

    # Filter for voltage > 0
    mask_positive = voltage > 0
    voltage_filtered = voltage[mask_positive]
    current_filtered = current[mask_positive]

    # Find the peak voltage (should be around 2.0V)
    peak_idx = np.argmax(voltage_filtered)

    # Only keep data up to the peak (increasing voltage)
    voltage_increasing = voltage_filtered[:peak_idx + 1]
    current_increasing = current_filtered[:peak_idx + 1]

    return voltage_increasing, current_increasing


# Set up the plot
plt.figure(figsize=(10, 6))

# Plot regular runs (red)
for i in range(1, 6):
    filepath = Path('Data') / f'tanzanian{i}.txt'
    try:
        voltage, current = load_and_filter_cv_data(filepath)

        # Apply moving average
        if len(voltage) > MOVING_WINDOW:
            voltage_smooth = voltage[:len(voltage) - MOVING_WINDOW + 1]
            current_smooth = moving_average(current, MOVING_WINDOW)
        else:
            voltage_smooth = voltage
            current_smooth = current

        plt.plot(voltage_smooth, current_smooth, 'r-', alpha=0.7, linewidth=1.5,
                 label='Regular' if i == 1 else '')
    except FileNotFoundError:
        print(f"Warning: {filepath} not found")

# Plot OCP runs (blue)
for i in range(1, 6):
    filepath = Path('Data') / f'tanzanian_ocp{i}.txt'
    try:
        voltage, current = load_and_filter_cv_data(filepath)

        # Apply moving average
        if len(voltage) > MOVING_WINDOW:
            voltage_smooth = voltage[:len(voltage) - MOVING_WINDOW + 1]
            current_smooth = moving_average(current, MOVING_WINDOW)
        else:
            voltage_smooth = voltage
            current_smooth = current

        plt.plot(voltage_smooth, current_smooth, 'b-', alpha=0.7, linewidth=1.5,
                 label='OCP' if i == 1 else '')
    except FileNotFoundError:
        print(f"Warning: {filepath} not found")

# Customize the plot
plt.xlabel('Applied Voltage (V)', fontsize=12)
plt.ylabel('Current Response (μA)', fontsize=12)
plt.title(f'Cyclic Voltammetry Data (Moving Window = {MOVING_WINDOW})', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save and show
plt.savefig('outputs/cv_plot.png', dpi=300, bbox_inches='tight')

print(f"Plot saved with moving window = {MOVING_WINDOW}")