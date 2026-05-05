from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import tabulate

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Configuration

CGA_DIR = "CGA Standards"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define a simple neural network
class CGANet(nn.Module):
    def __init__(self, input_size):
        super(CGANet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)  # Single output for CGA concentration
        )

    def forward(self, x):
        return self.network(x)


def read_data(file_path):
    """Load voltammogram data from a .txt file"""
    try:
        df = pd.read_csv(file_path, delimiter=',', header=None,
                         names=['Time', 'Applied Voltage', 'Detected Response'])
        df = df[(df['Applied Voltage'] >= 0.0)]
        # Keep the first half of the voltammogram
        mid_index = len(df) // 2
        df = df.iloc[:mid_index].reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def load_voltammogram(filepath):
    """Load voltammogram and return detected response as feature vector"""
    df = read_data(filepath)
    if df is not None:
        return df['Detected Response'].values
    return None


def train_model(model, X_train, y_train, X_val, y_val, num_epochs=500, batch_size=32):
    """Train the neural network"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).reshape(-1, 1).to(device)

    # Create data loader
    dataset = TensorDataset(X_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss / len(dataloader):.6f}, Val Loss: {val_loss.item():.6f}")
            model.train()

    return model


def evaluate_model(model, X, y):
    """Evaluate model and return predictions"""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        predictions = model(X_t).cpu().numpy().flatten()
    return predictions


# Main execution
if __name__ == "__main__":
    # Load all voltammogram files
    # txt_files = sorted(CGA_DIR.glob("*.txt"))
    txt_files = sorted(Path(CGA_DIR).glob("*.txt"))

    if len(txt_files) != 15:
        print(f"Warning: Expected 15 files, found {len(txt_files)}")

    print(f"Found {len(txt_files)} voltammogram files")

    # Load data
    voltammograms = []
    for f in txt_files:
        voltammograms.append(load_voltammogram(f))

    # Convert to numpy array
    X_all = np.array(voltammograms)

    # Define CGA concentrations (ppm)
    y_all = np.array([200] * 5 + [500] * 5 + [800] * 5)

    print(f"Data shape: {X_all.shape}")
    print(f"CGA concentrations: {y_all}")

    # Leave-one-out cross-validation
    results = []
    all_train_actual = []
    all_train_pred = []
    all_val_actual = []
    all_val_pred = []
    all_test_actual = []
    all_test_pred = []

    for i in range(len(X_all)):
        print(f"\n{'=' * 50}")
        print(f"Fold {i + 1}/{len(X_all)} - Leaving out sample {i + 1}")
        print(f"{'=' * 50}")

        # Create train/test split
        train_mask = np.ones(len(X_all), dtype=bool)
        train_mask[i] = False

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test = X_all[~train_mask]
        y_test = y_all[~train_mask]

        # Standardize features
        X_scaler = StandardScaler().fit(X_train)
        y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

        X_train_scaled = X_scaler.transform(X_train)
        y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).flatten()
        X_test_scaled = X_scaler.transform(X_test)
        y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

        # Create and train model
        input_size = X_train_scaled.shape[1]
        model = CGANet(input_size).to(device)

        # Use one sample from training as validation (simple approach)
        val_idx = 0
        X_train_final = np.delete(X_train_scaled, val_idx, axis=0)
        y_train_final = np.delete(y_train_scaled, val_idx)
        X_val = X_train_scaled[val_idx:val_idx + 1]
        y_val = y_train_scaled[val_idx:val_idx + 1]

        model = train_model(model, X_train_final, y_train_final, X_val, y_val, num_epochs=500)

        # Evaluate
        train_pred_scaled = evaluate_model(model, X_train_scaled, y_train_scaled)
        train_pred = y_scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten()

        val_pred_scaled = evaluate_model(model, X_val, y_val)
        val_pred = y_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
        val_actual = y_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

        test_pred_scaled = evaluate_model(model, X_test_scaled, y_test_scaled)
        test_pred = y_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()

        # Store for overall metrics
        all_train_actual.extend(y_train)
        all_train_pred.extend(train_pred)
        all_val_actual.extend(val_actual)
        all_val_pred.extend(val_pred)
        all_test_actual.extend(y_test)
        all_test_pred.extend(test_pred)

        # Store results
        results.append({
            'fold': i + 1,
            'test_sample': i + 1,
            'actual_ppm': y_test[0],
            'predicted_ppm': test_pred[0],
            'error_ppm': test_pred[0] - y_test[0],
            'error_pct': 100 * (test_pred[0] - y_test[0]) / y_test[0],
            'train_r2': r2_score(y_train, train_pred),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'val_r2': r2_score(val_actual, val_pred),
            'val_mae': mean_absolute_error(val_actual, val_pred),
            'test_r2': r2_score(y_test, test_pred),
            'test_mae': mean_absolute_error(y_test, test_pred)
        })

        print(f"Actual: {y_test[0]:.1f} ppm, Predicted: {test_pred[0]:.1f} ppm")

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Calculate overall metrics
    overall_train_r2 = r2_score(all_train_actual, all_train_pred)
    overall_train_mae = mean_absolute_error(all_train_actual, all_train_pred)
    overall_val_r2 = r2_score(all_val_actual, all_val_pred)
    overall_val_mae = mean_absolute_error(all_val_actual, all_val_pred)
    overall_test_r2 = r2_score(all_test_actual, all_test_pred)
    overall_test_mae = mean_absolute_error(all_test_actual, all_test_pred)

    # Print detailed results
    print("\n" + "=" * 80)
    print("LEAVE-ONE-OUT CROSS-VALIDATION RESULTS")
    print("=" * 80)
    print(tabulate.tabulate(
        df_results[['fold', 'test_sample', 'actual_ppm', 'predicted_ppm', 'error_ppm', 'error_pct']],
        headers='keys',
        tablefmt='psql',
        floatfmt='.2f',
        showindex=False
    ))

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (Per-Fold Averages)")
    print("=" * 80)
    summary = {
        'Metric': ['Mean R²', 'Mean MAE (ppm)', 'Mean Error (ppm)', 'Mean Abs Error %', 'Std Error (ppm)'],
        'Training': [
            df_results['train_r2'].mean(),
            df_results['train_mae'].mean(),
            df_results['error_ppm'].mean(),
            df_results['error_pct'].abs().mean(),
            df_results['error_ppm'].std()
        ],
        'Validation': [
            df_results['val_r2'].mean(),
            df_results['val_mae'].mean(),
            df_results['error_ppm'].mean(),
            df_results['error_pct'].abs().mean(),
            df_results['error_ppm'].std()
        ],
        'Testing': [
            df_results['test_r2'].mean(),
            df_results['test_mae'].mean(),
            df_results['error_ppm'].mean(),
            df_results['error_pct'].abs().mean(),
            df_results['error_ppm'].std()
        ]
    }

    print(tabulate.tabulate(
        pd.DataFrame(summary),
        headers='keys',
        tablefmt='psql',
        floatfmt='.4f',
        showindex=False
    ))

    # Print overall accuracy across all folds
    print("\n" + "=" * 80)
    print("OVERALL ACCURACY (All Predictions Combined)")
    print("=" * 80)
    overall_summary = {
        'Metric': ['Overall R²', 'Overall MAE (ppm)', 'Overall RMSE (ppm)'],
        'Training': [
            overall_train_r2,
            overall_train_mae,
            np.sqrt(np.mean((np.array(all_train_actual) - np.array(all_train_pred)) ** 2))
        ],
        'Validation': [
            overall_val_r2,
            overall_val_mae,
            np.sqrt(np.mean((np.array(all_val_actual) - np.array(all_val_pred)) ** 2))
        ],
        'Testing': [
            overall_test_r2,
            overall_test_mae,
            np.sqrt(np.mean((np.array(all_test_actual) - np.array(all_test_pred)) ** 2))
        ]
    }

    print(tabulate.tabulate(
        pd.DataFrame(overall_summary),
        headers='keys',
        tablefmt='psql',
        floatfmt='.4f',
        showindex=False
    ))

    print(
        f"\nTotal samples - Training: {len(all_train_actual)}, Validation: {len(all_val_actual)}, Testing: {len(all_test_actual)}")

    # Print actual vs predicted for each voltammogram
    print("\n" + "=" * 80)
    print("ACTUAL VS PREDICTED CGA CONCENTRATION (ppm)")
    print("=" * 80)
    prediction_table = {
        'Sample': list(range(1, len(all_test_actual) + 1)),
        'Actual (ppm)': all_test_actual,
        'Predicted (ppm)': [round(p, 2) for p in all_test_pred],
        'Error (ppm)': [round(p - a, 2) for p, a in zip(all_test_pred, all_test_actual)],
        'Error (%)': [round(100 * (p - a) / a, 2) for p, a in zip(all_test_pred, all_test_actual)]
    }
    print(tabulate.tabulate(
        pd.DataFrame(prediction_table),
        headers='keys',
        tablefmt='psql',
        floatfmt='.2f',
        showindex=False
    ))

    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(all_test_actual, all_test_pred, s=100, alpha=0.6, edgecolors='black', linewidth=1.5)

    # Plot perfect prediction line
    min_val = min(min(all_test_actual), min(all_test_pred))
    max_val = max(max(all_test_actual), max(all_test_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    plt.xlabel('Actual CGA Concentration (ppm)', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted CGA Concentration (ppm)', fontsize=12, fontweight='bold')
    plt.title(
        f'Actual vs Predicted CGA Concentration\nR² = {overall_test_r2:.4f}, MAE = {overall_test_mae:.2f} ppm',
        fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plot_path = 'cga_predictions.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPrediction plot saved to {plot_path}")
    plt.show()

    # Save results
    results_path = 'cga_results.pkl'
    df_results.to_pickle(results_path)
    print(f"Results saved to {results_path}")