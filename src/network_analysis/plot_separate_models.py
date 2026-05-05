from util import setup_mplt, DATADIR, PLOTDIR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error


def plot_predictions_vs_actual(results_path=None, save_plots=True):
    """
    Plot predicted vs actual values for each target separately.
    Recalculates R² from the actual prediction data to ensure accuracy.

    Parameters:
    -----------
    results_path : Path or str, optional
        Path to the results pickle file. If None, uses default path.
    save_plots : bool, optional
        Whether to save normalized_plots to PLOTDIR. Default True.
    """
    setup_mplt()
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'

    if results_path is None:
        results_path = DATADIR / 'separate_model_results_expanded_voltage_filtered.pkl'

    print(f"Loading results from {results_path}")

    # Load results
    df_results = pd.read_pickle(results_path)

    target_names = df_results['target'].unique()
    print(f"Found targets: {target_names}")

    # Create separate figure for each target
    for target_name in target_names:
        print(f"\n{'=' * 60}")
        print(f"Processing {target_name}...")
        print(f"{'=' * 60}")

        df_target = df_results[df_results['target'] == target_name]

        # Get unique experiment names for this target
        experiment_names = df_target['experiment_name'].unique()

        # Create figure with subplots for each experiment
        n_experiments = len(experiment_names)
        fig, axes = plt.subplots(1, n_experiments, figsize=(6 * n_experiments, 5))

        # Make sure axes is iterable even if there's only one experiment
        if n_experiments == 1:
            axes = [axes]

        # Calculate actual R² for ranking
        exp_r2_scores = []
        for exp_name in experiment_names:
            df_exp = df_target[df_target['experiment_name'] == exp_name]

            # Collect all test predictions and actuals
            all_test_pred = []
            all_test_actual = []

            for _, row in df_exp.iterrows():
                all_test_pred.extend(row[f'test_{target_name}_predictions'])
                all_test_actual.extend(row[f'test_{target_name}_actual'])

            all_test_pred = np.array(all_test_pred)
            all_test_actual = np.array(all_test_actual)

            # Calculate actual R²
            test_r2 = r2_score(all_test_actual, all_test_pred)
            exp_r2_scores.append((exp_name, test_r2))

        # Sort by R² (descending)
        exp_r2_scores.sort(key=lambda x: x[1], reverse=True)

        print(f"\nExperiments ranked by Test R² for {target_name}:")
        for i, (exp_name, r2) in enumerate(exp_r2_scores):
            print(f"  {i + 1}. {exp_name}: R² = {r2:.4f}")

        # Plot each experiment
        for idx, (exp_name, _) in enumerate(exp_r2_scores):
            ax = axes[idx]
            df_exp = df_target[df_target['experiment_name'] == exp_name]

            # Collect all predictions and actuals
            all_test_pred = []
            all_test_actual = []
            all_train_pred = []
            all_train_actual = []

            for _, row in df_exp.iterrows():
                all_test_pred.extend(row[f'test_{target_name}_predictions'])
                all_test_actual.extend(row[f'test_{target_name}_actual'])
                all_train_pred.extend(row[f'train_{target_name}_predictions'])
                all_train_actual.extend(row[f'train_{target_name}_actual'])

            all_test_pred = np.array(all_test_pred)
            all_test_actual = np.array(all_test_actual)
            all_train_pred = np.array(all_train_pred)
            all_train_actual = np.array(all_train_actual)

            # Calculate metrics
            train_r2 = r2_score(all_train_actual, all_train_pred)
            test_r2 = r2_score(all_test_actual, all_test_pred)
            train_mae = mean_absolute_error(all_train_actual, all_train_pred)
            test_mae = mean_absolute_error(all_test_actual, all_test_pred)

            # Calculate MSE loss (what the model actually minimized)
            train_mse = np.mean((all_train_pred - all_train_actual) ** 2)
            test_mse = np.mean((all_test_pred - all_test_actual) ** 2)

            print(f"\n  {short_title if 'short_title' in locals() else exp_name}:")
            print(f"    Train - R²: {train_r2:.4f}, MAE: {train_mae:.3f}, MSE: {train_mse:.4f}")
            print(f"    Test  - R²: {test_r2:.4f}, MAE: {test_mae:.3f}, MSE: {test_mse:.4f}")

            # Plot train data
            ax.scatter(all_train_actual, all_train_pred, alpha=0.4, label='Train', s=30, c='blue')

            # Plot test data
            ax.scatter(all_test_actual, all_test_pred, alpha=0.7, label='Test', s=60,
                       marker='s', c='red', edgecolors='darkred', linewidths=1)

            # Annotate EVERY test point with its test_coffee label
            for _, row in df_exp.iterrows():
                coffee_label = row['test_coffee']

                test_actual = row[f'test_{target_name}_actual']
                test_pred = row[f'test_{target_name}_predictions']

                for x, y in zip(test_actual, test_pred):
                    ax.annotate(
                        coffee_label,
                        (x, y),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=2,
                        alpha=0.8
                    )

            # Perfect prediction line
            all_vals = np.concatenate([all_train_actual, all_test_actual, all_train_pred, all_test_pred])
            min_val, max_val = all_vals.min(), all_vals.max()
            margin = (max_val - min_val) * 0.05
            ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin],
                    'k--', linewidth=2, label='Perfect Prediction', alpha=0.5)

            ax.set_xlabel(f'Actual {target_name}', fontsize=12)
            ax.set_ylabel(f'Predicted {target_name}', fontsize=12)

            # Create shorter title
            title_parts = exp_name.split('-')
            if 'NOISE' in exp_name:
                noise_idx = [i for i, p in enumerate(title_parts) if 'NOISE' in p][0]
                short_title = '-'.join(title_parts[noise_idx:noise_idx + 2])
            else:
                # Get architecture part (usually near the end)
                short_title = '-'.join([p for p in title_parts if any(c.isdigit() for c in p)][-2:])

            ax.set_title(f'{short_title}', fontsize=13, fontweight='bold')
            ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
            ax.grid(alpha=0.3)

            # Add metrics as text box in lower right
            textstr = f'Train R²: {train_r2:.4f}\nTest R²: {test_r2:.4f}\n'
            textstr += f'Train MAE: {train_mae:.3f}\nTest MAE: {test_mae:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.95, edgecolor='black', linewidth=1)
            ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right', bbox=props)

            # Make plot square
            ax.set_aspect('equal', adjustable='box')

        fig.suptitle(f'{target_name}: Predicted vs Actual Values', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_plots:
            plot_path = PLOTDIR / f'{target_name}_predictions_vs_actual.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {plot_path}")

        plt.show()


def plot_best_model_detailed(results_path=None, save_plots=True):
    """
    Plot detailed analysis of the best model for each target.

    Parameters:
    -----------
    results_path : Path or str, optional
        Path to the results pickle file. If None, uses default path.
    save_plots : bool, optional
        Whether to save normalized_plots to PLOTDIR. Default True.
    """
    setup_mplt()
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'

    if results_path is None:
        results_path = DATADIR / 'separate_model_results_expanded_voltage_filtered.pkl'

    print(f"Loading results from {results_path}")
    df_results = pd.read_pickle(results_path)

    target_names = df_results['target'].unique()

    for target_name in target_names:
        print(f"\n{'=' * 60}")
        print(f"Best model analysis for {target_name}")
        print(f"{'=' * 60}")

        df_target = df_results[df_results['target'] == target_name]

        # Find best model by recalculating R²
        best_exp = None
        best_r2 = -np.inf

        for exp_name in df_target['experiment_name'].unique():
            df_exp = df_target[df_target['experiment_name'] == exp_name]

            all_test_pred = []
            all_test_actual = []

            for _, row in df_exp.iterrows():
                all_test_pred.extend(row[f'test_{target_name}_predictions'])
                all_test_actual.extend(row[f'test_{target_name}_actual'])

            test_r2 = r2_score(np.array(all_test_actual), np.array(all_test_pred))

            if test_r2 > best_r2:
                best_r2 = test_r2
                best_exp = exp_name

        print(f"Best model: {best_exp}")
        print(f"Best Test R²: {best_r2:.4f}")

        df_best = df_target[df_target['experiment_name'] == best_exp]

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 10))

        # 1. Overall prediction scatter
        ax1 = plt.subplot(2, 2, 1)

        all_test_pred = []
        all_test_actual = []
        all_train_pred = []
        all_train_actual = []

        for _, row in df_best.iterrows():
            all_test_pred.extend(row[f'test_{target_name}_predictions'])
            all_test_actual.extend(row[f'test_{target_name}_actual'])
            all_train_pred.extend(row[f'train_{target_name}_predictions'])
            all_train_actual.extend(row[f'train_{target_name}_actual'])

        all_test_pred = np.array(all_test_pred)
        all_test_actual = np.array(all_test_actual)
        all_train_pred = np.array(all_train_pred)
        all_train_actual = np.array(all_train_actual)

        ax1.scatter(all_train_actual, all_train_pred, alpha=0.4, label='Train', s=30)
        ax1.scatter(all_test_actual, all_test_pred, alpha=0.7, label='Test', s=60, marker='s')

        all_vals = np.concatenate([all_train_actual, all_test_actual])
        min_val, max_val = all_vals.min(), all_vals.max()
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

        ax1.set_xlabel(f'Actual {target_name}')
        ax1.set_ylabel(f'Predicted {target_name}')
        ax1.set_title('Predicted vs Actual')
        ax1.legend(loc='upper left', framealpha=0.95)
        ax1.grid(alpha=0.3)

        train_r2 = r2_score(all_train_actual, all_train_pred)
        test_r2 = r2_score(all_test_actual, all_test_pred)
        ax1.text(0.95, 0.05, f'Train R²: {train_r2:.4f}\nTest R²: {test_r2:.4f}',
                 transform=ax1.transAxes, verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.95, edgecolor='black', linewidth=1))

        # 2. R² by fold
        ax2 = plt.subplot(2, 2, 2)

        fold_train_r2 = []
        fold_test_r2 = []
        fold_names = []

        for _, row in df_best.iterrows():
            test_pred = np.array(row[f'test_{target_name}_predictions'])
            test_actual = np.array(row[f'test_{target_name}_actual'])
            train_pred = np.array(row[f'train_{target_name}_predictions'])
            train_actual = np.array(row[f'train_{target_name}_actual'])

            fold_train_r2.append(r2_score(train_actual, train_pred))
            fold_test_r2.append(r2_score(test_actual, test_pred))
            fold_names.append(row['test_coffee'])

        x = np.arange(len(fold_names))
        ax2.plot(x, fold_train_r2, 'o-', label='Train R²', linewidth=2, markersize=8)
        ax2.plot(x, fold_test_r2, 's-', label='Test R²', linewidth=2, markersize=8)
        ax2.axhline(y=np.mean(fold_test_r2), color='orange', linestyle='--',
                    alpha=0.5, label=f'Mean Test: {np.mean(fold_test_r2):.3f}')
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('R² Score')
        ax2.set_title('R² Score by Fold')
        ax2.legend(framealpha=0.95)
        ax2.grid(alpha=0.3)

        # 3. R² by coffee
        ax3 = plt.subplot(2, 2, 3)

        colors = ['green' if r2 > 0.8 else 'orange' if r2 > 0.5 else 'red' for r2 in fold_test_r2]
        ax3.barh(range(len(fold_names)), fold_test_r2, color=colors, alpha=0.7)
        ax3.set_yticks(range(len(fold_names)))
        ax3.set_yticklabels(fold_names)
        ax3.set_xlabel('Test R²')
        ax3.set_title('Test R² by Coffee')
        ax3.axvline(x=np.mean(fold_test_r2), color='black', linestyle='--',
                    linewidth=2, label=f'Mean: {np.mean(fold_test_r2):.3f}')
        ax3.legend(framealpha=0.95)
        ax3.grid(axis='x', alpha=0.3)

        # 4. Error distribution
        ax4 = plt.subplot(2, 2, 4)

        test_errors = all_test_pred - all_test_actual
        train_errors = all_train_pred - all_train_actual

        bins = np.linspace(min(train_errors.min(), test_errors.min()),
                           max(train_errors.max(), test_errors.max()), 30)
        ax4.hist(train_errors, bins=bins, alpha=0.5, label='Train', edgecolor='black')
        ax4.hist(test_errors, bins=bins, alpha=0.5, label='Test', edgecolor='black')
        ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')

        ax4.set_xlabel('Prediction Error')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Error Distribution')
        ax4.legend(framealpha=0.95)
        ax4.grid(axis='y', alpha=0.3)

        fig.suptitle(f'{target_name}: Best Model Performance\n{best_exp}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_plots:
            plot_path = PLOTDIR / f'{target_name}_best_model_detailed.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {plot_path}")

        plt.show()


if __name__ == "__main__":
    # Plot predicted vs actual for each target
    print("=" * 60)
    print("Plotting predicted vs actual for all targets...")
    print("=" * 60)
    plot_predictions_vs_actual(save_plots=True)

    # Plot detailed analysis of best models
    print("\n" + "=" * 60)
    print("Plotting detailed analysis of best models...")
    print("=" * 60)
    plot_best_model_detailed(save_plots=True)