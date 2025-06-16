import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tabulate
from util import setup_mplt, DATADIR, PLOTDIR

#from nn_1_train_model import CoffeeNet4096x7 as CoffeeNet
#from nn_1_train_model import CoffeeNet1024x5 as CoffeeNet
#from nn_1_train_model import CoffeeNet1024x5v2 as CoffeeNet
from nn_1_train_model import CoffeeNet1024x5raw as CoffeeNet

from nn_1_train_model import evaluate_model

if __name__ == "__main__":
    setup_mplt()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #MODEL_NAME = 'CoffeeNet1024x5v2_200_2comb-10_bin128_redox'
    MODEL_NAME = 'CoffeeNet1024x5raw_100_2comb-10_bin64'

    #MODEL_NAME = 'CoffeeNet4096x7_400_2comb-10_bin128_redox'


    MODEL_PATH = DATADIR / f'{MODEL_NAME}.pth'
    TEST_DATA_PATH = DATADIR / 'test_64.pkl'

    EVALUATION_PLOT_PATH = PLOTDIR / f'test_evaluation_{MODEL_NAME}.pdf'
    ERROR_VIOLIN_PLOT_PATH = PLOTDIR / f'test_error_violin_{MODEL_NAME}.pdf'
    target_names = ['HPLC_Caff', 'HPLC_CGA', 'TDS']

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the saved model and normalization statistics
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    input_size = checkpoint['input_size']

    X_scaler = StandardScaler()
    X_scaler.mean_ = checkpoint['X_mean']
    X_scaler.scale_ = checkpoint['X_std']

    y_scaler = StandardScaler()
    y_scaler.mean_ = checkpoint['y_mean']
    y_scaler.scale_ = checkpoint['y_std']


    model = CoffeeNet(input_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set model to evaluation mode

    print(f"Model loaded from {MODEL_PATH}")

    # Load test data
    try:
        df_test = pd.read_pickle(TEST_DATA_PATH)
        print(f"Test data loaded from {TEST_DATA_PATH}")
    except FileNotFoundError:
        print(f"Error: Test data file not found at {TEST_DATA_PATH}")
        print("Please ensure 'test_data.pkl' exists in the DATADIR.")
        exit()

    print("\nTest Data Description:")
    print(df_test.describe())

    # Prepare test data
    X = np.array(df_test['cv_raw'].tolist())
    y = df_test[target_names].values

    X_standard = X_scaler.transform(X)
    y_standard = y_scaler.transform(y)


    predictions = evaluate_model(model, X_standard, y_standard)


    # Inverse transform predictions to original scale using training set statistics
    #predictions_original_scale_test = predictions_normalized_test.numpy() * (y_std_train + 1e-8) + y_mean_train
    predictions_original_scale_test = y_scaler.inverse_transform(predictions)

    # change targets for plots (display)
    target_names = ['Caffeine', 'CGA', 'TDS']
    print("\nEvaluation on Test Data:")
    results = defaultdict(dict)
    for i, name in enumerate(target_names):

        results[name]['r2']= r2_score(y[:, i], predictions_original_scale_test[:, i])
        results[name]['mae'] = mean_absolute_error(y[:, i], predictions_original_scale_test[:, i])
        results[name]['predictions']= predictions_original_scale_test[:, i]

        print(f"{name} - R2 Score: {results[name]['r2']:.4f}, MAE: {results[name]['mae']:.4f}")


    print(tabulate.tabulate(
        [[name, results[name]['r2'], results[name]['mae']] for name in target_names],

        headers=["Name", "R2 Score", "MAE"],
        tablefmt='pipe',
        floatfmt=".4f"
    ))

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Model Evaluation on Test Data ({MODEL_NAME})', fontsize=16)
    for i, name in enumerate(target_names):
        axes[i].scatter(y[:, i], predictions_original_scale_test[:, i], alpha=0.5, label='Predictions')
        axes[i].plot([y[:, i].min(), y[:, i].max()],
                     [y[:, i].min(), y[:, i].max()], 'r--', label='Ideal')
        axes[i].set_xlabel(f"Actual {name}")
        axes[i].set_ylabel(f"Predicted {name}")

        axes[i].set_title(f"Actual vs. Predicted {name}\nR2: {results[name]['r2']:.4f} MAE: {results[name]['mae']:.4f}",)
        axes[i].grid(True)
        axes[i].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.savefig(EVALUATION_PLOT_PATH)
    print(f"Evaluation plot saved to {EVALUATION_PLOT_PATH}")
    #plt.show()

    # Violin plot of prediction errors (%)
    errors = 100.0 * (predictions_original_scale_test - y) / y

    fig_violin, ax_violin_main = plt.subplots(figsize=(12, 7)) # Adjusted figure size
    fig.suptitle(f'Model Evaluation on Test Data ({MODEL_NAME})', fontsize=16)


    df_test['Model Caffeine'] = predictions_original_scale_test[:, 0]
    df_test['Model CGA'] = predictions_original_scale_test[:, 1]
    df_test['Model TDS'] = predictions_original_scale_test[:, 2]

    df_test['Caffeine Error'] = errors[:, 0]
    df_test['CGA Error'] = errors[:, 1]
    df_test['TDS Error'] = errors[:, 2]



    df_test.sort_values(by='Caffeine Error', ascending=True, inplace=True)
    print("\nTest Data sorted by Caffeine Error:")
    print(tabulate.tabulate(df_test[['Sample Name',
                                     'HPLC_Caff', 'Model Caffeine', 'Caffeine Error',
                                     'HPLC_CGA', 'Model CGA', 'CGA Error',
                                     'TDS', 'Model TDS', 'TDS Error']], headers='keys', tablefmt='psql'))

    # Plot HPLC_Caff and HPLC_CGA on the primary y-axis
    parts_main = ax_violin_main.violinplot(errors,
                                           positions=[1, 2, 3], # Specify positions
                                           showmeans=True, showmedians=False, widths=0.8)

    # for pc in parts_main['bodies']:
    #     pc.set_facecolor('skyblue')
    #     pc.set_edgecolor('black')
    #     pc.set_alpha(0.7)

    ax_violin_main.set_ylabel('Prediction Error % ppm')
    ax_violin_main.tick_params(axis='y',)

    # # Create a secondary y-axis for TDS
    # ax_violin_tds = ax_violin_main.twinx()
    # parts_tds = ax_violin_tds.violinplot(errors_secondary,
    #                                      positions=[3], # Specify position for TDS
    #                                      showmeans=True, showmedians=False, widths=0.8)
    # for pc in parts_tds['bodies']:
    #     pc.set_facecolor('lightcoral')
    #     pc.set_edgecolor('black')
    #     pc.set_alpha(0.7)

    #ax_violin_tds.set_ylabel('Prediction Error (TDS) %*UNITS?*')
    #ax_violin_tds.tick_params(axis='y')

    # Common x-axis settings
    ax_violin_main.set_xticks(np.arange(1, len(target_names) + 1))
    ax_violin_main.set_xticklabels(target_names)
    ax_violin_main.set_title('Distribution of Prediction Error Percent on Test Data')
    #ax_violin_main.grid(True, linestyle='--', alpha=0.7, axis='x') # Grid for x-axis from main
    ax_violin_main.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(ERROR_VIOLIN_PLOT_PATH)
    print(f"Error violin plot saved to {ERROR_VIOLIN_PLOT_PATH}")
    plt.show()