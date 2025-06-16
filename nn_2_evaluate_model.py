import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import tabulate
from util import setup_mplt, DATADIR, PLOTDIR

#from nn_1_train_model import CoffeeNet4096x7 as CoffeeNet
#from nn_1_train_model import CoffeeNet1024x5 as CoffeeNet
#from nn_1_train_model import CoffeeNet1024x5v2 as CoffeeNet
from nn_1_train_model import CoffeeNet1024x5raw as CoffeeNet

if __name__ == "__main__":
    setup_mplt()

    #MODEL_NAME = 'CoffeeNet1024x5v2_200_2comb-10_bin128_redox'
    MODEL_NAME = 'CoffeeNet1024x5raw_200_2comb-10_bin64'
    
    #MODEL_NAME = 'CoffeeNet4096x7_400_2comb-10_bin128_redox'


    MODEL_PATH = DATADIR / f'{MODEL_NAME}.pth'
    TEST_DATA_PATH = DATADIR / 'test_64.pkl'

    EVALUATION_PLOT_PATH = PLOTDIR / f'test_evaluation_{MODEL_NAME}.pdf'
    ERROR_VIOLIN_PLOT_PATH = PLOTDIR / f'test_error_violin_{MODEL_NAME}.pdf'

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the saved model and normalization statistics
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    input_size = checkpoint['input_size']
    X_mean_train = checkpoint['X_mean']
    X_std_train = checkpoint['X_std']
    y_mean_train = checkpoint['y_mean']
    y_std_train = checkpoint['y_std']

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
    #X_list_test = df_test['cv_bins'].tolist()
    X_list_test = df_test['cv_raw'].tolist()
    X_np_test = np.array(X_list_test)

    # Normalize test features using training set statistics
    X_np_test_normalized = (X_np_test - X_mean_train) / (X_std_train + 1e-8)
    
    y_np_test = df_test[['HPLC_Caff', 'HPLC_CGA', 'TDS']].values
    # Note: We don't normalize y_test here, as we want to compare against original scale.
    # The y_mean_train and y_std_train are used for un-normalizing predictions.

    X_tensor_test = torch.tensor(X_np_test_normalized, dtype=torch.float32).to(device)
    # y_tensor_test is not strictly needed for evaluation if we compare numpy arrays later

    # Make predictions
    with torch.no_grad():
        predictions_normalized_test = model(X_tensor_test).cpu()

    # Inverse transform predictions to original scale using training set statistics
    predictions_original_scale_test = predictions_normalized_test.numpy() * (y_std_train + 1e-8) + y_mean_train
    
    targets_original_scale_test = y_np_test

    target_names = ['Caffeine', 'CGA', 'TDS']

    print("\nEvaluation on Test Data:")
    for i, name in enumerate(target_names):
        r2 = r2_score(targets_original_scale_test[:, i], predictions_original_scale_test[:, i])
        mae = mean_absolute_error(targets_original_scale_test[:, i], predictions_original_scale_test[:, i])
        print(f"{name} - R2 Score: {r2:.4f}, MAE: {mae:.4f}")

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Model Evaluation on Test Data', fontsize=16)
    for i, name in enumerate(target_names):
        axes[i].scatter(targets_original_scale_test[:, i], predictions_original_scale_test[:, i], alpha=0.5, label='Predictions')
        axes[i].plot([targets_original_scale_test[:, i].min(), targets_original_scale_test[:, i].max()],
                     [targets_original_scale_test[:, i].min(), targets_original_scale_test[:, i].max()], 'r--', label='Ideal')
        axes[i].set_xlabel(f"Actual {name}")
        axes[i].set_ylabel(f"Predicted {name}")
        r2_val = r2_score(targets_original_scale_test[:, i], predictions_original_scale_test[:, i])
        axes[i].set_title(f"Actual vs. Predicted {name}\nR2: {r2_val:.2f}")
        axes[i].grid(True)
        axes[i].legend()
        
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.savefig(EVALUATION_PLOT_PATH)
    print(f"Evaluation plot saved to {EVALUATION_PLOT_PATH}")
    #plt.show()

    # Violin plot of prediction errors
    errors = predictions_original_scale_test - targets_original_scale_test
    
    fig_violin, ax_violin_main = plt.subplots(figsize=(12, 7)) # Adjusted figure size

    # Separate errors for primary and secondary axes
    errors_primary = [errors[:, 0], errors[:, 1]] # HPLC_Caff, HPLC_CGA
    errors_secondary = [errors[:, 2]]             # TDS

    target_names_primary = [target_names[0], target_names[1]]
    target_names_secondary = [target_names[2]]

    df_test['Model Caffeine'] = predictions_original_scale_test[:, 0]
    df_test['Model CGA'] = predictions_original_scale_test[:, 1]
    df_test['Model TDS'] = predictions_original_scale_test[:, 2]

    df_test['Caffeine Error'] = errors[:, 0]
    df_test['CGA Error'] = errors[:, 1]
    df_test['TDS Error'] = errors[:, 2]

  

    df_test.sort_values(by='Caffeine Error', ascending=True, inplace=True)
    print("\nTest Data sorted by Caffeine Error:")
    print(tabulate.tabulate(df_test[['Sample Name', 'HPLC_Caff', 'Model Caffeine', 'Caffeine Error', 'HPLC_CGA', 'Model CGA', 'CGA Error', 'TDS', 'Model TDS', 'TDS Error']], headers='keys', tablefmt='psql'))

    # Plot HPLC_Caff and HPLC_CGA on the primary y-axis
    parts_main = ax_violin_main.violinplot(errors_primary,                                        
                                           positions=[1, 2], # Specify positions
                                           showmeans=True, showmedians=False, widths=0.8)
    
    # for pc in parts_main['bodies']:
    #     pc.set_facecolor('skyblue')
    #     pc.set_edgecolor('black')
    #     pc.set_alpha(0.7)
    
    ax_violin_main.set_ylabel('Prediction Error (Caffeine, CGA) ppm')
    ax_violin_main.tick_params(axis='y',)

    # Create a secondary y-axis for TDS
    ax_violin_tds = ax_violin_main.twinx()
    parts_tds = ax_violin_tds.violinplot(errors_secondary,    
                                         positions=[3], # Specify position for TDS
                                         showmeans=True, showmedians=False, widths=0.8)
    # for pc in parts_tds['bodies']:
    #     pc.set_facecolor('lightcoral')
    #     pc.set_edgecolor('black')
    #     pc.set_alpha(0.7)

    ax_violin_tds.set_ylabel('Prediction Error (TDS) *UNITS?*')
    ax_violin_tds.tick_params(axis='y')

    # Common x-axis settings
    ax_violin_main.set_xticks(np.arange(1, len(target_names) + 1))
    ax_violin_main.set_xticklabels(target_names)
    ax_violin_main.set_title('Distribution of Prediction Errors on Test Data')
    #ax_violin_main.grid(True, linestyle='--', alpha=0.7, axis='x') # Grid for x-axis from main
    ax_violin_main.grid(True, linestyle='--', alpha=0.7) 
    
    plt.tight_layout()
    plt.savefig(ERROR_VIOLIN_PLOT_PATH)
    print(f"Error violin plot saved to {ERROR_VIOLIN_PLOT_PATH}")
    plt.show()