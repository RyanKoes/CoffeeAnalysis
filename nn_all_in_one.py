from util import setup_mplt, DATADIR, PLOTDIR
from nn_0_synthetic_data_gen import build_model_data, generate_combined_data
from nn_1_train_model import CoffeeNetBase, train_coffeenet, evaluate_model

from collections import defaultdict
import tabulate
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

if __name__ == "__main__":
    setup_mplt()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        print("Warning: No GPU found, using CPU for training. Abort.")
        exit()
    print(f"Using device: {device}")

    experiment_params =[ {
                'NORMALIZE': True,
                'REDOX': False,
                'BINS': 64,
                'USE_BINS': True,
                'num_epochs': 100,

                'network': nn.Sequential(
                            nn.Linear(64, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(128, 1024),
                            nn.BatchNorm1d(1024),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(1024, 1024),
                            nn.BatchNorm1d(1024),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(1024, 64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(64, 3)
                        ),
                'network_name': '1024x5'
              },

              {
                'NORMALIZE': True,
                'REDOX': False,
                'BINS': 64,
                'USE_BINS': True,
                'num_epochs': 200,

                'network': nn.Sequential(
                            nn.Linear(64, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(128, 64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(64, 3)
                        ),
                'network_name': '64-128-64'
              },
            {
                'NORMALIZE': True,
                'REDOX': False,
                'BINS': 64,
                'USE_BINS': False,
                'num_epochs': 200,
                'input_layer': lambda input_size: nn.Linear(input_size, 1024),

                'network': nn.Sequential(
                            # first layer will be added later
                            nn.BatchNorm1d(1024),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(1024, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(128, 64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(64, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(128, 3)
                        ),
                'network_name': 'nobins-1024-128-64-128-3'
              },
              {
                'NORMALIZE': True,
                'REDOX': False,
                'BINS': 64,
                'USE_BINS': False,
                'num_epochs': 100,
                'input_layer': lambda input_size: nn.Linear(input_size, 1024),

                'network': nn.Sequential(
                            # first layer will be added later
                            nn.BatchNorm1d(1024),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(1024, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(128, 64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(64, 3)
                        ),
                'network_name': 'nobins-1024-128-64-3'
              },

            {
                'NORMALIZE': True,
                'REDOX': False,
                'BINS': 64,
                'USE_BINS': False,
                'num_epochs': 300,
                'input_layer': lambda input_size: nn.Linear(input_size, 1024),

                'network': nn.Sequential(
                            # first layer will be added later
                            nn.BatchNorm1d(1024),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(1024, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(128, 64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(64, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(128, 3)
                        ),
                'network_name': 'nobins-1024-128-64-128-3'
              },
              {
                'NORMALIZE': True,
                'REDOX': False,
                'BINS': 64,
                'USE_BINS': False,
                'num_epochs': 300,
                'input_layer': lambda input_size: nn.Linear(input_size, 1024),

                'network': nn.Sequential(
                            # first layer will be added later
                            nn.BatchNorm1d(1024),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(1024, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(128, 64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(64, 3)
                        ),
                'network_name': 'nobins-1024-128-64-3'
              },
              {
                'NORMALIZE': True,
                'REDOX': False,
                'BINS': 64,
                'USE_BINS': False,
                'num_epochs': 300,
                'input_layer': lambda input_size: nn.Linear(input_size, 1024),

                'network': nn.Sequential(
                            # first layer will be added later
                            nn.BatchNorm1d(1024),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(1024, 3)
                        ),
                'network_name': 'nobins-1024-3'
              },
              {
                'NORMALIZE': True,
                'REDOX': False,
                'BINS': 64,
                'USE_BINS': False,
                'num_epochs': 300,
                'input_layer': lambda input_size: nn.Linear(input_size, 256),

                'network': nn.Sequential(
                            # first layer will be added later
                            nn.BatchNorm1d(256),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(256, 3)
                        ),
                'network_name': 'nobins-256-3'
              },
                            {
                'NORMALIZE': True,
                'REDOX': False,
                'BINS': 64,
                'USE_BINS': False,
                'num_epochs': 500,
                'input_layer': lambda input_size: nn.Linear(input_size, 256),

                'network': nn.Sequential(
                            # first layer will be added later
                            nn.BatchNorm1d(256),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(256, 3)
                        ),
                'network_name': 'nobins-256-3'
              },
                            {
                'NORMALIZE': True,
                'REDOX': False,
                'BINS': 64,
                'USE_BINS': False,
                'num_epochs': 1000,
                'input_layer': lambda input_size: nn.Linear(input_size, 256),

                'network': nn.Sequential(
                            # first layer will be added later
                            nn.BatchNorm1d(256),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(256, 3)
                        ),
                'network_name': 'nobins-256-3'
              },
              {
                'NORMALIZE': True,
                'REDOX': False,
                'BINS': 64,
                'USE_BINS': False,
                'num_epochs': 300,
                'input_layer': lambda input_size: nn.Linear(input_size, 3),

                'network': nn.Sequential(
                            # first layer will be added later

                        ),
                'network_name': 'nobins-3'
              },
                            {
                'NORMALIZE': True,
                'REDOX': False,
                'BINS': 64,
                'USE_BINS': False,
                'num_epochs': 300,
                'input_layer': lambda input_size: nn.Linear(input_size, 256),

                'network': nn.Sequential(
                            # first layer will be added later
                            nn.BatchNorm1d(256),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(256, 64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(64, 3)
                        ),
                'network_name': 'nobins-256-64-3'
              },
                                          {
                'NORMALIZE': True,
                'REDOX': False,
                'BINS': 64,
                'USE_BINS': False,
                'num_epochs': 500,
                'input_layer': lambda input_size: nn.Linear(input_size, 256),

                'network': nn.Sequential(
                            # first layer will be added later
                            nn.BatchNorm1d(256),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(256, 64),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Dropout(0.1),

                            nn.Linear(64, 3)
                        ),
                'network_name': 'nobins-256-64-3'
              },
    ]


    exp_results = []

    for experiment in experiment_params:

        # this stores all the experimental results and data to save across runs
        e = {}

        experiment_name = '2comb10-'
        experiment_name += 'NORM-' if experiment["NORMALIZE"] else "NONORM-"
        experiment_name += 'REDOX-' if experiment["REDOX"] else "OX-"
        experiment_name += str(experiment["BINS"]) + '-'
        experiment_name += f"{experiment['network_name']}-{experiment['num_epochs']}"

        e['experiment_name'] = experiment_name


        print("-"*40)
        print(f"Running experiment: {experiment_name}")

        train_data_path = DATADIR / f'{experiment_name}_train.pkl'
        test_data_path = DATADIR / f'{experiment_name}_test.pkl'

        e['train_data_path'] = train_data_path
        e['test_data_path'] = test_data_path

        # build experiment data

        if test_data_path.exists() and train_data_path.exists():
            print(f"Test/train data for {experiment_name} already exists. Loading...")
            df_test = pd.read_pickle(test_data_path)
            df_train_synthetic = pd.read_pickle(train_data_path)

        else:
            df_train, df_test = build_model_data(**experiment)
            df_train_synthetic = generate_combined_data(df_train)

            # store data
            df_test.to_pickle(test_data_path)
            df_train_synthetic.to_pickle(train_data_path)

        target_names = ['HPLC_Caff', 'HPLC_CGA', 'TDS']

        # setup train data
        if experiment['USE_BINS']:
            X = np.array(df_train_synthetic['cv_bins'].tolist())
        else:
            X = np.array(df_train_synthetic['cv_raw'].tolist())
        y = df_train_synthetic[target_names].values

        input_size = X.shape[1]

        X_scaler = StandardScaler().fit(X)
        y_scaler = StandardScaler().fit(y)

        X_standard = X_scaler.transform(X)
        y_standard = y_scaler.transform(y)

        # setup test data
        if experiment['USE_BINS']:
            X_test = np.array(df_test['cv_bins'].tolist())
        else:
            X_test = np.array(df_test['cv_raw'].tolist())
        y_test = df_test[target_names].values

        X_test_standard = X_scaler.transform(X_test)
        y_test_standard = y_scaler.transform(y_test)

        # train the model
        model = CoffeeNetBase()
        model.network = experiment['network']

        if 'input_layer' in experiment:
            model.network.insert(0,experiment['input_layer'](input_size))

        model.to(device)

        model_path = DATADIR / f'{experiment_name}.pth'

        if model_path.exists():
            print(f"Model {model_path} already exists. Loading...")
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model = train_coffeenet(model,
                                    X_standard, y_standard,
                                    X_test_standard, y_test_standard,
                                    num_epochs=experiment['num_epochs'])

            # Save the model
            torch.save({
                'model_state_dict': model.state_dict(),
                'X_mean': X_scaler.mean_.tolist(),
                'X_std': X_scaler.scale_.tolist(),
                'y_mean': y_scaler.mean_.tolist(),
                'y_std': y_scaler.scale_.tolist(),
                'input_size': input_size
            }, model_path)

        e['model_path'] = model_path

        # evaluate the model
        predictions = evaluate_model(model, X_standard, y_standard)

        # un standardize
        predictions_original_scale = y_scaler.inverse_transform(predictions)

        #
        for i, name in enumerate(target_names):
            e[f'train_{name}_r2'] = r2_score(y[:, i], predictions_original_scale[:, i])
            e[f'train_{name}_mae'] = mean_absolute_error(y[:, i], predictions_original_scale[:, i])
            e[f'train_{name}_predictions'] = predictions_original_scale[:, i]
            e[f'train_{name}_actual'] = y[:, i]
            e[f'train_{name}_error_pct'] = 100.0 * (predictions_original_scale[:, i] - y[:, i]) / y[:, i]

        # evaluate on test data
        test_predictions = evaluate_model(model, X_test_standard, y_test_standard)

        predictions_original_scale_test = y_scaler.inverse_transform(test_predictions)

        for i, name in enumerate(target_names):
            e[f'test_{name}_r2'] = r2_score(y_test[:, i], predictions_original_scale_test[:, i])
            e[f'test_{name}_mae'] = mean_absolute_error(y_test[:, i], predictions_original_scale_test[:, i])
            e[f'test_{name}_predictions'] = predictions_original_scale_test[:, i]
            e[f'test_{name}_actual']= y_test[:, i]
            e[f'test_{name}_error_pct'] = 100.0 * (predictions_original_scale_test[:, i] - y_test[:, i]) / y_test[:, i]

        print("-"*40)

        exp_results.append(e)



### AFTER RUNNING EVERYTHING PRINT THE RESULTS ###



df_results = pd.DataFrame(exp_results)

# add percent errors
for i, name in enumerate(target_names):
    df_results[f'train_{name}_error_pct_mean'] = df_results[f'train_{name}_error_pct'].apply(np.mean)
    df_results[f'test_{name}_error_pct_mean'] = df_results[f'test_{name}_error_pct'].apply(np.mean)


print(tabulate.tabulate(df_results [['experiment_name',
                   'train_HPLC_Caff_r2', 'test_HPLC_Caff_r2', #'train_HPLC_Caff_error_pct_mean', 'test_HPLC_Caff_error_pct_mean',
                   'train_HPLC_CGA_r2', 'test_HPLC_CGA_r2', #'train_HPLC_CGA_error_pct_mean', 'test_HPLC_CGA_error_pct_mean',
                   'train_TDS_r2', 'test_TDS_r2', #'train_TDS_error_pct_mean', 'test_TDS_error_pct_mean']],
                    ]],
                   floatfmt=".4f",
                   headers='keys', tablefmt='psql'))

