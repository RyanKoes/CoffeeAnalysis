from util import setup_mplt, DATADIR, PLOTDIR
from nn_0_synthetic_data_gen import build_model_data, generate_combined_data
from nn_1_train_model import CoffeeNetBase, train_coffeenet, evaluate_model

from collections import defaultdict
import tabulate
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tqdm import tqdm
from Models import VoltammogramConvNet, VoltammogramLSTMNet

import torch
import torch.nn as nn

if __name__ == "__main__":
    setup_mplt()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        print("Warning: No GPU found, using CPU for training. Abort.")
        exit()
    print(f"Using device: {device}")

    COMPLEX_MODELS = False
    SIMPLE_MODELS = True



    # do leave one out validation
    #kf = KFold(n_splits=10, shuffle=True, random_state=42)

    experiment_params =[
            {
                'NORMALIZE': True,
                'REDOX': False,
                'USE_BINS': False,
                'num_epochs': 1,
                'active': COMPLEX_MODELS,
                'network': lambda input_size: VoltammogramConvNet(input_size, num_outputs=3),
                'network_name': 'ConvNet'
            },
            {
                'NORMALIZE': True,
                'REDOX': False,
                'USE_BINS': False,
                'num_epochs': 1,
                'active': COMPLEX_MODELS,
                'network': lambda input_size: VoltammogramLSTMNet(input_size, num_outputs=3),
                'network_name': 'LSTMNet'
            },
            {
                'NORMALIZE': True,
                'REDOX': False,
                'USE_BINS': False,
                'num_epochs': 2000,
                'active': SIMPLE_MODELS,
                'network': lambda input_size: nn.Sequential(
                            nn.Linear(input_size, 1024),
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
                'USE_BINS': False,
                'num_epochs': 1000,
                'active': SIMPLE_MODELS,

                'network': lambda input_size:nn.Sequential(
                            nn.Linear(input_size, 1024),
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
                'USE_BINS': False,
                'num_epochs': 500,
                'active': SIMPLE_MODELS,

                'network': lambda input_size: nn.Sequential(
                            nn.Linear(input_size, 1024),
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
                'USE_BINS': False,
                'num_epochs': 500,
                'active': SIMPLE_MODELS,
                'network': lambda input_size: nn.Sequential(
                            nn.Linear(input_size, 1024),
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
                'USE_BINS': False,
                'num_epochs': 500,

                'active': SIMPLE_MODELS,
                'network': lambda input_size: nn.Sequential(
                            nn.Linear(input_size, 1024),
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
                'USE_BINS': False,
                'num_epochs': 300,
                'active': SIMPLE_MODELS,

                'network': lambda input_size: nn.Sequential(
                            nn.Linear(input_size, 256),
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
                'USE_BINS': False,
                'num_epochs': 500,
                'active': SIMPLE_MODELS,

                'network': lambda input_size: nn.Sequential(
                            nn.Linear(input_size, 256),
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
                'USE_BINS': False,
                'num_epochs': 1000,
                'active': SIMPLE_MODELS,

                'network': lambda input_size: nn.Sequential(
                            nn.Linear(input_size, 256),
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
                'USE_BINS': False,
                'num_epochs': 1000,
                'active': SIMPLE_MODELS,
                'add_noise': 10,
                'noise_level': 0.001,

                'network': lambda input_size: nn.Sequential(
                            nn.Linear(input_size, 256),
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
                'USE_BINS': False,
                'num_epochs': 1000,
                'active': SIMPLE_MODELS,
                'add_noise': 100,
                'noise_level': 0.001,

                'network': lambda input_size: nn.Sequential(
                            nn.Linear(input_size, 256),
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
                'USE_BINS': False,
                'num_epochs': 1000,
                'active': SIMPLE_MODELS,
                'add_noise': 200,
                'noise_level': 0.001,

                'network': lambda input_size: nn.Sequential(
                            nn.Linear(input_size, 256),
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
                'USE_BINS': False,
                'num_epochs': 1000,
                'active': SIMPLE_MODELS,
                'add_noise': 20,
                'noise_level': 0.005,

                'network': lambda input_size: nn.Sequential(
                            nn.Linear(input_size, 256),
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
                'USE_BINS': False,
                'num_epochs': 1000,
                'active': SIMPLE_MODELS,
                'add_noise': 30,
                'noise_level': 0.5,

                'network': lambda input_size: nn.Sequential(
                            nn.Linear(input_size, 256),
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
                'USE_BINS': False,
                'num_epochs': 1000,
                'active': SIMPLE_MODELS,
                'add_noise': 50,
                'noise_level': 5,

                'network': lambda input_size: nn.Sequential(
                            nn.Linear(input_size, 256),
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
                'USE_BINS': False,
                'num_epochs': 1000,
                'active': SIMPLE_MODELS,
                'add_noise': 100,
                'noise_level': .05,

                'network': lambda input_size: nn.Sequential(
                            nn.Linear(input_size, 256),
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
                'USE_BINS': False,
                'num_epochs': 300,
                'active': SIMPLE_MODELS,

                'network': lambda input_size: nn.Sequential(
                            nn.Linear(input_size, 3),
                        ),
                'network_name': 'nobins-3'
              },
                            {
                'NORMALIZE': True,
                'REDOX': False,
                'USE_BINS': False,
                'num_epochs': 1000,
                'active': SIMPLE_MODELS,

                'network': lambda input_size: nn.Sequential(
                            nn.Linear(input_size, 256),
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
                'USE_BINS': False,
                'num_epochs': 1000,
                'active': SIMPLE_MODELS,

                'network': lambda input_size: nn.Sequential(
                            nn.Linear(input_size, 256),
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
                'USE_BINS': False,
                'num_epochs': 1000,
                'active': SIMPLE_MODELS,
                'add_noise': 100,
                'noise_level': .01,
                'network': lambda input_size: nn.Sequential(
                            nn.Linear(input_size, 256),
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
                'USE_BINS': False,
                'num_epochs': 1000,
                'active': SIMPLE_MODELS,
                'add_noise': 200,
                'noise_level': .01,

                'network': lambda input_size: nn.Sequential(
                            nn.Linear(input_size, 256),
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

    for experiment in tqdm(experiment_params):

        if experiment['active'] == False:
            continue


        #experiment_name = '2comb10-'
        experiment_name = 'CoffeeNet'

        # normalization only affects bins.
        if experiment.get('bins') == True:
            experiment_name += 'NORM-' if experiment["NORMALIZE"] else "NONORM-"

        experiment_name += 'REDOX-' if experiment["REDOX"] else "OX-"

        if "add_noise" in experiment:
            if experiment["add_noise"]:
                experiment_name += f"NOISE{experiment['add_noise']}-{experiment['noise_level']}-"
            else:
                experiment_name += "NONOISE-"

        experiment_name += f"{experiment['network_name']}-{experiment['num_epochs']}"

        #print("-"*40)
        print(f"Running experiment: {experiment_name}", end="")

        all_data_path = DATADIR / f'{experiment_name}_all.pkl'

        # build experiment data
        if all_data_path.exists():
            print(f"All data for {experiment_name} already exists. Loading...")
            df_all = pd.read_pickle(all_data_path)

        else:
            #df_train, df_test = build_model_data(**experiment)
            #df_train_synthetic = generate_combined_data(df_train)

            df_all = build_model_data(test_train_split=False, **experiment)

            # store data
            df_all.to_pickle(all_data_path)


        #print (df_all[])

        target_names = ['HPLC_Caff', 'HPLC_CGA', 'TDS']



        coffees = df_all['Coffee Name'].unique()

        #for fold, (train_index, test_index) in enumerate(kf.split(X_all, y_all)):
        #for fold, (train_index, test_index) in enumerate(kf.split(coffees)):
        for fold, test_coffee in enumerate(coffees):

            print()
            print("-"*10)

            # if fold == 0:
            #     print (fold, flush=True, end="")
            # if fold == kf.get_n_splits() -1:
            #     print(f", {fold}", flush=True)
            # else:
            #     print(f", {fold}", end="", flush=True)

            # this is for kfolds
            #df_train = df_all[df_all['Coffee Name'].isin(coffees[train_index])]
            #df_test = df_all[df_all['Coffee Name'].isin(coffees[test_index])]

            test_mask = df_all['Coffee Name'] == test_coffee
            df_train = df_all[~test_mask]
            df_test = df_all[test_mask]

            # this stores all the experimental results and data to save across runs
            e = {
                'fold': fold,
                #'train_index': train_index,
                #'test_index': test_index,
                'test_coffee': test_coffee,
                'all_data_path': all_data_path,
                'experiment_name': experiment_name
            }

            # setup train data
            # X_train = X_all[train_index]
            # y_train = y_all[train_index]

            X_train = np.array(df_train['cv_raw'].to_list())
            y_train = np.array(df_train[target_names].values)

            input_size = X_train.shape[1]

            # X_mean = np.mean(X_train, axis = 1)

            # X_std = np.std(X_train, axis = 1)

            if 'add_noise' in experiment and experiment['add_noise']:
                noise_num = experiment['add_noise']
                noise_level = experiment['noise_level']

                X_list = []
                y_list = []

                for _ in range(noise_num):
                    noise = np.random.normal(
                        0,
                        noise_level,
                        X_train.shape
                    )

                    X_list.append(X_train + noise)
                    y_list.append(y_train)

                X_train = np.concatenate(X_list)
                y_train = np.concatenate(y_list)

            X_scaler = StandardScaler().fit(X_train)
            y_scaler = StandardScaler().fit(y_train)

            X_train_standard = X_scaler.transform(X_train)
            y_train_standard = y_scaler.transform(y_train)

            # setup test data
            # X_test = X_all[test_index]
            # y_test = y_all[test_index]

            X_test = np.array(df_test['cv_raw'].to_list())
            y_test = np.array(df_test[target_names].values)

            X_test_standard = X_scaler.transform(X_test)
            y_test_standard = y_scaler.transform(y_test)

            model = CoffeeNetBase()
            model.network = experiment['network'](input_size)

            # train the model
            model_path = DATADIR / f'{experiment_name}-fold-{fold}.pth'
            e['model_path'] = model_path
            if model_path.exists():
                #print(f"Model {model_path} already exists. Loading...")
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
            else:
                model.to(device)

                model = train_coffeenet(model,
                                        X_train_standard, y_train_standard,
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


            # evaluate the model
            predictions = evaluate_model(model, X_train_standard, y_train_standard)

            # un standardize
            predictions_original_scale = y_scaler.inverse_transform(predictions)

            #
            for i, name in enumerate(target_names):
                e[f'train_{name}_r2'] = r2_score(y_train[:, i], predictions_original_scale[:, i])
                e[f'train_{name}_mae'] = mean_absolute_error(y_train[:, i], predictions_original_scale[:, i])
                e[f'train_{name}_predictions'] = predictions_original_scale[:, i]
                e[f'train_{name}_actual'] = y_train[:, i]
                e[f'train_{name}_error_pct'] = 100.0 * (predictions_original_scale[:, i] - y_train[:, i]) / y_train[:, i]

            # evaluate on test data
            test_predictions = evaluate_model(model, X_test_standard, y_test_standard)

            predictions_original_scale_test = y_scaler.inverse_transform(test_predictions)

            for i, name in enumerate(target_names):
                e[f'test_{name}_r2'] = r2_score(y_test[:, i], predictions_original_scale_test[:, i])
                e[f'test_{name}_mae'] = mean_absolute_error(y_test[:, i], predictions_original_scale_test[:, i])
                e[f'test_{name}_predictions'] = predictions_original_scale_test[:, i]
                e[f'test_{name}_actual']= y_test[:, i]
                e[f'test_{name}_error_pct'] = 100.0 * (predictions_original_scale_test[:, i] - y_test[:, i]) / y_test[:, i]



            exp_results.append(e)



### AFTER RUNNING EVERYTHING PRINT THE RESULTS ###



df_results = pd.DataFrame(exp_results)

# add percent errors
for i, name in enumerate(target_names):
    df_results[f'train_{name}_error_pct_mean'] = df_results[f'train_{name}_error_pct'].apply(np.mean)
    df_results[f'test_{name}_error_pct_mean'] = df_results[f'test_{name}_error_pct'].apply(np.mean)


# aggregate by fold
df_avg = df_results[['experiment_name', 'fold',
                   'train_HPLC_Caff_r2', 'test_HPLC_Caff_r2',
                   'train_HPLC_CGA_r2', 'test_HPLC_CGA_r2',
                   'train_TDS_r2', 'test_TDS_r2',
                    ]].groupby(['experiment_name']).agg('mean')

df_avg.sort_values(by='test_HPLC_Caff_r2', ascending=False, inplace=True)

print(tabulate.tabulate(df_avg,
                   floatfmt=".4f",
                   headers='keys', tablefmt='psql'))

df_best = df_results[df_results['experiment_name'] == df_avg.iloc[0].name]

print(tabulate.tabulate(df_best[['experiment_name', 'fold',
                   'train_HPLC_Caff_r2', 'test_HPLC_Caff_r2',
                   'train_HPLC_CGA_r2', 'test_HPLC_CGA_r2',
                   'train_TDS_r2', 'test_TDS_r2'
                    ]], floatfmt=".4f",  headers='keys', tablefmt='psql'))


results_path = DATADIR / 'results.pkl'
df_results.to_pickle(results_path)