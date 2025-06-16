import pandas as pd
import numpy as np
import torch

import tabulate

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

from util import setup_mplt, DATADIR, PLOTDIR


# origninal model
# class CoffeeNet(nn.Module):
#     def __init__(self, input_size, output_size=3):
#         super(CoffeeNet, self).__init__()
#         self.fc1 = nn.Linear(input_size, 128)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, output_size)

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# 128 x 5
# class CoffeeNet(nn.Module):
#     def __init__(self, input_size, output_size=3):
#         super(CoffeeNet, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, output_size)
#         )
#     def forward(self, x):
#         return self.network(x)


class CoffeeNet1024x5v2(nn.Module):
    def __init__(self, input_size, output_size=3):
        super(CoffeeNet1024x5v2, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, output_size)
        )
    def forward(self, x):
        return self.network(x)

class CoffeeNet1024x5raw(nn.Module):
    def __init__(self, input_size, output_size=3):
        super(CoffeeNet1024x5raw, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, output_size)
        )
    def forward(self, x):
        return self.network(x)

class CoffeeNet4096x7(nn.Module):
    def __init__(self, input_size, output_size=3):
        super(CoffeeNet4096x7, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size)
        )
    def forward(self, x):
        return self.network(x)

# Define the neural network use the GPU
# class CoffeeNet(nn.Module):
#     def __init__(self, input_size, output_size=3):
#         super(CoffeeNet, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_size, 512),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, output_size)
#         )

#     def forward(self, x):
#         return self.network(x)


def train_coffeenet(model, X, y, num_epochs=100):
    """
    Train a model to predict y vales from X values.
    Args:
        model: The neural network model to train.
        X: Input features, a 2D numpy array.
        y: Target values, a 2D numpy array.
        num_epochs: Number of epochs to train the model.
    Returns:
        model: The trained model.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    # For this task, we'll use the whole dataset for training and evaluation
    # In a real scenario, you'd split into train/validation/test
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            if epoch == 0 and epoch_loss == 0: # Print for the first batch only
                print(f"First batch data is on device: {batch_X.device}")
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}, GPU Mem: {torch.cuda.memory_allocated(device)/1024**2:.2f}MB / {torch.cuda.memory_reserved(device)/1024**2:.2f}MB')
    return model

def evaluate_training_model(model, X, y):
    """
    Use the trained model to predict y values from X values.
    returns the predictions and evaluation metrics.
    """

    # Evaluation on training data
    model.eval()
    with torch.no_grad():
        predictions_normalized = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu()

    out_dim = y.shape[1]
    return[
         (
        #   r2_score(targets_original_scale[:, i], predictions_original_scale[:, i]),
        #   mean_absolute_error(targets_original_scale[:, i], predictions_original_scale[:, i])
            predictions_normalized[:, i].numpy(),
            r2_score(y[:, i], predictions_normalized[:, i]),
            mean_absolute_error(y[:, i], predictions_normalized[:, i])
         ) for i in range(out_dim)
        ]

if __name__ == "__main__":
    setup_mplt()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        print("Warning: No GPU found, using CPU for training. Abort.")
        exit()
    print(f"Using device: {device}")

    #          HPLC_Caff     HPLC_CGA          TDS
    # count  4752.000000  4752.000000  4752.000000
    # mean    680.363636   701.818182     1.527273
    # std     296.389226   171.073517     0.137842
    # min      36.400000   360.000000     1.300000
    # 25%     464.750000   557.750000     1.464750
    # 50%     746.450000   725.000000     1.500000
    # 75%     882.050000   870.000000     1.546000
    # max    1219.550000   920.000000     2.030000
    # <class 'pandas.core.frame.DataFrame'>
    # RangeIndex: 4752 entries, 0 to 4751
    # Data columns (total 6 columns):
    #  #   Column       Non-Null Count  Dtype
    # ---  ------       --------------  -----
    #  0   Sample Name  4752 non-null   object
    #  1   Coffee Name  4752 non-null   object
    #  2   HPLC_Caff    4752 non-null   float64
    #  3   HPLC_CGA     4752 non-null   float64
    #  4   TDS          4752 non-null   float64
    #  5   cv_bins      4752 non-null   object
    # dtypes: float64(3), object(3)
    # memory usage: 222.9+ KB
    # None
    # Index(['Sample Name', 'Coffee Name', 'HPLC_Caff', 'HPLC_CGA', 'TDS',
    #        'cv_bins'],
    #       dtype='object')

    test_name = '2comb-10_bin64'
    #net_name= 'CoffeeNet4096x7'
    net_name= 'CoffeeNet1024x5raw'
    #num_epochs = 400
    num_epochs = 100

    target_names = ['HPLC_Caff', 'HPLC_CGA', 'TDS']
    # Load data

    df = pd.read_pickle(DATADIR / f'train_{test_name}.pkl')
    print(f"Loaded data from {DATADIR / f'train_{test_name}.pkl'}")
    print(df.describe())

    X = np.array(df['cv_raw'].tolist())
    y = df[target_names].values

    X_scaler = StandardScaler().fit(X)
    y_scaler = StandardScaler().fit(y)

    X_standard = X_scaler.transform(X)
    y_standard = y_scaler.transform(y)

    # this is the size of the x input to the model
    input_size = X.shape[1]

    MODEL_NAME = f'{net_name}_{num_epochs}_{test_name}'
    MODEL_PATH = DATADIR / f'{MODEL_NAME}.pth'

    #model = CoffeeNet4096x7(128).to(device)
    #model = CoffeeNet1024x5v2(64).to(device)
    model = CoffeeNet1024x5raw(input_size).to(device)

    if MODEL_PATH.exists():
        print(f"Model {MODEL_NAME} already exists. Loading...")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {MODEL_PATH}")
    else:

        # use x_list to predict y_values
        model = train_coffeenet(model,
                                X_standard,
                                y_standard,
                                num_epochs=num_epochs)

        # Save the model
        torch.save({
            'model_state_dict': model.state_dict(),
            'X_mean': X_scaler.mean_.tolist(),
            'X_std': X_scaler.scale_.tolist(),
            'y_mean': y_scaler.mean_.tolist(),
            'y_std': y_scaler.scale_.tolist(),
            'input_size': input_size
        }, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")


    predictions = evaluate_training_model(model, X_standard, y_standard)

    print(tabulate.tabulate(
        [(name, x[1], x[2]) for name, x in zip (target_names, predictions)],
        headers=['R2 Score', 'MAE'],
        tablefmt='pipe',
        floatfmt=".4f"
    ) )


    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    predictions_original_scale = y_scaler.inverse_transform(
        np.array([x[0] for x in predictions]).reshape(-1, len(target_names))
    )
    for i, name in enumerate(target_names):
        # targets_original_scale = y_scaler.inverse_transform(
        #     y[:, i].reshape(-1, 1)).flatten()

        axes[i].scatter(y[:, i], predictions_original_scale[:, i], alpha=0.5)

        axes[i].plot([y[:, i].min(), y[:, i].max()],
                     [y[:, i].min(), y[:, i].max()], 'r--')
        axes[i].set_xlabel(f"Actual {name}")
        axes[i].set_ylabel(f"Predicted {name}")
        axes[i].set_title(f"Actual vs. Predicted {name}\nR2: {predictions[i][1]:.2f}, MAE: {predictions[i][2]:.2f}")
        axes[i].grid(True)

        print(f"{name} - R2 Score: {predictions[i][1]:.4f}, MAE: {predictions[i][2]:.4f}")



    plt.tight_layout()
    plt.savefig(PLOTDIR / f'{MODEL_NAME}_training_evaluation.pdf')
    print(f"Evaluation plot saved to {PLOTDIR / f'{MODEL_NAME}_training_evaluation.pdf'}")
    plt.show()
