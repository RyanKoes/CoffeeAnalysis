import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from util import DATADIR, setup_mplt
from nn_0_synthetic_data_gen import build_model_data
from nn_1_train_model import CoffeeNetBase, evaluate_model
from nn_model_window_search import get_network_architectures


def main():
    setup_mplt()

    # Load checkpoint
    ckpt_path = DATADIR / "BEST_TDS.pkl"
    # Allow full unpickling since this is a local, trusted file
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    metadata = checkpoint["metadata"]
    state_dict = checkpoint["model_state_dict"]

    print("Loaded BEST_TDS checkpoint from:", ckpt_path)
    print("Metadata:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")

    vmin, vmax = metadata["window"]
    arch_name = metadata["architecture"]

    # Rebuild data (same settings as search: full raw CV, no normalization, no bins)
    df_all = build_model_data(
        NORMALIZE=False,
        REDOX=False,
        USE_BINS=False,
        test_train_split=False,
    )

    # Voltage grid and indices for best window
    n_points = len(df_all["cv_raw"].iloc[0])
    full_voltage_array = np.linspace(0.0, 2.0, n_points)
    voltage_mask = (full_voltage_array >= vmin) & (full_voltage_array <= vmax)
    voltage_indices = np.where(voltage_mask)[0]

    # Get matching architecture
    architectures = get_network_architectures()
    arch = [a for a in architectures if a["network_name"] == arch_name][0]

    coffees = df_all["Coffee Name"].unique()

    train_actual, train_pred = [], []
    test_actual, test_pred = [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for test_coffee in coffees:
        test_mask = df_all["Coffee Name"] == test_coffee
        df_train = df_all[~test_mask]
        df_test = df_all[test_mask]

        X_train = np.array([
            np.asarray(cv)[voltage_indices]
            for cv in df_train["cv_raw"]
        ])
        X_test = np.array([
            np.asarray(cv)[voltage_indices]
            for cv in df_test["cv_raw"]
        ])

        y_train = df_train["TDS"].values.reshape(-1, 1)
        y_test = df_test["TDS"].values.reshape(-1, 1)

        X_scaler = StandardScaler().fit(X_train)
        y_scaler = StandardScaler().fit(y_train)

        X_train_s = X_scaler.transform(X_train)
        y_train_s = y_scaler.transform(y_train)
        X_test_s = X_scaler.transform(X_test)
        y_test_s = y_scaler.transform(y_test)

        model = CoffeeNetBase()
        model.network = arch["network"](X_train.shape[1])
        model.load_state_dict(state_dict)
        model.to(device)

        train_out = evaluate_model(model, X_train_s, y_train_s)
        test_out = evaluate_model(model, X_test_s, y_test_s)

        train_pred.extend(y_scaler.inverse_transform(train_out).flatten())
        train_actual.extend(y_train.flatten())

        test_pred.extend(y_scaler.inverse_transform(test_out).flatten())
        test_actual.extend(y_test.flatten())

    # Compute R^2 for test dots
    test_r2 = r2_score(test_actual, test_pred)
    print(f"Test R^2 from recomputed predictions: {test_r2:.4f}")

    # Plot actual vs predicted TDS
    plt.figure(figsize=(6, 6))

    plt.scatter(train_actual, train_pred, alpha=0.6, label="Train")
    plt.scatter(test_actual, test_pred, alpha=0.6, label="Test")

    lims = [
        min(train_actual + test_actual),
        max(train_actual + test_actual),
    ]
    plt.plot(lims, lims, "k--")

    plt.xlabel("Actual TDS")
    plt.ylabel("Predicted TDS")
    plt.title(
        f"BEST_TDS: {arch_name}\nWindow: {vmin}-{vmax} V\nTest R^2: {test_r2:.4f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
