import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

# === Load and preprocess data ===
df = pd.read_csv("green_coffees.csv")

# --- Drop rows missing required data ---
cols_needed = [
    "Spicy", "Chocolaty", "Nutty", "Buttery",
    "Fruity", "Flowery", "Winey", "Earthy",
    "Brightness", "Body", "Aroma", "Complexity", "Balance", "Sweetness"
]
df = df.dropna(subset=cols_needed)

# === Define features (X) and regression targets (y) ===
X = df[["Spicy", "Chocolaty", "Nutty", "Buttery",
        "Fruity", "Flowery", "Winey", "Earthy"]]
y = df[["Brightness", "Body", "Aroma", "Complexity", "Balance", "Sweetness"]]

# === Standardize inputs and outputs ===
X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

# === Prepare 5-fold CV ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)

all_mae = []
all_r2 = []

# Store per-attribute results
attr_mae = np.zeros((kf.get_n_splits(), y.shape[1]))
attr_r2 = np.zeros((kf.get_n_splits(), y.shape[1]))

for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]

    # === Define regression model ===
    model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(y_train.shape[1], activation='linear')  # six continuous outputs
    ])
    model.compile(optimizer='adam', loss='mse')

    # === Train model ===
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)

    # === Predict and inverse transform ===
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_test)

    # === Metrics (overall) ===
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    all_mae.append(mae)
    all_r2.append(r2)

    # === Metrics (per attribute) ===
    for i, col in enumerate(y.columns):
        attr_mae[fold, i] = mean_absolute_error(y_true[:, i], y_pred[:, i])
        attr_r2[fold, i] = r2_score(y_true[:, i], y_pred[:, i])

    print(f"Fold {fold + 1}: MAE = {mae:.3f}, R² = {r2:.3f}")

# === Average results ===
mean_mae = np.mean(attr_mae, axis=0)
mean_r2 = np.mean(attr_r2, axis=0)

print("\n===== Cross-validation results =====")
for i, col in enumerate(y.columns):
    print(f"{col}: MAE = {mean_mae[i]:.3f}, R² = {mean_r2[i]:.3f}")

print(f"\nOverall Avg MAE: {np.mean(all_mae):.3f} ± {np.std(all_mae):.3f}")
print(f"Overall Avg R²: {np.mean(all_r2):.3f} ± {np.std(all_r2):.3f}")

# === Plot results ===
plt.figure(figsize=(10, 4))

# Bar plot for R²
plt.subplot(1, 2, 1)
plt.bar(y.columns, mean_r2, color='skyblue')
plt.ylim(0, 1)
plt.title("Average R² per Attribute (5-fold CV)")
plt.ylabel("R² Score")
plt.xticks(rotation=45)

# Bar plot for MAE
plt.subplot(1, 2, 2)
plt.bar(y.columns, mean_mae, color='lightcoral')
plt.title("Average MAE per Attribute (5-fold CV)")
plt.ylabel("Mean Absolute Error")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
