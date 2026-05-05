import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === Load dataset ===
df = pd.read_csv("green_coffees.csv")

# Keep relevant columns
cols = ['Altitude (meters)',
        'Brightness', 'Body', 'Aroma', 'Complexity', 'Balance', 'Sweetness',
        'Spicy', 'Chocolaty', 'Nutty', 'Buttery', 'Fruity', 'Flowery', 'Winey', 'Earthy']
df = df[cols]

# Drop missing values in key columns
df = df.dropna(subset=['Altitude (meters)'] + cols[1:])

# --- Parse altitude strings into numeric averages ---
def parse_altitude(alt):
    if isinstance(alt, str):
        alt = alt.replace('masl', '').replace('m', '').replace(',', '').strip()
        if '-' in alt:
            parts = alt.split('-')
            try:
                vals = [float(p) for p in parts]
                return np.mean(vals)
            except:
                return np.nan
        else:
            try:
                return float(alt)
            except:
                return np.nan
    return np.nan

df['Altitude (meters)'] = df['Altitude (meters)'].apply(parse_altitude)

# --- Replace unrealistic altitude values ---
df.loc[df['Altitude (meters)'] > 2000, 'Altitude (meters)'] = 1500
df = df.dropna(subset=['Altitude (meters)'])

# === Features and targets ===
input_cols = ['Altitude (meters)', 'Brightness', 'Body', 'Aroma', 'Complexity', 'Balance', 'Sweetness']
target_cols = ["Spicy", "Chocolaty", "Nutty", "Buttery",
               "Fruity", "Flowery", "Winey", "Earthy"]

X = df[input_cols].values
y = df[target_cols].values

# Standardize inputs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 5-Fold Cross-Validation ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_mae_all, fold_r2_all = [], []

fold_num = 1
for train_idx, test_idx in kf.split(X_scaled):
    print(f"\n=== Fold {fold_num} ===")
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Define model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(y_train.shape[1], activation='linear')
    ])

    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)

    # Train model
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=8,
        verbose=0,
        callbacks=[es]
    )

    # Predict
    y_pred = model.predict(X_test)

    # Compute per-taste MAE and R²
    mae_per_taste = np.mean(np.abs(y_test - y_pred), axis=0)
    r2_per_taste = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(y.shape[1])]

    fold_mae_all.append(mae_per_taste)
    fold_r2_all.append(r2_per_taste)

    # Print fold results
    print("Per-taste results:")
    for name, mae_val, r2_val in zip(target_cols, mae_per_taste, r2_per_taste):
        print(f"  {name:10s} → MAE: {mae_val:.3f}, R²: {r2_val:.3f}")

    fold_num += 1

# === Average across folds ===
mae_avg = np.mean(fold_mae_all, axis=0)
r2_avg = np.mean(fold_r2_all, axis=0)

print("\n=== Average Results Across Folds ===")
for name, mae_val, r2_val in zip(target_cols, mae_avg, r2_avg):
    print(f"  {name:10s} → MAE: {mae_val:.3f}, R²: {r2_val:.3f}")

# === Plot per-taste MAE and R² ===
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.bar(target_cols, mae_avg, color='skyblue')
plt.title("Average MAE per Taste Attribute")
plt.ylabel("MAE")
plt.xticks(rotation=45)

plt.subplot(1,2,2)
plt.bar(target_cols, r2_avg, color='lightgreen')
plt.title("Average R² per Taste Attribute")
plt.ylabel("R²")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
