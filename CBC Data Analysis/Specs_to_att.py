import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === Load dataset ===
df = pd.read_csv("green_coffees.csv")

# Keep relevant columns
cols = ['Country', 'Altitude (meters)', 'Process',
        'Brightness', 'Body', 'Aroma', 'Complexity', 'Balance', 'Sweetness']
df = df[cols]
df = df.dropna(subset=['Country', 'Altitude (meters)', 'Process'])

# Parse altitude into numeric mean
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

# Replace altitudes > 2000 with 1500
df.loc[df['Altitude (meters)'] > 2000, 'Altitude (meters)'] = 1500

df = df.dropna(subset=['Altitude (meters)'])

# Features & targets
X = df[['Country', 'Altitude (meters)', 'Process']]
y = df[['Brightness', 'Body', 'Aroma', 'Complexity', 'Balance', 'Sweetness']]

# Preprocessor
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Country', 'Process']),
    ('num', StandardScaler(), ['Altitude (meters)'])
])

X_processed = preprocessor.fit_transform(X)

# === 5-Fold Cross-Validation ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores_global, mae_scores_global = [], []
taste_mae_all, taste_r2_all = [], []

fold = 1
for train_idx, test_idx in kf.split(X_processed):
    print(f"\n=== Fold {fold} ===")
    X_train, X_test = X_processed[train_idx], X_processed[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Define model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_processed.shape[1],)),
        Dense(32, activation='relu'),
        Dense(y.shape[1], activation='linear')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=8,
        verbose=0,
        callbacks=[es]
    )

    y_pred = model.predict(X_test)

    # Global metrics
    r2_global = r2_score(y_test, y_pred, multioutput='uniform_average')
    mae_global = mean_absolute_error(y_test, y_pred)

    # Per-attribute metrics
    taste_mae = np.mean(np.abs(y_test.values - y_pred), axis=0)
    taste_r2 = [r2_score(y_test.iloc[:, i], y_pred[:, i]) for i in range(y.shape[1])]

    taste_mae_all.append(taste_mae)
    taste_r2_all.append(taste_r2)

    r2_scores_global.append(r2_global)
    mae_scores_global.append(mae_global)

    print(f"Fold {fold} → Global R²: {r2_global:.3f}, Global MAE: {mae_global:.3f}")
    print("Per-attribute results:")
    for name, mae_val, r2_val in zip(y.columns, taste_mae, taste_r2):
        print(f"  {name:10s} → MAE: {mae_val:.3f}, R²: {r2_val:.3f}")

    fold += 1

# === Summary statistics ===
print("\n=== Cross-Validation Summary ===")
print(f"Average Global R²: {np.mean(r2_scores_global):.3f} ± {np.std(r2_scores_global):.3f}")
print(f"Average Global MAE: {np.mean(mae_scores_global):.3f} ± {np.std(mae_scores_global):.3f}")

# Average per-attribute metrics
taste_mae_mean = np.mean(taste_mae_all, axis=0)
taste_r2_mean = np.mean(taste_r2_all, axis=0)

print("\nAverage Per-Attribute Results:")
for name, mae_val, r2_val in zip(y.columns, taste_mae_mean, taste_r2_mean):
    print(f"  {name:10s} → MAE: {mae_val:.3f}, R²: {r2_val:.3f}")

# === Plot per-attribute MAE and R² ===
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(y.columns, taste_mae_mean, color='skyblue')
plt.title("Average MAE per Attribute (5-Fold CV)")
plt.ylabel("MAE")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.bar(y.columns, taste_r2_mean, color='lightgreen')
plt.title("Average R² per Attribute (5-Fold CV)")
plt.ylabel("R²")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
