import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# === Load and preprocess data ===
df = pd.read_csv("green_coffees.csv")

# Clean altitude and compute averages
def parse_altitude(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().replace("masl", "").strip()
    # If range like "1200 - 1500", take mean of numeric parts
    if "-" in s:
        parts = []
        for x in s.split("-"):
            x = x.strip().replace(",", "")
            # keep digits and dots
            x_clean = ''.join([c for c in x if c.isdigit() or c == '.'])
            if x_clean:
                try:
                    parts.append(float(x_clean))
                except:
                    pass
        if len(parts) >= 2:
            return np.mean(parts[:2])
        elif len(parts) == 1:
            return parts[0]
        else:
            return np.nan
    # otherwise keep only digits and dot
    num = ''.join([c for c in s if c.isdigit() or c == '.'])
    try:
        return float(num) if num else np.nan
    except:
        return np.nan

df["Altitude (meters)"] = df["Altitude (meters)"].apply(parse_altitude)

# ----- Fix implausible altitudes: replace values > 2000 with 1500 -----
df.loc[df["Altitude (meters)"] > 2000, "Altitude (meters)"] = 1500.0

# Drop missing values for relevant columns
cols_needed = ["Country", "Process", "Altitude (meters)",
               "Spicy", "Chocolaty", "Nutty", "Buttery", "Fruity", "Flowery", "Winey", "Earthy"]
df = df.dropna(subset=cols_needed)

# === Define features (X) and targets (y) ===
X = df[["Spicy", "Chocolaty", "Nutty", "Buttery", "Fruity", "Flowery", "Winey", "Earthy"]]

# Encode categorical outputs
country_enc = LabelEncoder()
process_enc = LabelEncoder()

y_country = country_enc.fit_transform(df["Country"])
y_process = process_enc.fit_transform(df["Process"])
y_altitude = df["Altitude (meters)"].values

# === Standardize inputs ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Prepare 5-fold CV ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)

country_accs, process_accs, altitude_maes = [], [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_country_train, y_country_test = y_country[train_idx], y_country[test_idx]
    y_process_train, y_process_test = y_process[train_idx], y_process[test_idx]
    y_alt_train, y_alt_test = y_altitude[train_idx], y_altitude[test_idx]

    # One-hot encode categorical outputs
    y_country_train_cat = to_categorical(y_country_train)
    y_process_train_cat = to_categorical(y_process_train)

    # === Define model ===
    model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(y_country_train_cat.shape[1] + y_process_train_cat.shape[1] + 1)
    ])
    model.compile(optimizer='adam', loss='mse')  # combined numeric + categorical

    # === Train model ===
    model.fit(X_train, np.concatenate([
        y_country_train_cat,
        y_process_train_cat,
        y_alt_train.reshape(-1, 1)
    ], axis=1), epochs=100, batch_size=16, verbose=0)

    # === Predict ===
    y_pred = model.predict(X_test, verbose=0)

    # Split predictions into parts
    n_country = len(country_enc.classes_)
    n_process = len(process_enc.classes_)
    y_pred_country = np.argmax(y_pred[:, :n_country], axis=1)
    y_pred_process = np.argmax(y_pred[:, n_country:n_country + n_process], axis=1)
    y_pred_alt = y_pred[:, -1]

    # === Metrics ===
    country_acc = accuracy_score(y_country_test, y_pred_country)
    process_acc = accuracy_score(y_process_test, y_pred_process)
    alt_mae = mean_absolute_error(y_alt_test, y_pred_alt)

    country_accs.append(country_acc)
    process_accs.append(process_acc)
    altitude_maes.append(alt_mae)

    print(f"Fold {fold + 1}: Country Acc = {country_acc:.3f}, Process Acc = {process_acc:.3f}, Altitude MAE = {alt_mae:.1f}")

# === Average results ===
avg_country_acc = np.mean(country_accs)
avg_process_acc = np.mean(process_accs)
avg_alt_mae = np.mean(altitude_maes)

print("\n===== Cross-validation results =====")
print(f"Avg Country Accuracy: {avg_country_acc:.3f} ± {np.std(country_accs):.3f}")
print(f"Avg Process Accuracy: {avg_process_acc:.3f} ± {np.std(process_accs):.3f}")
print(f"Avg Altitude MAE: {avg_alt_mae:.1f} ± {np.std(altitude_maes):.1f}")

# === Plot per-fold results ===
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(country_accs, marker='o')
plt.title("Country Accuracy (per fold)")
plt.xlabel("Fold")
plt.ylabel("Accuracy")

plt.subplot(1, 3, 2)
plt.plot(process_accs, marker='o', color='orange')
plt.title("Process Accuracy (per fold)")
plt.xlabel("Fold")
plt.ylabel("Accuracy")

plt.subplot(1, 3, 3)
plt.plot(altitude_maes, marker='o', color='green')
plt.title("Altitude MAE (per fold)")
plt.xlabel("Fold")
plt.ylabel("Mean Abs Error")

plt.tight_layout()
plt.show()

# === Summary bar chart of average results ===
plt.figure(figsize=(6, 4))
metrics = ['Country Accuracy', 'Process Accuracy', 'Altitude MAE']
values = [avg_country_acc, avg_process_acc, avg_alt_mae]

# Plot accuracies and MAE (MAE is on altitude's meter scale)
plt.bar(metrics[:2], values[:2], color=['skyblue', 'orange'], label='Accuracy')
plt.bar(metrics[2:], values[2:], color='lightgreen', label='MAE')

plt.title("Average Test Performance (5-fold CV)")
plt.ylabel("Accuracy / MAE")
# set y-limit with cushion
plt.ylim(0, max(values) * 1.2 if max(values) > 1 else 1.2)
plt.tight_layout()
plt.show()
