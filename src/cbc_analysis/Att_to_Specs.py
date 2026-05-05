import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === Load dataset ===
df = pd.read_csv("green_coffees.csv")

# Keep relevant columns
cols = ['Country', 'Altitude (meters)', 'Process',
        'Brightness', 'Body', 'Aroma', 'Complexity', 'Balance', 'Sweetness']
df = df[cols]
df = df.dropna(subset=['Country', 'Altitude (meters)', 'Process'])

# Parse altitude strings into numeric averages
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

# === Features (tasting) and targets (specs) ===
X = df[['Brightness', 'Body', 'Aroma', 'Complexity', 'Balance', 'Sweetness']]
y_country = df['Country']
y_process = df['Process']
y_altitude = df['Altitude (meters)']

# Encode categorical targets
enc_country = LabelEncoder()
enc_process = LabelEncoder()

y_country_enc = enc_country.fit_transform(y_country)
y_process_enc = enc_process.fit_transform(y_process)

num_countries = len(enc_country.classes_)
num_processes = len(enc_process.classes_)

# Scale input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Cross-validation setup ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

r2_scores, country_accs, process_accs = [], [], []

for train_idx, test_idx in kf.split(X_scaled):
    print(f"\n=== Fold {fold} ===")

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_country_train, y_country_test = y_country_enc[train_idx], y_country_enc[test_idx]
    y_process_train, y_process_test = y_process_enc[train_idx], y_process_enc[test_idx]
    y_alt_train, y_alt_test = y_altitude.iloc[train_idx], y_altitude.iloc[test_idx]

    # === Multi-task neural network ===
    inp = Input(shape=(X_scaled.shape[1],))

    shared = Dense(64, activation='relu')(inp)
    shared = Dense(32, activation='relu')(shared)

    # Outputs
    out_country = Dense(num_countries, activation='softmax', name='country')(shared)
    out_process = Dense(num_processes, activation='softmax', name='process')(shared)
    out_alt = Dense(1, activation='linear', name='altitude')(shared)

    model = Model(inputs=inp, outputs=[out_country, out_process, out_alt])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'country': 'sparse_categorical_crossentropy',
            'process': 'sparse_categorical_crossentropy',
            'altitude': 'mse'
        },
        metrics={
            'country': 'accuracy',
            'process': 'accuracy',
            'altitude': 'mae'
        }
    )

    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)

    history = model.fit(
        X_train,
        {'country': y_country_train, 'process': y_process_train, 'altitude': y_alt_train},
        validation_split=0.2,
        epochs=100,
        batch_size=8,
        verbose=0,
        callbacks=[es]
    )

    # === Evaluate ===
    preds = model.predict(X_test)
    pred_country = np.argmax(preds[0], axis=1)
    pred_process = np.argmax(preds[1], axis=1)
    pred_alt = preds[2].flatten()

    # Metrics
    acc_country = accuracy_score(y_country_test, pred_country)
    acc_process = accuracy_score(y_process_test, pred_process)
    r2_alt = r2_score(y_alt_test, pred_alt)

    country_accs.append(acc_country)
    process_accs.append(acc_process)
    r2_scores.append(r2_alt)

    print(f"Fold {fold} → Country acc: {acc_country:.3f}, Process acc: {acc_process:.3f}, Altitude R²: {r2_alt:.3f}")

    fold += 1

# === Summary ===
print("\n=== Cross-Validation Results ===")
print(f"Average Country Accuracy: {np.mean(country_accs):.3f} ± {np.std(country_accs):.3f}")
print(f"Average Process Accuracy: {np.mean(process_accs):.3f} ± {np.std(process_accs):.3f}")
print(f"Average Altitude R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
