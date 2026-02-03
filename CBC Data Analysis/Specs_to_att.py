import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
df = pd.read_csv("green_coffees.csv")

# --- Data Cleaning ---
# Keep relevant columns
cols = ['Country', 'Altitude (meters)', 'Process',
        'Brightness', 'Body', 'Aroma', 'Complexity', 'Balance', 'Sweetness']
df = df[cols]

# Drop rows with missing Country, Process, or Altitude
df = df.dropna(subset=['Country', 'Altitude (meters)', 'Process'])

# Function to parse altitude and return average as a float
def parse_altitude(alt):
    if isinstance(alt, str):
        # Remove non-numeric characters except '-' and ','
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
df = df.dropna(subset=['Altitude (meters)'])

# Define features (X) and targets (y)
X = df[['Country', 'Altitude (meters)', 'Process']]
y = df[['Brightness', 'Body', 'Aroma', 'Complexity', 'Balance', 'Sweetness']]

# --- Preprocessing ---
# Encode categorical features and scale numeric
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Country', 'Process']),
    ('num', StandardScaler(), ['Altitude (meters)'])
])

X_processed = preprocessor.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# --- Neural Network ---
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_processed.shape[1],)),
    Dense(32, activation='relu'),
    Dense(y.shape[1], activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=0)

# --- Evaluation ---
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
mse = mean_squared_error(y_test, y_pred)
print(f"R² Score: {r2:.3f}")
print(f"MSE: {mse:.3f}")

# --- Plot training history ---
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()

# --- Plot predictions vs actuals ---
for i, col in enumerate(y.columns):
    plt.figure(figsize=(5,5))
    plt.scatter(y_test.iloc[:, i], y_pred[:, i], alpha=0.6)
    plt.plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
             [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
             color='red', linestyle='--')
    plt.xlabel(f"Actual {col}")
    plt.ylabel(f"Predicted {col}")
    plt.title(f"{col}: Actual vs Predicted")
    plt.show()

# --- Display prediction comparison table ---
comparison = y_test.copy()
comparison['Predicted_Brightness'] = y_pred[:,0]
comparison['Predicted_Body'] = y_pred[:,1]
comparison['Predicted_Aroma'] = y_pred[:,2]
comparison['Predicted_Complexity'] = y_pred[:,3]
comparison['Predicted_Balance'] = y_pred[:,4]
comparison['Predicted_Sweetness'] = y_pred[:,5]
print(comparison.head())
