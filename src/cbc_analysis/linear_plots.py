import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm
import re
import os

# === Columns to analyze ===
columns_to_use = [
    "Brightness", "Body", "Aroma", "Complexity", "Balance",
    "Sweetness", "Spicy", "Chocolaty", "Nutty", "Buttery",
    "Fruity", "Flowery", "Winey", "Altitude (meters)"
]

# === Load CSV ===
file_path = "green_coffees.csv"  # ⬅️ update this to your actual filename
df = pd.read_csv(file_path)

# === Keep only relevant columns ===
df = df[[c for c in columns_to_use if c in df.columns]].copy()

# === Parse altitude (average if range, ignore >5000) ===
if "Altitude (meters)" in df.columns:
    def parse_altitude(value):
        if pd.isna(value):
            return np.nan
        value = str(value).replace(",", "")
        nums = re.findall(r"[\d\.]+", value)
        if not nums:
            return np.nan
        nums = [float(x) for x in nums]
        return np.mean(nums)


    tqdm.pandas(desc="Parsing altitudes")
    df["Altitude (meters)"] = df["Altitude (meters)"].progress_apply(parse_altitude)
    df = df[df["Altitude (meters)"] <= 5000]

# === Convert other columns to numeric ===
for col in tqdm(columns_to_use, desc="Coercing numeric columns"):
    if col != "Altitude (meters)" and col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# === Drop rows with missing numeric data ===
df_clean = df.dropna(subset=columns_to_use)
print(f"\nRows before cleaning: {len(df)}, after cleaning: {len(df_clean)}")

# === Prepare output folders ===
os.makedirs("graphs", exist_ok=True)
r2_results = []

# === Pairwise linear regressions ===
numeric_cols = [c for c in columns_to_use if c in df_clean.columns]
for i, x_col in enumerate(tqdm(numeric_cols, desc="Outer loop (X variable)")):
    for y_col in tqdm(numeric_cols, desc=f"Inner loop (Y vs {x_col})", leave=False):
        if x_col == y_col:
            continue

        X = df_clean[[x_col]].values
        y = df_clean[y_col].values

        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        r2_results.append((x_col, y_col, r2))

        # === Plot ===
        plt.figure(figsize=(5, 4))
        plt.scatter(X, y, alpha=0.6)
        plt.plot(X, y_pred, color='red')
        plt.title(f"{y_col} vs {x_col}\nR² = {r2:.3f}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.tight_layout()
        plt.savefig(f"graphs/{y_col}_vs_{x_col}.png")
        plt.close()

# === Save and summarize results ===
r2_df = pd.DataFrame(r2_results, columns=["X", "Y", "R2"])
r2_df.to_csv("r2_results.csv", index=False)

# === Print top and bottom 5 ===
print("\n🔝 Top 5 highest R² correlations:")
print(r2_df.sort_values("R2", ascending=False).head(5).to_string(index=False))

print("\n🔻 Bottom 5 lowest R² correlations:")
print(r2_df.sort_values("R2", ascending=True).head(5).to_string(index=False))

print(f"\n✅ Analysis complete! {len(r2_results)} regressions computed.")
print("Plots saved in: graphs/")
print("Full results saved in: r2_results.csv")
