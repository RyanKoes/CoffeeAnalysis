import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import re
from tqdm import tqdm

# === Columns to use for PCA ===
numeric_cols = [
    "Brightness", "Body", "Aroma", "Complexity", "Balance",
    "Sweetness", "Spicy", "Chocolaty", "Nutty", "Buttery",
    "Fruity", "Flowery", "Winey", "Altitude (meters)"
]

# === Load CSV ===
file_path = "green_coffees.csv"  # <- update as needed
df = pd.read_csv(file_path)

# === Keep only relevant columns + Country ===
columns_to_keep = numeric_cols + ["Country"]
df = df[[c for c in columns_to_keep if c in df.columns]].copy()

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

# === Convert numeric columns to numeric ===
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# === Drop rows with missing numeric data or missing Country ===
df_clean = df.dropna(subset=numeric_cols + ["Country"])
print(f"Rows after cleaning: {len(df_clean)}")

# === Standardize numeric data ===
X = df_clean[numeric_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Perform PCA ===
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_
print(f"Explained variance by PC1 and PC2: {explained_var}")

# === Prepare plot data ===
df_pca = pd.DataFrame(components, columns=["PC1", "PC2"])
df_pca["Country"] = df_clean["Country"].values

# === Create output folder ===
os.makedirs("graphs", exist_ok=True)

# === Plot PCA colored by country ===
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Country", palette="tab20", s=80, alpha=0.8)
plt.title("PCA of Coffee Flavor Attributes by Country")
plt.xlabel(f"PC1 ({explained_var[0] * 100:.1f}% variance)")
plt.ylabel(f"PC2 ({explained_var[1] * 100:.1f}% variance)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# === Save figure ===
plt.savefig("graphs/pca_by_country.png", dpi=300)
plt.show()
print("✅ PCA plot saved to graphs/pca_by_country.png")
