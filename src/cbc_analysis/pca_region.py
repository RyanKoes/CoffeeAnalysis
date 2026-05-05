import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import re
from tqdm import tqdm

# === Numeric columns for PCA ===
# numeric_cols = [
#     "Brightness", "Body", "Aroma", "Complexity", "Balance",
#     "Sweetness", "Spicy", "Chocolaty", "Nutty", "Buttery",
#     "Fruity", "Flowery", "Winey", "Altitude (meters)"
# ]

numeric_cols = [
    "Altitude (meters)", "Brightness"
]

# === Load CSV ===
file_path = "green_coffees.csv"  # <- update as needed
df = pd.read_csv(file_path)

# === Keep only relevant columns + Country ===
columns_to_keep = numeric_cols + ["Country"]
df = df[[c for c in columns_to_keep if c in df.columns]].copy()

# === Parse altitude ===
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

# === Map countries to regions ===
region_map = {
    "Bali": "Southeast Asia & Oceania",
    "Indonesia": "Southeast Asia & Oceania",
    "Papua New Guinea": "Southeast Asia & Oceania",
    "Vietnam": "Southeast Asia & Oceania",

    "Bolivia": "South America",
    "Brazil": "South America",
    "Colombia": "South America",
    "Peru": "South America",

    "Costa Rica": "Central America & Caribbean",
    "Dominican Republic": "Central America & Caribbean",
    "El Salvador": "Central America & Caribbean",
    "Guatemala": "Central America & Caribbean",
    "Haiti": "Central America & Caribbean",
    "Honduras": "Central America & Caribbean",
    "Jamaica": "Central America & Caribbean",
    "Mexico": "Central America & Caribbean",
    "Nicaragua": "Central America & Caribbean",
    "Panama": "Central America & Caribbean",
    "United States": "Central America & Caribbean",

    "Burundi": "Africa",
    "Ethiopia": "Africa",
    "Kenya": "Africa",
    "Rwanda": "Africa",
    "Tanzania": "Africa",
    "Uganda": "Africa",

    "India": "South Asia",
    "Yemen": "Middle East"
}

df_clean["Region"] = df_clean["Country"].map(region_map)

# === Standardize numeric data ===
X = df_clean[numeric_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Perform PCA ===
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_
print(f"Explained variance by PC1 and PC2: {explained_var}")

# === Prepare DataFrame for plotting ===
df_pca = pd.DataFrame(components, columns=["PC1", "PC2"])
df_pca["Region"] = df_clean["Region"].values
df_pca["Country"] = df_clean["Country"].values
df_pca["Name"] = df_clean.get("Name", pd.Series([""]*len(df_pca)))

# === Create output folder ===
os.makedirs("graphs", exist_ok=True)

# === Plot PCA colored by region ===
plt.figure(figsize=(12,9))
sns.scatterplot(
    data=df_pca,
    x="PC1", y="PC2",
    hue="Region",
    palette="tab10",
    s=80,
    alpha=0.8
)
plt.title("PCA of Coffee Flavor Attributes by Region", fontsize=16)
plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}% variance)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Region")
plt.tight_layout()

# === Save figure ===
plt.savefig("graphs/pca_by_region_altitude.png", dpi=300)
plt.show()
print("✅ PCA plot saved to graphs/pca_by_region.png")
