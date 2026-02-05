import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from tqdm import tqdm
import numpy as np

# === Columns to use ===
columns_to_use = [
    "Brightness", "Body", "Aroma", "Complexity", "Balance",
    "Sweetness", "Spicy", "Chocolaty", "Nutty", "Buttery",
    "Fruity", "Flowery", "Winey", "Altitude (meters)"
]

# === Load CSV ===
file_path = "green_coffees.csv"  # <- update if needed
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
print(f"Rows before cleaning: {len(df)}, after cleaning: {len(df_clean)}")

# === Create output folder if needed ===
os.makedirs("graphs", exist_ok=True)

# === Generate pair plot ===
sns.set(style="ticks", palette="pastel")
pairplot = sns.pairplot(df_clean[columns_to_use], diag_kind="hist", plot_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'k'})
pairplot.fig.suptitle("Coffee Flavor Attributes Pair Plot", y=1.02)
plt.tight_layout()

# === Save pair plot ===
pairplot.savefig("graphs/pairplot.png", dpi=300)
plt.show()

print("✅ Pair plot saved to graphs/pairplot.png")
