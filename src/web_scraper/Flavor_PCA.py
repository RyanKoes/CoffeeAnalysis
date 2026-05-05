import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV
csv_file = 'green_coffees.csv'  # replace with your filename
df = pd.read_csv(csv_file)

# Select numeric columns for PCA (tasting notes and grading numbers)
# Spicy	Chocolaty	Nutty	Buttery	Fruity	Flowery	Winey	Earthy
numeric_cols = ['Spicy', 'Chocolaty', 'Nutty', 'Buttery', 'Fruity', 'Flowery', 'Winey', 'Earthy']
X = df[numeric_cols]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Name'] = df['Name']

# Calculate the sum of the specified attributes
attribute_sum = df[numeric_cols].sum(axis=1)

# Normalize the sum to a range of 0 to 1
normalized_sum = (attribute_sum - attribute_sum.min()) / (attribute_sum.max() - attribute_sum.min())

# Plot
plt.figure(figsize=(10, 7))
scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=normalized_sum, cmap='Blues')

# Annotate points with the first three words of coffee names
for i, name in enumerate(pca_df['Name']):
    shortened_name = ' '.join(name.split()[:3])  # Take the first three words
    plt.text(pca_df['PC1'][i] + 0.05, pca_df['PC2'][i] + 0.05, shortened_name, fontsize=6)  # Reduced fontsize to 6

plt.title('PCA of Coffee Tasting Notes')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Tasting Sum (Normalized)')
plt.grid(True)
plt.tight_layout()
plt.show()