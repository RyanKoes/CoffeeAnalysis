import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the CSV
csv_file = 'green_coffees.csv'  # replace with your filename
df = pd.read_csv(csv_file)

# Select numeric columns for PCA (tasting notes and grading numbers)
numeric_cols = ['Brightness','Body','Aroma','Complexity','Balance','Sweetness',
                'Spicy','Chocolaty','Nutty','Buttery','Fruity','Flowery','Winey','Earthy']

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

# Plot
plt.figure(figsize=(10,7))
plt.scatter(pca_df['PC1'], pca_df['PC2'])

# Annotate points with coffee names
for i, name in enumerate(pca_df['Name']):
    plt.text(pca_df['PC1'][i]+0.05, pca_df['PC2'][i]+0.05, name, fontsize=6)  # Reduced fontsize to 6

plt.title('PCA of Coffee Tasting Notes and Grading')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.tight_layout()
plt.show()
