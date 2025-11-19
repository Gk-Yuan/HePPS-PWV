import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the uploaded file
df = pd.read_csv('data/CML/metadata_super.csv')

# Identify the 9th and 10th columns (0-based index 8 and 9) as the parameters
param_cols = list(df.columns[8:10])
print("Target Parameters (first 5 rows):")
print(df[param_cols].head())

# Select numeric predictors and drop rows with missing values
numeric_df = df.select_dtypes(include=[np.number]).dropna()
predictor_cols = [c for c in numeric_df.columns if c not in param_cols]
X = numeric_df[predictor_cols]

print("\nPredictor Variables (first 5 rows):")
print(X.head())

# Standardize predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
pcs = pca.fit_transform(X_scaled)

# Explained variance ratio
evr = pca.explained_variance_ratio_
print("\nExplained Variance Ratio:")
for i, var in enumerate(evr, 1):
    print(f"PC{i}: {var:.4f}")

# Scree plot
plt.figure()
plt.plot(range(1, len(evr) + 1), evr, marker='o')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.xticks(range(1, len(evr) + 1))
plt.show()

# Correlate the two parameters with the principal components
pc_df = pd.DataFrame(pcs, columns=[f'PC{i}' for i in range(1, pcs.shape[1] + 1)])
aligned_target = numeric_df[param_cols].reset_index(drop=True)
combined = pd.concat([pc_df, aligned_target], axis=1)
corr_df = combined.corr().loc[param_cols, pc_df.columns]

print("\nCorrelation Between Parameters and Principal Components:")
print(corr_df)
