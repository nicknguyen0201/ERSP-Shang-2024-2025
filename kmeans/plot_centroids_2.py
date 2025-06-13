import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Load your main dataset
df = pd.read_csv('processed_dataset_46482.csv')   # contains 20 feature cols + 'label'
X = df.drop(columns=['target']).values
y = df['target'].values

# 2. Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA → 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 4. Load your centroids (each should be 2 rows × 20 cols)
#    If your CSVs have headers matching the feature names, you can read directly.
pred_cent = pd.read_csv('predicted_centroids.csv').values    # shape (2,20)
elki_cent = pd.read_csv('new_data/test_result/centroids_processed_dataset_46482.csv').values         # shape (2,20)

# 5. Apply the same scaler & PCA transform to them
pred_cent_scaled = scaler.transform(pred_cent)
elki_cent_scaled = scaler.transform(elki_cent)

pred_cent_pca = pca.transform(pred_cent_scaled)   # shape (2,2)
elki_cent_pca = pca.transform(elki_cent_scaled)  # shape (2,2)

# 6. Plot data + both sets of centroids
plt.figure(figsize=(8,6))

# 6a) scatter the points
palette = {0:'tab:blue', 1:'tab:orange'}  # adjust if your labels differ
markers = {0:'o', 1:'s'}
for lbl in np.unique(y):
    mask = (y == lbl)
    plt.scatter(
        X_pca[mask,0], X_pca[mask,1],
        c=palette[lbl], marker=markers[lbl],
        s=60, alpha=0.5, edgecolor='k',
        label=f'Class {lbl}'
    )

# 6b) add predicted centroids
plt.scatter(
    pred_cent_pca[:,0], pred_cent_pca[:,1],
    c='black', marker='X', s=200,
    label='Predicted centroids',
    linewidths=1, edgecolor='white'
)

# 6c) add ELKI centroids
plt.scatter(
    elki_cent_pca[:,0], elki_cent_pca[:,1],
    c='gold', marker='D', s=200,
    label='ELKI centroids',
    linewidths=1, edgecolor='black'
)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('PCA Projection with Predicted & ELKI Centroids')
plt.legend(loc='best', fontsize='small', framealpha=0.8)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()

# 7. Save to PDF
plt.savefig('clusters_with_centroids.pdf', format='pdf', dpi=300)

# If you want to pop up a window, uncomment:
# plt.show()
