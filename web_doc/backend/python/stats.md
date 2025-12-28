# Statistics (`cx.stats`)

The `stats` submodule provides clustering algorithms, dimensionality reduction, and model evaluation metrics.

## Overview

```python
import pycyxwiz as cx
import numpy as np

# Generate sample data
data = np.random.randn(100, 5).tolist()

# Clustering
result = cx.stats.kmeans(data, k=3)
labels = result['labels']

# Dimensionality reduction
pca_result = cx.stats.pca(data, n_components=2)
transformed = pca_result['transformed']
```

## Clustering

### `kmeans(data, k, max_iter=300, init="kmeans++")`

K-Means clustering algorithm.

```python
import numpy as np

# Generate clustered data
np.random.seed(42)
cluster1 = np.random.randn(30, 2) + [0, 0]
cluster2 = np.random.randn(30, 2) + [5, 5]
cluster3 = np.random.randn(30, 2) + [0, 5]
data = np.vstack([cluster1, cluster2, cluster3]).tolist()

# Run K-Means
result = cx.stats.kmeans(data, k=3, max_iter=300, init="kmeans++")

print("Labels:", result['labels'][:10])  # Cluster assignments
print("Inertia:", result['inertia'])      # Sum of squared distances
print("Iterations:", result['n_iterations'])
print("Converged:", result['converged'])
```

**Parameters:**
- `data` (2D list): Input data (n_samples x n_features)
- `k` (int): Number of clusters
- `max_iter` (int): Maximum iterations. Default: 300
- `init` (str): Initialization method. Options: "kmeans++", "random". Default: "kmeans++"

**Returns:** dict with keys:
- `labels`: List of cluster assignments (0 to k-1)
- `centroids`: 2D list of cluster centers
- `inertia`: Sum of squared distances to nearest centroid
- `n_iterations`: Number of iterations run
- `converged`: Boolean, whether algorithm converged

**Example - Elbow Method:**
```python
inertias = []
k_values = range(1, 10)

for k in k_values:
    result = cx.stats.kmeans(data, k=k)
    inertias.append(result['inertia'])

# Plot elbow curve
import matplotlib.pyplot as plt
plt.plot(k_values, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

---

### `dbscan(data, eps, min_samples=5)`

Density-Based Spatial Clustering (DBSCAN).

```python
result = cx.stats.dbscan(data, eps=0.5, min_samples=5)

print("Labels:", result['labels'][:10])  # -1 indicates noise
print("Number of clusters:", result['n_clusters'])
print("Noise points:", result['n_noise'])
```

**Parameters:**
- `data` (2D list): Input data (n_samples x n_features)
- `eps` (float): Maximum distance between neighbors
- `min_samples` (int): Minimum points to form a cluster. Default: 5

**Returns:** dict with keys:
- `labels`: List of cluster labels (-1 for noise)
- `n_clusters`: Number of clusters found
- `n_noise`: Number of noise points

**When to use DBSCAN:**
- Unknown number of clusters
- Clusters with irregular shapes
- Dataset with outliers/noise

---

### `gmm(data, n_components, cov_type="full")`

Gaussian Mixture Model clustering.

```python
result = cx.stats.gmm(data, n_components=3, cov_type="full")

print("Labels:", result['labels'][:10])
print("Means:", result['means'])  # Cluster centers
print("Weights:", result['weights'])  # Mixing proportions
print("AIC:", result['aic'])  # Akaike Information Criterion
print("BIC:", result['bic'])  # Bayesian Information Criterion
```

**Parameters:**
- `data` (2D list): Input data (n_samples x n_features)
- `n_components` (int): Number of Gaussian components
- `cov_type` (str): Covariance type. Options:
  - `"full"`: Each component has its own covariance matrix
  - `"tied"`: All components share a covariance matrix
  - `"diag"`: Diagonal covariance (independent features)
  - `"spherical"`: Single variance per component

**Returns:** dict with keys:
- `labels`: List of cluster assignments
- `means`: 2D list of component means
- `weights`: List of mixing proportions
- `aic`: Akaike Information Criterion (lower is better)
- `bic`: Bayesian Information Criterion (lower is better)

**Model Selection with BIC:**
```python
bic_scores = []
n_range = range(1, 10)

for n in n_range:
    result = cx.stats.gmm(data, n_components=n)
    bic_scores.append(result['bic'])

best_n = n_range[bic_scores.index(min(bic_scores))]
print(f"Optimal number of components: {best_n}")
```

## Dimensionality Reduction

### `pca(data, n_components=2)`

Principal Component Analysis.

```python
# High-dimensional data
data = np.random.randn(100, 10).tolist()

result = cx.stats.pca(data, n_components=2)

transformed = result['transformed']  # Projected data (100 x 2)
components = result['components']    # Principal components
explained = result['explained_variance']  # Variance explained by each PC
```

**Parameters:**
- `data` (2D list): Input data (n_samples x n_features)
- `n_components` (int): Number of components to keep. Default: 2

**Returns:** dict with keys:
- `transformed`: 2D list, projected data
- `components`: 2D list, principal components
- `explained_variance`: List, variance ratio for each component

**Example - Visualize PCA:**
```python
import matplotlib.pyplot as plt

# Apply PCA
result = cx.stats.pca(data, n_components=2)
X_pca = result['transformed']

# Plot
plt.figure(figsize=(8, 6))
x = [p[0] for p in X_pca]
y = [p[1] for p in X_pca]
plt.scatter(x, y, c=labels, cmap='viridis')
plt.xlabel(f'PC1 ({result["explained_variance"][0]:.1%} variance)')
plt.ylabel(f'PC2 ({result["explained_variance"][1]:.1%} variance)')
plt.title('PCA Projection')
plt.colorbar(label='Cluster')
plt.show()
```

---

### `tsne(data, n_dims=2, perplexity=30)`

t-Distributed Stochastic Neighbor Embedding (t-SNE).

```python
# High-dimensional data
data = np.random.randn(100, 50).tolist()

# t-SNE embedding
embeddings = cx.stats.tsne(data, n_dims=2, perplexity=30)
# embeddings is a 2D list (100 x 2)
```

**Parameters:**
- `data` (2D list): Input data (n_samples x n_features)
- `n_dims` (int): Target dimensionality. Default: 2
- `perplexity` (int): Perplexity parameter. Default: 30

**Returns:** 2D list, embedded coordinates

**Note:** t-SNE is slower than PCA but often produces better visualizations for complex data structures.

**Choosing Perplexity:**
- Typical range: 5-50
- Larger values consider more neighbors
- Try multiple values and compare

## Evaluation Metrics

### `silhouette(data, labels)`

Compute silhouette score for clustering quality.

```python
# After clustering
result = cx.stats.kmeans(data, k=3)
labels = result['labels']

# Compute silhouette score
score = cx.stats.silhouette(data, labels)
print(f"Silhouette score: {score:.3f}")
# Score range: -1 to 1 (higher is better)
```

**Parameters:**
- `data` (2D list): Input data
- `labels` (list): Cluster assignments

**Returns:** float, silhouette score

**Interpretation:**
- **1.0**: Perfect clustering
- **0.0**: Overlapping clusters
- **< 0**: Possible wrong cluster assignments

---

### `confusion_matrix(y_true, y_pred)`

Compute confusion matrix and derived metrics.

```python
y_true = [0, 0, 1, 1, 2, 2, 0, 1, 2]
y_pred = [0, 0, 1, 2, 2, 2, 0, 1, 1]

result = cx.stats.confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
for row in result['matrix']:
    print(row)

print(f"\nAccuracy: {result['accuracy']:.3f}")
print(f"Precision (per class): {result['precision']}")
print(f"Recall (per class): {result['recall']}")
print(f"F1 Score (per class): {result['f1']}")
```

**Parameters:**
- `y_true` (list): True labels
- `y_pred` (list): Predicted labels

**Returns:** dict with keys:
- `matrix`: 2D list, confusion matrix
- `accuracy`: float, overall accuracy
- `precision`: list, precision per class
- `recall`: list, recall per class
- `f1`: list, F1 score per class

**Example - Plot Confusion Matrix:**
```python
import matplotlib.pyplot as plt
import numpy as np

result = cx.stats.confusion_matrix(y_true, y_pred)
cm = np.array(result['matrix'])

plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (Accuracy: {result["accuracy"]:.2%})')

# Add text annotations
for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black')

plt.colorbar()
plt.show()
```

---

### `roc(y_true, y_scores)`

Compute ROC curve and AUC score.

```python
y_true = [0, 0, 1, 1, 0, 1, 1, 0]
y_scores = [0.1, 0.4, 0.35, 0.8, 0.3, 0.7, 0.9, 0.2]  # Prediction probabilities

result = cx.stats.roc(y_true, y_scores)

print("False Positive Rates:", result['fpr'][:5])
print("True Positive Rates:", result['tpr'][:5])
print(f"AUC: {result['auc']:.3f}")
```

**Parameters:**
- `y_true` (list): Binary true labels (0 or 1)
- `y_scores` (list): Prediction scores/probabilities

**Returns:** dict with keys:
- `fpr`: List, false positive rates
- `tpr`: List, true positive rates
- `auc`: float, area under ROC curve

**Example - Plot ROC Curve:**
```python
import matplotlib.pyplot as plt

result = cx.stats.roc(y_true, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(result['fpr'], result['tpr'], 'b-', linewidth=2,
         label=f'ROC (AUC = {result["auc"]:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()
```

## Complete Example

```python
import pycyxwiz as cx
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic clustered data
np.random.seed(42)
n_samples = 300

# Three clusters
cluster1 = np.random.randn(n_samples // 3, 2) * 0.5 + [0, 0]
cluster2 = np.random.randn(n_samples // 3, 2) * 0.5 + [3, 3]
cluster3 = np.random.randn(n_samples // 3, 2) * 0.5 + [0, 3]
data = np.vstack([cluster1, cluster2, cluster3]).tolist()
true_labels = [0] * 100 + [1] * 100 + [2] * 100

# 1. K-Means clustering
kmeans_result = cx.stats.kmeans(data, k=3)
kmeans_labels = kmeans_result['labels']

# 2. DBSCAN clustering
dbscan_result = cx.stats.dbscan(data, eps=0.5, min_samples=5)
dbscan_labels = dbscan_result['labels']

# 3. GMM clustering
gmm_result = cx.stats.gmm(data, n_components=3)
gmm_labels = gmm_result['labels']

# Evaluate all methods
print("Clustering Evaluation")
print("=" * 40)

for name, labels in [("K-Means", kmeans_labels),
                     ("DBSCAN", dbscan_labels),
                     ("GMM", gmm_labels)]:
    silhouette = cx.stats.silhouette(data, labels)
    print(f"{name}: Silhouette = {silhouette:.3f}")

# PCA for visualization
pca_result = cx.stats.pca(data, n_components=2)
data_2d = pca_result['transformed']

# Plot all results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# True labels
x = [p[0] for p in data_2d]
y = [p[1] for p in data_2d]
axes[0, 0].scatter(x, y, c=true_labels, cmap='viridis', s=20)
axes[0, 0].set_title('True Labels')

# K-Means
axes[0, 1].scatter(x, y, c=kmeans_labels, cmap='viridis', s=20)
axes[0, 1].set_title(f'K-Means (k=3)')

# DBSCAN
axes[1, 0].scatter(x, y, c=dbscan_labels, cmap='viridis', s=20)
axes[1, 0].set_title(f'DBSCAN (eps=0.5)')

# GMM
axes[1, 1].scatter(x, y, c=gmm_labels, cmap='viridis', s=20)
axes[1, 1].set_title(f'GMM (n=3)')

plt.tight_layout()
plt.show()

# Confusion matrix for K-Means
conf = cx.stats.confusion_matrix(true_labels, kmeans_labels)
print(f"\nK-Means Accuracy: {conf['accuracy']:.2%}")
```

---

**Next**: [Time Series](timeseries.md) | [Back to Index](index.md)
