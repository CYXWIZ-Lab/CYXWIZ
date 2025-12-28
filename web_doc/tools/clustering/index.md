# Clustering Tools

Unsupervised learning tools for discovering patterns and groupings in data without predefined labels.

## Overview

The Clustering tools provide:
- **Partitioning Methods** - K-Means, K-Medoids
- **Hierarchical Clustering** - Agglomerative, Divisive
- **Density-Based** - DBSCAN, OPTICS, HDBSCAN
- **Model-Based** - Gaussian Mixture Models
- **Evaluation Metrics** - Silhouette, Calinski-Harabasz

## Tools Reference

### K-Means Clustering

The most popular partitioning algorithm for spherical clusters.

```
+------------------------------------------------------------------+
|  K-Means Clustering                                        [x]    |
+------------------------------------------------------------------+
|  DATA                                                             |
|  Features: [x] feature1 [x] feature2 [x] feature3 [ ] feature4    |
|  Samples: 10,000                                                  |
|                                                                   |
|  PARAMETERS                                                       |
|  Number of Clusters (k): [5     ]                                 |
|  Initialization: (o) k-means++  ( ) Random  ( ) Manual            |
|  Max Iterations: [300   ]                                         |
|  Convergence Tolerance: [1e-4  ]                                  |
|  Random State: [42    ]                                           |
|                                                                   |
|  [ Find Optimal k ]  [ Run Clustering ]                           |
|                                                                   |
|  ELBOW METHOD (Optimal k Selection)                               |
|  Inertia                                                          |
|  |                                                                |
|  |****                                                            |
|  |    ****                                                        |
|  |        ***                                                     |
|  |           **                                                   |
|  |             **---*----*----*                                   |
|  +-------------------------------------------> k                  |
|       2    3    4   [5]   6    7    8                              |
|                     ^                                              |
|              Suggested optimal k                                   |
|                                                                   |
|  RESULTS                                                          |
|  +-----------------------------------------------------------+   |
|  | Cluster | Size  | % of Data | Centroid (first 3 dims)     |   |
|  +-----------------------------------------------------------+   |
|  | 0       | 2,345 | 23.45%    | [0.12, -0.34, 0.56]         |   |
|  | 1       | 1,876 | 18.76%    | [-1.23, 0.45, -0.78]        |   |
|  | 2       | 2,123 | 21.23%    | [0.89, 1.23, 0.11]          |   |
|  | 3       | 1,567 | 15.67%    | [-0.45, -1.12, 0.34]        |   |
|  | 4       | 2,089 | 20.89%    | [1.34, -0.67, -0.89]        |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  METRICS                                                          |
|  Inertia: 12,345.67                                               |
|  Silhouette Score: 0.456                                          |
|  Calinski-Harabasz: 234.56                                        |
|  Davies-Bouldin: 0.789                                            |
|                                                                   |
|  [Visualize]  [Export Labels]  [Save Model]                       |
+------------------------------------------------------------------+
```

### Hierarchical Clustering

Build a tree of clusters using agglomerative (bottom-up) or divisive (top-down) approaches.

```
+------------------------------------------------------------------+
|  Hierarchical Clustering                                   [x]    |
+------------------------------------------------------------------+
|  Method: (o) Agglomerative  ( ) Divisive                          |
|                                                                   |
|  Linkage: (o) Ward  ( ) Complete  ( ) Average  ( ) Single         |
|  Distance: (o) Euclidean  ( ) Manhattan  ( ) Cosine               |
|                                                                   |
|  Number of Clusters: [3     ] (or cut dendrogram)                 |
|                                                                   |
|  [ Run Clustering ]                                               |
|                                                                   |
|  DENDROGRAM                                                       |
|                                                                   |
|           +------+                                                |
|     +-----|      |-----+                                          |
|     |     +------+     |                                          |
|  +--+--+            +--+--+                                        |
|  |     |            |     |                                        |
| +-+ +-+-+        +-+-+ +-+-+                                       |
| | | | | |        | | | | | |                                       |
| 1 2 3 4 5        6 7 8 9 10                                        |
|                                                                   |
| Cut height: [-------------------o----] 5.67                       |
|                                                                   |
|  CLUSTER ASSIGNMENTS                                              |
|  +-----------------------------------------------------------+   |
|  | Cluster | Size  | Members (sample IDs)                    |   |
|  +-----------------------------------------------------------+   |
|  | 0       | 3,456 | [1, 2, 3, 4, 5, ...]                    |   |
|  | 1       | 4,123 | [6, 7, 8, 9, 10, ...]                   |   |
|  | 2       | 2,421 | [11, 12, 13, ...]                       |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  [Save Dendrogram]  [Export Labels]  [Flatten to k clusters]      |
+------------------------------------------------------------------+
```

### DBSCAN (Density-Based)

Discover clusters of arbitrary shapes based on density.

```
+------------------------------------------------------------------+
|  DBSCAN Clustering                                         [x]    |
+------------------------------------------------------------------+
|  PARAMETERS                                                       |
|  Epsilon (eps): [0.5   ]  (neighborhood radius)                   |
|  Min Samples: [5     ]  (core point threshold)                    |
|  Metric: (o) Euclidean  ( ) Manhattan  ( ) Cosine                 |
|                                                                   |
|  [ Find Optimal eps ]  [ Run Clustering ]                         |
|                                                                   |
|  K-DISTANCE GRAPH (for eps selection)                             |
|  Distance                                                         |
|  |                                                                |
|  |                                    ****                        |
|  |                               *****                            |
|  |                          *****                                 |
|  |---------------------*****<-- Elbow point (eps ~ 0.5)           |
|  |         ************                                           |
|  +-------------------------------------------> k-th neighbor      |
|                                                                   |
|  RESULTS                                                          |
|  +-----------------------------------------------------------+   |
|  | Cluster | Size  | Type       | Density                    |   |
|  +-----------------------------------------------------------+   |
|  | 0       | 3,456 | Core       | High                       |   |
|  | 1       | 2,789 | Core       | High                       |   |
|  | 2       | 1,234 | Core       | Medium                     |   |
|  | -1      | 521   | Noise      | -                          |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  STATISTICS                                                       |
|  Number of Clusters: 3                                            |
|  Noise Points: 521 (5.21%)                                        |
|  Core Points: 7,479                                               |
|  Border Points: 234                                               |
|                                                                   |
|  [Visualize]  [Export Labels]  [Remove Noise]                     |
+------------------------------------------------------------------+
```

### Gaussian Mixture Models

Probabilistic clustering assuming data comes from mixture of Gaussians.

```
+------------------------------------------------------------------+
|  Gaussian Mixture Model                                    [x]    |
+------------------------------------------------------------------+
|  PARAMETERS                                                       |
|  Number of Components: [4     ]                                   |
|  Covariance Type: (o) Full  ( ) Tied  ( ) Diagonal  ( ) Spherical |
|  Max Iterations: [100   ]                                         |
|  Convergence Tolerance: [1e-3  ]                                  |
|  Initialization: (o) k-means  ( ) Random                          |
|                                                                   |
|  [ Find Optimal Components ]  [ Fit Model ]                       |
|                                                                   |
|  MODEL SELECTION (BIC/AIC)                                        |
|  Score                                                            |
|  |                                                                |
|  |    BIC                                                         |
|  |    *                                                           |
|  |     *                                                          |
|  |      *                                                         |
|  |       *                                                        |
|  |        *---*---*---*                                           |
|  +-------------------------------------------> Components         |
|       2    3   [4]   5    6    7                                   |
|                                                                   |
|  COMPONENT PARAMETERS                                             |
|  +-----------------------------------------------------------+   |
|  | Component | Weight | Mean            | Covariance         |   |
|  +-----------------------------------------------------------+   |
|  | 0         | 0.25   | [0.12, -0.34]   | [[1.2, 0.3],...]   |   |
|  | 1         | 0.30   | [-1.23, 0.45]   | [[0.8, -0.2],...]  |   |
|  | 2         | 0.22   | [0.89, 1.23]    | [[1.5, 0.1],...]   |   |
|  | 3         | 0.23   | [-0.45, -1.12]  | [[1.1, -0.4],...]  |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  METRICS                                                          |
|  Log-Likelihood: -12,345.67                                       |
|  BIC: 24,789.12                                                   |
|  AIC: 24,701.34                                                   |
|                                                                   |
|  [Visualize Ellipses]  [Export Probabilities]  [Sample]           |
+------------------------------------------------------------------+
```

## Clustering Evaluation

### Silhouette Analysis

```
+------------------------------------------------------------------+
|  Silhouette Analysis                                       [x]    |
+------------------------------------------------------------------+
|  Cluster Labels: [kmeans_labels v]                                |
|                                                                   |
|  [ Analyze ]                                                      |
|                                                                   |
|  SILHOUETTE PLOT                                                  |
|                                                                   |
|  Cluster 0  |==========>            | 0.56                        |
|  Cluster 1  |=========>             | 0.52                        |
|  Cluster 2  |============>          | 0.64                        |
|  Cluster 3  |======>                | 0.41                        |
|  Cluster 4  |===========>           | 0.58                        |
|             0    0.2   0.4   0.6   0.8   1.0                      |
|                                                                   |
|  OVERALL SILHOUETTE SCORE: 0.542                                  |
|                                                                   |
|  INTERPRETATION                                                   |
|  +-----------------------------------------------------------+   |
|  | Score Range | Interpretation                               |   |
|  +-----------------------------------------------------------+   |
|  | 0.71 - 1.00 | Strong structure                             |   |
|  | 0.51 - 0.70 | Reasonable structure <- Current              |   |
|  | 0.26 - 0.50 | Weak structure                               |   |
|  | < 0.25      | No structure                                 |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  [Export Report]  [Compare with Other k Values]                   |
+------------------------------------------------------------------+
```

### Evaluation Metrics Comparison

| Metric | Description | Range | Best |
|--------|-------------|-------|------|
| **Silhouette Score** | Cohesion vs separation | -1 to 1 | Higher |
| **Calinski-Harabasz** | Between/within variance ratio | 0 to infinity | Higher |
| **Davies-Bouldin** | Average cluster similarity | 0 to infinity | Lower |
| **Inertia** | Within-cluster sum of squares | 0 to infinity | Lower |

## Scripting Functions

### K-Means

```python
from cyxwiz.clustering import KMeans

# Fit model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans.fit(X)

# Get results
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_

# Predict new data
new_labels = kmeans.predict(X_new)
```

### DBSCAN

```python
from cyxwiz.clustering import DBSCAN

# Fit model
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

# Identify noise
noise_mask = labels == -1
n_noise = noise_mask.sum()
```

### Hierarchical Clustering

```python
from cyxwiz.clustering import AgglomerativeClustering
from cyxwiz.clustering import dendrogram, linkage

# Compute linkage matrix
Z = linkage(X, method='ward')

# Plot dendrogram
dendrogram(Z)

# Get flat clusters
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(X)
```

### GMM

```python
from cyxwiz.clustering import GaussianMixture

# Fit model
gmm = GaussianMixture(n_components=4, covariance_type='full')
gmm.fit(X)

# Get probabilities
probs = gmm.predict_proba(X)
labels = gmm.predict(X)

# Generate samples
samples = gmm.sample(100)
```

### Evaluation

```python
from cyxwiz.clustering.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

# Compute metrics
silhouette = silhouette_score(X, labels)
ch_score = calinski_harabasz_score(X, labels)
db_score = davies_bouldin_score(X, labels)
```

## Integration with Node Editor

### Clustering Nodes

| Node | Inputs | Outputs | Parameters |
|------|--------|---------|------------|
| **K-Means** | Features tensor | Labels, Centroids | k, init, max_iter |
| **DBSCAN** | Features tensor | Labels | eps, min_samples |
| **Hierarchical** | Features tensor | Labels, Dendrogram | n_clusters, linkage |
| **GMM** | Features tensor | Labels, Probabilities | n_components, cov_type |

### Example Pipeline

```
[Data Input] -> [StandardScaler] -> [PCA] -> [K-Means] -> [Cluster Labels]
                                       |
                                       v
                              [Silhouette Score]
```

## Visualization Options

| Visualization | Description |
|---------------|-------------|
| **Scatter Plot** | 2D/3D cluster visualization |
| **Dendrogram** | Hierarchical clustering tree |
| **Silhouette Plot** | Per-sample silhouette values |
| **Elbow Plot** | Inertia vs k for K-Means |
| **t-SNE/UMAP** | Dimensionality reduction + clusters |

---

**Next**: [Model Evaluation Tools](../model-evaluation/index.md) | [Dimensionality Reduction](../dimensionality-reduction/index.md)
