# Clustering tools

The K-Means, DBSCAN, Hierarchical, GMM and Cluster Evaluation panels, and the
clustering nodes in Data Studio, compute on the device selected in the Devices
tab (CUDA, OpenCL, or ArrayFire's CPU backend). There is no separate CPU code
path. Results
are float64 where the device supports it, otherwise float32.

Each algorithm is checked against float64 references of the library it follows:
scikit-learn's behaviour written in PyTorch, and scipy for hierarchical linkage.
On a GPU without float64 (for example an Intel Iris Xe) the same checks pass at
float32 precision (5e-5 relative; measured worst 4.9e-7), with the same labels.

## Algorithms

| Tool | Behaviour | Same as |
| --- | --- | --- |
| K-Means | Lloyd iterations from k-means++ (or random) starts; stops when the total squared centroid move is at most tol × the mean feature variance; best of n_init runs. An empty cluster keeps its previous centroid. The Convergence plot shows the inertia at every iteration. | scikit-learn `KMeans` (plain, not greedy, k-means++) |
| Elbow | K-Means at each k (up to samples − 1) with inertia and silhouette; the suggested k has the sharpest bend in the inertia curve | |
| DBSCAN | Core point: at least min_samples points (itself included) within eps. Clusters are numbered by their first core point; a border point joins the earliest cluster that reaches it; -1 is noise. | scikit-learn `DBSCAN`, same labels |
| k-distance graph | Each point's distance to its k-th nearest other point, sorted; the knee suggests eps | |
| Hierarchical | Agglomerative with ward (euclidean only), complete, average or single linkage on euclidean, manhattan or cosine distances; the linkage matrix and the cophenetic correlation are scipy's. The cut slider keeps every merge at or below the height. | scipy `linkage`, `fcluster`, `cophenet` |
| GMM | EM with full, tied, diag or spherical covariances (1e-6 added to each diagonal), started from k-means++ seeds; stops when the mean per-sample log-likelihood changes by less than tol. BIC and AIC count parameters per covariance type. | scikit-learn `GaussianMixture` |

## Cluster evaluation

- Silhouette (euclidean), Davies-Bouldin and Calinski-Harabasz, as scikit-learn.
- Labels may be any integers; noise (-1) is left out of every score, and its
  per-sample silhouette shows as 0. Per-cluster silhouettes are listed in
  ascending label order. A cluster of one point has silhouette 0.
- The scores need between 2 and n − 1 clusters; otherwise the panel shows why.

## Size

DBSCAN and hierarchical clustering hold an n × n matrix on the device (DBSCAN as
booleans), so they suit up to a few tens of thousands of rows. Hierarchical
clustering does one device step per merge. Silhouette and the k-distance graph
work in row blocks and do not hold the full matrix.
