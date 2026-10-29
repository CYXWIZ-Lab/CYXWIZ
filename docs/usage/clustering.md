# Clustering tools

Data Studio's Explore > Groups tab ([Data Studio](data_studio.md#groups)) and
the K-Means, DBSCAN, Hierarchical Clustering and GMM Clustering nodes compute on
the device selected in the Devices tab (CUDA, OpenCL, or ArrayFire's CPU
backend). There is no separate CPU code
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
| How many groups (K-Means) | K-Means at each k (up to samples − 1) with inertia and silhouette; the suggested k has the largest second difference of the inertias (the elbow) | |
| DBSCAN | Core point: at least min_samples points (itself included) within eps. Clusters are numbered by their first core point; a border point joins the earliest cluster that reaches it; -1 is noise. | scikit-learn `DBSCAN`, same labels |
| k-distance graph | Each point's distance to its (min samples − 1)-th nearest other point with the chosen metric, sorted (so a point is core when its value is at most eps); the knee (Kneedle: the point furthest below the chord) suggests eps | scikit-learn `NearestNeighbors` |
| Hierarchical | Agglomerative with ward (euclidean only), complete, average or single linkage on euclidean, manhattan or cosine distances; the linkage matrix and the cophenetic correlation are scipy's. The dendrogram draws the top 12 merges as scipy's `dendrogram(truncate_mode="lastp")`. | scipy `linkage`, `fcluster`, `cophenet`, `dendrogram` |
| GMM | EM with full, tied, diag or spherical covariances (1e-6 added to each diagonal), started from k-means++ seeds; stops when the mean per-sample log-likelihood changes by less than tol. BIC and AIC count parameters per covariance type. | scikit-learn `GaussianMixture` |

## Cluster evaluation

- Silhouette (euclidean), Davies-Bouldin and Calinski-Harabasz, as scikit-learn.
- Labels may be any integers; noise (-1) is left out of every score, and its
  per-sample silhouette shows as 0. Per-cluster silhouettes are listed in
  ascending label order. A cluster of one point has silhouette 0.
- The scores need between 2 and n − 1 clusters; otherwise the Groups tab says so.
- Compared with a column: the adjusted Rand index, as scikit-learn's
  `adjusted_rand_score` (noise rows left out).

## The nodes

- **standardise** (default off; Explore > Groups turns it on): z-scores with the
  population standard deviation, as scikit-learn's `StandardScaler`.
- Rows missing a feature (null, or not a finite number) are left out of the fit
  and get a null `cluster_id`; the output keeps every row in order. Features
  are read as float64.
- GMM Clustering's **probabilities** adds `probability_0` .. `probability_k-1`,
  each row's chance of each component.
- Add as a step / Keep as a node in the Groups tab set these from the tab's
  settings; test_groups_explore checks that the node gives the tab's groups.

## Size

DBSCAN and hierarchical clustering hold an n × n matrix on the device (DBSCAN as
booleans), so they suit up to a few tens of thousands of rows. Hierarchical
clustering does one device step per merge. Silhouette and the k-distance graph
work in row blocks and do not hold the full matrix.
