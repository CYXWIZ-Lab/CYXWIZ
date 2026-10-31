# Dimensionality reduction

Data Studio's Explore > Projection tab ([Data Studio](data_studio.md#projection))
and the PCA, t-SNE and UMAP nodes compute on the device selected in the Devices
tab (CUDA, OpenCL, or ArrayFire's CPU backend). There is no separate CPU code
path; a build without ArrayFire reports that the method needs ArrayFire. If a
run is refused or fails, the tab or the node says why.

## PCA

As scikit-learn `PCA`: the components are the eigenvectors of the covariance
(divided by n − 1), largest first, each signed so its largest entry is positive.
"Scale" divides each feature by its standard deviation first (constant features
are left as they are). Asking for more components than min(rows, features), or
0, gives that minimum. Checked against PyTorch to 1e-8 (5e-5 on a GPU without
float64, such as an Intel Iris Xe).

## t-SNE

Exact t-SNE as scikit-learn `TSNE(method="exact", init="pca")`:

- Each point's bandwidth is calibrated to the perplexity by binary search.
  Perplexity must be at least 1 and below the number of rows.
- The start is the PCA projection scaled to a first-axis standard deviation of 1e-4,
  so runs are repeatable.
- The first 250 iterations use early exaggeration 12 and momentum 0.5; later
  ones use momentum 0.8, with per-coordinate gains (minimum 0.01).
- KL divergence is reported every 50 iterations.

The learning rate is used as given. scikit-learn's "auto" value is max(n / 48, 50).
A rate much above n / 48 makes the early phase oscillate. t-SNE is chaotic: tiny
differences grow about tenfold every 10–20 iterations, so runs on different
devices end in different (equally valid) layouts. The parity test therefore
compares runs step for step at a small learning rate, or for the first 60
iterations, and checks that a full 1000-iteration run separates clusters. In
float32 (a GPU without float64) those short runs agree to 5e-2 relative (1.5e-2
measured on an Intel Iris Xe) and the KL divergence to 4e-4.

## UMAP

The neighbourhood graph and the a / b curve are umap-learn's:

- Exact k nearest neighbours (the point itself counts as one), umap-learn's
  smooth_knn_dist and membership strengths, and the fuzzy union.
- a / b come from the least-squares fit of umap-learn's find_ab_params
  (min_dist 0.1 gives a = 1.577, b = 0.895).

The start is the PCA projection (umap-learn `init="pca"` without its 1e-4 noise),
scaled to [0, 10].

The layout applies umap-learn's updates in expectation over each epoch, on the
whole graph at once:

- Edges are weighted by w / w_max, with 5 negative samples per edge.
- Each step is clipped to ±4.
- The learning rate falls linearly from 1.
- There are 500 epochs up to 10,000 rows, else 200.

Layouts therefore look like umap-learn's but are not identical to its stochastic
updates.

## Size

t-SNE and UMAP hold n × n matrices on the device and do n² work per step, so
they suit up to a few tens of thousands of rows. The Projection tab reads at
most 5,000 rows (a seeded sample beyond); the nodes read every row and check
first that the n × n matrix fits in memory.

## The nodes

- **PCA**: **keep_columns** (default off) keeps the input columns and adds
  Float64 `pc_0` .. beside them, null on rows missing a feature. Off, the
  output is Float32 `pc_*` (and the label as `y`) for the rows with every
  feature. **scale** standardises first.
- **t-SNE** and **UMAP**: `tsne_*` / `umap_*` beside the input columns
  (keep_columns, default on), standardised first (standardise, default on),
  2 or 3 dimensions. t-SNE takes perplexity, learning_rate and iterations (at
  least 250, the early exaggeration), and stops when the run is cancelled; UMAP
  takes n_neighbors and min_dist. Rows missing a feature are left out with
  nulls.
