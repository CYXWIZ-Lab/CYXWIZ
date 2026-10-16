# Dimensionality reduction

The Dimensionality Reduction panel (PCA, t-SNE, UMAP) and the PCA node in Data
Studio compute on the device selected in the Devices tab (CUDA, OpenCL, or
ArrayFire's CPU backend). There is no separate CPU code path; a build without
ArrayFire reports that the method needs ArrayFire. If a run is refused or fails,
the panel shows the reason under the settings.

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
they suit up to a few tens of thousands of rows.
