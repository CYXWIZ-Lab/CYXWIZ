---
title: Project rows into two dimensions
description: See the rows of a table in two dimensions with PCA, t-SNE or UMAP in Explore > Projection, and keep the projection as a node.
type: how-to
area: data-studio
audience: user
status: experimental
since: 1.0.8
owner: data-studio
last_reviewed: 2026-11-25
related:
  - explore.md
  - clustering.md
redirect_from:
  - dimensionality-reduction.md
---

Data Studio's **Explore > Projection** tab shows what the rows look like in
two dimensions: **PCA**, **t-SNE** or **UMAP** of the chosen number columns.
The **PCA**, **t-SNE** and **UMAP** nodes do the same inside a graph. **Tools
> Clustering and Projection > PCA / t-SNE / UMAP** opens the tab.

Everything computes on the device selected in the Devices tab (CUDA,
OpenCL, or ArrayFire's CPU backend); there is no separate CPU code path. A
build without ArrayFire reports that the method needs ArrayFire. If a run is
refused or fails, the tab or the node says why.

## Before you start

- A loaded dataset with number columns, shown in Data Studio's rail. See
  [Explore your data](explore.md).

## Project the rows

1. In **Data Studio > Explore**, open the **Projection** tab.
2. Pick the number columns and **Scale**, as in the
   [Groups tab](clustering.md).
3. Pick **PCA**, **t-SNE** or **UMAP**, and type its settings:
   - PCA shows each component's share of the variation (a bar each), the
     loadings of the first two (those of 0.4 or more stand out) and what
     the components are in words ("Component 1: flipper_length_mm,
     body_mass_g and bill_length_mm together, against bill_depth_mm").
   - t-SNE takes perplexity, learning rate (0: auto, the rows / 48 and at
     least 50, as scikit-learn) and iterations. It shows its iteration, KL
     divergence and a progress bar; **Cancel** stops it.
   - UMAP takes neighbours and min distance. It has no progress to show, so
     **Cancel** only stops waiting for it.
4. Optional: **Colour by** any column (text: a colour per value, the 17
   most common beyond 18 values; numbers: four quartile bins), or by the
   groups of the Groups tab when they were found on the same columns.
5. Optional: **Add as a step** or **Keep as a node** adds the PCA, t-SNE or
   UMAP node, which keeps the input columns and adds `pc_*`, `tsne_*` or
   `umap_*` beside them.

t-SNE and UMAP read at most 5,000 rows in the tab: beyond that a seeded
sample (the seed is typed) in table order, and the sentence says so. The
t-SNE and UMAP nodes work on every row.

## How the methods work

### PCA

As scikit-learn `PCA`: the components are the eigenvectors of the
covariance (divided by n − 1), largest first, each signed so its largest
entry is positive. **Scale** divides each feature by its standard deviation
first (constant features are left as they are). In the Projection tab,
asking for more components than min(rows, features), or 0, gives that
minimum; the node refuses more components than features or Train rows, with
the reason. Checked against PyTorch to 1e-8 (5e-5 on a GPU without
float64).

The node learns from the Train rows, as scikit-learn's
`PCA().fit(X_train).transform(X)`: with a Data Split in the graph, the
covariance (and the means and scales) come from the Train rows only, and
every row is then mapped onto those directions. Without a Split it learns
from every row. Its fitted state (the means, the scales and the directions)
is saved with the step like the scalers' (Fitted preprocessing state in
Properties), so a run in `transform_only` mode maps new rows with the
directions it learned.

### t-SNE

Exact t-SNE as scikit-learn `TSNE(method="exact", init="pca")`:

- Each point's bandwidth is calibrated to the perplexity by binary search.
  Perplexity must be at least 1 and below the number of rows.
- The start is the PCA projection scaled to a first-axis standard deviation
  of 1e-4, so runs are repeatable.
- The first 250 iterations use early exaggeration 12 and momentum 0.5;
  later ones use momentum 0.8, with per-coordinate gains (minimum 0.01).
- KL divergence is reported every 50 iterations.

The learning rate is used as given; scikit-learn's "auto" value is
max(n / 48, 50), and a rate much above n / 48 makes the early phase
oscillate. t-SNE is chaotic: tiny differences grow about tenfold every 10–20
iterations, so runs on different devices end in different, equally valid
layouts.

### UMAP

The neighbourhood graph and the a / b curve are umap-learn's:

- Exact k nearest neighbours (the point itself counts as one), umap-learn's
  smooth_knn_dist and membership strengths, and the fuzzy union.
- a / b come from the least-squares fit of umap-learn's find_ab_params
  (min_dist 0.1 gives a = 1.577, b = 0.895).

The start is the PCA projection (umap-learn `init="pca"` without its 1e-4
noise), scaled to [0, 10]. The layout applies umap-learn's updates in
expectation over each epoch, on the whole graph at once: edges weighted by
w / w_max with 5 negative samples per edge, each step clipped to ±4, the
learning rate falling linearly from 1, 500 epochs up to 10,000 rows, else
200. Layouts look like umap-learn's but are not identical to its stochastic
updates.

## The nodes

- **PCA**: **keep_columns** (default off) keeps the input columns and adds
  Float64 `pc_0` .. beside them, null on rows missing a feature. Off, the
  output is Float32 `pc_*` (and the label as `y`) for the rows with every
  feature. **scale** standardises first. The other linear steps (Truncated
  SVD, Whitening, Linear Projection) and Linear Solve are in
  [Linear algebra](../../usage/linear_algebra.md#the-nodes).
- **t-SNE** and **UMAP**: `tsne_*` / `umap_*` beside the input columns
  (keep_columns, default on), standardised first (standardise, default on),
  2 or 3 dimensions. t-SNE takes perplexity, learning_rate (0, the default:
  auto, the rows / 48 and at least 50; a fixed 200 overshoots on small
  tables and in 3D) and iterations (at least 250, the early exaggeration),
  and stops when the run is cancelled. UMAP takes n_neighbors and min_dist.
  Rows missing a feature are left out with nulls.

## Limits

t-SNE and UMAP hold n × n matrices on the device and do n² work per step, so
they suit up to a few tens of thousands of rows. The Projection tab reads at
most 5,000 rows (a seeded sample beyond); the nodes read every row and check
first that the n × n matrix fits in memory.

## Related

- [Explore your data](explore.md)
- [Find groups of rows](clustering.md)
- [Linear algebra](../../usage/linear_algebra.md)
