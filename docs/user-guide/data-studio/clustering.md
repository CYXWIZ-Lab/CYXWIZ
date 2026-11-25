---
title: Find groups of rows
description: Group rows with K-Means, DBSCAN, hierarchical clustering or a Gaussian mixture in Explore > Groups, and keep the grouping as a node.
type: how-to
area: data-studio
audience: user
status: experimental
since: 1.0.8
owner: data-studio
last_reviewed: 2026-11-25
related:
  - explore.md
  - dimensionality-reduction.md
  - statistics.md
redirect_from:
  - clustering.md
---

Data Studio's **Explore > Groups** tab shows which rows group together. The
**K-Means**, **DBSCAN**, **Hierarchical Clustering** and **GMM Clustering**
nodes do the same inside a graph. **Tools > Clustering and Projection**
opens the tab on a method (K-Means Clustering, DBSCAN, Hierarchical
Clustering, Gaussian Mixture Models, Cluster Evaluation).

Everything computes on the device selected in the Devices tab (CUDA,
OpenCL, or ArrayFire's CPU backend); there is no separate CPU code path.
Results are float64 where the device supports it, otherwise float32. Each
method is checked against float64 references of the library it follows
(scikit-learn, and scipy for hierarchical linkage). On a GPU without
float64 the same checks pass at float32 precision (5e-5 relative), with the
same labels.

## Before you start

- A loaded dataset with number columns, shown in Data Studio's rail. See
  [Explore your data](explore.md).

## Group the rows

1. In **Data Studio > Explore**, open the **Groups** tab.
2. Pick the number columns. Every number column, up to eight, is picked to
   start.
3. Set **Scale**: **Standardise** turns each column into z-scores, so a
   column in large units (grams) does not decide the groups alone; **As they
   are** uses the raw values.
4. Pick the **Method** and type its settings:

   | Method | Settings |
   | --- | --- |
   | K-Means | groups (2 to 50), seed |
   | DBSCAN | eps, min samples (the row itself counts), metric (euclidean, manhattan, cosine) |
   | Hierarchical | groups (where the tree is cut), linkage (ward, complete, average, single); metric when not ward |
   | Gaussian mixture | components, seed, covariance (full, tied, diag, spherical) |

5. Read the answer (see the next section). The same seed gives the same
   groups. Rows missing one of the columns are left out and counted.
6. Optional: select **How many groups?** for a chart of the method, then
   **Use 3** (or **Use eps 0.643**) to take its suggestion.
7. Optional: keep the grouping in the graph. **Add as a step** puts the
   method's node after the data in use, so the rest of the graph gets a
   `cluster_id` column (a Gaussian mixture also adds `probability_0` ..);
   **Keep as a node** adds it as a branch. The node gets the tab's settings
   and gives the same groups. **Export CSV...** saves the groups and a row
   per row used; **Copy** copies the summary.

## Read the answer

- **The scores**: silhouette, with its words (above 0.70 strong, 0.50
  reasonable, 0.25 weak structure), Davies-Bouldin, Calinski-Harabasz, and
  the rows used; and what the method says about itself (K-Means inertia and
  iterations, the tree's cophenetic correlation, the mixture's BIC and AIC).
- **The rows** on their first two principal components, coloured by group
  (DBSCAN's noise faint), with each group's size.
- **What each group is**: the mean of each column per group, in the
  column's own units, each group's silhouette, and a sentence per group
  ("high flipper_length_mm, low bill_depth_mm": columns at least half a
  standard deviation from the mean of all rows).
- **Compare with a column** (any column with 2 to 50 values): the counts
  per group and value, the adjusted Rand index (1: the groups are the
  column's values, 0: chance) and what it means ("Group 3 is all Gentoo
  (123)").

**How many groups?** draws, per method: K-Means inertia (the elbow) and
silhouette for 2 to 8 groups; Gaussian mixture BIC and AIC; the top 12
merges of the hierarchical tree, with the cut and the silhouette of each
cut; DBSCAN's sorted distance from each row to its (min samples − 1)-th
neighbour, with eps drawn and its knee. The suggestion comes in words.

## How the methods work

| Method | Behaviour | Same as |
| --- | --- | --- |
| K-Means | Lloyd iterations from k-means++ (or random) starts; stops when the total squared centroid move is at most tol × the mean feature variance; best of n_init runs. An empty cluster keeps its previous centroid. The Convergence plot shows the inertia at every iteration. | scikit-learn `KMeans` (plain, not greedy, k-means++) |
| How many groups (K-Means) | K-Means at each k (up to samples − 1) with inertia and silhouette; the suggested k has the largest second difference of the inertias (the elbow) | |
| DBSCAN | Core point: at least min_samples points (itself included) within eps. Clusters are numbered by their first core point; a border point joins the earliest cluster that reaches it; -1 is noise. | scikit-learn `DBSCAN`, same labels |
| k-distance graph | Each point's distance to its (min samples − 1)-th nearest other point with the chosen metric, sorted (a point is core when its value is at most eps); the knee (Kneedle: the point furthest below the chord) suggests eps | scikit-learn `NearestNeighbors` |
| Hierarchical | Agglomerative with ward (euclidean only), complete, average or single linkage on euclidean, manhattan or cosine distances; the linkage matrix and the cophenetic correlation are scipy's. The dendrogram draws the top 12 merges as scipy's `dendrogram(truncate_mode="lastp")`. | scipy `linkage`, `fcluster`, `cophenet`, `dendrogram` |
| GMM | EM with full, tied, diag or spherical covariances (1e-6 added to each diagonal), started from k-means++ seeds; stops when the mean per-sample log-likelihood changes by less than tol. BIC and AIC count parameters per covariance type. | scikit-learn `GaussianMixture` |

### The scores

- Silhouette (euclidean), Davies-Bouldin and Calinski-Harabasz, as
  scikit-learn.
- Labels may be any integers; noise (-1) is left out of every score, and its
  per-sample silhouette shows as 0. Per-cluster silhouettes are listed in
  ascending label order. A cluster of one point has silhouette 0.
- The scores need between 2 and n − 1 clusters; otherwise the Groups tab
  says so.
- Compared with a column: the adjusted Rand index, as scikit-learn's
  `adjusted_rand_score` (noise rows left out).

## The nodes

- **standardise** (default off; Explore > Groups turns it on): z-scores
  with the population standard deviation, as scikit-learn's
  `StandardScaler`.
- Rows missing a feature (null, or not a finite number) are left out of the
  fit and get a null `cluster_id`; the output keeps every row in order.
  Features are read as float64.
- GMM Clustering's **probabilities** adds `probability_0` ..
  `probability_k-1`, each row's chance of each component.
- **Add as a step** and **Keep as a node** in the Groups tab set these from
  the tab's settings.

## Limits

DBSCAN and hierarchical clustering hold an n × n matrix on the device
(DBSCAN as booleans), so they suit up to a few tens of thousands of rows.
Hierarchical clustering does one device step per merge. Silhouette and the
k-distance graph work in row blocks and do not hold the full matrix.

Placed before a Data Split in a training graph, a clustering step learns
from every row, the Validation and Test rows too, and Compile warns. See
[Steps learn from the Train rows](../../concepts/steps-learn-from-train-rows.md).

## Related

- [Explore your data](explore.md)
- [Project rows into two dimensions](dimensionality-reduction.md)
- [Test hypotheses and fit regressions](statistics.md)
