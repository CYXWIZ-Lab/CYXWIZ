---
title: Find customer segments with analysis nodes
description: Run a graph that scales a customer table, projects it with PCA and groups it with K-Means and DBSCAN, and read its three plots.
type: tutorial
area: general
audience: user
status: experimental
since: 1.0.8
owner: docs
last_reviewed: 2026-11-25
related:
  - index.md
  - ../user-guide/data-studio/clustering.md
  - ../user-guide/data-studio/dimensionality-reduction.md
  - ../user-guide/evaluation/plot-node.md
redirect_from:
  - ../usage/analysis_demo_graph.md
---

In this lesson you run a graph of analysis nodes, not a model: it prepares
a customer table, draws it on two axes, and lets two clustering methods
find its groups. You then compare what they found with the true groups.

**You need:** lesson 2 done, the graph
`examples/cyxgraph/analysis/customer_segments_analysis.cyxgraph` and its
data `customer_segments.csv` in the same folder: 248 made-up customers, so
no download is needed. About 10 minutes.

## 1. Look at the data

The table has five measures per customer: `visits`, `basket_size`,
`returns`, `age` and `spend`. `spend` is heavily right-skewed: a few
customers spend far more than the rest. A `segment` column says which of
four segments of 60 rows each customer belongs to; 8 scattered rows are
planted outliers.

## 2. Open the graph

1. Choose **File > Open** and open
   `examples/cyxgraph/analysis/customer_segments_analysis.cyxgraph`. The
   Data Input's path is relative, so it finds the CSV beside the graph.
2. Press **F** to fit the graph in view.

```text
Customer CSV -> Log spend -> Robust Scaler -> PCA (2) -+-> True segments (Plot)
                                                       +-> K-Means (k = 4) -> K-Means clusters (Plot)
                                                       +-> DBSCAN (eps 0.2) -> DBSCAN clusters (Plot)
```

| Node | What it does here |
| --- | --- |
| **Log spend** (Log Transform) | log(1 + spend), to tame the skew. |
| **Robust Scaler** | Centres each measure on its median and divides by its interquartile range, so outliers do not set the scale. |
| **PCA (2 components)** | Projects the five scaled measures onto two axes, `pc_0` and `pc_1`; `segment` becomes the code column `y`. |
| **K-Means (k = 4)** | Four clusters on `pc_0` and `pc_1` (seed 42, best of 10 starts). |
| **DBSCAN (eps 0.2)** | Density clusters on `pc_0` and `pc_1`; isolated points get cluster -1 (noise). |
| **True segments**, **K-Means clusters**, **DBSCAN clusters** | Plot nodes, each a scatter of `pc_0` against `pc_1`. |

The data comes in through the Data Input node only; every later node works
on the table flowing out of the node before it.

## 3. Draw the plots

1. Double-click **True segments**. The Plot window runs the nodes above it,
   on the device selected in **Preferences > Devices**, and draws the PCA
   projection coloured by the real segment.
2. Double-click **K-Means clusters**. Each colour is one cluster. They line
   up with the segments: nearly all of the 240 segment rows fall in the
   cluster of their segment.
3. Double-click **DBSCAN clusters**. DBSCAN finds four clusters and marks
   most of the 8 planted outliers as -1, noise. K-Means has no noise group,
   so it puts the outliers in its clusters.

You now see the same table three ways, and where the two methods agree with
the truth and with each other.

## 4. Change a setting

1. Click **K-Means (k = 4)** and, in **Properties**, set **Number of
   clusters** (`n_clusters`) to 3.
2. Double-click **K-Means clusters** again. Two segments now share a
   cluster.
3. Click **DBSCAN (eps 0.2)** and set `eps` (the neighbourhood radius) to
   0.5. Double-click **DBSCAN clusters** again: fewer points are noise, and
   clusters that come close can merge.

## Clean up

Close the graph without saving to keep the example as it was.

## Next steps

- [Find groups of rows](../user-guide/data-studio/clustering.md): Data Studio's
  **Explore > Groups** tab, with the cluster scores (silhouette,
  Davies-Bouldin, Calinski-Harabasz) and the clustering nodes.
- [Project rows into two dimensions](../user-guide/data-studio/dimensionality-reduction.md):
  **Explore > Projection**, and the PCA, t-SNE and UMAP nodes.
- [Scaling and transforms](../usage/data_transforms.md): the scalers, and
  the Power Transform and Quantile Transform nodes.
- [Plot data on the canvas](../user-guide/evaluation/plot-node.md)
