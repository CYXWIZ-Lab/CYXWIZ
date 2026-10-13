# Analysis demo graph: customer segments

`examples/cyxgraph/analysis/customer_segments_analysis.cyxgraph` shows the
analysis nodes working together on `customer_segments.csv` (next to it).

## The data

248 customers with five measures:
- `visits`, `basket_size`, `returns`, `age`
- `spend`, heavily right-skewed

Each row also has a `segment` column: four segments of 60 rows plus 8 scattered
outliers.

## The graph

```
Customer CSV -> Log spend -> Robust Scaler -> PCA (2) -+-> True segments (Plot)
                                                       +-> K-Means (k = 4) -> K-Means clusters (Plot)
                                                       +-> DBSCAN (eps 0.2) -> DBSCAN clusters (Plot)
```

| Node | What it does here |
| --- | --- |
| Log spend (Log Transform) | log(1 + spend), to tame the skew |
| Robust Scaler | Centres each measure on its median and divides by its interquartile range, so outliers do not set the scale |
| PCA (2 components) | Projects the five scaled measures onto two axes (`pc_0`, `pc_1`); `segment` becomes the code column `y` |
| K-Means | Four clusters on `pc_0`, `pc_1` (seed 42, best of 10 starts) |
| DBSCAN | Density clusters on `pc_0`, `pc_1`; isolated points get cluster -1 (noise) |

The data comes in through the Data Input node only; every later node works on
the table flowing out of the node before it.

## Run it

1. Open the graph (File > Open). The CSV path is relative and is found beside the graph.
2. Double-click a Plot node. It runs the nodes above it, on the device selected in
   the Devices tab, and draws the scatter:
   - **True segments:** the PCA projection coloured by the real segment.
   - **K-Means clusters:** each colour is one cluster. They line up with the segments
     (236 of 240 segment rows agree).
   - **DBSCAN clusters:** four clusters, with 6 of the 8 planted outliers marked -1.
3. Change a node's settings in Properties (for example k, eps or min_samples)
   and double-click the Plot again to see the effect.

The ctest `pipeline_executor_operator_routing` opens this file and runs all
three plots the same way, so the demo stays working.

## Tools without a node

t-SNE, UMAP, Box-Cox / Yeo-Johnson, the quantile transform and the cluster
scores (silhouette, Davies-Bouldin, Calinski-Harabasz) are tools only
(Data > Transform, Tools > Clustering and Projection). The tools do not yet read
the Data Input node's datasets; that is the next planned step.
