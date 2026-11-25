---
title: Train embeddings with pairs and triplets
description: Build a metric-learning graph with the Triplet or Pair Dataset Builder and its loss, so rows of one class land close together.
type: how-to
area: training
audience: user
status: stable
since: 1.0.7
owner: training
last_reviewed: 2026-11-25
related:
  - measure-and-export-embeddings.md
  - losses-and-optimizers.md
redirect_from:
  - ../../usage/metric_learning.md
---

Metric learning trains a model to place rows of the same class close
together and rows of different classes far apart. The model's output is an
embedding (a point in D dimensions), not class scores. To measure, test and
export such a model, see
[Measure and export embeddings](measure-and-export-embeddings.md).

## Before you start

- A table dataset (CSV, Parquet, Arrow) with a label column of class ids:
  whole numbers, one per row. Image and audio folders are not supported for
  metric learning.
- A Data Loader batch size that holds several rows per class: for 10
  classes, 64 rows gives about 6 per class.

## Build the graph

Wire a builder, the encoder (the layers between the builder and the loss)
and the matching loss, and link the Data Loader's **Labels** to the loss:

```text
Data (label = class id) -> Triplet Dataset Builder -> Dense ... -> Dense (D) -> Triplet Loss -> SGD / Adam ...
Data --Labels--> Triplet Loss

Data (label = class id) -> Pair Dataset Builder -> Dense ... -> Dense (D) -> Contrastive Loss
                                                                         or Cosine Embedding Loss -> SGD / Adam ...
Data --Labels--> the loss
```

There is one encoder: every row of a pair or triplet goes through the same
weights. The Pair Dataset Builder goes with the Contrastive or Cosine
Embedding Loss, the Triplet Dataset Builder with the Triplet Loss. Compile
refuses any other combination, a loss without its builder, and a margin out
of range.

## The nodes

| Node | Settings | What it does | PyTorch |
| --- | --- | --- | --- |
| Triplet Dataset Builder | `mining` (random) | Turns each batch of class-labelled rows into triplets: for every row, a positive of the same class and a negative of another class, from the same batch. With `random` mining the batch the model sees is `[anchors; positives; negatives]`. | `x[torch.cat([a, p, n])]` |
| Pair Dataset Builder | `mining` (random) | Turns each batch into pairs: every row gets a partner from the same batch, a same-class one and an other-class one in turn. The batch the model sees is `[firsts; seconds]`; the loss gets "similar or not" for each pair. | `x[torch.cat([a, b])]` |
| Triplet Loss | `margin` (1.0, above 0) | Mean over triplets of max(0, d(a, p) - d(a, n) + margin), d the Euclidean distance | `F.triplet_margin_loss(a, p, n, margin)` |
| Contrastive Loss | `margin` (1.0, 0 or more) | Mean over pairs of d² for a same-class pair and max(0, margin - d)² otherwise | `(s * d**2 + (1 - s) * F.relu(margin - d)**2).mean()` |
| Cosine Embedding Loss | `margin` (0.0, -1 to 1) | Mean over pairs of 1 - cos(a, b) for a same-class pair and max(0, cos(a, b) - margin) otherwise | `F.cosine_embedding_loss(a, b, y, margin)` with y = ±1 |

## Choose the mining

The builders' one setting, `mining`, decides how partners are picked:

| `mining` | Builders | What happens |
| --- | --- | --- |
| `random` (default) | Both | A random positive and negative (or partner) for each row from the batch, before the encoder, stacked. |
| `hard` | Both | The batch goes through the encoder as it is. The loss then takes, for each row, its farthest same-class row and its closest other-class row: the examples the model gets most wrong (batch-hard). Pairs: each row gives two pairs, one of each kind. |
| `semi_hard` | Triplet only | For every same-class pair (anchor, positive), the closest other-class row that is still farther from the anchor than the positive (FaceNet). When none is, the farthest negative. |

Start with `random`. Use `hard` or `semi_hard` once the loss levels off and
the easy triplets no longer teach anything; `semi_hard` is the gentler of
the two (`hard` can push a model with noisy labels toward collapse). Hard
mining also costs less, because the encoder runs over the batch once instead
of over the stacked batch, two or three times larger. For the Cosine
Embedding Loss, "far" and "close" are measured by cosine.

## How it trains

- **One pass.** The encoder runs once over the stacked batch (3T rows for
  triplets, 2P for pairs) and backpropagates once. The gradients of all the
  rows add up in the shared weights, as in PyTorch's
  `encoder(torch.cat([...]))`. Dropout and BatchNorm work; BatchNorm's
  statistics cover the whole stacked batch.
- **Picks from the batch.** A triplet needs a positive and a negative, so a
  row whose class has no other row in the batch is not an anchor, and a
  batch of a single class gives no triplet; it is skipped. A pair row takes
  the other kind of partner when its kind is missing.
- **Batch coverage.** Each epoch, the run logs a warning when some rows had
  no partner in their batch, with the share left out. Those rows taught
  nothing that epoch: raise the Data Loader batch size.
- **Reproducible.** The picks depend on the run's DataLoader seed, the
  epoch, the batch and the row only, so a run and a resumed run replay them.
- **Accuracy.** Triplets: the share with d(a, p) < d(a, n). Pairs: the share
  called correctly, a pair being called similar halfway between the loss's
  targets: Contrastive when d < margin / 2, Cosine Embedding when cos(a, b) >
  (1 + margin) / 2. The loss reaches 0 only when every pair or triplet also
  clears the margin.
- Smoke Run uses the same pair or triplet batches, so its loss and accuracy
  are the metric-learning ones.

With `margin` 1.0, a triplet with d(a, p) = 5 and d(a, n) = 1 has loss
5 - 1 + 1 = 5; one with d(a, p) = 1 and d(a, n) = 3 has loss 0 and counts as
correct. With Contrastive `margin` 2.0, a same-class pair at distance 5 has
loss 25, and an other-class pair at distance 1 has loss (2 - 1)² = 1.

## Example

`examples/cyxgraph/metric_learning/` has two runnable graphs on
`examples/datasets/metric_blobs.csv` (480 rows, 8 features, 4 classes,
label column `class`, split 70 / 15 / 15):

- `triplet_embeddings.cyxgraph`: Triplet Loss with semi-hard mining.
- `contrastive_pairs.cyxgraph`: Contrastive Loss with random pairs.

1. Open one of them.
2. Right-click the Data Input > **Configure...**, then **Apply**.
3. **Train** (F5), then **Train > Test**.

## Related

- [Measure and export embeddings](measure-and-export-embeddings.md)
- [Choose a loss and an optimizer](losses-and-optimizers.md)
- [Training](index.md)
