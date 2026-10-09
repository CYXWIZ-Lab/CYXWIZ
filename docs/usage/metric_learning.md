# Metric learning (pairs and triplets)

Metric learning trains a model to place rows of the same class close together
and rows of different classes far apart. The model's output is an embedding (a
point in D dimensions), not class scores. Wire it like this:

```
Data (label = class id) -> Triplet Dataset Builder -> Dense ... -> Dense (D) -> Triplet Loss -> SGD / Adam ...
Data --Labels--> Triplet Loss

Data (label = class id) -> Pair Dataset Builder -> Dense ... -> Dense (D) -> Contrastive Loss
                                                                         or Cosine Embedding Loss -> SGD / Adam ...
Data --Labels--> the loss
```

The layers between the builder and the loss are the encoder. There is one
encoder: every row of a pair or triplet goes through the same weights.

## The nodes

| Node | Settings | What it does | PyTorch |
| --- | --- | --- | --- |
| Triplet Dataset Builder | none | Turns each batch of class-labelled rows into triplets: for every row, a positive of the same class and a negative of another class, from the same batch. The batch the model sees is `[anchors; positives; negatives]`. | `x[torch.cat([a, p, n])]` |
| Pair Dataset Builder | none | Turns each batch into pairs: every row gets a partner from the same batch, a same-class one and an other-class one in turn. The batch the model sees is `[firsts; seconds]`; the loss gets "similar or not" for each pair. | `x[torch.cat([a, b])]` |
| Triplet Loss | `margin` (1.0, above 0) | mean over triplets of max(0, d(a, p) - d(a, n) + margin), d the Euclidean distance | `F.triplet_margin_loss(a, p, n, margin)` |
| Contrastive Loss | `margin` (1.0, 0 or more) | mean over pairs of d² for a same-class pair and max(0, margin - d)² otherwise | `(s * d**2 + (1 - s) * F.relu(margin - d)**2).mean()` |
| Cosine Embedding Loss | `margin` (0.0, -1 to 1) | mean over pairs of 1 - cos(a, b) for a same-class pair and max(0, cos(a, b) - margin) otherwise | `F.cosine_embedding_loss(a, b, y, margin)` with y = ±1 |

The Pair Dataset Builder goes with the Contrastive or Cosine Embedding Loss,
the Triplet Dataset Builder with the Triplet Loss; Compile refuses any other
combination, a loss without its builder, and a margin out of range.

## How it trains

- **One pass.** The encoder runs once over the stacked batch (3T rows for
  triplets, 2P for pairs) and backpropagates once. The gradients of all the
  rows add up in the shared weights, exactly as in PyTorch's
  `encoder(torch.cat([...]))`. Dropout and BatchNorm work; BatchNorm's
  statistics cover the whole stacked batch, as on the concatenated batch in
  PyTorch.
- **Picks from the batch.** For each row of a batch the builder picks a
  positive from the other rows of its class and a negative from the rows of
  the other classes. A triplet needs both, so a row whose class has no other
  row in the batch is not an anchor, and a batch of a single class gives no
  triplet; it is skipped. A pair row takes the other kind of partner when its
  kind is missing (a batch of one class gives only same-class pairs). Use a
  batch size that holds several rows per class: for 10 classes, 64 rows gives
  about 6 per class.
- **Reproducible.** The picks depend on the run's DataLoader seed, the epoch,
  the batch and the row only, so a run and a resumed run replay them.
- **Accuracy.** Triplets: the share of triplets with d(a, p) < d(a, n). Pairs:
  the share of pairs called correctly, a pair being called similar halfway
  between the loss's targets: Contrastive when d < margin / 2, Cosine
  Embedding when cos(a, b) > (1 + margin) / 2. The loss reaches 0 only when
  every pair or triplet also clears the margin.
- The label column holds the class ids: whole numbers, one per row.
- The data is a table dataset (CSV, Parquet, Arrow). Image and audio folders
  are not supported yet.
- Smoke Run uses the same pair or triplet batches, so its loss and accuracy are
  the metric-learning ones.

Examples: with `margin` 1.0, a triplet with d(a, p) = 5 and d(a, n) = 1 has
loss 5 - 1 + 1 = 5; one with d(a, p) = 1 and d(a, n) = 3 has loss 0 and counts
as correct. With Contrastive `margin` 2.0, a same-class pair at distance 5 has
loss 25, and an other-class pair at distance 1 has loss (2 - 1)² = 1.

## Not yet

Pair / Retrieval Metrics and the Embedding and Pair Score outputs are still
blocked (TOFIX140 A5 steps 3-4). The Test step does not yet evaluate
metric-learning models.
