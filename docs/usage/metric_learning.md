# Metric learning (triplets)

Metric learning trains a model to place rows of the same class close together
and rows of different classes far apart. The model's output is an embedding (a
point in D dimensions), not class scores. Wire it like this:

```
Data (label = class id) -> Triplet Dataset Builder -> Dense ... -> Dense (D) -> Triplet Loss -> SGD / Adam ...
Data --Labels--> Triplet Loss
```

The layers between the builder and the loss are the encoder. There is one
encoder: every anchor, positive and negative goes through the same weights.

## The nodes

| Node | Settings | What it does | PyTorch |
| --- | --- | --- | --- |
| Triplet Dataset Builder | none | Turns each batch of class-labelled rows into triplets: for every row, a positive of the same class and a negative of another class, from the same batch. The batch the model sees is `[anchors; positives; negatives]`. | `x[torch.cat([a, p, n])]` |
| Triplet Loss | `margin` (1.0) | mean over triplets of max(0, d(a, p) - d(a, n) + margin), d the Euclidean distance | `F.triplet_margin_loss(a, p, n, margin)` |

## How it trains

- **One pass.** The encoder runs once over the stacked batch of 3T rows and
  backpropagates once. The gradients from the anchor, positive and negative rows
  add up in the shared weights, exactly as in PyTorch's
  `encoder(torch.cat([a, p, n]))`. Dropout and BatchNorm work; BatchNorm's
  statistics cover all 3T rows, as on the concatenated batch in PyTorch.
- **Triplets from the batch.** For each row of a batch the builder picks a
  positive from the other rows of its class and a negative from the rows of
  the other classes. A row whose class has no other row in the batch is not an
  anchor, and a batch of a single class gives no triplet; it is skipped. Use a
  batch size that holds several rows per class: for 10 classes, 64 rows gives
  about 6 per class.
- **Reproducible.** The picks depend on the run's DataLoader seed, the epoch,
  the batch and the row only, so a run and a resumed run replay them.
- **Accuracy.** For metric learning the Engine reports the triplet accuracy:
  the share of triplets with d(a, p) < d(a, n). The loss reaches 0 only when
  every triplet also clears the margin.
- The label column holds the class ids: whole numbers, one per row.
- The data is a table dataset (CSV, Parquet, Arrow). Image and audio folders
  are not supported yet.
- Smoke Run uses the same triplet batches, so its loss and accuracy are the
  triplet ones.

Example: with `margin` 1.0, a triplet with d(a, p) = 5 and d(a, n) = 1 has loss
5 - 1 + 1 = 5; one with d(a, p) = 1 and d(a, n) = 3 has loss 0 and counts as
correct.

## Not yet

Contrastive and Cosine Embedding losses on pairs, Pair / Retrieval Metrics,
and the Embedding and Pair Score outputs are still blocked (TOFIX140 A5 steps
2-4). The Test step does not yet evaluate a Triplet Loss model.
