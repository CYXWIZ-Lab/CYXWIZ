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
| Triplet Dataset Builder | `mining` (random) | Turns each batch of class-labelled rows into triplets: for every row, a positive of the same class and a negative of another class, from the same batch. With `random` mining the batch the model sees is `[anchors; positives; negatives]`. | `x[torch.cat([a, p, n])]` |
| Pair Dataset Builder | `mining` (random) | Turns each batch into pairs: every row gets a partner from the same batch, a same-class one and an other-class one in turn. The batch the model sees is `[firsts; seconds]`; the loss gets "similar or not" for each pair. | `x[torch.cat([a, b])]` |
| Triplet Loss | `margin` (1.0, above 0) | mean over triplets of max(0, d(a, p) - d(a, n) + margin), d the Euclidean distance | `F.triplet_margin_loss(a, p, n, margin)` |
| Contrastive Loss | `margin` (1.0, 0 or more) | mean over pairs of d² for a same-class pair and max(0, margin - d)² otherwise | `(s * d**2 + (1 - s) * F.relu(margin - d)**2).mean()` |
| Cosine Embedding Loss | `margin` (0.0, -1 to 1) | mean over pairs of 1 - cos(a, b) for a same-class pair and max(0, cos(a, b) - margin) otherwise | `F.cosine_embedding_loss(a, b, y, margin)` with y = ±1 |

### Mining (the builders' one setting)

`mining` decides how the partners are picked:

| mining | Builders | What happens |
| --- | --- | --- |
| `random` (default) | both | The builder picks a random positive and negative (or partner) for each row from the batch, before the encoder, and stacks them. |
| `hard` | both | The batch goes through the encoder as it is. The loss then takes, for each row, its **farthest** same-class row and its **closest** other-class row: the examples the model gets most wrong (batch-hard). Pairs: each row gives two pairs, one of each kind. |
| `semi_hard` | Triplet only | For every same-class pair (anchor, positive), the closest other-class row that is still farther from the anchor than the positive (FaceNet). When none is, the farthest negative. |

When to use which: start with `random`. Use `hard` or `semi_hard` once the
loss levels off and the easy triplets no longer teach anything; `semi_hard` is
the gentler of the two (`hard` can push a model with noisy labels toward
collapse). Hard mining also costs less, because the encoder runs over the batch
once instead of over the stacked 2-3 times larger batch. For Cosine Embedding
Loss, "far" and "close" are measured by cosine.

### Batch coverage

Each epoch, the run logs a warning when some rows had no partner in their batch
(their class had no other row, or no other class was there), with the share
left out. Those rows taught nothing that epoch: raise the Data Loader batch
size so each batch holds several rows of each class.

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

## Measuring the model

Add **Pair Metrics** and/or **Retrieval Metrics** as side nodes: link their
Embeddings input from the layer that feeds the loss (the encoder output) and
their Class IDs from the Data Loader's Labels. They do not change training.

```
... -> Dense (D) --> Triplet Loss -> optimizer
            |------> Retrieval Metrics (k)
            |------> Pair Metrics (threshold)
Data Loader --Labels--> each of them
```

| Node | Setting | Reports |
| --- | --- | --- |
| Retrieval Metrics | `k` (10) | Every row is a query against all the other rows (leave-one-out). **Recall@k**: share of queries with a same-class row among their k nearest. **MRR**: mean of 1 / rank of the first same-class row. **1-NN agreement**: share whose nearest row is of their class. |
| Pair Metrics | `threshold` (0.5) | Each row is paired with a same-class or an other-class row of its batch in turn (seeded by the Data Loader seed); a pair is called similar when its embeddings are at most `threshold` apart. **Pair accuracy**, and the mean distance of same-class and of other-class pairs (pick the threshold between them). |

When they run:

- **Every validated epoch**, on the validation rows: a Console line
  (`Epoch 3 validation: pair accuracy 0.9100 ..., Recall@10 0.9700, MRR 0.8800, 1-NN 0.8500 over 240 rows`)
  and the dashboard series *Val Pair Accuracy*, *Val Recall@k*, *Val MRR* and
  *Val 1-NN Agreement* (in %).
- **At the end of training**, on the held-out test rows (Console line).
- **In the Test step** (below).

Distances are Euclidean; for a Cosine Embedding model the embeddings are
L2-normalised first, so the ranking is the cosine ranking. Up to 4096 rows of a
partition are used (the Console says when the cap is hit).

## Testing a metric model

Train > Test tests a metric model as a **1-NN classifier**: every test row takes
the class of its nearest other test row in the embedding space. Accuracy, the
confusion matrix and the per-class tab show that classifier (the Overview says
"1-NN Accuracy"). The test loss is the model's loss over the builder's pairs or
triplets of the test rows. With Pair / Retrieval Metrics in the graph, the
Overview also shows Recall@k, MRR, the pair accuracy and the mean distances.

## Exporting and serving

Two more side nodes on the encoder output turn the trained model into
something you can use:

| Node | Settings | What it does |
| --- | --- | --- |
| Embedding Output | `file_path` (exports/embeddings.parquet), `partition` (all / train / validation / test), `include_metadata` | After training, every row of the partition goes through the trained encoder and the Parquet file gets one row per data row: `e0 .. e(D-1)` and, with metadata, `class`, `partition` and `row` (position in the partition). A relative path is in the project folder. Load it with a Data Input and plot it: a 2-D embedding is a scatter of e0 against e1 coloured by class. |
| Pair Score Output | `score_mode` (distance / negative_distance / cosine_similarity), `threshold` (0.5) | Saved with the trained model (Export Model). The local inference server's `/v1/pair-score` uses this mode when a request gives none, and adds `"same": true/false` to every pair plus the `threshold` to the response: same when the distance is at most the threshold, the negative distance at least -threshold, or the cosine similarity at least the threshold. `/v1/model` shows the defaults under `pair_score_defaults`. A request that asks for another `score_mode` gets scores without same / different. |

Pick the threshold from Pair Metrics: its mean same-class and other-class
distances show where to cut.

Serving embeddings needs no node: `/v1/embeddings` returns them for any loaded
model.

## Example

`examples/cyxgraph/metric_learning/` has two runnable graphs on
`examples/datasets/metric_blobs.csv` (480 rows, 8 features, 4 classes, label
column `class`, split 70 / 15 / 15):

- `triplet_embeddings.cyxgraph`: Triplet Loss with semi-hard mining.
- `contrastive_pairs.cyxgraph`: Contrastive Loss with random pairs.

Both carry Retrieval Metrics (k 5), Pair Metrics, an Embedding Output
(`exports/<graph>_embeddings.parquet`) and a Pair Score Output. Open one, apply
the Data Input (right-click > Configure..., Apply), Train, then Test.
