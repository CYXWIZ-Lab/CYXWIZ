---
title: Measure and export embeddings
description: Score a metric-learning model with Pair and Retrieval Metrics, test it as a 1-NN classifier, and write or serve its embeddings.
type: how-to
area: training
audience: user
status: stable
since: 1.0.7
owner: training
last_reviewed: 2026-11-25
related:
  - metric-learning.md
  - ../evaluation/test-a-trained-model.md
---

A metric-learning model outputs embeddings, not class scores, so it is
measured by how well same-class rows find each other. Side nodes on the
encoder output measure it while it trains and write its embeddings after.
To build the model, see [Train embeddings with pairs and triplets](metric-learning.md).

## Measure while training

1. Add **Pair Metrics** and/or **Retrieval Metrics** to the graph.
2. Link their **Embeddings** input from the layer that feeds the loss (the
   encoder output) and their **Class IDs** from the Data Loader's
   **Labels**. They do not change training.
3. Train.

```text
... -> Dense (D) --> Triplet Loss -> optimizer
            |------> Retrieval Metrics (k)
            |------> Pair Metrics (threshold)
Data Loader --Labels--> each of them
```

| Node | Setting | Reports |
| --- | --- | --- |
| Retrieval Metrics | `k` (10) | Every row is a query against all the other rows (leave-one-out). **Recall@k**: the share of queries with a same-class row among their k nearest. **MRR**: the mean of 1 / rank of the first same-class row. **1-NN agreement**: the share whose nearest row is of their class. |
| Pair Metrics | `threshold` (0.5) | Each row is paired with a same-class or an other-class row of its batch in turn (seeded by the Data Loader seed); a pair is called similar when its embeddings are at most `threshold` apart. **Pair accuracy**, and the mean distance of same-class and of other-class pairs (pick the threshold between them). |

They run:

- **Every validated epoch**, on the validation rows: a Console line
  (`Epoch 3 validation: pair accuracy 0.9100 ..., Recall@10 0.9700, MRR 0.8800, 1-NN 0.8500 over 240 rows`)
  and the Training Dashboard series *Val Pair Accuracy*, *Val Recall@k*,
  *Val MRR* and *Val 1-NN Agreement* (in %).
- **At the end of training**, on the held-out test rows (a Console line).
- **In the test** (next section).

Distances are Euclidean; for a Cosine Embedding model the embeddings are
L2-normalised first, so the ranking is the cosine ranking. Up to 4096 rows
of a partition are used (the Console says when the cap is hit).

## Test a metric model

**Train > Test** tests a metric model as a **1-NN classifier**: every test
row takes the class of its nearest other test row in the embedding space.
Accuracy, the confusion matrix and the per-class view show that classifier
(the headline says *1-NN accuracy*). The test loss is the model's loss over
the builder's pairs or triplets of the test rows. With Pair or Retrieval
Metrics in the graph, the overview also shows Recall@k, MRR, the pair
accuracy and the mean distances. See
[Test a trained model](../evaluation/test-a-trained-model.md).

## Write or serve the embeddings

Two more side nodes on the encoder output:

| Node | Settings | What it does |
| --- | --- | --- |
| Embedding Output | `file_path` (`exports/embeddings.parquet`), `partition` (all / train / validation / test), `include_metadata` | After training, every row of the partition goes through the trained encoder, and the Parquet file gets one row per data row: `e0 .. e(D-1)` and, with metadata, `class`, `partition` and `row` (position in the partition). A relative path is in the project folder. Load it with a Data Input and plot it: a 2-D embedding is a scatter of e0 against e1 coloured by class. |
| Pair Score Output | `score_mode` (distance / negative_distance / cosine_similarity), `threshold` (0.5) | Saved with the trained model (Export Model). The local inference server's `/v1/pair-score` uses this mode when a request gives none, and adds `"same": true/false` to every pair plus the `threshold` to the response: same when the distance is at most the threshold, the negative distance at least -threshold, or the cosine similarity at least the threshold. `/v1/model` shows the defaults under `pair_score_defaults`. A request that asks for another `score_mode` gets scores without same / different. |

Pick the threshold from Pair Metrics: its mean same-class and other-class
distances show where to cut. Serving embeddings needs no node:
`/v1/embeddings` returns them for any loaded model.

The examples in `examples/cyxgraph/metric_learning/` carry Retrieval Metrics
(k 5), Pair Metrics, an Embedding Output
(`exports/<graph>_embeddings.parquet`) and a Pair Score Output.

## Related

- [Train embeddings with pairs and triplets](metric-learning.md)
- [Test a trained model](../evaluation/test-a-trained-model.md)
- [Training](index.md)
