# Branching examples

Graphs that are not one straight chain. Both classify
`examples/datasets/metric_blobs.csv` (480 rows, 8 features, 4 classes).

| Graph | What it shows |
| --- | --- |
| `blobs_multibranch_classifier.cyxgraph` | One input read by three branches: a residual (Add) and a wide Tanh branch, concatenated before the classifier (1,972 parameters). |
| `blobs_split_heads_classifier.cyxgraph` | A Split: a shared Dense layer's 16 outputs cut into 6 and 10 for two heads, concatenated before the classifier (356 parameters). |

Run one: open it in the Engine (**File > Open**), double-click **Metric blobs**
and **Apply**, then **Compile** (F7) and **Train** (F5).

Check its PyTorch export (needs Python with `torch`):

```bash
python cyxwiz-engine/tests/pytorch_export/check_pytorch_export.py --bin build/bin/Release \
    examples/cyxgraph/branching/blobs_split_heads_classifier.cyxgraph
```

The check exports the graph, runs a forward and backward pass on an input of
the graph's `sample_shape` (`[8]`), and compares the parameter count with the
model the Engine builds. The TensorFlow and Keras exports write a straight
chain only and refuse both graphs, naming the node where they branch.

More: [Branched graphs](../../../docs/usage/branched_graphs.md).
