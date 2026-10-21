# Built-in layers and losses written as Python nodes

Dense, Layer Norm, eight activations (ReLU, Leaky ReLU, ELU, Sigmoid,
Tanh, Softplus, GELU, SiLU) and four losses (Cross Entropy, MSE, L1, Huber)
written only in Python with pycyxwiz, as a bundle the Engine loads. Each
computes what the built-in node of the same name computes; read them to see
how a layer or a loss is made, and copy one to start your own.

Licensed under the Apache License 2.0 (the SDK's `LICENSE` and `NOTICE`).

| File | What it is |
|---|---|
| `builtin_layers.py` | The fourteen nodes (`@nodes.node` for Dense, Layer Norm, Leaky ReLU and ELU; `@nodes.activation` for the other activations; `@nodes.loss` for the losses) |
| `nodes/builtin-layers.cyxnode/` | The bundle the Engine reads (`node.json` + `node.py`), made by `nodes.bundle` |
| `blobs_python_layers.cyxgraph` | Data Input -> Split -> Loader -> Dense 16 -> ReLU -> Layer Norm -> Dense 4 -> Cross Entropy (all Python) -> Adam -> Output |
| `blobs.csv` | Four classes over eight features, 480 rows |
| `test_builtin_layers.py` | Checks the nodes (below) |

## Use them in the Engine

1. Copy `nodes/builtin-layers.cyxnode` into your project's `nodes/` folder
   (or your user node folder).
2. Plugin Manager > Python node bundles > builtin-layers > **Approve
   code...**. The fourteen nodes appear under *Python examples* in the node
   editor, as "Dense (Python)", "ReLU (Python)", "MSE (Python)" and so on.
3. Open `blobs_python_layers.cyxgraph` (with `blobs.csv` beside it), open
   the Data Input and apply the file, then **Train**. It learns the four
   classes (about 97% after 10 epochs).

A loss node has the built-in losses' pins: connect the last layer to
**Predictions** and the Data Input's labels (through the Loader) to
**Targets**, and its **Loss** output to the optimizer. Cross Entropy takes
class labels (the Engine gives them one-hot, as wide as the last layer);
MSE, L1 and Huber take a number to predict, with a last layer of one unit.
Huber's setting `delta` is where it turns from squared to linear (the
built-in Huber calls it `beta`).

## Check them

```bash
python test_builtin_layers.py <folder holding pycyxwiz>
```

(in an installed CyxWiz, the folder is `runtime/base/<base>`). It runs the
gradient check on each node, compares each with the Engine's built-in module
(forward, input gradient and Dense's weight gradients agree to 1e-5), and
checks the bundle is current. The losses are checked the same way: Cross
Entropy and MSE against the built-in losses, L1 and Huber against their
formulas. After editing `builtin_layers.py`, bundle again:

```python
from pycyxwiz import nodes
import builtin_layers as b
nodes.bundle(*b.ALL, folder="nodes", name="builtin-layers")
```

and update the `content_hash` in `blobs_python_layers.cyxgraph` (or open the
graph in the Engine and save it).
