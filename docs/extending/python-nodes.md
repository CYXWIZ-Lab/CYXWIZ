# Describing a Python module as a node

`pycyxwiz.nodes` adds what the Engine needs to show a module as a node: an
identity, a category, settings with types and limits, and the shape rule.
Write the module as in [python-modules.md](python-modules.md), then
decorate it.

> Checked 2026-10-17 against pycyxwiz built from branch
> `tofix125-extension-nodes`. The examples are
> `cyxwiz-backend/python/tests/example_nodes.py`, which
> `test_python_nodes.py` and `test_python_bundle.py` run. Using a bundle in
> the Engine comes next (stage P4).

```python
import pycyxwiz as cyx
from pycyxwiz import nodes
```

## 1. Tier 1: an activation from f and its derivative

For an elementwise function without learnable parameters, write `f(x)` and
`df(x)`. The forward and backward steps are generated: forward returns
`f(x)`, backward returns `grad_output * df(x)`. The output shape equals the
input shape.

```python
@nodes.activation(id="com.example.acts", name="Softsign", version="1.0.0",
                  description="x / (1 + |x|)")
class Softsign:
    def f(self, x):
        return x / (x.abs() + 1.0)

    def df(self, x):
        d = x.abs() + 1.0
        return 1.0 / (d * d)
```

The decorated class is a `pycyxwiz.Module`.

## 2. Tier 2: a module with settings and parameters

```python
@nodes.node(id="com.example.acts", name="Snake", version="1.0.0", category="Activation",
            role="activation", description="x + sin^2(a x) / a, learnable a")
class Snake(cyx.Module):
    initial_a = nodes.Param(float, default=1.0, min=0.01, description="Starting value of a")

    def build(self, input_shape):
        self.add_parameter("a", cyx.Tensor.full([1], self.initial_a))

    def forward(self, x): ...      # as in python-modules.md
    def backward(self, grad_output): ...
```

| Part | Meaning |
|---|---|
| `nodes.Param(type, default, min, max, choices, description)` | A setting. `type` is `float`, `int`, `bool` or `str`. Read it as `self.initial_a`; it is read-only inside the node |
| `build(self, input_shape)` | Optional. Called once when the module is made, with one sample's shape (no batch). Create parameters here when their size depends on the input |
| `output_shape(self, input_shape)` | Optional. One sample's output shape. Without it the output shape equals the input shape |
| `role` | `activation` or `layer` (default `layer`) |

## 3. Identity rules

| Field | Rule | Example |
|---|---|---|
| `id` | Lower case: `a-z`, `0-9`, `.`, `-`, `_`. Shared by every node you publish together | `com.example.acts` |
| `name` | Starts with a letter; letters, digits, `_`. Unique within the `id` | `Snake` |
| `version` | Three numbers | `1.0.0` |
| `category` | Where the palette lists it | `Activation` |
| `description` | One line; the docstring's first line when left out | |

A wrong value is refused when the class is decorated, with the rule in the
message.

## 4. Make and check a node the way the Engine will

```python
snake = nodes.create(Snake, {"initial_a": "2.5"}, [16])   # settings as a graph stores them
nodes.output_shape(Snake, {}, [7, 16])                   # [7, 16]
print(nodes.gradcheck(Snake, {"initial_a": "1.0"}))      # Passed: ...
print(nodes.describe(Snake))                              # the manifest entry
```

| Function | Does |
|---|---|
| `create(cls, settings, input_shape)` | Converts and checks the settings, makes the module, calls `build` |
| `output_shape(cls, settings, input_shape)` | The output shape for one sample |
| `gradcheck(cls, settings)` | `pycyxwiz.gradcheck` on fresh modules made with these settings |
| `describe(cls)` | Identity, tier, role, shape rule, pins and settings: the fields a bundle's `node.json` holds |

A refused setting raises `ValueError` with the reason the Engine will show
under the field:

| Setting | Message |
|---|---|
| `{"initial_a": "-1"}` | `initial_a must be at least 0.01` |
| `{"initial_a": "abc"}` | `initial_a must be a number, not 'abc'` |
| `{"speed": "1"}` | `Snake has no setting speed` |

## 5. Bundle the nodes

Put the decorated classes in one `.py` file, import it, and bundle:

```python
from my_nodes import Softsign, Snake
nodes.bundle(Softsign, Snake, folder="nodes", name="example-acts")
# -> nodes/example-acts.cyxnode/
```

The bundle is a folder:

```
nodes/
  example-acts.cyxnode/
    node.json     the manifest, generated: never edit it by hand
    node.py       your .py file, copied unchanged
```

| Rule | Why |
|---|---|
| All nodes of a bundle share one `id` and live in one `.py` file | One file is the bundle's code |
| `name` (default: the first node's name in lower case) uses `a-z`, `0-9`, `-`, `_` | It is the folder name |
| The gradient check runs with each node's default settings and is recorded | Pass `check_gradients=False` to skip it; it is then recorded as `not_checked` |
| A failed check is recorded as `failed`, and the bundle is still written | Finite differences can fail at kinks; the Engine warns instead of refusing |
| `content_hash` is the SHA-256 of `node.py` | The Engine uses it to notice a changed bundle and to approve code |
| Bundling again replaces `node.json` and `node.py` | Edit the `.py`, bundle again |

`content_hash` covers `node.py` byte for byte. If you keep bundles in git,
stop git from changing their line endings, or the Engine reports
"node.py changed since the bundle was written". Add to `.gitattributes`:

```
*.cyxnode/** -text
```

The manifest of the example above (`node.json`, Snake's entry shortened):

```json
{
  "bundle_format": 1,
  "id": "com.example.acts",
  "nodes": [
    {
      "name": "Snake",
      "version": "1.0.0",
      "entry": "node.py:Snake",
      "tier": "module",
      "display_name": "Snake",
      "category": "Activation",
      "role": "activation",
      "description": "x + sin^2(a x) / a, learnable a",
      "shape_rule": "same_as_input",
      "inputs": [{"name": "Input", "type": "Tensor"}],
      "outputs": [{"name": "Output", "type": "Tensor"}],
      "parameters": [
        {"name": "initial_a", "type": "float", "default": "1.0", "validation": "0.01-",
         "description": "Starting value of a", "choices": []}
      ],
      "verification": {"status": "passed", "checked_on": "2026-10-17", "tolerance": 0.02,
                       "max_relative_error": 0.0013, "where": "input"}
    }
  ],
  "requires": {"pycyxwiz": ">=0.2.0", "python_packages": []},
  "content_hash": "sha256:bcffe763..."
}
```

`verification.status` is `passed`, `failed` or `not_checked` (with `error`
saying why, for example a default setting the node refuses).

## 6. Use the bundle in the Engine

The Engine lists every bundle it finds in these folders, without starting
Python:

| Folder | Scope |
|---|---|
| `<project>/nodes/` | This project; wins over a bundle with the same id elsewhere |
| `%APPDATA%\cyxwiz\nodes\` (`~/.cyxwiz/nodes/` elsewhere) | Every project of this user |
| `nodes/` inside a loaded plugin's folder | Delivered with that plugin |

Copy or bundle into one of them; the Engine notices new, edited and removed
bundles within two seconds, without a restart.

**Approval.** A bundle's code runs only after you approve that exact code
(its `content_hash`). Until then its nodes appear in the palette, save and
load like any node, and compiling a graph that uses one says *the code of
Python node bundle ... is not approved yet*. To approve:

- **Plugin Manager** (Apps > Plugin Manager): the section *Python node
  bundles* has a card per bundle with its folder, its nodes, the gradient
  check from bundling, and **Approve code...**. The dialog *Run this Python
  code?* shows what the bundle adds and its code hash; **View code** opens
  `node.py` in the Script Editor.
- **Properties** of one of its nodes: the Gradient check card offers
  **Approve code...** too.

Editing `node.py` (and bundling again) needs a new approval. **Actions >
Revoke approval** on the card takes it back. Approvals are kept per user in
`%APPDATA%\cyxwiz\node_approvals.json`. Bundles that cannot be read are
listed with the reason.

**Python** starts by itself the first time an approved Python node is built
(a project must be open). Settings are checked against the manifest before
that, with the same messages as `nodes.create`, so a refused value shows
under its field without Python.

**Errors** in your `forward` or `backward` stop training with the node's
name, the Python message and the traceback.

**In the Engine's windows** a Python node shows as one: Properties reads
*Python node · com.example.acts 1.0.0 · Activation* and shows the check
from bundling next to its own **Check gradient**; Info lists the bundle,
its approval and code hash; the Node Browser counts *nodes from 2 plugins
and 1 Python bundle*; **Apps > Python node bundles** opens a bundle's
nodes.

## 7. Create a node from a script in the Engine

In the Script Editor, open (and save) a `.py` file of your project that
defines nodes with `@nodes.activation` or `@nodes.node`, then **Run > Create
node from script...**:

1. The dialog lists the nodes it found, with their pins and settings.
2. **Check gradients** runs the check on each with its default settings.
3. Choose the bundle name and **Create nodes**: the bundle is written to
   `<project>/nodes/<name>.cyxnode` and approved for you, since you wrote
   it. Its nodes appear in the Node Browser at once.

A node whose check fails is still created; fix its `backward` and create it
again. A Python error while reading the script shows its message and
traceback in the dialog.

**Speed.** Each batch a Python node takes the interpreter lock and calls
your Python code. Measured 2026-10-17 (GTX 1050 Ti, `bench_python_node_cost`):

| Softsign | forward + backward of a [32, 32] batch | Time per epoch, tiny causal-LM example |
|---|---|---|
| C++ plugin | 221 us | 0.0998 s |
| Python node | 474 us | 0.1015 s |

About 0.25 ms more per batch: invisible in a model whose layers do real
work, noticeable with very small batches. Python nodes do not run in
parallel with each other or with a script running in the Engine (one
interpreter lock), and converting to NumPy inside a node copies to the CPU
and back.

## 8. Examples: ten built-in layers as Python nodes

`examples/python_nodes/builtin_layers/` (in the SDK:
`examples/python-nodes/builtin-layers/`) holds Dense, Layer Norm, ReLU, Leaky
ReLU, ELU, Sigmoid, Tanh, Softplus, GELU and SiLU written as Python nodes,
bundled as `builtin-layers.cyxnode`, with an example graph that feeds them
from a Data Input (`blobs_python_layers.cyxgraph`, four classes from
`blobs.csv`).

| Shows | Where |
|---|---|
| A layer with learnable weights and a setting that changes the output shape | `Dense` (`units`, `build`, `output_shape`, a matrix product through `batch_matmul`) |
| A normalisation with learnable gain and shift and a hand-written backward | `LayerNorm` |
| Tier 1 activations from f and df | `ReLU`, `Sigmoid`, `Tanh`, `Softplus`, `GELU`, `SiLU` |
| Activations with a setting | `LeakyReLU`, `ELU` |

`test_builtin_layers.py` checks each against the Engine's built-in module
(forward and gradients agree to 1e-5) and runs the gradient check; the
Engine test `test_python_builtin_layers` trains the example graph and every
activation from a Data Input. A comparison (`x > 0`) gives a uint8 mask:
multiply it into a float tensor (`Tensor.ones(shape) * (x > 0)`) before
using it in arithmetic, as `relu_mask` in the example does.

## 9. Saving and exporting a model with Python nodes

| Path | What happens |
|---|---|
| Export a CyxWiz package (`.cyxmodel`) | Works. The node's weights are saved, the graph is always included, and the manifest lists the bundle's node types with version and code hash |
| Import the package | The node is rebuilt from the installed, approved bundle and its weights loaded. Missing bundle: refused, "is not installed". Other code than at training: loads, with the warning "is not the one this graph was saved with" |
| ONNX export, generated code (PyTorch, TensorFlow, Keras, pycyxwiz) | Refused with the node's name: the bundle's `node.py` is not carried into them yet |
| Server Node deployment | Refused: the Server Node does not run extension nodes yet |

## 10. Not yet

| Coming | Stage |
|---|---|
| Nodes with several inputs or outputs | Later |
