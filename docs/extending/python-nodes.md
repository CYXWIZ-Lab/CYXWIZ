# Describing a Python module as a node

`pycyxwiz.nodes` adds what the Engine needs to show a module as a node: an
identity, a category, settings with types and limits, and the shape rule.
Write the module as in [python-modules.md](python-modules.md), decorate it,
bundle it, and the Engine lists it in the node palette.

> Checked against plugin API 1.2.0 and pycyxwiz 0.2.0.
> The Python samples on this page run, in order, as one script
> (`scripts/docs/check_extension_docs.py` runs them). To run them yourself
> outside the Engine, first make pycyxwiz importable
> ([python-modules.md](python-modules.md) section 1). The source of
> `pycyxwiz.nodes` is `cyxwiz-backend/python/pycyxwiz_nodes.py`; its tests are
> `cyxwiz-backend/python/tests/test_python_nodes.py`,
> `test_python_bundle.py` and `test_python_losses.py`.

```python
import pycyxwiz as cyx
from pycyxwiz import nodes
```

There are three decorators: `@nodes.activation` (an elementwise function
from `f` and its derivative), `@nodes.node` (any module with `forward` and
`backward`) and `@nodes.loss`. There is no `@nodes.layer`: a layer is
`@nodes.node` with `role="layer"`, the default.

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

print(issubclass(Softsign, cyx.Module))   # True: the decorator made a module
```

The class needs no base class; the decorated class is a `pycyxwiz.Module`.
`f` may use the forward-only module functions such as `pycyxwiz.tanh`
([python-modules.md](python-modules.md) section 3); `df` is still yours:

```python
@nodes.activation(id="com.example.acts", name="TanhAct", version="1.0.0",
                  display_name="Tanh (mine)")
class TanhAct:
    """tanh(x)"""

    def f(self, x):
        return cyx.tanh(x)

    def df(self, x):
        t = cyx.tanh(x)
        return 1.0 - t * t
```

## 2. Tier 2: a module with settings and parameters

```python
@nodes.node(id="com.example.acts", name="Snake", version="1.0.0", category="Activation",
            role="activation", description="x + sin^2(a x) / a, learnable a")
class Snake(cyx.Module):
    initial_a = nodes.Param(float, default=1.0, min=0.01, description="Starting value of a")

    def build(self, input_shape):
        self.add_parameter("a", cyx.Tensor.full([1], self.initial_a))

    def forward(self, x):
        self.x = x
        a = self.parameter("a").broadcast_to(x.shape())
        self.a = a
        self.s = (x * a).sin()
        return x + (self.s * self.s) / a

    def backward(self, grad_output):
        a = self.a
        sin2 = (self.x * a * 2.0).sin()
        da = (self.x * sin2) / a - (self.s * self.s) / (a * a)
        self.set_gradient("a", (grad_output * da).sum().reshape([1]))
        return grad_output * (sin2 + 1.0)
```

| Part | Meaning |
|---|---|
| `nodes.Param(...)` | A setting (section 5). Read it as `self.initial_a`; it is read-only inside the node |
| `__init__(self)` | Optional. The Engine calls it with no arguments, so it may not require any. If you define it, call `super().__init__()` first. Settings can already be read in it |
| `build(self, input_shape)` | Optional. Called once after `__init__`, with one sample's shape (no batch dimension): `[16]` for a batch `[32, 16]`. Create parameters here when their size depends on the input |
| `output_shape(self, input_shape)` | Optional. One sample's output shape from one sample's input shape. Without it the output shape equals the input shape. It runs on a module that was never built: use the settings and `input_shape` only, not parameters or anything `build` sets |
| `forward`, `backward` | As in [python-modules.md](python-modules.md); both are required |
| `set_training(self, training)` | Optional, as in [python-modules.md](python-modules.md) section 5; the Engine switches to evaluation for validation and testing |

`build` is called with other shapes than in your graph too: the gradient
check builds fresh modules for `[3]` and `[4, 5]` (one sample of the check's
batches `[2, 3]` and `[3, 4, 5]`). A node that only works on rank-1 samples
gives "Could not check: ..." with its own error there, and a bundle records
that as `not_checked`. Check such a node yourself with the shapes it takes:
`nodes.gradcheck(cls, input_shapes=[[4, 3]])` (section 6).

**role** says what the graph checks treat the node as:

| role | Meaning |
|---|---|
| `layer` (default) | A model layer, as the built-in Dense: a graph needs at least one model layer, and the last one sets the model's output size |
| `activation` | Not a model layer, as the built-in ReLU: it keeps the size of the layer before it |

`@nodes.activation` takes no `role` (it is always `activation`).

## 3. A loss

A loss takes the model's predictions and the targets and gives one value.
Subclass `pycyxwiz.Loss` and decorate it with `@nodes.loss`:

```python
@nodes.loss(id="com.example.losses", name="Huber", version="1.0.0", target="continuous")
class Huber(cyx.Loss):
    """Squared error below delta, linear above"""
    delta = nodes.Param(float, default=1.0, min=1e-6, description="Where it turns linear")

    def forward(self, predictions, targets):
        d = predictions - targets
        a = d.abs()
        small = (a < self.delta) * 1.0          # float32 mask (section 3 of python-modules.md)
        return (d * d * 0.5 * small + (a - 0.5 * self.delta) * self.delta * (1.0 - small)).mean()

    def backward(self, predictions, targets):
        d = predictions - targets
        return d.clip(-self.delta, self.delta) * (1.0 / predictions.num_elements())
```

| Part | Meaning |
|---|---|
| `forward(self, predictions, targets)` | One value: a Tensor of one element (`.mean()` and `.sum()` give one). Anything else stops training with "forward must return one value" |
| `backward(self, predictions, targets)` | dLoss/dpredictions, with the predictions' shape. Include the mean scaling (divide by the element or row count) as the built-in losses do: the optimizer uses this gradient as it is |
| `predictions` | The last layer's outputs as they are, `[batch, outputs]`. Nothing is applied before your loss: a classification loss applies its own softmax, as `CrossEntropy` in the examples (section 13) does |
| `target` | `categorical` (default): the targets are one-hot rows as wide as the model's output, and training reports accuracy by argmax. `continuous`: the targets are the values to predict, and training reports regression metrics |
| Settings | `nodes.Param`, as for any node |

A loss has no `build`, no input shape and no learnable parameters. If you
define `__init__`, call `super().__init__()` first and require no arguments.

In the graph a loss node has the built-in losses' pins: **Predictions**
(the last layer), **Targets** (the Data Input's labels, through the Loader)
and **Loss** (to the optimizer). It is the graph's training objective, not
a layer of the model.

## 4. Decorator arguments

```
nodes.activation(id, name, version, category="Activation", description="", display_name=None)
nodes.node(id, name, version, category="Custom", role="layer", description="", display_name=None)
nodes.loss(id, name, version, target="categorical", category="Loss", description="", display_name=None)
```

| Argument | Rule | Example |
|---|---|---|
| `id` | Lower case: starts with `a-z` or `0-9`, then `a-z`, `0-9`, `.`, `-`, `_`. Shared by every node you bundle together | `com.example.acts` |
| `name` | Starts with a letter; letters, digits, `_`. Unique within the `id`. The node's type name in graphs | `Snake` |
| `version` | Three numbers | `1.0.0` |
| `display_name` | What the palette and the node show. Default: `name` | `Tanh (mine)` |
| `category` | Where the palette lists it. Defaults: `Activation`, `Custom`, `Loss` | `Activation` |
| `description` | One line. Default: the first line of the class docstring | |
| `role` | `@nodes.node` only: `layer` or `activation` (section 2) | `activation` |
| `target` | `@nodes.loss` only: `categorical` or `continuous` (section 3) | `continuous` |

A wrong value is refused when the class is decorated, with the rule in the
message, for example `id 'Com.Example' must be lower case: a-z, 0-9, '.',
'-', '_' (for example com.you.my-nodes)`. `@nodes.node` also refuses a class
that is not a `pycyxwiz.Module` subclass or lacks `forward` or `backward`;
`@nodes.activation` refuses a class without `f` or `df`.

## 5. Settings: nodes.Param

`nodes.Param(type=float, default=None, min=None, max=None, choices=None, description="")`

| Argument | Meaning |
|---|---|
| `type` | `float`, `int`, `bool` or `str` |
| `default` | The starting value. Without one: `0.0`, `0`, `False`, `""`, or the first of `choices`. The default itself must meet `min`, `max` and `choices`, or the class is refused ("the default of this Param must be at most 1") |
| `min`, `max` | Limits for `float` and `int`; the Engine refuses a value outside them under the field |
| `choices` | A list of allowed values; the Engine shows a drop-down |
| `description` | Shown under the setting in Properties |

The setting's name is the attribute name (`initial_a`). The Engine stores
every setting as text in the graph and converts it before your code sees it:
`"2.5"` becomes `2.5` for a `float`, `"true"`, `"1"`, `"yes"` (and `"false"`,
`"0"`, `"no"`, `""`) become a `bool`. In `node.json` a `str` setting is
written with the type `string`, and a setting with `choices` lists them; its
`validation` is the rule as text (`0.01-` for min 0.01, `0-1`, `a|b`).

```python
@nodes.node(id="com.example.acts", name="Scale", version="1.0.0")
class Scale(cyx.Module):
    factor = nodes.Param(float, default=2.0, min=0.0, max=10.0, description="Multiplies the input")
    mode = nodes.Param(str, choices=["plain", "negate"], description="Sign of the result")

    def __init__(self):
        super().__init__()
        self.sign = -1.0 if self.mode == "negate" else 1.0   # settings are readable here

    def forward(self, x):
        return x * (self.factor * self.sign)

    def backward(self, grad_output):
        return grad_output * (self.factor * self.sign)

print(nodes.describe(Scale)["parameters"][1])
# {'name': 'mode', 'type': 'string', 'default': 'plain', 'validation': 'plain|negate', ...}
```

## 6. Make and check a node the way the Engine will

```python
snake = nodes.create(Snake, {"initial_a": "2.5"}, [16])  # settings as a graph stores them
print(snake.parameter("a").at(0))                        # 2.5
print(nodes.output_shape(Snake, {}, [7, 16]))            # [7, 16]
print(nodes.gradcheck(Snake, {"initial_a": "1.0"}))      # Passed: ...
print(nodes.gradcheck(Huber))                            # a loss: on value targets
print(nodes.describe(Snake)["shape_rule"])               # same_as_input
```

| Function | Does |
|---|---|
| `create(cls, settings=None, input_shape=None)` | Converts and checks the settings, calls `__init__()` with no arguments, then `build(input_shape)` when both exist. `input_shape` is one sample's shape. A loss: `create(cls, settings)` |
| `output_shape(cls, settings=None, input_shape=())` | One sample's output shape, from `output_shape` on a module that was not built (not for a loss) |
| `gradcheck(cls, settings=None, **options)` | `pycyxwiz.gradcheck` on fresh modules made with these settings, or `pycyxwiz.gradcheck_loss` for a loss (one-hot targets for `categorical`, values for `continuous`). `options` are passed on |
| `describe(cls)` | Identity, tier, role, shape rule, pins and settings: the fields a bundle's `node.json` holds |
| `is_node(cls)`, `is_loss(cls)` | Whether a class carries one of the decorators, or `@nodes.loss` |

The options of both checkers, with their defaults: `input_shapes` (empty:
`[[2, 3], [3, 4, 5]]` for a module, prediction shapes `[[2, 3], [4, 5]]`
for a loss), `seed=125`, `tolerance=0.02`, `step=0.01`, `input_range=2.0`.
`pycyxwiz.gradcheck_loss(make, targets="one_hot")` takes a `make()` with no
arguments; `targets` is `"one_hot"` or `"values"`. A loss whose backward
leaves out the mean scaling fails it. The meaning of each option is in
[python-modules.md](python-modules.md) section 7.

```python
print(nodes.gradcheck(Snake, input_shapes=[[8, 16]], tolerance=0.01))
```

A refused setting raises `ValueError` with the reason the Engine shows under
the field:

| Setting | Message |
|---|---|
| `{"initial_a": "-1"}` | `initial_a must be at least 0.01` |
| `{"initial_a": "abc"}` | `initial_a must be a number, not 'abc'` |
| `{"speed": "1"}` | `Snake has no setting speed` |

```python
try:
    nodes.create(Snake, {"initial_a": "-1"}, [16])
except ValueError as error:
    print(error)   # initial_a must be at least 0.01
```

## 7. The node file

The nodes of a bundle live in one `.py` file, which becomes the bundle's
`node.py`. The Engine loads it on its own:

- **One self-contained file.** `node.py` is loaded from its path; its folder
  is not on `sys.path`. Helper modules of your own and relative imports
  (`from .util import x`) do not work: put helpers in the same file, as
  `rows` and `softmax` are in the examples (section 13).
- **Packages come from the Engine's Python.** `import math` and packages
  installed in the Python the Engine runs for the project work. Nothing is
  installed for a bundle: `requires.python_packages` in `node.json` is always
  `[]`. A node that imports a package the user lacks fails when it is built.
- **Top-level code runs on every load.** The Engine imports the whole file
  when a node is first built, and **Create node from script** imports it
  three times (reading, checking, writing). Keep the top level to imports,
  constants, helpers and classes. Put anything else (tests, bundling, prints)
  under `if __name__ == "__main__":`; the Engine never imports the file
  under that name.

## 8. Bundle the nodes

`nodes.bundle(*classes, folder="nodes", name=None, check_gradients=True)`
writes `<folder>/<name>.cyxnode/`. The classes must come from a `.py` file
that Python loaded, by `import my_nodes` or by `python my_nodes.py`:
`bundle` copies that file. `folder` is relative to the
current folder; a script run from the Engine's Script Editor runs in the
project folder, so `"nodes"` is `<project>/nodes`.

This sample writes a small `my_nodes.py` and bundles it. Your own file is
the same: decorated classes, and bundling under the `__main__` guard so that
`python my_nodes.py` in a terminal writes the bundle:

```python
import importlib
import sys

with open("my_nodes.py", "w", encoding="utf-8") as f:
    f.write('''import pycyxwiz as cyx
from pycyxwiz import nodes


@nodes.activation(id="com.example.acts", name="Softsign", version="1.0.0",
                  description="x / (1 + |x|)")
class Softsign:
    def f(self, x):
        return x / (x.abs() + 1.0)

    def df(self, x):
        d = x.abs() + 1.0
        return 1.0 / (d * d)


if __name__ == "__main__":
    print(nodes.bundle(Softsign, folder="nodes", name="example-acts"))
''')

sys.path.insert(0, "")              # the current folder, where my_nodes.py is
import my_nodes
importlib.reload(my_nodes)          # read the file again after an edit
print(nodes.bundle(my_nodes.Softsign, folder="nodes", name="example-acts"))
# nodes\example-acts.cyxnode (nodes/example-acts.cyxnode elsewhere)
```

In the Engine, Python keeps running between scripts, so a second
`import my_nodes` returns the module as first imported: call
`importlib.reload(my_nodes)` after editing the file, as above, or use
**Create node from script** (section 11), which reads the file afresh. You
can also run `my_nodes.py` itself with **Run Script** (F5) in the Script Editor: a saved
script runs as its file (`__file__` is its path, tracebacks name it), so the
`bundle(...)` under its `__main__` guard copies that file. **Run selection**
and **Run section** run part of the file without a file name; `bundle` then
stops with "Softsign is not defined in a .py file; save it in a file and
import it to bundle it".

The bundle is a folder:

```
nodes/
  example-acts.cyxnode/
    node.json     the manifest, generated: never edit it by hand
    node.py       your .py file, copied byte for byte
```

| Rule | Why |
|---|---|
| All nodes of a bundle share one `id` and live in one `.py` file | One file is the bundle's code |
| `name` (default: the first node's name in lower case) starts with `a-z` or `0-9`, then `a-z`, `0-9`, `-`, `_` | It is the folder name |
| The gradient check runs with each node's default settings and is recorded | Pass `check_gradients=False` to skip it; it is then recorded as `not_checked` |
| A failed check is recorded as `failed`, and the bundle is still written | Finite differences can fail at kinks, and a random node (a dropout) always fails; the Engine warns instead of refusing |
| `content_hash` is the SHA-256 of `node.py` | The Engine uses it to notice a changed bundle and to approve code |
| Bundling again replaces `node.json` and `node.py` | Edit the `.py`, bundle again |

```python
import json
import os

with open(os.path.join("nodes", "example-acts.cyxnode", "node.json"), encoding="utf-8") as f:
    manifest = json.load(f)
print(manifest["content_hash"][:7], manifest["nodes"][0]["verification"]["status"])   # sha256: passed
```

The manifest (`node.json`) of this bundle:

```json
{
  "bundle_format": 1,
  "id": "com.example.acts",
  "nodes": [
    {
      "name": "Softsign",
      "version": "1.0.0",
      "entry": "node.py:Softsign",
      "tier": "activation",
      "display_name": "Softsign",
      "category": "Activation",
      "role": "activation",
      "description": "x / (1 + |x|)",
      "shape_rule": "same_as_input",
      "inputs": [{"name": "Input", "type": "Tensor"}],
      "outputs": [{"name": "Output", "type": "Tensor"}],
      "parameters": [],
      "verification": {"status": "passed", "checked_on": "2026-10-28", "tolerance": 0.02,
                       "max_relative_error": 4.5e-05, "where": "input"}
    }
  ],
  "requires": {"pycyxwiz": ">=0.2.0", "python_packages": []},
  "content_hash": "sha256:5d8f1fad..."
}
```

| Field | Values |
|---|---|
| `tier` | `activation` (`@nodes.activation`), `module` (`@nodes.node`), `loss` |
| `shape_rule` | `same_as_input`, or `python` when the class defines `output_shape` (not for a loss) |
| `parameters` | One entry per setting: `name`, `type` (`float`, `int`, `bool`, `string`), `default` as text, `validation`, `description`, `choices` |
| `verification.status` | `passed`, `failed` or `not_checked` (with `error` saying why, for example a default setting the node refuses) |

A loss's entry has `"tier": "loss"`, `"role": "loss"`, its `"target"`, the
pins Predictions (Tensor), Targets (Labels) -> Loss (Loss), and no
`shape_rule`.

`content_hash` covers `node.py` byte for byte. If you keep bundles in git,
stop git from changing their line endings, or the Engine reports
"node.py changed since the bundle was written". Add to `.gitattributes` at
the repository root:

```
**/*.cyxnode/** -text
```

**Versions and names.** Graphs save each node's `version` and code hash.
Raise `version` whenever you change a node's settings, pins or code. A graph
saved with another version says so when it is opened ("Saved with version
1.0.0, installed version is 1.1.0."); new code under the same version gives
"Its code changed since the graph was saved, with the same version number."
Settings and parameters are stored by name: a renamed setting makes older
graphs fail with "Snake has no setting <old name>", and a renamed parameter
(`add_parameter`) is not filled from weights saved under the old name in an
older `.cyxmodel`.

## 9. Use the bundle in the Engine

The Engine lists every bundle it finds in these folders, without starting
Python:

| Folder | Scope |
|---|---|
| `<project>/nodes/` | This project |
| `%APPDATA%\cyxwiz\nodes\` (`~/.cyxwiz/nodes/` elsewhere) | Every project of this user |
| `nodes/` inside a loaded plugin's folder | Delivered with that plugin |

Copy or bundle into one of them. The Engine looks every 2 seconds: new,
edited and removed bundles show within 2 seconds, without a restart.

**One id, one bundle.** Two bundles with the same `id` do not both load:
the first one found loads and the other's card says "another bundle with
the id ... is already loaded". The folders are searched in the order of the
table: a project's bundle wins over the user folder's, which wins over a
plugin's. Within one folder, bundles are read in name order. A bundle whose id a loaded C++ plugin
already uses does not load: "the id ... is already used by a loaded
plugin".

**A bundle that cannot be read** is listed in the Plugin Manager with the
reason, and its nodes are not available: graphs that use them show them as
**Not installed**. The common reason is "node.py changed since the bundle
was written; bundle it again with pycyxwiz.nodes.bundle()", after a hand
edit of `node.py` or a git checkout that changed its line endings.

**Approval.** A bundle's code runs only after you approve that exact code.
An approval is for the bundle's `id` and `content_hash`, not its folder: a
copy of an approved bundle elsewhere is approved too, and any change to
`node.py` needs a new approval. Until then its nodes appear in the palette,
save and load like any node, and compiling a graph that uses one says
"the code of Python node bundle ... is not approved yet; approve it in the
Plugin Manager to run it". To approve:

- **Plugin Manager** (Apps > Plugin Manager): the section *Python node
  bundles* has a card per bundle with its folder, its nodes, the gradient
  check from bundling, and **Approve code...**. The dialog *Run this Python
  code?* shows what the bundle adds and its code hash; **View code** opens
  `node.py` in the Script Editor.
- **Properties** of one of its nodes: the Gradient check card offers
  **Approve code...** too.
- **Create node from script** (section 11) approves the bundle it writes,
  every time it writes it.

**Actions > Revoke approval** on the card takes an approval back. Approvals
are kept per user in `%APPDATA%\cyxwiz\node_approvals.json`
(`~/.cyxwiz/node_approvals.json` elsewhere). `nodes.bundle()` run by hand
does not approve.

**Python** starts by itself when an open graph uses an approved Python node,
or the first time one is built. It needs an open project. Until it runs,
compiling says "Python is starting for the Python nodes (it needs an open
project); try again in a moment". A CyxWiz built without Python says
"Python is not available in this build of CyxWiz". Settings are checked
against the manifest before Python is needed, with the same messages as
`nodes.create`, so a refused value shows under its field without Python.

**In the Engine's windows** a Python node shows as one: Properties reads
*Python node · com.example.acts 1.0.0 · Activation* and shows the check
from bundling next to its own **Check gradient**; Info lists the bundle,
its approval and code hash; the Node Browser counts *nodes from 2 plugins
and 1 Python bundle*; **Apps > Python node bundles** opens a bundle's
nodes.

## 10. When a node fails in the Engine

An error in your code stops training. The training status reads
*Training failed: ...* with the node's name and type id, what failed and
the Python traceback, for example:

```
Extension node 'Snake' (com.example.acts:Snake): forward failed: Snake.forward raised ValueError: ...
```

The full text, with every traceback line, is in the Console (its Logs
session) and in `engine_log.txt` next to `cyxwiz-engine.exe`. The Engine
also stops training when a node breaks the contract, with these messages:

| Message | Cause |
|---|---|
| `expected output shape [16] per sample, got [32, 8] for input [32, 16]` | `forward` returned another shape than `output_shape` (or the input's shape) says |
| `changed the batch size from 32 to 1` | `forward` changed dimension 0 |
| `backward returned gradient shape [...], expected the input shape [...]` | `backward` returned another shape than its input |
| `its gradients do not match its parameters (...)` | A parameter has no gradient, or one of another shape |
| `forward must return one value, got shape [...]` | A loss returned more than one element |

`print()` inside `forward` or `backward` does not reach any Engine window
during training: the Engine shows a script's output only while that script
runs. To look inside a node, reproduce the failure in the Script Editor or a
terminal: make the node as the Engine does and feed it a batch by hand.

```python
snake = nodes.create(Snake, {"initial_a": "0.5"}, [16])   # the graph's settings, one sample's shape
batch = cyx.Tensor.random([8, 16]) * 2.0 - 1.0            # 8 samples
out = snake.forward(batch)
print(out.shape())                                        # [8, 16]
grad = snake.backward(cyx.Tensor.ones(out.shape()))
print(grad.shape(), snake.gradient("a").shape())          # [8, 16] [1]
```

An edit reaches the Engine only through a new bundle: change the `.py`
file, bundle it again (or **Create node from script**), and approve the new
code. The Engine loads each version of `node.py` once, by its hash, so a
new bundle is picked up without restarting.

## 11. Create a node from a script in the Engine

In the Script Editor, open and save a `.py` file of your project that
defines nodes with `@nodes.activation`, `@nodes.node` or `@nodes.loss`,
then **Run > Create node from script...** (it needs an open project, a saved
`.py` file, and no script running):

1. The dialog lists the nodes it found, with their pins and settings. Only
   classes defined in that file count, not ones it imports.
2. **Check gradients** runs the check on each with its default settings.
3. Choose the bundle name (suggested from the file name) and **Create
   nodes**: the bundle is written to `<project>/nodes/<name>.cyxnode` and
   approved for you, since you wrote it. Its nodes appear in the Node
   Browser within 2 seconds.

A node whose check fails is still created; fix its `backward` and create it
again. A Python error while reading the script shows its message and
traceback in the dialog. The dialog imports the file afresh each time, so
an edited file needs no restart.

## 12. Speed

Each batch a Python node takes the interpreter lock and calls your Python
code. Measured 2026-10-17 (GTX 1050 Ti, `bench_python_node_cost`):

| Softsign | forward + backward of a [32, 32] batch | Time per epoch, tiny causal-LM example |
|---|---|---|
| C++ plugin | 221 us | 0.0998 s |
| Python node | 474 us | 0.1015 s |

About 0.25 ms more per batch: invisible in a model whose layers do real
work, noticeable with very small batches. Python nodes do not run in
parallel with each other or with a script running in the Engine (one
interpreter lock), and converting to NumPy inside a node copies to the CPU
and back.

## 13. Examples: built-in layers and losses as Python nodes

`examples/python_nodes/builtin_layers/` (in the SDK:
`examples/python-nodes/builtin-layers/`) holds Dense, Layer Norm, ReLU, Leaky
ReLU, ELU, Sigmoid, Tanh, Softplus, GELU and SiLU, and the losses Cross
Entropy, MSE, L1 and Huber, written as Python nodes in `builtin_layers.py`,
bundled as `nodes/builtin-layers.cyxnode`, with an example graph that feeds
them from a Data Input (`blobs_python_layers.cyxgraph`, four classes from
`blobs.csv`; every layer and the loss are Python).

| Shows | Where |
|---|---|
| A layer with learnable weights and a setting that changes the output shape | `Dense` (`units`, `build`, `output_shape`, a matrix product through `batch_matmul`) |
| A normalisation with learnable gain and shift and a hand-written backward | `LayerNorm` |
| Tier 1 activations from f and df | `ReLU`, `Sigmoid`, `Tanh`, `Softplus`, `GELU`, `SiLU` |
| Activations with a setting | `LeakyReLU`, `ELU` |
| A classification loss (its own softmax over raw outputs, one-hot targets) | `CrossEntropy` |
| Regression losses, one with a setting | `MSE`, `L1`, `Huber` (`delta`) |
| Helpers kept in the same file | `matmul`, `rows`, `relu_mask` (a float32 mask from a comparison), `softmax` |

`test_builtin_layers.py` checks each against the Engine's built-in module
or loss (forward and gradients agree to 1e-5) and runs the gradient check;
the Engine test `test_python_builtin_layers` trains the example graph and
every activation from a Data Input, and each Python loss gives the same
epoch losses as the built-in loss of the same name.

## 14. Saving and exporting a model with Python nodes

| Path | What happens |
|---|---|
| Export a CyxWiz package (`.cyxmodel`) | Works. The node's weights are saved, the graph is always included, and the manifest lists the bundle's node types with version and code hash |
| Import the package | The node is rebuilt from the installed, approved bundle and its weights loaded. Missing bundle: refused, "is not installed". Other code than at training: loads, with the warning "is not the one this graph was saved with" |
| ONNX export, generated code (PyTorch, TensorFlow, Keras, pycyxwiz) | Refused with the node's name: the bundle's `node.py` is not carried into them yet |
| Server Node deployment | Refused: the Server Node does not run extension nodes yet |

## 15. Not yet

| Coming | Stage |
|---|---|
| Nodes with several inputs or outputs (a loss has its fixed two) | Later |
| Losses whose targets are class indices or 0/1 values, or that take weights | Later |
| Losses from C++ plugins | Later |
| Installing a bundle's Python packages for it | Later |
| Autograd: `backward` is written by hand | Later |
