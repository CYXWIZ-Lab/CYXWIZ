# Describing a Python module as a node

`pycyxwiz.nodes` adds what the Engine needs to show a module as a node: an
identity, a category, settings with types and limits, and the shape rule.
Write the module as in [python-modules.md](python-modules.md), then
decorate it.

> Checked 2026-10-17 against pycyxwiz built from branch
> `tofix125-extension-nodes`. The examples are the ones
> `cyxwiz-backend/python/tests/test_python_nodes.py` runs. Bundling a node
> and using it in the Engine come next (stages P3 step 3.5 and P4).

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

## 5. Not yet

| Coming | Stage |
|---|---|
| `nodes.bundle()`: write the node into a folder the Engine reads | P3 step 3.5 |
| The node in the Engine's palette, Properties and training | P4 |
| Nodes with several inputs or outputs | Later |
