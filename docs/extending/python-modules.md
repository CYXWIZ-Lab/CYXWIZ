# Writing a module in Python

A Python class derived from `pycyxwiz.Module` is a real CyxWiz module: the
C++ model calls its `forward` and `backward`, and the optimizer updates the
parameters it declares. This is the first half of the Python route; turning
such a class into a node in the Engine's palette comes next (stage P4).

> Checked 2026-10-17 against pycyxwiz built from branch
> `tofix125-extension-nodes`. The examples below are the ones
> `cyxwiz-backend/python/tests/test_python_modules.py` runs.

## 1. A module without parameters: Softsign

```python
import pycyxwiz as cyx

class Softsign(cyx.Module):
    """x / (1 + |x|)"""

    def forward(self, x):
        self.x = x
        return x / (x.abs() + 1.0)

    def backward(self, grad_output):
        d = self.x.abs() + 1.0
        return grad_output / (d * d)
```

| Rule | Why |
|---|---|
| `forward(self, x)` returns a `Tensor` | The next module receives it |
| `backward(self, grad_output)` returns the gradient for `x`, with `x`'s shape | The previous module receives it |
| Keep in `self` what `backward` needs from `forward` | `backward` runs after `forward` on the same batch |
| Work with `Tensor` operations (`abs`, `sin`, `cos`, `exp`, `log`, `sqrt`, `pow`, `+ - * /`, `sum`, `mean`, ...) | They run on the device CyxWiz trains on; converting to NumPy copies to the CPU and back |

## 2. A module with a learnable parameter: Snake

```python
class Snake(cyx.Module):
    """x + sin^2(a x) / a with a learnable a"""

    def __init__(self, initial_a=1.0):
        super().__init__()
        self.add_parameter("a", cyx.Tensor.full([1], initial_a))

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

| Method | Use |
|---|---|
| `super().__init__()` | Call it first in your `__init__`; without it the object is not a module |
| `add_parameter(name, tensor)` | Declare a learnable tensor (usually in `__init__`) |
| `parameter(name)` | Its current value; read it in `forward`, because the optimizer replaces it after each step |
| `set_gradient(name, tensor)` | In `backward`: the loss gradient of the parameter, with the parameter's shape |
| `gradient(name)`, `parameter_names()` | Read back what you stored |

Parameters are added only to modules defined in Python; built-in modules such
as `LinearModule` refuse `add_parameter`.

You may instead override `get_parameters`, `set_parameters`,
`get_gradients` and `has_parameters` yourself; the helpers above do it for
you.

## 3. Train it

```python
model = cyx.Sequential()
model.add_linear(4, 16)
model.add(Snake(1.0))          # any module: yours or a built-in one
model.add_linear(16, 1)

loss_fn = cyx.MSELoss()
optimizer = cyx.Adam(0.01)
model.train()
for step in range(150):
    prediction = model.forward(xs)
    loss = loss_fn.forward(prediction, ys)
    model.backward(loss_fn.backward(prediction, ys))
    model.update_parameters(optimizer)
```

- `Sequential.add` keeps your object alive while the model holds it.
- The model names your parameters `layer<N>.<name>` (`layer1.a` above).
- `get_name()` is your class name unless you define `get_name`.

## 4. Check your gradient

Before training, compare your `backward` with a numerical gradient of your
`forward`. `gradcheck` is the same checker as **Check gradient** in the
Engine's Properties:

```python
result = cyx.gradcheck(lambda input_shape: Snake(1.0))
print(result)        # Passed: largest relative error 1.3e-03 in input (tolerance 2.0e-02)
assert result        # True when passed
```

| Item | Meaning |
|---|---|
| `make(input_shape)` | Returns a fresh module; `input_shape` includes the batch. Called once per shape |
| `input_shapes` | Default `[[2, 3], [3, 4, 5]]` |
| `result.passed`, `result.max_relative_error`, `result.tolerance` | The outcome (tolerance 2e-2 by default) |
| `result.worst` | Where the largest error is: `input` or `parameter <name>` |
| `result.error` | Set when the module could not be checked (an exception, a factory that returns no module) |

A wrong gradient is far above the tolerance: dividing Softsign's gradient by
`1 + |x|` instead of its square gives `Failed: largest relative error
5.0e-01 in input`. The check probes inputs away from exactly zero, where
functions such as `|x|` have a kink.

## 5. When something goes wrong

An exception in your `forward` or `backward` stops the model with a
`RuntimeError` that names your class and method and carries the Python
traceback:

```
Broken.forward raised ValueError: deliberate failure in forward

At:
  .../my_nodes.py(12): forward
```

A `forward` or `backward` that does not return a `Tensor` is refused with
"must return a Tensor".

## 6. Next

To describe the module as a node (identity, settings, shape rule), see
[python-nodes.md](python-nodes.md).

### Not yet

| Coming | Stage |
|---|---|
| Bundling a module into a node and using it in the Engine | P3, P4 |
