# Writing a module in Python

A Python class derived from `pycyxwiz.Module` is a real CyxWiz module: the
C++ model calls its `forward` and `backward`, and the optimizer updates the
parameters it declares. This page is the first half of the Python route.
The second half, [python-nodes.md](python-nodes.md), turns such a class into
a node in the Engine's palette, bundles it and uses it in a graph.

> Checked against plugin API 1.3.0 and pycyxwiz 0.2.0.
> The Python samples on this page run, in order, as one script
> (`scripts/docs/check_extension_docs.py` runs them). Every pycyxwiz name is
> in [python-api-reference.md](python-api-reference.md); the tensor
> operations are in [tensor-operations.md](tensor-operations.md).

## 1. Get pycyxwiz

**Inside the Engine** pycyxwiz is already there: in the Script Editor and in
the Console, `import pycyxwiz` works. A script run from the Script Editor
runs in the project folder, with the project folder on `sys.path`, and as
its saved file (`__file__` is its path).

**Outside the Engine** (a terminal, your own editor, a test) you need:

| What | Where |
|---|---|
| Python 3.12 | pycyxwiz is built for Python 3.12 only. An installed Engine carries one: `<base>\python\python.exe` on Windows, `<base>/python/bin/python3.12` on Linux and macOS |
| The folder with pycyxwiz | Installed Engine: `<base>` itself; it holds `pycyxwiz.pyd` and the Engine's DLLs. Source build: `build\lib\Release` holds `pycyxwiz.cp312-win_amd64.pyd`; its DLLs are in `build\bin\Release` |
| ArrayFire | Installed Engine: `<base>\arrayfire\bin`. Source build: `C:\Program Files\ArrayFire\v3\lib` |
| CUDA (a source build with CUDA only) | `%CUDA_PATH%\bin` |

`<base>` is the one folder in `runtime\base` of the installation, for example
`%LOCALAPPDATA%\Programs\CyxWiz\runtime\base\cyxwiz-base-0.2.0-6-win64-x86_64`
([plugin-quickstart.md](plugin-quickstart.md) lists the other install
locations).

On Windows, Python does not search `PATH` for the DLLs a module needs. Name
the folders before the import:

<!-- not run: the paths depend on your installation -->
```python
import os
import sys

BASE = r"C:\Users\you\AppData\Local\Programs\CyxWiz\runtime\base\cyxwiz-base-0.2.0-6-win64-x86_64"
folders = [BASE, os.path.join(BASE, "arrayfire", "bin")]
for folder in folders:
    os.add_dll_directory(folder)
# ArrayFire loads its compute backend later, through PATH.
os.environ["PATH"] = os.pathsep.join(folders) + os.pathsep + os.environ["PATH"]
sys.path.insert(0, BASE)

import pycyxwiz
print(pycyxwiz.get_version())   # 0.2.0
```

For a source build, put `build\lib\Release` on `sys.path` and add
`build\bin\Release` (and `%CUDA_PATH%\bin` when the build uses CUDA) to the
DLL folders; `cyxwiz-backend/python/tests/pycyxwiz_test_env.py` does this
for the tests. "DLL load failed while importing pycyxwiz" means a DLL folder
is missing; "No module named 'pycyxwiz'" means the `sys.path` entry is.

On Linux and macOS the module is `pycyxwiz.cpython-312-*.so` in `<base>`:
put `<base>` on `sys.path`. pycyxwiz finds the base's libraries itself and
points ArrayFire at `<base>/arrayfire` (it sets `AF_PATH` when you have not).
CyxWiz 1.0.7 on macOS does not: the import works but the first tensor fails
with "ArrayFire couldn't locate any backends". There, start Python with
`AF_PATH` set:

<!-- not run: the paths depend on your installation -->
```bash
base=~/Applications/CyxWiz/runtime/base/cyxwiz-base-0.2.0-6-macos-x86_64
AF_PATH="$base/arrayfire" "$base/python/bin/python3.12" my_script.py
```

## 2. A module without parameters: Softsign

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

y = Softsign().forward(cyx.Tensor.full([2, 3], 3.0))
print(y.shape(), y.at(0, 0))   # [2, 3] 0.75
```

| Rule | Why |
|---|---|
| `forward(self, x)` returns a `Tensor` | The next module receives it. Anything else stops the model with "Softsign.forward must return a Tensor, not ..." |
| `backward(self, grad_output)` returns the gradient for `x`, with `x`'s shape | The previous module receives it |
| Keep in `self` what `backward` needs from `forward` | `backward` runs after `forward`, on the same batch |
| Write `backward` yourself | There is no autograd: nothing is derived from `forward` for you |

## 3. Tensors in a module

`x` is a whole batch, not one sample. **Dimension 0 is the batch**: 32 rows
of 4 columns arrive as `[32, 4]`. Work on all rows at once and keep
dimension 0; in the Engine a node that returns another batch size stops
training with "changed the batch size from 32 to ...".

Tensors are `float32` unless you make them otherwise. What you can do with
them is listed in [tensor-operations.md](tensor-operations.md): arithmetic
(`+ - * / **` with tensors or numbers), elementwise functions (`abs`, `exp`,
`log`, `sqrt`, `sin`, `cos`, `sign`, `pow`, `clip`), reductions (`sum`,
`mean`, `max`, `min`, `var`, `std`, with `dim` and `keepdim`), shapes
(`reshape`, `transpose`, `permute`, `unsqueeze`, `squeeze`, `broadcast_to`,
`slice`, `cat`, `stack`) and `batch_matmul` for matrix products. They run on
the device CyxWiz trains on. `to_numpy()` and `from_numpy()` copy to the CPU
and back, and need NumPy in the Python you run.

**Comparisons give `uint8`**, not `float32`: `x > 0.0` is 1 where true and 0
elsewhere. A `uint8` tensor times a number or a `float32` tensor gives
`float32`, but `uint8` with `uint8` stays `uint8`: `-mask` and
`mask + mask` wrap around, and `mask.sum()` overflows past 255. Turn a mask
into `float32` before anything else:

```python
x = cyx.Tensor.random([2, 3]) * 2.0 - 1.0   # values in [-1, 1)
mask = x > 0.0
print(mask.get_data_type())                 # DataType.UInt8
positive = mask * 1.0                       # float32 from here on
print(positive.get_data_type())             # DataType.Float32
relu = x * positive
```

**Module functions.** `pycyxwiz.tanh`, `relu`, `sigmoid`, `softmax`,
`gelu`, `elu`, `leaky_relu`, `mish`, `silu`, `swish` and `dropout` take a
tensor and give a tensor. They compute the forward value only: use them in
`forward`, and still write the derivative in `backward`.
`pycyxwiz.dropout` draws a new random mask on each call and does not return
it, so no `backward` can repeat it; write dropout as in section 5 instead.

```python
class Tanh(cyx.Module):
    def forward(self, x):
        self.t = cyx.tanh(x)
        return self.t

    def backward(self, grad_output):
        return grad_output * (1.0 - self.t * self.t)   # tanh'(x) = 1 - tanh(x)^2
```

## 4. A module with a learnable parameter: Snake

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
| `super().__init__()` | Call it first in your `__init__`. Without it, making the module fails with "pycyxwiz.Module.__init__() must be called when overriding __init__" (for a loss, `pycyxwiz.Loss.__init__()`) |
| `add_parameter(name, tensor)` | Declare a learnable tensor (in `__init__`, or in a node's `build`). Each name once |
| `parameter(name)` | Its current value. Read it in `forward`: the optimizer replaces it after each step |
| `set_gradient(name, tensor)` | In `backward`: the loss gradient of the parameter, with the parameter's shape (another shape is refused) |
| `gradient(name)`, `parameter_names()` | Read back what you stored |

A parameter's name is the key its value is saved under; read
[python-nodes.md](python-nodes.md) section 8 before renaming one.
Parameters are added only to modules defined in Python; built-in modules such
as `LinearModule` refuse `add_parameter`. You may instead override
`get_parameters`, `set_parameters`, `get_gradients` and `has_parameters`
yourself; the helpers above do it for you.

## 5. Training and evaluation mode

The model tells each module whether it is training: `Sequential.train()`
and `Sequential.eval()` in Python; the Engine switches modes itself. Read
the mode with `self.is_training()`; a new module starts in training mode. To
react when it changes, define `set_training(self, training)` and call
`super().set_training(training)` in it.

A dropout written this way keeps its mask for `backward`:

```python
class MyDropout(cyx.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.is_training() or self.p == 0.0:
            self.scale = None
            return x
        keep = cyx.Tensor.random(x.shape()) >= self.p    # uint8
        self.scale = keep * (1.0 / (1.0 - self.p))       # float32: 0 or 1/(1-p)
        return x * self.scale

    def backward(self, grad_output):
        return grad_output if self.scale is None else grad_output * self.scale

d = MyDropout(0.5)
print(d.is_training())   # True
d.set_training(False)
print(d.is_training())   # False
```

Random draws (starting weights, masks) come from the run's random stream:
inside a training run the Engine seeds it with the run's model seed, so the
same seed trains the same way. Outside a run, or for values of your own that
must repeat, pass a seed: `cyx.Tensor.random([2, 3], seed=7)` gives the same
values every time.

```python
check_a = cyx.Tensor.random([4], seed=7)
check_b = cyx.Tensor.random([4], seed=7)
print([check_a.at(i) == check_b.at(i) for i in range(4)])   # [True, True, True, True]
```

## 6. Train it

```python
xs = cyx.Tensor.random([64, 4]) * 2.0 - 1.0     # 64 samples, 4 features
ys = xs.sum(dim=1, keepdim=True)                # [64, 1]: what to predict

model = cyx.Sequential()
model.add_linear(4, 16)
model.add(Snake(1.0))          # any module: yours or a built-in one
model.add(MyDropout(0.1))
model.add_linear(16, 1)

loss_fn = cyx.MSELoss()
optimizer = cyx.Adam(0.01)
model.train()
for step in range(30):
    prediction = model.forward(xs)
    loss = loss_fn.forward(prediction, ys)
    model.backward(loss_fn.backward(prediction, ys))
    model.update_parameters(optimizer)
    if step in (0, 29):
        print(step, loss.at(0))   # the loss falls
```

- `Sequential.add` keeps your object alive while the model holds it.
- The model names your parameters `layer<N>.<name>` (`layer1.a` above).
- `get_name()` is your class name unless you define `get_name`.
- A loss is written the same way: subclass `pycyxwiz.Loss` with
  `forward(predictions, targets)` and `backward(predictions, targets)`; see
  [python-nodes.md](python-nodes.md) section 3.

## 7. Check your gradient

Before training, compare your `backward` with a numerical gradient of your
`forward`. `gradcheck` is the same checker as **Check gradient** in the
Engine's Properties:

```python
result = cyx.gradcheck(lambda input_shape: Snake(1.0))
print(result)        # Passed: largest relative error ... in input (tolerance 2.0e-02)
assert result        # True when passed
```

`gradcheck(make, input_shapes=[], seed=125, tolerance=0.02, step=0.01,
input_range=2.0)`:

| Argument or field | Meaning |
|---|---|
| `make(input_shape)` | Returns a fresh module. `input_shape` includes the batch. Called once per shape |
| `input_shapes` | The input shapes to try. Empty means `[[2, 3], [3, 4, 5]]` |
| `seed` | Seed of the random inputs and probes, so a check repeats exactly |
| `tolerance` | The largest relative error that passes |
| `step` | The step of the central differences |
| `input_range` | How far from zero the random inputs go. Inputs are kept away from exactly zero, where functions such as `abs(x)` have a kink |
| `result.passed`, `result.max_relative_error`, `result.tolerance` | The outcome; `bool(result)` is `result.passed` |
| `result.worst` | Where the largest error is: `input` or `parameter <name>` |
| `result.error` | Set when the module could not be checked (an exception, a factory that returns no module) |

A wrong gradient is far above the tolerance. Dividing Softsign's gradient by
`1 + |x|` instead of its square:

```python
class WrongSoftsign(Softsign):
    def backward(self, grad_output):
        return grad_output / (self.x.abs() + 1.0)

bad = cyx.gradcheck(lambda input_shape: WrongSoftsign())
print(bad)           # Failed: largest relative error 5.0e-01 in input ...
assert not bad.passed
```

`MyDropout` draws new noise on every forward, so in training mode it fails
the check; check it with `p = 0`:

```python
assert cyx.gradcheck(lambda input_shape: MyDropout(0.0))
```

## 8. When something goes wrong

An exception in your `forward` or `backward` stops the model with a
`RuntimeError` that names your class and method and carries the Python
traceback:

```python
class Broken(cyx.Module):
    def forward(self, x):
        raise ValueError("deliberate failure in forward")

    def backward(self, grad_output):
        return grad_output

failing = cyx.Sequential()
failing.add(Broken())
try:
    failing.forward(cyx.Tensor.ones([2, 3]))
except RuntimeError as error:
    print(error)   # Broken.forward raised ValueError: deliberate failure in forward ...
```

The message ends with the traceback: `At:` and the file and line of each
frame. In the Engine the same text is in the training error;
[python-nodes.md](python-nodes.md) section 10 says where to find it and how
to reproduce a failure outside training.

## 9. Next

To describe the module as a node (identity, settings, shape rule), bundle
it and use it in the Engine, see [python-nodes.md](python-nodes.md).
