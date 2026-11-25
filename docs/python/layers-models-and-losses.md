---
title: Layers, models and losses
description: Build a Sequential model from pycyxwiz modules, add your own module, use the layer classes, and pick and write losses.
type: how-to
area: python
audience: user
status: experimental
since: 1.0.8
owner: python
last_reviewed: 2026-11-25
related:
  - training-loops.md
  - reference/modules-and-models.md
  - reference/layers.md
  - reference/losses.md
  - ../sdk/guides/python-modules.md
---

Use this when you build a model in Python. The model you train is a
`Sequential`: a list of **modules** run in order. Next to the modules,
pycyxwiz has **layer** classes you call one at a time, and **losses** that
measure how far the model's output is from the target.

> Checked against pycyxwiz 0.2.0. The samples on this page run, in order,
> as one script.

## Build a Sequential

```python
import numpy as np
import pycyxwiz as cx

cx.initialize()

model = cx.Sequential()
model.add_linear(4, 16)          # Linear(4 -> 16): weight [16, 4] and bias [16]
model.add_relu()
model.add_dropout(0.2)
model.add_linear(16, 3)
print(len(model))                # 4
print(sorted(model.get_parameters()))
# ['layer0.bias', 'layer0.weight', 'layer3.bias', 'layer3.weight']

x = cx.Tensor.random([8, 4], seed=1)     # a batch of 8 rows of 4 features
print(model.forward(x).shape())          # [8, 3]
```

Dimension 0 is the batch: a model takes `[batch, features]` and keeps the
batch size. Parameters are named `layer<index>.<name>` after the module's
place in the model.

The `add_*` methods add a built-in module: `add_linear`, `add_relu`,
`add_relu6`, `add_leaky_relu`, `add_elu`, `add_gelu`, `add_swish`,
`add_mish`, `add_sigmoid`, `add_tanh`, `add_softmax`, `add_hardswish`,
`add_hardsigmoid`, `add_dropout` and `add_flatten`. `add(module)` appends a
module object:

```python
model2 = cx.Sequential()
model2.add(cx.LinearModule(4, 8))
model2.add(cx.GELUModule())
model2.add(cx.LinearModule(8, 2))
print(model2.forward(x).shape())         # [8, 2]
```

The module classes are `LinearModule`, the activations (`ReLUModule`,
`GELUModule`, `TanhModule`, ...), `DropoutModule`, `FlattenModule`,
`AttentionModule`, `PositionalEncoding`, `LearnedPositionalEmbedding` and
`SqueezeExcitation`; their settings are on
[Modules and Sequential models](reference/modules-and-models.md).

Training and evaluation behave differently for dropout: call `model.train()`
before training and `model.eval()` before you measure or predict.

```python
model.eval()
print(np.allclose(model.forward(x).to_numpy(), model.forward(x).to_numpy()))   # True
model.train()
```

`model.summary()` writes the module list to the log (the Engine's
**Console > Logs**, or the terminal); `cx.model.summary(model, [4])`
returns the shapes, parameter counts and multiply-adds as a dict.

## Write your own module

A class derived from `cx.Module` with `forward` and `backward` is a module
like any other: it goes into a `Sequential`, and the optimizer trains the
parameters it declares. There is no automatic differentiation: `backward`
returns the gradient for the input and stores the gradient of each
parameter.

```python
class Scale(cx.Module):
    """y = s * x with one learnable number s."""

    def __init__(self):
        super().__init__()
        self.add_parameter("s", cx.Tensor.full([1], 2.0))

    def forward(self, x):
        self.x = x
        return x * self.parameter("s").at(0)

    def backward(self, grad_output):
        ds = float((grad_output * self.x).sum().to_numpy())
        self.set_gradient("s", cx.Tensor.full([1], ds))
        return grad_output * self.parameter("s").at(0)

model3 = cx.Sequential()
model3.add_linear(4, 2)
model3.add(Scale())
print(sorted(model3.get_parameters()))   # ['layer0.bias', 'layer0.weight', 'layer1.s']
print(cx.gradcheck(lambda shape: Scale()).passed)   # True
```

`gradcheck` compares your `backward` with differences of `forward`, as the
Engine's **Check gradient** does. The full rules, state tensors and more
examples are in [Write a module in Python](../sdk/guides/python-modules.md);
[Python nodes](../sdk/guides/python-nodes.md) turns such a class into a node.

## Layer classes

The layer classes (`Dense`, `Conv2D`, `MaxPool2D`, `BatchNorm2D`, `LSTM`,
`GRU`, `MultiHeadAttention`, `TransformerEncoderLayer` and the others on
[Layers](reference/layers.md)) each have `forward`, `backward`,
`get_parameters` and `set_parameters`. Use them to compute one step:

```python
dense = cx.Dense(4, 3)
out = dense.forward(x)
print(out.shape(), {k: v.shape() for k, v in dense.get_parameters().items()})
# [8, 3] {'bias': [3], 'weight': [3, 4]}
grad_in = dense.backward(cx.Tensor.ones([8, 3]))
print(grad_in.shape())                   # [8, 4]
```

Know these limits before you build on them:

- A layer class is not a module: `Sequential.add(cx.Conv2D(...))` raises
  "Sequential.add expects a pycyxwiz.Module". Train a model as a
  `Sequential` of modules.
- An optimizer's `step(parameters, gradients)` does not change a layer's
  weights from Python, so a layer class cannot be trained on its own; see
  [Optimizers and training loops](training-loops.md).
- `Conv2D`, `MaxPool2D` and the other 2-D layers take images as
  `[height, width, channels, batch]`, not `[batch, channels, height, width]`:

```python
images = cx.Tensor.random([16, 16, 3, 2], seed=2)   # 2 RGB images of 16 x 16
conv = cx.Conv2D(3, 8, 3, stride=1, padding=1)
print(conv.forward(images).shape())                  # [16, 16, 8, 2]
print(cx.MaxPool2D(2, stride=2).forward(conv.forward(images)).shape())   # [8, 8, 8, 2]
```

Convolutional and recurrent models train in the Engine's graph editor,
where each of these layers is a node.

## Names that are not classes

Some short names at the top of pycyxwiz are values of an enumeration, not
classes: `cx.ReLU`, `cx.Sigmoid`, `cx.Tanh`, `cx.Linear`, `cx.Dropout`,
`cx.Flatten` and others are `ModuleType` or `ActivationType` values, and
`cx.BatchNorm` is not `cx.BatchNorm2D`. Use the classes ending in `Module`
(`cx.ReLUModule()`), the `add_*` methods, or the functions (`cx.relu(x)`).

```python
print(cx.ReLU)                           # ModuleType.ReLU
print(cx.relu(cx.Tensor.full([1], -1.0)).at(0))   # 0.0
```

## Losses

| Loss | Predictions | Targets |
| --- | --- | --- |
| `MSELoss()` | Any shape | The same shape |
| `CrossEntropyLoss(reduction, ignore_index, class_weights, label_smoothing)` | `[batch, classes]` scores (logits) | Class indices as `int32` or `int64` `[batch]`, or one-hot `float32` `[batch, classes]` |
| `BCEWithLogitsLoss()` | Scores, before a sigmoid | 0 or 1, the same shape |
| `FocalLoss`, `SoftDiceLoss`, `JaccardLoss`, `TverskyLoss` | Scores | One-hot or masks |
| `TripletLoss`, `ContrastiveLoss` | Embeddings | See [Losses](reference/losses.md) |

```python
scores = cx.Tensor.random([4, 3], seed=3)
classes = cx.Tensor.from_numpy(np.array([0, 2, 1, 2], dtype=np.int64))
one_hot = cx.Tensor.from_numpy(np.eye(3, dtype=np.float32)[[0, 2, 1, 2]])
ce = cx.CrossEntropyLoss()
print(round(ce.forward(scores, classes).at(0), 4) == round(ce.forward(scores, one_hot).at(0), 4))   # True
print(ce.backward(scores, classes).shape())   # [4, 3]: dLoss / dscores
```

`forward` returns the loss as a one-element tensor (read it with `.at(0)`);
`backward` returns the gradient you pass to `model.backward`. Class indices
stored as `float32` raise "CrossEntropy class-index targets must be Int32 or
Int64": convert the labels with `astype(np.int64)`.

To write a loss, derive from `cx.Loss` and define `forward(self,
predictions, targets)` and `backward(self, predictions, targets)`:

```python
class MeanAbsoluteError(cx.Loss):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return (predictions - targets).abs().mean().reshape([1])

    def backward(self, predictions, targets):
        n = predictions.num_elements()
        return (predictions - targets).sign() * (1.0 / n)

mae = MeanAbsoluteError()
p = cx.Tensor.full([2, 2], 3.0)
q = cx.Tensor.full([2, 2], 1.0)
print(mae.forward(p, q).at(0), mae.backward(p, q).to_numpy().tolist())   # 2.0 [[0.25, 0.25], [0.25, 0.25]]
```

`cx.gradcheck_loss(lambda: MeanAbsoluteError(), targets="values")` checks
such a backward.

## Related

- [Optimizers and training loops](training-loops.md): train the model.
- [Modules and Sequential models](reference/modules-and-models.md),
  [Layers](reference/layers.md), [Losses](reference/losses.md): every class.
