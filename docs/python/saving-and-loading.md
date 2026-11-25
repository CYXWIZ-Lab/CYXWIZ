---
title: Save and load models in pycyxwiz
description: Save a pycyxwiz Sequential's weights to a file and load them into the same model, keep weights as NumPy arrays, and know how this differs from the Engine's model package.
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
  - ../concepts/model-formats.md
---

Use this when you want a trained model again later: in another script, in
another session, or on another machine. A `Sequential` saves its weights;
you rebuild the same model in code and load them into it.

> Checked against pycyxwiz 0.2.0. The samples on this page run, in order,
> as one script, and write their files in the folder the script runs in.

## Save and load

```python
import os
import numpy as np
import pycyxwiz as cx

cx.initialize()

def build():
    model = cx.Sequential()
    model.add_linear(4, 8)
    model.add_relu()
    model.add_linear(8, 2)
    return model

model = build()
model.set_name("sensor-classifier")
model.set_description("4 readings in, 2 classes out")
print(model.save("models/sensor"))            # True
print(os.path.exists("models/sensor.cyxmodel"))   # True: .cyxmodel is added, folders are made

again = build()
print(again.load("models/sensor"))            # True
print(again.get_name(), "|", again.get_description())   # sensor-classifier | 4 readings in, 2 classes out

x = cx.Tensor.random([3, 4], seed=1)
print(np.allclose(again.forward(x).to_numpy(), model.forward(x).to_numpy()))   # True
```

`save(path)` and `load(path)` add `.cyxmodel` when the path does not end in
it, and return `True` or `False`; the reason for a `False` is in the log
(**Console > Logs** in the Engine), for example "Failed to open file" for a
path that is not there:

```python
print(again.load("models/missing"))           # False
```

## What the file holds

The file holds the model's name and description, the name of each module,
and each module's weights with their shapes and types. It does not hold
the code that builds the model, the optimizer's state, or the training
history. So:

- **Build the same model before you load.** `load` checks only that the
  number of modules matches ("Module count mismatch" otherwise, and
  `False`). It does not check the layer sizes: weights from a 4 -> 8 -> 2
  model load into a 4 -> 16 -> 2 model, which then fails on its first
  `forward`. Keep the function that builds the model next to the file, as
  `build()` above.
- **Training goes on from the weights, not from the optimizer's state.** A
  new optimizer starts its moments (Adam's averages, momentum) from zero.

This file is not the model package the Engine writes with **File > Export
Model**: that package also holds the graph, the training configuration and
the assets, so it opens again on the canvas. See
[Model formats](../concepts/model-formats.md).

## Keep the weights as NumPy arrays

To hand the weights to other code, or to keep several sets in one file,
save the parameter dict with NumPy:

```python
weights = {name: t.to_numpy() for name, t in model.get_parameters().items()}
np.savez("sensor_weights.npz", **weights)

stored = np.load("sensor_weights.npz")
print(sorted(stored.files))   # ['layer0.bias', 'layer0.weight', 'layer2.bias', 'layer2.weight']

third = build()
third.set_parameters({name: cx.Tensor.from_numpy(stored[name]) for name in stored.files})
print(np.allclose(third.forward(x).to_numpy(), model.forward(x).to_numpy()))   # True
```

A linear layer's `weight` is `[out_features, in_features]` and its `bias`
`[out_features]`, the same layout as PyTorch's `nn.Linear`.

## Related

- [Optimizers and training loops](training-loops.md#keep-the-best-weights):
  keep the best weights while training.
- [Model formats](../concepts/model-formats.md): the Engine's package,
  checkpoints and weight files.
