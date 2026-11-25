---
title: Optimizers and training loops
description: Train a pycyxwiz Sequential with an optimizer, in mini-batches, with validation, a learning rate schedule, frozen layers and the best weights kept.
type: how-to
area: python
audience: user
status: experimental
since: 1.0.8
owner: python
last_reviewed: 2026-11-25
related:
  - layers-models-and-losses.md
  - evaluation-and-metrics.md
  - reference/optimizers.md
  - reference/model.md
  - ../user-guide/training/index.md
---

Use this when you train a model from a script. pycyxwiz has no `fit`
function: you write the loop, and each step is the same four calls.

> Checked against pycyxwiz 0.2.0. The samples on this page run, in order,
> as one script. The data is made up in the first sample (a fixed seed), so
> nothing needs downloading.

## The four calls of a step

```python
import numpy as np
import pycyxwiz as cx

cx.initialize()

rng = np.random.default_rng(0)
X = rng.normal(size=(600, 2)).astype(np.float32)
y = (X[:, 0] * X[:, 1] > 0).astype(np.int64)        # class 1 in two opposite quarters
X_train, y_train, X_val, y_val = X[:500], y[:500], X[500:], y[500:]

model = cx.Sequential()
model.add_linear(2, 32)
model.add_relu()
model.add_linear(32, 2)
loss_fn = cx.CrossEntropyLoss()
optimizer = cx.Adam(learning_rate=0.01)

xb = cx.Tensor.from_numpy(X_train[:50])
yb = cx.Tensor.from_numpy(y_train[:50])
scores = model.forward(xb)                    # 1. forward
loss = loss_fn.forward(scores, yb)            # 2. the loss
model.backward(loss_fn.backward(scores, yb))  # 3. gradients, back through the model
model.update_parameters(optimizer)            # 4. the optimizer updates every parameter
print(loss.shape())                           # [1]
```

`update_parameters` applies the optimizer to the gradients the last
`backward` left in each module; you do not clear gradients between steps.

## Optimizers

| Class | Defaults |
| --- | --- |
| `SGD` | `learning_rate=0.01, momentum=0.0, weight_decay=0.0, nesterov=False` |
| `Adam` | `learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8` |
| `AdamW` | as Adam, `weight_decay=0.01` |
| `RMSprop`, `AdaGrad`, `NAdam`, `Adadelta`, `LAMB` | See [Optimizers](reference/optimizers.md) |

`cx.create_optimizer(cx.OptimizerType.AdamW, 0.001)` makes one by type.
`get_learning_rate()` and `set_learning_rate(lr)` read and change the rate.

`optimizer.step(parameters, gradients)` takes the two dicts of
`get_parameters()` and `get_gradients()`, but from Python it changes
nothing: the dicts hold copies, and the update is lost. Train through
`Sequential.update_parameters`. The same holds for `LRWarmup.step`; to warm
up the rate, change it yourself (below).

## A full loop: shuffle, batches, validation

```python
def accuracy(model, X, y):
    model.eval()
    predicted = model.forward(cx.Tensor.from_numpy(X)).to_numpy().argmax(axis=1)
    model.train()
    return float((predicted == y).mean())

batch_size = 50
history = []
model.train()
for epoch in range(30):
    order = rng.permutation(len(X_train))          # a new order every epoch
    total, batches = 0.0, 0
    for start in range(0, len(order), batch_size):
        rows = order[start:start + batch_size]
        xb = cx.Tensor.from_numpy(X_train[rows])
        yb = cx.Tensor.from_numpy(y_train[rows])
        scores = model.forward(xb)
        loss = loss_fn.forward(scores, yb)
        model.backward(loss_fn.backward(scores, yb))
        model.update_parameters(optimizer)
        total += loss.at(0)
        batches += 1
    history.append((total / batches, accuracy(model, X_val, y_val)))

print("last loss %.3f, validation accuracy %.2f" % history[-1])   # about 0.06 and 0.99
```

Keep the conversion to tensors per batch, as here, when the data is bigger
than you want on the device at once; for a small dataset, converting it
once and taking batches with `Tensor.index_select(0, rows.tolist())`
saves the copies.

## Change the learning rate during training

A warmup over the first steps, then a step decay:

```python
optimizer = cx.SGD(learning_rate=0.1, momentum=0.9)
base_rate = 0.1
for step in range(200):
    if step < 20:
        optimizer.set_learning_rate(base_rate * (step + 1) / 20)   # warmup
    elif step % 100 == 0:
        optimizer.set_learning_rate(optimizer.get_learning_rate() * 0.5)
    rows = rng.integers(0, len(X_train), size=batch_size)
    xb = cx.Tensor.from_numpy(X_train[rows])
    yb = cx.Tensor.from_numpy(y_train[rows])
    scores = model.forward(xb)
    model.backward(loss_fn.backward(scores, yb))
    model.update_parameters(optimizer)
print(round(optimizer.get_learning_rate(), 3))   # 0.05
```

To find a good rate first, `cx.model.lr_range_test(model, loss, optimizer,
x, y)` trains with a rising rate, puts the weights back and suggests a rate
(`analysis["steepest_rate"]`); `examples/python/model_tools.py` runs it.

## Keep the best weights

```python
best_accuracy, best_weights = -1.0, None
for epoch in range(5):
    for start in range(0, len(X_train), batch_size):
        xb = cx.Tensor.from_numpy(X_train[start:start + batch_size])
        yb = cx.Tensor.from_numpy(y_train[start:start + batch_size])
        scores = model.forward(xb)
        model.backward(loss_fn.backward(scores, yb))
        model.update_parameters(optimizer)
    now = accuracy(model, X_val, y_val)
    if now > best_accuracy:
        best_accuracy = now
        best_weights = {k: v.to_numpy() for k, v in model.get_parameters().items()}

model.set_parameters({k: cx.Tensor.from_numpy(v) for k, v in best_weights.items()})
print(accuracy(model, X_val, y_val) == best_accuracy)   # True
```

`get_parameters()` gives copies, so copy them to NumPy to keep them; stop
early by leaving the loop when validation has not improved for a few
epochs.

## Freeze layers

```python
model.freeze_layer(0)                          # by index; also freeze_up_to(i), freeze_except_last(n)
print(sorted(model.get_frozen_parameter_names()))   # ['layer0.bias', 'layer0.weight']
print(model.is_layer_trainable(0), sorted(model.get_trainable_parameters()))
# False ['layer2.bias', 'layer2.weight']
model.unfreeze_all()
```

`update_parameters` leaves the parameters of frozen layers as they are.

## Check gradients

`cx.gradcheck(make)` checks a module's `backward` against differences of
its `forward`, for the input and each parameter, the same check as the
Engine's **Check gradient**:

```python
result = cx.gradcheck(lambda shape: cx.LinearModule(shape[-1], 3), input_shapes=[[2, 4], [5, 4]])
print(result.passed, result.max_relative_error < result.tolerance)   # True True
```

`make` receives the input shape (batch included) and returns a fresh module.
Without `input_shapes` the check also tries a 3-D input `[3, 4, 5]`, which a
linear module refuses: `passed` is then `False` and `error` says "Input must
be 1D or 2D tensor". `cx.gradcheck_loss` does the same for a loss.

## Related

- [Evaluate a model](evaluation-and-metrics.md): the confusion matrix,
  ROC, calibration.
- [Save and load models](saving-and-loading.md): keep the trained weights.
- [Train a model in the Engine](../user-guide/training/index.md): the same
  optimizers and losses as nodes, with the training dashboard.
