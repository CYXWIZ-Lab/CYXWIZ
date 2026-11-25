---
title: Quickstart with pycyxwiz
description: Import pycyxwiz, make a tensor, train a small classifier on the committed handwritten-digits sample, predict and save it.
type: tutorial
area: python
audience: user
status: experimental
since: 1.0.8
owner: python
last_reviewed: 2026-11-25
related:
  - index.md
  - tensors-and-devices.md
  - training-loops.md
  - evaluation-and-metrics.md
  - saving-and-loading.md
---

What you will build: a small neural network that reads 8x8 images of
handwritten digits and says which digit each one is, trained from a Python
script in about a minute.

**You need:** the CyxWiz Engine with a project open (or a Python 3.12 that
can import pycyxwiz, see [Run in the Engine or standalone](engine-and-standalone.md)),
NumPy, and the sample `examples/datasets/digits_8x8.csv`. The sample is in
the source repository, and Engine packages from 1.0.8 ship it in their
`examples` folder:
1,797 digits from the UCI Optical Recognition of Handwritten Digits data
(CC BY 4.0), 64 pixel columns with grey levels 0 to 16, then the digit in
`class`.

> Checked against pycyxwiz 0.2.0. The samples on this page run, in order,
> as one script.

Type the parts in the Script Editor as one script, or one by one in the
Python console. The path in part 3 is relative to the folder the script
runs in: change it to where your copy of the sample is.

## 1. Import and start the backend

```python
import numpy as np
import pycyxwiz as cx

cx.initialize()                 # picks the GPU when there is one, else the CPU
print(cx.get_version())         # 0.2.0
```

`initialize()` is safe to call again; in the Engine the backend is already
running and it returns `True` at once.

## 2. Make a tensor

```python
t = cx.Tensor.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
print(t.shape(), t.get_data_type())       # [2, 2] DataType.Float32
print((t * 10.0 + 1.0).to_numpy().tolist())   # [[11.0, 21.0], [31.0, 41.0]]
print(float(t.mean().to_numpy()))         # 2.5
```

A `Tensor` lives on the compute device; `from_numpy` and `to_numpy` copy to
it and back. More in [Tensors and devices](tensors-and-devices.md).

## 3. Load the digits

```python
data = np.loadtxt("examples/datasets/digits_8x8.csv", delimiter=",", skiprows=1, dtype=np.float32)
X = data[:, :64] / 16.0                   # pixels scaled to 0..1
y = data[:, 64].astype(np.int64)          # the digit, 0..9

order = np.random.default_rng(0).permutation(len(X))
X, y = X[order], y[order]
X_train, y_train = X[:1500], y[:1500]
X_test, y_test = X[1500:], y[1500:]
print(X_train.shape, X_test.shape)        # (1500, 64) (297, 64)
```

## 4. Build the model

```python
model = cx.Sequential()
model.add_linear(64, 64)    # 64 pixels in, 64 hidden units
model.add_relu()
model.add_linear(64, 10)    # one score per digit

loss_fn = cx.CrossEntropyLoss()           # takes class indices as int64
optimizer = cx.Adam(learning_rate=0.01)
print(cx.model.summary(model, [64])["parameters"])   # 4810
```

## 5. Train

Each step runs the batch forward, measures the loss, runs the gradient
backward through the model and lets the optimizer update the weights.

```python
model.train()
for epoch in range(20):
    total = 0.0
    for start in range(0, len(X_train), 100):
        xb = cx.Tensor.from_numpy(X_train[start:start + 100])
        yb = cx.Tensor.from_numpy(y_train[start:start + 100])
        scores = model.forward(xb)
        loss = loss_fn.forward(scores, yb)
        model.backward(loss_fn.backward(scores, yb))
        model.update_parameters(optimizer)
        total += loss.at(0)
    if epoch % 5 == 4:
        print("epoch", epoch + 1, "loss", round(total / 15, 4))
```

The loss falls to about 0.14 by epoch 5 and about 0.04 by epoch 20. The
starting weights come from the library's random stream, so the numbers can
differ a little on another machine or device.

## 6. Test and predict

```python
model.eval()
scores = model.forward(cx.Tensor.from_numpy(X_test)).to_numpy()
predicted = scores.argmax(axis=1)
print("test accuracy", round(float((predicted == y_test).mean()), 3))   # about 0.98

first = cx.Tensor.from_numpy(X_test[:1])
print("first test digit:", int(model.forward(first).to_numpy().argmax()), "label:", int(y_test[0]))
```

## 7. Save the weights

```python
print(model.save("digits_mlp"))           # True: wrote digits_mlp.cyxmodel

again = cx.Sequential()
again.add_linear(64, 64)
again.add_relu()
again.add_linear(64, 10)
print(again.load("digits_mlp"))           # True
same = again.forward(cx.Tensor.from_numpy(X_test)).to_numpy().argmax(axis=1)
print(bool((same == predicted).all()))    # True
```

`save` adds `.cyxmodel` to the name. The file holds the weights of each
module; you load it into a model built with the same modules. See
[Save and load models](saving-and-loading.md).

## Clean up

Delete `digits_mlp.cyxmodel` from the folder the script ran in.

## Next steps

- [Optimizers and training loops](training-loops.md): shuffling, the
  learning rate, freezing layers.
- [Evaluate a model](evaluation-and-metrics.md): the confusion matrix and
  per-class scores.
- [The digits as a graph](../user-guide/graph-editor/index.md): the same
  sample trains in the Engine's graph editor from the starter graph
  `examples/cyxgraph/mnist_mlp.cyxgraph`.
