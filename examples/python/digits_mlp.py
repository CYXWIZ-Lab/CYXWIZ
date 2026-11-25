"""Train a small digit classifier with pycyxwiz, from the Script Editor.

Reads examples/datasets/digits_8x8.csv (1,797 handwritten digits, 8x8 grey
levels 0..16, column `class` 0-9), the data of the Multiclass image
classification starter graph, and trains the same network in code:
64 -> 128 -> 64 -> 10 with ReLU, cross entropy and Adam. It prints the loss
of each epoch and the accuracy on the 10 % of rows it kept back.

In the Engine: Script > Open Script..., pick this file, then Run (F5).
From a terminal: python examples/python/digits_mlp.py, with pycyxwiz
importable. Needs numpy.
"""
from pathlib import Path

import numpy as np
import pycyxwiz as cyx

DATA = Path(__file__).resolve().parents[1] / "datasets" / "digits_8x8.csv"

# 64 pixel columns, then the class. Dividing by 16 is the starter's Normalize.
table = np.loadtxt(DATA, delimiter=",", skiprows=1, dtype=np.float32)
pixels = table[:, :64] / 16.0
labels = table[:, 64].astype(np.int32)

# Keep 10 % of the rows back for the test, with a fixed seed.
rng = np.random.default_rng(42)
order = rng.permutation(len(labels))
test_count = len(labels) // 10
x_test, y_test = pixels[order[:test_count]], labels[order[:test_count]]
x_train, y_train = pixels[order[test_count:]], labels[order[test_count:]]

model = cyx.Sequential()
model.add_linear(64, 128)
model.add_relu()
model.add_linear(128, 64)
model.add_relu()
model.add_linear(64, 10)

loss = cyx.CrossEntropyLoss()
optimizer = cyx.Adam(learning_rate=0.001)

batch_size = 128
model.train()
for epoch in range(1, 11):
    shuffled = rng.permutation(len(y_train))
    total = 0.0
    for start in range(0, len(shuffled), batch_size):
        rows = shuffled[start:start + batch_size]
        xb = cyx.Tensor.from_numpy(x_train[rows])
        yb = cyx.Tensor.from_numpy(y_train[rows])
        prediction = model.forward(xb)
        total += float(loss.forward(prediction, yb).to_numpy().mean()) * len(rows)
        model.backward(loss.backward(prediction, yb))
        model.update_parameters(optimizer)
    print(f"epoch {epoch:2d}: loss {total / len(y_train):.4f}")

model.eval()
scores = cyx.softmax(model.forward(cyx.Tensor.from_numpy(x_test))).to_numpy()
predicted = scores.argmax(axis=1)
print(f"test accuracy {np.mean(predicted == y_test):.3f} on {len(y_test)} rows")
