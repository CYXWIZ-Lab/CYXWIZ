"""pycyxwiz.model: the Model window and Results > Test > Calibration from Python.

A small network on made-up data (three classes of points in 4 dimensions, a
fixed seed, so nothing to download): its summary, a learning rate range
test, training at the suggested rate, an explanation of one test row, and
the calibration of its test probabilities.

    python model_tools.py <folder holding pycyxwiz> [<folder of its DLLs>]

Needs numpy. The numbers it prints are in python_doc/pycyxwiz.md (6.5).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cyxwiz-backend", "python", "tests"))
from pycyxwiz_test_env import import_pycyxwiz  # noqa: E402

cx = import_pycyxwiz()
cx.initialize()
mt = cx.model

# Three overlapping clouds of 200 points each; 400 rows to train, 200 to test.
rng = np.random.default_rng(7)
centres = np.array([[0, 0, 0, 0], [1.5, 1, 0, -1], [0, 1.5, 1.5, 0]], dtype=np.float32)
labels = np.repeat(np.arange(3), 200)
points = (centres[labels] + rng.normal(0, 0.9, (600, 4))).astype(np.float32)
order = rng.permutation(600)
x_train, y_train = points[order[:400]], labels[order[:400]].astype(np.int32)
x_test, y_test = points[order[400:]], labels[order[400:]]


def network():
    model = cx.Sequential()
    model.add_linear(4, 32)
    model.add_relu()
    model.add_linear(32, 3)
    return model


model = network()

# The summary: parameters and multiply-adds, counted as PyTorch counts them.
s = mt.summary(model, [4])
for row in s["rows"]:
    print(f"{row['name']:<16} {str(row['output']):<6} {row['parameters']:>4} parameters  {row['multiply_adds']:>4} multiply-adds")
print("total", s["parameters"], "parameters,", s["multiply_adds"], "multiply-adds a sample")

# A range test on a fresh optimizer; the model's weights come back after it.
test = mt.lr_range_test(model, cx.CrossEntropyLoss(), cx.Adam(learning_rate=1e-3), x_train, y_train,
                        start=1e-5, end=10.0, steps=100, batch_size=32)
a = test["analysis"]
print(f"range test: {len(test['points'])} steps; lowest loss {a['lowest_loss']:.3f} at {a['lowest_rate']:.3g}; "
      f"steepest fall at {a['steepest_rate']:.3g}")

# Train at the suggested rate.
optimizer = cx.Adam(learning_rate=a["steepest_rate"])
loss = cx.CrossEntropyLoss()
model.train()
for epoch in range(30):
    for at in range(0, 400, 32):
        xb = cx.Tensor.from_numpy(x_train[at:at + 32])
        yb = cx.Tensor.from_numpy(y_train[at:at + 32])
        prediction = model.forward(xb)
        loss.forward(prediction, yb)
        model.backward(loss.backward(prediction, yb))
        model.update_parameters(optimizer)
model.eval()
probabilities = cx.softmax(model.forward(cx.Tensor.from_numpy(x_test))).to_numpy()
predicted = probabilities.argmax(axis=1)
print(f"test accuracy {np.mean(predicted == y_test):.3f} on {len(y_test)} rows")

# Why the first wrong row was read as it was.
wrong = int(np.flatnonzero(predicted != y_test)[0])
sal = mt.saliency(model, x_test[wrong])
ig = mt.integrated_gradients(model, x_test[wrong], steps=64)
print(f"row {wrong}: a {y_test[wrong]} read as {predicted[wrong]}; saliency",
      [round(v, 3) for v in sal["absolute"]], "IG", [round(v, 3) for v in ig["attributions"]],
      f"(sum {ig['attribution_sum']:.3f}, score moved {ig['score'] - ig['baseline_score']:.3f})")

# Calibration: is 80 % sure right 80 % of the time? A temperature fitted on
# the even test rows, checked on the odd ones.
right = (predicted == y_test).astype(int).tolist()
confidence = probabilities.max(axis=1).tolist()
odd = list(range(1, len(y_test), 2))
before = mt.reliability([confidence[i] for i in odd], [right[i] for i in odd])
fit = mt.fit_temperature(probabilities, y_test.tolist(), rows=list(range(0, len(y_test), 2)))
scaled = np.array(mt.with_temperature(probabilities, fit["temperature"]))
after = mt.reliability(scaled.max(axis=1)[odd].tolist(), [right[i] for i in odd])
print(f"calibration error {before['ece']:.4f}; temperature {fit['temperature']:.3f} brings it to {after['ece']:.4f}")
