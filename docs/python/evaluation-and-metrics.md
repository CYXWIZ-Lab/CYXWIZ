---
title: Evaluate a model in pycyxwiz
description: Measure a pycyxwiz model with accuracy, the confusion matrix, per-class precision and recall, ROC and AUC, calibration and explanations.
type: how-to
area: python
audience: user
status: experimental
since: 1.0.8
owner: python
last_reviewed: 2026-11-25
related:
  - training-loops.md
  - reference/stats.md
  - reference/model.md
  - ../user-guide/evaluation/index.md
  - examples.md
---

Use this when a model is trained and you want to know how good it is.
Predictions come out as tensors; the metrics are in `pycyxwiz.stats`
(confusion matrix, ROC) and `pycyxwiz.model` (calibration, explanations);
the calibration and explanation functions are the code the Engine's
**Results > Test** and **Tools > Model** use.

> Checked against pycyxwiz 0.2.0. The samples on this page run, in order,
> as one script; the first one trains a small model on the committed sample
> `examples/datasets/digits_8x8.csv` (see the [quickstart](quickstart.md)).

## Get predictions

```python
import numpy as np
import pycyxwiz as cx

cx.initialize()
data = np.loadtxt("examples/datasets/digits_8x8.csv", delimiter=",", skiprows=1, dtype=np.float32)
X, y = data[:, :64] / 16.0, data[:, 64].astype(np.int64)
order = np.random.default_rng(0).permutation(len(X))
X_train, y_train, X_test, y_test = X[order[:1500]], y[order[:1500]], X[order[1500:]], y[order[1500:]]

model = cx.Sequential()
model.add_linear(64, 64)
model.add_relu()
model.add_linear(64, 10)
loss_fn, optimizer = cx.CrossEntropyLoss(), cx.Adam(learning_rate=0.01)
for epoch in range(10):
    for s in range(0, 1500, 100):
        xb, yb = cx.Tensor.from_numpy(X_train[s:s + 100]), cx.Tensor.from_numpy(y_train[s:s + 100])
        scores = model.forward(xb)
        model.backward(loss_fn.backward(scores, yb))
        model.update_parameters(optimizer)

model.eval()                                   # dropout and batch norm in evaluation mode
scores = model.forward(cx.Tensor.from_numpy(X_test))
probabilities = cx.softmax(scores).to_numpy()  # one row per sample, each row sums to 1
predicted = probabilities.argmax(axis=1)
print(probabilities.shape, round(float(probabilities[0].sum()), 4))   # (297, 10) 1.0
```

A model ending in `add_linear` gives scores (logits); `cx.softmax` turns
them into probabilities.

## Accuracy and the confusion matrix

```python
print("accuracy", round(float((predicted == y_test).mean()), 3))   # about 0.97

cm = cx.stats.confusion_matrix(y_test.tolist(), predicted.tolist())
print(sorted(cm))                   # ['accuracy', 'f1', 'matrix', 'precision', 'recall']
print(cm["matrix"][0])              # row 0: how the true zeros were classified
worst = int(np.argmin(cm["recall"]))
print("lowest recall: digit", worst, round(cm["recall"][worst], 3))
```

`matrix[i][j]` counts samples of true class `i` predicted as `j`;
`precision`, `recall` and `f1` have one value per class, in class order.

## ROC and AUC for a yes/no question

```python
is_zero = (y_test == 0).astype(int).tolist()
roc = cx.stats.roc(is_zero, probabilities[:, 0].tolist())
print(sorted(roc), roc["auc"] > 0.95)   # ['auc', 'fpr', 'tpr'] True
```

`roc(labels, scores)` takes 0/1 labels and a score per sample (higher means
"yes") and returns the curve (`fpr`, `tpr`) and the area under it.

## Is the model's confidence right?

A calibrated model that says 0.8 is right 8 times in 10.
`reliability` bins the confidence and gives the expected calibration error;
`fit_temperature` finds one temperature that calibrates the probabilities
without changing any prediction:

```python
confidence = probabilities.max(axis=1).tolist()
correct = (predicted == y_test).astype(int).tolist()
before = cx.model.reliability(confidence, correct, bins=10)
print(sorted(before))               # ['bins', 'ece', 'rows']

fit = cx.model.fit_temperature(probabilities, y_test.tolist())
calibrated = np.array(cx.model.with_temperature(probabilities, fit["temperature"]))
print((calibrated.argmax(axis=1) == predicted).all())   # True: the same predictions
```

Fit the temperature on one part of the test rows (`rows=`) and measure it
on the other, as `examples/python/model_tools.py` does. For a yes/no score,
`fit_platt` / `apply_platt` and `fit_isotonic` / `apply_isotonic` are the
other two calibrations.

## Why did it say that?

```python
row = X_test[0]
saliency = cx.model.saliency(model, row)
ig = cx.model.integrated_gradients(model, row, steps=64)
print(len(saliency["absolute"]), saliency["target"] == predicted[0])   # 64 True
print(round(ig["attribution_sum"], 2), round(ig["score"] - ig["baseline_score"], 2))
```

`saliency` and `smooth_grad` give how much each input value moves the score
of the predicted class (`target=-1`) or of the class you name;
`integrated_gradients` splits the score's change from a baseline of zeros
among the inputs, so the attributions add up to about that change.

## Size and cost

```python
summary = cx.model.summary(model, [64])
print(summary["parameters"], summary["multiply_adds"])   # 4810 4736
for r in summary["rows"]:
    print(r["name"], r["output"], r["parameters"])
```

Multiply-adds are counted for one sample, as PyTorch's FLOP counter counts
them (halved).

## Related

- [pycyxwiz.stats](reference/stats.md) and
  [pycyxwiz.model](reference/model.md): every metric and its options.
- [Test a model in the Engine](../user-guide/evaluation/index.md): the same
  measures on the **Results** tab.
- [Examples](examples.md): `model_tools.py` runs the range test, an
  explanation and calibration end to end.
