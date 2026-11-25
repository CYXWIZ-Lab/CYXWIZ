---
title: Train a model from the Script Editor
description: Write the digits network of lesson 2 in Python with pycyxwiz, and run it in the Engine's Script Editor.
type: tutorial
area: scripting
audience: user
status: experimental
since: 1.0.8
owner: docs
last_reviewed: 2026-11-25
related:
  - index.md
  - first-model-from-a-starter.md
  - ../user-guide/scripting/script-editor.md
  - ../python/reference/index.md
---

Everything the graph of lesson 2 does, you can also write in Python. The
Engine ships `pycyxwiz`, its compute library for Python, and runs scripts
in its **Script Editor**. In this lesson you run a short script that trains
the same digits network as the **Multiclass image classification** starter,
and read how it maps to the nodes.

**You need:** lesson 2 done, a Python the Engine can use (the chip on the
start page reads ready; see [First launch](../install/first-launch.md)),
a project open, the script `examples/python/digits_mlp.py` and its data
`examples/datasets/digits_8x8.csv`. About 15 minutes.

## 1. Open and run the script

1. Choose **Script > Open Script...** (**Ctrl+O**) and
   pick `examples/python/digits_mlp.py` in the Engine's folder. It opens in
   a Script Editor tab.
2. Select **Run** at the right of the tab strip, or press **F5** with the
   editor focused.
3. The status bar reads **Running** with the script's name. The output goes
   to the Console's **Python REPL** session:

   ```text
   epoch  1: loss 2.1850
   ...
   epoch 10: loss 0.1978
   test accuracy 0.966 on 179 rows
   ```

   Your numbers can differ a little; the accuracy should be above 0.9.

To stop a script, select **Stop** (**Shift+F5**).

## 2. Read the script, node by node

The script reads the data with NumPy, the way the **Data Input**,
**Normalize** and **Data Split** nodes do:

```python
DATA = Path(__file__).resolve().parents[1] / "datasets" / "digits_8x8.csv"

table = np.loadtxt(DATA, delimiter=",", skiprows=1, dtype=np.float32)
pixels = table[:, :64] / 16.0
labels = table[:, 64].astype(np.int32)
```

A saved script runs as its file, so `__file__` finds the data next to it.
The script keeps 10 % of the rows back for the test, with a fixed seed.

The model is the chain of **Dense** and **ReLU** nodes, and the loss and
optimizer nodes become two objects:

```python
model = cyx.Sequential()
model.add_linear(64, 128)
model.add_relu()
model.add_linear(128, 64)
model.add_relu()
model.add_linear(64, 10)

loss = cyx.CrossEntropyLoss()
optimizer = cyx.Adam(learning_rate=0.001)
```

What **Train** does for each batch of the **Data Loader** is four calls:

```python
prediction = model.forward(xb)
total += float(loss.forward(prediction, yb).to_numpy().mean()) * len(rows)
model.backward(loss.backward(prediction, yb))
model.update_parameters(optimizer)
```

`cyx.Tensor.from_numpy` turns a NumPy array into a pycyxwiz tensor, and
`.to_numpy()` turns it back. The test at the end runs `model.forward` on
the held-back rows and takes the class with the highest score.

## 3. Change it

1. Change `range(1, 11)` to `range(1, 21)` for 20 epochs.
2. Press **Ctrl+S**, then **F5**.

To run a part of a script, select its lines and press **F9**, or put `%%`
lines around a section and press **Ctrl+Enter**. Both run in the same
namespace, so `model` stays defined between runs.

## 4. Try a line in the console

The Console's **Python REPL** session keeps the script's names after a
run. Type at its prompt:

```python
model.eval()
cyx.softmax(model.forward(cyx.Tensor.from_numpy(x_test[:1]))).to_numpy().round(3)
```

It prints the ten class probabilities of the first test row.

## What the script does not do

The graph also validates every epoch, keeps the best checkpoint, fills the
Training Dashboard and writes **Results > Test**. A script does only what
it says. Use graphs for that, and scripts for experiments, data work and
checks.

## Next steps

- The Script Editor in full: [Write and run scripts](../user-guide/scripting/script-editor.md)
- The console and its ready-made names: [Use the Python console](../user-guide/scripting/python-console.md)
- Every class and function of pycyxwiz: [pycyxwiz API reference](../python/reference/index.md)
- Turn a script into a node: [Your first Python node](../sdk/get-started/first-python-node.md)
