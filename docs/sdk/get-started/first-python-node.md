---
title: Your first Python node
description: Write an activation in Python, check its gradient, make it a node with Create node from script, and train a graph with it.
type: tutorial
area: sdk
audience: plugin-developer
status: stable
since: 1.0.7
owner: sdk
last_reviewed: 2026-11-25
related:
  - ../guides/python-nodes.md
  - ../guides/python-modules.md
  - ../../usage/extension_nodes.md
  - ../../python/reference/index.md
---

In this lesson you write a new activation, Softsign, in a few lines of
Python, check that its derivative is right, turn it into a node of the
Engine's Nodes panel, and train a graph that uses it. You need an
installed Engine and a project; nothing else (the Engine carries Python
3.12 and `pycyxwiz`).

> Checked against plugin API 1.4.0 and pycyxwiz 0.2.0.
> The Python samples on this page run, in order, as one script
> (`scripts/docs/check_extension_docs.py` runs them).

## 1. Write the activation

An activation works element by element, so you only write the function
`f(x)` and its derivative `df(x)`. Softsign is `x / (1 + |x|)`; its
derivative is `1 / (1 + |x|)^2`.

1. Open your project (**File > Open Project...**), or create one.
2. Choose **Script > New Script...**, name the script `my_nodes.py` and keep
   the project folder. The script opens in the Script Editor.
3. Type this code and save it (**Ctrl+S**):

```python
import pycyxwiz as cyx
from pycyxwiz import nodes


@nodes.activation(id="com.example.first", name="Softsign", version="1.0.0",
                  description="x / (1 + |x|)")
class Softsign:
    def f(self, x):
        return x / (x.abs() + 1.0)

    def df(self, x):
        d = x.abs() + 1.0
        return 1.0 / (d * d)
```

`@nodes.activation` turns the class into a full `pycyxwiz.Module`: the
forward step returns `f(x)` and the backward step returns
`grad_output * df(x)`. The `id` names your set of nodes; keep it the same
for every node you add to it later.

## 2. Check the derivative

A wrong derivative does not crash: the model trains badly. The gradient
check compares your `df` with small numerical steps of your `f`.

1. Add these lines at the end of `my_nodes.py`:

```python
if __name__ == "__main__":
    print(nodes.gradcheck(Softsign))
```

2. In the Script Editor, choose **Run > Run Script** (**F5**). The
   script's output appears in the Console's Python session:
   `Passed: largest relative error ... (tolerance 2.0e-02)`.

The check ran here too, as it would in the Engine:

```python
result = nodes.gradcheck(Softsign)
print(result)                                  # Passed: ...
print(nodes.describe(Softsign)["shape_rule"])  # same_as_input
```

Keep the check under `if __name__ == "__main__":`. The Engine imports the
file when it builds the node, and code outside that guard would run every
time.

## 3. Make it a node

1. In the Script Editor, with `my_nodes.py` open and saved, choose **Run >
   Create node from script...**. The dialog lists **Softsign** with its
   **Input** and **Output** pins.
2. Click **Check gradients**. Softsign shows that it passed.
3. Keep the **Bundle name** the dialog suggests (`my-nodes`, from the file
   name) and click **Create nodes**.

The Engine writes the bundle to `<project>/nodes/my-nodes.cyxnode/` (your
code as `node.py` and a generated `node.json`) and approves its code, since
you wrote it. Within two seconds **Softsign** appears in the Nodes panel
under **Activation**.

## 4. Train with it

1. Open a graph that trains a small classifier, or build one: see
   [Build a model in the graph editor](../../user-guide/graph-editor/index.md).
2. Drag **Softsign** from the Nodes panel onto the canvas and connect it
   between two layers, where a ReLU would go.
3. Select the node. **Properties** reads *Python node · com.example.first
   1.0.0 · Activation* and shows the gradient check from step 3.
4. Press **Compile** (**F7**), then **Train** (**F5**).

The first build of the node starts Python for the project, so the first
compile can say "Python is starting for the Python nodes (it needs an open
project); try again in a moment". Compile again after a moment.

## 5. Change it

Edit `my_nodes.py`, save it, and run **Create node from script...** again.
The new code replaces the bundle and is approved again; graphs that use
Softsign pick it up without a restart. Raise `version` when you change a
node's settings, pins or code.

## What you did

- Wrote a node as ordinary Python with `pycyxwiz.nodes`.
- Checked its derivative with `nodes.gradcheck`.
- Bundled it into the project's `nodes` folder, where the Engine finds it.

## Next

- [Describe a Python module as a node](../guides/python-nodes.md): layers
  with learnable parameters and settings, losses, bundles and approval in
  full.
- [Write a module in Python](../guides/python-modules.md): `forward`,
  `backward` and training from a script.
- [Your own nodes in the Engine](../../usage/extension_nodes.md): the
  form in **Nodes > Make a Python node...**, which writes this kind of file
  for you, and the Plugin Manager.
- [pycyxwiz API reference](../../python/reference/index.md)
