# Extending CyxWiz with your own nodes

CyxWiz lets you add nodes it does not ship and train them next to the
built-in ones. A node you add is an **extension node**: it appears in the
node editor, saves in `.cyxgraph` files, compiles into the model and trains.

> These pages are written while the feature is built (TOFIX125) and change
> with it. Each page says what it was checked against. Last checked:
> 2026-10-17, plugin API 1.1.0, branch `tofix125-extension-nodes`.

## Two routes

| Route | Who it is for | State |
|---|---|---|
| **Python node**: a class written with `pycyxwiz`, bundled into a node | Most users; Python is the usual language for machine learning | In progress: Python modules train in pycyxwiz today; nodes in the Engine come in stage P4 |
| **C++ plugin**: a shared library the Engine loads | Programmers who need native speed or Engine access | Works in a source build of CyxWiz (this guide) |

Both routes give the Engine the same thing: a `cyxwiz::Module` with a
forward and a hand-written backward step, which the Engine checks
numerically. Both follow one [node contract](node-contract.md).

## Pages

| Page | Read it to |
|---|---|
| [python-modules.md](python-modules.md) | Write a trainable module in Python and train it with pycyxwiz |
| [python-nodes.md](python-nodes.md) | Describe a Python module as a node: identity, settings, shape rule |
| [plugin-quickstart.md](plugin-quickstart.md) | Build the example plugin, load it and train with its nodes |
| [plugin-developer-guide.md](plugin-developer-guide.md) | Write your own C++ plugin with trainable nodes |
| [node-contract.md](node-contract.md) | Know what every extension node must guarantee |
| [troubleshooting.md](troubleshooting.md) | Fix "not installed", refused settings, failed gradient checks |

## What works today and what does not

| Works | Not yet |
|---|---|
| Trainable nodes from a C++ plugin: one tensor in, one tensor out | Nodes with several inputs or outputs |
| Typed settings shown in Properties | Autograd: you write the backward step yourself |
| Numerical gradient check in Properties | Training on a remote Server Node (refused with a message) |
| Saving and reopening graphs; a missing plugin shows its nodes as "Not installed" | Exporting a model with extension nodes to ONNX or Python |
| Plugins load again when the Engine starts | A packaged SDK you can use outside the CyxWiz source tree |
| Apps menu lists every loaded plugin | Python nodes in the Engine |
| Trainable modules written in Python, trained with pycyxwiz | |
