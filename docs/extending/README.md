# Extending CyxWiz with your own nodes

CyxWiz lets you add nodes it does not ship and train them next to the
built-in ones. A node you add is an **extension node**: it appears in the
node editor, saves in `.cyxgraph` files, compiles into the model and trains.

> These pages are written while the feature is built (TOFIX125) and change
> with it. Each page says what it was checked against. Last checked:
> 2026-10-17, plugin API 1.2.0, branch `tofix125-extension-nodes`.

## Two routes

| Route | Who it is for | State |
|---|---|---|
| **Python node**: a class written with `pycyxwiz`, bundled into a node | Most users; Python is the usual language for machine learning | Works: write and bundle with pycyxwiz, or Create node from script in the Engine; approve, train |
| **C++ plugin**: a shared library the Engine loads | Programmers who need native speed or Engine access | Works in a source build of CyxWiz (this guide) |

Both routes give the Engine the same thing: a `cyxwiz::Module` with a
forward and a hand-written backward step, which the Engine checks
numerically. Both follow one [node contract](node-contract.md).

## Pages

| Page | Read it to |
|---|---|
| [python-modules.md](python-modules.md) | Write a trainable module in Python and train it with pycyxwiz |
| [python-nodes.md](python-nodes.md) | Describe a Python module as a node, bundle it, approve and use it in the Engine; ten built-in layers as Python example nodes |
| [plugin-quickstart.md](plugin-quickstart.md) | Build the example plugin, load it and train with its nodes; build your own against the SDK |
| [plugin-developer-guide.md](plugin-developer-guide.md) | Write your own C++ plugin with trainable nodes |
| [plugin-api-reference.md](plugin-api-reference.md) | Look up every C++ type, `plugin.json` field, CMake name and what the Engine does with it |
| [node-contract.md](node-contract.md) | Know what every extension node must guarantee |
| [troubleshooting.md](troubleshooting.md) | Fix "not installed", refused settings, failed gradient checks |

## What works today and what does not

| Works | Not yet |
|---|---|
| Trainable nodes from a C++ plugin: one tensor in, one tensor out | Nodes with several inputs or outputs |
| Typed settings shown in Properties | Autograd: you write the backward step yourself |
| Numerical gradient check in Properties | Training on a remote Server Node (refused with a message) |
| Saving and reopening graphs; a missing plugin shows its nodes as "Not installed". A trained model saves as a CyxWiz package and imports with its extension weights | Exporting a model with extension nodes to ONNX or Python code, or deploying it on a Server Node (refused with the node's name) |
| Plugins load again when the Engine starts | |
| Apps menu lists every loaded plugin and Python node bundle | |
| C++ plugins built outside the CyxWiz sources, against the plugin SDK that every installation carries (`find_package(CyxWizSDK)`, template, plugin check; Apache-2.0) | |
| Trainable modules written in Python, trained with pycyxwiz; `gradcheck`; `pycyxwiz.nodes` decorators and `bundle()` | |
| Python nodes in the Engine: bundles listed, approved per code, trained; Create node from script | |
