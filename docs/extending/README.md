# Extending CyxWiz with your own nodes

CyxWiz lets you add nodes it does not ship and train them next to the
built-in ones. A node you add is an **extension node**: it appears in the
node editor, saves in `.cyxgraph` files, compiles into the model and trains.

A C++ plugin can also add **dataset formats** (Data Input opens their files)
and **dataset analytics** (Data Studio's Explore > Plugins runs them): see
[Data plugins](data-plugin-guide.md).

> Checked against plugin API 1.3.0 and pycyxwiz 0.2.0.
> Each guide carries this line; a test fails when the plugin API or pycyxwiz
> version changes until the guide is checked again (see
> [How these pages stay true](#how-these-pages-stay-true)).

## Two routes

| Route | Who it is for | What you need |
|---|---|---|
| **Python node**: a class written with `pycyxwiz`, bundled into a `.cyxnode` folder in your project | Most users; Python is the usual language for machine learning | An installed CyxWiz Engine (it bundles Python 3.12 and pycyxwiz) |
| **C++ plugin**: a shared library the Engine loads | Programmers who need native speed | An installed CyxWiz Engine (it carries the plugin SDK and `cyxwiz-plugin-check`) and the same C++ compiler the Engine was built with |

Both routes give the Engine the same thing: a module with a forward step
and a hand-written backward step, which the Engine checks numerically
before it trains. Both follow one [node contract](node-contract.md).

### Start here

- **Python**: [Writing a module in Python](python-modules.md), then
  [Describing a Python module as a node](python-nodes.md), then
  [Your own nodes in the Engine](../usage/extension_nodes.md).
- **C++**: [Plugin quickstart](plugin-quickstart.md), then the
  [developer guide](plugin-developer-guide.md).
- **Dataset formats and analytics (C++)**: the quickstart's sections 1 and
  3, then [Data plugins](data-plugin-guide.md).

## Pages

| Page | Read it to |
|---|---|
| [../usage/extension_nodes.md](../usage/extension_nodes.md) | Use extension nodes in the Engine: install, approve, the example layers and losses, save and export |
| [python-modules.md](python-modules.md) | Write a trainable module in Python and train it with pycyxwiz |
| [python-nodes.md](python-nodes.md) | Describe a Python module or loss as a node, bundle it, approve and use it in the Engine; ten built-in layers and four losses as Python example nodes |
| [plugin-quickstart.md](plugin-quickstart.md) | Build the template plugin against the SDK, check it, load it and train with its node |
| [plugin-developer-guide.md](plugin-developer-guide.md) | Write your own C++ plugin with trainable nodes |
| [data-plugin-guide.md](data-plugin-guide.md) | Add a dataset format to Data Input and an analytic to Explore > Plugins, following the SDK's example (ARFF and coefficient of variation) from build to code |
| [node-contract.md](node-contract.md) | Know what every extension node must guarantee, in C++ and in Python |
| [troubleshooting.md](troubleshooting.md) | Fix "not installed", refused settings, failed gradient checks, plugins that do not load |

Reference pages, generated from the code (do not edit them by hand):

| Page | What it lists |
|---|---|
| [python-api-reference.md](python-api-reference.md) | Every public name of pycyxwiz, with its signature and description |
| [tensor-operations.md](tensor-operations.md) | The tensor operations a node can use, C++ and Python side by side |
| [sdk-contents.md](sdk-contents.md) | Every header the plugin SDK installs and the types it declares |

The C++ [plugin API reference](plugin-api-reference.md) explains every
plugin type, `plugin.json` field, CMake name and what the Engine does with
it; it is written by hand and checked against the code.

## What works today and what does not

| Works | Not yet |
|---|---|
| Trainable nodes with one tensor in and one tensor out: layers and activations, from Python or C++ | Nodes with several inputs or outputs (a loss has its fixed two) |
| Losses written in Python (`@nodes.loss`): predictions and targets in, the graph's training objective; classification or regression | Losses from C++ plugins |
| Typed settings shown in Properties; a numerical gradient check in Properties | Autograd: you write the backward step yourself |
| Saving and reopening graphs; a missing extension shows its nodes as "Not installed" and keeps them | Training on a remote Server Node (refused with a message naming the node) |
| A trained model saves as a CyxWiz package that records the extensions it needs, and imports with their weights | Exporting a model with extension nodes to ONNX or Python code (refused with the node's name) |
| Python: write, check and bundle with pycyxwiz, or **Create node from script** in the Script Editor; each bundle runs only after you approve its code | Python packages declared by a bundle: a bundle uses the packages of the Engine's Python |
| C++: plugins built outside the CyxWiz sources against the SDK every installation carries (`find_package(CyxWizSDK)`, a template, `cyxwiz-plugin-check`; Apache-2.0); plugins load again when the Engine starts | Loading a new build of a plugin while the Engine runs: unload it first (see the [quickstart](plugin-quickstart.md)) |
| The Apps menu lists every loaded plugin and Python node bundle | |

## How these pages stay true

CyxWiz changes quickly, so the extension docs are tied to the code by tests
(`ctest -R extension_docs`):

- **Generated reference.** `scripts/docs/generate_extension_reference.py`
  writes the three generated pages from the built pycyxwiz and the SDK
  headers. The test `extension_docs_reference` fails when a function, class,
  header or tensor operation is added, removed or changed without the page;
  run the script to update them.
- **Samples that run.** `scripts/docs/check_extension_docs.py` (test
  `extension_docs_samples`) runs every Python sample of these guides with
  the pycyxwiz under test, checks that every C++ and CMake sample is an
  excerpt of a file the build compiles or the SDK installs, and checks
  every relative link.
- **Version line.** Every guide states the plugin API and pycyxwiz version
  it was checked against. Raising either version fails the test until each
  guide has been read against the code again and its line updated.

A change to the SDK headers, pycyxwiz or how the Engine loads extensions
updates these pages in the same commit.
