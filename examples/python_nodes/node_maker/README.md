# Make a Python node: the three starters

The Engine's **Nodes > Make a Python node...** starts each kind of node from
one that works. These are those three, as the window writes them into a
project's `nodes_src/` folder:

| File | Kind | Node | What it computes |
|---|---|---|---|
| `nodes_src/soft_clip.py` | Activation (`f` and `df`) | Soft Clip | x / (1 + \|x\|) |
| `nodes_src/swish.py` | Layer shown as Activation, one setting `beta` | Swish (beta) | x * sigmoid(beta * x) |
| `nodes_src/log_cosh.py` | Loss on values | Log-cosh | mean(log(cosh(prediction - target))) |
| `check_node_maker.py` | Runs the gradient check the window's **Check** runs | | |

Licensed under the Apache License 2.0 (the SDK's `LICENSE` and `NOTICE`).

## Use them in the Engine

1. Open your project and copy `nodes_src/` into the project folder (or copy
   the files into its `nodes_src/`).
2. **Nodes > Make a Python node... > Open** lists them. Pick one: the form
   fills in with its kind, names, settings and code.
3. **Check**, then **Create node**. The bundle is written to the project's
   `nodes/` folder with the project's own id (`project.<project>.<file>`) and
   approved for you; the node is in the Nodes panel under **Plugins**, in its
   group (Activation, or Training for the loss).

The files begin with the line `# Made with Make a Python node in CyxWiz.`:
that line is how **Open** knows the window can read the file back. A file
without it opens in the Script Editor instead.

## Check them without the Engine

    python check_node_maker.py <folder holding pycyxwiz> [<folder of its DLLs>]

Each prints `ok` with its largest relative error (all three are below
0.0001; the limit is 0.02).

How the window works: [docs/usage/extension_nodes.md](../../../docs/usage/extension_nodes.md#make-a-python-node).
