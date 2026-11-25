---
title: The Studio bar
description: Every button, menu and status chip on the bar above the CyxWiz Studio canvas, and how the bar folds when narrow.
type: reference
area: general
audience: user
status: stable
since: 1.0.7
owner: main-window
last_reviewed: 2026-11-24
related:
  - index.md
  - ../user-guide/graph-editor/compile-and-run.md
  - ../user-guide/graph-editor/canvas.md
  - ../user-guide/training/studio-debugger.md
redirect_from:
  - toolbar.md
---

The Studio bar is the toolbar at the top of the CyxWiz Studio canvas. It
is one row:

> the graph · Save · Edit · View · the mode · the framework · Code · More ...
> Compile › Local Debug › Train

The Studio's tab names the graph, for example **CyxWiz Studio ·
mnist_mlp** (**Untitled** when the graph has no file), with a dot after the
name while it has unsaved changes. A window of its own shows the same
title.

## The graph

**The graph's button** shows the file name (or *Untitled graph*) and a dot
when the graph changed since it was saved. Hover for the file path, or "Not
saved to a file yet". Click it for the graph menu:

| Item | What it does |
| --- | --- |
| **Save** (Ctrl+S) | Writes the graph's file. A graph with no file asks for a name. On an empty canvas it writes nothing and says *Nothing to save yet*: "The canvas is empty. Add a node, then save." |
| **Save As...** (Ctrl+Shift+S) | Writes a new file; the open graph moves to it |
| **Load...** | Opens another graph in place of this one. The loaded graph starts its own undo history: Ctrl+Z does not bring back the previous graph |
| **Export a copy of the graph...** | Writes a copy where you choose; the open graph stays on its own file |
| **Make a Python node...** | Opens the Python node maker |
| **Clear the canvas...** | Asks **Clear the canvas?** with the number of nodes and links, then removes them. Ctrl+Z brings them back. The file changes only when you save |

While a run keeps the graph busy (training, a test, a pipeline, a Data
Input loading its data), **Load...** and **Clear the canvas...** are greyed
out with the reason ("Not while ...").

**Save** is also a button on the bar. Ctrl+S and Ctrl+Shift+S save the
graph while the canvas has the focus; in the Script Editor they save the
script.

A starter graph, a pattern and an imported model open as *Untitled
graph*, so Save asks for a name and never overwrites the example files.

## Edit and View

- **Edit**: **Select all** (Ctrl+A), **Duplicate** (Ctrl+D), **Delete
  selected** (Delete). With nothing selected, Duplicate and Delete are
  greyed out with "Select nodes first".
- **View · 100%** (the current zoom on the button): **Zoom** **-** and
  **+** (from 25% to 200%; the view keeps its centre), **Zoom to 100%**,
  **Fit all nodes** (F), **Minimap** (M), **Find in the graph** (Ctrl+F).

See [The canvas](../user-guide/graph-editor/canvas.md) for moving around,
selecting and undo.

## The mode, the framework and Code

The mode decides how **Train** and **Code** treat the graph. Hover each for
what it does:

| Mode | Use it to |
| --- | --- |
| **Code Gen** | Turn the graph into Python code for a framework |
| **Data Pipeline** | Run the data steps only (**Execute Pipeline**); training nodes run in Train |
| **Local Training** | Train the whole graph on this computer |

- **The framework** (only in **Code Gen** mode): **PyTorch**,
  **TensorFlow**, **Keras** or **PyCyxWiz**. In the other modes the Code
  menu names the framework it uses, so code is never written for a
  framework you cannot see.
- **Code**:
  - **Generate <framework> Code into Script Editor** opens a Script Editor
    tab named after the graph and the framework (`mnist_mlp_pytorch.py`). A
    second Generate replaces that tab's text.
  - **Save <framework> Code as .py...** asks where, then writes the file.
  - When the graph cannot become code, both are greyed out with the reason
    ("The canvas is empty. Add a node first.", "Graph must have at least
    one Output node.").
  - A line under the bar says what was written where, for example
    "Generated 118 lines of PyTorch code into Script Editor tab
    mnist_mlp_pytorch.py · 14:32". **Hide** removes it.
- **Execute Pipeline** (in **Data Pipeline** mode) runs the data steps.
  While it runs, the bar shows its progress and **Cancel**.
- A graph with simulation nodes shows **Run Sim** instead, and during a run
  **Pause** (or **Resume**), **Step**, **Stop Sim** and the simulation time.
  See [Graph simulation](../usage/graph_simulation.md).

## Compile, Local Debug and Train

The right side of the bar holds the three steps from a graph to a trained
model. **Compile** and **Local Debug** each have a name (click to run the
step) and a chip with their last result (click for its menu).

| Step | Key | Click the name | The chip |
| --- | --- | --- | --- |
| **Compile** | F7 | Checks the graph and opens the compile report | **Empty**, **Not checked**, **Checking...**, **Ready**, **N errors**, **Failed**. Menu: **Open the compile report**, **Show the first finding on the canvas** |
| **Local Debug** | F6 | Runs one forward and backward pass in the Studio Debugger | **Not run**, **Running...**, **Passed <time>**, **Failed**, **Graph changed**. Menu: **Run Local Debug**, **Open the Studio Debugger** |
| **Train** | F5 | Starts training | No chip; while training, the run's line takes its place |

- Compile checks the graph as you edit (shortly after each change). In
  **Data Pipeline** mode it does not (a data graph has no model): the chip
  says **Not checked** until you press Compile.
- Local Debug says **Graph changed** when it ran on the graph before your
  last edit (an **Apply** on a Data Input counts). Only one Local Debug runs
  at a time.
- **Train** is greyed out, with the reason on hover, when it cannot start:
  "Training is not available yet (no dataset loaded?)", "The graph is not
  ready to train. It needs a dataset input, model layers and a loss.",
  "Train waits while a Data Input is loading its data. It can start when
  that ends.", or, when the project requires it, "This project requires a
  passing Local Debug (F6) on this graph".
- While training, Train's place shows **Epoch 3 / 10** with **Pause** (or
  **Resume**: "Paused at epoch 3 / 10") and **Stop**. A run waiting for the
  GPU says **Waiting for the GPU**. A reinforcement-learning (PPO) graph
  shows its update and steps the same way.

What Compile checks, what each run mode does, and what Train does with
your data before it trains:
[Compile and run](../user-guide/graph-editor/compile-and-run.md#what-train-does-first). Local
Debug in depth: [The Studio Debugger](../user-guide/training/studio-debugger.md).

## An empty canvas

An empty canvas says **Start a graph**, with three buttons:

- **Search nodes**: the search box at the top right of the canvas takes the
  keyboard;
- **Open the Nodes panel**: drag a node from it onto the canvas;
- **Open an example graph**: one of the five task starters of the
  [start page](start-page.md#task-starter-graphs), opened as *Untitled
  graph*.

## Find in the graph

Ctrl+F (or **View > Find in the graph** on the bar) opens a find bar in the
canvas's top-left corner. It finds nodes by name, by type, and by any
setting's name or value ("dense", "relu", "0.001").

- Every match gets a ring; the current one is stronger and the view
  centres on it.
- **Enter** or **F3** goes to the next match, **Shift+Enter** or
  **Shift+F3** to the previous one; **Esc** closes the bar.
- "2 of 3" says where you are; "No match" when nothing matches.

## A narrow canvas

Nothing is cut off. When the row is too narrow it folds, in this order, and
unfolds as it widens:

1. The mode becomes a menu, **View** drops the zoom from its button, and
   the simulation buttons keep their icons only.
2. One at a time, into **More** at the end of the left side: Local Debug,
   Compile, View, Edit, Code (with Execute Pipeline), the framework, Save.
3. A run's line shortens to its state (*Epoch 3 / 10*, *Paused 3 / 10*,
   *Waiting*); hover it for the whole line.
4. The mode goes into **More**, and last the graph's button keeps only its
   icon and the unsaved dot.

**Train** (or a run's line) and a running pipeline's progress always stay
on the bar. In the default layout the Studio is this narrow: the graph's
button, **More** and **Train**. While Compile is in **More**, a dot on
**More** keeps its state in sight: green when the graph compiles, red when
it does not, grey before the first check. Hover **More** for the line
("Compile: Not ready to train: 2 errors").

## The canvas footer

Along the bottom of the canvas: the node and link counts, the selection
("2 selected"), frozen layer totals, and the mode ("Data Pipeline mode ·
training nodes run in Train"). While a run would refuse a load or clear, it
says "Graph replacement locked: ..." with the reason.

## Related

- [Graph editor](../user-guide/graph-editor/index.md)
- [Keyboard shortcuts](../reference/shortcuts.md)
