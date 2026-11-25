---
title: Notebooks
description: Open Jupyter and .cyx notebooks in the Script Editor, run cells, read their outputs and figures, and restart the kernel.
type: how-to
area: scripting
audience: user
status: stable
since: 1.0.7
owner: scripting
last_reviewed: 2026-11-25
related:
  - script-editor.md
  - cyx-documents.md
  - variable-explorer.md
  - python-environment.md
  - ../evaluation/plots-from-python.md
redirect_from:
  - ../../usage/notebooks.md
  - usage/notebooks.md
---

Use this when you explore with code cell by cell. The Script Editor opens
Jupyter notebooks (`.ipynb`) as cells, with the outputs they were saved
with, and `.cyx` notebooks (cells in a plain-text file, see
[The .cyx document format](cyx-documents.md)).

## Open or make a notebook

- Double-click a `.ipynb` or `.cyx` file in the **Asset Browser**, or use
  **File > Open Script...** (Ctrl+O). A `.ipynb` file always opens as a
  notebook; a `.cyx` file opens as one when it has cell markers.
- To turn the open script into cells, choose **View > Notebook Mode** in
  the Script Editor's menu, or press **Ctrl+Shift+M**. Press it again to go
  back to plain text.

Keep notebooks in `.cyx` or `.ipynb` files. A `.py` file saved in notebook
mode gets the cell marker lines written into it, and Python cannot run
those lines outside notebook mode.

## Work with cells

The notebook bar holds **+ Code**, **+ Markdown**, **Run All**, **Run
Above**, **Run Below**, **Interrupt**, **Restart**, **Clear outputs**,
**Outline** and **Variables**, and the kernel chip at the right. In a narrow
panel some of them move into **...**.

A notebook has two modes, as in Jupyter. The status bar says which, with the
selected cell (**Notebook · Cell 3 of 12**):

- **Command mode** (Esc): keys act on cells.
- **Edit mode** (Enter, or double-click a cell): you type in the cell, with
  the Script Editor's completion, signature help and go to definition.

| Key (command mode) | What it does |
| --- | --- |
| Shift+Enter | Run the cell and select the next |
| Ctrl+Enter | Run the cell and stay on it |
| Up, Down | Select the cell above or below |
| A, B | Add a code cell above or below |
| M, Y | Make the cell markdown or code |
| D, D | Delete the cell |
| C, O | Fold the cell, fold its output |

Each cell has a run button, its type (**Code**, **Text (markdown)**,
**Raw**), move up and down, delete, and **...** with **Edit**, **Run from
here**, **Duplicate**, **Split at cursor**, **Merge with the cell below**,
**Copy cell**, **Cut cell**, **Paste cell below**, fold and output actions.

## Run cells

- **Run All** runs every code cell in order; **Run Above** runs the cells
  from the top through the selected one; **Run Below** the selected cell and
  every cell after it.
- A running cell reads **Running**, the ones waiting **Queued**. After an
  error, the cells after it say "Not run: Run All stopped after the error in
  [n]".
- **Interrupt** stops the running cell at its next Python line; queued
  cells are not run.
- **Debug cell** at the right of the tab strip debugs the selected cell. See
  [Debug a script](debug-a-script.md).

Each code cell's output shows under it: printed text, the value of its last
expression (`Out[n]`), tables (with **Open in Table Viewer**), figures and
errors (with **Show full traceback**).

## The kernel

Notebook cells run in the Engine's own Python (the project's environment
when the project has one, else the Engine's Python). Each notebook has its
own variables; they stay until **Restart**. The chip at the right of the
notebook bar says which Python, and what it is doing:

| Chip | Meaning |
| --- | --- |
| `Python 3.12.9 · sentiment · Idle` | Running in the project's environment, nothing running. |
| `... · Busy` | A cell is running; **Interrupt** stops it at its next Python line. |
| `... · Waiting` | Another script or a Console command holds Python; the cells wait. |
| `Python · ... · Starts on first run` | Python starts when the first cell runs. |
| `... · Restart the Engine` | The project needs a Python that the running one cannot become in place; the tooltip names both interpreters, and the status bar says *Restart to use this project's Python* with a **Restart** link. |

**Restart** clears this notebook's variables and starts the `[n]` count
again; the outputs stay. It waits until no cell, script or Console command
is running, and it does not change the Python the Engine runs. It is safe
while a graph trains: training does not use the notebook's Python.

When you open another project, Python moves to its environment without a
restart in the usual case. When it cannot, cells say a restart is needed.
See [Changing project](python-environment.md#changing-project).

## Figures

`plt.show()` puts the figure under the cell. `%matplotlib inline` (any
`%matplotlib` line) is understood: from that line on, figures go to the
cell, also after `matplotlib.use('Agg')`. Other `%` magics and `!` shell
lines are not supported; the cell stops with a syntax error. Beside each
figure, **Copy**, **Save as...** and **Open in window** (Plot Output, with
zoom and pan). See
[Plots from Python](../evaluation/plots-from-python.md).

## Save

**Save** (Ctrl+S) writes a `.ipynb` file as Jupyter notebook format 4 with
its outputs; what the Engine does not draw (metadata, attachments, other
outputs) is written back unchanged. A `.cyx` notebook keeps the cells'
text only, not their outputs.

Unsaved notebooks (and an unsaved graph on the canvas) are listed when you
close the Engine, under **Unsaved work**, with **Save and close**, **Close
without saving** and **Keep working**.

## Related

- [The .cyx document format](cyx-documents.md)
- [The Variable Explorer](variable-explorer.md)
- [Keyboard shortcuts](../../reference/shortcuts.md)
