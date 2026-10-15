# Notebooks in the Script Editor

The Script Editor opens Jupyter notebooks (`.ipynb`) as cells, with the
outputs they were saved with.

## Opening one

- Double-click the notebook in the **Asset Browser** (Python files `.py`,
  `.pyw`, `.cyx` open the same way), or
- **File > Open Script...** in the Engine or the Script Editor.

## The kernel

Notebook cells run in the Engine's own Python (the project's environment
when the project has one, else the Engine's Python). The chip at the right
of the notebook bar says which, and what it is doing:

| Chip | Meaning |
| --- | --- |
| `Python 3.12.9 · sentiment · Idle` | Running in the project's environment, nothing running. |
| `... · Busy` | A cell is running; **Interrupt** stops it at its next Python line. |
| `... · Waiting` | Another script or a Console command holds Python; the cells wait. |
| `Python · ... · Starts on first run` | Python starts when the first cell runs. |
| `... · Restart the Engine` | The project now asks for another Python (for example a `python` virtual environment made after Python started). Python runs inside the Engine, so it switches when the Engine restarts; the tooltip names both interpreters. The status bar shows the same note. |

**Restart** clears this notebook's variables and starts the `[n]` count
again; the outputs stay. It waits until no cell, script or Console command
is running, and it does not change the Python the Engine runs (see
*Restart the Engine* above). It is safe while a graph trains: training does
not use the notebook's Python.

## Figures

`plt.show()` puts the figure under the cell. `%matplotlib inline` (any
`%matplotlib` line) is understood: from that line on, figures go to the cell,
also after `matplotlib.use('Agg')`. Other `%` magics and `!` shell lines are
not supported; the cell stops with a syntax error. See
[plotting.md](plotting.md).

## Closing the Engine

The Engine closes cleanly after notebooks, scripts or the Console ran Python
(it used to crash on exit after any Python run, 2026-10-15). Unsaved
notebooks are listed in the close dialog with **Save scripts and close**,
**Close without saving** and **Keep working**.
