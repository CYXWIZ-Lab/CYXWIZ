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
| `... · Restart the Engine` | The project asks for a Python that the running one cannot become in place (see *Changing project* below); the tooltip names both interpreters, and the status bar says *Restart to use this project's Python* with a **Restart** link. |

A project without an environment gets one when it is created or first opened
(`python/`, about 10 seconds). The status bar shows "Setting up the project's
Python environment..." meanwhile, and Python waits for it, then starts in the
new environment: no restart needed.

**Changing project.** Python runs inside the Engine and stays running when you
open another project (or close one). When the new project's environment is
built on the same Python installation and version as the running one (the
usual case: every project environment is made from the Engine's Python),
Python moves to it in place, with no restart: the old environment's
site-packages and the old project's `scripts` folder leave `sys.path`, the
modules imported from them are forgotten, and the new ones take their place.
The log says "Python now runs ... (switched without a restart)".

A restart of the Engine is needed when:

- the new environment uses another Python installation or version, or
- a compiled module of the old environment is already loaded (for example
  `numpy`, imported by a cell, a script or a Python node): it cannot be
  unloaded from a running Python.

Then the status bar says *Restart to use this project's Python* (hover: why),
and **Restart** (or **File > Restart**) reopens the project in a new Engine.
Until then cells, scripts and Python nodes do not run in the wrong
environment: they say a restart is needed.

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
