---
title: Python and project environments
description: Which Python the Engine runs, the environment each project gets, how to install packages into it, and what happens when you change project.
type: how-to
area: scripting
audience: user
status: stable
since: 1.0.7
owner: scripting
last_reviewed: 2026-11-25
related:
  - ../../install/first-launch.md
  - python-console.md
  - startup-scripts.md
  - ../../main-window/preferences.md
---

Use this to choose the Python the Engine uses, to add packages such as
pandas or scikit-learn to a project, or to understand a "restart needed"
message. Scripts, notebook cells, the Python console and Python nodes all
run in one Python inside the Engine.

## Which Python runs

The Engine embeds one Python version (3.12 in the release packages) and can
only use an installed Python of that same version, with `venv` and `pip`.
Which one is set in **File > Preferences... > Python > Interpreter**:

- Empty: the Python shipped with the Engine, when there is one.
- A path: type it and press Enter, select **Browse...**, or select **Find
  Python** to list the Pythons on this computer and pick one. The line under
  the field says whether it can be used; **Use this Python** takes it.
- **Runtime details** shows the running Python: its paths, version and
  packages folder, with **Copy**.

The first time the Engine starts without a usable Python, the **Python for
scripting** dialog helps you install and pick one. See
[First launch: Python for scripting](../../install/first-launch.md#python-for-scripting).

A Python that is already running keeps its interpreter until the Engine
restarts.

## The project's environment

Each project gets its own virtual environment in its `python` folder, made
from the interpreter above, so packages you install for one project do not
change another.

- It is made when the project is created, or when a project without one is
  first opened (about ten seconds). The status bar shows "Setting up the
  project's Python environment..." meanwhile, and Python waits for it, then
  starts in the new environment.
- **Preferences > Python > New projects > Create a virtual environment for
  each new project** (on by default) turns this off: a new project then runs
  in the interpreter itself.
- **Packages to install** lists packages, separated by spaces or commas
  (`numpy pandas`, with versions such as `pandas>=2,<3`), that pip installs
  into each new environment before Python starts there. A package that fails
  is reported in the Console log; the environment is still used.
- The project's `scripts` folder is on Python's import path, so a script
  can `import` another script there.

## Install packages

Install into the open project's environment from the Console:

1. In the Console, select **+** and choose **Commands**.
2. Type a pip command and press Enter, for example:

   ```text
   pip install scikit-learn
   ```

3. pip runs in the background with the project environment's own pip; its
   output shows under the command. The Engine stays responsive.
4. Run your code again.

`pip list`, `pip show numpy` and `pip install --upgrade polars` work the
same way; `pip3` is the same command. When code fails with `No module named
'...'`, the Python console shows the `pip install` line to use. A compiled
package that was already imported (NumPy, for example) needs a restart of
the Engine to load a new version.

If the Console says "Project virtual environment not found", the
environment is still being made, or the project has none; wait, or turn on
**Create a virtual environment for each new project** and reopen the
project.

## Changing project

New in 1.0.8. Python runs inside the Engine and stays running when you open
another project (or close one). When the new project's environment is built
on the same Python installation and version as the running one (the usual
case: every project environment is made from the Engine's Python), Python
moves to it in place, with no restart: the old environment's packages and
the old project's `scripts` folder leave the import path, the modules
imported from them are forgotten, and the new ones take their place. The
log says "Python now runs ... (switched without a restart)".

A restart of the Engine is needed when:

- the new environment uses another Python installation or version, or
- a compiled module of the old environment is already loaded (for example
  `numpy`, imported by a cell, a script or a Python node): it cannot be
  unloaded from a running Python.

Then the status bar says *Restart to use this project's Python* (hover it
for why), and **Restart** (or **File > Restart**) reopens the project in a
new Engine. Until then cells, scripts and Python nodes do not run in the
wrong environment: they say a restart is needed.

## What starts with Python

Each Python session (before its first command or script, and again after
**Restart** in the console or a notebook) imports `numpy as np` and
`pycyxwiz` with its short names, then runs your startup scripts. Both
imports can be turned off. See [Startup scripts](startup-scripts.md).

To see every command the Engine sends to Python, with how long it took,
turn on **Tools > Diagnostics > Log Every Python Command**; the lines go to the Console
log.

## If something goes wrong

**None set, and no Python ships with this Engine: Python features are off
until one is chosen.**: pick an interpreter in Preferences > Python, or
install Python 3.12 first.

**This one cannot be used**: the interpreter is not the Python version the
Engine embeds, or cannot be run. Install the matching version.

**The project's Python environment could not be created.**: the interpreter
has no `venv` module, or the project folder cannot be written. On Linux,
install the `venv` package of your Python (see
[Linux: Python for scripting](../../install/linux.md#python-for-scripting)).

## Related

- [Use the Python console](python-console.md)
- [Startup scripts](startup-scripts.md)
- [Preferences: Python](../../usage/preferences.md#python)
