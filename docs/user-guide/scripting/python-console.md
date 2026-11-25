---
title: Use the Python console
description: Type Python in the Console's Python REPL session, recall commands, read output and tracebacks, and install a missing package.
type: how-to
area: scripting
audience: user
status: stable
since: 1.0.7
owner: scripting
last_reviewed: 2026-11-25
related:
  - script-editor.md
  - variable-explorer.md
  - python-environment.md
  - ../../usage/console_logs.md
---

Use this to try a line of Python, look at a value, or call `pycyxwiz`
without writing a script. The Python console is the **Python REPL** session
of the Console (bottom dock). It shares its namespace with the Script
Editor: a variable a script makes is there in the console, and the other way
round.

## Open it

1. Open a project. Without one the session says "Open a project to use the
   Python REPL."
2. Choose **Script > Open Python Console**, or press **F12** (when the
   Script Editor has the focus, F12 goes to a definition instead). You can
   also select **+** in the Console and pick **Python REPL**.
3. The header shows the Python in use and its state. Python starts with your
   first command, script or cell ("Python starts when you run your first
   command.").

The Console's other sessions (**Logs**, **Commands**, **Agent LLM** and the
local shells) are described in [Console and logs](../../usage/console_logs.md).

## Run code

1. Type Python in the input at the bottom, for example:

   ```python
   import numpy as np
   a = np.linspace(0.0, 1.0, 5)
   a * 2
   ```

2. Press **Enter**. The input runs when it is complete: after a line that
   opens a block (`for`, `def`, `if`) Enter adds a line, and an empty line
   closes the block and runs it.
3. The transcript shows the code with `>>>` and `...` prompts, then what it
   printed, the value of a last expression, warnings, and the error and
   traceback when it fails, with how long it took.

| Key | What it does |
| --- | --- |
| Enter | Runs the input when it is complete |
| Shift+Enter | New line |
| Ctrl+Enter | Runs now |
| Ctrl+Up / Ctrl+Down | Earlier / later command (the last 100) |
| Tab | Completion |
| Ctrl+L | Clears the output (variables stay) |
| Ctrl+C | Interrupts the running command |

Drag the bar above the input to make it taller. **Auto-scroll on** (bottom
right) follows new output; click it to stop.

## Ready-made names

When Python starts, the Engine imports `numpy as np` and `pycyxwiz` (as
`cyx`), and defines short names: `eye`, `zeros`, `ones`, `svd`, `eig`,
`inv`, `solve`, `fft`, `ifft`, `conv`, `lowpass`, `highpass`, `kmeans`,
`dbscan`, `pca`, `tsne`, `acf`, `arima` and more, plus `pm()` to print a
matrix. When DuckDB and Polars are installed it also defines `sql()`,
`read_csv()`, `db`, `pl`, `df()`, `col()`, `pl_csv()` and `scan_csv()`. Type
`help` for the list.

```python
I = eye(3)
pm(I)
sql("SELECT 1 + 1 AS result")
```

Turn the imports off, or add your own, in **Preferences > Python > When
Python starts**. See [Startup scripts](startup-scripts.md).

## The header buttons

- **Interrupt** (while a command runs, Ctrl+C): Python stops at its next
  line ("Interrupting... Python stops at its next line.").
- **Restart**: clears all variables and imports. The interpreter keeps
  running inside the Engine, so it does not change which Python runs.
- **Copy all**: copies the whole transcript.
- **Clear**: clears the output; variables are kept.

Right-click an entry for **Copy entry**, **Copy code**, **Copy output**,
**Copy traceback** and **Edit in input** (puts its code back in the input).

## Install a missing package

When a command fails with `No module named '...'`, the entry shows
**Install packages into the project environment:** with the `pip install`
command (for example `pip install scikit-learn` for `sklearn`). Click it to
copy it, then:

1. In the Console, select **+** and open a **Commands** session.
2. Paste the command, for example `pip install scikit-learn`, and press
   Enter. It runs the project environment's pip in the background.
3. Run your code again in the Python REPL session.

`pip list` and `pip show <name>` work the same way. See
[Python and project environments](python-environment.md#install-packages).

## While a script is paused

During a debug run that is paused, a command runs in the paused frame, and
the entry says where ("Evaluated in ..., where the run is paused"). See
[Debug a script](debug-a-script.md).

## If something goes wrong

**Python is not available**: the Engine has no usable Python. See
[First launch: Python for scripting](../../install/first-launch.md#python-for-scripting).

**Setting up**: the project's Python environment is being made (about ten
seconds the first time); commands wait for it.

**Restart needed**: the project uses an environment the running Python
cannot move to. Restart the Engine. See
[Python and project environments](python-environment.md#changing-project).

**Slash commands are not Python**: an input starting with `/` is for the
**Agent LLM** session, not Python.

**Command interrupted (timeout)**: the command ran past its time limit and
was stopped.

## Related

- [The Variable Explorer](variable-explorer.md)
- [Write and run scripts](script-editor.md)
- [Python API reference](../../python/reference/index.md)
