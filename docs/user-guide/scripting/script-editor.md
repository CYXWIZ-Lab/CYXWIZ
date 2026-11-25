---
title: Write and run scripts
description: Create a Python script in the Script Editor, run it whole, by selection or by section, read its output, and save it.
type: how-to
area: scripting
audience: user
status: stable
since: 1.0.7
owner: scripting
last_reviewed: 2026-11-25
related:
  - debug-a-script.md
  - notebooks.md
  - python-console.md
  - variable-explorer.md
  - ../../main-window/preferences.md
---

Use this when you want to write Python that runs inside the Engine: load
and look at data, call `pycyxwiz`, or try code before it becomes a node.
The Script Editor is a tab of the main window, next to the CyxWiz Studio
canvas; open it from **View > Panels > Script Editor** if it is closed.

## Before you start

- A project is open. A new script goes into the project's `scripts` folder,
  and Python runs in the project's environment. See
  [Python and project environments](python-environment.md).
- The Engine has a Python it can use. Without one the editor still opens
  and saves files, and the status bar says **Python unavailable**.

## Create or open a script

1. Choose **File > New Script...** or **Script > New Script...** (Ctrl+N).
2. In **New script**, type a **Name**, pick the **Type** (**Python script
   (.py)** or **CyxWiz script (.cyx)**) and, if you want another place than
   the project's `scripts` folder, a **Folder**.
3. Select **Create and open**. The script opens in a new tab.

To open an existing file, choose **Open Script...** (Ctrl+O) or
double-click it in the Asset Browser. The editor opens `.py`, `.pyw`,
`.cyx` and `.ipynb` files as scripts or notebooks; it also opens other text
files for editing, without the run buttons. The **+** at the end of the tab
strip opens an empty `Untitled` tab.

A file over 4 MB opens read-only in pages of 512 lines; the status bar says
**Large file, read only** with the lines shown.

## Run a script

1. Write the script, for example:

   ```python
   import numpy as np
   import pycyxwiz as cyx

   values = np.arange(10, dtype=np.float32)
   print("mean", values.mean())
   print(cyx.linalg.eye(3))
   ```

2. Select **Run** at the right of the tab strip, press **F5**, or choose
   **Run > Run Script** in the editor's menu (also **Script > Run Script**).
3. If the script has unsaved changes, **Save Before Running?** asks first:
   **Save & Run** saves and runs it.
4. The status bar reads **Running** with the script's name and the time.
   What the script prints, its warnings and a traceback when it fails show in
   the Console's **Python REPL** session, as one entry per run.
5. To stop it, select **Stop** (Shift+F5). Python stops at its next line;
   a call that runs in C (a long NumPy operation) finishes first.

Only one script or command runs at a time in the Engine's Python. While a
notebook cell or a Console command runs, **Run** is greyed out with
"Another script is running".

F5 runs the script only when the Script Editor has the focus; on the canvas
it starts training.

## Run part of a script

- **Run selection** (F9): runs the selected lines. Common indentation is
  removed first, so you can run the body of a function.
- **Run section** (Ctrl+Enter): runs the lines between two `%%` lines around
  the cursor, as in MATLAB. A file without `%%` lines is one section.

Both run in the same namespace as the whole script, so a variable one run
makes is there for the next. They are in the **Run** menu, and in the
**...** menu when the panel is narrow.

## Edit faster

- **Completion**: Ctrl+Space, or type; Enter or Tab inserts. With the list
  open, Ctrl+Space shows or hides the details.
- **Signature**: Ctrl+Shift+Space, and on its own when you type `(` or `,`.
- **Go to definition**: F12 or Ctrl+click.
- **Problems**: pyflakes checks the file as you type. The count in the status
  bar opens the **Problems** list; click a row to go to the line.
- **Find** (Ctrl+F) and **Replace** (Ctrl+H) in the file; **Find in Files**
  (Ctrl+Shift+F) across the project.
- **Several cursors**: Alt+click, Ctrl+Alt+Up / Down, Ctrl+Shift+L (every
  occurrence of the word).
- **Fold**: Ctrl+Shift+[ and Ctrl+Shift+].

Completion, signatures, definitions and problems come from Jedi and pyflakes
bundled with the Engine; they run in the background and need Python
running. Every key is listed in [Keyboard shortcuts](../../reference/shortcuts.md).

## The status bar

From left to right: the state (**Ready**, **Running**, **Debugging: paused**),
**Modified** or **Saved**, the cursor position (**Ln, Col**), and at the
right **Spaces**, the encoding (**UTF-8** or **UTF-8 with BOM**), the line
endings (**LF** or **CRLF**), the Python in use, the problem count and
**Sandbox on** or **Sandbox off**. Click **Spaces**, the encoding or the line
endings to change them for this file.

## Save

- **Save** (Ctrl+S) and **Save As** (Ctrl+Shift+S) in the editor's **File**
  menu, or the main **File** menu when the editor has the focus. **File >
  Save All** (Ctrl+Alt+S) saves every open script and the graph.
- The file is written as one step (a crash during the write does not leave
  half a file), as UTF-8, with the line endings it had.
- **Preferences > Script Editor** sets the tab size, whitespace, minimap,
  word wrap, line endings of new files, trimming and the final newline. See
  [Preferences](../../usage/preferences.md#script-editor).

If the file changes on disk while you have unsaved edits, the editor keeps
your edits and asks: **Reload from disk**, **Keep my edits**, or open the
version on disk in a read-only tab. Without unsaved edits it reloads.

## Make a node from a script

**Run > Create node from script...** turns the functions of a saved `.py`
file into nodes in the project's `nodes` folder. It is greyed out until the
script is a saved `.py` file and a project is open. See
[Python nodes](../../sdk/guides/python-nodes.md).

## About the sandbox

Scripts run with your user's rights: they can read and write your files and
start programs. In this build, **Security > Enable Sandbox** does not
restrict scripts run with **Run** (F5), a selection or a section. Run only
scripts you trust.

## Related

- [Debug a script](debug-a-script.md)
- [Notebooks](notebooks.md)
- [Use the Python console](python-console.md)
- [Plots from Python](../evaluation/plots-from-python.md)
