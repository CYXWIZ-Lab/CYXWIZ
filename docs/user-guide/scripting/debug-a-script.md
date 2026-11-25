---
title: Debug a script
description: Stop a Python script or notebook cell at a breakpoint, step through it, and read its variables, watches and call stack.
type: how-to
area: scripting
audience: user
status: stable
since: 1.0.7
owner: scripting
last_reviewed: 2026-11-25
related:
  - script-editor.md
  - notebooks.md
  - python-console.md
  - ../../reference/shortcuts.md
---

Use this when a script does not do what you expect and you want to stop it
on a line and look at its values. The debugger is part of the Script
Editor; it works on scripts and on notebook cells. (To debug a graph, use
the [Studio Debugger](../training/studio-debugger.md).)

## Before you start

- The script is open in the Script Editor, and nothing else runs in the
  Engine's Python (no script, cell or Console command).

## Set breakpoints

1. In the Script Editor, click the gutter left of a line number. A dot marks
   the breakpoint. Click it again to remove it.
2. Optional: right-click the dot for its settings:
   - **Enabled**: off keeps the breakpoint but does not stop there.
   - **Stop only when**: a Python expression, for example `i == 40`; empty
     stops every time.
   - **Stop from this hit on**: skip the first hits (0 stops on every hit).

   Select **Apply** (or press Enter), **Cancel**, or **Remove**.

While a debug run is paused, F9 toggles the breakpoint on the cursor's line
(outside debugging F9 runs the selection).

## Start and step

1. Select **Debug** at the right of the tab strip, or press **F10**. In a
   notebook, select a code cell and select **Debug cell**.
2. The script runs to the first breakpoint. The status bar reads
   **Debugging: paused**, the line is highlighted, and the debug bar and the
   side area appear.
3. Step with the buttons of the debug bar or the keys:

| Action | Key |
| --- | --- |
| **Continue** to the next breakpoint | F5 |
| **Pause** at the next line | (button) |
| **Step over** the line | F10 |
| **Step into** the call | F11 |
| **Step out** of the function | Shift+F11 |
| **Stop** debugging | Shift+F5 |

The step buttons work only while the run is paused.

## Read the values

The side area has four parts:

- **VARIABLES**: the selected frame's values; switch between **Locals** and
  **Globals**. While the run goes on, it says "Running... values show when
  the run pauses."
- **WATCH**: type an expression in **+ Add an expression** and press Enter.
  It is evaluated in the selected frame at each pause. Right-click a watch
  to remove it.
- **CALL STACK**: the frames of the paused run. Click one to go to its line;
  Variables and Watch follow it.
- **BREAKPOINTS**: every breakpoint, with its tick (**Enabled**) and a link
  to its line; right-click for its condition and hit count. **Stop on
  uncaught errors** pauses where an exception is raised and not caught; a
  change takes effect at the next **Debug**.

To run any Python in the paused frame, type it in the Console's **Python
REPL** session and press Enter. The entry says "Evaluated in ... where the
run is paused". Use it to call a function or change a value before you
continue.

## Why the run stopped

The state tells you why it paused: **Paused at a breakpoint**, **Paused
after a step**, or **Stopped on an error** (with Stop on uncaught errors).
When the run ends, the status bar reads **Ready** again and the Console's
Python REPL session shows its output.

## If something goes wrong

**Debug: another run holds the interpreter**: a script, a notebook cell or a
Console command is running. Wait for it or stop it, then select **Debug**
again.

**The breakpoint is not hit**: the line never runs, the breakpoint is
disabled, or its condition is false. Check the BREAKPOINTS list.

## Related

- [Write and run scripts](script-editor.md)
- [Notebooks](notebooks.md)
- [Keyboard shortcuts](../../reference/shortcuts.md)
