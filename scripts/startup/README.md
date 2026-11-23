# Startup scripts

A startup script is a Python file the Engine runs in each Python session
before the session's first command or script, and again after **Restart**.
It runs in the session's namespace, so what it imports or defines is there
in the Python console and in every script you run.

## Set them

**Edit > Preferences > Python > When Python starts > Startup scripts**:

- **Add a script...** picks a `.py` (or `.cyx`) file. Add as many as you
  need; they run in the order shown.
- The arrow moves a script up the order; the cross removes it from the list
  (the file stays where it is).
- A script that is no longer there is shown in the warning colour; sessions
  start without it.

The list is saved with the Engine settings and is the same for every
project. A change applies to the next session (use **Restart** in the
Python console to start one now).

The two switches below it, **import numpy as np** and **import pycyxwiz**,
run before the scripts, so a script can use `np` and `pycyxwiz` when they
are on.

## What you see

- What a script prints goes to the Console log, as
  `Startup script printed: ...`.
- A script that is missing, or stops with an error, is reported in the
  Console log; the session starts anyway and the next script still runs.

## Example

[`example_startup.py`](example_startup.py) defines two helpers,
`read_json(path)` and `describe(values)`, and prints one line. Add it, open
the Python console, and type `describe([1, 2, 3])`.

## Tips

- Keep scripts quick: the session's first command waits for them.
- Add only scripts you trust: they run with your user's rights.
- Try a script in the Script Editor (F5) before adding it.
