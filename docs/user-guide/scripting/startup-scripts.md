---
title: Startup scripts
description: Run your own Python files at the start of every Python session, so their imports and helpers are there in the console and in scripts.
type: how-to
area: scripting
audience: user
status: experimental
since: 1.0.8
owner: scripting
last_reviewed: 2026-11-25
related:
  - python-environment.md
  - python-console.md
  - ../../usage/preferences.md
---

Use this when you type the same imports or helper functions in every
session. A startup script is a Python file the Engine runs in each Python
session before the session's first command or script, and again after
**Restart**. It runs in the session's namespace, so what it imports or
defines is there in the Python console and in every script you run.

## Before you start

- Write and try the script in the Script Editor first (F5).
- Keep it quick: the session's first command waits for it.
- Add only scripts you trust: they run with your user's rights.

## Add a startup script

1. Choose **File > Preferences... > Python**.
2. Under **When Python starts > Startup scripts**, select **Add a
   script...** and pick a `.py` (or `.cyx`) file. Add as many as you need;
   they run in the order shown.
3. Optional: the arrow next to a script runs it before the one above; the
   cross removes it from the list (the file stays where it is).
4. Optional: to use the scripts now, select **Restart** in the Python
   console. Otherwise they run from the next session.

The list is saved at once with the Engine settings and is the same for
every project.

The two switches under **Imports**, **import numpy as np** and **import
pycyxwiz** (with its short names: `cyx`, `svd`, `eig`, `zeros`, `ones` and
the other MATLAB-style functions), run before the scripts, so a script can
use `np` and `pycyxwiz` when they are on. Both are on by default.

## Example

This script defines two helpers and prints one line. Save it as
`startup_helpers.py`, add it, open the Python console and type
`describe([1, 2, 3])`:

```python
import json
import math
from pathlib import Path


def read_json(path):
    """Read a JSON file into Python values."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def describe(values):
    """Count, mean and standard deviation of a list of numbers."""
    n = len(values)
    if n == 0:
        return {"count": 0, "mean": None, "sd": None}
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    return {"count": n, "mean": mean, "sd": sd}


print("Startup helpers ready: read_json(path), describe(values)")
```

The same file is `scripts/startup/example_startup.py` in the Engine's
source repository.

## What you see

- What a script prints goes to the Console log, as `Startup script printed:
  ...`, and "Ran the startup script ..." at Info level.
- A script that is not there is shown in the warning colour in the list; the
  session starts without it, and the log says "The startup script ... is not
  there; the session starts without it".
- A script that stops with an error is reported in the log ("The startup
  script ... stopped: ..."); the session starts anyway and the next script
  still runs.

## Related

- [Python and project environments](python-environment.md)
- [Use the Python console](python-console.md)
- [Preferences: Python](../../usage/preferences.md#python)
