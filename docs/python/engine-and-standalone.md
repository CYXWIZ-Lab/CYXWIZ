---
title: Run pycyxwiz in the Engine or standalone
description: What the Engine's Python session prepares for pycyxwiz, the short names it defines, and how to import pycyxwiz from a plain Python 3.12 outside the Engine.
type: how-to
area: python
audience: user
status: experimental
since: 1.0.8
owner: python
last_reviewed: 2026-11-25
related:
  - index.md
  - ../user-guide/scripting/python-environment.md
  - ../user-guide/scripting/startup-scripts.md
  - ../sdk/guides/python-modules.md
  - ../user-guide/evaluation/plots-from-python.md
---

Use this to know what is already set up when your code runs inside the
Engine, and what to set up when it runs in a Python of your own.

> Checked against pycyxwiz 0.2.0.

## Inside the Engine

The Python console, the Script Editor, notebooks, `.cyx` documents and
Python nodes all run in the one Python the Engine embeds (3.12 in the
release packages), in the project's environment; see
[Python and project environments](../user-guide/scripting/python-environment.md).
pycyxwiz comes with the Engine: nothing to install.

### What the session prepares

Before the session's first command, the Engine runs, in this order:

1. `import numpy as np`, when **File > Preferences... > Python > When
   Python starts > Imports > import numpy as np** is on (the default).
2. `import pycyxwiz`, with `cyx` as a short name for it and the
   MATLAB-style functions under their own names, when **import pycyxwiz**
   is on (the default):

   | Names in the session | Same as |
   | --- | --- |
   | `svd`, `eig`, `qr`, `chol`, `lu`, `det`, `rank`, `trace`, `norm`, `cond`, `inv`, `transpose`, `solve`, `lstsq`, `matmul`, `eye`, `zeros`, `ones` | `pycyxwiz.linalg.*` |
   | `fft`, `ifft`, `conv`, `conv2`, `spectrogram`, `lowpass`, `highpass`, `bandpass`, `findpeaks`, `sine`, `square`, `noise` | `pycyxwiz.signal.*` (no `filter`: it would hide Python's own) |
   | `kmeans`, `dbscan`, `gmm`, `pca`, `tsne`, `silhouette`, `confusion_matrix`, `roc` | `pycyxwiz.stats.*` |
   | `acf`, `pacf`, `decompose`, `stationarity`, `arima`, `diff`, `rolling_mean`, `rolling_std` | `pycyxwiz.timeseries.*` |
   | `printmat(matrix, precision=4)`, `pm` | Prints a matrix in rows, MATLAB style |

3. DuckDB helpers when the `duckdb` package is installed: `db` (an
   in-memory connection), `sql(query)`, `read_csv`, `read_parquet`,
   `read_json`.
4. Polars helpers when `polars` is installed: `pl`, `df(...)`, `lf(...)`,
   `pl_csv`, `pl_parquet`, `pl_json`, `pl_excel`, `scan_csv`,
   `scan_parquet`, `col`, `lit`, `when`.
5. Your startup scripts, in order
   ([Startup scripts](../user-guide/scripting/startup-scripts.md)).

So in the console this works without any import:

<!-- not run: uses the names the Engine's session defines -->
```python
x = solve([[3.0, 2.0], [1.0, 4.0]], [[7.0], [9.0]])
pm(x)
t = cyx.Tensor.ones([2, 2])
```

The short names are a convenience of the console; in a script you keep or
share, import what you use (`import pycyxwiz as cx`), so it also runs
outside the Engine. A name you assign, such as `norm = 3`, hides the
session's function of that name until the session restarts.

### The backend is the Engine's

- **The device is the Engine's.** pycyxwiz uses the backend the Engine
  started; `cx.initialize()` returns `True` at once. Tensors are on the
  device the Engine computes on.
- **Do not call `cx.shutdown()`** in the Engine: it frees the device's
  cached memory and marks the backend the Engine shares as stopped.
- **Do not switch devices from a script** (`Device.set_active`): tensors
  the Engine holds stay on the old backend
  ([Tensors and devices](tensors-and-devices.md#devices)).
- **Log output** of pycyxwiz (for example `Sequential.summary()`, a failed
  `load`) goes to **Console > Logs**, not to the script's output.
- **RL metrics** a script sends with `cx.rl_update_metric(name, value)`
  reach the RL Training Dashboard: the Engine sets the callback when Python
  starts.

### Plots: `cyxwiz`, not `pycyxwiz`

Plot windows come from a second module the Engine ships, `cyxwiz`
(without "py"): `import cyxwiz` then `cyxwiz.scatter(...)`,
`cyxwiz.hist(...)`; `plt.show()` also shows matplotlib figures in the
Engine. See [Plots from Python](../user-guide/evaluation/plots-from-python.md).
Give the two modules different short names when you use both, for example
`import pycyxwiz as cx` and `import cyxwiz as plots`.

## Outside the Engine

pycyxwiz is a compiled module for **Python 3.12** on 64-bit Windows, Linux
and macOS; other Python versions cannot import it. You need the folder that
holds it and, on Windows, the folders of the DLLs it loads:

| What | Installed Engine | Source build |
| --- | --- | --- |
| The module | `pycyxwiz.pyd` (Windows) or `pycyxwiz.cpython-312-*.so` in the Engine's base folder | `build\lib\Release\pycyxwiz.cp312-win_amd64.pyd` |
| Its libraries | The same base folder, and `arrayfire\bin` in it | `build\bin\Release`, ArrayFire's `lib` folder, and the CUDA `bin` folder for a CUDA build |
| A Python 3.12 | The one the Engine ships in its `python` folder, or your own | Your own |

On Windows, Python does not look for a module's DLLs on `PATH`: name the
folders before the import.

<!-- not run: the paths depend on your installation -->
```python
import os
import sys

BASE = r"C:\path\to\the\Engine\base\folder"           # the folder that holds pycyxwiz.pyd
folders = [BASE, os.path.join(BASE, "arrayfire", "bin")]
for folder in folders:
    os.add_dll_directory(folder)
os.environ["PATH"] = os.pathsep.join(folders) + os.pathsep + os.environ["PATH"]   # ArrayFire loads its backends later
sys.path.insert(0, BASE)

import pycyxwiz as cx
cx.initialize()
print(cx.get_version())
```

Where the base folder of an installed Engine is, the Linux and macOS
details, and the error each missing piece gives, are in
[Write a module in Python](../sdk/guides/python-modules.md#1-get-pycyxwiz).

Standalone, you start and stop the backend yourself: call `cx.initialize()`
first (it picks CUDA, then OpenCL, then the CPU) and, optionally,
`cx.shutdown()` at the end. Without `initialize()` ArrayFire still runs on
its own default device, but the CyxWiz backend is not set up; call it.
NumPy must be installed in that Python for `from_numpy` and `to_numpy`.

The examples in `examples/python` take the folder of pycyxwiz on their
command line; see [Examples](examples.md).

## Related

- [Script with Python](../user-guide/scripting/index.md): the Script
  Editor, the console and notebooks.
- [Startup scripts](../user-guide/scripting/startup-scripts.md): your own
  imports at the start of every session.
