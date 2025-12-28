# Python Scripting

CyxWiz Engine includes a powerful embedded Python scripting system for interactive data analysis, model building, and training automation.

## Overview

The scripting system provides:

- **Embedded Python Interpreter** - Full Python 3.x environment
- **Notebook-Style Editor** - Cell-based `.cyx` script format
- **Interactive Console** - REPL for quick commands
- **Plot Integration** - Matplotlib renders inline
- **Debugging** - Breakpoints and variable inspection
- **MATLAB Compatibility** - Familiar function aliases

## Quick Start

### Console (Interactive)

Press `` Ctrl+` `` to open the Console:

```python
>>> import pycyxwiz as cx
>>> t = cx.Tensor.random([3, 3])
>>> print(t.shape())
[3, 3]
```

### Script Editor

Press `Ctrl+N` to create a new script:

```python
%%code
import pycyxwiz as cx
import numpy as np

# Create data
data = np.random.randn(100, 10).astype(np.float32)
tensor = cx.Tensor.from_numpy(data)
print("Shape:", tensor.shape())
```

## Documentation

| Topic | Description |
|-------|-------------|
| [Cell Notebooks](cell-notebook.md) | The `.cyx` file format and cell operations |
| [Command Window](command-window.md) | Interactive REPL usage |
| [Debugging](debugging.md) | Breakpoints and stepping through code |
| [MATLAB Compatibility](matlab-compatibility.md) | MATLAB-style function aliases |

## Key Features

### Cell-Based Notebooks

Write code in discrete cells:

```python
%%code
# Cell 1: Setup
import pycyxwiz as cx

%%code
# Cell 2: Processing
result = cx.linalg.svd([[1, 2], [3, 4]])
print(result)

%%markdown
## Documentation
This cell contains markdown text.
```

### Inline Plotting

Matplotlib figures render directly in the editor:

```python
%%code
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.title('Sine Wave')
plt.show()  # Renders below the cell
```

### Async Execution

Scripts run asynchronously - the UI stays responsive:

- Progress indicator shows running cells
- Cancel with **Script > Stop** or `Ctrl+C`
- Output streams in real-time

### Sandboxed Execution

Optional security features:

- Memory limits
- Execution timeouts
- Blocked imports
- File access restrictions

## Available Modules

### Built-in

| Module | Description |
|--------|-------------|
| `pycyxwiz` | CyxWiz backend bindings |
| `numpy` | Numerical computing |
| `matplotlib` | Plotting and visualization |
| `pandas` | Data manipulation (if installed) |
| `scipy` | Scientific computing (if installed) |

### pycyxwiz Submodules

```python
import pycyxwiz as cx

cx.linalg     # Linear algebra (SVD, QR, LU, etc.)
cx.signal     # Signal processing (FFT, filters)
cx.stats      # Statistics and clustering
cx.timeseries # Time series analysis
```

See [pycyxwiz API Reference](../../backend/python/index.md) for complete documentation.

## File Types

| Extension | Description |
|-----------|-------------|
| `.cyx` | CyxWiz notebook script (cell-based) |
| `.py` | Standard Python script |

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| New Script | `Ctrl+N` |
| Save | `Ctrl+S` |
| Run All | `F5` |
| Run Cell | `Shift+Enter` |
| Stop Execution | `Ctrl+C` |
| Toggle Console | `` Ctrl+` `` |
| Insert Cell Below | `Ctrl+Enter` |
| Delete Cell | `Ctrl+Shift+D` |

## Script Execution States

| State | Display | Meaning |
|-------|---------|---------|
| Idle | `[ ]` | Not executed |
| Queued | `[*]` | Waiting to run |
| Running | `[*]` | Currently executing |
| Success | `[1]` | Completed (shows execution count) |
| Error | `[!]` | Failed with exception |

## Best Practices

### Organize with Cells

```python
%%code
# Cell 1: Imports
import pycyxwiz as cx
import numpy as np

%%code
# Cell 2: Configuration
CONFIG = {
    'learning_rate': 0.001,
    'epochs': 100
}

%%code
# Cell 3: Data Loading
data = load_data()

%%code
# Cell 4: Training
train(data, CONFIG)
```

### Use Markdown for Documentation

```python
%%markdown
# Experiment: Learning Rate Sweep

This notebook tests different learning rates.

## Setup
- Dataset: MNIST
- Model: 3-layer MLP
- Optimizer: Adam
```

### Handle Errors Gracefully

```python
%%code
try:
    result = risky_operation()
except Exception as e:
    print(f"Error: {e}")
    # Fallback logic
```

## Integration with Node Editor

Generated code from the Node Editor can be pasted into scripts:

1. Build model in Node Editor
2. **Nodes > Generate Code > PyCyxWiz**
3. Paste into a script cell
4. Add training loop and evaluation

## Related Documentation

- [pycyxwiz API](../../backend/python/index.md)
- [Training Dashboard](../panels/training-dashboard.md)
- [Console Panel](../panels/console.md)

---

**Next**: [Cell Notebooks](cell-notebook.md)
