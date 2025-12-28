# Basic Scripting Tutorial

Learn to write and execute Python scripts in the CyxWiz Engine using the Script Editor and Console.

## Overview

CyxWiz provides two ways to run Python code:

1. **Console** - Interactive REPL for quick commands
2. **Script Editor** - Write and save `.cyx` or `.py` files

## Part 1: Using the Console

### Opening the Console

1. Go to **View > Console** (or press `` Ctrl+` ``)
2. The Console panel appears at the bottom of the screen

### Running Commands

Type Python commands at the `>>>` prompt:

```python
>>> 2 + 2
4

>>> import pycyxwiz as cx
>>> cx.get_version()
'0.1.0'

>>> print("Hello, CyxWiz!")
Hello, CyxWiz!
```

### Console Features

| Feature | Usage |
|---------|-------|
| **History** | Up/Down arrows to navigate previous commands |
| **Auto-complete** | Tab to complete variable/function names |
| **Multi-line** | Shift+Enter for continuation |
| **Clear** | Type `clear` or `Ctrl+L` |

### Checking GPU Availability

```python
>>> import pycyxwiz as cx
>>> print("CUDA:", cx.cuda_available())
CUDA: True
>>> print("OpenCL:", cx.opencl_available())
OpenCL: False

>>> devices = cx.get_available_devices()
>>> for d in devices:
...     print(f"{d.name}: {d.memory_total // (1024**3)} GB")
NVIDIA RTX 3080: 10 GB
```

## Part 2: Creating a Script

### Opening the Script Editor

1. Go to **File > New Script** (or press `Ctrl+N`)
2. Choose script type:
   - **Python (.py)** - Standard Python file
   - **CyxWiz Script (.cyx)** - Notebook-style cells

### The .cyx Format

CyxWiz scripts (`.cyx`) use a cell-based format similar to Jupyter notebooks:

```python
%%code
# This is a code cell
import pycyxwiz as cx
print("Hello from cell 1")

%%code
# This is another code cell
data = [1, 2, 3, 4, 5]
print(sum(data))

%%markdown
# Documentation Cell
This is markdown text that won't be executed.
```

### Cell Types

| Marker | Type | Description |
|--------|------|-------------|
| `%%code` | Code | Executable Python code |
| `%%markdown` | Markdown | Documentation (not executed) |
| `%%raw` | Raw | Plain text (not rendered) |

## Part 3: Your First Script

Create a new script and write:

```python
%%code
# Cell 1: Import and setup
import pycyxwiz as cx
import numpy as np

print("CyxWiz Version:", cx.get_version())
print("NumPy Version:", np.__version__)

%%code
# Cell 2: Create a tensor
data = np.random.randn(3, 4).astype(np.float32)
tensor = cx.Tensor.from_numpy(data)

print("Tensor shape:", tensor.shape())
print("Tensor dtype:", tensor.get_data_type())

%%code
# Cell 3: Basic operations
t1 = cx.Tensor.random([4, 4])
t2 = cx.Tensor.random([4, 4])

result = t1 + t2  # Element-wise addition
print("Sum result shape:", result.shape())

%%code
# Cell 4: Neural network layer
dense = cx.Dense(10, 5)  # 10 inputs, 5 outputs
input_tensor = cx.Tensor.random([2, 10])  # Batch of 2, 10 features

output = dense.forward(input_tensor)
print("Output shape:", output.shape())  # [2, 5]
```

### Running the Script

**Run All Cells:**
- Press `F5` or go to **Script > Run All**

**Run Single Cell:**
- Place cursor in cell
- Press `Shift+Enter` or **Script > Run Cell**

**Run Cells Above/Below:**
- **Script > Run Above** - Run all cells above current
- **Script > Run Below** - Run current and all below

## Part 4: Execution States

Each cell shows its execution state:

| Display | State | Meaning |
|---------|-------|---------|
| `[ ]` | Idle | Not yet run |
| `[*]` | Running | Currently executing |
| `[1]` | Success | Completed (execution number) |
| `[!]` | Error | Failed with exception |

### Viewing Output

Cell output appears below each cell:
- Text output (print statements)
- Error messages with tracebacks
- Plots (matplotlib figures render inline)

## Part 5: Working with Plots

```python
%%code
import matplotlib.pyplot as plt
import numpy as np

# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot
plt.figure(figsize=(8, 4))
plt.plot(x, y, 'b-', label='sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine Wave')
plt.legend()
plt.grid(True)
plt.show()  # Plot appears inline
```

The plot renders directly below the cell!

## Part 6: Training a Simple Model

```python
%%code
# Setup
import pycyxwiz as cx
import numpy as np

# Create synthetic data
np.random.seed(42)
X = np.random.randn(100, 4).astype(np.float32)
y = np.random.randint(0, 3, 100)  # 3 classes

# Convert to tensors
X_tensor = cx.Tensor.from_numpy(X)

%%code
# Build model
layer1 = cx.Dense(4, 16)
relu = cx.ReLU()
layer2 = cx.Dense(16, 3)
softmax = cx.Softmax()

# Forward pass
def forward(x):
    x = layer1.forward(x)
    x = relu.forward(x)
    x = layer2.forward(x)
    x = softmax.forward(x)
    return x

%%code
# Test forward pass
output = forward(X_tensor)
print("Output shape:", output.shape())  # [100, 3]
print("First prediction:", output.to_numpy()[0])
```

## Part 7: Saving and Loading Scripts

### Save
- Press `Ctrl+S` or **File > Save**
- First save prompts for filename and location

### Save As
- **File > Save As** to save with new name

### Open Existing
- **File > Open** or double-click in Asset Browser

## Script Editor Features

| Feature | Shortcut | Description |
|---------|----------|-------------|
| Run All | `F5` | Execute all cells |
| Run Cell | `Shift+Enter` | Execute current cell |
| New Cell | `Ctrl+Enter` | Insert cell below |
| Delete Cell | `Ctrl+Shift+D` | Remove current cell |
| Move Up | `Ctrl+Shift+Up` | Move cell up |
| Move Down | `Ctrl+Shift+Down` | Move cell down |
| Split Cell | `Ctrl+Shift+S` | Split at cursor |
| Merge Cells | `Ctrl+Shift+M` | Merge with below |

## What You Learned

- Using the Console for interactive Python
- Creating `.cyx` notebook-style scripts
- Running cells individually or all at once
- Creating plots that render inline
- Basic model building with pycyxwiz

## Next Steps

- Learn about [debugging](../engine/scripting/debugging.md) with breakpoints
- Explore the [pycyxwiz API](../backend/python/index.md) for all available functions
- Try the [Complete Workflow Tutorial](workflow-tutorial.md)

---

**Next**: [Complete Workflow Tutorial](workflow-tutorial.md)
