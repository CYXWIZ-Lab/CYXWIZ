# Cell Notebooks (.cyx Format)

CyxWiz uses a cell-based notebook format (`.cyx`) similar to Jupyter notebooks, optimized for the desktop application.

## File Format

A `.cyx` file contains multiple cells separated by cell markers:

```python
%%code
# This is a code cell
import pycyxwiz as cx
print("Hello!")

%%code
# Another code cell
result = 2 + 2

%%markdown
# Documentation
This is a markdown cell for notes.

%%raw
Plain text, no processing.
```

## Cell Types

### Code Cells (`%%code`)

Executable Python code:

```python
%%code
import pycyxwiz as cx
import numpy as np

# Create tensor
data = np.random.randn(10, 5).astype(np.float32)
tensor = cx.Tensor.from_numpy(data)

# Operations
result = tensor + tensor
print("Result shape:", result.shape())
```

**Features:**
- Syntax highlighting for Python
- Auto-completion
- Execution counter `[1]`, `[2]`, etc.
- Output display below cell

### Markdown Cells (`%%markdown`)

Documentation and notes:

```python
%%markdown
# Experiment Notes

## Objective
Test different learning rates.

## Results
- LR 0.001: Best accuracy
- LR 0.01: Faster but less stable
```

**Supported Markdown:**
- Headers (`#`, `##`, `###`)
- Bold/Italic (`**bold**`, `*italic*`)
- Lists (bullet and numbered)
- Code blocks (syntax highlighted)
- Links and images

### Raw Cells (`%%raw`)

Plain text without processing:

```python
%%raw
This text is displayed as-is.
No markdown rendering.
No code execution.
```

## Cell Structure

Each cell has:

| Component | Description |
|-----------|-------------|
| **ID** | Unique identifier (e.g., `cell-a1b2c3d4`) |
| **Type** | `code`, `markdown`, or `raw` |
| **Source** | Cell content (code or text) |
| **Outputs** | Execution results (code cells only) |
| **State** | Idle, Queued, Running, Success, Error |
| **Execution Count** | Order of execution (`[1]`, `[2]`, etc.) |

## Cell Operations

### Running Cells

| Action | Shortcut | Description |
|--------|----------|-------------|
| Run Cell | `Shift+Enter` | Execute current cell, move to next |
| Run Cell (Stay) | `Ctrl+Enter` | Execute current cell, stay in place |
| Run All | `F5` | Execute all cells in order |
| Run Above | `Ctrl+Shift+Enter` | Run all cells above current |
| Run Below | - | Run current and all below |

### Editing Cells

| Action | Shortcut | Description |
|--------|----------|-------------|
| Insert Above | `Ctrl+Shift+A` | New cell above current |
| Insert Below | `Ctrl+Shift+B` | New cell below current |
| Delete Cell | `Ctrl+Shift+D` | Remove current cell |
| Cut Cell | `Ctrl+Shift+X` | Cut cell to clipboard |
| Copy Cell | `Ctrl+Shift+C` | Copy cell to clipboard |
| Paste Cell | `Ctrl+Shift+V` | Paste cell from clipboard |
| Duplicate | `Ctrl+D` | Duplicate current cell |

### Reorganizing Cells

| Action | Shortcut | Description |
|--------|----------|-------------|
| Move Up | `Ctrl+Shift+Up` | Move cell up |
| Move Down | `Ctrl+Shift+Down` | Move cell down |
| Merge Up | `Ctrl+Shift+M` | Merge with cell above |
| Merge Down | - | Merge with cell below |
| Split Cell | `Ctrl+Shift+S` | Split at cursor position |

### Changing Cell Type

| Action | Shortcut | Description |
|--------|----------|-------------|
| To Code | `Ctrl+Shift+1` | Convert to code cell |
| To Markdown | `Ctrl+Shift+2` | Convert to markdown cell |
| To Raw | `Ctrl+Shift+3` | Convert to raw cell |

## Execution States

### State Indicators

| Display | State | Description |
|---------|-------|-------------|
| `[ ]` | Idle | Cell has not been run |
| `[*]` | Running | Currently executing |
| `[1]` | Success | Completed (execution number) |
| `[!]` | Error | Failed with exception |

### Execution Flow

```
Idle → Queued → Running → Success/Error
```

When you click "Run All":
1. All cells are marked **Queued**
2. Cells execute sequentially
3. Each cell transitions: **Running** → **Success/Error**
4. Execution stops on error (unless configured otherwise)

## Output Types

Code cells can produce various outputs:

### Text Output

```python
%%code
print("Hello, World!")
# Output: Hello, World!
```

### Return Values

```python
%%code
2 + 2
# Output: 4
```

### Error Output

```python
%%code
1 / 0
# Output: ZeroDivisionError: division by zero
```

### Plot Output

```python
%%code
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [1, 4, 9])
plt.show()
# Output: [Image of plot]
```

### Rich Output

- Tables (pandas DataFrames)
- Images (PNG/JPEG)
- HTML content

## Cell Settings

### Per-Cell Options

Right-click a cell for options:

- **Collapse Input** - Hide the code/text
- **Collapse Output** - Hide the output
- **Toggle Breakpoint** - Add/remove debugging breakpoint
- **Clear Output** - Remove execution output
- **Lock Cell** - Prevent editing

### Breakpoints

Add breakpoints for debugging:

```python
%%code
x = 10
# [Breakpoint on line 2]
y = x * 2  # Execution pauses here
print(y)
```

See [Debugging](debugging.md) for details.

## Best Practices

### Organize Your Notebook

```python
%%markdown
# Project Title

%%code
# Imports
import pycyxwiz as cx

%%markdown
## Data Loading

%%code
data = load_data()

%%markdown
## Processing

%%code
processed = process(data)
```

### Keep Cells Focused

Each cell should do one thing:

```python
%%code
# Good: Single purpose
def load_data():
    return np.load('data.npy')

%%code
# Good: Single purpose
data = load_data()
print(data.shape)
```

### Use Descriptive Markdown

```python
%%markdown
## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 100 | May need more |
| LR | 0.001 | Using Adam |
| Batch | 32 | GPU memory limited |
```

### Handle Long-Running Cells

```python
%%code
# Add progress indicators for long operations
for i in range(epochs):
    train_epoch()
    if i % 10 == 0:
        print(f"Epoch {i}/{epochs} complete")
```

## File Format Details

### Internal Structure

`.cyx` files are plain text with cell markers:

```
%%code
# Cell 1 content
import something

%%code
# Cell 2 content
do_something()

%%markdown
# Notes
Some documentation.
```

### Metadata (Optional)

The file can start with a JSON metadata block:

```
%%metadata
{
  "title": "My Notebook",
  "created": "2025-01-15",
  "version": "1.0"
}

%%code
# First cell
```

## Importing/Exporting

### Export to Python

**Script > Export as Python (.py)**

Converts cells to a single Python file:

```python
# Exported from: my_notebook.cyx

# Cell 1
import pycyxwiz as cx

# Cell 2
data = load_data()
```

### Export to Jupyter

**Script > Export as Jupyter (.ipynb)**

Creates a Jupyter notebook compatible file.

### Import from Jupyter

**File > Import > Jupyter Notebook**

Converts `.ipynb` to `.cyx` format.

---

**Next**: [Command Window](command-window.md) | [Debugging](debugging.md)
