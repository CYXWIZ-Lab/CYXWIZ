# Command Window

The Command Window (Console) provides an interactive Python REPL for quick commands and experimentation.

## Overview

The Command Window is a MATLAB-style interactive command interface:

- Execute Python commands immediately
- View output and errors inline
- Navigate command history
- Auto-complete variables and functions

## Opening the Command Window

- **Menu**: View > Console
- **Shortcut**: `` Ctrl+` ``
- **Panel**: Appears at the bottom of the workspace

## Basic Usage

### Executing Commands

Type at the `>>>` prompt and press Enter:

```python
>>> 2 + 2
4

>>> import pycyxwiz as cx
>>> cx.get_version()
'0.1.0'

>>> print("Hello!")
Hello!
```

### Multi-Line Input

Use `Shift+Enter` for continuation:

```python
>>> def greet(name):
...     return f"Hello, {name}!"
...
>>> greet("CyxWiz")
'Hello, CyxWiz!'
```

Or use triple quotes:

```python
>>> """
... Multi-line
... string
... """
'\nMulti-line\nstring\n'
```

## Features

### Command History

| Action | Key |
|--------|-----|
| Previous command | `Up Arrow` |
| Next command | `Down Arrow` |
| Search history | `Ctrl+R` |
| Clear history | `Ctrl+Shift+H` |

History persists across sessions.

### Auto-Completion

Press `Tab` to complete:

```python
>>> import pycyxwiz as cx
>>> cx.lin<Tab>
cx.linalg

>>> cx.linalg.sv<Tab>
cx.linalg.svd
```

Completion works for:
- Module names
- Function names
- Variable names
- Object attributes
- File paths (in strings)

### Inline Help

Use `?` for quick help:

```python
>>> cx.linalg.svd?
Signature: cx.linalg.svd(A, full_matrices=False)
Docstring: Singular Value Decomposition: U, S, Vt = svd(A)
...
```

Or `help()`:

```python
>>> help(cx.linalg.svd)
```

## Console Commands

### Built-in Commands

| Command | Description |
|---------|-------------|
| `clear` | Clear the console output |
| `reset` | Reset the Python environment |
| `who` | List defined variables |
| `whos` | List variables with details |
| `history` | Show command history |

### Examples

```python
>>> x = 10
>>> y = [1, 2, 3]
>>> who
Defined variables: x, y

>>> whos
Name    Type    Size
----    ----    ----
x       int     -
y       list    3

>>> clear
(console cleared)
```

## Working with Data

### Quick Tensor Operations

```python
>>> import pycyxwiz as cx
>>> t = cx.Tensor.random([3, 3])
>>> t.shape()
[3, 3]

>>> t2 = t + t
>>> t2.to_numpy()
array([[...]], dtype=float32)
```

### Using MATLAB-Style Functions

```python
>>> import pycyxwiz as cx
>>> A = cx.linalg.eye(3)
>>> B = cx.linalg.zeros(3, 3)
>>> U, S, Vt = cx.linalg.svd([[1, 2], [3, 4]])
```

### Quick Plots

```python
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> plt.plot(np.sin(np.linspace(0, 10, 100)))
>>> plt.show()
# Plot appears in a popup or inline
```

## Output Display

### Return Values

Non-None return values are displayed:

```python
>>> 2 + 2
4

>>> None
>>> # (nothing displayed)
```

### Large Outputs

Large outputs are truncated:

```python
>>> list(range(1000))
[0, 1, 2, 3, 4, ... (995 more items) ... 999]
```

### Error Messages

Errors show traceback:

```python
>>> 1 / 0
Traceback (most recent call last):
  File "<console>", line 1, in <module>
ZeroDivisionError: division by zero
```

## Configuration

### Console Settings

Access via **Edit > Preferences > Console**:

| Setting | Description | Default |
|---------|-------------|---------|
| Font Size | Console text size | 14 |
| Max Lines | History limit | 10000 |
| Auto-Complete | Enable Tab completion | On |
| Syntax Colors | Color code syntax | On |
| Show Line Numbers | Display line numbers | Off |

### Key Bindings

Customize in **Edit > Preferences > Keyboard**:

| Action | Default | Description |
|--------|---------|-------------|
| Execute | `Enter` | Run current line |
| Continue | `Shift+Enter` | New line without executing |
| Clear | `Ctrl+L` | Clear console |
| Interrupt | `Ctrl+C` | Stop execution |

## Tips & Tricks

### Quick Variable Inspection

```python
>>> tensor
<Tensor shape=[3, 3] dtype=float32>

>>> tensor.shape(), tensor.get_data_type()
([3, 3], DataType.Float32)
```

### Use Underscore for Last Result

```python
>>> 2 + 2
4
>>> _ * 10
40
```

### Magic Variables

```python
>>> _   # Last result
>>> __  # Second-to-last result
>>> ___ # Third-to-last result
```

### Quick Imports

Common modules are pre-imported:

```python
>>> np  # numpy (if available)
>>> cx  # pycyxwiz (if available)
>>> plt # matplotlib.pyplot (if available)
```

## Integration with Scripts

### Run Script from Console

```python
>>> exec(open('my_script.py').read())
```

Or use the `%run` magic:

```python
>>> %run my_script.py
```

### Access Script Variables

Variables defined in scripts are accessible:

```python
# After running a script that defines 'model'
>>> model
<SimpleClassifier object>
>>> model.forward(test_data)
```

### Share Console Variables with Script

Console variables are available in the Script Editor:

```python
# In Console:
>>> my_var = 42

# In Script Editor:
%%code
print(my_var)  # Works! Prints 42
```

## Troubleshooting

### Console Not Responding

1. Check if a script is running (look for `[*]`)
2. Press `Ctrl+C` to interrupt
3. If frozen, use **Script > Force Stop**

### Auto-Complete Not Working

1. Ensure the module is imported
2. Wait for initialization to complete
3. Try `Tab` multiple times

### Output Not Appearing

1. Ensure the command has output (try `print()`)
2. Check for infinite loops
3. Look for errors in the output

---

**Next**: [Debugging](debugging.md) | [MATLAB Compatibility](matlab-compatibility.md)
