# Debugging

CyxWiz includes a powerful debugging system for Python scripts with breakpoints, stepping, and variable inspection.

## Overview

The debugger provides:

- **Breakpoints** - Pause execution at specific lines
- **Stepping** - Step through code line by line
- **Variable Inspection** - View local and global variables
- **Call Stack** - See the execution path
- **Watch Expressions** - Monitor specific values
- **Conditional Breakpoints** - Break only when conditions are met

## Debugger States

| State | Description |
|-------|-------------|
| Disconnected | Debugger not attached |
| Running | Script executing normally |
| Paused | Execution paused at breakpoint |
| Stepping | Executing step command |

## Setting Breakpoints

### Visual Method

1. Click in the **gutter** (left margin) of a code cell
2. A red dot appears indicating the breakpoint
3. Click again to remove

### Keyboard Method

1. Place cursor on the line
2. Press `F9` to toggle breakpoint

### Via Code

```python
%%code
import pdb

x = 10
pdb.set_trace()  # Execution pauses here
y = x * 2
```

## Breakpoint Types

### Standard Breakpoint

Pauses every time the line is reached.

### Conditional Breakpoint

Right-click on a breakpoint → **Edit Condition**:

```python
# Break only when i > 50
for i in range(100):
    process(i)  # Breakpoint with condition: i > 50
```

### Hit Count Breakpoint

Right-click → **Hit Count**:

- Break after N hits: `>= 10`
- Break on every Nth hit: `% 5 == 0`

## Debugging Controls

### Toolbar Buttons

| Button | Shortcut | Action |
|--------|----------|--------|
| Continue | `F5` | Resume execution |
| Step Over | `F10` | Execute current line, skip functions |
| Step Into | `F11` | Step into function calls |
| Step Out | `Shift+F11` | Run until current function returns |
| Stop | `Shift+F5` | Terminate debugging |
| Restart | `Ctrl+Shift+F5` | Restart script |

### Step Over (`F10`)

Executes the current line. If it's a function call, runs the entire function:

```python
x = 10
y = calculate(x)  # Steps over, doesn't enter calculate()
z = y * 2
```

### Step Into (`F11`)

Enters function calls:

```python
x = 10
y = calculate(x)  # Steps into calculate()
z = y * 2
```

### Step Out (`Shift+F11`)

Runs until the current function returns:

```python
def calculate(x):
    a = x * 2
    b = a + 1  # If paused here, Step Out runs to return
    return b
```

## Variable Inspection

### Variables Panel

When paused, the **Variables** panel shows:

| Section | Contents |
|---------|----------|
| Locals | Variables in current scope |
| Globals | Module-level variables |
| Special | `__name__`, `__file__`, etc. |

### Viewing Values

```
Variables
├── Locals
│   ├── x: 10 (int)
│   ├── data: array([1, 2, 3]) (ndarray)
│   └── model: <SimpleClassifier> (object)
└── Globals
    ├── cx: <module 'pycyxwiz'> (module)
    └── np: <module 'numpy'> (module)
```

### Expanding Objects

Click the arrow to expand:

```
├── model: <SimpleClassifier> (object)
│   ├── dense1: <Dense> (layer)
│   │   ├── weights: array[[...]] (ndarray, 784x128)
│   │   └── bias: array[...] (ndarray, 128)
│   ├── dense2: <Dense> (layer)
│   └── learning_rate: 0.001 (float)
```

### Hover Inspection

Hover over variables in the editor to see their values:

```python
x = 10
y = x * 2  # Hover over 'x' shows: x = 10
```

## Watch Expressions

Add expressions to monitor:

1. Open the **Watch** panel
2. Click **+** to add expression
3. Enter Python expression

### Examples

| Expression | Shows |
|------------|-------|
| `len(data)` | Array length |
| `model.learning_rate` | Object attribute |
| `x > 100` | Boolean result |
| `np.mean(losses)` | Computed value |

Watch expressions update each time execution pauses.

## Call Stack

When paused, the **Call Stack** shows the execution path:

```
Call Stack
├── [0] train_epoch() at line 45
├── [1] forward() at line 23
├── [2] dense.forward() at line 8
└── [3] <module> at line 52
```

Click on a frame to:
- View that location in the editor
- See variables at that scope

## Debug Console

While paused, use the Debug Console to evaluate expressions:

```python
debug> x
10
debug> x * 2
20
debug> model.get_parameters()
{'weight': <Tensor>, 'bias': <Tensor>}
debug> model.learning_rate = 0.0001  # Modify variables!
```

## Breakpoint Management

### Breakpoints Panel

View all breakpoints:

```
Breakpoints
├── [x] my_script.cyx:15 (cell-abc123)
├── [ ] my_script.cyx:28 (cell-def456) - DISABLED
└── [x] my_script.cyx:42 (cell-ghi789) - Condition: i > 50
```

### Operations

| Action | Description |
|--------|-------------|
| Toggle | Enable/disable breakpoint |
| Delete | Remove breakpoint |
| Edit | Change condition |
| Disable All | Temporarily disable all |
| Delete All | Remove all breakpoints |

## Exception Breakpoints

Break when exceptions occur:

1. Open **Debug > Exception Breakpoints**
2. Enable types:
   - **All Exceptions** - Break on any exception
   - **Uncaught Exceptions** - Break only if not handled
   - **Specific Types** - e.g., `ValueError`, `RuntimeError`

### Example

```python
%%code
try:
    result = risky_operation()  # Breaks if exception raised
except ValueError as e:
    handle_error(e)
```

With "All Exceptions" enabled, debugger pauses at the exception site.

## Debugging Async Code

For scripts with async execution:

```python
%%code
import asyncio

async def fetch_data():
    await asyncio.sleep(1)  # Breakpoint works here
    return "data"

asyncio.run(fetch_data())
```

Breakpoints work in async functions. The debugger tracks the async context.

## Tips & Best Practices

### Start with a Breakpoint

Set a breakpoint at the start of problematic code:

```python
%%code
def process_data(data):
    # Set breakpoint here
    result = []
    for item in data:  # Step through to find issue
        processed = transform(item)
        result.append(processed)
    return result
```

### Use Conditional Breakpoints for Loops

```python
%%code
for i in range(10000):
    process(i)  # Conditional: i == 9999
```

### Watch Key Variables

Monitor variables that affect behavior:

```
Watch:
- loss
- accuracy
- model.learning_rate
- len(batch)
```

### Check Tensor Shapes

Common debugging pattern:

```python
debug> x.shape()
[32, 784]
debug> weights.shape()
[784, 128]  # Correct!
```

### Modify and Continue

Change variables while paused:

```python
debug> learning_rate = 0.001  # Was 0.01, causing divergence
debug> (continue execution with F5)
```

## Troubleshooting

### Breakpoint Not Hit

1. Ensure the line is reachable (check control flow)
2. Verify the cell has been saved
3. Check that breakpoint is enabled (red dot, not hollow)

### Variables Not Showing

1. Step to a line after variable definition
2. Expand object containers
3. Check for typos in variable names

### Debugger Frozen

1. Try `Ctrl+C` to interrupt
2. Use **Debug > Stop**
3. Restart the script

---

**Next**: [MATLAB Compatibility](matlab-compatibility.md)
