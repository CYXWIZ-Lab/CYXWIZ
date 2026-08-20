# CyxWiz Startup Scripts

This directory contains example startup scripts that run automatically when CyxWiz Engine starts.

## What are Startup Scripts?

Startup scripts are Python (.cyx) files that execute automatically on application launch. They're useful for:

- **Importing common libraries**: Auto-load math, numpy, pandas, etc.
- **Loading datasets**: Automatically load frequently used data
- **Configuring environment**: Set up paths, preferences
- **Custom initialization**: Run your own setup code

## How to Use

### Enable Startup Scripts

1. Build the Engine, then edit `startup_scripts.txt` next to the Engine executable
2. Add script paths (one per line):
   ```
   scripts/startup/welcome.cyx
   scripts/startup/init_imports.cyx
   ```
3. Restart CyxWiz Engine
4. Scripts will run automatically and output to Console

### Configuration File Format

```
# Comments start with #
# One script path per line
# Paths can be absolute or relative

scripts/startup/welcome.cyx
scripts/startup/init_imports.cyx
C:/Users/me/my_custom_startup.cyx
```

### Example Scripts

This directory includes:

**welcome.cyx**:
- Displays welcome message
- Shows quick tips
- Lists useful commands

**init_imports.cyx**:
- Auto-imports math, random, json
- Attempts to load numpy, pandas
- Imports pycyxwiz if available

## Creating Your Own Startup Scripts

Example: `my_startup.cyx`

```python
# My custom startup script

print("Loading my custom setup...")

# Import libraries
import math
import random

# Define helper functions
def quick_plot(data):
    """Quick plotting helper"""
    # Implementation here
    pass

# Load data
try:
    # Auto-load my dataset
    data = load_data("my_dataset.csv")
    print(f"Loaded dataset: {len(data)} rows")
except:
    print("Dataset not found")

print("Custom setup complete!")
```

Then add to `startup_scripts.txt`:
```
scripts/startup/my_startup.cyx
```

## Tips

**Keep scripts fast**: Avoid long-running operations (>5 seconds)

**Handle errors gracefully**: Use try/except to prevent blocking startup

**Use print() for feedback**: Messages appear in Console

**Test scripts individually**: Run scripts manually before adding to startup

**Safe mode**: Hold Shift on startup to skip all startup scripts (future feature)

## Troubleshooting

**Scripts not running?**
- Check `startup_scripts.txt` exists next to the Engine executable
- Verify script paths are correct (absolute or relative)
- Check Console for error messages

**Application slow to start?**
- Remove slow scripts from configuration
- Optimize scripts to run faster
- Consider lazy loading (import on first use)

**Script errors?**
- Scripts continue even if one fails
- Check Console for error output
- Test script manually first (F5 in Script Editor)

## Advanced

### Conditional Execution

```python
# Only run on Windows
import platform
if platform.system() == "Windows":
    print("Windows-specific setup")
```

### Project Detection

```python
# Different setup for different projects
import os
project_name = os.path.basename(os.getcwd())
if project_name == "my_ml_project":
    # Load ML-specific libraries
    import tensorflow as tf
```

### Performance Timing

```python
import time
start = time.time()

# Your initialization code here

elapsed = time.time() - start
print(f"Startup completed in {elapsed:.2f}s")
```

## Examples Gallery

See more examples in the [CyxWiz Documentation](../docs/startup_scripts.md).

---

**Happy scripting!** 🐍
