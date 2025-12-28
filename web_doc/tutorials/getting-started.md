# Getting Started with CyxWiz

This tutorial walks you through your first experience with the CyxWiz Engine, from launching the application to exploring the interface.

## Launching the Engine

1. **Start the application**:
   - Windows: Run `cyxwiz-engine.exe`
   - macOS/Linux: Run `./cyxwiz-engine`

2. **Wait for initialization**:
   - The splash screen shows loading progress
   - Python interpreter and GPU backends are initialized

## Understanding the Interface

When the Engine loads, you'll see the main workspace:

```
+------------------------------------------------------------------+
|  File  Edit  View  Nodes  Train  Dataset  Script  Plots  Deploy  |
+------------------------------------------------------------------+
|                    |                            |                 |
|   Asset Browser    |     Central Workspace      |   Properties    |
|                    |     (Node Editor/Script)   |                 |
|   [Project Files]  |                            |   [Selected     |
|   - Scripts/       |                            |    Item Props]  |
|   - Models/        |                            |                 |
|   - Datasets/      |                            |                 |
+--------------------+----------------------------+-----------------+
|                                                                   |
|   Console / Script Editor / Training Dashboard                    |
|                                                                   |
+------------------------------------------------------------------+
```

### Key Panels

| Panel | Location | Purpose |
|-------|----------|---------|
| **Menu Bar** | Top | Access all features |
| **Asset Browser** | Left | Navigate project files |
| **Central Workspace** | Center | Node Editor or Script Editor |
| **Properties** | Right | Edit selected item properties |
| **Console** | Bottom | Python REPL and output logs |

## Creating Your First Project

### Step 1: Create a New Project

1. Go to **File > New Project** (or press `Ctrl+Shift+N`)
2. Choose a location and name for your project
3. Click **Create**

A new project folder is created with this structure:

```
my_project/
├── scripts/        # Python and .cyx scripts
├── models/         # Saved models
├── datasets/       # Data files
└── project.cyxproj # Project configuration
```

### Step 2: Explore the Asset Browser

The Asset Browser shows your project files:

- **Scripts/** - Python (`.py`) and CyxWiz (`.cyx`) scripts
- **Models/** - Graph files (`.cyxgraph`) and trained models (`.cyxmodel`)
- **Datasets/** - CSV, HDF5, image folders

**Actions:**
- Double-click to open files
- Right-click for context menu
- Drag files to panels

### Step 3: Open the Node Editor

1. Go to **View > Node Editor** (or press `Ctrl+1`)
2. The Node Editor appears in the central workspace

You'll see an empty canvas - this is where you'll build neural networks visually.

### Step 4: Open the Console

1. Go to **View > Console** (or press `` Ctrl+` ``)
2. The Console panel appears at the bottom

Try a Python command:

```python
>>> print("Hello, CyxWiz!")
Hello, CyxWiz!
```

## Quick Tour of Menus

### File Menu
- **New Project** - Create a new project
- **Open Project** - Open existing project
- **New Script** - Create a new Python/.cyx script
- **Save** / **Save As** - Save current work

### View Menu
- **Node Editor** - Visual model builder
- **Script Editor** - Code editing
- **Console** - Python REPL
- **Properties** - Configuration panel
- **Training Dashboard** - Real-time plots

### Nodes Menu
- **Add Node** - Open node palette
- **Validate Graph** - Check for errors
- **Generate Code** - Export to Python

### Train Menu
- **Start Training** - Begin training
- **Stop Training** - Halt training
- **Configure** - Set training parameters

### Tools Menu
- Access 70+ data science and ML tools
- Organized by category (Statistics, Clustering, etc.)

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| New Project | `Ctrl+Shift+N` |
| Open Project | `Ctrl+O` |
| Save | `Ctrl+S` |
| Node Editor | `Ctrl+1` |
| Script Editor | `Ctrl+2` |
| Console | `` Ctrl+` `` |
| Command Palette | `Ctrl+Shift+P` |
| Run Script | `F5` |

## Changing the Theme

1. Go to **View > Theme**
2. Choose from:
   - **Dark** (default)
   - **Light**
   - **Classic**
   - **Nord**
   - **Dracula**

## Checking GPU Status

1. Look at the bottom status bar
2. GPU info shows: `GPU: NVIDIA RTX 3080 (CUDA)`
3. If no GPU: `GPU: CPU Mode`

To verify in Python:

```python
>>> import pycyxwiz as cyx
>>> print(cyx.cuda_available())
True
```

## What's Next

Now that you're familiar with the interface:

1. **[Basic Node Tutorial](basic-node-tutorial.md)** - Build your first neural network visually
2. **[Basic Scripting Tutorial](basic-scripting-tutorial.md)** - Write Python scripts

## Troubleshooting

### Engine Won't Start
- Check that you have OpenGL 3.3+ support
- Verify all dependencies are installed

### No GPU Detected
- Install CUDA Toolkit (for NVIDIA GPUs)
- Install OpenCL drivers (for AMD/Intel)
- The Engine will fall back to CPU mode

### Python Import Errors
- Ensure the Python environment is correctly configured
- Check the Console for error messages

---

**Next**: [Basic Node Tutorial](basic-node-tutorial.md)
