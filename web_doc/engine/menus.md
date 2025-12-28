# Menus Reference

This document provides complete documentation for all menu items in the CyxWiz Engine.

## Menu Bar Overview

```
File | Edit | View | Nodes | Train | Dataset | Script | Plots | Deploy | Tools | Help
```

---

## File Menu

Project and file management operations.

### Project Operations

| Item | Shortcut | Description |
|------|----------|-------------|
| **New Project...** | `Ctrl+Shift+N` | Create a new CyxWiz project |
| **Open Project...** | `Ctrl+Shift+O` | Open existing project folder |
| **Close Project** | - | Close current project |
| **Recent Projects** | - | Submenu with last 10 projects |

### Script Operations

| Item | Shortcut | Description |
|------|----------|-------------|
| **New Script** | `Ctrl+N` | Create new .py or .cyx script |
| **Open Script...** | `Ctrl+O` | Open script file |
| **Save** | `Ctrl+S` | Save current script |
| **Save As...** | `Ctrl+Shift+S` | Save with new name |
| **Save All** | `Ctrl+Alt+S` | Save all open files |

### Auto-Save

| Item | Description |
|------|-------------|
| **Enable Auto-Save** | Toggle checkbox for automatic saving |
| **Auto-Save Interval** | Configure interval (default: 60s) |

### Application

| Item | Shortcut | Description |
|------|----------|-------------|
| **Account Settings** | - | Login/logout, profile settings |
| **Exit** | `Alt+F4` | Close application (with confirmation) |

---

## Edit Menu

Text editing and code manipulation.

### Basic Editing

| Item | Shortcut | Description |
|------|----------|-------------|
| **Undo** | `Ctrl+Z` | Undo last action |
| **Redo** | `Ctrl+Y` | Redo undone action |
| **Cut** | `Ctrl+X` | Cut selection |
| **Copy** | `Ctrl+C` | Copy selection |
| **Paste** | `Ctrl+V` | Paste clipboard |
| **Delete** | `Delete` | Delete selection |
| **Select All** | `Ctrl+A` | Select all content |

### Find & Replace

| Item | Shortcut | Description |
|------|----------|-------------|
| **Find...** | `Ctrl+F` | Open find dialog |
| **Find Next** | `F3` | Find next occurrence |
| **Replace...** | `Ctrl+H` | Open replace dialog |
| **Find in Files...** | `Ctrl+Shift+F` | Search across project |
| **Replace in Files...** | `Ctrl+Shift+H` | Replace across project |

### Navigation

| Item | Shortcut | Description |
|------|----------|-------------|
| **Go to Line...** | `Ctrl+G` | Jump to line number |

### Line Operations

| Item | Shortcut | Description |
|------|----------|-------------|
| **Duplicate Line** | `Ctrl+D` | Duplicate current line |
| **Move Line Up** | `Alt+Up` | Move line up |
| **Move Line Down** | `Alt+Down` | Move line down |
| **Delete Line** | `Ctrl+Shift+K` | Delete current line |

### Indentation

| Item | Shortcut | Description |
|------|----------|-------------|
| **Indent** | `Tab` | Increase indentation |
| **Outdent** | `Shift+Tab` | Decrease indentation |

### Comments

| Item | Shortcut | Description |
|------|----------|-------------|
| **Toggle Line Comment** | `Ctrl+/` | Comment/uncomment line |
| **Toggle Block Comment** | `Ctrl+Shift+/` | Comment/uncomment block |

### Transform Text

| Item | Description |
|------|-------------|
| **To Uppercase** | Convert selection to uppercase |
| **To Lowercase** | Convert selection to lowercase |
| **To Title Case** | Convert selection to title case |

### Sort Lines

| Item | Description |
|------|-------------|
| **Sort Ascending** | Sort selected lines A-Z |
| **Sort Descending** | Sort selected lines Z-A |
| **Join Lines** | Merge selected lines |

### Preferences

| Item | Description |
|------|-------------|
| **Preferences...** | Open settings dialog |

---

## View Menu

Panel visibility and layout management.

### Core Panels

| Item | Shortcut | Description |
|------|----------|-------------|
| **Node Editor** | `Ctrl+1` | Show/hide node editor |
| **Script Editor** | `Ctrl+2` | Show/hide script editor |
| **Console** | `Ctrl+3` | Show/hide Python console |
| **Properties** | `Ctrl+4` | Show/hide properties panel |
| **Asset Browser** | `Ctrl+5` | Show/hide asset browser |
| **Viewport** | `Ctrl+6` | Show/hide viewport |

### Training Panels

| Item | Description |
|------|-------------|
| **Training Dashboard** | Real-time training plots |
| **Training Plot (Global)** | Network training metrics |
| **Task Progress** | Background task status |

### Data Panels

| Item | Description |
|------|-------------|
| **Dataset Manager** | Configure datasets |
| **Table Viewer** | Inspect data tables |
| **Query Console** | Run CyxQL queries |

### Advanced Panels

| Item | Description |
|------|-------------|
| **Pattern Browser** | Pre-built model patterns |
| **Variable Explorer** | Python variable inspection |
| **Output Renderer** | Visualize outputs |

### Developer Tools

| Item | Description |
|------|-------------|
| **Custom Node Editor** | Create custom nodes |
| **Theme Editor** | Edit color themes |
| **Profiler** | Performance profiling |
| **Memory Monitor** | Memory usage tracking |

### Minimaps

| Item | Description |
|------|-------------|
| **Node Editor Minimap** | Toggle minimap in node editor |
| **Script Editor Minimap** | Toggle minimap in script editor |

### Debug Logging

| Item | Description |
|------|-------------|
| **Idle Log** | Log idle frame events |
| **Verbose Python Log** | Detailed Python output |

### Theme Selection

| Item | Description |
|------|-------------|
| **Dark** | Dark color theme |
| **Light** | Light color theme |
| **Classic** | Classic ImGui theme |
| **Nord** | Nord color palette |
| **Dracula** | Dracula color palette |

### Layout

| Item | Description |
|------|-------------|
| **Reset Layout** | Reset to default layout |
| **Save Layout** | Save current layout |
| **Save Project Settings** | Save project-specific settings |

---

## Nodes Menu

Node editor operations.

### Selection

| Item | Shortcut | Description |
|------|----------|-------------|
| **Select All** | `Ctrl+A` | Select all nodes |
| **Clear Selection** | `Escape` | Deselect all |
| **Frame Selected** | `F` | Zoom to selected nodes |
| **Frame All** | `Home` | Zoom to show all nodes |

### Clipboard

| Item | Shortcut | Description |
|------|----------|-------------|
| **Copy** | `Ctrl+C` | Copy selected nodes |
| **Cut** | `Ctrl+X` | Cut selected nodes |
| **Paste** | `Ctrl+V` | Paste nodes |
| **Duplicate** | `Ctrl+D` | Duplicate selected |
| **Delete** | `Delete` | Delete selected |

### History

| Item | Shortcut | Description |
|------|----------|-------------|
| **Undo** | `Ctrl+Z` | Undo graph change |
| **Redo** | `Ctrl+Y` | Redo graph change |

### Alignment

| Item | Description |
|------|-------------|
| **Align Left** | Align selected nodes left |
| **Align Center** | Align horizontally center |
| **Align Right** | Align selected nodes right |
| **Align Top** | Align selected nodes top |
| **Align Middle** | Align vertically middle |
| **Align Bottom** | Align selected nodes bottom |

### Distribution

| Item | Description |
|------|-------------|
| **Distribute Horizontal** | Space nodes horizontally |
| **Distribute Vertical** | Space nodes vertically |
| **Auto Layout** | Automatic arrangement |

### Grouping

| Item | Description |
|------|-------------|
| **Create Group** | Group selected nodes |
| **Ungroup** | Ungroup selected |
| **Create Subgraph** | Encapsulate as subgraph |
| **Expand Subgraph** | Expand subgraph node |

### File Operations

| Item | Shortcut | Description |
|------|----------|-------------|
| **Save Graph** | `Ctrl+S` | Save graph to file |
| **Load Graph** | `Ctrl+O` | Load graph from file |
| **Clear Graph** | - | Clear all nodes |

### Code Generation

| Item | Description |
|------|-------------|
| **Generate PyTorch** | Generate PyTorch code |
| **Generate TensorFlow** | Generate TensorFlow code |
| **Generate Keras** | Generate Keras code |
| **Generate PyCyxWiz** | Generate PyCyxWiz code |

### Pattern Operations

| Item | Description |
|------|-------------|
| **Save as Pattern** | Save selection as pattern |
| **Load Pattern** | Load pattern template |

### Validation

| Item | Description |
|------|-------------|
| **Validate Graph** | Check for errors |

---

## Train Menu

Training operations.

### Training Control

| Item | Shortcut | Description |
|------|----------|-------------|
| **Start Training** | `F5` | Begin training |
| **Pause Training** | `F6` | Pause (TODO) |
| **Stop Training** | `Shift+F5` | Stop training (TODO) |

### Server Operations

| Item | Description |
|------|-------------|
| **Connect to Server** | Connect to Central Server |
| **Submit to Network** | Submit job to network |
| **View Job Status** | Monitor submitted jobs |

---

## Dataset Menu

Dataset management.

| Item | Description |
|------|-------------|
| **Import Dataset** | Import new dataset |
| **Dataset Manager** | Open dataset panel |
| **Refresh Datasets** | Reload dataset list |

---

## Script Menu

Python scripting operations.

### Script Management

| Item | Shortcut | Description |
|------|----------|-------------|
| **New Script** | `Ctrl+N` | Create new script |
| **Open Script** | `Ctrl+O` | Open script file |
| **Save Script** | `Ctrl+S` | Save current script |

### Execution

| Item | Shortcut | Description |
|------|----------|-------------|
| **Run Script** | `F5` | Execute current script |
| **Run Selection** | `Ctrl+Enter` | Execute selected code |
| **Stop Execution** | `Shift+F5` | Cancel running script |

### Console

| Item | Description |
|------|-------------|
| **Clear Console** | Clear console output |
| **Restart Kernel** | Restart Python interpreter |

---

## Plots Menu

Plotting and visualization.

| Item | Description |
|------|-------------|
| **New Line Plot** | Create line plot window |
| **New Scatter Plot** | Create scatter plot window |
| **New Bar Chart** | Create bar chart window |
| **New Histogram** | Create histogram window |
| **New Heatmap** | Create heatmap window |
| **Test Control** | Plot testing interface |

---

## Deploy Menu

Model deployment operations.

### Export

| Item | Description |
|------|-------------|
| **Export Model** | Export trained model |
| **Export as ONNX** | Export to ONNX format |
| **Export as PyTorch** | Export to .pt format |
| **Export as TensorFlow** | Export to SavedModel |
| **Export as GGUF** | Export to GGUF format |

### Import

| Item | Description |
|------|-------------|
| **Import Model** | Import existing model |

### Network

| Item | Description |
|------|-------------|
| **Deploy to Node** | Deploy model to Server Node |
| **Model Marketplace** | Browse/publish models |

---

## Tools Menu

Comprehensive analysis toolkit. See [Tools Documentation](../tools/index.md) for details.

### Categories

| Category | Tools |
|----------|-------|
| **Model Analysis** | Model Summary, Architecture Diagram, LR Finder |
| **Data Science** | Data Profiler, Correlation Matrix, Missing Values, Outliers |
| **Statistics** | Descriptive Stats, Hypothesis Test, Distribution Fitter, Regression |
| **Clustering** | K-Means, DBSCAN, Hierarchical, GMM, Cluster Evaluation |
| **Model Evaluation** | Confusion Matrix, ROC-AUC, PR Curve, Cross-Validation, Learning Curves |
| **Transformations** | Normalization, Standardization, Log Transform, Box-Cox, Feature Scaling |
| **Linear Algebra** | Matrix Calculator, Eigendecomposition, SVD, QR, Cholesky |
| **Signal Processing** | FFT, Spectrogram, Filter Designer, Convolution, Wavelet |
| **Optimization** | Gradient Descent, Convexity, Linear Programming, Quadratic Programming |
| **Calculus** | Differentiation, Integration |
| **Time Series** | Decomposition, ACF/PACF, Stationarity, Seasonality, Forecasting |
| **Text Processing** | Tokenization, Word Frequency, TF-IDF, Embeddings, Sentiment |
| **Utilities** | Calculator, Unit Converter, Random Generator, Hash Generator, JSON Viewer, Regex Tester |

### Other Tools

| Item | Description |
|------|-------------|
| **Resume from Checkpoint** | Load training checkpoint |
| **Save Checkpoint** | Save current state |
| **Run Quick Test** | Run model tests |
| **Compare Test Results** | Compare multiple runs |
| **Export Test Report** | Generate test report |
| **Clear Cache** | Clear temporary files |
| **Run Garbage Collection** | Free unused memory |

---

## Help Menu

Documentation and support.

| Item | Shortcut | Description |
|------|----------|-------------|
| **Documentation** | `F1` | Open online docs |
| **Keyboard Shortcuts** | - | View all shortcuts |
| **Tutorial** | - | Interactive tutorial |
| **About CyxWiz** | - | Version and credits |
| **Check for Updates** | - | Check for new version |
| **Report Issue** | - | Open issue tracker |

---

## Context Menus

### Node Editor Context Menu

Right-click on canvas:

| Section | Items |
|---------|-------|
| **Add Node** | All layer categories |
| **Paste** | Paste copied nodes |
| **Frame All** | Zoom to fit |
| **Clear** | Clear graph |

Right-click on node:

| Item | Description |
|------|-------------|
| **Duplicate** | Duplicate node |
| **Delete** | Remove node |
| **Disconnect All** | Remove all connections |
| **Properties** | Show in properties panel |

### Asset Browser Context Menu

| Item | Description |
|------|-------------|
| **Open** | Open file |
| **Open in Script Editor** | Open script for editing |
| **View in Table** | Open data in Table Viewer |
| **Rename** | Rename file |
| **Delete** | Delete file |
| **Copy Path** | Copy file path |
| **Show in Explorer** | Open containing folder |

---

**Next**: [Keyboard Shortcuts](shortcuts.md) | [Node Editor](node-editor/index.md)
