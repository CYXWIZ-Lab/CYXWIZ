# CyxWiz Engine

The CyxWiz Engine is the flagship desktop application for designing, training, and deploying machine learning models. It provides a visual interface similar to professional game engines and scientific computing environments.

## Overview

The Engine combines:
- **Visual Node Editor** - Drag-and-drop ML pipeline building
- **Python Integration** - Full scripting support with embedded interpreter
- **Real-Time Visualization** - Live training metrics and plotting
- **70+ Tool Panels** - Comprehensive data science toolkit
- **Code Generation** - Export to PyTorch, TensorFlow, Keras

## Quick Start

1. Launch the Engine application
2. Create a new project via **File > New Project**
3. Open the **Node Editor** from the View menu
4. Drag layers from the context menu to build your model
5. Configure parameters in the **Properties** panel
6. Click **Train > Start Training** to begin

## Main Interface

```
+------------------------------------------------------------------+
|  File  Edit  View  Nodes  Train  Dataset  Script  Plots  Deploy  |
+------------------------------------------------------------------+
|                    |                            |                 |
|   Asset Browser    |     Node Editor            |   Properties    |
|                    |                            |                 |
|   [Project Files]  |   [Visual Graph]           |   [Selected     |
|   - Scripts/       |                            |    Node Params] |
|   - Models/        |   +------+    +-------+    |                 |
|   - Datasets/      |   |Input |---->|Dense  |   |   Units: 128    |
|                    |   +------+    +-------+    |   Activation:   |
|                    |                   |        |   [ReLU      v] |
|                    |               +-------+    |                 |
+--------------------+               |Output |    +-----------------+
|                                    +-------+                      |
|   Console / Script Editor / Training Dashboard                    |
|                                                                   |
|   >>> import pycyxwiz                                             |
|   >>> model = pycyxwiz.Sequential()                               |
|                                                                   |
+------------------------------------------------------------------+
```

## Documentation Sections

### Core Features

| Section | Description |
|---------|-------------|
| [Menus Reference](menus.md) | Complete menu bar documentation |
| [Keyboard Shortcuts](shortcuts.md) | All shortcuts and customization |
| [Themes](themes.md) | Visual customization options |
| [Command Palette](command-palette.md) | Quick command access (Ctrl+P) |

### Panels

| Panel | Description |
|-------|-------------|
| [Node Editor](node-editor/index.md) | Visual ML pipeline builder |
| [Script Editor](panels/script-editor.md) | Python code editing |
| [Console](panels/console.md) | Python REPL and logs |
| [Properties](panels/properties.md) | Node parameter editing |
| [Asset Browser](panels/asset-browser.md) | Project file management |
| [Viewport](panels/viewport.md) | Training status and system info |
| [Training Dashboard](panels/training-dashboard.md) | Real-time loss/accuracy plots |
| [Dataset Panel](panels/dataset-panel.md) | Dataset configuration |
| [Table Viewer](panels/table-viewer.md) | Data inspection |

### Node Editor Deep Dive

| Topic | Description |
|-------|-------------|
| [Node Types](node-editor/node-types.md) | All 80+ node types |
| [Connections](node-editor/connections.md) | Linking nodes and data flow |
| [Code Generation](node-editor/code-generation.md) | Exporting to Python frameworks |
| [Patterns](node-editor/patterns.md) | Pre-built architecture templates |
| [Validation](node-editor/validation.md) | Graph validation rules |

### Training System

| Topic | Description |
|-------|-------------|
| [Training Configuration](training.md) | Setting up training runs |
| [Local Training](training-local.md) | Training on your machine |
| [Distributed Training](training-distributed.md) | Using the network |
| [Checkpoints](checkpoints.md) | Saving and resuming |
| [Visualization](visualization.md) | Real-time monitoring |

### Python Scripting

| Topic | Description |
|-------|-------------|
| [Getting Started](scripting.md) | Python integration overview |
| [pycyxwiz Module](pycyxwiz.md) | Backend Python bindings |
| [cyxwiz_plotting](plotting-api.md) | Plotting Python bindings |
| [Example Scripts](examples.md) | Sample training scripts |

## Feature Highlights

### Visual Model Building

Build neural networks visually without writing code:

- **80+ Node Types** - Layers, activations, optimizers, loss functions
- **Smart Connections** - Type-safe pin connections with validation
- **Shape Inference** - Automatic output shape calculation
- **Code Export** - Generate PyTorch, TensorFlow, Keras, or PyCyxWiz

### Comprehensive Tool Suite

Access 70+ analysis tools via the **Tools** menu:

- **Data Science** - Profiling, correlation, missing values
- **Statistics** - Hypothesis testing, regression analysis
- **Clustering** - K-Means, DBSCAN, GMM, hierarchical
- **Model Evaluation** - ROC curves, confusion matrices
- **Signal Processing** - FFT, wavelets, spectrograms
- **Linear Algebra** - SVD, eigendecomposition, QR
- **Time Series** - Forecasting, decomposition, seasonality
- **Text Processing** - Tokenization, TF-IDF, embeddings

### Real-Time Training

Monitor training progress live:

- **Loss/Accuracy Plots** - Updated every batch
- **Live Metrics** - Current epoch, learning rate
- **GPU Utilization** - Memory and compute usage
- **Export Data** - Save training history to CSV

### Project Management

Organize your ML work:

- **Project Structure** - Scripts, models, datasets folders
- **Asset Browser** - File tree with filters
- **Recent Projects** - Quick access list
- **Auto-Save** - Never lose work

## System Requirements

### Minimum
- **OS**: Windows 10, macOS 10.15, Ubuntu 18.04
- **CPU**: 4 cores
- **RAM**: 8 GB
- **GPU**: OpenGL 3.3 compatible
- **Storage**: 500 MB (application)

### Recommended
- **OS**: Windows 11, macOS 13, Ubuntu 22.04
- **CPU**: 8+ cores
- **RAM**: 16+ GB
- **GPU**: NVIDIA RTX series (for CUDA acceleration)
- **Storage**: 10+ GB (with datasets)

## File Types

| Extension | Description |
|-----------|-------------|
| `.cyxgraph` | Node editor graph files |
| `.cyxmodel` | Trained model files |
| `.cyx` | CyxWiz script files |
| `.py` | Python script files |

## Related Documentation

- [Backend API](../backend/index.md) - Underlying compute library
- [Protocol Reference](../protocol/index.md) - Network communication
- [Developer Guide](../developer/index.md) - Building and contributing

---

**Next**: [Menus Reference](menus.md) | [Node Editor](node-editor/index.md)
