# Node Editor

The Node Editor is the visual heart of CyxWiz Engine, allowing you to design neural network architectures through an intuitive drag-and-drop interface.

## Overview

The Node Editor provides:
- **80+ Node Types** - Comprehensive ML building blocks
- **Visual Connections** - Type-safe data flow links
- **Shape Inference** - Automatic dimension calculation
- **Code Generation** - Export to PyTorch, TensorFlow, Keras
- **Patterns Library** - Pre-built architecture templates
- **Undo/Redo** - Full history support

## Interface

```
+------------------------------------------------------------------+
|  [Toolbar: Save | Load | Generate Code | Validate | Train]       |
+------------------------------------------------------------------+
|                                                                   |
|     +----------+        +----------+        +----------+          |
|     | DataInput|------->| Dense    |------->| ReLU     |          |
|     | [MNIST]  |        | units:128|        |          |          |
|     +----------+        +----------+        +----------+          |
|                              |                   |                |
|                              |              +----------+          |
|                              +------------->| Dense    |          |
|                                             | units:10 |          |
|                                             +----------+          |
|                                                  |                |
|                                             +----------+          |
|                                             | Softmax  |          |
|                                             |          |          |
|                                             +----------+          |
|                                                  |                |
|                                             +----------+          |
|                                             | Output   |          |
|                                             |          |          |
|                                             +----------+          |
|                                                                   |
|                                           [Minimap]               |
+------------------------------------------------------------------+
```

## Quick Start

### Creating Your First Model

1. **Open the Node Editor**
   - View > Node Editor or `Ctrl+1`

2. **Add an Input Node**
   - Right-click on canvas
   - Select "Data Pipeline > Dataset Input"

3. **Add Layers**
   - Right-click > "Core Layers > Dense"
   - Configure units in Properties panel

4. **Connect Nodes**
   - Drag from output pin to input pin

5. **Add Output**
   - Right-click > "Output > Output"
   - Connect final layer

6. **Validate**
   - Click "Validate" in toolbar
   - Fix any reported errors

7. **Generate Code**
   - Click framework button (PyTorch/TensorFlow/Keras)
   - Code appears in Script Editor

## Documentation Sections

| Section | Description |
|---------|-------------|
| [Node Types](node-types.md) | All 80+ available node types |
| [Connections](connections.md) | Creating and managing links |
| [Code Generation](code-generation.md) | Exporting to Python frameworks |
| [Patterns](patterns.md) | Pre-built architecture templates |
| [Validation](validation.md) | Graph validation rules |
| [Skip Connections](skip-connections.md) | Residual and dense connections |

## Node Categories

### Data Pipeline
Entry points and data handling:
- Dataset Input
- Data Loader
- Augmentation
- Data Split
- Normalize
- One-Hot Encode

### Core Layers
Fundamental neural network layers:
- Dense (Fully Connected)
- Conv1D, Conv2D, Conv3D
- DepthwiseConv2D
- MaxPool2D, AvgPool2D
- GlobalMaxPool, GlobalAvgPool
- AdaptiveAvgPool

### Normalization
Batch and layer normalization:
- BatchNorm
- LayerNorm
- GroupNorm
- InstanceNorm

### Regularization
Prevent overfitting:
- Dropout
- Flatten

### Recurrent
Sequence processing:
- RNN
- LSTM
- GRU
- Bidirectional
- TimeDistributed
- Embedding

### Attention & Transformer
Modern attention mechanisms:
- MultiHeadAttention
- SelfAttention
- CrossAttention
- LinearAttention
- TransformerEncoder
- TransformerDecoder
- PositionalEncoding

### Activations
Non-linear functions:
- ReLU, LeakyReLU, PReLU
- ELU, SELU, GELU
- Swish, Mish
- Sigmoid, Tanh
- Softmax

### Shape Operations
Tensor manipulation:
- Reshape
- Permute
- Squeeze, Unsqueeze
- View
- Split

### Merge Operations
Combining tensors:
- Concatenate
- Add
- Multiply
- Average

### Loss Functions
Training objectives:
- MSELoss
- CrossEntropyLoss
- BCELoss, BCEWithLogits
- L1Loss, SmoothL1Loss
- HuberLoss, NLLLoss

### Optimizers
Training algorithms:
- SGD
- Adam, AdamW
- RMSprop
- Adagrad
- NAdam

### Schedulers
Learning rate management:
- StepLR
- CosineAnnealing
- ReduceOnPlateau
- ExponentialLR
- WarmupScheduler

### Regularization Nodes
Additional regularization:
- L1Regularization
- L2Regularization
- ElasticNet

### Utility
Helper nodes:
- Lambda
- Identity
- Constant
- Parameter

## Pin Types

Nodes connect via typed pins:

| Type | Color | Purpose |
|------|-------|---------|
| Tensor | Blue | General tensor data |
| Labels | Green | Classification labels |
| Parameters | Orange | Model parameters |
| Loss | Red | Loss values |
| Optimizer | Purple | Optimizer state |
| Dataset | Cyan | Dataset reference |

## Toolbar Actions

| Button | Action |
|--------|--------|
| Save | Save graph to .cyxgraph file |
| Load | Load graph from file |
| Clear | Clear all nodes |
| PyTorch | Generate PyTorch code |
| TensorFlow | Generate TensorFlow code |
| Keras | Generate Keras code |
| PyCyxWiz | Generate PyCyxWiz code |
| Validate | Check graph validity |
| Train | Start training |

## Navigation

### Mouse Controls

| Action | Result |
|--------|--------|
| Left-click | Select node |
| Ctrl+Left-click | Add to selection |
| Left-drag (empty) | Box select |
| Middle-drag | Pan canvas |
| Right-drag | Pan canvas |
| Scroll wheel | Zoom |
| Right-click | Context menu |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `F` | Frame selected |
| `Home` | Frame all |
| `Delete` | Delete selected |
| `Ctrl+C/X/V` | Copy/Cut/Paste |
| `Ctrl+D` | Duplicate |
| `Ctrl+A` | Select all |
| `Ctrl+Z/Y` | Undo/Redo |
| `Ctrl+F` | Search nodes |

## Minimap

The minimap (bottom-right) shows:
- Overview of entire graph
- Current viewport rectangle
- Node distribution

**Toggle:** View > Node Editor Minimap

**Interaction:**
- Click to navigate
- Drag rectangle to move view

## Context Menu

### Canvas Menu (right-click empty space)

```
Add Node >
  Data Pipeline >
  Core Layers >
  Normalization >
  Recurrent >
  Attention >
  Activations >
  Shape Operations >
  Merge Operations >
  Loss Functions >
  Optimizers >
  Schedulers >
  Output >
---
Paste
Frame All
Clear Graph
```

### Node Menu (right-click node)

```
Duplicate
Delete
Disconnect All
---
Properties
Add to Group
Create Subgraph
```

## Working with Groups

Groups visually organize related nodes:

1. Select multiple nodes
2. Right-click > Create Group
3. Enter group name
4. Choose color

Groups:
- Can be collapsed/expanded
- Have custom colors
- Show node count
- Can be nested

## Subgraphs

Subgraphs encapsulate complexity:

1. Select nodes to encapsulate
2. Right-click > Create Subgraph
3. Double-click to expand/collapse

Subgraphs:
- Reduce visual clutter
- Enable reuse
- Maintain connections

## Tips & Best Practices

1. **Start with Data** - Always begin with DatasetInput
2. **End with Output** - Required for training
3. **Validate Often** - Catch errors early
4. **Use Patterns** - Start from proven architectures
5. **Name Your Nodes** - Double-click title to rename
6. **Organize Spatially** - Use alignment tools
7. **Save Frequently** - Use auto-save

## Troubleshooting

### "Invalid connection" error
- Check pin types match
- Verify data flow direction

### "Missing input" warning
- Connect all required input pins
- Add DatasetInput node

### "Cycle detected" error
- Remove circular connections
- Use skip connections properly

### "Shape mismatch"
- Check layer dimensions
- Use Reshape if needed

---

**Next**: [Node Types](node-types.md) | [Connections](connections.md)
