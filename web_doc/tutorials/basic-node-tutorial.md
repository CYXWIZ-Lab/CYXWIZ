# Basic Node Tutorial

Learn to build a simple neural network using the visual Node Editor. No coding required!

## Goal

Create a 2-layer Dense (fully-connected) network:

```
Input (784) → Dense (128, ReLU) → Dense (10, Softmax) → Output
```

This architecture can classify MNIST digits (28x28 images = 784 pixels → 10 digit classes).

## Prerequisites

- CyxWiz Engine running
- A project open (see [Getting Started](getting-started.md))

## Step 1: Open the Node Editor

1. Go to **View > Node Editor** (or press `Ctrl+1`)
2. You'll see an empty canvas with a grid background

### Navigation Controls

| Action | Control |
|--------|---------|
| Pan | Middle-mouse drag or Space+drag |
| Zoom | Mouse wheel |
| Select | Left-click |
| Multi-select | Ctrl+click or drag box |

## Step 2: Add an Input Node

1. **Right-click** on the canvas to open the context menu
2. Navigate to **Data > DatasetInput**
3. Click to place the node

The **DatasetInput** node appears:

```
+------------------+
|  DatasetInput    |
+------------------+
| Dataset: [None]  |
|   Batch: 32      |
+------------------+
|        [output]→ |
+------------------+
```

### Configure the Input

1. Select the DatasetInput node
2. In the **Properties** panel (right side):
   - Set **Input Shape**: `784` (flattened 28x28 image)
   - Set **Batch Size**: `32`

## Step 3: Add the First Dense Layer

1. **Right-click** on the canvas
2. Navigate to **Layers > Dense**
3. Place it to the right of the Input node

The **Dense** node appears:

```
+------------------+
|     Dense        |
+------------------+
|   Units: 128     |
| Activation: ReLU |
+------------------+
| →[input] [output]→|
+------------------+
```

### Configure the Dense Layer

1. Select the Dense node
2. In Properties:
   - **Units**: `128`
   - **Activation**: `ReLU`
   - **Use Bias**: checked

## Step 4: Connect Input to Dense

1. Click on the **output pin** (right side) of DatasetInput
2. Drag to the **input pin** (left side) of Dense
3. Release to create a connection

A colored line now connects the nodes:

```
+-------------+          +-------------+
| DatasetInput|----→-----| Dense       |
+-------------+          +-------------+
```

### Checking the Connection

When connected properly:
- The link turns green (valid connection)
- The Properties panel shows "Input Shape: 784, Output Shape: 128"

## Step 5: Add the Second Dense Layer

1. Right-click → **Layers > Dense**
2. Place it to the right of the first Dense

### Configure This Layer

1. **Units**: `10` (one for each digit class)
2. **Activation**: `Softmax` (for classification probabilities)

### Connect the Layers

1. Draw a connection from the first Dense output to the second Dense input

Your graph now looks like:

```
DatasetInput → Dense(128, ReLU) → Dense(10, Softmax)
```

## Step 6: Add an Output Node

1. Right-click → **Output > ModelOutput**
2. Place it to the right of the second Dense
3. Connect the second Dense to the Output

Final graph:

```
+-------------+    +-------------+    +-------------+    +-------------+
| DatasetInput|-→--| Dense(128)  |-→--| Dense(10)   |-→--| ModelOutput |
+-------------+    | ReLU        |    | Softmax     |    +-------------+
                   +-------------+    +-------------+
```

## Step 7: Validate the Graph

1. Go to **Nodes > Validate Graph** (or press `Ctrl+Shift+V`)
2. A green checkmark appears if valid

If there are errors, they appear in the Console:
- "Missing input connection" → Ensure all inputs are connected
- "Shape mismatch" → Check layer dimensions

## Step 8: View the Generated Code

1. Go to **Nodes > Generate Code**
2. Choose a framework:
   - **PyTorch**
   - **TensorFlow/Keras**
   - **PyCyxWiz**

### PyTorch Output

```python
import torch
import torch.nn as nn

class GeneratedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_1 = nn.Linear(784, 128)
        self.relu_1 = nn.ReLU()
        self.dense_2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.relu_1(x)
        x = self.dense_2(x)
        x = self.softmax(x)
        return x
```

### PyCyxWiz Output

```python
import pycyxwiz as cx

# Layer definitions
dense_1 = cx.Dense(784, 128)
relu_1 = cx.ReLU()
dense_2 = cx.Dense(128, 10)
softmax = cx.Softmax()

# Forward pass
def forward(x):
    x = dense_1.forward(x)
    x = relu_1.forward(x)
    x = dense_2.forward(x)
    x = softmax.forward(x)
    return x
```

## Step 9: Save the Graph

1. Go to **File > Save As**
2. Choose a name: `my_first_model.cyxgraph`
3. The graph is saved in your project's `models/` folder

## Bonus: Using the Minimap

1. Look at the bottom-right corner of the Node Editor
2. A minimap shows your entire graph
3. Drag the viewport indicator to navigate large graphs

## Bonus: Quick Node Search

1. Press `Ctrl+A` or click the search box (top-right)
2. Type a node name: `dense`
3. Press Enter to add the node

This is faster than using the context menu!

## What You Learned

- Adding nodes from the context menu
- Connecting nodes with links
- Configuring node properties
- Validating the graph
- Generating Python code

## Common Node Types

| Category | Nodes |
|----------|-------|
| **Data** | DatasetInput, DataLoader |
| **Layers** | Dense, Conv2D, LSTM, Flatten |
| **Activations** | ReLU, Sigmoid, Tanh, Softmax, GELU |
| **Normalization** | BatchNorm, LayerNorm |
| **Regularization** | Dropout, L2Regularization |
| **Output** | ModelOutput, LossOutput |

See [Node Types Reference](../engine/node-editor/node-types.md) for all 80+ nodes.

## Troubleshooting

### Nodes Won't Connect
- Check that you're connecting output → input (left to right)
- Verify data types are compatible

### Shapes Don't Match
- Use Flatten before Dense layers if coming from Conv2D
- Check the Properties panel for shape information

### Graph Won't Validate
- Ensure there's a DatasetInput and ModelOutput
- Check for disconnected nodes

---

**Next**: [Basic Scripting Tutorial](basic-scripting-tutorial.md)
