# MNIST Demo Walkthrough

A comprehensive guide to training and testing the MNIST handwritten digit dataset using CyxWiz Engine.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [System Capabilities](#system-capabilities)
4. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
   - [Step 1: Launch the Engine](#step-1-launch-the-engine)
   - [Step 2: Load MNIST Dataset](#step-2-load-mnist-dataset)
   - [Step 3: Build the Model Architecture](#step-3-build-the-model-architecture)
   - [Step 4: Configure Training Parameters](#step-4-configure-training-parameters)
   - [Step 5: Start Training](#step-5-start-training)
   - [Step 6: Monitor Training Progress](#step-6-monitor-training-progress)
   - [Step 7: Evaluate Results](#step-7-evaluate-results)
5. [Alternative: Using Pre-built Patterns](#alternative-using-pre-built-patterns)
6. [Alternative: Python Scripting](#alternative-python-scripting)
7. [Logging and Assertions](#logging-and-assertions)
8. [Known Limitations](#known-limitations)
9. [Troubleshooting](#troubleshooting)

---

## Overview

This guide demonstrates how to train a neural network on the MNIST dataset (70,000 handwritten digit images, 28x28 pixels) using the CyxWiz Engine's visual node editor and local training capabilities.

**What you will accomplish:**
- Load and preprocess the MNIST dataset
- Build a Multi-Layer Perceptron (MLP) architecture
- Train the model with real-time visualization
- Evaluate accuracy on test data

**Estimated time:** 15-30 minutes (depending on hardware)

---

## Prerequisites

### Software Requirements
- CyxWiz Engine (Release build recommended)
- Windows 10/11, macOS, or Linux

### Hardware Requirements
- **Minimum:** CPU with 4+ cores, 8GB RAM
- **Recommended:** NVIDIA GPU with CUDA support for faster training

### Build the Engine
```bash
# From project root
cmake --preset windows-release
cmake --build build --config Release -j 8

# Run the engine
./build/bin/Release/cyxwiz-engine.exe
```

---

## System Capabilities

### What CyxWiz Engine Can Do (Current State)

| Feature | Status | Notes |
|---------|--------|-------|
| Visual Node Editor | Fully functional | 90+ node types available |
| MLP/Dense Networks | Fully trainable | Sequential architecture |
| MNIST Data Loading | Automatic | Downloads from internet if needed |
| Real-time Training Plots | Functional | Loss, accuracy, learning rate |
| Python Scripting | Functional | Interactive console + pycyxwiz |
| Model Save/Load | Functional | .cyxgraph format |
| Code Generation | Functional | PyTorch, TensorFlow, Keras, PyCyxWiz |

### Trainable Layer Types
- **Dense (Fully Connected)** - Core building block
- **Dropout** - Regularization
- **BatchNorm** - Normalization (limited)
- **Flatten** - Shape transformation
- **Activations** - ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU, GELU, Swish

### Not Yet Trainable (Nodes Exist but No Backend)
- Conv2D, Conv3D, DepthwiseConv2D
- MaxPool, AvgPool, GlobalPool
- LSTM, GRU, Transformer layers
- Attention mechanisms

---

## Step-by-Step Walkthrough

### Step 1: Launch the Engine

1. Run the CyxWiz Engine executable:
   ```bash
   ./build/bin/Release/cyxwiz-engine.exe
   ```

2. You'll see the main window with:
   - **Menu Bar** at top (File, Edit, View, Tools, Help)
   - **Node Editor** in center
   - **Properties Panel** on right
   - **Console** at bottom
   - **Dataset Panel** (View > Dataset Manager if not visible)

### Step 2: Load MNIST Dataset

#### Method A: Via Dataset Panel (Recommended)

1. Open **View > Dataset Manager** if not already visible
2. Click **"Add Dataset"** button
3. In the dialog:
   - **Name:** `mnist`
   - **Source:** Select "Built-in" or "Download"
   - **Type:** Classification
4. Click **"Load"**
5. The engine will automatically download MNIST (~11MB) if not cached

#### Method B: Via Python Console

1. Open the Console panel (View > Console)
2. Run the following commands:

```python
# Load MNIST using the data registry
from pycyxwiz import DataRegistry

# Load training data
train_data = DataRegistry.load("mnist", split="train")
print(f"Training samples: {len(train_data)}")

# Load test data
test_data = DataRegistry.load("mnist", split="test")
print(f"Test samples: {len(test_data)}")
```

**Expected Output:**
```
Training samples: 60000
Test samples: 10000
```

#### Method C: Via DataInput Node

1. In Node Editor, right-click > **Data > DataInput**
2. In Properties panel, set:
   - **Dataset:** mnist
   - **Split:** train
3. The node will show "MNIST (60000 samples)" when connected

### Step 3: Build the Model Architecture

For MNIST, we'll build an MLP (Multi-Layer Perceptron) since CNN layers are not yet trainable in the backend.

#### Visual Node Editor Approach

1. **Create Input Node:**
   - Right-click in Node Editor > **Input/Output > Input**
   - In Properties: Set shape to `[784]` (28x28 flattened)

2. **Add First Dense Layer:**
   - Right-click > **Layers > Dense**
   - Connect Input output to Dense input
   - In Properties: Set units to `512`

3. **Add Activation:**
   - Right-click > **Activations > ReLU**
   - Connect Dense output to ReLU input

4. **Add Dropout (Regularization):**
   - Right-click > **Layers > Dropout**
   - Connect ReLU output to Dropout input
   - In Properties: Set rate to `0.2`

5. **Add Second Dense Layer:**
   - Right-click > **Layers > Dense**
   - Connect Dropout output to Dense input
   - In Properties: Set units to `256`

6. **Add Second Activation:**
   - Right-click > **Activations > ReLU**
   - Connect Dense output to ReLU input

7. **Add Second Dropout:**
   - Right-click > **Layers > Dropout**
   - In Properties: Set rate to `0.2`

8. **Add Output Layer:**
   - Right-click > **Layers > Dense**
   - In Properties: Set units to `10` (for 10 digit classes)

9. **Add Softmax:**
   - Right-click > **Activations > Softmax**
   - Connect to Dense output

10. **Create Output Node:**
    - Right-click > **Input/Output > Output**
    - Connect Softmax output to Output input

#### Final Architecture Diagram
```
Input [784]
    |
Dense (512) --> ReLU --> Dropout (0.2)
    |
Dense (256) --> ReLU --> Dropout (0.2)
    |
Dense (10) --> Softmax
    |
Output [10]
```

### Step 4: Configure Training Parameters

1. **Open Training Panel:**
   - Go to **View > Training Panel** or find it in the interface

2. **Set Training Configuration:**

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| **Epochs** | 10-20 | Start with 10 for quick test |
| **Batch Size** | 64 | Increase to 128 if GPU available |
| **Learning Rate** | 0.001 | Adam optimizer default |
| **Optimizer** | Adam | Best for most cases |
| **Loss Function** | CrossEntropyLoss | For classification |
| **Validation Split** | 0.1 | 10% of training data |

3. **Advanced Options (Optional):**
   - **Weight Decay:** 1e-4 (L2 regularization)
   - **LR Scheduler:** StepLR with step=5, gamma=0.5
   - **Early Stopping:** patience=5, min_delta=0.001

### Step 5: Start Training

1. **Verify Graph Validity:**
   - Look for green checkmark in toolbar
   - Or go to **Tools > Validate Graph**
   - Fix any red error nodes before proceeding

2. **Compile the Graph:**
   - Go to **Tools > Compile Graph** or press `Ctrl+B`
   - Console will show compilation status:
   ```
   [INFO] Compiling graph...
   [INFO] Found 1 input node, 1 output node
   [INFO] Sequential model with 8 layers
   [INFO] Total parameters: 533,770
   [INFO] Compilation successful
   ```

3. **Start Training:**
   - Click **"Start Training"** button in Training Panel
   - Or go to **Tools > Start Training** or press `F5`

### Step 6: Monitor Training Progress

During training, observe:

1. **Training Plot Panel** (View > Training Plot):
   - **Loss Curve:** Should decrease over epochs
   - **Accuracy Curve:** Should increase toward 95%+
   - **Learning Rate:** Shows scheduler effect if enabled

2. **Console Output:**
   ```
   [Epoch 1/10] Loss: 0.4523, Acc: 86.12%, Val_Loss: 0.2134, Val_Acc: 93.45%
   [Epoch 2/10] Loss: 0.1876, Acc: 94.32%, Val_Loss: 0.1456, Val_Acc: 95.67%
   [Epoch 3/10] Loss: 0.1234, Acc: 96.21%, Val_Loss: 0.1123, Val_Acc: 96.78%
   ...
   [Epoch 10/10] Loss: 0.0456, Acc: 98.67%, Val_Loss: 0.0823, Val_Acc: 97.45%

   Training complete!
   Best validation accuracy: 97.45% at epoch 8
   ```

3. **Memory Monitor** (View > Memory Monitor):
   - GPU VRAM usage
   - System RAM usage
   - Training batch throughput

### Step 7: Evaluate Results

#### View Final Metrics

1. **Training Summary:**
   - Final training accuracy
   - Final validation accuracy
   - Best epoch checkpoint

2. **Test Set Evaluation:**
   ```python
   # In Console
   from pycyxwiz import Model

   model = Model.current()
   test_acc = model.evaluate("mnist", split="test")
   print(f"Test Accuracy: {test_acc:.2f}%")
   ```

   **Expected Result:** ~97-98% accuracy on test set

#### Save Your Model

1. **Save Graph:**
   - **File > Save As** or `Ctrl+Shift+S`
   - Save as `mnist_mlp.cyxgraph`

2. **Export Trained Weights:**
   - **File > Export Model**
   - Choose format: `.cyxmodel` (native) or `.onnx`

---

## Alternative: Using Pre-built Patterns

CyxWiz Engine includes 25 pre-built architecture patterns. For MNIST:

1. **Open Pattern Library:**
   - **File > New from Pattern** or `Ctrl+Shift+N`

2. **Select MNIST-compatible Pattern:**
   - **LeNet-5** - Classic CNN (Note: Conv layers not trainable yet)
   - **MLP Classifier** - Fully connected network (RECOMMENDED)
   - **Simple Dense** - Minimal architecture

3. **Click "Apply Pattern"**

4. **The graph will be populated automatically**

### Available Patterns Suitable for MNIST

| Pattern | Description | Parameters |
|---------|-------------|------------|
| MLP Classifier | 784→512→256→10 | ~535K |
| Simple Dense | 784→128→10 | ~101K |
| Deep MLP | 784→512→256→128→10 | ~600K |

---

## Alternative: Python Scripting

For users who prefer code over visual editing:

```python
# mnist_train.py - Run in CyxWiz Console

import pycyxwiz as cyx
from pycyxwiz import Sequential, Dense, Dropout, Activation
from pycyxwiz import Adam, CrossEntropyLoss
from pycyxwiz import DataRegistry

# Load data
train_data = DataRegistry.load("mnist", split="train")
test_data = DataRegistry.load("mnist", split="test")

# Build model
model = Sequential([
    Dense(784, 512),
    Activation("relu"),
    Dropout(0.2),
    Dense(512, 256),
    Activation("relu"),
    Dropout(0.2),
    Dense(256, 10),
    Activation("softmax")
])

# Compile
model.compile(
    optimizer=Adam(lr=0.001),
    loss=CrossEntropyLoss(),
    metrics=["accuracy"]
)

# Train
history = model.fit(
    train_data,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=True
)

# Evaluate
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc:.2%}")

# Save
model.save("mnist_model.cyxmodel")
```

---

## Logging and Assertions

### Enable Verbose Logging

1. **In Engine Settings:**
   - **Edit > Preferences > Logging**
   - Set log level to "Debug"

2. **Via Console:**
   ```python
   import pycyxwiz as cyx
   cyx.set_log_level("debug")
   ```

### Adding Assertions to Training

To verify computed results at various stages:

```python
# Assert input shape
assert train_data.shape == (60000, 784), f"Unexpected shape: {train_data.shape}"

# Assert output shape
output = model.predict(test_data[:1])
assert output.shape == (1, 10), f"Unexpected output shape: {output.shape}"

# Assert probability distribution
assert abs(output.sum() - 1.0) < 1e-5, "Softmax output should sum to 1"

# Assert accuracy bounds
assert test_acc > 0.90, f"Accuracy too low: {test_acc}"
```

### Logging Computed Results

The training system logs intermediate results that can be viewed:

1. **Layer Outputs:**
   ```python
   # Enable layer output logging
   model.set_debug_mode(True)

   # After forward pass, access outputs
   for name, output in model.layer_outputs.items():
       print(f"{name}: shape={output.shape}, mean={output.mean():.4f}")
   ```

2. **Gradient Magnitudes:**
   ```python
   # After backward pass
   for name, grad in model.gradients.items():
       print(f"{name}: grad_norm={grad.norm():.4f}")
   ```

3. **Training Metrics Per Batch:**
   - Visible in Training Plot Panel
   - Or via callback:
   ```python
   def on_batch_end(batch, logs):
       print(f"Batch {batch}: loss={logs['loss']:.4f}")

   model.fit(..., callbacks=[on_batch_end])
   ```

---

## Known Limitations

### Current System Limitations

| Limitation | Description | Workaround |
|------------|-------------|------------|
| **No CNN Training** | Conv2D, MaxPool nodes exist but backend training not implemented | Use MLP architecture |
| **Sequential Only** | No branching/residual connections | Use linear layer stacks |
| **Local Training Only** | Network deployment not yet available | Run on local machine |
| **No Recurrent Layers** | LSTM, GRU training not implemented | Use Dense for sequence tasks |
| **Limited Batch Norm** | May have stability issues | Use Dropout instead |
| **Single GPU** | Multi-GPU training not supported | Use largest single GPU |

### Expected Accuracy with MLP

Without convolutional layers, expect:
- **MLP (784→512→256→10):** ~97-98% accuracy
- **Simple Dense (784→128→10):** ~96-97% accuracy
- **CNN (if supported):** Would achieve ~99%+ accuracy

### Memory Considerations

| Model Size | GPU VRAM | System RAM |
|------------|----------|------------|
| Simple MLP (~100K params) | ~500MB | ~2GB |
| Standard MLP (~500K params) | ~1GB | ~4GB |
| Large MLP (~2M params) | ~2GB | ~8GB |

---

## Troubleshooting

### Common Issues

#### "Dataset not found"
```
Solution:
1. Check internet connection (MNIST auto-downloads)
2. Manually download from http://yann.lecun.com/exdb/mnist/
3. Place files in ~/.cyxwiz/datasets/mnist/
```

#### "Training stuck at epoch 1"
```
Solution:
1. Reduce batch size (try 32)
2. Lower learning rate (try 0.0001)
3. Check for NaN in loss (indicates exploding gradients)
```

#### "Out of memory"
```
Solution:
1. Reduce batch size
2. Use smaller model (fewer units per layer)
3. Close other GPU applications
4. Enable gradient checkpointing if available
```

#### "Graph validation failed"
```
Solution:
1. Check all nodes are connected
2. Verify input/output shapes match
3. Ensure exactly 1 Input and 1 Output node
4. Look for red highlight on problematic nodes
```

#### "Loss is NaN"
```
Solution:
1. Reduce learning rate by 10x
2. Add gradient clipping
3. Check for division by zero in custom layers
4. Normalize input data
```

### Getting Help

- **Console Errors:** Check the Console panel for detailed error messages
- **Log Files:** Located at `~/.cyxwiz/logs/`
- **GitHub Issues:** Report bugs at project repository

---

## Next Steps

After completing this demo:

1. **Experiment with architectures** - Try different layer sizes, dropout rates
2. **Try other datasets** - Fashion-MNIST, CIFAR-10 (when CNN support added)
3. **Export to other frameworks** - Use code generation for PyTorch/TensorFlow
4. **Contribute** - Help implement CNN training backend!

---

## Appendix: MNIST Dataset Details

| Property | Value |
|----------|-------|
| **Images** | 70,000 (60K train, 10K test) |
| **Image Size** | 28x28 pixels, grayscale |
| **Classes** | 10 (digits 0-9) |
| **File Size** | ~11MB compressed |
| **Format** | IDX (auto-converted to tensor) |

### Sample Visualization

```
Training sample #0:
Label: 5
Image:
        ████████
      ██████████████
    ████          ████
    ████            ██
      ████████████████
            ██████████
                    ██
    ██              ██
    ████████████████
```

---

*Document Version: 1.0*
*Last Updated: 2025-12-11*
*CyxWiz Engine Version: 0.2.x*
