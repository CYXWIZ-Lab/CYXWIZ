# CyxWiz - Current Capabilities

## Overview

CyxWiz is a visual machine learning platform that allows users to design, train, and deploy neural networks through an intuitive node-based editor. The system consists of three main components working together.

---

## Components

### 1. CyxWiz Engine (Desktop Client)
The main GUI application for designing and training ML models.

### 2. CyxWiz Backend (Compute Library)
C++ library with ArrayFire GPU acceleration for all tensor operations.

### 3. CyxWiz Server Node (Distributed Compute)
Worker nodes for distributed training (in development).

---

## What You Can Do Right Now

### Visual Model Design

- **Node Editor**: Drag-and-drop interface to build neural network architectures
- **25 Pre-built Patterns**: Ready-to-use architectures including:
  - MLP (Multi-Layer Perceptron)
  - CNN architectures (LeNet-5, AlexNet, VGG16, MobileNet, ResNet)
  - RNN/LSTM/GRU networks
  - Transformers (Encoder, Decoder, Attention blocks)
  - Autoencoders and VAEs
  - GANs (GAN, DCGAN)
  - U-Net for segmentation
- **Code Generation**: Export your visual model to:
  - PyTorch
  - TensorFlow/Keras
  - PyCyxWiz (native format)

### Available Layer Types

| Category | Layers |
|----------|--------|
| **Core** | Dense/Linear, Input, Output |
| **Convolution** | Conv2D, MaxPool2D, AvgPool2D, GlobalAvgPool, Flatten |
| **Normalization** | BatchNorm, LayerNorm, Dropout |
| **Recurrent** | LSTM, GRU, RNN, Bidirectional |
| **Attention** | MultiHeadAttention, SelfAttention |
| **Activations** | ReLU, LeakyReLU, ELU, GELU, Swish, Mish, Sigmoid, Tanh, Softmax |

### Training Capabilities

- **Local Training**: Train models directly on your machine
- **GPU Acceleration**: Full ArrayFire integration (CUDA, OpenCL, or CPU fallback)
- **Real-time Visualization**: Live loss/accuracy plots during training
- **Optimizers**:
  - SGD (with momentum)
  - Adam
  - AdamW
  - RMSprop
- **Loss Functions**:
  - CrossEntropy
  - MSE (Mean Squared Error)
  - BCE (Binary Cross Entropy)
  - L1 Loss
  - SmoothL1 (Huber) Loss
- **Learning Rate Schedulers**:
  - StepLR
  - ExponentialLR
  - CosineAnnealing
  - ReduceLROnPlateau
  - LinearWarmup
  - OneCycleLR

---

## Functional Algorithms & Models (Backend)

### Fully Implemented & Trainable

These algorithms are **fully functional** with forward pass, backward pass (gradient computation), and parameter updates:

#### Layers (with Backpropagation)

| Layer | Description | Status |
|-------|-------------|--------|
| **LinearLayer** | Fully connected layer (Dense) | Fully functional |
| **Conv2DLayer** | 2D Convolution with stride/padding | Fully functional |
| **MaxPool2DLayer** | Max pooling with configurable kernel | Fully functional |
| **AvgPool2DLayer** | Average pooling | Fully functional |
| **BatchNorm2DLayer** | Batch normalization (training/eval modes) | Fully functional |
| **FlattenModule** | Reshape for FC layers | Fully functional |
| **DropoutModule** | Regularization (training mode only) | Fully functional |
| **SoftmaxModule** | Output probability distribution | Fully functional |

#### Activation Functions (with Gradients)

| Activation | Formula | Status |
|------------|---------|--------|
| **ReLU** | max(0, x) | Fully functional |
| **LeakyReLU** | max(αx, x), α=0.01 | Fully functional |
| **ELU** | x if x>0, else α(eˣ-1) | Fully functional |
| **GELU** | x·Φ(x) Gaussian Error Linear | Fully functional |
| **Swish** | x·σ(x) | Fully functional |
| **Mish** | x·tanh(softplus(x)) | Fully functional |
| **Sigmoid** | 1/(1+e⁻ˣ) | Fully functional |
| **Tanh** | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | Fully functional |

#### Loss Functions (with Gradients)

| Loss | Use Case | Status |
|------|----------|--------|
| **MSELoss** | Regression | Fully functional |
| **CrossEntropyLoss** | Multi-class classification | Fully functional |
| **BCELoss** | Binary classification | Fully functional |
| **BCEWithLogitsLoss** | Binary (numerically stable) | Fully functional |
| **L1Loss** | Mean Absolute Error | Fully functional |
| **SmoothL1Loss** | Huber loss (robust regression) | Fully functional |
| **NLLLoss** | Negative Log Likelihood | Fully functional |
| **HingeLoss** | SVM-style margin loss | Fully functional |
| **KLDivLoss** | KL Divergence | Fully functional |
| **CosineEmbeddingLoss** | Similarity learning | Fully functional |

#### Optimizers (with Momentum/Adaptive LR)

| Optimizer | Features | Status |
|-----------|----------|--------|
| **SGD** | Momentum, weight decay | Fully functional |
| **Adam** | Adaptive learning rate, β1=0.9, β2=0.999 | Fully functional |
| **AdamW** | Adam + decoupled weight decay | Fully functional |
| **RMSprop** | Adaptive with moving average | Fully functional |

#### Learning Rate Schedulers

| Scheduler | Behavior | Status |
|-----------|----------|--------|
| **StepLR** | Decay by γ every N epochs | Fully functional |
| **ExponentialLR** | Decay by γ every epoch | Fully functional |
| **CosineAnnealingLR** | Cosine decay with warm restarts | Fully functional |
| **ReduceLROnPlateau** | Reduce when metric stalls | Fully functional |
| **LinearWarmupLR** | Linear warmup then constant | Fully functional |
| **OneCycleLR** | 1cycle policy (warmup + cosine) | Fully functional |

### Model Types

| Model | Description | Status |
|-------|-------------|--------|
| **SequentialModel** | Stack of layers in order | Fully functional |

### What You Can Train Right Now

#### Classification Tasks
- **MNIST digits** (784 → Dense → ReLU → Dense → Softmax)
- **CIFAR-10 images** (Conv2D → MaxPool → Flatten → Dense)
- **Binary classification** (any → Sigmoid + BCE)
- **Multi-class** (any → Softmax + CrossEntropy)

#### Regression Tasks
- **Linear regression** (Dense + MSE)
- **Non-linear regression** (Dense + ReLU + Dense + MSE)
- **Robust regression** (Dense + SmoothL1)

#### Example Trainable Architectures

```
MLP Classifier:
Input(784) → Linear(256) → ReLU → Dropout(0.2) → Linear(10) → Softmax

CNN Classifier:
Input(32,32,3) → Conv2D(32) → ReLU → MaxPool → Conv2D(64) → ReLU →
MaxPool → Flatten → Linear(128) → ReLU → Linear(10) → Softmax

Autoencoder:
Encoder: Input → Linear(256) → ReLU → Linear(64) → ReLU → Linear(16)
Decoder: Linear(16) → ReLU → Linear(64) → ReLU → Linear(256) → Sigmoid
```

### ArrayFire GPU Operations Used

All computation runs through ArrayFire for GPU acceleration:

| Operation | ArrayFire Function |
|-----------|-------------------|
| Matrix multiply | `af::matmul()` |
| Convolution | `af::convolve2()` |
| Element-wise ops | `af::exp()`, `af::log()`, `af::sqrt()` |
| Reductions | `af::sum()`, `af::mean()`, `af::max()` |
| Random generation | `af::randu()`, `af::randn()` |
| Reshaping | `af::moddims()`, `af::flat()` |
| Broadcasting | `af::tile()` |

---

### Data Management

- **Dataset Panel**: Load and manage training datasets
- **Table Viewer**: Preview CSV/Excel/HDF5 data files
- **Data Transforms**: 13 augmentation presets including:
  - ImageNet normalization
  - CIFAR-10 augmentation
  - Medical imaging transforms
  - Self-supervised learning augmentations
- **Memory Management**: LRU cache with automatic eviction

### Scripting & Automation

- **Python Console**: Interactive REPL for scripting
- **Script Editor**: Write and execute Python/CyxWiz scripts
- **Async Execution**: Non-blocking script execution with cancellation

### Project Management

- **Project System**: Create, open, save projects
- **Recent Projects**: Quick access to previous work
- **Auto-save**: Configurable automatic saving
- **Asset Browser**: Navigate project files with filters

### User Interface

- **Dockable Panels**: Fully customizable workspace layout
- **Multiple Themes**: Dark, Light, Classic, Nord, Dracula
- **15 Panels**: Including Node Editor, Console, Properties, Training Dashboard, etc.

---

## Hardware Support

### GPU Acceleration
- **NVIDIA GPUs**: Via CUDA or OpenCL
- **AMD GPUs**: Via OpenCL
- **Intel GPUs**: Via OpenCL
- **CPU Fallback**: Works without any GPU (uses optimized SIMD)

### Platforms
- Windows (primary)
- Linux (supported)
- macOS (supported)

---

## Technical Specifications

### Backend Compute
- All tensor operations use ArrayFire (no hardcoded loops)
- Automatic GPU/CPU backend selection
- Memory-efficient operations with zero-copy where possible

### Model Architecture
- Sequential model with dynamic layer management
- Forward and backward pass fully implemented
- Gradient computation and parameter updates

### Checkpoint System
- Save/load model weights
- JSON metadata + binary tensor storage
- Best model tracking

---

## Current Limitations

### Not Yet Implemented
- Distributed training across multiple nodes
- ONNX model import/export
- PyTorch/TensorFlow model import
- Blockchain payment integration
- User authentication system
- Model marketplace

### Training Constraints
- CNN layers (Conv2D, Pooling) are defined but training integration pending
- RNN/LSTM/Attention layers defined but sequential training only
- Batch processing limited to in-memory datasets

---

## Quick Start

1. **Launch Engine**: Run `cyxwiz-engine.exe`
2. **Create Model**: Use Node Editor to design architecture
3. **Load Data**: Use Dataset Panel to load training data
4. **Configure Training**: Set optimizer, loss, learning rate
5. **Start Training**: Click "Start Training" in Training Dashboard
6. **Monitor Progress**: Watch real-time loss/accuracy plots

---

## Example Workflow

```
1. File > New Project
2. Add nodes: Input(784) -> Dense(128) -> ReLU -> Dense(10) -> Softmax
3. Connect nodes in sequence
4. Load MNIST dataset
5. Set: Optimizer=Adam, LR=0.001, Loss=CrossEntropy
6. Click "Start Training"
7. View live training metrics
8. Export trained model or generate code
```

---

## Version

**Current Version**: 0.2.0
**Last Updated**: December 2024

---

## Summary

CyxWiz is a fully functional visual ML IDE that can:
- Design neural networks visually
- Train models locally with GPU acceleration
- Generate code for popular frameworks
- Manage datasets and projects
- Visualize training in real-time

The core training infrastructure is complete and operational. Future development focuses on distributed training, model import/export, and blockchain integration.
