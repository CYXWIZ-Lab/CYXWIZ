# CyxWiz Backend

**High-Performance ML Compute Library with GPU Acceleration**

CyxWiz Backend is a shared library (DLL/SO) that provides the core ML computation primitives for the CyxWiz distributed ML platform. It is used by both the **CyxWiz Engine** (desktop client) and **CyxWiz Server Node** (compute worker).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CyxWiz Backend Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐                           ┌──────────────────┐        │
│  │   CyxWiz Engine  │                           │ CyxWiz Server    │        │
│  │  (Desktop Client)│                           │     Node         │        │
│  │                  │                           │ (Compute Worker) │        │
│  │  - Visual IDE    │                           │  - Job Executor  │        │
│  │  - Node Editor   │                           │  - GPU Training  │        │
│  │  - Local Train   │                           │  - Distributed   │        │
│  └────────┬─────────┘                           └────────┬─────────┘        │
│           │                                              │                   │
│           │  Links to DLL/SO                             │                   │
│           │                                              │                   │
│           ▼                                              ▼                   │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                     cyxwiz-backend.dll / .so                       │      │
│  │                                                                    │      │
│  │  ┌────────────────────────────────────────────────────────────┐   │      │
│  │  │                     PUBLIC API (C++)                        │   │      │
│  │  │  #include "cyxwiz/cyxwiz.h"                                │   │      │
│  │  └────────────────────────────────────────────────────────────┘   │      │
│  │                              │                                     │      │
│  │  ┌───────────────────────────┼───────────────────────────────┐    │      │
│  │  │                    CORE LAYER                              │    │      │
│  │  │                                                            │    │      │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │    │      │
│  │  │  │  Tensor  │  │  Device  │  │  Memory  │  │  Engine  │   │    │      │
│  │  │  │          │  │  Manager │  │  Manager │  │          │   │    │      │
│  │  │  │ - Shape  │  │          │  │          │  │- Init    │   │    │      │
│  │  │  │ - Data   │  │ - CPU    │  │ - Alloc  │  │- Shutdown│   │    │      │
│  │  │  │ - Math   │  │ - CUDA   │  │ - Pool   │  │- Version │   │    │      │
│  │  │  │ - GPU    │  │ - OpenCL │  │ - Track  │  │          │   │    │      │
│  │  │  │          │  │ - Metal  │  │          │  │          │   │    │      │
│  │  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │    │      │
│  │  └────────────────────────────────────────────────────────────┘    │      │
│  │                              │                                     │      │
│  │  ┌───────────────────────────┼───────────────────────────────┐    │      │
│  │  │                  NEURAL NETWORK LAYER                      │    │      │
│  │  │                                                            │    │      │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │    │      │
│  │  │  │  Layers  │  │Activations│ │  Losses  │  │Optimizers│   │    │      │
│  │  │  │          │  │          │  │          │  │          │   │    │      │
│  │  │  │- Dense   │  │- ReLU    │  │- MSE     │  │- SGD     │   │    │      │
│  │  │  │- Conv2D  │  │- Sigmoid │  │- CE      │  │- Adam    │   │    │      │
│  │  │  │- Pool2D  │  │- Tanh    │  │- BCE     │  │- AdamW   │   │    │      │
│  │  │  │- BatchN  │  │- GELU    │  │- NLL     │  │- RMSprop │   │    │      │
│  │  │  │- Dropout │  │- Swish   │  │- Huber   │  │- AdaGrad │   │    │      │
│  │  │  │- Flatten │  │- Mish    │  │- KL      │  │          │   │    │      │
│  │  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │    │      │
│  │  │                                                            │    │      │
│  │  │  ┌──────────────────────────────────────────────────────┐  │    │      │
│  │  │  │                  Sequential Model                     │  │    │      │
│  │  │  │  - Add layers dynamically                             │  │    │      │
│  │  │  │  - Forward/Backward pass                              │  │    │      │
│  │  │  │  - Save/Load weights                                  │  │    │      │
│  │  │  │  - Transfer learning (Freeze/Unfreeze)                │  │    │      │
│  │  │  └──────────────────────────────────────────────────────┘  │    │      │
│  │  └────────────────────────────────────────────────────────────┘    │      │
│  │                              │                                     │      │
│  │  ┌───────────────────────────┼───────────────────────────────┐    │      │
│  │  │                 ALGORITHMS LAYER                           │    │      │
│  │  │                                                            │    │      │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │    │      │
│  │  │  │ Linear   │  │ Signal   │  │Clustering│  │  Time    │   │    │      │
│  │  │  │ Algebra  │  │Processing│  │          │  │ Series   │   │    │      │
│  │  │  │          │  │          │  │          │  │          │   │    │      │
│  │  │  │- SVD     │  │- FFT     │  │- KMeans  │  │- ACF     │   │    │      │
│  │  │  │- Eigen   │  │- Conv    │  │- DBSCAN  │  │- ARIMA   │   │    │      │
│  │  │  │- QR      │  │- Filter  │  │- GMM     │  │- Decomp  │   │    │      │
│  │  │  │- LU      │  │- STFT    │  │- PCA     │  │- Diff    │   │    │      │
│  │  │  │- Solve   │  │- Wavelet │  │- t-SNE   │  │- Rolling │   │    │      │
│  │  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │    │      │
│  │  │                                                            │    │      │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │    │      │
│  │  │  │  Data    │  │  Model   │  │ Feature  │  │Interpret-│   │    │      │
│  │  │  │Transform │  │Evaluation│  │Importance│  │  ability │   │    │      │
│  │  │  │          │  │          │  │          │  │          │   │    │      │
│  │  │  │- Norm    │  │- Metrics │  │- Permute │  │- GradCAM │   │    │      │
│  │  │  │- Std     │  │- ROC/AUC │  │- SHAP    │  │- Saliency│   │    │      │
│  │  │  │- BoxCox  │  │- Conf.Mat│  │- Mutual  │  │          │   │    │      │
│  │  │  │- YeoJohn │  │- CV      │  │          │  │          │   │    │      │
│  │  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │    │      │
│  │  └────────────────────────────────────────────────────────────┘    │      │
│  │                              │                                     │      │
│  │  ┌───────────────────────────┼───────────────────────────────┐    │      │
│  │  │                  HARDWARE ABSTRACTION                      │    │      │
│  │  │                                                            │    │      │
│  │  │  ┌──────────────────────────────────────────────────────┐  │    │      │
│  │  │  │                    ArrayFire                          │  │    │      │
│  │  │  │  GPU-accelerated numerical computing library          │  │    │      │
│  │  │  │                                                       │  │    │      │
│  │  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │  │    │      │
│  │  │  │  │  CUDA   │  │ OpenCL  │  │  Metal  │  │   CPU   │  │  │    │      │
│  │  │  │  │ (NVIDIA)│  │(AMD/Intel)│ │ (Apple) │  │(Fallback)│ │  │    │      │
│  │  │  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  │  │    │      │
│  │  │  └──────────────────────────────────────────────────────┘  │    │      │
│  │  └────────────────────────────────────────────────────────────┘    │      │
│  │                              │                                     │      │
│  │  ┌───────────────────────────┼───────────────────────────────┐    │      │
│  │  │                   PYTHON BINDINGS                          │    │      │
│  │  │                                                            │    │      │
│  │  │           pycyxwiz (pybind11)                              │    │      │
│  │  │                                                            │    │      │
│  │  │  import pycyxwiz as cx                                     │    │      │
│  │  │  cx.initialize()                                           │    │      │
│  │  │  tensor = cx.Tensor.from_numpy(np_array)                   │    │      │
│  │  │                                                            │    │      │
│  │  └────────────────────────────────────────────────────────────┘    │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
cyxwiz-backend/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This file
├── include/
│   └── cyxwiz/
│       ├── cyxwiz.h            # Main header (include this)
│       ├── api_export.h        # DLL export macros
│       ├── tensor.h            # Multi-dimensional array
│       ├── device.h            # Device management (CPU/GPU)
│       ├── memory_manager.h    # Memory allocation/tracking
│       ├── engine.h            # Backend initialization
│       │
│       ├── layer.h             # Base Layer + Dense, Conv2D, Pool, etc.
│       ├── activation.h        # Activation functions
│       ├── loss.h              # Loss functions
│       ├── optimizer.h         # Optimizers (SGD, Adam, etc.)
│       ├── sequential.h        # SequentialModel container
│       ├── scheduler.h         # Learning rate schedulers
│       │
│       ├── layers/
│       │   └── linear.h        # LinearLayer implementation
│       ├── activations/
│       │   ├── relu.h
│       │   ├── sigmoid.h
│       │   └── tanh.h
│       │
│       ├── linear_algebra.h    # Matrix decompositions, solvers
│       ├── signal_processing.h # FFT, filters, convolution
│       ├── clustering.h        # K-Means, DBSCAN, GMM
│       ├── dimensionality_reduction.h  # PCA, t-SNE, UMAP
│       ├── data_transform.h    # Normalization, standardization
│       ├── time_series.h       # ACF, ARIMA, decomposition
│       ├── model_evaluation.h  # Metrics, ROC, confusion matrix
│       ├── feature_importance.h # SHAP, permutation importance
│       ├── model_interpretability.h # Grad-CAM, saliency
│       ├── optimization.h      # Numerical optimization
│       ├── text_processing.h   # Tokenization, TF-IDF
│       └── utilities.h         # Helper functions
│
├── src/
│   ├── core/
│   │   ├── tensor.cpp          # Tensor implementation
│   │   ├── device.cpp          # Device management
│   │   ├── memory_manager.cpp  # Memory tracking
│   │   └── engine.cpp          # Init/Shutdown
│   │
│   └── algorithms/
│       ├── layer.cpp           # Layer implementations
│       ├── layers/
│       │   └── linear.cpp
│       ├── activation.cpp
│       ├── activations/
│       │   ├── relu.cpp
│       │   ├── sigmoid.cpp
│       │   └── tanh.cpp
│       ├── loss.cpp
│       ├── optimizer.cpp
│       ├── sequential.cpp
│       ├── scheduler.cpp
│       ├── linear_algebra.cpp
│       ├── signal_processing.cpp
│       ├── clustering.cpp
│       ├── dimensionality_reduction.cpp
│       ├── data_transform.cpp
│       ├── time_series.cpp
│       ├── model_evaluation.cpp
│       ├── feature_importance.cpp
│       ├── model_interpretability.cpp
│       ├── optimization.cpp
│       ├── text_processing.cpp
│       └── utilities.cpp
│
└── python/
    └── bindings.cpp            # pybind11 Python bindings
```

## Quick Start

### C++ Usage

```cpp
#include "cyxwiz/cyxwiz.h"
using namespace cyxwiz;

int main() {
    // Initialize backend
    Initialize();

    // Create a simple MLP for MNIST
    SequentialModel model;
    model.Add<LinearModule>(784, 256);    // Input: 28x28 = 784
    model.Add<ReLUModule>();
    model.Add<LinearModule>(256, 128);
    model.Add<ReLUModule>();
    model.Add<LinearModule>(128, 10);     // Output: 10 classes

    // Create optimizer and loss
    auto optimizer = std::make_unique<AdamOptimizer>(0.001);
    CrossEntropyLoss loss;

    // Training loop
    for (int epoch = 0; epoch < 10; epoch++) {
        // Forward pass
        Tensor output = model.Forward(input_batch);

        // Compute loss
        Tensor loss_value = loss.Forward(output, labels);

        // Backward pass
        Tensor grad = loss.Backward(output, labels);
        model.Backward(grad);

        // Update weights
        model.UpdateParameters(optimizer.get());
    }

    // Save model
    model.Save("mnist_model");

    Shutdown();
    return 0;
}
```

### Python Usage

```python
import pycyxwiz as cx
import numpy as np

# Initialize
cx.initialize()

# Check available devices
devices = cx.get_available_devices()
for d in devices:
    print(f"{d.name}: {d.type}, {d.memory_total / 1e9:.1f} GB")

# Create tensor from NumPy
data = np.random.randn(100, 784).astype(np.float32)
tensor = cx.Tensor.from_numpy(data)

# Create layers
layer1 = cx.Dense(784, 256)
relu = cx.ReLU()
layer2 = cx.Dense(256, 10)

# Forward pass
x = layer1.forward(tensor)
x = relu.forward(x)
x = layer2.forward(x)

# Convert back to NumPy
result = x.to_numpy()

# Use linear algebra
import pycyxwiz.linalg as la

A = [[1, 2], [3, 4]]
U, S, Vt = la.svd(A)
print(f"Singular values: {S}")

# Shutdown
cx.shutdown()
```

## API Reference

### Core Components

#### Tensor
Multi-dimensional array with GPU acceleration.

```cpp
// Create tensors
Tensor t1({100, 784}, DataType::Float32);   // Shape: [100, 784]
Tensor t2 = Tensor::Zeros({10, 10});        // All zeros
Tensor t3 = Tensor::Random({256, 256});     // Random [0, 1)

// Operations
Tensor sum = t1 + t2;
Tensor prod = t1 * t2;
Tensor reshaped = t1.Reshape({100, 28, 28});

// GPU transfer
Device* gpu = new Device(DeviceType::CUDA, 0);
t1.ToDevice(gpu);
t1.ToCPU();
```

#### Device
GPU/CPU device management.

```cpp
// List available devices
auto devices = Device::GetAvailableDevices();
for (const auto& info : devices) {
    std::cout << info.name << ": " << info.memory_total << " bytes\n";
}

// Select device
Device gpu(DeviceType::CUDA, 0);
gpu.SetActive();
```

### Neural Network Layers

| Layer | Description | Parameters |
|-------|-------------|------------|
| `DenseLayer` / `LinearModule` | Fully connected | `in_features`, `out_features`, `use_bias` |
| `Conv2DLayer` | 2D Convolution | `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding` |
| `MaxPool2DLayer` | Max Pooling | `pool_size`, `stride`, `padding` |
| `AvgPool2DLayer` | Average Pooling | `pool_size`, `stride`, `padding` |
| `GlobalAvgPool2DLayer` | Global Average Pooling | - |
| `BatchNorm2DLayer` | Batch Normalization | `num_features`, `eps`, `momentum` |
| `DropoutLayer` / `DropoutModule` | Dropout regularization | `p` (drop probability) |
| `FlattenLayer` / `FlattenModule` | Flatten spatial dims | `start_dim` |

### Activation Functions

| Activation | Formula | Module |
|------------|---------|--------|
| ReLU | `max(0, x)` | `ReLUModule` |
| Sigmoid | `1 / (1 + exp(-x))` | `SigmoidModule` |
| Tanh | `tanh(x)` | `TanhModule` |
| LeakyReLU | `x if x > 0 else alpha * x` | `LeakyReLUModule(alpha)` |
| ELU | `x if x > 0 else alpha * (exp(x) - 1)` | `ELUModule(alpha)` |
| GELU | Gaussian Error Linear Unit | `GELUModule` |
| Swish/SiLU | `x * sigmoid(x)` | `SwishModule` |
| Mish | `x * tanh(softplus(x))` | `MishModule` |
| Softmax | `exp(x) / sum(exp(x))` | `SoftmaxModule(dim)` |

### Loss Functions

| Loss | Use Case | Class |
|------|----------|-------|
| MSE | Regression | `MSELoss` |
| L1 | Regression (robust) | `L1Loss` |
| SmoothL1/Huber | Regression (outliers) | `SmoothL1Loss(delta)` |
| CrossEntropy | Multi-class classification | `CrossEntropyLoss` |
| BCE | Binary classification | `BCELoss` |
| BCEWithLogits | Binary (numerically stable) | `BCEWithLogitsLoss` |
| NLL | Classification with log-softmax | `NLLLoss` |
| KLDivergence | Distribution matching | `KLDivLoss` |

### Optimizers

| Optimizer | Description | Parameters |
|-----------|-------------|------------|
| `SGDOptimizer` | Stochastic Gradient Descent | `learning_rate`, `momentum` |
| `AdamOptimizer` | Adaptive Moment Estimation | `learning_rate`, `beta1`, `beta2`, `epsilon` |
| `AdamWOptimizer` | Adam with weight decay | `learning_rate`, `beta1`, `beta2`, `epsilon`, `weight_decay` |

### SequentialModel

Container for building models layer by layer.

```cpp
SequentialModel model;
model.Add<LinearModule>(784, 256);
model.Add<ReLUModule>();
model.Add<DropoutModule>(0.5f);
model.Add<LinearModule>(256, 10);

// Training
model.SetTraining(true);
Tensor output = model.Forward(input);
Tensor grad = model.Backward(loss_grad);
model.UpdateParameters(optimizer);

// Inference
model.SetTraining(false);
Tensor prediction = model.Forward(test_input);

// Save/Load
model.Save("model_path");  // Creates .json + .bin
model.Load("model_path");

// Transfer Learning
model.FreezeUpTo(4);       // Freeze first 4 layers
model.FreezeExceptLast(2); // Keep only last 2 trainable
model.UnfreezeAll();       // Unfreeze all
```

## Data Flow: Engine to Server Node

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         P2P TRAINING FLOW                                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  CyxWiz Engine (User's Machine)                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                                                                    │  │
│  │   1. User builds model in Node Editor                              │  │
│  │      ┌─────────┐   ┌─────────┐   ┌─────────┐                      │  │
│  │      │ Input   │──▶│  Dense  │──▶│  ReLU   │──▶ ...               │  │
│  │      │ (784)   │   │ (256)   │   │         │                      │  │
│  │      └─────────┘   └─────────┘   └─────────┘                      │  │
│  │                                                                    │  │
│  │   2. Engine uses cyxwiz-backend to:                               │  │
│  │      - Validate model architecture                                 │  │
│  │      - Serialize model definition                                  │  │
│  │      - Preview local training (optional)                           │  │
│  │                                                                    │  │
│  │   3. Engine sends to Server Node via gRPC:                        │  │
│  │      - Model definition (JSON)                                     │  │
│  │      - Hyperparameters (lr, epochs, batch_size)                   │  │
│  │      - Dataset reference or stream                                 │  │
│  │                                                                    │  │
│  └──────────────────────────┬─────────────────────────────────────────┘  │
│                             │                                             │
│                             │  gRPC P2P Connection                       │
│                             ▼                                             │
│  CyxWiz Server Node (Remote GPU Machine)                                 │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                                                                    │  │
│  │   4. Server Node uses SAME cyxwiz-backend DLL to:                 │  │
│  │                                                                    │  │
│  │      a) Reconstruct model from definition                          │  │
│  │         SequentialModel model;                                     │  │
│  │         model.Add<LinearModule>(784, 256);                         │  │
│  │         model.Add<ReLUModule>();                                   │  │
│  │         ...                                                        │  │
│  │                                                                    │  │
│  │      b) Load data onto GPU                                         │  │
│  │         Device gpu(DeviceType::CUDA, 0);                           │  │
│  │         gpu.SetActive();                                           │  │
│  │         tensor.ToDevice(&gpu);                                     │  │
│  │                                                                    │  │
│  │      c) Execute training loop                                      │  │
│  │         for epoch in range(epochs):                                │  │
│  │             output = model.Forward(batch);                         │  │
│  │             loss = loss_fn.Forward(output, labels);                │  │
│  │             grad = loss_fn.Backward(output, labels);               │  │
│  │             model.Backward(grad);                                  │  │
│  │             model.UpdateParameters(optimizer);                     │  │
│  │                                                                    │  │
│  │      d) Stream progress back to Engine                             │  │
│  │         - Current epoch, batch                                     │  │
│  │         - Loss, accuracy metrics                                   │  │
│  │         - GPU utilization                                          │  │
│  │                                                                    │  │
│  │   5. On completion, send trained weights back                      │  │
│  │      - model.Save() -> binary weights file                         │  │
│  │      - Stream back to Engine via gRPC                              │  │
│  │                                                                    │  │
│  └──────────────────────────┬─────────────────────────────────────────┘  │
│                             │                                             │
│                             │  Trained Weights                           │
│                             ▼                                             │
│  CyxWiz Engine                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                                                                    │  │
│  │   6. Load trained model                                            │  │
│  │      model.Load("trained_weights");                                │  │
│  │                                                                    │  │
│  │   7. Use for inference locally                                     │  │
│  │      Tensor prediction = model.Forward(input);                     │  │
│  │                                                                    │  │
│  │   8. Export to ONNX for deployment                                 │  │
│  │      model.ExportONNX("model.onnx");                               │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

## Building

### Prerequisites
- CMake 3.20+
- C++20 compiler (MSVC 2022, GCC 11+, Clang 14+)
- ArrayFire (optional, for GPU acceleration)
- Python 3.8+ (for Python bindings)
- pybind11 (via vcpkg)

### Build Commands

```bash
# From repository root
cmake --preset windows-release
cmake --build build/windows-release --target cyxwiz-backend

# Output: build/bin/Release/cyxwiz-backend.dll (Windows)
#         build/lib/Release/libcyxwiz-backend.so (Linux)
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CYXWIZ_HAS_ARRAYFIRE` | ON if found | Enable GPU acceleration |
| `CYXWIZ_BUILD_PYTHON` | ON | Build Python bindings |
| `CYXWIZ_DEBUG` | Debug builds | Enable debug logging |

## GPU Acceleration

The backend automatically uses GPU when available:

1. **ArrayFire Detection**: CMake checks for ArrayFire installation
2. **Backend Selection**: Runtime selection of CUDA/OpenCL/CPU
3. **Transparent API**: Same code works on CPU and GPU

```cpp
// Check GPU availability at runtime
auto devices = Device::GetAvailableDevices();
bool has_cuda = false;
for (const auto& d : devices) {
    if (d.type == DeviceType::CUDA) {
        has_cuda = true;
        Device gpu(DeviceType::CUDA, d.device_id);
        gpu.SetActive();
        break;
    }
}

if (!has_cuda) {
    // Falls back to CPU automatically
    Device cpu(DeviceType::CPU, 0);
    cpu.SetActive();
}
```

## Thread Safety

- **Tensor operations**: Thread-safe for independent tensors
- **Device selection**: Use separate Device instances per thread
- **Model training**: Single-threaded per model instance
- **Memory Manager**: Thread-safe allocation/deallocation

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2024-12 | Initial release with core ML primitives |

## License

Part of the CyxWiz distributed ML platform. See repository root for license.
