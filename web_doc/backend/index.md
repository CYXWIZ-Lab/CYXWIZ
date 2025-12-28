# Backend API Reference

The `cyxwiz-backend` library provides the core computational functionality for CyxWiz, including tensor operations, neural network layers, optimizers, and GPU-accelerated algorithms.

## Overview

The backend is a shared library (DLL/SO) used by:
- **CyxWiz Engine** - For local training and inference
- **CyxWiz Server Node** - For distributed job execution
- **pycyxwiz** - Python bindings for scripting

## Architecture

```
cyxwiz-backend/
├── include/cyxwiz/
│   ├── cyxwiz.h           # Main header (include this)
│   ├── api_export.h       # DLL export macros
│   ├── tensor.h           # Tensor class
│   ├── device.h           # GPU/CPU device management
│   ├── layer.h            # Base layer class
│   ├── optimizer.h        # Optimizer base
│   ├── loss.h             # Loss functions
│   ├── activation.h       # Activation functions
│   ├── model.h            # Model training interface
│   ├── sequential.h       # Sequential model
│   ├── scheduler.h        # LR schedulers
│   └── [algorithm].h      # Domain algorithms
└── src/
    ├── core/              # Core implementations
    └── algorithms/        # Algorithm implementations
```

## Documentation Sections

| Section | Description |
|---------|-------------|
| [Tensor Operations](api/tensor.md) | Core tensor class and operations |
| [Device Management](api/device.md) | GPU/CPU device selection |
| [Neural Network Layers](api/layers.md) | Layer implementations |
| [Optimizers](api/optimizers.md) | Training optimizers |
| [Loss Functions](api/loss.md) | Loss function implementations |
| [Activation Functions](api/activations.md) | Activation implementations |
| [Sequential Model](api/sequential.md) | High-level training API |
| [LR Schedulers](api/schedulers.md) | Learning rate scheduling |
| [Clustering](api/clustering.md) | Clustering algorithms |
| [Data Transforms](api/transforms.md) | Preprocessing functions |
| [Model Evaluation](api/evaluation.md) | Metrics and evaluation |
| [Linear Algebra](api/linalg.md) | Matrix operations |
| [Signal Processing](api/signal.md) | DSP functions |
| [Python Bindings](api/python.md) | pycyxwiz module |

## Quick Start

### C++ Usage

```cpp
#include <cyxwiz/cyxwiz.h>

int main() {
    // Initialize backend
    cyxwiz::Initialize();

    // Create tensor
    cyxwiz::Tensor x({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});

    // Build model
    cyxwiz::Sequential model;
    model.Add(std::make_unique<cyxwiz::Linear>(2, 4));
    model.Add(std::make_unique<cyxwiz::ReLU>());
    model.Add(std::make_unique<cyxwiz::Linear>(4, 1));

    // Forward pass
    cyxwiz::Tensor output = model.Forward(x);

    // Cleanup
    cyxwiz::Shutdown();
    return 0;
}
```

### Python Usage

```python
import pycyxwiz as cyx

# Initialize
cyx.initialize()

# Create tensor
x = cyx.Tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])

# Build model
model = cyx.Sequential()
model.add(cyx.Linear(2, 4))
model.add(cyx.ReLU())
model.add(cyx.Linear(4, 1))

# Forward pass
output = model.forward(x)
print(output.data())
```

## Core Classes

### Tensor

The fundamental data structure for multi-dimensional arrays.

```cpp
class CYXWIZ_API Tensor {
public:
    // Constructors
    Tensor();
    Tensor(const std::vector<float>& data, const std::vector<int>& shape);
    Tensor(const std::vector<int>& shape);  // Uninitialized

    // Shape operations
    std::vector<int> Shape() const;
    int NumElements() const;
    int NumDimensions() const;

    // Data access
    std::vector<float> ToVector() const;
    float& operator[](int index);

    // Math operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    // Matrix operations
    Tensor MatMul(const Tensor& other) const;
    Tensor Transpose() const;

    // Reductions
    Tensor Sum(int axis = -1) const;
    Tensor Mean(int axis = -1) const;
    Tensor Max(int axis = -1) const;
    Tensor Min(int axis = -1) const;

    // Element-wise operations
    Tensor Exp() const;
    Tensor Log() const;
    Tensor Sqrt() const;
    Tensor Pow(float exponent) const;

    // Reshape
    Tensor Reshape(const std::vector<int>& new_shape) const;
    Tensor Flatten() const;
};
```

### Device

Manages GPU/CPU device selection.

```cpp
class CYXWIZ_API Device {
public:
    enum class Type { CPU, CUDA, OpenCL };

    static void SetDevice(Type type, int device_id = 0);
    static Type GetDeviceType();
    static int GetDeviceId();
    static std::string GetDeviceInfo();
    static size_t GetAvailableMemory();
    static size_t GetTotalMemory();
    static void Synchronize();
};
```

### Layer

Base class for all neural network layers.

```cpp
class CYXWIZ_API Layer {
public:
    virtual ~Layer() = default;

    virtual Tensor Forward(const Tensor& input) = 0;
    virtual Tensor Backward(const Tensor& grad_output) = 0;
    virtual std::vector<Tensor*> Parameters() = 0;
    virtual std::vector<Tensor*> Gradients() = 0;
    virtual std::string Name() const = 0;

    void SetTraining(bool training);
    bool IsTraining() const;
};
```

### Optimizer

Base class for all optimizers.

```cpp
class CYXWIZ_API Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void Step() = 0;
    virtual void ZeroGrad() = 0;
    virtual void SetLearningRate(float lr) = 0;
    virtual float GetLearningRate() const = 0;

    void RegisterParameters(const std::vector<Tensor*>& params,
                           const std::vector<Tensor*>& grads);
};
```

## Available Layers

| Layer | Description | Parameters |
|-------|-------------|------------|
| `Linear` | Fully connected | in_features, out_features |
| `Conv2d` | 2D convolution | in_ch, out_ch, kernel, stride, padding |
| `BatchNorm2d` | Batch normalization | num_features |
| `Dropout` | Dropout regularization | probability |
| `ReLU` | ReLU activation | - |
| `Sigmoid` | Sigmoid activation | - |
| `Tanh` | Tanh activation | - |
| `Softmax` | Softmax activation | dim |

## Available Optimizers

| Optimizer | Description | Key Parameters |
|-----------|-------------|----------------|
| `SGD` | Stochastic gradient descent | lr, momentum, weight_decay |
| `Adam` | Adaptive moments | lr, betas, eps, weight_decay |
| `AdamW` | Adam with decoupled decay | lr, betas, eps, weight_decay |
| `RMSprop` | Root mean square prop | lr, alpha, eps |

## Available Loss Functions

| Loss | Description | Use Case |
|------|-------------|----------|
| `MSELoss` | Mean squared error | Regression |
| `CrossEntropyLoss` | Cross entropy | Classification |
| `BCELoss` | Binary cross entropy | Binary classification |
| `BCEWithLogitsLoss` | BCE with sigmoid | Binary (logits) |
| `L1Loss` | Mean absolute error | Robust regression |

## GPU Acceleration

The backend uses ArrayFire for GPU acceleration:

```cpp
// Check available backends
if (af::isBackendAvailable(AF_BACKEND_CUDA)) {
    cyxwiz::Device::SetDevice(cyxwiz::Device::Type::CUDA);
} else if (af::isBackendAvailable(AF_BACKEND_OPENCL)) {
    cyxwiz::Device::SetDevice(cyxwiz::Device::Type::OpenCL);
} else {
    cyxwiz::Device::SetDevice(cyxwiz::Device::Type::CPU);
}
```

### Backend Priority

1. **CUDA** - NVIDIA GPUs (best performance)
2. **OpenCL** - AMD/Intel GPUs
3. **CPU** - Fallback (always available)

## Memory Management

The backend uses RAII for memory management:

```cpp
// Tensors are automatically deallocated
{
    Tensor x({1.0f, 2.0f}, {2});
    Tensor y = x * 2.0f;  // Auto GPU allocation
}  // Memory freed here

// Force synchronization before measuring
Device::Synchronize();
```

## Thread Safety

The backend is thread-safe for:
- Independent tensor operations
- Model inference (with separate instances)

**Not thread-safe:**
- Model training (use one model per thread)
- Device switching

## Error Handling

```cpp
try {
    Tensor x({1.0f, 2.0f, 3.0f}, {2, 2});  // Shape mismatch
} catch (const cyxwiz::TensorError& e) {
    std::cerr << "Tensor error: " << e.what() << std::endl;
}
```

## Version Information

```cpp
// Get version string
const char* version = cyxwiz::GetVersionString();
// Returns: "0.1.0"

// Version macros
CYXWIZ_VERSION_MAJOR  // 0
CYXWIZ_VERSION_MINOR  // 1
CYXWIZ_VERSION_PATCH  // 0
```

## Building

### CMake Integration

```cmake
find_package(cyxwiz-backend REQUIRED)
target_link_libraries(myapp PRIVATE cyxwiz::backend)
```

### Compile Definitions

| Define | Purpose |
|--------|---------|
| `CYXWIZ_DEBUG` | Enable debug logging |
| `CYXWIZ_ENABLE_LOGGING` | Verbose output |
| `CYXWIZ_HAS_ARRAYFIRE` | ArrayFire available |

## Platform Notes

### Windows

- Library: `cyxwiz-backend.dll`
- Import lib: `cyxwiz-backend.lib`
- Requires: MSVC 2022+

### Linux

- Library: `libcyxwiz-backend.so`
- Requires: GCC 10+

### macOS

- Library: `libcyxwiz-backend.dylib`
- Requires: Clang 12+

---

**Next**: [Tensor Operations](api/tensor.md) | [Neural Network Layers](api/layers.md)
