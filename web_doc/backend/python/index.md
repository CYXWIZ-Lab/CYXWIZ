# pycyxwiz Python API

`pycyxwiz` is the Python binding for the CyxWiz backend, providing GPU-accelerated tensor operations, neural network layers, and analysis tools.

## Installation

The module is automatically available when running scripts in the CyxWiz Engine. For standalone use:

```bash
pip install pycyxwiz
```

## Quick Start

```python
import pycyxwiz as cx

# Initialize backend
cx.initialize()

# Check GPU availability
print("CUDA available:", cx.cuda_available())
print("Version:", cx.get_version())

# Create tensors
t1 = cx.Tensor.random([3, 3])
t2 = cx.Tensor.ones([3, 3])
result = t1 + t2

print("Result shape:", result.shape())
```

## Module Structure

```
pycyxwiz
├── Core
│   ├── initialize() / shutdown()
│   ├── Tensor
│   ├── Device / DeviceInfo
│   └── DataType / DeviceType (enums)
│
├── Neural Network
│   ├── Layers: Dense, Conv2D, MaxPool2D, BatchNorm2D, Flatten, Dropout
│   ├── Activations: ReLU, Sigmoid, Tanh, GELU, Softmax, etc.
│   ├── Loss: MSELoss, CrossEntropyLoss
│   └── Optimizers: SGD, Adam, AdamW
│
├── Submodules
│   ├── linalg - Linear algebra (SVD, QR, LU, etc.)
│   ├── signal - Signal processing (FFT, filters)
│   ├── stats - Statistics and clustering
│   └── timeseries - Time series analysis
│
└── Utilities
    ├── cuda_available() / opencl_available() / metal_available()
    ├── get_available_devices()
    └── set_device() / get_device()
```

## Documentation

| Section | Description |
|---------|-------------|
| [Device Management](device.md) | GPU/CPU selection and info |
| [Linear Algebra](linalg.md) | Matrix operations (MATLAB-style) |
| [Signal Processing](signal.md) | FFT, filters, spectrograms |
| [Statistics](stats.md) | Clustering, PCA, metrics |
| [Time Series](timeseries.md) | ACF, ARIMA, decomposition |
| [Activations](activations.md) | Activation functions |

## Core API

### Initialization

```python
cx.initialize()     # Initialize backend (auto-called on first use)
cx.shutdown()       # Cleanup resources
cx.get_version()    # Returns version string, e.g., '0.1.0'
```

### Tensor Class

```python
# Creation
t = cx.Tensor()                           # Empty tensor
t = cx.Tensor([3, 4], cx.DataType.Float32)  # With shape and dtype
t = cx.Tensor.zeros([3, 4])               # Zeros
t = cx.Tensor.ones([3, 4])                # Ones
t = cx.Tensor.random([3, 4])              # Random [0, 1)

# NumPy conversion
import numpy as np
arr = np.random.randn(3, 4).astype(np.float32)
t = cx.Tensor.from_numpy(arr)
arr_back = t.to_numpy()

# Properties
t.shape()           # [3, 4]
t.num_elements()    # 12
t.num_bytes()       # 48 (12 * 4 bytes)
t.get_data_type()   # DataType.Float32
t.num_dimensions()  # 2

# Arithmetic
c = a + b           # Element-wise addition
c = a - b           # Element-wise subtraction
c = a * b           # Element-wise multiplication
c = a / b           # Element-wise division
```

### Data Types

```python
cx.DataType.Float32   # 32-bit float (default)
cx.DataType.Float64   # 64-bit float
cx.DataType.Int32     # 32-bit integer
cx.DataType.Int64     # 64-bit integer
cx.DataType.UInt8     # 8-bit unsigned integer
```

### Device Types

```python
cx.DeviceType.CPU     # CPU backend
cx.DeviceType.CUDA    # NVIDIA CUDA
cx.DeviceType.OPENCL  # OpenCL
cx.DeviceType.METAL   # Apple Metal
cx.DeviceType.VULKAN  # Vulkan (experimental)
```

## Neural Network Layers

### Dense (Linear)

```python
layer = cx.Dense(in_features=784, out_features=128, use_bias=True)

# Forward pass
output = layer.forward(input_tensor)  # [batch, 784] → [batch, 128]

# Get/set parameters
params = layer.get_parameters()  # {'weight': Tensor, 'bias': Tensor}
layer.set_parameters(params)

# Properties
layer.in_features   # 784
layer.out_features  # 128
layer.has_bias      # True
```

### Conv2D

```python
layer = cx.Conv2D(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    use_bias=True
)

# Input: [batch, channels, height, width]
output = layer.forward(input_tensor)  # [N, 3, H, W] → [N, 64, H, W]
```

### Pooling Layers

```python
# Max Pooling
pool = cx.MaxPool2D(pool_size=2, stride=2, padding=0)
output = pool.forward(input)

# Average Pooling
pool = cx.AvgPool2D(pool_size=2, stride=2, padding=0)
output = pool.forward(input)

# Global Average Pooling
pool = cx.GlobalAvgPool2D()
output = pool.forward(input)  # [N, C, H, W] → [N, C, 1, 1]
```

### Normalization

```python
# Batch Normalization
bn = cx.BatchNorm2D(num_features=64, eps=1e-5, momentum=0.1)
output = bn.forward(input)  # Normalizes over batch
```

### Utility Layers

```python
# Flatten
flatten = cx.Flatten()
output = flatten.forward(input)  # [N, C, H, W] → [N, C*H*W]

# Dropout
dropout = cx.Dropout(p=0.5)
output = dropout.forward(input)  # Randomly zeros elements during training
```

## Activation Functions

### Class-Based

```python
relu = cx.ReLU()
sigmoid = cx.Sigmoid()
tanh = cx.Tanh()
gelu = cx.GELU()
softmax = cx.Softmax(dim=-1)
leaky_relu = cx.LeakyReLU(negative_slope=0.01)
elu = cx.ELU(alpha=1.0)
swish = cx.Swish()
mish = cx.Mish()

output = relu.forward(input)
```

### Functional API

```python
output = cx.relu(input)
output = cx.sigmoid(input)
output = cx.tanh(input)
output = cx.softmax(input, dim=-1)
output = cx.gelu(input)
output = cx.leaky_relu(input, negative_slope=0.01)
output = cx.elu(input, alpha=1.0)
output = cx.swish(input)
output = cx.silu(input)  # Alias for swish
output = cx.mish(input)
output = cx.flatten(input)
output = cx.dropout(input, p=0.5, training=True)
```

## Loss Functions

```python
# Mean Squared Error
mse = cx.MSELoss()
loss = mse.forward(predictions, targets)
grad = mse.backward(predictions, targets)

# Cross Entropy (with softmax)
ce = cx.CrossEntropyLoss()
loss = ce.forward(logits, targets)  # logits are pre-softmax
grad = ce.backward(logits, targets)
```

## Optimizers

```python
# SGD with momentum
optimizer = cx.SGD(learning_rate=0.01, momentum=0.9)

# Adam
optimizer = cx.Adam(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8
)

# AdamW (with weight decay)
optimizer = cx.AdamW(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=0.01
)

# Usage
optimizer.step(parameters, gradients)
optimizer.zero_grad()
optimizer.set_learning_rate(0.0001)
lr = optimizer.get_learning_rate()
```

## Submodules

### Linear Algebra (`cx.linalg`)

```python
# Matrix operations
A_inv = cx.linalg.inv(A)
U, S, Vt = cx.linalg.svd(A)
Q, R = cx.linalg.qr(A)
x = cx.linalg.solve(A, b)
```

See [Linear Algebra Reference](linalg.md) for complete API.

### Signal Processing (`cx.signal`)

```python
# FFT
result = cx.signal.fft(x, sample_rate=1.0)

# Filters
b, a = cx.signal.lowpass(cutoff=10, fs=100)
filtered = cx.signal.filter(x, b, a)
```

See [Signal Processing Reference](signal.md) for complete API.

### Statistics (`cx.stats`)

```python
# Clustering
result = cx.stats.kmeans(data, k=3)
result = cx.stats.dbscan(data, eps=0.5)

# Dimensionality reduction
result = cx.stats.pca(data, n_components=2)
```

See [Statistics Reference](stats.md) for complete API.

### Time Series (`cx.timeseries`)

```python
# Analysis
result = cx.timeseries.acf(data, max_lag=20)
result = cx.timeseries.decompose(data, period=12)

# Forecasting
forecast = cx.timeseries.arima(data, horizon=10)
```

See [Time Series Reference](timeseries.md) for complete API.

## Utility Functions

```python
# Device availability
cx.cuda_available()     # True if CUDA devices available
cx.opencl_available()   # True if OpenCL devices available
cx.metal_available()    # True if Metal available (macOS)

# Device management
devices = cx.get_available_devices()  # List of DeviceInfo
cx.set_device(device)                 # Set active device
device = cx.get_device(cx.DeviceType.CUDA, 0)  # Get specific device
```

## Example: Training Loop

```python
import pycyxwiz as cx
import numpy as np

# Data
X_train = np.random.randn(1000, 784).astype(np.float32)
y_train = np.random.randint(0, 10, 1000)

# Model
layer1 = cx.Dense(784, 128)
relu = cx.ReLU()
layer2 = cx.Dense(128, 10)

# Training setup
loss_fn = cx.CrossEntropyLoss()
optimizer = cx.Adam(learning_rate=0.001)

# Training loop
for epoch in range(10):
    # Forward
    X = cx.Tensor.from_numpy(X_train)
    h = relu.forward(layer1.forward(X))
    logits = layer2.forward(h)

    # Loss
    y = cx.Tensor.from_numpy(y_train.astype(np.int64))
    loss = loss_fn.forward(logits, y)
    print(f"Epoch {epoch}: Loss = {loss:.4f}")

    # Backward
    grad = loss_fn.backward(logits, y)
    # ... backprop through layers ...

    # Update
    params = list(layer1.get_parameters().values()) + \
             list(layer2.get_parameters().values())
    grads = list(layer1.get_gradients().values()) + \
            list(layer2.get_gradients().values())
    optimizer.step(params, grads)
```

---

**Next**: [Device Management](device.md)
