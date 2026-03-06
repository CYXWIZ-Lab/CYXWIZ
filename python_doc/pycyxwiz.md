# PyCyxWiz Python API Documentation

Version: generated from `cyxwiz-backend/python/bindings.cpp`

PyCyxWiz is the Python binding for the CyxWiz backend (ArrayFire-accelerated math and ML core). It provides:
- Tensors with CPU/GPU acceleration
- Neural network layers, losses, optimizers, and sequential models
- MATLAB-style numerical submodules (linear algebra, signal processing, statistics, time series)
- Audio preprocessing
- Tokenization utilities
- Reinforcement learning helpers
- DuckDB-powered data loading
- Distributed data-parallel training primitives

This document is intended to be a professional, practical reference. It focuses on what you need to use the library effectively in Python.

Related:
- `python_doc/performance_guidance.md`

---

**Contents**
1. Getting Started
2. Core API
3. Neural Network API
4. Functional API
5. Data Loading (DuckDB)
6. MATLAB-Style Submodules
7. Time Series
8. Audio Processing
9. Tokenization
10. Reinforcement Learning
11. Distributed Training
12. Utilities and Conventions

---

## 1. Getting Started

### 1.1 Build and Import
PyCyxWiz is built as a pybind11 module named `pycyxwiz`.

Typical build flags:
- `CYXWIZ_BUILD_PYTHON=ON`
- Make sure Python 3.8+ and pybind11 are available.

After building, the module is placed under:
- `build/<preset>/python/pycyxwiz.*`

Import:
```python
import pycyxwiz as cx
```

### 1.1.1 How the Python Binding Is Exposed
The binding is a compiled native extension built via pybind11 from `cyxwiz-backend/python/bindings.cpp`. The module name is **`pycyxwiz`**. Submodules are:
- `pycyxwiz.linalg`
- `pycyxwiz.signal`
- `pycyxwiz.stats`
- `pycyxwiz.timeseries`
- `pycyxwiz.distributed`

The binding is a thin layer over the C++ API. Prefer:
- `Tensor` for high-performance compute and model/layer workflows
- NumPy arrays for ndarray-compatible `pycyxwiz.linalg` functions
- Python lists only for legacy MATLAB-style compatibility paths

### 1.2 Initialization
Always initialize before use and shut down after you are done:
```python
import pycyxwiz as cx
cx.initialize()
# ... use the library ...
cx.shutdown()
```

### 1.2.1 Error Handling
- Most failures raise `RuntimeError` exceptions in Python.
- Some structs expose `success` or `valid` flags (e.g., `AudioData`, `AudioFeatures`, `WindowResult`, `EnvInfo`). Check those fields if you need to avoid exceptions or if a function returns a result object.

### 1.3 Quick Example
```python
import pycyxwiz as cx
import numpy as np

cx.initialize()

x = np.random.randn(8, 4).astype(np.float32)
t = cx.Tensor.from_numpy(x)

layer = cx.Dense(4, 3)
out = layer.forward(t)

print(out)
print(out.to_numpy().shape)

cx.shutdown()
```

### 1.4 Common Usage Patterns

**Device selection**
```python
import pycyxwiz as cx
cx.initialize()

if cx.cuda_available():
    dev = cx.Device(cx.DeviceType.CUDA, 0)
    dev.set_active()
else:
    dev = cx.Device(cx.DeviceType.CPU, 0)
    dev.set_active()
```

**Tensor round-trip with NumPy**
```python
import numpy as np
import pycyxwiz as cx

cx.initialize()
x = np.arange(12, dtype=np.float32).reshape(3, 4)
t = cx.Tensor.from_numpy(x)
y = t.to_numpy()
```

---

## 2. Core API

### 2.1 Module-Level Functions
- `initialize()`
- `shutdown()`
- `get_version()`

### 2.2 Enums
- `DeviceType`: `CPU`, `CUDA`, `OPENCL`, `METAL`, `VULKAN`
- `DataType`: `Float32`, `Float64`, `Int32`, `Int64`, `UInt8`

### 2.3 Device
```python
info_list = cx.Device.get_available_devices()
current = cx.Device.get_current_device()

device = cx.Device(cx.DeviceType.CUDA, 0)
device.set_active()
```

`DeviceInfo` fields (read-only):
- `type`, `device_id`, `name`, `memory_total`, `memory_available`, `compute_units`
- `supports_fp64`, `supports_fp16`

`Device` methods:
- `get_type()`
- `get_device_id()`
- `get_info()`
- `set_active()`
- `is_active()`
- `get_available_devices()` (static)
- `get_current_device()` (static)

### 2.4 Tensor
`Tensor` is the fundamental multi-dimensional array type.

Constructors:
- `Tensor()`
- `Tensor(shape: List[int], dtype: DataType = Float32)`

Metadata:
- `shape()`
- `num_elements()`
- `num_bytes()`
- `get_data_type()`
- `num_dimensions()`

Arithmetic (element-wise):
- `+`, `-`, `*`, `/`

Device:
- `get_device()`

Factories:
- `Tensor.zeros(shape, dtype=Float32)`
- `Tensor.ones(shape, dtype=Float32)`
- `Tensor.random(shape, dtype=Float32)`

NumPy conversion:
- `Tensor.from_numpy(array)` copies data from NumPy (expects contiguous array)
- `Tensor.to_numpy()` returns a new NumPy array copy

Data type mapping:
- `np.float32` <-> `DataType.Float32`
- `np.float64` <-> `DataType.Float64`
- `np.int32` <-> `DataType.Int32`
- `np.int64` <-> `DataType.Int64`
- `np.uint8` <-> `DataType.UInt8`

Precision and dtype behavior:
- `Tensor` compute dtype is explicit and user-controlled via `DataType`.
- Default tensor dtype is `DataType.Float32`.
- Unsupported NumPy dtypes (for example `float16`, `int16`, `bool`) are not accepted by `Tensor.from_numpy`.
- `Tensor.to_numpy()` preserves the tensor dtype.

Device note:
- `Tensor.to_numpy()` expects CPU-backed data. Keep tensors on CPU before converting.

Example:
```python
import numpy as np
import pycyxwiz as cx

a = np.ones((2, 3), dtype=np.float32)
t = cx.Tensor.from_numpy(a)
print(t.shape())
print(t.to_numpy())
```

**Basic arithmetic**
```python
import pycyxwiz as cx
import numpy as np

cx.initialize()
a = cx.Tensor.from_numpy(np.ones((2, 2), dtype=np.float32))
b = cx.Tensor.from_numpy(np.full((2, 2), 3.0, dtype=np.float32))
c = a + b
d = a * b
```

---

## 3. Neural Network API

### 3.1 Optimizers
Enums:
- `OptimizerType`: `SGD`, `Adam`, `AdamW`, `RMSprop`, `AdaGrad`, `NAdam`, `Adadelta`, `LAMB`
- `WarmupType`: `None_`, `Linear`, `Cosine`

Base class `Optimizer`:
- `step(parameters, gradients)`
- `zero_grad()`
- `set_learning_rate(lr)`
- `get_learning_rate()`

Factory:
- `create_optimizer(type, learning_rate=0.001)`

Concrete optimizers:
- `SGD(learning_rate=0.01, momentum=0.0)`
- `Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)`
- `AdamW(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01)`
- `RMSprop(learning_rate=0.001, alpha=0.99, epsilon=1e-8, momentum=0.0)`
- `AdaGrad(learning_rate=0.01, epsilon=1e-10)`
- `NAdam(learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8)`
- `Adadelta(rho=0.9, epsilon=1e-6)`
- `LAMB(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-6, weight_decay=0.01)`

Learning rate warmup:
- `LRWarmup` (use the factory)
- `create_lr_warmup(optimizer_type, learning_rate=0.001, warmup_steps=1000, warmup_type=WarmupType.Linear)`

`LRWarmup` methods:
- `step(parameters, gradients)`
- `zero_grad()`
- `get_current_lr()`
- `get_warmup_progress()`
- `is_warmup_complete()`

**Optimizer usage with gradients**
```python
import pycyxwiz as cx
import numpy as np

cx.initialize()
layer = cx.Dense(4, 2)
opt = cx.Adam(learning_rate=1e-3)

# Forward
x = cx.Tensor.from_numpy(np.random.randn(8, 4).astype(np.float32))
y = layer.forward(x)

# Dummy gradient coming from loss
grad = cx.Tensor.from_numpy(np.random.randn(8, 2).astype(np.float32))
layer.backward(grad)

# Apply update
params = layer.get_parameters()
grads = layer.get_gradients()
opt.step(params, grads)
```

### 3.2 Layers
Base class `Layer`:
- `forward(input)`
- `backward(grad_output)`
- `get_parameters()`
- `set_parameters(params)`

Core layers:
- `LinearLayer(in_features, out_features, use_bias=True)`
- `Conv2D(in_channels, out_channels, kernel_size, stride=1, padding=0, use_bias=True)`
- `MaxPool2D(pool_size, stride=-1, padding=0)`
- `AvgPool2D(pool_size, stride=-1, padding=0)`
- `GlobalAvgPool2D()`
- `BatchNorm2D(num_features, eps=1e-5, momentum=0.1)`
- `LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)`
- `InstanceNorm2D(num_features, eps=1e-5, affine=False)`
- `GroupNorm(num_groups, num_channels, eps=1e-5, affine=True)`
- `Conv1D(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, use_bias=True)`
- `Embedding(num_embeddings, embedding_dim, padding_idx=-1, max_norm=0.0)`
- `LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False, dropout=0.0)`
- `GRU(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False, dropout=0.0)`
- `MultiHeadAttention(embed_dim, num_heads, dropout=0.0, use_bias=True)`
- `TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, norm_first=False)`
- `TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, norm_first=False)`
- `Flatten()`
- `Dropout(p=0.5)`
- `ConvTranspose2D(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, use_bias=True)`
- `Upsample2D(scale_factor=2, mode=UpsampleMode.Nearest)`
- `PixelShuffle(upscale_factor)`

Aliases:
- `Dense` is an alias for `LinearLayer`
- `BatchNorm` is an alias for `BatchNorm2D`

Parameter conventions:
- `get_parameters()` returns a dict of name -> `Tensor` (e.g., `{'weight': Tensor, 'bias': Tensor}`).
- `set_parameters(params)` accepts a dict with matching keys.
- `get_gradients()` is available on some layers (e.g., `LinearLayer`) and on `Sequential`.

Additional layer utilities:
- `TransformerDecoderLayer.generate_causal_mask(size)`
- `MultiHeadAttention.forward_qkv(query, key, value, attn_mask=None)`
- `TransformerEncoderLayer.forward_with_mask(input, src_mask=None)`
- `TransformerDecoderLayer.forward_with_memory(tgt, memory, tgt_mask=None, memory_mask=None)`

**CNN forward example**
```python
import pycyxwiz as cx
import numpy as np

cx.initialize()
x = cx.Tensor.from_numpy(np.random.randn(4, 3, 32, 32).astype(np.float32))
conv = cx.Conv2D(3, 16, 3, stride=1, padding=1)
pool = cx.MaxPool2D(2, stride=2)
out = pool.forward(conv.forward(x))
```

### 3.3 Losses
Base class `Loss`:
- `forward(predictions, targets)`
- `backward(predictions, targets)`

Losses:
- `MSELoss()`
- `CrossEntropyLoss()`
- `FocalLoss(alpha=0.25, gamma=2.0)`
- `TripletLoss(margin=1.0, distance_type=TripletDistanceType.Euclidean)`
  - `set_negative(negative)`
- `ContrastiveLoss(margin=1.0)`
  - `set_labels(labels)`

**Cross-entropy example**
```python
import pycyxwiz as cx
import numpy as np

cx.initialize()
pred = cx.Tensor.from_numpy(np.random.randn(5, 10).astype(np.float32))  # logits
target = cx.Tensor.from_numpy(np.array([1, 3, 2, 0, 4], dtype=np.int64))
loss = cx.CrossEntropyLoss()
value = loss.forward(pred, target)
```

### 3.4 Activations
Enum:
- `ActivationType`: `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `LeakyReLU`, `ELU`, `GELU`, `Swish`, `SiLU`, `Mish`, `Hardswish`, `SELU`, `PReLU`

Factory:
- `create_activation(type, alpha=0.01)`

Activation classes:
- `ReLU()`
- `LeakyReLU(alpha=0.01)`
- `ELU(alpha=1.0)`
- `GELU()`
- `Swish()`
- `Sigmoid()`
- `Tanh()`
- `Softmax(axis=-1)`
- `Mish()`
- `Hardswish()`
- `SELU()`
- `PReLU(num_parameters=1, init=0.25)`

Alias:
- `SiLU` is an alias for `Swish`

**Activation usage**
```python
import pycyxwiz as cx
import numpy as np

cx.initialize()
x = cx.Tensor.from_numpy(np.random.randn(2, 3).astype(np.float32))
y = cx.ReLU().forward(x)
```

### 3.5 Modules and Sequential
Modules are used by `Sequential` (the high-level model container).

Enum:
- `ModuleType`: `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `Dropout`, `BatchNorm`, `Flatten`, `LeakyReLU`, `ELU`, `GELU`, `Swish`, `Mish`

Base `Module` methods:
- `forward(input)`
- `backward(grad_output)`
- `get_parameters()`
- `set_parameters(params)`
- `get_gradients()`
- `has_parameters()`
- `get_name()`
- `set_training(training)` / `is_training()`
- `freeze()` / `unfreeze()` / `is_trainable()`

Common modules:
- `LinearModule(in_features, out_features, use_bias=True)`
- `ReLUModule()`
- `SigmoidModule()`
- `TanhModule()`
- `SoftmaxModule(dim=-1)`
- `DropoutModule(p=0.5)`
- `FlattenModule(start_dim=1)`
- `LeakyReLUModule(negative_slope=0.01)`
- `ELUModule(alpha=1.0)`
- `GELUModule()`
- `SwishModule()`
- `MishModule()`

Create from enum:
- `create_module(type, params={})`

`Sequential` model:
```python
model = cx.Sequential()
model.add_linear(784, 256)
model.add_relu()
model.add_linear(256, 10)
```

**Training loop sketch**
```python
import pycyxwiz as cx
import numpy as np

cx.initialize()
model = cx.Sequential()
model.add_linear(4, 8)
model.add_relu()
model.add_linear(8, 2)

loss_fn = cx.MSELoss()
opt = cx.SGD(learning_rate=1e-2)

x = cx.Tensor.from_numpy(np.random.randn(16, 4).astype(np.float32))
target = cx.Tensor.from_numpy(np.random.randn(16, 2).astype(np.float32))

pred = model.forward(x)
loss = loss_fn.forward(pred, target)
grad = loss_fn.backward(pred, target)
model.backward(grad)
model.update_parameters(opt)
```

`Sequential` methods:
- `forward(input)` / `backward(grad_output)`
- `get_parameters()` / `set_parameters(params)`
- `get_gradients()`
- `update_parameters(optimizer)`
- `set_training(training)` / `train()` / `eval()`
- `size()` / `__len__()` / `summary()`
- `save(path)` / `load(path)`
- `set_name(name)` / `get_name()`
- `set_description(description)` / `get_description()`
- `freeze_layer(layer_idx)` / `freeze_up_to(layer_idx)` / `freeze_except_last(n)` / `unfreeze_all()`
- `is_layer_trainable(layer_idx)`

---

## 4. Functional API

Convenient functional operations (all operate on `Tensor`):
- `relu(x)`
- `sigmoid(x)`
- `tanh(x)`
- `softmax(x, dim=-1)`
- `gelu(x)`
- `leaky_relu(x, negative_slope=0.01)`
- `elu(x, alpha=1.0)`
- `swish(x)`
- `silu(x)`
- `mish(x)`
- `flatten(x)`
- `dropout(x, p=0.5, training=True)`

Usage pattern:
```python
import pycyxwiz as cx
import numpy as np

cx.initialize()
x = cx.Tensor.from_numpy(np.random.randn(3, 3).astype(np.float32))
y = cx.relu(x)
z = cx.dropout(y, p=0.2, training=True)
```

---

## 5. Data Loading (DuckDB)

The `DataLoader` uses DuckDB internally. If DuckDB is not available, `DataLoader.is_available()` and `duckdb_available()` return false.

Classes:
- `DataLoaderConfig`
- `ColumnInfo`
- `BatchIterator`
- `DataLoader`

`DataLoaderConfig` fields:
- `batch_size` (default 1024)
- `memory_limit_mb` (default 4096)
- `num_threads` (default 4)
- `verbose` (default false)

`DataLoader` methods:
- `load_parquet(path, columns=[])`
- `load_csv(path, columns=[], delimiter=',', has_header=True)`
- `load_json(path, columns=[])`
- `query(sql)`
- `query_columns(sql)`
- `create_batch_iterator(sql, batch_size=0)`
- `get_schema(path)`
- `get_columns(path)`
- `get_row_count(path)`
- `convert_csv_to_parquet(csv_path, parquet_path, compression='snappy')`
- `convert_json_to_parquet(json_path, parquet_path, compression='snappy')`
- `get_config()` / `set_config(config)`

`BatchIterator` supports Python iteration:
```python
loader = cx.DataLoader()
for batch in loader.create_batch_iterator("SELECT * FROM 'big.parquet'", 2048):
    # batch is a Tensor
    pass
```

**SQL usage notes**
- Queries are executed by DuckDB against files directly:
  - `"SELECT * FROM 'data.parquet'"`
  - `"SELECT a, b FROM 'data.csv' WHERE c > 10"`
  - `"SELECT * FROM 'a.parquet' JOIN 'b.parquet' ON a.id = b.id"`
- Use `query_columns(sql)` when you want one tensor per column.

**Load CSV and select columns**
```python
import pycyxwiz as cx

loader = cx.DataLoader()
t = loader.load_csv("data.csv", columns=["col1", "col2"])
print(t.shape())
```

**CSV -> NumPy -> Frobenius norm (`cx.linalg.norm`)**
```python
import pycyxwiz as cx
import pycyxwiz.linalg as la
import numpy as np

loader = cx.DataLoader()
t = loader.load_csv("data.csv", columns=["col1", "col2"])
A = t.to_numpy().astype(np.float64, copy=False)
n = la.norm(A)
print("Frobenius norm:", n)
```

---

## 6. MATLAB-Style Submodules

These submodules are MATLAB-like and useful for quick numerical results.  
`pycyxwiz.linalg` now supports Tensor, ndarray, and list workflows:

- Tensor path: `Tensor` inputs for selected linalg ops (core Tensor-native path)
- ndarray path: direct NumPy input for major linalg calls (no `.tolist()` needed)
- list path: compatibility mode for existing scripts

Key conventions:
- Tensor overloads currently available for: `solve`, `lstsq`, `matmul`, `norm`, `inv`, `transpose`.
- Tensor overloads return `Tensor` for matrix outputs (`norm` returns scalar).
- `pycyxwiz.linalg` supports ndarray inputs for: `diag`, `svd`, `eig`, `qr`, `chol`, `lu`, `det`, `rank`, `trace`, `norm`, `cond`, `inv`, `transpose`, `solve`, `lstsq`, `matmul`.
- List inputs remain supported for backward compatibility.
- `eye`, `zeros`, and `ones` are still list-returning helpers in current bindings.
- Precision for these linalg bindings is `double`-style (`float64` semantics).

Data interop pattern:
```python
import pycyxwiz.linalg as la
import numpy as np

A = np.array([[1, 2], [3, 4]], dtype=np.float64)
B = np.array([[5, 6], [7, 8]], dtype=np.float64)
C = la.matmul(A, B)   # ndarray-compatible path

# Legacy compatibility path (still supported):
C2 = la.matmul(A.tolist(), B.tolist())
```

Tensor-first pattern:
```python
import pycyxwiz as cx
import pycyxwiz.linalg as la
import numpy as np

A = cx.Tensor.from_numpy(np.array([[3.0, 2.0], [1.0, 4.0]], dtype=np.float64))
b = cx.Tensor.from_numpy(np.array([7.0, 9.0], dtype=np.float64))
x = la.solve(A, b)          # returns Tensor
print(type(x), x.shape())
```

### 6.0 Simple MATLAB-Style Calculations

These are common calculations. Use ndarray when possible; list examples are legacy-compatible.

**Create matrices**
```python
import pycyxwiz.linalg as la
import numpy as np

I = la.eye(3)
Z = la.zeros(2, 3)
O = la.ones(2, 2)
D = la.diag(np.array([1.0, 2.0, 3.0], dtype=np.float64))
```

**Matrix multiplication**
```python
import pycyxwiz.linalg as la
import numpy as np

A = np.array([[1, 2], [3, 4]], dtype=np.float64)
B = np.array([[5, 6], [7, 8]], dtype=np.float64)
C = la.matmul(A, B)
```

**Transpose / Inverse / Determinant**
```python
import pycyxwiz.linalg as la
import numpy as np

A = np.array([[2.0, 1.0], [5.0, 3.0]], dtype=np.float64)
At = la.transpose(A)
Ai = la.inv(A)
detA = la.det(A)
```

**Solve linear system**
```python
import pycyxwiz.linalg as la
import numpy as np

A = np.array([[3.0, 2.0], [1.0, 4.0]], dtype=np.float64)
b = np.array([7.0, 9.0], dtype=np.float64)  # 1D or 2D RHS supported
x = la.solve(A, b)
```

**Least squares**
```python
import pycyxwiz.linalg as la
import numpy as np

A = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype=np.float64)
b = np.array([1.0, 2.0, 2.0], dtype=np.float64)
x = la.lstsq(A, b)
```

**Decompositions**
```python
import pycyxwiz.linalg as la
import numpy as np

A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
U, S, Vt = la.svd(A)
Q, R = la.qr(A)
L = la.chol(np.array([[4.0, 2.0], [2.0, 3.0]], dtype=np.float64))
```

**FFT / IFFT**
```python
import pycyxwiz.signal as sp

x = sp.sine(freq=5, fs=100, n=1000)
X = sp.fft(x, sample_rate=100)
x_back = sp.ifft(X["complex"])
```

**Convolution**
```python
import pycyxwiz.signal as sp

x = [1, 2, 3, 4]
h = [1, 0, -1]
y = sp.conv(x, h, mode="same")
```

**Basic statistics**
```python
import pycyxwiz.stats as st

data = [[1, 2], [1, 3], [2, 2], [8, 9], [9, 8]]
km = st.kmeans(data, k=2)
```

### 6.1 `pycyxwiz.linalg`
Matrix creation:
- `eye(n)` or `eye(rows, cols)`
- `zeros(n)` or `zeros(rows, cols)`
- `ones(n)` or `ones(rows, cols)`
- `diag(d)`

Decompositions:
- `svd(A, full_matrices=False)` -> `(U, S, Vt)`
- `eig(A)` -> `(eigenvalues, eigenvectors)`
- `qr(A)` -> `(Q, R)`
- `chol(A)` -> `L`
- `lu(A)` -> `(L, U, P)`

Properties:
- `det(A)`
- `rank(A, tol=1e-10)`
- `trace(A)`
- `norm(A)`
- `cond(A)`

Operations:
- `inv(A)`
- `transpose(A)`
- `solve(A, b)`
- `lstsq(A, b)`
- `matmul(A, B)`

### 6.1.0 Tensor-First Quickstart (New Path)

Use this when your data is already in `cx.Tensor` and you want linalg calls to stay on the Tensor path.

```python
import pycyxwiz as cx
import pycyxwiz.linalg as la
import numpy as np

cx.initialize()

A = cx.Tensor.from_numpy(np.array([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64))
b1 = cx.Tensor.from_numpy(np.array([1.0, 2.0], dtype=np.float64))       # 1D RHS
b2 = cx.Tensor.from_numpy(np.array([[1.0], [2.0]], dtype=np.float64))   # 2D RHS

x1 = la.solve(A, b1)              # Tensor shape: [2]
x2 = la.solve(A, b2)              # Tensor shape: [2, 1]
x_ls = la.lstsq(A, b1)            # Tensor
C = la.matmul(A, A)               # Tensor [2, 2]
Ai = la.inv(A)                    # Tensor [2, 2]
At = la.transpose(A)              # Tensor [2, 2]
n = la.norm(A)                    # Python float

print(x1.shape(), x2.shape(), C.shape(), n)
print(x1.to_numpy())              # Convert only at boundary when needed

cx.shutdown()
```

Practical notes:

- For Tensor overloads, supported ops are: `solve`, `lstsq`, `matmul`, `norm`, `inv`, `transpose`.
- Use `float64` for linalg examples unless you explicitly want `float32`.
- Keep data in Tensor across multiple operations, then call `to_numpy()` only when needed.

CSV -> Tensor -> Frobenius norm:

```python
import pycyxwiz as cx
import pycyxwiz.linalg as la

cx.initialize()
loader = cx.DataLoader()
loader.load_csv("data.csv")
X = loader.query("SELECT col1, col2, col3 FROM data")  # returns Tensor
f_norm = la.norm(X)                                     # Tensor path
print(f_norm)
cx.shutdown()
```

Note: `pycyxwiz.linalg.matmul(A, B)` supports both ndarray and lists. NumPy `@` remains valid when you want pure NumPy execution.

**MATLAB-style example**
```python
import pycyxwiz.linalg as la
import numpy as np

A = np.array([[1, 2], [3, 4]], dtype=np.float64)
B = np.array([[5, 6], [7, 8]], dtype=np.float64)
C = la.matmul(A, B)
print(C)
```

**Use case: compute matrix norm (Frobenius norm)**
```python
import pycyxwiz.linalg as la
import numpy as np

A = np.array([
    [3.0, 4.0],
    [0.0, 12.0],
], dtype=np.float64)

n = la.norm(A)
print(n)  # 13.0
```

**Use case: solve a linear system (Ax = b)**
```python
import pycyxwiz.linalg as la
import numpy as np

A = np.array([[3.0, 2.0], [1.0, 4.0]], dtype=np.float64)
b = np.array([7.0, 9.0], dtype=np.float64)
x = la.solve(A, b)
print(x)
```

**Use case: compute eigenvalues**
```python
import pycyxwiz.linalg as la
import numpy as np

A = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64)
vals, vecs = la.eig(A)
print(vals)
```

### 6.1.1 What Happens Internally for `la.solve(A, b)`

When you call:

```python
x = la.solve(A, b)
```

the execution path depends on input type.

ndarray path (`A`, `b` are NumPy arrays):

1. Python dispatches to the ndarray overload in `bindings.cpp` (`NumpyArrayDouble`).
2. `A` is validated as 2D and converted to `std::vector<std::vector<double>>`.
3. `b` is accepted as 1D or 2D and converted to matrix form (`n x 1` for 1D input).
4. Binding calls `cyxwiz::LinearAlgebra::Solve(A, b)` in C++.
5. C++ checks dimensions (`A` square, rows match `b`).
6. If ArrayFire GPU backend is active, it executes `af::solve(...)`.
7. Otherwise, CPU fallback runs LU decomposition + forward/back substitution.
8. Result returns to binding; if original `b` was 1D, result is flattened to 1D NumPy output.

Tensor path (`A`, `b` are `Tensor`):

1. Python dispatches to the Tensor overload in `bindings.cpp`.
2. Binding calls `cyxwiz::LinearAlgebra::Solve(const Tensor&, const Tensor&)`.
3. C++ checks shapes (`A` must be 2D square, `b` must be 1D or 2D, row count must match).
4. Core executes ArrayFire-first solve (`af::solve`) directly from Tensor input.
5. If ArrayFire path throws, core falls back to CPU solve path.
6. Result is returned as `Tensor` (`1D` output if input `b` was 1D, otherwise `2D`).

### 6.1.2 Copy Semantics (Is the Conversion Copy Necessary?)

Short answer: it depends on path.

ndarray path:

- Current C++ linear algebra API uses `std::vector<std::vector<double>>` signatures.
- ndarray input is copied into that structure before compute.

Tensor path (for `solve`, `lstsq`, `matmul`, `norm`, `inv`, `transpose`):

- No `Tensor -> std::vector<std::vector<double>>` conversion in Python bindings.
- Core Tensor overloads run Tensor-native interfaces directly.

Practical copy behavior:

- If input ndarray is not contiguous or not `float64`, pybind may create a temporary casted copy first.
- Binding then copies ndarray data into `std::vector<std::vector<double>>`.
- ArrayFire path converts host Tensor buffers to/from `af::array` for compute.
- Binding finally copies result into NumPy output array.

So ndarray boundary copies remain overhead, while Tensor path has removed the vector-of-vector conversion layer for key linalg ops.

Remaining optimization target:

- Reduce host/device transfer overhead inside Tensor compute path (persistent device buffers / true device-resident tensor internals).

### 6.2 `pycyxwiz.signal`
FFT and convolution:
- `fft(x, sample_rate=1.0)` -> dict with `magnitude`, `phase`, `frequencies`, `complex`
- `ifft(X)`
- `conv(x, h, mode='same')`
- `conv2(x, h, mode='same')`
- `spectrogram(x, window_size=256, hop_size=128, sample_rate=1.0, window='hann')`

Filters:
- `lowpass(cutoff, fs, order=4)` -> `{'b':..., 'a':...}`
- `highpass(cutoff, fs, order=4)`
- `bandpass(low, high, fs, order=4)`
- `filter(x, b, a)`

Signal generation:
- `sine(freq, fs, n, amp=1.0, phase=0.0)`
- `square(freq, fs, n, amp=1.0)`
- `noise(n, amp=1.0)`

**Use case: filter a signal**
```python
import pycyxwiz.signal as sp

x = sp.sine(freq=10, fs=200, n=2000)
f = sp.lowpass(cutoff=20, fs=200, order=4)
y = sp.filter(x, f["b"], f["a"])
```

**Use case: quick spectrogram**
```python
import pycyxwiz.signal as sp

x = sp.sine(freq=5, fs=100, n=1000)
spec = sp.spectrogram(x, window_size=128, hop_size=64, sample_rate=100)
print(spec["S"][0][:5])
```

### 6.3 `pycyxwiz.stats`
Clustering:
- `kmeans(data, k, max_iter=300, init='kmeans++')`
- `dbscan(data, eps, min_samples=5)`
- `gmm(data, n_components, cov_type='full')`

Dimensionality reduction:
- `pca(data, n_components=2)`
- `tsne(data, n_dims=2, perplexity=30)`

Metrics:
- `silhouette(data, labels)`
- `confusion_matrix(y_true, y_pred)` -> dict with matrix, accuracy, precision, recall, f1
- `roc(y_true, y_scores)` -> dict with fpr, tpr, auc

**Use case: K-means clustering**
```python
import pycyxwiz.stats as st

data = [
    [1.0, 2.0], [1.2, 2.1], [0.9, 1.8],
    [5.0, 5.1], [5.2, 4.9], [4.9, 5.2],
]
res = st.kmeans(data, k=2)
print(res["labels"])
print(res["centroids"])
```

---

## 7. Time Series (`pycyxwiz.timeseries`)

Core functions:
- `acf(data, max_lag=-1)` -> dict
- `pacf(data, max_lag=-1)`
- `decompose(data, period, method='additive')`
- `stationarity(data)` -> dict (ADF, KPSS, suggested differencing)
- `arima(data, horizon, p=-1, d=-1, q=-1)` -> dict
- `diff(data, order=1)`
- `rolling_mean(data, window)`
- `rolling_std(data, window)`

Windowing:
- `WindowConfig` fields: `window_size`, `forecast_horizon`, `stride`, `lag_values`, `rolling_windows`, `add_diff_features`, `normalize`
- `WindowResult` fields: `X`, `y`, `num_windows`, `input_features`, `target_features`, `success`, `error_message`

Windowing helpers:
- `create_windows(data, config)`
- `create_multivariate_windows(data, target_col, config)`
- `add_features(data, lag_values=[], rolling_windows=[], add_diff=False)`
- `chronological_split(num_samples, train_ratio=0.7, val_ratio=0.15)`

**Forecast window example**
```python
import pycyxwiz.timeseries as ts

cfg = ts.WindowConfig()
cfg.window_size = 24
cfg.forecast_horizon = 1
cfg.stride = 1
cfg.add_diff_features = True

series = [float(i) for i in range(100)]
result = ts.create_windows(series, cfg)
print(result.num_windows, result.input_features)
```

---

## 8. Audio Processing

Classes:
- `AudioData` (fields: `samples`, `sample_rate`, `num_samples`, `duration_seconds`, `valid`, `error_message`)
- `SpectrogramConfig` (fields: `n_fft`, `hop_length`, `win_length`, `center`, `window_type`)
- `MelConfig` (adds: `n_mels`, `fmin`, `fmax`)
- `MFCCConfig` (adds: `n_mfcc`, `use_energy`)
- `AudioFeatures` (fields: `data`, `rows`, `cols`, `valid`, `error_message`)

Static utility class `AudioProcessing`:
- `load_audio(filepath, target_sr=16000)`
- `compute_spectrogram(audio, config=SpectrogramConfig())`
- `compute_mel_spectrogram(audio, config=MelConfig())`
- `compute_mfcc(audio, config=MFCCConfig())`
- `add_noise(audio, snr_db=20.0)`
- `time_stretch(audio, rate=1.0)`
- `pitch_shift(audio, semitones=0.0)`
- `resample(audio, target_sr)`
- `normalize(audio)`
- `trim_silence(audio, threshold_db=-40.0)`

**MFCC example**
```python
import pycyxwiz as cx

audio = cx.AudioProcessing.load_audio("sample.wav", target_sr=16000)
cfg = cx.MFCCConfig()
cfg.n_mfcc = 13
mfcc = cx.AudioProcessing.compute_mfcc(audio, cfg)
print(mfcc.rows, mfcc.cols)
```

Usage notes:
- `AudioProcessing.load_audio` returns an `AudioData` object with a `valid` flag.
- `AudioFeatures.valid` indicates whether feature extraction succeeded.

---

## 9. Tokenization

Enum:
- `TokenizerType`: `Whitespace`, `Word`, `Character`

`Vocabulary`:
- `build_from_documents(documents, min_freq=1, max_vocab_size=-1, lowercase=True)`
- `set_vocabulary(words)`
- `add_word(word)`
- `word_to_index(word)`
- `index_to_word(index)`
- `has_word(word)`
- `size()`
- `save_to_file(filepath)`
- `load_from_file(filepath)`
- Properties: `pad_index`, `unk_index`, `bos_index`, `eos_index`

`Tokenizer`:
- `Tokenizer(type=TokenizerType.Word)`
- `encode(text)` / `decode(token_ids)`
- `encode_batch(texts)` / `decode_batch(batch)`
- `pad_batch(batch, max_length=-1)`
- `train(documents, min_freq=1, max_vocab_size=-1)`
- `set_vocabulary(vocab)` / `get_vocabulary()`
- `set_lowercase(value)`
- `set_max_length(value)`
- `set_padding(value)`
- `set_truncation(value)`
- `set_add_bos(value)`
- `set_add_eos(value)`
- Property: `vocab_size`

**Tokenizer example**
```python
import pycyxwiz as cx

tok = cx.Tokenizer(cx.TokenizerType.Word)
tok.train(["Hello world", "Hello cyxwiz"])
ids = tok.encode("Hello world")
text = tok.decode(ids)
```

Usage notes:
- `Tokenizer.train` builds the vocabulary internally; you can also build a `Vocabulary` directly and pass it via `set_vocabulary`.
- Use `pad_batch` to normalize sequence lengths before model input.

---

## 10. Reinforcement Learning

Data structures:
- `RLTransition` (fields: `state`, `action`, `reward`, `next_state`, `done`)
- `RLBatch` (fields: `states`, `actions`, `rewards`, `next_states`, `dones`, `size`)
- `StepResult` (fields: `observation`, `reward`, `done`, `truncated`, `info`)
- `EnvInfo` (fields: `name`, `observation_dim`, `action_dim`, `discrete_actions`, `num_actions`, `action_low`, `action_high`, `valid`, `error_message`)

Replay buffer:
- `ReplayBuffer(capacity=100000, seed=42)`
- `push(transition)` or `push(state, action, reward, next_state, done)`
- `sample(batch_size)` -> `RLBatch`
- `size()`, `capacity()`, `can_sample(batch_size)`, `clear()`

Epsilon schedule:
- `EpsilonSchedule(start=1.0, end=0.01, decay_steps=10000)`
- `step()` / `reset()`
- Properties: `epsilon`, `current_step`

**Replay buffer usage**
```python
import pycyxwiz as cx

rb = cx.ReplayBuffer(capacity=10000)
rb.push([0.0, 1.0], [1.0], 0.5, [0.1, 1.1], False)
if rb.can_sample(32):
    batch = rb.sample(32)
```

RL dashboard bridge:
- `rl_set_metric_callback(callback)`
- `rl_update_metric(name, value)`
- `rl_should_stop()` / `rl_set_stop(value)`
- `rl_is_paused()` / `rl_set_paused(value)`

Usage notes:
- `rl_set_metric_callback` allows Python RL loops to stream metrics into the engine UI.
- `rl_should_stop()` and `rl_is_paused()` can be polled inside training loops.

### 10.1 Callback Call Flow (`rl_set_metric_callback` / `rl_update_metric`)

Current Python callback surface in `pycyxwiz` is the RL metric callback bridge.

Call flow:

1. `rl_set_metric_callback(callback)` stores a process-global callback.
2. `rl_update_metric(name, value)` checks whether a callback is registered.
3. If present, `rl_update_metric` releases the Python GIL before entering the callback bridge.
4. The bridge reacquires the GIL and calls `callback(name, value)`.
5. `rl_set_metric_callback(None)` clears the callback.

Important behavior:

- Callback registration is global (latest registration replaces prior one).
- If no callback is set, `rl_update_metric` is a no-op.
- Callback exceptions propagate back to the `rl_update_metric` caller.
- Pause/stop flags are independent atomics: use `rl_should_stop()` and `rl_is_paused()` in your loop.

Recommended usage pattern:

```python
import time
import pycyxwiz as cx

def on_metric(name: str, value: float):
    # Keep callback lightweight (UI/logging/aggregation only)
    print(f"{name}: {value:.4f}")

cx.rl_set_metric_callback(on_metric)

for step in range(100000):
    if cx.rl_should_stop():
        break

    while cx.rl_is_paused():
        time.sleep(0.05)

    # ... training step ...
    reward = 1.23
    cx.rl_update_metric("episode_reward", reward)

cx.rl_set_metric_callback(None)
```

Best practices:

- Keep callback work small; do heavy compute outside the callback.
- Use stable metric names (`episode_reward`, `loss`, `epsilon`, etc.).
- Clear callback when done to avoid stale global handler state.

---

## 11. Distributed Training (`pycyxwiz.distributed`)

Enums:
- `ReduceOp`: `SUM`, `PRODUCT`, `MIN`, `MAX`, `AVERAGE`
- `BackendType`: `CPU`, `NCCL`

Config objects:
- `DistributedConfig` (fields: `backend`, `rank`, `world_size`, `local_rank`, `master_addr`, `master_port`, `timeout_ms`)
  - `from_environment()`
  - `is_valid()`
- `DDPConfig` (fields: `broadcast_parameters`, `bucket_size_mb`, `find_unused_parameters`)
- `DistributedTrainingConfig` (fields: `epochs`, `batch_size`, `shuffle`, `seed`, `save_on_master_only`, `checkpoint_every_n_epochs`, `checkpoint_dir`, `verbose`, `log_every_n_batches`, `validation_split`)

Global distributed functions:
- `init(config=DistributedConfig.from_environment())`
- `finalize()`
- `get_rank()`
- `get_world_size()`
- `get_local_rank()`
- `is_distributed()`
- `is_master()`
- `get_default_process_group()`

Distributed primitives:
- `ProcessGroup` (base class) with `barrier()` and rank/world info
- `DistributedDataParallel(model, config=DDPConfig())`
- `DistributedSampler(dataset_size, shuffle=True, seed=0, drop_last=False)`
- `DistributedBatchIterator(sampler, batch_size)`

Trainer:
- `DistributedTrainer(model, loss, optimizer, process_group=None)`
  - `fit(X_train, y_train, config)`
  - `fit(X_train, y_train, X_val, y_val, config)`
  - `evaluate(X_test, y_test)` -> `(loss, accuracy)`
  - `save_checkpoint(path)` / `load_checkpoint(path)`
- `get_model()`

**Distributed init example**
```python
import pycyxwiz.distributed as dist

cfg = dist.DistributedConfig.from_environment()
if cfg.is_valid():
    dist.init(cfg)
```

Environment variables commonly used:
- `RANK`, `WORLD_SIZE`, `LOCAL_RANK`
- `MASTER_ADDR`, `MASTER_PORT`

---

## 12. Utilities and Conventions

### 12.1 Device Availability Checks
- `cuda_available()`
- `opencl_available()`
- `metal_available()`

### 12.2 Device Helpers
- `get_device(type, device_id=0)`
- `set_device(device)`
- `get_available_devices()`

### 12.3 DuckDB Availability
- `duckdb_available()`

### 12.4 Notes on Matrix Multiplication
- For `Tensor` workflows, keep computation in `Tensor` space.
- For NumPy arrays, `pycyxwiz.linalg.matmul(A, B)` now accepts ndarray directly.
- For pure Python lists, `pycyxwiz.linalg.matmul(A, B)` remains supported.

---

## Appendix: Example Snippets

### A. Linear Algebra
```python
import pycyxwiz.linalg as la
import numpy as np
A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
print(la.matmul(A, A))
```

### B. Signal Processing
```python
import pycyxwiz.signal as sp
x = sp.sine(freq=5, fs=100, n=1000)
fft = sp.fft(x, sample_rate=100)
print(fft['frequencies'][:5])
```

### C. Time Series Windowing
```python
import pycyxwiz.timeseries as ts
cfg = ts.WindowConfig()
cfg.window_size = 12
cfg.forecast_horizon = 1
result = ts.create_windows([1,2,3,4,5,6,7,8,9,10,11,12,13], cfg)
print(result.num_windows)
```

### D. DataLoader
```python
import pycyxwiz as cx
loader = cx.DataLoader()
print(loader.get_columns('data.csv'))
```

---

End of document.
