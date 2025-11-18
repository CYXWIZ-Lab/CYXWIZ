# Phase 6: pycyxwiz Python Bindings

**Status:** 🚧 In Progress
**Start Date:** 2025-11-17
**Goal:** Expose ArrayFire backend to Python scripts for 10-100x performance boost

---

## Overview

Currently, Python scripts in CyxWiz use NumPy/Pandas (CPU-only) for array operations, while the C++ backend has ArrayFire (GPU-accelerated). This creates:
- ❌ Performance bottleneck (scripts are slow)
- ❌ Data conversion overhead (NumPy ↔ Backend)
- ❌ Inconsistent APIs (learn two systems)

**Solution:** Build `pycyxwiz` Python module that exposes the C++ backend directly to Python!

---

## Architecture

```
┌─────────────────────────────────────┐
│   Python Script (.cyx)              │
│                                     │
│   import pycyxwiz as cx             │
│   tensor = cx.Tensor([1,2,3])      │
│   result = cx.matmul(tensor, ...)  │
└──────────────┬──────────────────────┘
               │ Python Bindings (pybind11)
               ↓
┌─────────────────────────────────────┐
│   CyxWiz Backend (C++)              │
│                                     │
│   cyxwiz::Tensor                    │
│   ArrayFire arrays                  │
│   GPU/CPU acceleration              │
└─────────────────────────────────────┘
```

**Key Insight:** Python scripts will use the SAME backend as C++ code, with zero-copy data sharing!

---

## Tasks Breakdown

### Task 1: Setup Python Bindings Infrastructure ⚙️

**Goal:** Configure build system for Python module

**Subtasks:**
1. ✅ Check pybind11 availability (already in vcpkg.json)
2. Add pybind11 target to CMakeLists.txt
3. Create `cyxwiz-backend/python/` directory
4. Create skeleton bindings file
5. Configure module build settings
6. Test basic import

**Files:**
- `cyxwiz-backend/CMakeLists.txt` (modify)
- `cyxwiz-backend/python/tensor_bindings.cpp` (create)

**Estimated Time:** 1-2 hours

---

### Task 2: Expose Tensor Class 🔢

**Goal:** Make cyxwiz::Tensor available in Python

**Subtasks:**
1. Bind Tensor constructor
2. Bind shape() method
3. Bind device info methods
4. Bind element access (optional)
5. Add __repr__ for printing
6. Add __str__ for display

**Python API:**
```python
import pycyxwiz as cx

# Create tensors
t1 = cx.Tensor([1, 2, 3, 4])
t2 = cx.Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # shape, data

# Inspect
print(t1)  # <Tensor shape=[4] device=CUDA>
print(t1.shape())  # [4]
print(t1.device())  # "CUDA" or "CPU"
```

**Files:**
- `cyxwiz-backend/python/tensor_bindings.cpp`

**Estimated Time:** 2-3 hours

---

### Task 3: Expose Math Operations ➕➖✖️➗

**Goal:** Basic arithmetic and math functions

**Subtasks:**
1. Bind operators (+, -, *, /)
2. Bind comparison operators (==, !=, <, >)
3. Bind matmul (matrix multiplication)
4. Bind trigonometric functions (sin, cos, tan)
5. Bind statistical functions (sum, mean, std)
6. Bind reduction operations (min, max)

**Python API:**
```python
import pycyxwiz as cx

a = cx.Tensor([1, 2, 3])
b = cx.Tensor([4, 5, 6])

# Arithmetic
c = a + b      # Element-wise add
d = a * 2      # Scalar multiply
e = cx.matmul(a, b.T)  # Matrix multiply

# Math functions
f = cx.sin(a)  # Trigonometric
g = cx.mean(a) # Statistics
```

**Files:**
- `cyxwiz-backend/python/ops_bindings.cpp`

**Estimated Time:** 3-4 hours

---

### Task 4: Expose Neural Network Layers 🧠

**Goal:** Make Layer classes available in Python

**Subtasks:**
1. Bind base Layer class
2. Bind DenseLayer (fully connected)
3. Bind Activation layers (ReLU, Sigmoid, Softmax)
4. Bind Conv2D layer
5. Bind Pooling layers
6. Bind Dropout layer

**Python API:**
```python
import pycyxwiz as cx

# Build model
layers = [
    cx.layers.Dense(784, 128, activation='relu'),
    cx.layers.Dropout(0.2),
    cx.layers.Dense(128, 10, activation='softmax')
]

# Each layer is callable
x = cx.Tensor([batch_size, 784])
h = layers[0](x)  # Forward pass
```

**Files:**
- `cyxwiz-backend/python/layer_bindings.cpp`

**Estimated Time:** 4-5 hours

---

### Task 5: Expose Optimizers 🎯

**Goal:** Training algorithms available in Python

**Subtasks:**
1. Bind base Optimizer class
2. Bind SGD optimizer
3. Bind Adam optimizer
4. Bind AdamW optimizer
5. Bind RMSprop optimizer
6. Bind learning rate schedulers

**Python API:**
```python
import pycyxwiz as cx

# Create optimizer
optimizer = cx.optimizers.Adam(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999
)

# Use with model
model = cx.Model()
model.compile(optimizer=optimizer, loss='cross_entropy')
```

**Files:**
- `cyxwiz-backend/python/optimizer_bindings.cpp`

**Estimated Time:** 3-4 hours

---

### Task 6: Expose Model Class 🏗️

**Goal:** High-level training interface

**Subtasks:**
1. Bind Model constructor
2. Bind add() for adding layers
3. Bind compile() for configuration
4. Bind fit() for training
5. Bind predict() for inference
6. Bind save/load for persistence

**Python API:**
```python
import pycyxwiz as cx

# Build model
model = cx.Model()
model.add(cx.layers.Dense(784, 128, activation='relu'))
model.add(cx.layers.Dense(128, 10, activation='softmax'))

# Compile
model.compile(
    optimizer=cx.optimizers.Adam(lr=0.001),
    loss='cross_entropy',
    metrics=['accuracy']
)

# Train (GPU-accelerated automatically!)
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# Predict
predictions = model.predict(X_test)
```

**Files:**
- `cyxwiz-backend/python/model_bindings.cpp`

**Estimated Time:** 4-5 hours

---

### Task 7: NumPy Interoperability 🔄

**Goal:** Convert between NumPy and CyxWiz tensors

**Subtasks:**
1. Implement from_numpy() converter
2. Implement to_numpy() converter
3. Handle different data types
4. Handle different shapes
5. Optimize for zero-copy when possible

**Python API:**
```python
import pycyxwiz as cx
import numpy as np

# NumPy → CyxWiz
np_array = np.array([1, 2, 3, 4])
cx_tensor = cx.from_numpy(np_array)

# CyxWiz → NumPy
back_to_numpy = cx_tensor.to_numpy()

# Hybrid workflow
import pandas as pd

# Use pandas for I/O
df = pd.read_csv("data.csv")

# Convert to CyxWiz for computation
X = cx.from_numpy(df[['feature1', 'feature2']].values)
y = cx.from_numpy(df['label'].values)

# Train on GPU
model.fit(X, y)  # Fast!
```

**Files:**
- `cyxwiz-backend/python/numpy_interop.cpp`

**Estimated Time:** 3-4 hours

---

### Task 8: Device Management 🖥️

**Goal:** Control GPU/CPU execution

**Subtasks:**
1. Bind Device class
2. Bind device info functions
3. Bind device selection
4. Bind memory management
5. Add context managers for device scope

**Python API:**
```python
import pycyxwiz as cx

# Query devices
devices = cx.get_devices()
for dev in devices:
    print(f"{dev.name}: {dev.memory_gb}GB")

# Select device
cx.set_device('cuda:0')  # Use GPU 0
cx.set_device('cpu')     # Use CPU

# Device context
with cx.device('cuda:1'):
    # This code runs on GPU 1
    tensor = cx.Tensor([1, 2, 3])
```

**Files:**
- `cyxwiz-backend/python/device_bindings.cpp`

**Estimated Time:** 2-3 hours

---

### Task 9: Documentation & Examples 📚

**Goal:** Document the Python API

**Subtasks:**
1. Write API reference docs
2. Create tutorial notebooks
3. Update script templates to use pycyxwiz
4. Create example scripts
5. Add docstrings to all bindings
6. Create migration guide (NumPy → pycyxwiz)

**Deliverables:**
- `developer_docs/phase6/PYCYXWIZ_API_REFERENCE.md`
- `developer_docs/phase6/PYCYXWIZ_TUTORIAL.md`
- `developer_docs/phase6/NUMPY_MIGRATION_GUIDE.md`
- Updated templates in `scripts/templates/`
- Example scripts in `examples/pycyxwiz/`

**Estimated Time:** 4-5 hours

---

### Task 10: Testing & Benchmarks 🧪

**Goal:** Verify correctness and measure performance

**Subtasks:**
1. Create unit tests for each binding
2. Test NumPy compatibility
3. Benchmark vs NumPy operations
4. Test GPU vs CPU performance
5. Memory leak testing
6. Create performance comparison report

**Test Scripts:**
```python
# tests/python/test_tensor.py
import pycyxwiz as cx
import numpy as np

def test_tensor_creation():
    t = cx.Tensor([1, 2, 3])
    assert t.shape() == [3]

def test_numpy_conversion():
    np_arr = np.array([1, 2, 3])
    cx_tensor = cx.from_numpy(np_arr)
    back = cx_tensor.to_numpy()
    assert np.array_equal(np_arr, back)
```

**Benchmarks:**
```python
# benchmarks/matmul_benchmark.py
import pycyxwiz as cx
import numpy as np
import time

# NumPy (CPU)
start = time.time()
result_np = np.matmul(A_np, B_np)
numpy_time = time.time() - start

# pycyxwiz (GPU)
start = time.time()
result_cx = cx.matmul(A_cx, B_cx)
cx_time = time.time() - start

print(f"NumPy: {numpy_time:.3f}s")
print(f"pycyxwiz: {cx_time:.3f}s")
print(f"Speedup: {numpy_time/cx_time:.1f}x")
```

**Files:**
- `tests/python/test_*.py`
- `benchmarks/*.py`
- `developer_docs/phase6/PERFORMANCE_REPORT.md`

**Estimated Time:** 5-6 hours

---

## Total Timeline

| Task | Estimated Time | Priority |
|------|----------------|----------|
| 1. Setup Infrastructure | 1-2 hours | Critical |
| 2. Expose Tensor | 2-3 hours | Critical |
| 3. Expose Math Ops | 3-4 hours | High |
| 4. Expose Layers | 4-5 hours | High |
| 5. Expose Optimizers | 3-4 hours | Medium |
| 6. Expose Model | 4-5 hours | High |
| 7. NumPy Interop | 3-4 hours | Critical |
| 8. Device Management | 2-3 hours | Medium |
| 9. Documentation | 4-5 hours | High |
| 10. Testing | 5-6 hours | Critical |

**Total:** 32-41 hours (~1 week of focused work)

---

## Success Criteria

### ✅ Functionality
- [ ] Python can import pycyxwiz module
- [ ] Tensors can be created from Python lists
- [ ] Basic math operations work (add, multiply, matmul)
- [ ] Neural network layers are functional
- [ ] Model training runs on GPU
- [ ] NumPy conversion is bidirectional
- [ ] Device selection works (CPU/GPU)

### ✅ Performance
- [ ] Matrix multiplication is 10-25x faster than NumPy
- [ ] Neural network training is 20-50x faster
- [ ] Memory usage is comparable to C++ backend
- [ ] No significant overhead from Python bindings

### ✅ Usability
- [ ] API is intuitive for Python developers
- [ ] Error messages are clear
- [ ] Documentation is comprehensive
- [ ] Examples cover common use cases

### ✅ Integration
- [ ] Templates use pycyxwiz instead of NumPy
- [ ] Sandbox allows pycyxwiz import
- [ ] Works with both sandbox ON and OFF
- [ ] Compatible with startup scripts

---

## Dependencies

### Required:
- ✅ pybind11 (already in vcpkg.json)
- ✅ ArrayFire backend (already implemented)
- ✅ cyxwiz::Tensor class (already implemented)
- ✅ Neural network layers (already implemented)

### Check First:
- [ ] Current Tensor API in C++
- [ ] Current Layer API in C++
- [ ] Current Model API in C++
- [ ] Build system structure

---

## Risks & Mitigation

### Risk 1: Memory Management
**Issue:** Python and C++ have different memory models
**Mitigation:** Use pybind11's smart pointer support, careful RAII

### Risk 2: GIL (Global Interpreter Lock)
**Issue:** Python GIL may limit parallelism
**Mitigation:** Release GIL during long operations (pybind11 supports this)

### Risk 3: ArrayFire Initialization
**Issue:** ArrayFire requires initialization, Python might call multiple times
**Mitigation:** Singleton pattern, check if already initialized

### Risk 4: Debugging Difficulty
**Issue:** Harder to debug Python ↔ C++ boundary
**Mitigation:** Extensive logging, good error messages, unit tests

---

## Phase 6 Folder Structure

```
cyxwiz-backend/
├── python/
│   ├── CMakeLists.txt          # pybind11 module build
│   ├── pycyxwiz.cpp            # Main module definition
│   ├── tensor_bindings.cpp     # Tensor class bindings
│   ├── ops_bindings.cpp        # Math operations
│   ├── layer_bindings.cpp      # Neural network layers
│   ├── optimizer_bindings.cpp  # Optimizers
│   ├── model_bindings.cpp      # Model class
│   ├── device_bindings.cpp     # Device management
│   └── numpy_interop.cpp       # NumPy conversion

developer_docs/phase6/
├── PHASE6_PLAN.md              # This file
├── PYCYXWIZ_API_REFERENCE.md   # API documentation
├── PYCYXWIZ_TUTORIAL.md        # Tutorial
├── NUMPY_MIGRATION_GUIDE.md    # Migration guide
└── PERFORMANCE_REPORT.md       # Benchmarks

examples/pycyxwiz/
├── 01_basic_tensors.cyx        # Basic usage
├── 02_math_operations.cyx      # Math ops
├── 03_neural_network.cyx       # Build NN
├── 04_training.cyx             # Train model
└── 05_numpy_interop.cyx        # NumPy integration

tests/python/
├── test_tensor.py
├── test_ops.py
├── test_layers.py
├── test_model.py
└── test_numpy_interop.py

benchmarks/
├── matmul_benchmark.py
├── neural_network_benchmark.py
└── memory_benchmark.py
```

---

## Implementation Order

### Week 1: Core Infrastructure
1. **Day 1-2:** Task 1 (Setup) + Task 2 (Tensor)
2. **Day 3:** Task 3 (Math Ops)
3. **Day 4-5:** Task 7 (NumPy Interop) + Basic Testing

**Milestone:** Can create tensors, do math, convert to/from NumPy

---

### Week 2: Neural Networks
4. **Day 6-7:** Task 4 (Layers)
5. **Day 8-9:** Task 5 (Optimizers) + Task 6 (Model)
6. **Day 10:** Task 8 (Device Management)

**Milestone:** Can build and train neural networks on GPU

---

### Week 3: Polish & Documentation
7. **Day 11-12:** Task 9 (Documentation & Examples)
8. **Day 13-14:** Task 10 (Testing & Benchmarks)
9. **Day 15:** Final testing, bug fixes, release

**Milestone:** Production-ready pycyxwiz module

---

## Expected Performance Gains

### NumPy (CPU-only) vs pycyxwiz (GPU)

| Operation | Size | NumPy (ms) | pycyxwiz (ms) | Speedup |
|-----------|------|------------|---------------|---------|
| Matrix Multiply | 1000x1000 | 50 | 2 | **25x** |
| Element-wise ops | 1M elements | 10 | 0.5 | **20x** |
| Neural network forward | 784→128→10 | 5 | 0.3 | **16x** |
| Neural network training | 1 epoch MNIST | 1000 | 50 | **20x** |

**Expected Overall:** 10-100x speedup depending on operation

---

## Post-Phase 6 Benefits

### For Users:
✅ Scripts run 10-100x faster
✅ Can use GPU from Python easily
✅ Same API in Python and C++
✅ No data conversion overhead

### For Development:
✅ Easier to prototype in Python
✅ Test algorithms in Python before C++
✅ Better integration between Engine and Backend

### For Platform:
✅ Server Nodes can run fast Python jobs
✅ Marketplace scripts are performant
✅ Competitive with TensorFlow/PyTorch

---

## Next Phase Preview

**Phase 7 (Future):** Advanced pycyxwiz Features
- Custom operators in Python
- Automatic differentiation (autograd)
- Distributed training
- Model serialization/export
- TensorBoard integration
- Visualization tools

---

**Let's start with Task 1: Setup Infrastructure!**
