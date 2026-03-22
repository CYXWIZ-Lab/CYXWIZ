# pycyxwiz Python Bindings: Data Structures and Linear Algebra

This guide documents the current Python binding surface for the CyxWiz backend (`pycyxwiz`), with emphasis on:

- Core data structures you manipulate from Python
- How those structures are used in training/data workflows
- How to translate common NumPy linear algebra code to `pycyxwiz.linalg`

## 1. Initialization and Session Lifecycle

```python
import pycyxwiz as cx

cx.initialize()
print(cx.get_version())
# ... work ...
cx.shutdown()
```

Inside CyxWiz Engine scripting, `pycyxwiz` is also auto-imported and aliased as `cyx`.

## 2. Core Python Data Structures

## 2.1 `Tensor`

`Tensor` is the primary numeric container exposed by the backend.

```python
t = cx.Tensor([32, 128], cx.DataType.Float32)
z = cx.Tensor.zeros([32, 128])
o = cx.Tensor.ones([32, 128])
r = cx.Tensor.random([32, 128])
```

Key APIs:

- `shape()`, `num_elements()`, `num_bytes()`, `num_dimensions()`, `get_data_type()`
- Element-wise operators: `+`, `-`, `*`, `/` (shape + dtype must match)
- NumPy bridge:
  - `cx.Tensor.from_numpy(np_array)`
  - `tensor.to_numpy()`

Supported NumPy dtypes for `from_numpy`:

- `float32`, `float64`, `int32`, `int64`, `uint8`

Notes:

- `from_numpy`/`to_numpy` copy memory (not a zero-copy view).
- `to_numpy` message indicates data should be on CPU.

## 2.2 `Sequential` and Parameter Dictionaries

`Sequential` is the main high-level model container for Python.

```python
model = cx.Sequential()
model.add_linear(784, 256)
model.add_relu()
model.add_linear(256, 10)
```

Core training-facing methods:

- `forward(input: Tensor) -> Tensor`
- `backward(grad_output: Tensor) -> Tensor`
- `get_parameters() -> dict[str, Tensor]`
- `get_gradients() -> dict[str, Tensor]`
- `update_parameters(optimizer)`

Parameter/gradient keys use layer-qualified naming:

- `layer0.weight`, `layer0.bias`, `layer1....`

Model control and persistence:

- `train()`, `eval()`, `set_training(bool)`
- `freeze_layer(i)`, `freeze_up_to(i)`, `freeze_except_last(n)`, `unfreeze_all()`
- `save(path)`, `load(path)`

## 2.3 `DataLoader` (DuckDB-backed SQL loader)

Python bindings expose the DuckDB SQL-oriented loader (`data_loader.h`), not the separate dataset-iterator class in `dataloader.h`.

```python
cfg = cx.DataLoaderConfig()
cfg.batch_size = 1024
cfg.verbose = True
loader = cx.DataLoader(cfg)
```

Main structs/classes:

- `DataLoaderConfig`
  - `batch_size`, `memory_limit_mb`, `num_threads`, `verbose`
- `ColumnInfo`
  - `name`, `type`, `nullable`, `index`
- `BatchIterator`
  - `has_next()`, `next()`, `reset()`, Python iterator protocol

Main `DataLoader` APIs:

- `DataLoader.is_available()`, `DataLoader.get_version()`
- `load_csv`, `load_parquet`, `load_json`
- `query(sql)`, `query_columns(sql)`
- `create_batch_iterator(sql, batch_size=0)`
- `get_schema(path)`, `get_columns(path)`, `get_row_count(path)`
- `convert_csv_to_parquet`, `convert_json_to_parquet`

Important conversion behavior:

- Query/file outputs are converted to `Float32 Tensor` matrices `[rows, cols]`.
- Unsupported/non-numeric SQL column types become `0.0` in the output tensor.
- For categorical/string columns, encode/cast in SQL (`CASE`, `CAST`) before fetching.

## 2.4 Optimizers and Losses

Optimizers:

- `SGD`, `Adam`, `AdamW`, `RMSprop`, `AdaGrad`, `NAdam`, `Adadelta`, `LAMB`
- Base type: `Optimizer`

Losses:

- `MSELoss`, `CrossEntropyLoss`, `FocalLoss`, `TripletLoss`, `ContrastiveLoss`, ...

Typical manual step:

```python
pred = model.forward(x)
loss = loss_fn.forward(pred, y)
grad = loss_fn.backward(pred, y)
model.backward(grad)
model.update_parameters(optimizer)
```

## 2.5 NLP and RL Structs

NLP:

- `Vocabulary`
- `Tokenizer`
- `TokenizerType` (`Whitespace`, `Word`, `Character`)

RL:

- `RLTransition`, `RLBatch`, `StepResult`, `EnvInfo`
- `ReplayBuffer`
- `EpsilonSchedule`

These are plain Python-exposed struct/class wrappers over backend C++ data containers.

## 2.6 Callback Calls (RL Metric Bridge)

`pycyxwiz` currently exposes callback registration for RL metric streaming:

- `cx.rl_set_metric_callback(callback)`
- `cx.rl_update_metric(name, value)`
- `cx.rl_set_metric_callback(None)` to clear

Call flow:

1. Python registers callback with `rl_set_metric_callback`.
2. `rl_update_metric` checks if callback exists.
3. Binding releases GIL before entering C++ bridge.
4. Bridge reacquires GIL and invokes `callback(name, value)`.
5. Callback receives metric name (`str`) and value (`float`).

Behavior and constraints:

- Callback slot is process-global (new registration replaces previous).
- `rl_update_metric` is a no-op when callback is not set.
- Exceptions in callback propagate to caller of `rl_update_metric`.
- Use `rl_should_stop()` and `rl_is_paused()` to control training loop execution.

Example:

```python
import time
import pycyxwiz as cx

def on_metric(name: str, value: float):
    print(f"{name}: {value:.4f}")

cx.rl_set_metric_callback(on_metric)

for step in range(100000):
    if cx.rl_should_stop():
        break
    while cx.rl_is_paused():
        time.sleep(0.05)

    # training...
    cx.rl_update_metric("episode_reward", 12.34)

cx.rl_set_metric_callback(None)
```

## 3. Linear Algebra in `pycyxwiz` vs NumPy

Linear algebra APIs live under `cx.linalg`.

```python
import pycyxwiz as cx
import numpy as np
```

## 3.1 Mapping Table

| NumPy | `pycyxwiz` |
|---|---|
| `np.eye(n)` | `cx.linalg.eye(n)` |
| `np.eye(r, c)` | `cx.linalg.eye(r, c)` |
| `np.zeros((r, c))` | `cx.linalg.zeros(r, c)` |
| `np.ones((r, c))` | `cx.linalg.ones(r, c)` |
| `np.diag(v)` | `cx.linalg.diag(v)` |
| `np.linalg.svd(A, full_matrices=False)` | `cx.linalg.svd(A, full_matrices=False)` |
| `np.linalg.eig(A)` | `cx.linalg.eig(A)` |
| `np.linalg.qr(A)` | `cx.linalg.qr(A)` |
| `np.linalg.cholesky(A)` | `cx.linalg.chol(A)` |
| `scipy.linalg.lu(A)` | `cx.linalg.lu(A)` |
| `np.linalg.det(A)` | `cx.linalg.det(A)` |
| `np.linalg.matrix_rank(A)` | `cx.linalg.rank(A)` |
| `np.trace(A)` | `cx.linalg.trace(A)` |
| `np.linalg.norm(A, ord='fro')` | `cx.linalg.norm(A)` |
| `np.linalg.cond(A)` | `cx.linalg.cond(A)` |
| `np.linalg.inv(A)` | `cx.linalg.inv(A)` |
| `A.T` | `cx.linalg.transpose(A)` |
| `np.linalg.solve(A, B)` | `cx.linalg.solve(A, B)` |
| `np.linalg.lstsq(A, B, rcond=None)[0]` | `cx.linalg.lstsq(A, B)` |
| `A @ B` / `np.matmul(A, B)` | `cx.linalg.matmul(A, B)` |

## 3.2 Input/Output Shape and Type Expectations

`cx.linalg` now has NumPy-native overloads for:

- `diag`
- `svd`, `eig`, `qr`, `chol`, `lu`
- `det`, `rank`, `trace`, `norm`, `cond`
- `inv`, `transpose`, `solve`, `lstsq`, `matmul`

`cx.linalg` also has Tensor-input overloads for:

- `solve`, `lstsq`, `matmul`, `norm`, `inv`, `transpose`

Input rules for ndarray overloads:

- Matrix operands are 2D arrays.
- `diag(d)` requires 1D input.
- `solve(A, b)` and `lstsq(A, b)` accept `b` as 1D or 2D.
- Inputs are cast to contiguous `float64` by the binding (`forcecast`).

Return behavior:

- ndarray overloads return NumPy arrays (or tuples of arrays).
- Tensor overloads return `Tensor` for matrix outputs (`norm` returns scalar).
- Legacy list overloads are still available for compatibility.
- `eye/zeros/ones` currently still return list-of-lists.

Legacy helpers are only needed when intentionally using list-path APIs:

```python
def to_cx_mat(a: np.ndarray):
    return np.asarray(a, dtype=np.float64).tolist()

def to_np(a):
    return np.asarray(a, dtype=np.float64)
```

## 3.2.1 Tensor-First Quickstart (Recommended for Repeated Compute)

If your workflow already uses `Tensor`, use Tensor inputs directly in `cx.linalg` to avoid list/ndarray marshalling in the hot path.

```python
import pycyxwiz as cx
import numpy as np

cx.initialize()

A = cx.Tensor.from_numpy(np.array([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64))
b = cx.Tensor.from_numpy(np.array([1.0, 2.0], dtype=np.float64))

x = cx.linalg.solve(A, b)      # Tensor output, shape [2]
C = cx.linalg.matmul(A, A)     # Tensor output, shape [2, 2]
Ai = cx.linalg.inv(A)          # Tensor output
At = cx.linalg.transpose(A)    # Tensor output
n = cx.linalg.norm(A)          # scalar float

print(x.shape(), C.shape(), n)
print(x.to_numpy())            # Convert at API boundary only

cx.shutdown()
```

Shape behavior for `solve` / `lstsq`:

- If `b` is 1D Tensor `[n]`, result is 1D Tensor `[n]`.
- If `b` is 2D Tensor `[n, k]`, result is 2D Tensor `[n, k]` for `solve`, `[m, k]` for `lstsq` (`A` is `[n, m]`).

CSV -> Tensor -> Frobenius norm:

```python
import pycyxwiz as cx

cx.initialize()
loader = cx.DataLoader()
loader.load_csv("data.csv")
X = loader.query("SELECT col1, col2, col3 FROM data")  # Tensor
f_norm = cx.linalg.norm(X)                              # Tensor path
print(f_norm)
cx.shutdown()
```

## 3.3 Example: ndarray-First Linalg Calls

```python
import pycyxwiz as cx
import numpy as np

A = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
b = np.array([9.0, 8.0], dtype=np.float64)  # 1D RHS supported

# Solve Ax = b
x = cx.linalg.solve(A, b)
print("x =", x)

# SVD
U, S, Vt = cx.linalg.svd(A, full_matrices=False)
print("U,S,Vt shapes:", U.shape, S.shape, Vt.shape)

# Matmul
C = cx.linalg.matmul(A, A)
print("C shape =", C.shape)

# Eig and LU
e_vals, e_vecs = cx.linalg.eig(A)
L, U_lu, P = cx.linalg.lu(A)
print("eig values dtype:", e_vals.dtype)
print("LU shapes:", L.shape, U_lu.shape, P.shape)
```

## 3.4 When to Prefer NumPy vs `cx.linalg`

Prefer NumPy when:

- You need advanced broadcasting and SciPy-heavy pipelines.
- You need widest ecosystem compatibility and mature CPU kernels.

Prefer `cx.linalg` when:

- You want MATLAB-style APIs inside CyxWiz scripts.
- You are chaining with other `pycyxwiz` modules.
- You want ArrayFire-backed execution paths where available.

Pragmatic pattern:

- Do feature/data prep in NumPy/Pandas/Polars.
- Convert to `Tensor` with `Tensor.from_numpy(...)` for model execution/training.
- Use ndarray-enabled `cx.linalg` where you want CyxWiz operators.
- Use `Tensor.to_numpy()` at boundaries where NumPy/SciPy tooling is needed.

## 3.5 Example: CSV -> Matrix -> Frobenius Norm (MATLAB-Style)

```python
import pycyxwiz as cx
import numpy as np

cx.initialize()

cfg = cx.DataLoaderConfig()
loader = cx.DataLoader(cfg)

# Load numeric columns from CSV with SQL, then convert Tensor -> NumPy
loader.load_csv("data.csv")
X_tensor = loader.query("SELECT col1, col2, col3 FROM data")
X = X_tensor.to_numpy().astype(np.float64, copy=False)

# Frobenius norm in pycyxwiz (ndarray path)
f_norm = cx.linalg.norm(X)
print("Frobenius norm:", f_norm)

cx.shutdown()
```

## 4. Minimal End-to-End Pattern

```python
import pycyxwiz as cx
import numpy as np

cx.initialize()

# NumPy -> Tensor
X_np = np.random.randn(128, 16).astype(np.float32)
y_np = np.random.randn(128, 4).astype(np.float32)
X = cx.Tensor.from_numpy(X_np)
y = cx.Tensor.from_numpy(y_np)

model = cx.Sequential()
model.add_linear(16, 32)
model.add_relu()
model.add_linear(32, 4)

loss_fn = cx.MSELoss()
opt = cx.Adam(learning_rate=1e-3)

pred = model.forward(X)
loss = loss_fn.forward(pred, y)
grad = loss_fn.backward(pred, y)
model.backward(grad)
model.update_parameters(opt)

# Tensor -> NumPy
pred_np = pred.to_numpy()
print(pred_np.shape)

cx.shutdown()
```

## 5. Backend Compute Model (ArrayFire + Device Backends)

`cyxwiz-backend` is built with optional ArrayFire acceleration:

- If ArrayFire is found at build time, `CYXWIZ_HAS_ARRAYFIRE` is enabled and many math/ML paths can execute on ArrayFire backends.
- Enabled backends can include CUDA and OpenCL (`CYXWIZ_ENABLE_CUDA`, `CYXWIZ_ENABLE_OPENCL`), with CPU available as fallback.
- Device selection is exposed in Python (`cx.get_device(...)`, `cx.set_device(...)`, `cx.get_available_devices()`).

Important nuance:

- The backend is not GPU-only. Many operations include CPU fallback code paths when GPU backend is unavailable or a backend call fails.
- `DataLoader` (DuckDB SQL loader) is CPU-side data processing and returns tensors for downstream ML use.
- `cx.linalg` can use ArrayFire internally when available, and ndarray overloads reduce Python marshalling overhead versus list-path usage.

## 5.1 CPU Backend and MKL Clarification

ArrayFire does handle backend selection/fallback, but MKL behavior is more specific:

- This repo does **not** directly link MKL in `cyxwiz-backend` CMake.
- `cyxwiz-backend` links ArrayFire and delegates CPU/GPU math backend details to ArrayFire.
- On CPU, ArrayFire uses whatever BLAS/LAPACK provider it was built with (for example MKL, OpenBLAS, or platform-specific providers).
- So it is not "pycyxwiz detects Intel CPU and then switches to MKL" in this codebase; MKL availability is determined by the ArrayFire build/package you installed.

Practical takeaway:

- Yes: fallback behavior is managed through ArrayFire backends.
- No: MKL is not guaranteed unless your ArrayFire CPU backend was built against MKL.

## 6. Binding Investigation: Where `tolist()` Is Still Slower

List-path overloads still use C++ STL conversions:

- matrix inputs: `std::vector<std::vector<double>>`
- vector inputs: `std::vector<double>`

If you call list-path overloads, Python still pays:

1. `np.ndarray -> Python list` conversion (`A.tolist()`)
2. `Python list -> C++ std::vector` conversion in pybind11
3. Compute in backend (possibly GPU via ArrayFire)
4. `C++ std::vector -> Python list` conversion on return
5. `Python list -> np.ndarray` conversion if needed

So list-path usage is still slower and more memory-heavy than ndarray-path usage.

In short:

- Backend compute can be fast.
- ndarray overloads remove most Python list boxing for covered `cx.linalg` calls.
- Remaining bottlenecks are uncovered APIs/submodules and backend algorithm maturity.
- NumPy/MATLAB can still be faster on many CPU workloads depending on ArrayFire backend/provider and matrix sizes.

## 6.1 `cx.linalg.solve(A, b)` Backend Call Flow

ndarray path (`A`, `b` are `np.ndarray`):

1. Python call `cx.linalg.solve(A, b)` enters pybind ndarray overload.
2. Binding validates `A` is 2D; `b` is 1D or 2D.
3. Binding converts ndarray inputs to `std::vector<std::vector<double>>`.
4. C++ `LinearAlgebra::Solve` (vector signature) runs dimension checks and dispatches:
   - ArrayFire path (`af::solve`) when GPU backend is active.
   - CPU path using LU decomposition + forward/back substitution otherwise.
5. Binding converts C++ matrix result back to ndarray.
6. If original `b` was 1D, binding returns 1D ndarray.

Tensor path (`tA`, `tb` are `Tensor`):

1. Python call enters pybind Tensor overload.
2. Binding calls `LinearAlgebra::Solve(const Tensor&, const Tensor&)`.
3. Core validates shapes (`A`: 2D square, `b`: 1D or 2D, row count match).
4. Core executes ArrayFire-first solve directly from Tensor input.
5. If ArrayFire path fails, core falls back to CPU solve logic.
6. Result returns as `Tensor` (1D shape if `b` was 1D, else 2D).

Is ndarray step 3 copy necessary?

- Yes, for ndarray/list signatures, because those APIs are still vector-of-vector based.
- This remains boundary overhead for ndarray path.

Tensor overload note:

- For `solve`, `lstsq`, `matmul`, `norm`, `inv`, `transpose`, Tensor path is now core Tensor-native.
- Remaining cost is host/device transfer behavior in current Tensor internals, not vector/list marshalling.

## 7. TODO Roadmap: Remaining Work After ndarray + Tensor-First Phase 2

Goal: keep closing ergonomics and performance gaps versus NumPy/MATLAB while preserving backward compatibility.

Reference architecture proposal:

- `docs/pycyxwiz_tensor_first_architecture.md`

## 7.1 Completed Patch Scope (`python/bindings.cpp`, `linear_algebra.*`)

- Status: done
- `np.ndarray` overloads using `py::array_t<double, c_style | forcecast>`
- Shape validation for matrix/vector cases
- ndarray returns for decomposition and matrix ops
- List-based overload retention for compatibility
- Tensor overloads for key ops (`solve`, `lstsq`, `matmul`, `norm`, `inv`, `transpose`)
- Core Tensor-native linalg implementations for the same key ops

Current ndarray-first usage:

```python
x = cx.linalg.solve(A, b)      # A, b are np.ndarray
C = cx.linalg.matmul(A, B)     # no .tolist()
U, S, Vt = cx.linalg.svd(A)
```

## 7.2 Completed: Core Tensor-First Refactor (Phase 2)

- Added core Tensor signatures in `LinearAlgebra` for:
  - `solve`, `lstsq`, `matmul`, `norm`, `inv`, `transpose`
- Updated Python Tensor overloads to call these Tensor-native core signatures directly.
- Removed Tensor-path dependency on binding-side matrix/list conversion helpers.
- Kept ndarray/list wrappers for compatibility.

Next TODO after phase 2:

- Reduce host/device transfer overhead in Tensor execution path (persistent device-resident tensor data flow).
- Extend Tensor-native core coverage to remaining linalg routines (`svd`, `eig`, `qr`, `chol`, `lu`, etc.).

## 7.3 TODO: Coverage Expansion

Apply the same ndarray-native pattern to:

- `stats` submodule (`kmeans`, `pca`, etc.)
- `signal` submodule (`conv2`, `spectrogram` matrix outputs, etc.)
- `timeseries` where matrix-like inputs are used
- Optional consistency upgrade for `linalg.eye/zeros/ones` output type

## 7.4 TODO: Performance and Correctness Validation

Add automated tests:

- Input validation (ndim, dtype casting, non-contiguous arrays)
- Numeric equivalence against list-path results
- Round-trip shape/dtype checks for ndarray-enabled outputs

Add benchmarks:

- Compare list path vs ndarray path for `matmul`, `solve`, `svd` across sizes
- Compare Tensor path vs ndarray path vs list path
- Report conversion cost vs compute cost

## 7.5 TODO: API Quality and Documentation

- Keep examples ndarray-first across all docs
- Add Tensor-first examples as primary path for repeated compute
- Keep list API documented as legacy path
- Add migration notes: replace `A.tolist()` with direct ndarray inputs

## 7.6 Optional Future Work (Best Performance Path)

For model/training flows that already use `Tensor`, add tensor-native linear algebra APIs (for example `matmul_tensor`) so data stays in backend tensor form and avoids repeated conversions at Python boundaries.
