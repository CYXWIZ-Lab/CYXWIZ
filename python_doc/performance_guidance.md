# PyCyxWiz Performance Guidance

This short guide explains when to use `Tensor`, NumPy ndarray compatibility, and legacy list-style utilities.

## Summary
- Use **`Tensor`** for highest performance, GPU acceleration, and multi-op workloads.
- Use **NumPy ndarray-compatible `pycyxwiz.linalg`** for convenient linear algebra without `.tolist()`.
- Use **legacy list-style calls** for quick MATLAB-like prototypes or backward compatibility.

## Why List Inputs Are Not the Fast Path
List-style APIs are convenient, but list-based usage:
- Requires Python-to-C++ data conversion
- Is not optimized for large-scale training workloads
- Is best for quick math, inspection, or prototyping

## ndarray Compatibility Patch (What Changed)
`pycyxwiz.linalg` now accepts NumPy arrays directly for major operations:

- `diag`
- `svd`, `eig`, `qr`, `chol`, `lu`
- `det`, `rank`, `trace`, `norm`, `cond`
- `inv`, `transpose`, `solve`, `lstsq`, `matmul`

This removes the required `.tolist()` conversion for these calls.

## Recommended High-Performance Path
For serious compute:
1. Use NumPy for data prep
2. Convert to `Tensor`
3. Run layers/ops on `Tensor`

Example:
```python
import numpy as np
import pycyxwiz as cx

cx.initialize()

x = np.random.randn(1024, 1024).astype(np.float32)
t = cx.Tensor.from_numpy(x)

layer = cx.Dense(1024, 512)
out = layer.forward(t)
```

## ndarray Linalg Path (Good Default for Python Numerics)
If your data is already in NumPy and you want CyxWiz linalg:

Example:
```python
import pycyxwiz.linalg as la
import numpy as np

A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[5.0, 6.0], [7.0, 8.0]])
C = la.matmul(A, B)

b = np.array([7.0, 9.0])
x = la.solve(A, b)
```

## Legacy List Path (Compatibility)
Still supported, but slower for larger workloads:

```python
import pycyxwiz.linalg as la

A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C = la.matmul(A, B)
```

## Rule of Thumb
- If peak performance matters: **`Tensor`**
- If convenience with NumPy matters: **ndarray-compatible `pycyxwiz.linalg`**
- If quick compatibility/prototyping matters: **list-style MATLAB calls**

End of document.
