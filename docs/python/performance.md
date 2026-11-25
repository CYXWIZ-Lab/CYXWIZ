---
title: Make pycyxwiz code fast
description: Which pycyxwiz path is fast and why: tensors on the device, NumPy arrays for linalg, whole-tensor operations, fewer copies, and device memory.
type: how-to
area: python
audience: user
status: experimental
since: 1.0.8
owner: python
last_reviewed: 2026-11-25
related:
  - tensors-and-devices.md
  - maths-and-optimization.md
  - ../concepts/compute-back-ends.md
---

Use this when a script is slower than it should be. pycyxwiz computes in
C++ on the device; time goes where Python hands data across, one element or
one list at a time.

> Checked against pycyxwiz 0.2.0. The samples on this page run, in order,
> as one script. The times in the comments were measured on one laptop
> (an NVIDIA GTX 1050 Ti with CUDA); yours differ, the order between them
> does not.

## The rule of thumb

| What you do | Use | Why |
| --- | --- | --- |
| Training, layers, many operations in a row | `Tensor` | The data stays on the device between operations |
| Linear algebra on data you have in NumPy | `pycyxwiz.linalg` with NumPy arrays | No conversion to lists |
| A quick calculation, a few numbers | Lists | Convenient; the slowest path for big inputs |

## Measure it

```python
import time
import numpy as np
import pycyxwiz as cx
import pycyxwiz.linalg as la

cx.initialize()

def ms(f, repeat=5):
    f()                                   # the first call pays one-off setup
    start = time.perf_counter()
    for _ in range(repeat):
        f()
    return (time.perf_counter() - start) / repeat * 1000

A = np.random.default_rng(0).normal(size=(300, 300))
T = cx.Tensor.from_numpy(A.astype(np.float32))
lists = ms(lambda: la.matmul(A.tolist(), A.tolist()))          # about 50 ms
arrays = ms(lambda: la.matmul(A, A))                           # about 6 ms
tensors = ms(lambda: la.matmul(T, T).to_numpy())               # about 0.5 ms, the copy back included
print(lists > arrays > tensors)                                # True
```

Tensor work is queued on the device and runs when a result is needed: time
it with a step that reads the result, such as `to_numpy()`, or you measure
only the queueing.

## Work on whole tensors, not elements

```python
t = cx.Tensor.random([100, 100], seed=1)
one_by_one = ms(lambda: sum(t.at(i, j) for i in range(100) for j in range(100)), repeat=1)   # about 45 ms
at_once = ms(lambda: float(t.sum().to_numpy()))                                              # under 1 ms
print(one_by_one > 10 * at_once)                                                              # True
```

Each `at` or `set` crosses from Python to the device and back. Express the
work as tensor operations (`sum`, `mean`, `clip`, `*`, comparisons and
masks), or do it in NumPy once and convert the result.

## Copy less

- **Convert once.** `from_numpy` and `to_numpy` each copy the whole array.
  Make the tensor once and reuse it; read results back only when you need
  the numbers (at the end of an epoch, not every step).
- **Batch your rows.** A forward pass over 256 rows costs little more than
  over one row: give the model batches, never one sample at a time in a
  loop.
- **Use `float32`.** Layers compute in `float32`; a `float64` array is
  twice the bytes to copy and is not what the model trains in. Convert with
  `astype(np.float32)` before `from_numpy`.
- **Contiguous arrays copy faster.** A sliced view (`X[:, ::2]`) is copied
  correctly but needs an extra pass; `np.ascontiguousarray` once before a
  loop avoids it.

```python
x = np.random.default_rng(1).normal(size=(256, 64)).astype(np.float32)
model = cx.Sequential()
model.add_linear(64, 256)
model.add_relu()
model.add_linear(256, 10)
xt = cx.Tensor.from_numpy(x)                    # once
batch = ms(lambda: model.forward(xt).to_numpy(), repeat=20)              # all 256 rows: about 1.3 ms
rows = ms(lambda: [model.forward(cx.Tensor.from_numpy(x[i:i + 1])) for i in range(256)], repeat=1)
print(rows > 10 * batch)                        # True: one row at a time is far slower
```

## Device memory

ArrayFire keeps the memory of freed tensors for reuse, so the memory a
process holds on the GPU stays up after tensors go. That is normal and
makes the next allocation fast. When another program needs the memory,
give it back:

```python
print(cx.system.release_unused_device_memory()["ran"])   # True
```

`cx.system.device_memory_held()` and `device_memory_free()` show the
figures ([Memory on the device](signals-statistics-and-text.md#memory-on-the-device)).

## The device itself

The GPU is fastest for large tensors and many operations; for small ones
the CPU can win, because each GPU call has a fixed cost. Inside the Engine
the device is the Engine's; standalone, `cx.initialize()` picks CUDA, then
OpenCL, then the CPU ([Tensors and devices](tensors-and-devices.md#devices)).

## Related

- [Tensors and devices](tensors-and-devices.md)
- [Compute back ends](../concepts/compute-back-ends.md): how the Engine
  chooses and uses a device.
