---
title: Tensors and devices
description: Create pycyxwiz tensors, choose their data type, move data to and from NumPy, reshape and reduce them, and choose the GPU or the CPU.
type: how-to
area: python
audience: user
status: experimental
since: 1.0.8
owner: python
last_reviewed: 2026-11-25
related:
  - reference/tensors-and-devices.md
  - ../sdk/tensor-operations.md
  - performance.md
  - ../concepts/compute-back-ends.md
---

Use this when you compute with `pycyxwiz.Tensor`: the multi-dimensional
array that every layer, loss and optimizer of pycyxwiz takes and returns.
A tensor's data lives on the compute device (a GPU, or the CPU), where
ArrayFire does the work.

> Checked against pycyxwiz 0.2.0. The samples on this page run, in order,
> as one script.

## Create tensors

```python
import numpy as np
import pycyxwiz as cx

cx.initialize()
T = cx.Tensor

a = T.zeros([2, 3])                       # float32 unless you ask otherwise
b = T.ones([2, 3], cx.DataType.Float64)
c = T.full([2, 2], 7.0)
d = T.range_n([2, 3])                     # 0, 1, 2, ... in row-major order
r = T.random([2, 3], seed=5)              # uniform in [0, 1); one seed, one result
print(d.to_numpy().tolist())              # [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
print(repr(b))                            # <Tensor shape=[2, 3] dtype=float64>
print(d.shape(), d.num_elements(), d.num_dimensions(), d.num_bytes())   # [2, 3] 6 2 24
```

Without a seed, `Tensor.random` draws from the library's random stream; the
same seed always gives the same values.

## Data types

| `cx.DataType` | NumPy | Notes |
| --- | --- | --- |
| `Float32` | `float32` | The default, and what layers and losses train in |
| `Float64` | `float64` | Kept as float64 through `from_numpy` and `to_numpy` |
| `Int32`, `Int64` | `int32`, `int64` | Class indices for `CrossEntropyLoss` |
| `UInt8` | `uint8` | What comparisons give |

Other NumPy types (`float16`, `int16`, `bool`) raise `RuntimeError:
Unsupported NumPy dtype`: convert with `astype` first.

## NumPy in and out

```python
x = np.arange(6, dtype=np.float32).reshape(2, 3)
t = T.from_numpy(x)                       # copies the array to the device
back = t.to_numpy()                       # copies it back, same dtype
print(np.array_equal(x, back), back.dtype)   # True float32

view = np.arange(9, dtype=np.float32).reshape(3, 3)[:, ::2]
print(T.from_numpy(view).to_numpy().tolist())   # [[0.0, 2.0], [3.0, 5.0], [6.0, 8.0]]
```

Both directions copy: changing `x` afterwards does not change `t`. A
non-contiguous view, such as every second column, is copied correctly.

## Arithmetic and functions

```python
x = T.from_numpy(np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32))
y = x * 2.0 + 1.0                         # numbers on either side: 2.0 * x, 1.0 - x
print(y.to_numpy().tolist())              # [[3.0, -3.0], [7.0, -7.0]]
print((x * x).to_numpy().tolist())        # element-wise: [[1.0, 4.0], [9.0, 16.0]]
print([round(float(v), 3) for v in x.abs().sqrt().to_numpy().ravel()])   # [1.0, 1.414, 1.732, 2.0]
print(cx.relu(x).to_numpy().tolist())     # [[1.0, 0.0], [3.0, 0.0]]
```

`*` is element-wise. A matrix product is `pycyxwiz.linalg.matmul(a, b)` for
2-D tensors, or `a.batch_matmul(b)` for 3-D batches; there is no `@`.

Element-wise functions: `abs`, `exp`, `log`, `sqrt`, `sin`, `cos`, `sign`,
`pow` (or `**`), `clip(min_val, max_val)`, `logical_not`. Activations are
module functions: `cx.relu`, `cx.sigmoid`, `cx.tanh`, `cx.softmax`,
`cx.gelu` and the others on [Activations](reference/activations.md).

## Comparisons give uint8

```python
mask = x > 0.0
print(mask.get_data_type(), mask.to_numpy().tolist())   # DataType.UInt8 [[1, 0], [1, 0]]
positive = mask * 1.0                     # float32 from here on
print(float((x * positive).sum().to_numpy()))   # 4.0: the sum of the positive values
```

`uint8` with `uint8` stays `uint8` and wraps around past 255; multiply a
mask by `1.0` before you add masks up.

## Shapes

```python
m = T.range_n([2, 3])
print(m.reshape([3, 2]).shape(), m.transpose().shape())          # [3, 2] [3, 2]
print(T.zeros([2, 3, 4]).permute([2, 0, 1]).shape())             # [4, 2, 3]
print(m.unsqueeze(0).shape(), m.unsqueeze(0).squeeze().shape())  # [1, 2, 3] [2, 3]
print(m.slice(1, 0, 2).to_numpy().tolist())       # columns 0 and 1: [[0.0, 1.0], [3.0, 4.0]]
print(m.index_select(1, [2, 0]).to_numpy().tolist())   # [[2.0, 0.0], [5.0, 3.0]]
print(T.cat([m, m], 0).shape(), T.stack([m, m], 0).shape())      # [4, 3] [2, 2, 3]
```

`slice(dim, start, end, step)` takes rows or columns; a tensor has no
`t[i]` indexing and no `len(t)`. Read and write one element with
`t.at(i, j)` and `t.set(i, j, value)`, but do not loop over elements in
Python: a reduction or an element-wise operation is far faster
([Performance](performance.md)).

## Reductions

```python
print(float(m.sum().to_numpy()))                  # 15.0
print(m.sum(dim=0).to_numpy().tolist())           # [3.0, 5.0, 7.0]
print(m.mean(dim=1, keepdim=True).shape())        # [2, 1]
print(float(m.max().to_numpy()), round(float(m.std().to_numpy()), 4))   # 5.0 1.7078
```

`sum`, `mean`, `max`, `min`, `prod`, `var` and `std` (population) reduce
everything, or one dimension with `dim`; `keepdim=True` keeps that dimension
with size 1. A full reduction has shape `[]`: read it with
`float(t.to_numpy())`.

## Devices

`initialize()` starts the backend on the first that works of CUDA, OpenCL
and the CPU. List what this machine has and what is in use:

```python
for info in cx.get_available_devices():
    print(info.type, info.device_id, info.name, info.kind)
# for example: DeviceType.CUDA 0 NVIDIA GeForce GTX 1050 Ti DeviceKind.GPU
current = cx.Device.get_current_device()
print(current.get_type(), current.get_info().name)
print(cx.cuda_available(), cx.opencl_available())
```

To compute on another device, make it active **before** you create the
tensors you use on it:

```python
start = cx.Device.get_current_device()
cpu = cx.Device(cx.DeviceType.CPU, 0)
result = cpu.activate_exact()             # or cpu.set_active()
print(result.success, result.effective_type)   # True DeviceType.CPU

on_cpu = T.range_n([2, 2]) * 2.0          # made and computed on the CPU
print(on_cpu.to_numpy().tolist())         # [[0.0, 2.0], [4.0, 6.0]]

start.set_active()                        # back to the device you started on
```

A tensor stays on the backend it was made on. After you switch from CUDA to
the CPU (or back), every older tensor fails with "There was a mismatch
between an array and the current backend", even `to_numpy()`. Switch once, at
the start of the script, or bring data across through NumPy: `to_numpy()`
before the switch, `from_numpy()` after it. Inside the Engine, the device is
the one the Engine computes on; change it in the Engine
([Compute back ends](../concepts/compute-back-ends.md)) rather than from a
script.

How much memory the process holds on the device, and how to give unused
memory back, is in
[Signals, statistics and text](signals-statistics-and-text.md#memory-on-the-device).

## Related

- [Tensors and devices reference](reference/tensors-and-devices.md):
  every member of `Tensor`, `Device` and `DeviceInfo`.
- [Tensor operations for nodes](../sdk/tensor-operations.md): the same
  operations next to their C++ names.
