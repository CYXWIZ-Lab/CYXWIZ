---
title: Use the backend library from C++ and C
description: Call cyxwiz-backend directly from a C++ or C program, outside a plugin, to start it, pick a device, and work with tensors.
type: how-to
area: sdk
audience: plugin-developer
status: experimental
since: 1.0.8
owner: sdk
last_reviewed: 2026-11-25
related:
  - ../get-started/first-cpp-plugin.md
  - ../sdk-contents.md
  - ../tensor-operations.md
  - ../../python/quickstart.md
---

Use this when you want the Engine's compute library, `cyxwiz-backend`, in
your own program: tensors, layers, losses and optimizers on the GPU or the
CPU, with no Engine and no plugin around them. C++ programs use the full
API; C programs, and other languages through a C foreign-function interface,
use the smaller C API in `cyxwiz/cyxwiz_c.h`.

> Checked against plugin API 1.4.0 and pycyxwiz 0.2.0.

## Before you start

- The plugin SDK: its `include/` folder holds every backend header and its
  `lib/` folder the library to link (`cyxwiz-backend.lib` on Windows,
  `libcyxwiz-backend.so` on Linux, `libcyxwiz-backend.dylib` on macOS). See
  [Find the SDK](../get-started/first-cpp-plugin.md#find-the-sdk). A source
  build has the same headers in `cyxwiz-backend/include`.
- For C++: the Engine's compiler family and C++20, as for a plugin (see
  [The compiler](../get-started/first-cpp-plugin.md#the-compiler)). The C API
  has no such rule.
- At run time, the backend library and ArrayFire's libraries must be on the
  library search path (`PATH` on Windows, `LD_LIBRARY_PATH` on Linux).

## Start and stop the backend

Call `cyxwiz::Initialize()` once before anything else and
`cyxwiz::Shutdown()` at the end (both in `cyxwiz/cyxwiz.h`). `Initialize`
activates the first compute route that works among those the library was
built with, in this order: CUDA, oneAPI (only when the environment variable
`CYXWIZ_ENABLE_UNCERTIFIED_ONEAPI_TRAINING` is `1`), OpenCL, then the CPU.
It returns `false` only when none works. A second call returns `true` at
once.

<!-- not compiled: written from cyxwiz/cyxwiz.h and cyxwiz/device.h; no committed program shows it -->
```cpp
#include <cyxwiz/cyxwiz.h>
#include <cstdio>

int main() {
    if (!cyxwiz::Initialize()) return 1;
    std::printf("cyxwiz-backend %s\n", cyxwiz::GetVersionString());

    for (const cyxwiz::DeviceInfo& d : cyxwiz::Device::GetAvailableDevices()) {
        std::printf("%d:%d %s\n", static_cast<int>(d.type), d.device_id, d.name.c_str());
    }

    cyxwiz::Shutdown();
    return 0;
}
```

## Choose a device

`Device::GetAvailableDevices()` lists every device ArrayFire sees, as
`DeviceInfo` (type, id, name, memory, whether it is usable). To compute on
one, make a `cyxwiz::Device` and activate it:

- `SetActive()` switches to it; if that fails, it logs why and falls back
  to the CPU.
- `ActivateExact(validate_execution)` switches to it or reports why not
  (`DeviceActivationResult`: `success`, `stage`, `message`), with no
  fallback. Use it when a silent switch to the CPU would be wrong.

The active device is per thread: a worker thread calls
`Device::ActivateProcessDeviceOnThisThread()` first, or it computes on
ArrayFire's default device.

## Tensors, a model and a training step

Tensors are row-major and hold `float` by default. `Tensor::Zeros`, `Ones`,
`Random` and `Tensor(shape, data)` make them; the operators and the
functions on [Tensor operations](../tensor-operations.md) compute with them;
`ReadData<float>()` gives a host pointer to the values. A
`SequentialModel` (`cyxwiz/sequential.h`) chains modules, and a training
step is forward, loss, backward, update:

<!-- not compiled: written from cyxwiz/sequential.h, cyxwiz/losses/regression.h and cyxwiz/optimizers/adam.h; no committed program shows it -->
```cpp
#include <cyxwiz/cyxwiz.h>

void TrainStep(cyxwiz::SequentialModel& model, cyxwiz::Loss& loss,
               cyxwiz::Optimizer& optimizer,
               const cyxwiz::Tensor& x, const cyxwiz::Tensor& y) {
    model.SetTraining(true);
    cyxwiz::Tensor prediction = model.Forward(x);
    cyxwiz::Tensor value = loss.Forward(prediction, y);   // the loss, for logging
    model.Backward(loss.Backward(prediction, y));
    model.UpdateParameters(&optimizer);
}

// cyxwiz::SequentialModel model;
// model.Add<cyxwiz::LinearModule>(4, 8);
// model.Add<cyxwiz::ReLUModule>();
// model.Add<cyxwiz::LinearModule>(8, 1);
// cyxwiz::MSELoss loss;
// cyxwiz::AdamOptimizer optimizer(0.001);
```

The same classes are what pycyxwiz binds, so the
[Python quickstart](../../python/quickstart.md) and
[Optimizers and training loops](../../python/training-loops.md) show the
same flow step by step.

## The C API

`cyxwiz/cyxwiz_c.h` is a plain C interface over the same library: opaque
handles (`CyxWizTensor`, `CyxWizDevice`, `CyxWizOptimizer`), and functions
for the backend's start and stop, devices, tensors, five element-wise and
matrix operations, optimizers and memory counters.

| Group | Functions |
| --- | --- |
| Start and stop | `cyxwiz_initialize`, `cyxwiz_shutdown`, `cyxwiz_get_version` |
| Devices | `cyxwiz_device_get_count`, `cyxwiz_device_create`, `cyxwiz_device_set_active`, `cyxwiz_device_destroy` |
| Tensors | `cyxwiz_tensor_create`, `_create_with_data`, `_zeros`, `_ones`, `_random`, `_destroy`; `_num_elements`, `_num_bytes`, `_num_dimensions`, `_get_shape`, `_data`, `_data_const` |
| Maths | `cyxwiz_tensor_add`, `_sub`, `_mul`, `_div`, `_matmul` |
| Optimizers | `cyxwiz_optimizer_create` (SGD, Adam, AdamW, RMSprop, AdaGrad), `_set_learning_rate`, `_get_learning_rate`, `_destroy` |
| Memory | `cyxwiz_memory_get_allocated_bytes`, `_get_peak_bytes`, `_reset_peak` |
| Errors | `cyxwiz_get_last_error`, `cyxwiz_clear_last_error` |

Every function that makes a tensor returns a new handle that you free with
`cyxwiz_tensor_destroy`. On failure it returns `NULL`, and
`cyxwiz_get_last_error()` gives the reason (kept per thread).
`cyxwiz_device_set_active` falls back to the CPU as `SetActive` does. The C
API has no models, layers or losses: train from C++ or Python.

The repository's `examples/c_api/c_api_example.c` runs the whole API: start,
count devices, make tensors, add them, read the result, make an optimizer,
read the memory counters and clean up. Its core:

<!-- excerpt: examples/c_api/c_api_example.c -->
```c
size_t shape[] = {3, 4};  // 3x4 matrix
CyxWizTensor* tensor_a = cyxwiz_tensor_zeros(shape, 2, CYXWIZ_DTYPE_FLOAT32);
CyxWizTensor* tensor_b = cyxwiz_tensor_ones(shape, 2, CYXWIZ_DTYPE_FLOAT32);
// ...
CyxWizTensor* tensor_c = cyxwiz_tensor_add(tensor_a, tensor_b);
// ...
const float* data = (const float*)cyxwiz_tensor_data_const(tensor_c);
```

Build it against the SDK, with `$sdk` set as in
[Find the SDK](../get-started/first-cpp-plugin.md#find-the-sdk). With GCC or
Clang on Linux and macOS:

```bash
cc examples/c_api/c_api_example.c -I"$sdk/include" -L"$sdk/lib" -lcyxwiz-backend -o c_api_example
```

With MSVC on Windows (Developer PowerShell):

```powershell
cl examples\c_api\c_api_example.c /I"$sdk\include" "$sdk\lib\cyxwiz-backend.lib"
```

## Related

- [Your first C++ plugin](../get-started/first-cpp-plugin.md): the SDK, its
  folders and the compiler rule
- [What the plugin SDK contains](../sdk-contents.md): every header
- [Tensor operations](../tensor-operations.md): C++ and Python side by side
