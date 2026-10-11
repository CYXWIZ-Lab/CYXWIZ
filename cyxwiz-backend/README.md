# CyxWiz Backend

`cyxwiz-backend` is the shared C++20 computation library used by the Engine and Server Node.

## Responsibilities

- tensor storage, shape, indexing, reductions, and linear algebra;
- model, sequential, layer, activation, loss, optimizer, and scheduler contracts;
- data loading and batching primitives;
- evaluation, time-series, signal, text, and selected classical-ML utilities;
- device selection and runtime placement evidence;
- optional Python and C API bindings.

The public C++ entry point is `include/cyxwiz/cyxwiz.h`. Implementation lives under `src/core` and `src/algorithms`.

## ArrayFire

ArrayFire is required: configure fails early if `find_package(ArrayFire)` does not find it, whatever else is enabled. The backend computes on ArrayFire only; the CPU option is ArrayFire's CPU backend, and there are no hand-written CPU kernels. The backend links ArrayFire publicly and defines `CYXWIZ_HAS_ARRAYFIRE`, which only sources shared with the ArrayFire-free installer still test.

Not every operation currently has identical placement behavior. Callers and tests must distinguish requested backend, resolved backend, actual device, ArrayFire CPU backend selection, and native CPU compute fallback. ArrayFire execution backends are CPU, CUDA, oneAPI, and OpenCL; Metal/Vulkan are not ArrayFire backends in the current Engine path. Device selection is not proof of complete GPU execution.

## Build and test

From the repository root:

```powershell
cmake --preset windows-debug -DCYXWIZ_BUILD_ENGINE=OFF -DCYXWIZ_BUILD_SERVER_NODE=OFF -DCYXWIZ_BUILD_TESTS=ON
cmake --build --preset windows-debug
ctest --test-dir build -C Debug --output-on-failure
```

See [the root README](../README.md), [installation guide](../INSTALL.md), and [contribution guide](../CONTRIBUTING.md).
