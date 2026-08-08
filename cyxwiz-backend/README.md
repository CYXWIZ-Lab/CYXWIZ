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

ArrayFire is discovered with `find_package(ArrayFire QUIET)`. When found, `CYXWIZ_HAS_ARRAYFIRE` is enabled and the backend links to ArrayFire. Without it, the backend still builds with reduced capability.

Not every operation currently has identical placement behavior. Callers and tests must distinguish requested backend, resolved backend, actual device, and fallback. Device selection is not proof of complete GPU execution.

## Build and test

From the repository root:

```powershell
cmake --preset windows-debug -DCYXWIZ_BUILD_ENGINE=OFF -DCYXWIZ_BUILD_SERVER_NODE=OFF -DCYXWIZ_BUILD_TESTS=ON
cmake --build --preset windows-debug
ctest --test-dir build -C Debug --output-on-failure
```

See [the root README](../README.md), [installation guide](../INSTALL.md), and [contribution guide](../CONTRIBUTING.md).
