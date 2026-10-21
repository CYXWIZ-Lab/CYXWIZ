# CyxWiz examples

This directory contains public sample inputs and integration examples. Examples demonstrate a narrow API or workflow; they are not production configuration, checkpoints, or benchmark claims.

## Layout

- `c_api_example.c` — basic C API lifecycle and tensor operations.
- `cyx/` and `cyx_tests/` — scripting examples and focused script checks.
- `cyxgraph/` — saved graph examples and contract fixtures.
- `data/` — small redistributable input samples used by examples and the Engine build.
- `python/` and `python_tests/` — Python API examples and parity checks.
- `plugins/` — plugin host examples.

Build CyxWiz first using the [installation guide](../INSTALL.md). Examples that use optional Python, ONNX, GPU, or plugin features require those capabilities to be enabled and verified in the build.

Run an example from the repository root unless its own README says otherwise. Do not infer production readiness, performance, or complete device placement from an example completing successfully.
