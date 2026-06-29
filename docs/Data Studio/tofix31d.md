# tofix31d - Pinned-memory boundary

## Scope

This is the fourth small ticket extracted from `tofix31.md`.

The goal is to keep `DataLoader.pin_memory` truthful. This ticket does not add
real pinned host-memory transfers. It makes the current unsupported behavior
visible through compiler issues, not only logs and UI text.

## Current engine contract

- `pin_memory` is serialized for graph compatibility.
- Current Arrow/Parquet/tabular batchers do not allocate pinned host memory.
- Current training execution does not have a pinned host-to-device transfer
  path.
- `pin_memory=true` is ignored at runtime.

## Changes made

- `cyxwiz-engine/src/core/graph_compiler.cpp`
  - Emits a normal compiler warning issue when a selected `DataLoader` has
    `pin_memory=true`.
  - Keeps the previous log warning.
  - Does not invalidate the graph, because the setting is compatibility-only
    and safely ignored.
- `cyxwiz-engine/tests/test_graph_compiler_deferred_nodes.cpp`
  - Existing DataLoader contract test now asserts that `pin_memory=true`
    surfaces as an unsupported compiler warning.

## Existing truthfulness already present

- `cyxwiz-engine/src/gui/data_io_dialogs.cpp`
  - Shows `pin_memory` as unsupported.
  - Explains that loaded graphs with `pin_memory=true` are ignored by current
    batchers.
- `cyxwiz-engine/src/gui/node_editor_nodes.cpp`
  - Initializes `pin_memory=false` as serialized compatibility only.

## Not done in this ticket

- No pinned allocation primitive.
- No pinned host-memory pool.
- No async H2D copy path.
- No DataLoader ownership of pinned buffers.
- No benchmark proving transfer improvement.

## Future implementation requirements

A real pinned-memory implementation should be a separate ticket with:

- backend abstraction for pinned host allocation/free,
- batcher-owned buffer lifecycle,
- CPU fallback when the active backend cannot use pinned memory,
- transfer benchmark covering CPU, CUDA/ArrayFire, and no-GPU paths,
- explicit shutdown cleanup to avoid leaking pinned pages.
