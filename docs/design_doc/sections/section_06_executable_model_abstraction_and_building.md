## 6) Executable model abstraction and building

### 6.1 `IExecutableModel`
Interface contract that decouples compiler artifacts from runtime mode:
- runtime model execution from plan graph,
- tensor/pin cache semantics,
- layer materialization boundaries.

### 6.2 Sequential vs graph executables
- `SequentialExecutableModel` for linear-layer-style execution.
- `GraphExecutableModel` for pin-based execution and caching.
  - stores:
    - `plan` reference,
    - layer and operator id maps,
    - pin/tensor cache (`FindCachedTensor` path).
  - exposes `CanRunLinearPlan` and `CanRunCurrentPlan` checks.

### 6.3 Model-builder split
From builder code:
- `BuildSequentialFromConfig` (legacy/sequential path),
- `BuildExecutableFromConfig`,
- `BuildGraphExecutableFromConfig`.

This is a deliberate migration path from older layer-by-layer execution toward generalized graph executable behavior.

---
