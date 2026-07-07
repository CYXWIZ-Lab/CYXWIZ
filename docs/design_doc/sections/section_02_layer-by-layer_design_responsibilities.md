## 2) Layer-by-layer design responsibilities

### 2.1 application layer
Primary responsibilities:
- Process lifecycle and global initialization.
- Window and startup-mode dispatch (`STARTUP_PROJECT`, `STARTUP_GRAPH`, `STARTUP_NONE`).
- Delegates GUI ownership to `MainWindow`.

### 2.2 frontend/gui layer
Primary responsibilities:
- Editing the logical graph (nodes/pins/edges/metadata).
- Triggering compile, preflight, and launch commands from UI events.
- Surfacing compile reports and runtime status.

### 2.3 core compiler/runtime layer
Primary responsibilities:
- Translate editor graph to stable contracts (`TrainingConfiguration`, `CompiledGraphPlan`).
- Validate structural correctness and domain rules.
- Build executable models for runtime.
- Run training loops and materialization passes.
- Expose runtime capability boundaries before execution.

### 2.4 plugin layer
Primary responsibilities:
- Extend node catalog and execution behavior.
- Keep core clean while allowing external runtime registration.

---
