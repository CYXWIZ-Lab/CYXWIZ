## 9) Training orchestration and execution

### 9.1 Orchestrator layer (`TrainingManager`)
- Holds current training task state.
- Provides:
  - start,
  - pause,
  - resume,
  - stop,
  - active executor/plan references,
  - callback streams for UI updates.

### 9.2 Launcher flow
UI entry points call compile + preflight + launcher functions:
- compile/validate returns:
  - `graph_plan`,
  - compile issues,
  - preflight diagnostics.
- `StartTrainingFromGraph` chooses:
  - sequence mode branch,
  - direct training branch.
- Asynchronous launch path constructs `StartGraphTrainingFromCompiledConfig` and kicks a worker path.

### 9.3 `TrainingExecutor` modes and path split
`TrainingExecutor` has explicit mode constructors:
- legacy
- Arrow
- Parquet
- external
- sequence external

Runtime behavior includes:
- initialization and executable initialization,
- batcher creation for selected source type,
- validation interval handling,
- callback emission and pause/stop checks.

### 9.4 Batcher setup (`training_batcher_setup.*`)
- `ResolveTabularTrainingInputSize` drives input-size inference.
- Arrow and Parquet builders expose dataset/batch loops.
- Class weight and class-balancing hooks are part of compile-time configuration flow.

### 9.5 checkpoints and best-model behavior
Execution tracks:
- checkpoint intervals,
- validation metric tracking,
- best-checkpoint restore path,
- stop conditions (user stop, validation breakpoints, internal errors).

---
