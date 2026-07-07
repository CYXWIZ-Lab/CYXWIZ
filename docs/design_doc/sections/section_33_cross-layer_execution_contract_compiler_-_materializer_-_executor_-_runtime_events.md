## 33) Cross-layer execution contract (compiler -> materializer -> executor -> runtime events)

### 33.1 Canonical execution states

```text
Idle -> GraphReady -> CompileDone -> PreflightDone -> MaterializeDone -> ExecInitDone ->
Running -> Paused -> Stopping -> Completed -> Archived
```

### 33.2 State transition guards and blocking conditions

| From -> To          | Guard required                                      | Blocking issue families |
|---------------------|-----------------------------------------------------|------------------------|
| Idle -> GraphReady   | at least one valid graph snapshot                     | C_MISSING_*            |
| GraphReady -> CompileDone | graph compile pass + role extraction              | C_MISSING_* , C_PIN_MISMATCH |
| CompileDone -> PreflightDone | preflight summary returns non-error           | P_*                   |
| PreflightDone -> MaterializeDone | source + operators materialize if needed | M_*                   |
| MaterializeDone -> ExecInitDone | mode-specific executor init succeeds        | E_*                   |
| ExecInitDone -> Running | training thread/task started                        | E_THREAD_*             |
| Running -> Paused | user or external pause command                        | none                  |
| Running -> Stopping | stop requested or hard runtime fault                    | R_*                   |
| Paused -> Running | explicit resume accepted                                | R_*                   |
| Completed -> Archived | run summary written and lifecycle finalization         | none                  |

### 33.3 Ownership model per transition

- `GraphReady` and `CompileDone`: `main_window.cpp` owns orchestration and delegates to `GraphCompiler`.
- `PreflightDone`: `preflight_validator.cpp` is sole validator of runtime semantics.
- `MaterializeDone`: `pipeline_materializer.cpp` owns operator binding.
- `ExecInitDone` and `Running`: `TrainingExecutor` owns process lifecycle; `TrainingManager` owns UI-facing state mirror.
- `Paused`, `Stopping`, `Completed`, `Archived`: `TrainingManager` and `main_window.cpp` own callback persistence and status propagation.

### 33.4 Failure recovery map by contract breach

```text
Compile fail => clear launch state, present Compile issue list, require graph edit.
Preflight fail => keep graph frozen, highlight node/edge path.
Materialization fail => keep execution mode disabled, offer operator compatibility rewrite hints.
Exec init fail => retain materialized source metadata and request executor mode fallback.
Runtime fault => save partial state (if possible), surface run_id + exception trace.
```

### 33.5 Event and telemetry contract (minimum)

Every run has:
- `run_id`
- `phase` (`compile`, `preflight`, `materialize`, `train`, `validate`, `checkpoint`, `finish`)
- `component` (`compiler`, `materializer`, `executor`, `ui`)
- `code` (C_*, P_*, M_*, E_*, R_*)
- `message`

UI and executor should preserve these fields from launch start to final summary.

### 33.6 Minimal claim to evidence extension

Add this new claim ID when hardening is complete:

```text
Claim-ID        | Section | Claim text                                              | Evidence files | Evidence symbol | Owner
---------------+---------+----------------------------------------------------------+---------------+----------------+--------------
C-10-state-contract | 33 | State transitions are deterministic and failure coded | core/graph_compiler.cpp, core/pipeline_materializer.cpp, core/training_manager.cpp, core/training_executor.cpp, gui/main_window.cpp | core/graph_compiler.cpp:2749 GraphCompiler::Compile ; core/pipeline_materializer.cpp:74 PipelineMaterializer::Materialize ; core/training_manager.cpp:54 StartTrainingCommon ; core/training_manager.cpp:562 StopTraining ; core/training_manager.cpp:602 PauseTraining ; core/training_manager.cpp:609 ResumeTraining ; core/training_executor.cpp:168 Initialize ; core/training_executor.cpp:217 Train ; core/training_executor.cpp:1311 Stop ; core/training_executor.cpp:1316 Pause ; core/training_executor.cpp:1324 Resume ; gui/main_window.cpp:2963 MainWindow::StartTrainingFromGraph ; gui/main_window.cpp:3021 GraphCompiler Compile call ; gui/main_window.cpp:3062 materialization-failure branch ; gui/main_window.cpp:3235 PreflightValidator::Validate ; gui/main_window.cpp:3493 preflight trace setup | Runtime
```

### 33.7 Next small-cycle tasks

1. Tie `Phase 7` evidence review to this state machine.
2. Add run-state transition assertions in a short smoke-loop checklist.
3. Capture a snapshot update for 0b + section 26 after closure.
