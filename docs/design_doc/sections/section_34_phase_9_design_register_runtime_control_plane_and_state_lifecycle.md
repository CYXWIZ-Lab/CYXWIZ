## 34) Phase 9 design register (runtime control-plane and lifecycle synchronization)

### 34.1 Objective
Map the full runtime control plane from GUI launch action to live training completion.

### 34.2 Design summary
Training is launched in three explicit layers:

```text
UI toolbar/shortcut
   -> synchronous compile/preflight gate
      -> StartGraphTrainingFromCompiledConfig
         -> async preparation task (materialize + validate launch)
            -> TrainingManager::StartTraining*
               -> background TrainingThread
                  -> TrainingExecutor::Train
                     -> UI callbacks + trace + callbacks
```

### 34.3 Launch gates and preconditions

#### 34.3.1 UI launch entry points
- Toolbar button/shortcut resolves into `StartTrainingFromGraph(nodes, links)`.
- Both `toolbar` start action and equivalent shortcuts invoke this same path.
- `StartTrainingFromGraph` performs:
  - compile gate (`BuildCompileResult`) and blocks on compile `Error`.
  - Local Debug staleness check.
  - re-compilation to obtain a fresh `TrainingConfiguration`.
  - dispatch through `StartGraphTrainingFromCompiledConfig`.

#### 34.3.2 Launch-result contract
`StartGraphTrainingFromCompiledConfig` returns:
- `started` boolean,
- status title/detail + effective dataset details + effective epochs/batch size,
- dataset/label/dispatch metadata.

Failure examples:
- no dataset configured,
- invalid config,
- missing dispatch callback,
- sequence mode requiring Arrow/Parquet but no registered source,
- task queue failure while starting async preparation or dispatch.

### 34.4 Execution plan and thread ownership

```text
UI thread
  | BuildCompileResult + BuildGraph
  v
graph_training_launcher (sync)
  | enqueue LambdaTask("Prepare graph training")
  v
AsyncTaskManager worker
  | materialize + resolve dataset name
  | validate sequence column requirements
  | call dispatch (TrainingManager::StartTraining*)
  v
TrainingManager::StartTraining*
  | create std::thread (TrainingThreadFunc)
  | hold current_executor_ + current_task_id_
  v
TrainingManager::TrainingThreadFunc
  | create callbacks
  | call TrainingExecutor::Train
  | collect terminal metrics
  | emit completion + cleanup
  v
TrainingExecutor::Train loop
  | owns batching, model execution, pause-stop checks
```

### 34.5 Dispatch and dataset routing contract

- The graph launcher is source-aware and resolves launch path inside one function:
  - sequence mode enabled:
    - `BuildSequenceBatcherFromArrowDataset` + config patching (`ApplySequenceBatcherBuildResultToTrainingConfig`),
    - `TrainingManager::StartTrainingSequence`.
  - registered non-sequence dataset loaders:
    - `loaders::GetByRegisteredDataset(dataset_name)`,
    - `LaunchTraining(...)`.
  - legacy fallback:
    - resolve `DataRegistry::GetDataset(dataset_name)`,
    - call `TrainingManager::StartTraining(...)` on legacy `DatasetHandle`.
- This keeps one UI launch function while preserving old and new runtime paths.

### 34.6 Materialization integration before dispatch

- During async launch task:
  - `PipelineMaterializer::Materialize(nodes, links, registry, effective_dataset_name, ...)`.
  - If `operators_applied > 0`, effective dataset name is updated.
  - For sequence launch, post-materialize schema is validated before training dispatch.
- If materialization fails, launch task throws and maps to preparation failure.
- Preparation progress is relayed into dashboard and TrainingTrace events.

### 34.7 State machine (runtime layer)

```text
Idle -> CompileGate -> LaunchQueued -> Materializing -> Dispatching -> Running
Running -> Paused -> Running
Running -> Stopping -> Completed/Stopped
Running/Stopped -> Cleanup -> Idle
```

Transition owners:
- `CompileGate` and `LaunchQueued`: `MainWindow::StartTrainingFromGraph` + `graph_training_launcher`.
- `Materializing`: `GraphTrainingLaunchResult` async task.
- `Dispatching` and `Running`: `TrainingManager` + `TrainingExecutor`.
- UI completion transitions: `TrainingManager::TrainingThreadFunc` cleanup path.

### 34.8 Pause / stop semantics

- `TrainingManager::StopTraining()`:
  - sets stop flag,
  - calls `TrainingExecutor::Stop()`,
  - cancels async task by task id.
- `TrainingExecutor::Stop()`:
  - sets `stop_requested_ = true`,
  - clears pause (`is_paused_ = false`) so loop can exit.
- `TrainingManager::PauseTraining()` -> `TrainingExecutor::Pause()` -> `is_paused_ = true`.
- `TrainingManager::ResumeTraining()` -> `TrainingExecutor::Resume()` -> `is_paused_ = false`.
- `TrainingExecutor::WaitWhilePaused()` is invoked in epoch loop before work and before batch fetch; training loop continues only when unpaused.
- `TrainingManager` thread cleanup sets `is_training_ = false`, emits final completion state, optionally preserves trained model/optimizer.

### 34.9 Callback and metrics contract

- Batch callback (per-step):
  - updates current epoch/batch/loss in cached metrics,
  - updates live panel batch progress and curve points.
- Epoch callback:
  - updates train/val metrics,
  - sequence-specific token/entity metrics,
  - marks validation metrics.
- completion callback receives final `TrainingMetrics`.
- UI panel is updated with final summary and completion status, plus run comparison record.

### 34.10 Failure and blocked-start behavior

- compile failure → `CompileGraph` popup in `Compile` or `BlockedTrain`.
- preflight/local-debug staleness failure under strict mode → blocked popup and no dispatch.
- materialization/runtime dispatch exception → preparation failed state, node editor deactivation, no training thread.
- `TrainingExecutor::Initialize` failure → training loop stops with invalid setup status.
- runtime thread stop path still marks launch as completed with run metadata if model artifacts exist.

### 34.11 Cross-layer ownership map (control plane)

- `main_window.cpp`: launch policy and user-facing status.
- `graph_training_launcher.cpp`: async prep graph, materialization, dataset routing.
- `training_manager.cpp`: long-running lifecycle orchestration and thread ownership.
- `training_executor.cpp`: epoch loop, batcher mode dispatch, stop/pause control, model lifecycle.
- `training_plot_panel` path: receives state updates (`SetTrainingState`, `SetBatchProgress`, `SetTrainingComplete`).

### 34.12 Evidence anchors

| Layer | Symbolic anchor |
|-------|-----------------|
| UI command -> training graph start | `cyxwiz-engine/src/gui/main_window.cpp:3034 MainWindow::StartTrainingFromGraph` |
| Start + debug staleness gates | `cyxwiz-engine/src/gui/main_window.cpp:3038` `:3048` `:3058` `:3155` |
| Dispatch contract + loader fallback | `cyxwiz-engine/src/gui/main_window.cpp:3155` `:3157` `:3161` `:3166` |
| Async launch + materialization task | `cyxwiz-engine/src/gui/graph_training_launcher.cpp:261` `:383` `:387` `:475` `:491` |
| Launch failure path + completion callback | `cyxwiz-engine/src/gui/graph_training_launcher.cpp:513` `:527` |
| Manager lifecycle orchestration | `cyxwiz-engine/src/core/training_manager.cpp:54` `:157` `:624` `:840` `:841` |
| Lifecycle control primitives | `cyxwiz-engine/src/core/training_manager.cpp:562` `:602` `:609` |
| Executor initialize/train/pause/stop | `cyxwiz-engine/src/core/training_executor.cpp:168` `:217` `:1311` `:1316` `:1324` |

### 34.13 Phase closure
- This section closes the remaining control-plane gap from compile to runtime loop ownership.
- Pending follow-up in this register:
  - if required, add a dedicated “control-plane recovery matrix” that maps each control transition to exact error recovery guidance.
