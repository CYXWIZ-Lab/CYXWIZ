## 35) Phase 9 design register (runtime diagnostics, traceability, and failure contracts)

### 35.1 Objective
Define the runtime diagnostics architecture from launch to completion, including
trace persistence, failure semantics, and how UI/runtime state is reconstructed.

### 35.2 Observability topology

```text
Node graph UI action
  -> GraphTrainingLauncher (async prepare)
    -> Materialization + dataset dispatch
      -> TrainingManager (thread+task orchestration)
        -> TrainingExecutor (epoch/batch runtime loop)
          -> CrashRunRecorder + TrainingTraceCollector
            -> Debug/runtime consumers (status, recommendations, dashboards)
              -> main_window runtime-trace mode
```

### 35.3 Dual recorders (separation of concerns)

`CrashRunRecorder` (compact run heartbeat)
- Singleton with mutable active run context.
- Tracks coarse run lifecycle with `status`, `last_stage`, `epoch`, `batch`,
  `loss`, `accuracy`, warnings, terminal reason, and `suspected_crash`.
- Persists into:
  - `.cyxwiz/debug_runs/current_run.json`.
- Terminal transitions are explicit:
  - `MarkCompleted` -> `status="completed"`
  - `MarkEarlyStopped(reason)` -> `status="early_stopped"`
  - `MarkCancelled()` -> `status="cancelled"`
  - `MarkFailed(reason)` -> `status="failed"`

`TrainingTraceCollector` (high-fidelity diagnostic stream)
- Singleton with deque-based in-memory event storage and periodic persist.
- Keeps bounded recent event history and materialization/task warning streams.
- Persists into:
  - `.cyxwiz/debug_runs/current_training_trace.json`.
- Records:
  - per-stage execution metrics (`RecordStage`)
  - runtime messages (`RecordRuntimeEvent`, `RecordRuntimeWarning`)
  - task progress (`RecordTaskProgress`)
  - validation + checkpoint + terminal metrics
  - terminal completion (`FinishRun`)

### 35.4 Event contracts and persistence shape

`TrainingTraceEvent` fields (non-exhaustive contract):
- `timestamp`, `run_id`, `stage`, `thread_id`
- progress: `epoch`, `batch`, `total_batches`
- metrics: `loss`, `accuracy`, `validation_loss`, `validation_accuracy`,
  `duration_ms`
- runtime/task metadata: `status`, `message`, `metric_scope`,
  `task_id`, `task_name`, `task_stage`, `task_progress`
- topology and storage hints: `node_id`, `node_name`,
  `estimated_memory_bytes`, `processed_items`, `total_items`
- `checkpoint_path`, `is_best_checkpoint`, `terminal_reason`

`TrainingTraceSettings` retention controls:
- `persist_enabled` (default true)
- `persist_every_n_events` (default 10)
- `max_recent_events` (default 200)

`CrashRunSummary` fields for recovery-oriented state:
- `run_id`, `status`, `dataset_name`, `backend`, `last_stage`,
  `last_event_time`, `warning`, `terminal_reason`, `failure_reason`, `panel_events`
- optional Windows postmortem fields when available.

### 35.5 Stage model

Shared stages in `TrainingTraceStage`:
`Start`, `GetNextBatch`, `Forward`, `ComputeLoss`, `Backward`,
`UpdateParameters`, `BatchCallback`, `UIPlotUpdate`, `EpochComplete`,
`Complete`, `EarlyStopped`, `Failed`, `Cancelled`.

`TrainingExecutor` emits these stages at loop checkpoints:
- batch fetch/loop: `GetNextBatch`
- forward/loss: `Forward`, `ComputeLoss`
- backward/update: `Backward`, `UpdateParameters`
- callback points: `BatchCallback`, `UIPlotUpdate`
- epoch boundaries: `EpochComplete`

`CrashRunRecorder` maps terminal outcomes via:
- `MarkCompleted`, `MarkEarlyStopped`, `MarkCancelled`, `MarkFailed`.

### 35.6 Control and recovery contracts (ASCII)

```text
StartTrainingFromGraph
  -> Compile gate (must pass)
  -> GraphTrainingLauncher::StartGraphTrainingFromCompiledConfig
     -> Async task: materialize + resolve dataset route
        -> on success: TrainingManager::StartTraining*
           -> TrainingThread executes TrainingExecutor::Train
              -> per-step stage emission
              -> terminal event emitted on end condition
        -> on prepare failure: task error + no Train execution
```

Pause/stop semantics:
- `StopTraining()` sets stop flag in manager and executor.
- Execution loops check stop at epoch and batch boundaries.
- `Stop()` clears pause so loop can exit deterministically.
- `PauseTraining()` sets pause flag; loop blocks via `WaitWhilePaused()`.
- `ResumeTraining()` clears pause; loop continues.

Failure mapping at terminalization:
- stop path -> `CrashRunRecorder::MarkCancelled` + `TrainingTraceCollector::FinishRun("cancelled")`
- early stop -> `MarkEarlyStopped(reason)` + `FinishRun("early_stopped")`
- success -> `MarkCompleted()` + `FinishRun("completed")`
- exception -> `MarkFailed(coded_error)` + `FinishRun("failed")`

### 35.7 Recovery surface

Runtime-trace mode consumes persisted artifacts even without rerunning:
- `main_window` reads both `CrashRunRecorder::LoadLastRun()` and
  `TrainingTraceCollector::LoadLastTrace()` in runtime trace flows.
- Debug recommendations can then reference the last heartbeat + detailed event stream.

### 35.8 Evidence anchors

| Layer | Symbol | Notes |
|---|---|---|
| Debug/runtime launch and materialization tracing | `cyxwiz-engine/src/gui/graph_training_launcher.cpp` | `StartRun` and `RecordTaskProgress` during setup (`357`, `359`, `395`, `513`) |
| Lifecycle start and task tracking | `cyxwiz-engine/src/core/training_manager.cpp` | `StartTrainingCommon`, task polling, and completion marking (`88`, `95`, `129-134`) |
| Lifecycle control (`Stop`/`Pause`/`Resume`) | `cyxwiz-engine/src/core/training_manager.cpp` | `StopTraining`, `PauseTraining`, `ResumeTraining` (`562`, `576`, `589`) |
| Execution-stage instrumentation and terminal writeback | `cyxwiz-engine/src/core/training_executor.cpp` | stage recording and terminal status (`464`, `477`, `528`, `917`, `691`, `846-853`, `870`, `880`) |
| Trace collector contract + storage cadence | `cyxwiz-engine/src/core/training_trace_collector.h/.cpp` | event fields/settings and persistence path (`13-122`, `132`, `145`, `206`, `253`, `316`, `352`, `386`, `422`, `475`) |
| Failure/terminal heartbeat contract | `cyxwiz-engine/src/core/crash_run_recorder.h/.cpp` | enum + terminal states + load contract (`13-85`, `180`, `282`, `296`, `310`, `324`, `339`) |
| Runtime trace rehydrate from debug runs | `cyxwiz-engine/src/gui/main_window.cpp` | load points (`3478`, `3724`) |

