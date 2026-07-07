# 48) Studio Debugger training-trace contracts

## 48.1 Scope and boundary

This section documents contracts that connect Studio Debugger and runtime execution to `TrainingTraceCollector`.

- how a live trace is started and reused
- how training stages are emitted from executor loops
- task-lifecycle trace emission from async workers
- materialization-to-trace coupling
- persistence format and durability policy
- UI consumption contract in Studio Debugger (live + historical views)
- recommendation wiring that consumes trace warnings and terminal state

Evidence coverage is limited to code artifacts in:
- `cyxwiz-engine/src/core/training_trace_collector.*`
- `cyxwiz-engine/src/core/async_task_manager.*`
- `cyxwiz-engine/src/core/training_manager.cpp`
- `cyxwiz-engine/src/core/training_executor.cpp`
- `cyxwiz-engine/src/gui/graph_training_launcher.cpp`
- `cyxwiz-engine/src/gui/main_window.cpp`
- `cyxwiz-engine/src/gui/panels/studio_debugger_panel.*`
- `cyxwiz-engine/src/core/debug_recommendation_engine.cpp`
- `cyxwiz-engine/src/core/debug_support_bundle_builder.cpp`

## 48.2 Trace model and singleton ownership

`TrainingTraceCollector` is a process-wide singleton (`Instance()`).

```text
Training trace event
  fields: timestamp, run_id, stage, thread_id, epoch/batch/totals
          loss/accuracy, validation_loss/validation_accuracy
          duration_ms
          cpu/AF memory counters
          status, message, metric_scope, checkpoint_path, terminal_reason
          task_id/task_name/task_stage/task_progress
          node_id/node_name/estimated_memory_bytes/processed_items/total_items
  emitted via: stages, runtime events, task progress, validation/ checkpoint / terminal
```

```text
Training trace summary
  latest_ fields: run_id, status, latest_stage/timestamp/epoch/batch/metrics
  latest warning list and bounded event windows
  materialization_events is a derived sub-stream from full events
```

Contract highlights:

- `TrainingTraceCollector` has in-memory bounded `events_` and `materialization_events_`, with default capacity settings from `TrainingTraceSettings`.
- all event payload writes are best-effort and cannot crash runtime (all persistence and memory calls are exception-safe wrappers).

## 48.3 Start/attach contract

The collector may be started by multiple pathways:

- graph launch preparation (`graph_training_launcher.cpp`) starts trace with generated `train-<ms>` id before materialization.
- trainer manager (`training_manager.cpp`) starts or attaches trace before spawning training:
  - if no running trace: `StartRun("train-" + ms)`
  - if trace already running: emit `TrainingSetup` runtime warning event and continue attach.
- executor path (`training_executor.cpp`) also re-checks snapshot and attaches to existing run if already active.

```text
Start/attach decision
  if !snapshot.available or snapshot.status != "running"
    -> StartRun(run_id)
  else
    -> RecordRuntimeEvent("TrainingLoop", "attached")
```

## 48.4 Stage emission contract

`TrainingTraceStage` comes from `CrashRunRecorder` enum and is stringified through `StageName`:

- `Start`, `GetNextBatch`, `Forward`, `ComputeLoss`, `Backward`, `UpdateParameters`,
  `BatchCallback`, `UIPlotUpdate`, `EpochComplete`, `EarlyStopped`, `Failed`, `Cancelled`

Executor stages and mapping:

- training loop:
  - every batch: `GetNextBatch`, `Forward`, `ComputeLoss`
  - if validation occurs: `ValidationCompleted` is emitted separately
- compute path:
  - `BatchCallback`, `UpdateParameters`, `EpochComplete`
- termination events:
  - `RecordTerminalEvent(early_stopped, ...)` on learning-rate plateau path
  - checkpoint save path emits `BestCheckpointUpdated` / `CheckpointSaved`

Contract semantics:

- `RecordStage` skips emission when no `run_id_`.
- each stage call stamps:
  - local time, stage name, thread id
  - epoch/batch context and optional loss/accuracy/duration
  - warning accumulation when status is not `"ok"` and message is present
- non-finite loss is explicitly mapped to `"failed"` and warning text is included.

## 48.5 Task-progress contract (async task plane)

Async work emits training-trace task events from lifecycle hooks.

- `AsyncTask::ReportProgress` -> `TaskProgress`
- `AsyncTask::MarkCompleted` -> `TaskCompleted`
- `AsyncTask::MarkFailed` -> `TaskFailed`
- `AsyncTaskManager::WorkerThread` start/stop transitions:
  - `TaskStarted` when execution begins
  - `TaskCancelled` when stop requested and state still running after execute
- `AsyncTask::RequestCancel` emits `TaskCancelRequested`

This includes:

- UI-run preparation tasks (`"Prepare graph training"`) with progress mapped to percentage.
- training manager startup task (`"Train ..."` path) with `TrainingSetup` progress event.

```text
AsyncTask lifecycle
  start -> TaskStarted
  progress updates -> TaskProgress
  completion -> TaskCompleted
  exception -> TaskFailed
  cancel flag -> TaskCancelled
```

## 48.6 Materialization coupling contract

Materialization callbacks in `graph_training_launcher` push task progress to trace with optional node metadata:

- `task_stage` from materializer event stage
- `node_id`, `node_name`
- memory estimate + processed/total counters

Collector behavior:

- any event that has node context is inserted into `materialization_events_`.
- fallback `LoadLastTrace` can rebuild materialization events from full events:
  `node_id >= 0` and (`metric_scope == "task"` or `task_id != 0`).

This enables Studio Debugger materialization views even if only compact events were recorded.

## 48.7 Persistence contract

Trace directory and serialization contract:

- directory: `<cwd>/.cyxwiz/debug_runs`
- current run file: `current_training_trace.json`
- persisted JSON keys: `run_id`, `status`, `events`, `materialization_events`, `warnings`

Write policy:

- write on `StartRun`
- write every N events (`persist_every_n_events`, min 1)
- immediate write on non-`ok` stage/task statuses
- `FinishRun` writes final status

`LoadLastTrace()` returns optional summary and reconstructs materialization stream if needed.

## 48.8 UI contract in Studio Debugger

Studio debug panel owns a dedicated trace settings section and render-time refresh path.

- settings (`renderTraceSettings`) configure:
  - persist enabled
  - write interval
  - max recent events
  - values are clamped (min/max) before applying to collector.

- live refresh:
  - `RefreshLiveTrainingTrace()` loads latest trace each render call
  - updates active session (`session_`) and current comparison session

- overview + training tabs:
  - `RenderLiveTrainingStatus` shows run id, status, latest epoch/batch/stage/loss/accuracy
  - latest warning classification color/category plus raw warning payload
  - `RenderTrainingTrace` prints:
    - latest stage/loss/accuracy
    - count and latest warnings
    - latest task/validation/checkpoint/terminal highlights
    - rolling event list
  - `RenderMaterializationTrace`, `RenderRuntimeTimeline`, `RenderMemoryTrace`, and `RenderLayerTimingBreakdown`
    read directly from `TrainingTraceSummary` and derive render rows.
- parse contract for layer timings:
  - accepts only `ModelForward` and `ModelBackward`
  - parses `layer`, `name`, `input`, `output`, `duration_ms` from message tokens.

```text
UI render flow
  RefreshLiveTrainingTrace
    -> session_.training_trace optional
    -> RenderLiveTrainingStatus
    -> RenderTrainingTrace
    -> RenderMaterializationTrace / RuntimeTimeline / MemoryTrace / LayerTimingBreakdown
```

## 48.9 Recommendation contract

`DebugRecommendationEngine::Build(...)` includes both `last_run` and `training_trace` inputs.

- all accumulated training warnings become recommendation warnings with category `"Training Trace"`.
- if trace is `running`, `latest_stage == "ComputeLoss"` and latest loss is non-finite,
  a critical recommendation is emitted with explicit action text.
- recommendations are surfaced in:
  - session build path for Studio Debugger run results
  - persisted run records
  - export support bundle field `training_trace`.

## 48.10 Run orchestration integration and contracts

Build-session integration in `MainWindow::BuildStudioDebuggerSessionFromSnapshot` uses three modes:

- runtime-only (`RuntimeTrace`):
  - load last crash run + last training trace
  - build recommendations
  - return without executing graph debug

- non-runtime path:
  - compile + preflight + optional smoke + optional local debug + optional runtime attach
  - save session to `DebugRunStore` in all terminal branches

This makes trace contract consistent across:
- pre-execution visibility (`RunOnlyRuntime`),
- active debug workflows (`LocalDebug`, `SmokeRun`, `FullWorkflow`),
- and persisted support artifacts (`DebugRunStore`, support bundle).

## 48.11 ASCII contracts

```text
Control-plane
AsyncTask::RequestProgress/Completion
  -> AsyncTaskManager WorkerThread
    -> TaskStarted / TaskProgress / TaskCompleted / TaskFailed / TaskCancelled
      -> TrainingTraceCollector::RecordTaskProgress
        -> StudioDebuggerPanel live/mat/timeline renders
```

```text
Training run attach + finish
Graph start path/Launcher -> StartRun(run_id)
  -> executor stage emissions in loop
  -> optional Validation/Checkpoint/Terminal events
  -> FinishRun("completed"|"early_stopped"|"cancelled"|"failed")
  -> Write to current_training_trace.json
```

```text
UI live consumption
RuntimeTrace mode requested
  -> MainWindow loads last trace
    -> session.training_trace
      -> Studio panel renders live status and recommendations
```

## 48.12 Evidence anchors

| Claim | Source |
|---|---|
| Trace model fields, settings, singleton methods | `cyxwiz-engine/src/core/training_trace_collector.h:13-139` |
| Event JSON schema, run id/write path, persistence behavior | `cyxwiz-engine/src/core/training_trace_collector.cpp:20-27`, `cyxwiz-engine/src/core/training_trace_collector.cpp:28-86`, `cyxwiz-engine/src/core/training_trace_collector.cpp:525-544`, `cyxwiz-engine/src/core/training_trace_collector.cpp:475-523` |
| Stage enum string map used by collector | `cyxwiz-engine/src/core/crash_run_recorder.h:13-27`, `cyxwiz-engine/src/core/crash_run_recorder.cpp:161-177` |
| Trace start/attach logic from executor and manager | `cyxwiz-engine/src/core/training_executor.cpp:474-483`, `cyxwiz-engine/src/core/training_manager.cpp:86-103` |
| Stage emission in training loops (GetNextBatch/Forward/ComputeLoss/Backward/UpdateParameters/BatchCallback/EpochComplete) | `cyxwiz-engine/src/core/training_executor.cpp:920-937`, `cyxwiz-engine/src/core/training_executor.cpp:980-990`, `cyxwiz-engine/src/core/training_executor.cpp:1488-1507`, `cyxwiz-engine/src/core/training_executor.cpp:1525-1528`, `cyxwiz-engine/src/core/training_executor.cpp:1599-1605`, `cyxwiz-engine/src/core/training_executor.cpp:1807-1810` |
| Validation/checkpoint/terminal + finish statuses | `cyxwiz-engine/src/core/training_executor.cpp:635-696`, `cyxwiz-engine/src/core/training_executor.cpp:661-667`, `cyxwiz-engine/src/core/training_executor.cpp:847-853`, `cyxwiz-engine/src/core/training_executor.cpp:870-882`, `cyxwiz-engine/src/core/training_executor.cpp:865-886` |
| Async task progress lifecycle to trace | `cyxwiz-engine/src/core/async_task_manager.cpp:70-129`, `cyxwiz-engine/src/core/async_task_manager.cpp:371-394`, `cyxwiz-engine/src/core/async_task_manager.cpp:415-423` |
| Materialization -> task progress -> materialization trace stream | `cyxwiz-engine/src/gui/graph_training_launcher.cpp:357-406`, `cyxwiz-engine/src/gui/graph_training_launcher.cpp:394-419`, `cyxwiz-engine/src/core/training_trace_collector.cpp:252-300` |
| Trace settings UI, load/refresh, render contracts | `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:700-737`, `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:843-880`, `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:1256-1370`, `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:1399-1437`, `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:1661-1805`, `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:1809-1856`, `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:286-314` |
| Runtime/preload recommendations from training trace | `cyxwiz-engine/src/core/debug_recommendation_engine.cpp:150-164` |
| MainWindow runtime-mode trace preload and recommendations integration | `cyxwiz-engine/src/gui/main_window.cpp:3444-3510`, `cyxwiz-engine/src/gui/main_window.cpp:3750-3762`, `cyxwiz-engine/src/gui/main_window.cpp:3494-3502` |
| Trace included in support bundle | `cyxwiz-engine/src/core/debug_support_bundle_builder.cpp:111-114` |
