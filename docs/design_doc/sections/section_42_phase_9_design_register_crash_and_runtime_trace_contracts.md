# 42) Phase 9 design register (crash and runtime trace contracts)

## 42.1 Objective
Define the complete observability contract for runtime crash handling and training trace generation, including heartbeat schema, stage/event recording, persistence policy, WER integration, and UI/diagnostic consumption flow.

## 42.2 ASCII contracts

```text
TrainingManager::StartTrainingCommon
  -> trace snapshot check (TrainingTraceCollector::Snapshot)
  -> StartRun("train-<ms>") if no running run
  -> RecordTaskProgress(task_name, "TrainingSetup", status=running)
  -> AsyncTaskManager::RunAsync(task_name, training thread)
      -> TrainingExecutor::Train()
         -> CrashRunRecorder::StartTrainingRun(config_, epochs,...)
         -> TrainingTraceCollector::StartRun(run_id)
         -> for each epoch/batch:
              MarkStage/Get/Forward/ComputeLoss/Backward/BatchCallback/EpochComplete
              Mark/RecordValidationMetrics (if validation run)
              Mark/RecordCheckpointSaved (best checkpoint)
              Mark/RecordTaskProgress (manager-level progress callbacks)
              Mark/RecordRuntimeEvent (backend debug, warnings)
         -> terminal path:
              cancelled / early_stopped / completed / failed -> FinishRun / RecordTerminalEvent
         -> propagate final state to TrainingPlotPanel (terminal_status / reason)
```

```text
Studio Debug Runtime Trace mode
  BuildStudioDebuggerSession(mode=RuntimeTrace)
    -> LoadLastRun() from .cyxwiz/debug_runs/current_run.json
    -> LoadLastTrace() from .cyxwiz/debug_runs/current_training_trace.json
    -> recommendation engine consumes both and returns run-level diagnosis
  StudioDebuggerPanel render pipeline:
    -> session bootstrap refresh + trace settings from collector
    -> live render of crash + training trace + materialization + runtime + memory
```

## 42.3 Data contracts and persistence endpoints

### 42.3.1 Crash heartbeat contract (`CrashRunRecorder`)
- Runtime heartbeat file: `.cyxwiz/debug_runs/current_run.json`.
- Public summary contract: `CrashRunSummary`:
  - `run_id`, `status`, `last_stage`, `last_event_time`, `epoch`, `batch`,
    `total_batches`, `loss`, `accuracy`, `dataset_name`, `backend`,
    `terminal_reason`, `failure_reason`, `warning`, `panel_events`, WER metadata.
- Runtime event enum `TrainingTraceStage` mapped through `StageName(...)`:
  - `Start`, `GetNextBatch`, `Forward`, `ComputeLoss`, `Backward`,
    `UpdateParameters`, `BatchCallback`, `EpochComplete`, `Complete`, `EarlyStopped`,
    `Failed`, `Cancelled`.
- Writer is always guarded under mutex and writes JSON fields for:
  - active config (`dataset_name`, `backend`, dimensions, loss/acc state, reasons)
  - stage+event arrays (`panel_events`).

### 42.3.2 Training timeline contract (`TrainingTraceCollector`)
- Runtime timeline file: `.cyxwiz/debug_runs/current_training_trace.json`.
- Event contract: `TrainingTraceEvent` with fields for `timestamp`, `run_id`, `stage`,
  `thread_id`, epoch/batch counters, loss/accuracy/memory snapshots, status/message,
  task and validation dimensions, checkpoint details, warnings, and terminal reason.
- Summary contract: `TrainingTraceSummary`:
  - `run_id`, `status`, `latest_*`, `recent_events`, `materialization_events`, `warnings`.
- Mutable settings contract: `TrainingTraceSettings`
  - `persist_enabled`
  - `persist_every_n_events`
  - `max_recent_events`
- Persistence policy:
  - Write lock under mutex.
  - On `RecordStage`: conditional writes on N-event interval and non-`ok` statuses.
  - On warning/terminal/runtime-event: immediate write to keep crash-critical evidence.
  - Memory/event buffers are bounded by `max_recent_events`.

### 42.3.3 Memory snapshot side-channel
- `PopulateMemorySnapshot()` records:
  - CPU allocated/peak bytes from `MemoryManager`.
  - ArrayFire allocated/locked bytes and buffer counts (best-effort, exception-safe).
- Snapshot is attached to each stage/task record.

## 42.4 Recorder lifecycle contract

### 42.4.1 Start/initialize
- `TrainingManager::StartTrainingCommon`:
  - If no running trace, starts run (`training-task-<ms>` style timestamp).
  - Otherwise emits runtime event indicating attachment to existing trace.
  - Records task-level progress record (`task_name`, stage `TrainingSetup`).
- `TrainingExecutor::Train`:
  - Starts crash recorder run with config metadata.
  - Starts training trace if absent, else records runtime event for loop attachment.
  - Connects backend debug callback:
    - `source` prefixed `Model` => `RecordRuntimeEvent`
    - other sources => `MarkBackendEvent` + `RecordRuntimeWarning`.

### 42.4.2 Progress/events during active training
- Stage-level instrumentation appears in:
  - legacy `RunTrainingEpoch`
  - `RunTrainingEpochArrow`
  - `RunTrainingEpochSequence`
- For each active batch and epoch:
  - `GetNextBatch`, `Forward`, `ComputeLoss`, `Backward`, optional `UpdateParameters`,
    `BatchCallback`, `EpochComplete`.
- UI plot updates are traced as `UIPlotUpdate` from:
  - epoch callback (`terminal_status`, timings, val metrics)
  - per-batch callback (`running loss/accuracy`).
- Validation checkpoints:
  - `RecordValidationMetrics` when validation is executed.
  - `RecordCheckpointSaved` when best validation checkpoint succeeds.

### 42.4.3 Terminalization and failure semantics
- Terminal status machine in `TrainingExecutor`:
  - `cancelled` if stop requested
  - `early_stopped` if plugin/validation policy set terminal status
  - `completed` otherwise
  - `failed` on caught exceptions
- Terminal recording rules:
  - `MarkCancelled` / `MarkEarlyStopped` / `MarkCompleted` / `MarkFailed`
  - `FinishRun(<status>)` and `RecordTerminalEvent(status, reason, epoch, loss, acc)`.
- Training plot terminal row source:
  - `SetTrainingComplete(total_time, terminal_status, terminal_reason, ...)` from manager.

## 42.5 Crash diagnosis contract

### 42.5.1 Crash suspicion and WER attachment
- `LoadLastRun()` sets `suspected_crash=true` when persisted status is still `running`.
- Adds user-visible `warning` when no clean terminal marker exists.
- On supported platform, scans `ProgramData`/`LocalAppData` AppCrash folders for `.wer`,
  extracts report id/module/exception/time and marks `windows_crash_available`.

### 42.5.2 Recovery/telemetry posture
- Heartbeat updates are incremental and overwrite state in a single file; this preserves:
  - last known stage/event
  - latest terminal reason/failure strings
  - panel and backend event tails.
- Non-fatal contract: writer failures are swallowed to prevent telemetry from aborting training.

## 42.6 UI/consumers contract

### 42.6.1 Studio debugger runtime mode
- `MainWindow::BuildStudioDebuggerSession` with `StudioDebuggerRunMode::RuntimeTrace`
  - does not re-run debug/execution
  - loads last-run + training-trace files and feeds recommendation engine.
- `StudioDebuggerPanel`:
  - hydrates `session_.last_run` and `session_.training_trace` from disk on set/load.
  - renders live trace in `RenderLiveTrainingStatus` and `RenderTrainingTrace`.
  - trace settings UI maps directly to `TrainingTraceCollector::Configure`:
    - persist toggle, interval, max-event budget.

### 42.6.2 Plot panel telemetry visibility
- `TrainingPlotPanel::RenderTrainingWarningSummary` reads singleton snapshot and shows latest warnings.
- Terminal state is driven from `SetTrainingComplete(..., terminal_status, terminal_reason)`
  and displayed as final run status rows.
- In non-plotting-module mode, `RecordPanelEvent` writes to `CrashRunRecorder`
  (`MarkPanelEvent`) to preserve GUI-level diagnostics in the same heartbeat stream.

## 42.7 Lean guardrail assessment
- Positive:
  - Single canonical file contracts for crash+trace avoid duplicated ad-hoc formats.
  - Bounded ring buffers and immediate write throttles contain memory/IO growth.
  - Clear terminal-status model maps directly to UI and recommendations.
- Residual risk:
  - frequent event writes are synchronous and can compete with hot loops; this is mitigated by
    periodic interval writes, but crash-critical events still force writes.
  - multiple writers/readers coordinate via mutexes only (not file locks), so cross-process
    concurrency is implicitly unsupported.

## 42.8 Evidence anchors

| Contract | Evidence |
|---|---|
| Crash heartbeat contract, status derivation, WER scan | `cyxwiz-engine/src/core/crash_run_recorder.h`, `cyxwiz-engine/src/core/crash_run_recorder.cpp` |
| Training trace event model, buffering, persistence, runtime settings | `cyxwiz-engine/src/core/training_trace_collector.h`, `cyxwiz-engine/src/core/training_trace_collector.cpp` |
| Training lifecycle integration + trace/heartbeat instrumentation | `cyxwiz-engine/src/core/training_manager.cpp`, `cyxwiz-engine/src/core/training_executor.cpp` |
| Runtime UI load path and recommendation input contracts | `cyxwiz-engine/src/gui/main_window.cpp`, `cyxwiz-engine/src/gui/panels/studio_debugger_panel.h`, `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp` |
| Dashboard surface and terminal/warning rendering from traces | `cyxwiz-engine/src/gui/panels/training_plot_panel.h`, `cyxwiz-engine/src/gui/panels/training_plot_panel.cpp` |
