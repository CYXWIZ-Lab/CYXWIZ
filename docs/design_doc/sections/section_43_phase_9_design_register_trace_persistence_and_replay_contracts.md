# 43) Trace persistence and replay contracts

## 43.1 Scope and intent

This section captures the lowest-level design contract for how CyxWiz persists runtime diagnostics and crash/heartbeat state for training recovery and post-mortem visibility. It defines:

- where trace artifacts are stored,
- what gets written,
- how often persistence is triggered,
- what fields are guaranteed when loading artifacts later,
- and what qualifies as “suspected crash” from the persisted snapshot.

## 43.2 Contract surface (two persistence lanes)

1) Crash/heartbeat lane (`CrashRunRecorder`)
- persists a running run summary into `.cyxwiz/debug_runs/current_run.json`
- is driven by training control events (start, stage, panel/backend action, completion/failure/stop)
- provides “last known state + action log” for immediate recovery diagnostics

2) Metric/telemetry lane (`TrainingTraceCollector`)
- persists the rich event stream into `.cyxwiz/debug_runs/current_training_trace.json`
- records metrics, timing, memory usage, checkpoints, task progress, and terminal events
- is used by dashboard and trace UI views for live/replay continuity

## 43.3 Crash heartbeat persistence contract (state recovery)

### 43.3.1 File location and record identity
- Debug directory: `.cyxwiz/debug_runs`
- Crash lane file: `current_run.json`
- Trace lane file: `current_training_trace.json`
- both are JSON documents written with `std::ofstream(..., ios::trunc)` so each write replaces prior content.
- run identity is generated from epoch milliseconds as `"train-" + millis`.

### 43.3.2 State fields and state transitions
- On `StartTrainingRun`, recorder initializes:
  - `status = "running"`,
  - `run_id`,
  - dataset/backend/domain/context,
  - epoch/batch/sample counters,
  - empty failure/terminal reason,
  - empty panel event log.
- Transition methods (`MarkCompleted`, `MarkEarlyStopped`, `MarkCancelled`, `MarkFailed`) set terminal `status` and terminal reason, update `last_stage`, then write.
- `MarkFailed` also stores `failure_reason` and copies it into `terminal_reason`.
- All writes are serialized under mutex and currently executed synchronously.

### 43.3.3 Suspected crash semantics
- `LoadLastRun()` loads `current_run.json` if present.
- If loaded `status == "running"`, loader marks:
  - `suspected_crash = true`,
  - `status = "suspected crash"`,
  - adds a warning that indicates no clean completion marker was seen.
- On Windows, loader attempts to attach WER report metadata (`Fault Module`, `Exception Code`, `EventTime`, report id/path) when available and runtime crash suspicion exists.

## 43.4 Trace persistence contract (telemetry stream persistence)

### 43.4.1 In-memory buffering guarantees
- The collector keeps deques for:
  - `events_` (general latest events),
  - `materialization_events_` (node/task materialization subset),
  - warning strings.
- defaults are bounded:
  - `persist_every_n_events = 10`,
  - `max_recent_events = 200`.
- `Configure()` enforces minimum invariants:
  - `persist_every_n_events >= 1`,
  - `max_recent_events >= 20`.

### 43.4.2 Flush behavior
- Every record path writes into memory then:
  - increments `events_since_write_`,
  - writes immediately when status transitions are non-ok / non-running (warnings/errors),
  - or writes every `max(1, persist_every_n_events)` events.
- `RecordRuntimeEvent`, `RecordTaskProgress`, `RecordValidationMetrics`, `RecordCheckpointSaved`, and `RecordTerminalEvent` force immediate writes.
- All persistence write/read paths are guarded with `try/catch` and intentionally do not abort training/monitoring.

### 43.4.3 Event schema materialized to JSON
- `TrainingTraceEvent` serializes explicit telemetry/state fields including:
  - timestamps/thread/run IDs,
  - stage and metric scope,
  - epoch/batch totals and losses/accuracies,
  - task fields (`task_id`, `task_name`, `task_stage`, `task_progress`),
  - node fields (`node_id`, `node_name`),
  - checkpoint fields (`checkpoint_path`, `is_best_checkpoint`),
  - terminal fields (`terminal_reason`, `status`),
  - memory envelope (`cpu_allocated_bytes`, `cpu_peak_bytes`, AF memory + buffer counters),
  - duration fields.
- `TrainingTraceSummary` reads top-level keys:
  - `run_id`, `status`, `events`, `materialization_events`, `warnings`.
- `EventFromJson` provides defaulted deserialization, tolerating missing keys.
- On load, if `materialization_events` is missing, consumer can fall back to task-related markers in events (node id + task/ID signals).

## 43.5 Snapshot and replay behavior

- `Snapshot()` returns:
  - current run id + status,
  - latest event aggregate for dashboards,
  - full recent and materialization event deques,
  - warning list.
- `LoadLastTrace()`:
  - returns empty optional when file absent or parse fails,
  - hydrates warnings + both event lanes,
  - computes latest metrics from final event when available.
- On read failure, system returns optional empty without throwing to keep caller behavior stable.

## 43.6 Design diagrams (ASCII)

```text
CrashLane: Training start/stop

TrainingExecutor -> CrashRunRecorder::StartTrainingRun
    -> initialize run snapshot (status=running, ids, metadata)
    -> WriteLocked() -> current_run.json
Training loop stage callbacks -> MarkStage / MarkPanelEvent / MarkBackendEvent
    -> WriteLocked() for each callback
Terminal -> MarkCompleted | MarkFailed | MarkCancelled | MarkEarlyStopped
    -> final WriteLocked() + deactivate recorder
Next launch -> LoadLastRun()
    if status == "running" => suspected_crash true
```

```text
TraceLane: Runtime telemetry stream

TrainingExecutor/Manager -> TrainingTraceCollector::StartRun
    -> clear deques and write initial current_training_trace.json
Events:
    RecordStage / RecordTaskProgress / RecordValidationMetrics / RecordCheckpointSaved / RecordTerminalEvent
    -> append in-memory event, trim to max_recent_events
    -> flush on interval or warning/terminal event
Debug/load path -> Snapshot() for UI | LoadLastTrace() for restore/reopen
```

## 43.7 Evidence anchors

| Claim | Source |
|---|---|
| Two JSON artifacts under `.cyxwiz/debug_runs` with fixed names | `cyxwiz-engine/src/core/crash_run_recorder.cpp:27`, `cyxwiz-engine/src/core/training_trace_collector.cpp:20-24` |
| Run status and suspected-crash recovery rule | `cyxwiz-engine/src/core/crash_run_recorder.cpp:339-378`, `cyxwiz-engine/src/core/crash_run_recorder.h:31` |
| Event stream schema and defaulted fields | `cyxwiz-engine/src/core/training_trace_collector.h:53-84`, `cyxwiz-engine/src/core/training_trace_collector.cpp:49-92` |
| Persistence flush policy and bounds | `cyxwiz-engine/src/core/training_trace_collector.h:67-68`, `cyxwiz-engine/src/core/training_trace_collector.cpp:192-198`, `cyxwiz-engine/src/core/training_trace_collector.cpp:425-439` |
| Non-throwing persistence writes/reads | `cyxwiz-engine/src/core/training_trace_collector.cpp:525-546`, `cyxwiz-engine/src/core/crash_run_recorder.cpp:389-420` |
| Panel event retention cap and truncation | `cyxwiz-engine/src/core/crash_run_recorder.cpp:250-255`, `cyxwiz-engine/src/core/crash_run_recorder.cpp:273-278` |

