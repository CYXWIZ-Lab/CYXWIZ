# 44) Studio Debugger runtime-trace recovery contracts

## 44.1 Objective

This section documents how Studio Debugger sessions consume persisted crash/trace artifacts (`current_run.json`, `current_training_trace.json`) and how that feeds the debugger, recommendations, and run-history comparison UI.

## 44.2 Session data contract in debugger layer

### 44.2.1 Snapshot model
- `StudioDebuggerSnapshot` carries both:
  - structured debug artifacts:
    - `traces`
    - `studio_events`
    - `issues`
    - `recommendations`
    - `run_history`
  - runtime recovery artifacts:
    - `CrashRunSummary last_run`
    - `TrainingTraceSummary training_trace`
- `StudioDebuggerPanel::SetSession()` hydrates:
  - `last_run` from `CrashRunRecorder::LoadLastRun()`
  - `training_trace` from `TrainingTraceCollector::LoadLastTrace()`
  - `run_history` from `DebugRunStore::ListRecent(8)`.

### 44.2.2 Run mode contract
`StudioDebuggerRunMode` currently defines:
- `FullWorkflow`
- `Preflight`
- `LocalDebug`
- `SmokeRun`
- `RuntimeTrace`

`RuntimeTrace` is the explicit mode that is intended for post-run recovery/replay and does not execute graph debug.

## 44.3 RuntimeTrace mode contract (BuildSession -> recommendation path)

### 44.3.1 What happens when `StudioDebuggerRunMode::RuntimeTrace` is selected
- `MainWindow::BuildStudioDebuggerSessionFromSnapshot` still generates a run id and starts a `DebugSession` snapshot.
- In the same call:
  - it loads `CrashRunRecorder::LoadLastRun()` into `session.last_run`
  - it loads `TrainingTraceCollector::LoadLastTrace()` into `session.training_trace`
  - it immediately builds recommendations from these two plus standard trace/issue/smoke inputs.
  - it marks session as successful and sets a summary message that states run was loaded from artifacts.
- It saves the session to debug run store and returns successfully without graph execution.

### 44.3.2 Additional runtime artifact loading after other modes
- In the normal full/local/smoke workflow, after local debug and optional smoke execution:
  - if run mode includes runtime scope, it again loads both `CrashRunSummary` and `TrainingTraceSummary`.
  - these are still used by recommendation engine even when not in `RuntimeTrace` mode.
- This means recommendations are always crash-aware for full debug sessions that reached runtime section.

## 44.4 UI contracts in `StudioDebuggerPanel`

### 44.4.1 Trace persistence settings surface
- `RenderTraceSettings()` is a panel-level editor with three writable controls:
  - `Persist training trace to disk` boolean
  - `Write every N events`
  - `Keep recent events`
- Settings lazy-load on first render from `TrainingTraceCollector::Instance().GetSettings()`.
- Input clamping is enforced before commit:
  - N >= 1
  - recent events >= 20
  - recent events <= 5000
- Changes call `TrainingTraceCollector::Instance().Configure(...)` immediately.
- Note: lower N favors crash evidence fidelity; higher N saves I/O.

### 44.4.2 Live training trace refresh and status rendering
- `RenderLiveTrainingStatus()` always calls `RefreshLiveTrainingTrace()` which re-loads persisted trace JSON via `LoadLastTrace()`.
- If available, it renders:
  - run id
  - status
  - epoch/batch/stage
  - loss/accuracy
  - latest warning (if present)

### 44.4.3 Last-run and training-trace views
- `RenderLastRun()`:
  - lazily loads `CrashRunRecorder::LoadLastRun()` into `session_.last_run` if empty.
  - renders dataset/backend/metrics/last event + WER details if attached.
  - if `windows_crash_available == false`, it surfaces explicit fallback hint.
- `RenderTrainingTrace()`:
  - refreshes live trace, classifies latest task/validation/checkpoint/terminal entries.
  - prints compact warnings and recent recent events (latest up to 8).
  - passes summary data to subrenderers for materialization/runtime/memory/layer timings.

### 44.4.4 Store-backed session history contract
- `RenderRunHistory()` uses `session_.run_history` as list source and allows selecting older run ids.
- `LoadStoredRun(run_id)` loads a stored session from `DebugRunStore::Load` and replaces active snapshot.
- `RenderRunComparison()` compares selected snapshot with current run id and highlights delta in errors/shapes/warnings/recommendations.
- `current_session_` is lazily loaded from debug store only when comparison is first rendered and current run id is known.

## 44.5 Debug run store contract (persisted snapshots)

- Store root is `./.cyxwiz/debug_runs/studio`.
- Each session is written to `<run_id>/session.json`.
- `DebugRunStoreRecord` contract:
  - `summary`: compact counts and graph/run metadata
  - vectors for `issues`, `traces`, `studio_events`, `recommendations`
- `Save()` stores both summary fields and full arrays using JSON serialization with indentation.
- `Load()` maps JSON back to full vectors through the same schema.
- `ListRecent(max_runs)`:
  - reads each `<run_id>/session.json`
  - parses summary only
  - sorts by timestamp descending
  - truncates to `max_runs`.

## 44.6 End-to-end recovery flow (ASCII)

```text
RuntimeTrace request
  -> BuildStudioDebuggerSessionFromSnapshot(mode=RuntimeTrace)
     -> DebugSession start (graph snapshot trace + event marker)
     -> LoadLastRun() + LoadLastTrace()
     -> DebugRecommendationEngine.Build(traces, issues, smoke, last_run, training_trace)
     -> save_session
     -> panel Session has last_run + training_trace + recommendations

```

```text
User opens Studio Debugger panel
  -> SetSession(snapshot)
     -> hydrate session.last_run / session.training_trace
     -> run_history <= ListRecent(8)
  -> Render:
     Trace Settings (runtime control)
     Run History + load old run
     Crash/Last Run view
     Training Trace view
     Recommendations and comparison
```

## 44.7 Evidence anchors

| Claim | Source |
|---|---|
| Runtime modes and runtime-trace shortcut | `cyxwiz-engine/src/gui/panels/studio_debugger_panel.h:35-45` |
| Session hydration in `SetSession` and persisted artifact loading | `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:324-349` |
| Debugger `RenderTraceSettings` clamp + configure contract | `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:698-741` |
| Live refresh + last run/training trace rendering contracts | `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:844-861`, `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:1183-1234`, `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:1256-1323` |
| RuntimeTrace branch loads crash and trace artifacts without execution | `cyxwiz-engine/src/gui/main_window.cpp:3456-3517` |
| Non-runtime path also enriches with last run + trace when runtime flag is active | `cyxwiz-engine/src/gui/main_window.cpp:3751-3758` |
| Debug store path + save/load/list schema | `cyxwiz-engine/src/core/debug_run_store.h:13-38`, `cyxwiz-engine/src/core/debug_run_store.cpp:14-24`, `cyxwiz-engine/src/core/debug_run_store.cpp:222-238`, `cyxwiz-engine/src/core/debug_run_store.cpp:275-314`, `cyxwiz-engine/src/core/debug_run_store.cpp:295-318` |
| Recommendation engine consumes both last run and training trace | `cyxwiz-engine/src/core/debug_recommendation_engine.h:12-22`, `cyxwiz-engine/src/core/debug_recommendation_engine.cpp:33-84`, `cyxwiz-engine/src/core/debug_recommendation_engine.cpp:96-123` |

