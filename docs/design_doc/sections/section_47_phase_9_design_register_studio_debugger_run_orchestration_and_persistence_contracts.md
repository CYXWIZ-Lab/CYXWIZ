# 47) Studio Debugger run orchestration and persistence contracts

## 47.1 Scope and boundary

This section documents the run orchestration path that creates, executes, and persists a Studio Debugger session.

Coverage:

- UI control path that starts a debugger run
- execution mode composition (full/local/smoke/runtime/preflight)
- asynchronous worker contract and UI state transitions
- session graph capture and trace seeding
- compile/preflight/smoke/local-debug/runtime behavior branches
- session persistence into `DebugRunStore` and run-history recall contract

## 47.2 Start contract: who owns the run request

The run is started in `StudioDebuggerPanel::RenderToolbar()` and delegated to callback provided by `MainWindow`.

- Panel-level controls:
  - selected mode from `StudioDebuggerRunMode` (`FullWorkflow`, `Preflight`, `LocalDebug`, `SmokeRun`, `RuntimeTrace`)
  - run button disabled only by internal `run_in_progress_` guard
  - on click, the callback produces a `std::function<StudioDebuggerSnapshot()>`
- `MainWindow` wires callback via `SetRunDebugCallback(...)`:
  - captures current graph nodes/links from node editor at click time
  - calls `BuildStudioDebuggerSessionFromSnapshot(session, mode, sample_index, nodes, links)`
  - returns the built snapshot to UI thread on completion

## 47.3 Async execution contract

`StudioDebuggerPanel` executes the callback with `AsyncTaskManager::RunAsync`:

- creates task name `"Studio Debugger Run"`
- sets `run_in_progress_ = true`
- stores `pending_task_id_`
- completion callback executes on UI thread via task manager queue processing

Completion policy:

- if async task reports failure (`success == false`):
  - `StudioDebuggerSnapshot failed` is created
  - `failed.success = false`
  - `failed.failure_summary = error message or generic fallback`
  - `SetSession(failed)` to refresh panel state
  - `run_status_message_ = failed.failure_summary`
- on success:
  - if task result exists: `SetSession(*result)` and clear status message
  - else sets a generic “completed without a result” message
- always on completion:
  - `run_in_progress_ = false`
  - `pending_task_id_ = 0`

Async task execution contract comes from `AsyncTaskManager::RunAsync`:

- executes a lambda `f(task)` on worker thread
- if lambda exits without cancellation/exception: task is marked completed
- if lambda throws: task is marked failed with exception text
- completion callback is queued to main thread via `ProcessCompletedCallbacks()`

## 47.4 UI state guard contract

`run_in_progress_` is also used to lock user actions:

- `Clear()` is no-op when a run is active
- run/status widgets are disabled during active execution
- run mode combo/lens remain editable, but result row actions become stale-safe by gating run button and disabling clear during execution

## 47.5 Build-session contract (`MainWindow::BuildStudioDebuggerSessionFromSnapshot`)

`BuildStudioDebuggerSession(session, mode, sample)` is the canonical builder; `BuildStudioDebuggerSessionFromSnapshot` accepts moved graph objects.

Initial state contract:

- resets incoming session
- computes `graph_hash` from `HashGraphStructure(nodes, links)`
- allocates `run_id = MakeDebuggerRunId(graph_hash)`
- maps requested mode into boolean feature gates:
  - `run_full`
  - `run_local_debug`
  - `run_smoke`
  - `run_runtime`

Mode-name contract:

```text
FullWorkflow -> "FullWorkflow"
Preflight    -> "Preflight"
LocalDebug   -> "LocalDebug"
SmokeRun     -> "SmokeRun"
RuntimeTrace -> "RuntimeTrace"
```

Graph snapshot seeding:

- calls `DebugSessionManager::StartSession(run_id, mode_name, graph_hash, nodes, links, sample_index)`
- copies:
  - `session.traces` seeded with compile graph snapshot trace
  - `session.studio_events` seeded with `"DebugSession.Start"`
- sets `sample_summary` as `"Studio Debugger mode: <mode> | sample <n>"`

Persistent-save closure contract:

- closure `save_session()` writes a `DebugRunStoreRecord` containing:
  - `run_id`, `timestamp`, `graph_hash`, `success`, `summary`, `issues`, `traces`, `studio_events`, `recommendations`
- saves via `DebugRunStore::Save(record)`
- refreshes `session.run_history = DebugRunStore::ListRecent(8)`

Mandatory first event:

- `StudioEventRecord { action="StudioDebugger.Run", status="started", message="Studio Debugger started mode: <mode>" }`

## 47.6 Mode branch contracts

Runtime-only branch (`RuntimeTrace`):

- load `CrashRunRecorder::LoadLastRun()` and `TrainingTraceCollector::LoadLastTrace()` when available
- build recommendations from current trace set + issues + smoke/training/callsite context
- force `session.success = true`
- set summary text indicating trace load only
- persist session and return immediately

Compile gate + preflight branch:

- compile with `GraphCompiler`
- always append compile trace:
  - `phase="Compile"`, `role=CompileArtifact`
  - status `passed` or `failed`
  - payload includes node/link counts and issue count
- on compile failure:
  - set `session.success = false`
- append `Compile` studio event with `status` `passed|failed`

Preflight validation branch:

- run `PreflightValidator::Validate`
- append preflight trace:
  - `phase="Preflight"`
  - `role=CompileArtifact` when ready, else `Warning`
  - `status="ready"` when pass, else `"blocked"`
- append corresponding `"Preflight"` studio event
- if requested mode is exactly `Preflight`, recommendations are built from current traces only and session returns there.

Smoke branch (`FullWorkflow` or `SmokeRun`):

- `TextPreprocessingTracer::TraceSample` adds preprocessing traces
- `SmokeRunExecutor::RunTextSmoke` may add smoke traces
- append `"TextPreprocessingTrace"` and `"SmokeRun"` studio events with status/result summary
- merge smoke issues into `session.issues` when present

Local-debug branch (`FullWorkflow` or `LocalDebug`):

- instantiate `DebugExecutor` and call `exe.Run()`
- map each `LayerTrace` to `DebugTraceRecord`:
  - `phase="Forward"`
  - `role=Activation`
  - status mapping:
    - (`has_nan` or `has_inf`) -> `"warning"`
    - else shape match -> `"ok"` or `"shape_mismatch"`
  - payload includes predicted/actual shape + shape match + nan/inf flags
- map each `GradNorm` to `DebugTraceRecord`:
  - `phase="Backward"`
  - `role=Gradient`
  - status mapping:
    - nan -> `"nan"`
    - zero -> `"zero"`
    - else `"ok"`
  - payload includes `l2_norm`, `is_nan`, `is_zero`
- append `"LocalDebug"` studio event with status from `session.debug_result.success`

Local-debug exception contract:

- on exception: `session.failure_summary = "Debug run threw: ..."`
- adds error issue with `TrainingExecutionFailed`
- appends `"LocalDebug"` event with status `"failed"`
- save and return false

Runtime enrichment contract:

- if `run_runtime` also active and debug has already run, attach crash heartbeat and training trace snapshots from latest collectors (best-effort loads)
- build recommendations against:
  - `session.traces`, `session.issues`, `session.smoke_result`, `session.last_run`, `session.training_trace`

Finalization:

- `save_session()` is called in all terminal paths (runtime-only, compile-fail, preflight mode, post-branch normal completion)
- returned boolean is `session.success`

## 47.7 Persistence and run-history recovery contract

Persistence is split by stage:

- each session builder path ends with `DebugRunStore::Save`
- `session.run_history` is a direct derived slice `ListRecent(8)` and is used by panel rendering (`RenderRunHistory`)
- run history row selection invokes `LoadStoredRun`, which replaces the active snapshot with persisted record contents

`DebugRunStore` semantics used by this flow:

- `Save` writes `${cwd}/.cyxwiz/debug_runs/studio/<run_id>/session.json`
- file payload stores:
  - issues, traces, events, recommendations arrays
  - derived counts: issue_count, trace_count, event_count, recommendation_count
- load/list path reads `session.json` and builds typed records

## 47.8 ASCII contract map

```text
Studio Debugger run command
  -> StudioDebuggerPanel toolbar click
     -> callback factory returns callable snapshot builder
        -> AsyncTaskManager::RunAsync("Studio Debugger Run", task)
           -> worker executes BuildStudioDebuggerSessionFromSnapshot
           -> completion callback on main thread
              -> SetSession(result) or set failed session
```

```text
Studio session build (mode matrix)
FullWorkflow = run_local_debug + run_smoke + run_runtime
Preflight only = run_full=false, run_local_debug=false, run_smoke=false, run_runtime=false
LocalDebug = run_local_debug=true only
SmokeRun = run_smoke=true
RuntimeTrace = run_runtime=true only
```

```text
Run persistence
save_session() -> DebugRunStore::Save -> ListRecent(8) -> session.run_history -> UI list
```

## 47.9 Evidence anchors

| Claim | Source |
|---|---|
| Panel run button and asynchronous dispatch to callback | `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:484-514`, `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:338-357`, `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:510-549`, `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:339-347` |
| Run/clear state guards | `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:338-347`, `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:481-514`, `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:537-548` |
| MainWindow callback wiring and graph capture | `cyxwiz-engine/src/gui/main_window.cpp:633-651`, `cyxwiz-engine/src/gui/main_window.h:236-244` |
| Run-id + mode flag decomposition + mode-name mapping + save_session lambda | `cyxwiz-engine/src/gui/main_window.cpp:3444-3499` |
| Runtime-only branch and early return contract | `cyxwiz-engine/src/gui/main_window.cpp:3500-3526` |
| Compile/preflight path and branch exits | `cyxwiz-engine/src/gui/main_window.cpp:3526-3620`, `cyxwiz-engine/src/gui/main_window.cpp:3620-3605`, `cyxwiz-engine/src/gui/main_window.cpp:3570-3606` |
| Smoke execution and trace append contract | `cyxwiz-engine/src/gui/main_window.cpp:3620-3663` |
| Local-debug execution mapping, layer/gradient trace synthesis, and failure fallback | `cyxwiz-engine/src/gui/main_window.cpp:3655-3744` |
| Recommendation build and persistence-finalization | `cyxwiz-engine/src/gui/main_window.cpp:3746-3764` |
| `DebugSessionManager::StartSession` snapshot initialization and graph snapshot trace | `cyxwiz-engine/src/core/debug_session_manager.cpp:34-68` |
| Studio session persistence schema and file path | `cyxwiz-engine/src/core/debug_run_store.cpp:14-19`, `cyxwiz-engine/src/core/debug_run_store.cpp:222-267`, `cyxwiz-engine/src/core/debug_run_store.cpp:306-326` |
| Async task lifecycle (RunAsync, queueing, callback invocation, cancellation/failure semantics) | `cyxwiz-engine/src/core/async_task_manager.h:124-170`, `cyxwiz-engine/src/core/async_task_manager.cpp:219-238`, `cyxwiz-engine/src/core/async_task_manager.cpp:326-420`, `cyxwiz-engine/src/core/async_task_manager.cpp:371-405` |
