# 37) Phase 9 design register (async control-plane, task lifecycle, and UI visibility contracts)

## 37.1 Objective
Document the runtime task control plane used by training and editor workflows:
- task creation and dispatch,
- execution and cancellation semantics,
- callback-threading policy,
- and task visibility in background-task UI.

## 37.2 Why this layer exists
The engine uses a dedicated async task fabric for operations that can block the UI:
- data loading and import tasks,
- training launch/materialization prep,
- training runtime progress loop,
- project/build side operations,
- diagnostics export and codegen jobs.

That isolates long-running work while preserving a single progress/terminal event vocabulary.

## 37.3 Control-plane data model

```text
Caller -> AsyncTaskManager
  -> RunAsync / Submit
     -> AsyncTask (id, name, state, progress, messages, callbacks)
        -> WorkerThread executes task lambda
           -> ReportProgress / MarkCompleted / MarkFailed
           -> optional completion callback queued for main-thread processing
  -> UI (TaskProgressPanel + status bar) pulls task snapshots
```

### 37.3.1 Object contracts
- `AsyncTask`
  - immutable identity (`id_`, `name_`) and atomic state:
    `Pending -> Running -> (Completed | Failed | Cancelled)`
  - cancellable by design-time flag (`cancellable_`).
  - callbacks:
    - `ProgressCallback`: `(float progress, const std::string& message)`
    - `CompletionCallback`: `(bool success, const std::string& error)`
  - mutable message/error fields are mutex-protected.
- `AsyncTaskManager`
  - singleton with worker pool (`workers_`) and priority queue.
  - stores tasks in `active_tasks_` + `completed_tasks_` history.
  - processes completion callbacks on main-thread via internal queue (`pending_callbacks_`).
- `TaskInfo` is the read model consumed by UI:
  - `id`, `name`, `state`, `progress`, `status_message`, `error_message`,
    `start_time`, `end_time`, `cancellable`.

## 37.4 Contracts and state transitions

### 37.4.1 Submission contract
- `Submit(task, priority)`:
  - task is inserted into active map,
  - task is pushed into queue,
  - worker wakes and executes asynchronously.
- `RunAsync(name, func, progress_cb, completion_cb)`:
  - wraps lambda in `LambdaTask`,
  - auto-calls `MarkCompleted()` when function returns without exception,
  - auto-calls `MarkFailed(e.what())` on `std::exception`.

### 37.4.2 Worker execution contract
- Before execute:
  - state transitions to `Running`,
  - `start_time_` is set,
  - runtime emits `TaskStarted`.
- During execute:
  - `ReportProgress` clamps `[0.0..1.0]`, stores message,
  - always emits `TaskProgress`.
- Completion branch:
  - `MarkCompleted` -> `Completed`, progress=1.0.
  - `MarkFailed` -> `Failed`, status text prefixed `Failed:`.
  - if function never called `MarkCompleted` and no cancel request, `RunAsync` wrapper marks completed.
  - if `ShouldStop()` remains true after execution and state still `Running`, state is set to `Cancelled`.
- After execute:
  - active map erase, completed ring append,
  - completed callback marshaled to main-thread queue.

### 37.4.3 Cancellation contract
- `Cancel(task_id)`:
  - applies `RequestCancel()` when task is still active (including queued/just-started),
  - `RequestCancel` only sets a flag and emits `TaskCancelRequested`.
- Cancellable tasks:
  - `RequestCancel` is ignored when `cancellable_==false`.
- Cancellation does **not** force immediate teardown from queue:
  - worker executes function and then emits `Cancelled` if flag observed post-exec.

## 37.5 Threading and callback policy
- `ReportProgress` can invoke callback on worker thread.
- `CompletionCallback` is deferred and executed by
  `AsyncTaskManager::ProcessCompletedCallbacks()`.
- Main window calls this method each frame in `Render()`, ensuring completion side effects
  run on main thread.
- `TaskProgressPanel` reads snapshots via `GetRecentTasks` / `GetActiveTasks`;
  no direct cross-thread UI mutation from worker threads.

## 37.6 Contract examples in engine flows

### 37.6.1 Training control-plane flow
- `TrainingManager::StartTrainingCommon` submits task named `"Training Model"` and stores `current_task_id_`.
- task lambda:
  - polls `TrainingManager::IsTrainingActive()`,
  - emits progress as `"Epoch X/Y - Loss: ..."` every 100ms.
- `TrainingManager::StopTraining()` performs:
  - executor stop (`current_executor_->Stop()`),
  - `AsyncTaskManager::Cancel(current_task_id_)`.
- task completion:
  - `training_task_` marks `Cancelled` on stop path if the runner reports cancellation.

### 37.6.2 Graph materialization launch flow
- `GraphTrainingLauncher::StartGraphTrainingFromCompiledConfig` creates a `LambdaTask`
  named `"Prepare graph training"`.
- task uses task progress callback style and `RecordTaskProgress`:
  - `ShouldStop()` checkpoints gate preparation phases.
- on failure:
  - task completion callback flips the panel to `SetPreparationFailed`.

## 37.7 UI visibility contract

### 37.7.1 Background tasks panel
- `TaskProgressPanel`:
  - Active tasks section displays `Pending/Running`;
  - optional completed section displays `Completed/Failed/Cancelled`;
  - each running/pending task supports a per-task cancel button.

### 37.7.2 Status bar contract
- `MainWindow::RenderStatusBar`:
  - if active count > 0, show count + clickable `TaskStatusIndicator`,
  - else show ready state.
- `TaskStatusIndicator` tooltip enumerates active task names and percentages.
- clicking indicator opens Tasks panel.

## 37.8 Failure and determinism contract
- Task failure text is preserved in `TaskInfo.error_message` and surfaced by panel.
- Completion state is canonical for status reporting; progress polling does not define
  terminal state.
- For a task cancelled by user:
  - user-visible terminal state is `Cancelled`,
  - execution can still emit `cancelled` terminal `RecordTaskProgress` stage in trace collector.

## 37.9 Evidence anchors

| Layer | Evidence |
|---|---|
| Core async contract (types/state/callbacks) | `cyxwiz-engine/src/core/async_task_manager.h` |
| Core async behavior (run loop, cancel, completion queue) | `cyxwiz-engine/src/core/async_task_manager.cpp` |
| Training control-plane usage | `cyxwiz-engine/src/core/training_manager.cpp` and `cyxwiz-engine/src/core/training_manager.h` |
| Materialization async preparation | `cyxwiz-engine/src/gui/graph_training_launcher.cpp` |
| UI task rendering | `cyxwiz-engine/src/gui/panels/task_progress_panel.h` and `cyxwiz-engine/src/gui/panels/task_progress_panel.cpp` |
| Main-loop callback pumping + status bar visibility | `cyxwiz-engine/src/gui/main_window.cpp` |

## 37.10 Lean contract review
- Essentials kept:
  - single task core, strong `TaskInfo` read model,
  - main-thread completion callback drain,
  - one visible status channel (`TaskProgressPanel` + status bar indicator).
- Complexity risk:
  - completed callback queue + active/completed maps is a second shared-state path; currently contained in manager and necessary for UI-safety.
- Simplification boundary:
  - avoid spreading ad-hoc background threading elsewhere; route new UI-visible background work through `AsyncTaskManager::RunAsync` for observability consistency.
