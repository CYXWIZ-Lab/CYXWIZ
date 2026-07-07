# 52) Training lifecycle and callback-order contracts (control-plane, pause/stop, and plugin hooks)

## 52.1 Scope and boundary

This section records the full runtime control contract for training execution from
the manager API to the executor loop, including:

- thread ownership and state transitions,
- callback surfaces (`TrainingManager`-driven callbacks and plugin hooks),
- pause/stop semantics,
- mode-specific execution dispatch,
- completion/terminal outcomes,
- and callback-order invariants used by host components and observability.

The contract is scoped to:

- `TrainingManager` start/stop/pause control-plane,
- `TrainingExecutor` lifecycle and loop boundaries,
- `PluginTrainingHookManager` + `ITrainingHook` integration,
- and the UI/dashboard update path driven by batch and epoch callbacks.

## 52.2 Lifecycle machine (manager + executor)

The effective state is composed from two layers.

**Manager layer (`TrainingManager`)**

- `StartTraining*` entry methods are serialized with `is_training_` + `mutex_`.
- On start, state is set to `is_training_=true`, `stop_requested_=false`.
- `current_executor_` is moved into the training worker thread and published.
- `current_task_id_` is registered with `AsyncTaskManager`.
- On stop, both manager flag and executor stop are triggered.

**Executor layer (`TrainingExecutor`)**

- `is_training_`, `stop_requested_`, and `is_paused_` are atomic state bits.
- Execution starts only when `Train()` is entered and `is_training_` becomes true.
- `Stop()` sets `stop_requested_=true` and clears `is_paused_` so blocked loops can exit.
- `Pause()` and `Resume()` only manipulate `is_paused_` + metrics messages.
- `~TrainingExecutor()` calls `Stop()` (RAII safety if object lifetime ends while active).

```text
UI/GUI or automation
  -> TrainingManager::StartTraining*
     -> StartTrainingCommon
        -> is_training_=true, stop_requested_=false
        -> task registration + worker thread
           -> TrainingThreadFunc
              -> Executor->Train()
                 -> looped execution until epoch end / stop / failure
                    -> terminal metrics + callbacks
                 -> is_training_ => false

Stop path
  -> TrainingManager::StopTraining
     -> is_training_ untouched
     -> stop_requested_=true
     -> current_executor_->Stop()
        -> stop_requested_=true
        -> is_paused_=false (break pause waits)

Pause path
  -> TrainingManager::PauseTraining
     -> current_executor_->Pause()
        -> is_paused_=true
     -> loops stall at WaitWhilePaused()

Resume path
  -> TrainingManager::ResumeTraining
     -> current_executor_->Resume()
        -> is_paused_=false
     -> loops continue
```

## 52.3 Training executor mode lattice contract

`TrainingExecutor` is created with an explicit dataset-mode constructor and uses
that mode consistently in `Train()` dispatch.

- `DatasetMode::Legacy` from legacy `DatasetHandle` constructor.
- `DatasetMode::Arrow` from `ArrowDataset` constructor.
- `DatasetMode::Parquet` from `ParquetBackedDataset` constructor.
- `DatasetMode::External` from external `IBatcher` constructor (Image/Audio/Text).
- `DatasetMode::SequenceExternal` from `ISequenceBatcher` + label vector constructor.

The mode determines batcher selection (`Legacy`, `Arrow`, `Parquet`, `External`,
`SequenceExternal`) and controls validation/train loop routing.

## 52.4 Callback order contract

### 52.4.1 Plugin hook order (declared contract, enforced subset)

`ITrainingHook` declares `OnTrainingStart`, `OnTrainingEnd`, `OnEpochStart`,
`OnEpochEnd`, `OnBatchStart`, `OnBatchEnd`, and `ShouldStopEarly`.

The executor currently drives this effective order:

1. Before loop:
   - `NotifyTrainingStart`
2. For each epoch:
   - `ShouldStopEarly`
   - `NotifyEpochStart`
   - run epoch through dataset mode path
   - optional validation flow
   - `NotifyEpochEnd`
3. After loop:
   - `NotifyTrainingEnd`

There is currently no executor call site for:

- `NotifyBatchStart`
- `NotifyBatchEnd`

That means plugin batch-level hook surfaces are declared but not executed in current
training loops.

### 52.4.2 Host/dashboard callback order

`TrainingManager::TrainingThreadFunc` binds two host callbacks:

- per-epoch callback: updates cached metrics, plot panel, and history.
- per-batch callback: updates live batch progress and smooth x-axis curve points.

Executor receives both from manager and executes only `batch_cb` and `epoch_cb`.

```text
Plugin hooks (inside TrainingExecutor)
  NotifyTrainingStart
  [for each epoch]
    ShouldStopEarly
    NotifyEpochStart
    RunTrainingEpoch* -> batch processing
    [optional validation]
    NotifyEpochEnd
  NotifyTrainingEnd

Host callbacks
  batch_cb (if provided) -> every batch, after optimizer/metrics update
  epoch_cb (if provided) -> after epoch metrics are finalized
```

## 52.5 Pause/stop safety and race contracts

### 52.5.1 Stop contract

- Stop is asynchronous and cooperative:
  - manager sets stop flag and calls executor stop;
  - loops check `ShouldStop()` at epoch boundaries and inside batch loops;
  - `Stop()` also clears pause so any pause wait exits quickly.

### 52.5.2 Pause contract

- Pause does not set stop.
- When paused, loops call `WaitWhilePaused()` and sleep in 100 ms steps.
- Resume clears `is_paused_`; loops continue from the next check point.

### 52.5.3 Completion contract

At loop exit and before final callbacks, executor computes terminal status:

- `completed` when loops end without explicit stop,
- `cancelled` when `stop_requested_` is set,
- `early_stopped` when plugin-early-stop or validation patience triggers.

`is_training_` is set false in normal completion and also in exception paths.

## 52.6 Loop-level behavior contract (batching mode aware)

- Legacy mode: `RunTrainingEpoch(DatasetBatcher&)`.
- Arrow, Parquet, Image/Audio/Text: `RunTrainingEpochArrow(IBatcher&)`.
- Sequence text model mode: `RunTrainingEpochSequence(ISequenceBatcher&)`.

Each batch loop shares the same guard order:

1. `ShouldStop()`
2. `WaitWhilePaused()`
3. fetch next batch
4. forward/backward/update metrics and optionally callback

Validation follows similar guarding (`ShouldStop`), with phase flips for
sequence and modern IBatcher paths so validation uses explicit `Val` phase.

## 52.7 Termination + traceability side effects

- On exit, executor marks terminal status/reason and updates `TrainingMetrics`:
  `is_training=false`, `is_complete=true`, terminal metadata.
- If checkpoints are enabled and a best checkpoint exists, metrics are restored
  after run end when not user-cancelled.
- `TrainingTraceCollector` and `CrashRunRecorder` finalization occur per terminal
  state (`cancelled`, `early_stopped`, `completed`).
- `AsyncTaskManager` task transitions to running/completed based on final metrics
  snapshot from `GetCurrentMetrics()`.

## 52.8 Open contract gap (non-compliance inventory)

This surface has one explicit mismatch:

- plugin `ITrainingHook::OnBatchStart/OnBatchEnd` are declared and registered
  but never invoked from `TrainingExecutor`.

This is an active behavior gap: either the contract should be narrowed to the
executed hook points, or execution should call those hooks at stable batch-loop
points with clear error-order guarantees.

## 52.9 Evidence anchors

| Claim family | Source |
|---|---|
| TrainingManager single-flight start policy + async task bootstrap | `cyxwiz-engine/src/core/training_manager.h:19-37`, `cyxwiz-engine/src/core/training_manager.h:49-69`, `cyxwiz-engine/src/core/training_manager.cpp:54-70`, `cyxwiz-engine/src/core/training_manager.cpp:80-107`, `cyxwiz-engine/src/core/training_manager.cpp:106-163` |
| StartTraining* constructors and mode-specific entry points | `cyxwiz-engine/src/core/training_manager.h:59-123`, `cyxwiz-engine/src/core/training_manager.cpp:166-560`, `cyxwiz-engine/src/core/training_manager.cpp:596-614` |
| Executor mode contract in constructors | `cyxwiz-engine/src/core/training_executor.h:218-229`, `cyxwiz-engine/src/core/training_executor.cpp:94-162` |
| Mode dispatch and run-loop entry | `cyxwiz-engine/src/core/training_executor.cpp:288-333`, `cyxwiz-engine/src/core/training_executor.cpp:415-433` |
| `Train()` callback ordering + pause/stop checkpoints | `cyxwiz-engine/src/core/training_executor.cpp:217-268`, `cyxwiz-engine/src/core/training_executor.cpp:537-544`, `cyxwiz-engine/src/core/training_executor.cpp:702-713`, `cyxwiz-engine/src/core/training_executor.cpp:832-842` |
| Pause/stop/wait primitives | `cyxwiz-engine/src/core/training_executor.cpp:1311-1346` |
| Batcher loop stop/pause guard for all modern paths | `cyxwiz-engine/src/core/training_executor.cpp:912-917`, `cyxwiz-engine/src/core/training_executor.cpp:1446-1448`, `cyxwiz-engine/src/core/training_executor.cpp:1716-1718` |
| Host batch/epoch callback wiring and completion state merge | `cyxwiz-engine/src/core/training_manager.cpp:624-797`, `cyxwiz-engine/src/core/training_manager.cpp:800-863` |
| Manager stop/pause/ resume/ join operations | `cyxwiz-engine/src/core/training_manager.cpp:562-613`, `cyxwiz-engine/src/core/training_manager.cpp:585-599` |
| Plugin hook interface contract | `cyxwiz-engine/src/plugin/interfaces/i_training_hook.h:8-35` |
| Hook registration/removal + callback invocation methods | `cyxwiz-engine/src/plugin/registries/plugin_training_hook_manager.h:18-30`, `cyxwiz-engine/src/plugin/registries/plugin_training_hook_manager.cpp:12-70`, `cyxwiz-engine/src/plugin/plugin_manager.cpp:313-330`, `cyxwiz-engine/src/plugin/plugin_manager.cpp:371-376` |
| Plugin state model used around load/init/shutdown lifecycle | `cyxwiz-engine/src/plugin/plugin_types.h:144-163`, `cyxwiz-engine/src/plugin/plugin_manager.cpp:133-136`, `cyxwiz-engine/src/plugin/plugin_manager.cpp:228-233`, `cyxwiz-engine/src/plugin/plugin_manager.cpp:371-381` |
