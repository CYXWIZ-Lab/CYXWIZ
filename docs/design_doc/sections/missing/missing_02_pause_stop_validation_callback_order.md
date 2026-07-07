# M-02) Training pause/stop/validation callback ordering and determinism boundary

## M-02.1 Question
What is the exact callback and control-order contract when pause, stop, and validation gate transitions are combined?

## M-02.2 Observed engine contract

`TrainingManager` owns thread/task control and passes host callbacks into `TrainingExecutor`; `TrainingExecutor` owns plugin epoch-level callbacks and loop guards.

```text
TrainingManager::StartTraining*
  -> StartTrainingCommon
     -> build executor
     -> worker thread
        -> TrainingThreadFunc
           -> Train(executor)
              -> Set is_training_=true
              -> epoch loop:
                 -> ShouldStop()?  (manager + plugin + user stop state)
                 -> NotifyEpochStart
                 -> WaitWhilePaused() checkpoints in batch path
                 -> batch loop:
                    -> ShouldStop()? (batch-level checkpoint)
                    -> WaitWhilePaused() (pause gate)
                    -> forward/backward/update + batch callback
                 -> optional validation
                 -> NotifyEpochEnd
              -> completion branch:
                 -> is_training_=false
                 -> cache completion status
                 -> manager callback(false)
                 -> on_training_end_
```

## M-02.3 Determinism caveat in current code
- `ShouldStop` is polled at epoch and batch boundaries and before validation paths.
- `WaitWhilePaused` runs while paused and is broken by `Stop()` via `is_paused_=false`.
- Host callbacks are provided from manager (`batch_cb`, `epoch_cb`) and invoked from executor loop.
- Plugin hooks are only guaranteed at `TrainingStart/StartEpoch/EndEpoch/TrainingEnd` and `ShouldStopEarly`.
- `OnBatchStart`/`OnBatchEnd` are not invoked in the current batch loop, so batch-level plugin callback determinism is currently a known gap.

## M-02.4 Risk and required follow-up
- Missing deterministic trace in tests for **nested pause/unpause and stop** around validation-phase transitions.
- Missing hook-level determinism contract at true batch boundaries due absent `OnBatchStart/OnBatchEnd`.

## M-02.5 Suggested contract for next pass
- Add a fixed callback-sequence marker stream in tests covering:
  - pause in `train` phase,
  - pause entering validation phase,
  - stop during batch loop,
  - stop during validation loop,
  - stop during `Resume` wake-up race points.

## M-02.6 Evidence anchors
- `cyxwiz-engine/src/core/training_manager.cpp:54` (`StartTrainingCommon` wires thread + callback contract)
- `cyxwiz-engine/src/core/training_manager.cpp:157` (Training thread dispatch into executor)
- `cyxwiz-engine/src/core/training_manager.cpp:109` (thread loop and stop polling)
- `cyxwiz-engine/src/core/training_executor.cpp:447` (training loop + plugin hook sequencing)
- `cyxwiz-engine/src/core/training_executor.cpp:508` (epoch loop and stop gate)
- `cyxwiz-engine/src/core/training_executor.cpp:517` (plugin early-stop and stop gate)
- `cyxwiz-engine/src/core/training_executor.cpp:528` (WaitWhilePaused at batch boundary)
- `cyxwiz-engine/src/core/training_executor.cpp:563` (batch loop dispatch point)
- `cyxwiz-engine/src/core/training_executor.cpp:913` (per-batch stop + pause check)
- `cyxwiz-engine/src/core/training_executor.cpp:949` (validation loop stop check)
- `cyxwiz-engine/src/core/training_executor.cpp:1098` (RunValidation stop check)
- `cyxwiz-engine/src/core/training_executor.cpp:1342` (`WaitWhilePaused` implementation and resume behavior)
- `cyxwiz-engine/src/plugin/registries/plugin_training_hook_manager.cpp:45` (NotifyEpochStart)
- `cyxwiz-engine/src/plugin/registries/plugin_training_hook_manager.cpp:47` (NotifyEpochEnd)
- `cyxwiz-engine/src/plugin/registries/plugin_training_hook_manager.cpp:49` (NotifyBatchStart)
- `cyxwiz-engine/src/plugin/registries/plugin_training_hook_manager.cpp:50` (NotifyBatchEnd)
- `cyxwiz-engine/src/plugin/registries/plugin_training_hook_manager.cpp:54` (`ShouldStopEarly` callback contract)
- `cyxwiz-engine/src/plugin/interfaces/i_training_hook.h:25` (plugin hook contract surface)
