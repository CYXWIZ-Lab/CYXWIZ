# 36) Phase 9 design register (training dashboard telemetry, UI contracts, and Python bridge)

## 36.1 Objective
Formalize the full runtime contract for the Training Dashboard as the user-visible control plane for training behavior, from graph launch to execution completion.
This section captures:
- data pipeline ownership,
- thread-safety and update ordering,
- failure/terminal visibility contracts,
- and Python/C++ integration boundaries.

## 36.2 End-to-end telemetry topology

```text
MainWindow
  -> StartTrainingFromCompiledConfig
     -> GraphTrainingLauncher::StartGraphTrainingFromCompiledConfig
        -> async Prepare task
           -> PipelineMaterializer::Materialize
              -> TrainingTraceCollector::RecordTaskProgress
              -> TrainingPlotPanel::SetPreparationState / RecordMaterializationProgress
        -> dispatch callback
           -> TrainingManager::StartTraining*
              -> StartTrainingCommon
                 -> Async task registration (Tasks panel)
                 -> panel bootstrap (Clear + SetTrainingState)
                 -> start TrainingThread
                    -> TrainingThreadFunc
                       -> Train() epoch_callback
                          -> TrainingTraceCollector::MarkStage(TrainingTraceStage::EpochComplete)
                          -> TrainingPlotPanel::AddLossPoint / AddAccuracyPoint / SetTrainingState
                       -> batch_callback
                          -> TrainingPlotPanel::SetBatchProgress
                          -> TrainingPlotPanel::AddLossPoint(AddAccuracyPoint) with frac_epoch
                    -> AddRunComparisonRecord + SetTrainingComplete
                       (always in thread tail path)
     -> panel remains authoritative sink for live UI state and run summary
```

## 36.3 Public panel contract in C++

`TrainingPlotPanel` is the canonical sink for user-visible telemetry and is called from non-UI threads.

### 36.3.1 Thread-safety boundary
- Methods documented under "Training state updates (thread-safe)" in interface docs are lock-protected using an internal mutex.
- `Render()` snapshots all state under lock while rendering summary cards and charts.
- All background calls use `std::weak_ptr<TrainingPlotPanel>` and must lock before use.

### 36.3.2 Required caller contract
| Caller | Method(s) | Contract |
|---|---|---|
| GraphTrainingLauncher | `Clear`, `SetPreparationState`, `SetPreparationFailed`, `SetMaterializationComplete`, `RecordMaterializationProgress` | The launcher owns "pre-training" UX status; it owns progress percentages from dataset/materialization stages and transitions to false/failed state explicitly. |
| TrainingManager::StartTrainingCommon | `Clear`, `ShowCustomMetrics`, `SetTrainingState`, `SetVisible` | Reset panel and set training-mode state before worker thread starts. |
| TrainingThread (epoch path) | `AddLossPoint`, `AddAccuracyPoint`, `AddCustomMetric`, `SetTrainingState`, `SetTrainingComplete` | Epoch-level metrics are treated as stable checkpoints for official epoch sampling; state updates include timing and throughput. |
| TrainingThread (batch path) | `SetBatchProgress`, `AddLossPoint`, `AddAccuracyPoint` | Enables intra-epoch live UI; intentionally writes fractional epoch points for smooth curves. |
| Run completion path | `AddRunComparisonRecord`, `SetTrainingComplete` | Produces terminal summary row for training run comparison/export path. |

### 36.3.3 Observable state fields and state machine (panel side)
- `is_training_` and `is_preparing_` are mutually asserted by updates.
- `preparation_failed_` blocks overwrite while active training is running.
- `terminal_status_`/`terminal_reason_` are cleared on start, then set at completion.
- `current_epoch_`, `total_epochs_`, `current_batch_`, and `total_batches_` are the primary progress source for UI text.
- `materialization_events_` stores bounded recent materialization steps (max 24 events).
- `run_comparison_records_` stores persisted run comparison rows in panel scope.

### 36.3.4 Event emission contract
- `RecordPanelEvent(action, detail)` is called for:
  - panel lifecycle (`Created`, `Destroyed`, visibility transitions),
  - batch metric writes and loss curve truncation (`WriteLoss`, `WriteAccuracy`, `TrimData`, `WriteBatchProgress`),
  - materialization and run export failures (`RecordPanelEvent` from helper methods).
- In non-plotting builds, this is forwarded to `CrashRunRecorder::MarkPanelEvent`.

## 36.4 State transition contract (preparation -> execution -> completion)

```text
State A: Idle
  --main window -> launch path--> Preparation
  -> SetPreparationState(true, msg, progress)
  -> RecordMaterializationProgress(event) [0..24 bounded events]
  -> SetMaterializationComplete(dataset, ops, status="completed")
  --or--> SetPreparationFailed(error)

State B: Preparation complete, entering training
  -> StartTrainingCommon()
  -> SetTrainingState(true, epoch=0, total_epochs,0,0)
  -> per-batch SetBatchProgress updates epoch/batch immediately
  -> per-epoch SetTrainingState + epoch metrics
  -> optional SetBatchProgress never regresses current epoch

State C: Terminalization
  -> SetTrainingComplete(total_time,status,reason,checkpoint info, best metrics)
  -> AddRunComparisonRecord
  -> on_training_end_(success,metrics) callback
```

## 36.5 Cross-thread correctness and responsiveness contracts
- `SetBatchProgress` exists to avoid stale epoch text for long first epoch runs.
  - It updates epoch counter at first batch event if that batch belongs to a new epoch.
  - This prevents user-facing UI freezing at `Epoch 0/N` while a long epoch runs.
- `AddLossPoint` and `AddAccuracyPoint` use `epoch` as `double` (not int) to support intra-epoch samples.
- `TrimDataIfNeeded` trims each series independently and logs remove counts by metric name.
- `ShowCustomMetrics` and sequence-specific metric keys are gated by configuration, keeping panel clutter bounded.

## 36.6 Python and external API contract

The dashboard has a global pointer bridge to support Python tooling.

### 36.6.1 Global bridge
- C++ globals:
  - `set_training_plot_panel(cyxwiz::TrainingPlotPanel* panel)`
  - `get_training_plot_panel()`
- Backed by module-level static pointer (`g_training_plot_panel`) in `training_plot_panel_global.cpp`.
- `get_training_plot_panel` returns the current pointer for external scripts.

### 36.6.2 Python surface
- `TrainingPlotPanel` object is bound into module as `cyxwiz_plotting.TrainingPlotPanel`.
- Methods exported mirror runtime update and export APIs.
- `get_training_plot_panel()` is exported with reference return policy (same live C++ object).
- `set_training_plot_panel()` is exported as the integration point for host-side injection.

Important note:
- MainWindow registers the panel object with the scripting engine for script access, while the raw global setter is available as an explicit boundary function for host wiring.

## 36.7 Failure and terminal semantics

### 36.7.1 UI-level terminalization
- `SetTrainingComplete` always writes:
  - elapsed time,
  - terminal status,
  - terminal reason,
  - checkpoint fields,
  - optional validation checkpoint metrics,
  - epoch index for terminal checkpoint.
- `SetPreparationFailed` is guarded by `if (is_training_) return;`, so a running training session is not overwritten by preparation errors.

### 36.7.2 Runtime failure visibility
- `TrainingManager::StopTraining` cancels both executor and AsyncTaskManager task.
- Async task completion uses terminal status strings (`early_stopped`, `cancelled`, default success).
- `on_training_end_` is always invoked with `success = !stop_requested_` and final metrics.
- Task completion reason text is included in task-marking and panel completion state.

## 36.8 Evidence anchors

| Layer | Evidence |
|---|---|
| Dashboard API surface | `cyxwiz-engine/src/gui/panels/training_plot_panel.h` |
| Panel implementation details | `cyxwiz-engine/src/gui/panels/training_plot_panel.cpp` |
| Materialization -> dashboard wiring | `cyxwiz-engine/src/gui/graph_training_launcher.cpp` |
| Launch orchestration and panel pass-through | `cyxwiz-engine/src/gui/main_window.cpp` |
| Runtime training callback fan-out | `cyxwiz-engine/src/core/training_manager.h` and `cyxwiz-engine/src/core/training_manager.cpp` |
| Python binding + registry contract | `cyxwiz-engine/src/gui/panels/training_plot_panel_global.cpp` and `cyxwiz-engine/python/plot_bindings.cpp` |

## 36.9 Lean guardrail read
- Essential kept in this section:
  - one writable, lock-safe dashboard sink,
  - one async path for preparation and one execution path for training data,
  - bounded queues for both materialization and metric series.
- Compatibility debt present but bounded:
  - global pointer bridge and scripting binding remain available, but host-level lifecycle coupling still exists through `MainWindow` + Python startup.
- Suggested minimal simplification:
  - move the optional Python global setter behind a single adapter interface to make explicit ownership at startup and avoid accidental multiple publishers.
