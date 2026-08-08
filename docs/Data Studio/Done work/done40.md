# tofix40 - Training Lifecycle Visibility, Debugger Truth, and Dashboard UX

## Status

Open.

Progress:

- Started dashboard terminal-state surfacing.
- Training Dashboard now distinguishes early-stopped runs from completed runs.
- Training Dashboard now receives terminal reason from backend metrics.
- Training Dashboard now receives restored best-checkpoint path, epoch, validation loss, and validation accuracy for the run summary.
- Started structured trace surfacing for validation/checkpoints.
- Training trace events now have first-class validation loss/accuracy fields.
- Training trace events now have checkpoint path and best-checkpoint fields.
- Training executor now records `ValidationCompleted`, `BestCheckpointUpdated`, and terminal early-stop events.
- Added generic task-progress trace fields for task id, task name, task stage, progress, memory estimate, processed items, and total items.
- Existing `AsyncTask` progress lifecycle now emits structured trace events for started, progress, completed, failed, cancel requested, and cancelled.
- Release engine build succeeded after generic task-progress trace wiring.
- Graph training launch now queues a generic `Prepare graph training` async task.
- Graph materialization and training dispatch now run through that preparation task instead of blocking the Train click path.
- Release engine build succeeded after async graph-training preparation task wiring.
- Training Dashboard now has a `PREPARING` state with progress and status text before epoch 1 starts.
- Graph training launch updates the dashboard preparation state during materialization, dataset resolution, sequence validation, and training start.
- Release engine build succeeded after Training Dashboard preparation-state wiring.
- Training Dashboard now shows a generic active engine task summary from `AsyncTaskManager`, not just graph-training preparation messages.
- Graph training now marks the node editor active during background preparation, not only after epoch training starts.
- Failed or cancelled preparation clears the node editor active state.
- Studio Debugger Training Trace now summarizes latest task progress, validation metrics, best checkpoint, and terminal reason from structured trace fields.
- Added generic `PipelineOperatorProgress` callback support for materializer operators.
- `PipelineMaterializer` now forwards progress callbacks into materializer operators.
- TF-IDF materialization now reports tokenization, vocabulary selection, TF-IDF matrix planning, Arrow column allocation, row building, finalization, processed counts, and estimated raw feature memory.
- Release engine build succeeded after generic materializer progress wiring.
- Focused `test_computation_truth_tfidf_loss` build and run succeeded after TF-IDF progress wiring.
- Training Dashboard now has a `PREPARATION FAILED` state for async preparation/materialization failures before epoch 1.
- Studio Debugger now classifies training warnings as data-transfer, device-fallback, GPU, memory, or generic warning based on existing trace warning text.
- Training Dashboard now shows the latest classified training warnings from the live training trace.
- Training Dashboard now has optional UI-only moving-average smoothing overlays for loss and accuracy curves; raw metric series are unchanged.
- Node editor now shows a visible `training` badge on graph nodes while training/preparation is active.
- Materializer progress events now carry source graph node id/name through the generic progress path and training trace.
- Studio Debugger task summary now shows the graph node responsible for the latest materializer task event when available.
- Release engine build succeeded after adding materializer node id/name trace metadata.
- Focused `test_computation_truth_tfidf_loss` build and run succeeded after materializer node metadata changes.
- Node editor now highlights the active materializer node with its current trace stage when training trace node metadata is available.
- Training Dashboard status header no longer uses a nested child panel; status, preparation, warnings, checkpoint, and terminal information now flow in the main dashboard layout.
- Added optional no-op-by-default profiling wrapper for Tracy-compatible semantic zones.
- Graph training preparation, pipeline materialization, training dispatch, and TF-IDF materialization now have CyxWiz semantic profiling zones when Tracy is enabled.
- Release engine build succeeded after optional Tracy-compatible profiling zones.
- Focused `test_computation_truth_tfidf_loss` build and run succeeded after TF-IDF profiling instrumentation.
- Added opt-in `CYXWIZ_ENABLE_TRACY` CMake flag. When Tracy is installed it links `Tracy::TracyClient` and defines `CYXWIZ_HAS_TRACY`; otherwise profiling zones remain no-op.
- Release engine build succeeded with the default Tracy-off path after adding the optional CMake flag.
- CountVectorizer materialization now reports generic progress stages, processed counts, memory estimates, and has a CyxWiz semantic profiling zone.
- Release engine build succeeded after CountVectorizer materializer progress wiring.
- Focused `test_text_arrow_materializer` build and run succeeded after text materializer progress changes.
- Focused `test_computation_truth_tfidf_loss` build and run succeeded after text materializer progress changes.
- TextTokenizer materialization now reports generic progress stages for reading text, vocabulary load/train, tokenization, Arrow column allocation, row writing, finalization, memory estimate, and has a CyxWiz semantic profiling zone.
- Release engine build succeeded after TextTokenizer materializer progress wiring.
- Focused `test_text_arrow_materializer` build and run succeeded after TextTokenizer materializer progress wiring.
- TimeSeriesWindow materialization now reports generic progress stages for reading value/time/feature columns, planning generated windows, Arrow column allocation, window writing, finalization, memory estimate, and has a CyxWiz semantic profiling zone.
- Release engine build succeeded after TimeSeriesWindow materializer progress wiring.
- Focused `test_pipeline_executor_operator_routing` build and run succeeded after TimeSeriesWindow materializer progress wiring.
- TimeSeriesFeatures materialization now reports generic progress stages for reading source values, planning engineered lag/rolling columns, building features, appending Arrow columns, memory estimate, and has a CyxWiz semantic profiling zone.
- Release engine build succeeded after TimeSeriesFeatures materializer progress wiring.
- Focused `test_pipeline_executor_operator_routing` run succeeded after TimeSeriesFeatures materializer progress wiring.
- PCA materialization now reports generic progress stages for feature resolution, feature reading, matrix planning/packing, PCA compute, Arrow output writing, memory estimate, and has a CyxWiz semantic profiling zone.
- Release engine build succeeded after PCA materializer progress wiring.
- Focused `test_pipeline_executor_operator_routing` run succeeded after PCA materializer progress wiring.
- Common tabular preprocessing operators now report generic progress and have CyxWiz semantic profiling zones: StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OrdinalEncoder, TargetEncoder, and OutlierDetector.
- Release engine build succeeded after common tabular preprocessing progress wiring.
- Focused `test_pipeline_executor_operator_routing` run succeeded after common tabular preprocessing progress wiring.
- Tree/classical model operators now report generic progress and have CyxWiz semantic profiling zones: DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, and TreeModelPredictor.
- Release engine build succeeded after tree/classical model progress wiring.
- Focused `test_pipeline_executor_operator_routing` run succeeded after tree/classical model progress wiring.
- Signal-processing operators now report generic progress and have CyxWiz semantic profiling zones: FFT, Convolve1D, and FilterDesigner.
- Release engine build succeeded after signal-processing progress wiring.
- Focused `test_pipeline_executor_operator_routing` run succeeded after signal-processing progress wiring.
- Time-series analysis operators now report generic progress and have CyxWiz semantic profiling zones: TimeSeriesDecomposition, ARIMA, ExponentialSmoothing, ACF, PACF, StationarityTest, and SeasonalityDetector.
- Release engine build succeeded after time-series analysis progress wiring.
- Focused `test_pipeline_executor_operator_routing` run succeeded after time-series analysis progress wiring.

This task exists because the engine now records some useful training facts, but the GUI does not expose them clearly. Users should not need raw JSON files or developer explanation to understand what the engine is doing, why it is slow, why it stopped, or which checkpoint is best.

## Scope

This is not the numerical correctness task. Numerical truth remains in `tofix39`.

This task is about user-visible training lifecycle truth:

- materialization progress
- loader preparation
- validation metrics
- checkpoint visibility
- early-stop explanation
- training graph visibility
- dashboard layout and scrolling
- Studio Debugger training panels

## Current observed problem

The sentiment TF-IDF graph made the issue visible.

When the user clicks Train, the engine performs a heavy synchronous preparation step before epoch 1:

- dataset rows: 52681
- TF-IDF features: 5000
- raw feature values: 52681 * 5000
- raw float estimate: about 1.05 GB
- extra peak memory: Arrow builders, strings, vectors, metadata, maps, and temporary materialization state

The user sees the GUI pause, but the engine is actually doing work:

- reading dataset
- tokenizing text
- building term counts
- selecting top vocabulary terms
- building TF-IDF values
- building a wide Arrow table
- creating train, validation, and test loaders

This is currently not visible enough in the Training Dashboard.

Important scope rule:

This must not be implemented as a TF-IDF-only fix. TF-IDF is only the current test case that exposed the weakness. The engine needs a generic task-progress contract for all expensive preparation and execution phases.

## My additional findings

These are the extra findings that should be tracked in this task:

- The backend has richer truth than the GUI currently shows. `current_run.json`, training trace, and checkpoint metadata contain facts that should be surfaced.
- Early stop is now recorded by the backend, but the dashboard does not clearly show the terminal reason as a first-class result.
- Validation metrics are not visible enough in the live/debug trace. Best checkpoint metadata has validation loss and accuracy, but users should not need to open checkpoint files manually.
- The best checkpoint is a model-quality decision point. It should be visible as its own card or panel with epoch, validation loss, validation accuracy, train metrics, and path.
- Run comparison should compare validation metrics and terminal reasons, not only training curves.
- The graph view is part of the explanation surface. Hiding or replacing the graph during training removes the user's ability to reason about which node is preparing, running, falling back, or failing.
- Materialization is a training phase, not an invisible implementation detail. It should be represented as a task with progress, warnings, memory estimate, cancellation, and failure reason.
- CPU/GPU fallback and pinned-memory limitations should be surfaced as warnings. They should not require log inspection.
- Curve smoothing should be UI-only. Raw metric points must remain the source of truth.
- The dashboard has too much hidden information behind nested scroll regions. Critical training facts should be visible by default with one main page scroll.
- Tracy should be used for low-level CPU, GPU, memory, and timeline profiling. CyxWiz should not reimplement Tracy. CyxWiz Studio should focus on engine semantics: graph compile, materialize, train, validate, checkpoint, fallback, and explain.

## Source-of-truth model

Use a clear layered truth model:

- Backend trace JSON: canonical lifecycle truth.
- Checkpoint metadata: canonical saved-model truth.
- Training Dashboard: concise live and final training truth.
- Studio Debugger: deep forensic training detail.
- Graph view: node-level execution explanation.

## Generic async preparation and materialization task

Move dataset materialization, loader preparation, and other expensive pre-execution work out of blocking UI paths and into generic background tasks.

This system must support any heavy engine task, including:

- dataset reads
- streaming dataset inspection
- tabular feature transforms
- text vectorization
- image preprocessing
- audio preprocessing
- time-series windowing
- graph compilation
- JIT preflight
- CPU/GPU placement checks
- memory planning
- train/validation/test loader creation
- checkpoint restore
- export packaging

Required behavior:

- Train click returns control to the GUI quickly.
- A task appears immediately with stage, progress text, warnings, and cancel action.
- The task reports memory estimate before large allocations.
- Materialization failure is reported before training starts.
- Cancellation leaves the run in a clear `cancelled` state.
- The task contract is reusable by Studio Debugger, Training Dashboard, Data Studio, and future engine panels.
- Every task records structured events with task name, stage, progress, warnings, and failure reason.

Generic user-visible stage examples:

- Preparing
- Reading input
- Inspecting schema
- Planning memory
- Compiling graph
- Materializing features
- Creating output table
- Creating loader
- Moving data to device
- Restoring checkpoint
- Starting execution

Task-specific details should be attached as structured metadata. For TF-IDF, show:

- row count
- feature count
- raw feature-value count
- raw memory estimate
- peak-memory warning
- vocabulary selection summary
- output table width

For image preprocessing, show:

- sample count
- decoded image size
- resize/crop policy
- channel format
- batch memory estimate

For time-series windowing, show:

- source row count
- window size
- stride
- generated sequence count
- feature width
- memory estimate

For graph compilation or JIT preflight, show:

- checked nodes
- unsupported nodes
- CPU/GPU placement
- fallback reasons
- estimated temporary memory

## Validation, checkpoint, and early-stop trace contract

The debug trace should record validation facts directly, not only train-stage events.

Required events:

- RunStarted
- PrepareStarted
- MaterializationStarted
- MaterializationProgress
- MaterializationCompleted
- LoaderCreated
- EpochStarted
- TrainEpochSummary
- ValidationStarted
- ValidationCompleted
- EpochCompleted
- CheckpointSaved
- BestCheckpointUpdated
- EarlyStopped
- Completed
- Failed
- Cancelled

Required fields:

- run_id
- graph_path
- epoch
- total_epochs
- train_loss
- train_accuracy
- validation_loss
- validation_accuracy
- test_loss when available
- test_accuracy when available
- learning_rate
- checkpoint_path
- is_best_checkpoint
- terminal_status
- terminal_reason
- stop_source
- backend_device
- fallback_reason
- materialized_rows
- materialized_features
- estimated_memory_bytes
- elapsed_ms

## Training Dashboard requirements

The dashboard should show important truth by default.

Required visible cards:

- Current status: preparing, materializing, training, validating, early stopped, completed, failed, cancelled
- Terminal reason: for example `validation_loss_plateau_patience_8`
- Best checkpoint: epoch, train metrics, validation metrics, path
- Latest epoch metrics: train and validation side by side
- Data preparation summary: rows, feature count, memory estimate, loader split
- Warnings: CPU fallback, GPU fallback, large materialization, pinned-memory unavailable

Graph curves:

- Raw curves remain the default.
- Add optional smoothing as a view setting.
- Smoothing must be clearly labeled.
- Smoothing must never overwrite raw metric data.
- Support EMA or moving average with a visible control.

## Studio Debugger requirements

Studio Debugger should provide the deep view for engineers.

Required panels:

- Run timeline
- Materialization timeline
- Validation timeline
- Checkpoints
- CPU/GPU placement and fallback
- Memory and allocation estimates
- Data loader lifecycle
- Node execution events
- Warnings and terminal reason

Tracy integration:

- Use Tracy for CPU timeline profiling.
- Use Tracy for GPU timeline profiling where available.
- Use Tracy for memory allocation visibility where practical.
- Do not duplicate Tracy's profiler UI inside CyxWiz.
- Link CyxWiz semantic events to Tracy spans where possible.

## Graph view during training

The graph should remain visible while training is active.

Required behavior:

- Keep the graph visible in read-only mode during training.
- Preserve pre-training graph state.
- Preserve compiled graph state.
- Add live node overlays for preparing, materializing, running, validating, checkpointing, failed, and skipped.
- Show CPU/GPU placement badges on nodes.
- Show warning badges for fallback, large allocation, unsupported pinned memory, and materializer failures.
- Allow clicking a node to see node-specific trace events.

## Layout and scrolling

The Training Dashboard should avoid nested scrollbars and hidden critical information.

Required layout:

- One main vertical scroll region.
- Sticky run summary at the top.
- Important information visible by default.
- Secondary details lower on the same page or in debugger tabs.
- No small inner scroll panes for critical metrics.
- Responsive layout for smaller screens.

## Secondary UX follow-up: Get Started screen redesign

This is not part of the training lifecycle/debugger core, but it should be tracked as a small design improvement because the first screen sets the product tone.

Current design target:

- Dark-themed desktop application dashboard for CyxWiz Engine.
- Deep navy/black gradient background.
- Electric blue accent for primary actions and highlights.
- Modern developer-focused style similar to IDE and ML workflow tools.
- High-contrast white/gray text on dark background.
- Minimal, sleek, and technical visual language.

Left panel:

- Large `Get started` heading.
- Search bar for recent files, projects, or commands.
- `Task starter graphs` section with cards:
  - Binary image classification
  - Multiclass image classification
  - Text classification
- Each starter card should have:
  - rounded corners
  - subtle border
  - icon
  - short description
  - blue `Open` button
- `Recent Projects` section grouped by time:
  - This month
  - Older
- Recent entries should show file/folder icon, project name, and timestamp.

Right panel:

- Prominent `Create a new project` button.
- Workflow lanes:
  - Classic ML
  - Deep Learning
- Domain starters:
  - Tabular
  - Vision
  - NLP
- File actions:
  - Open project
  - Open folder
  - Clone repository, marked planned if not implemented
- Secondary `Continue without project` action at the bottom.

Acceptance criteria:

- The first screen feels like a professional ML workflow launcher, not a placeholder.
- New users can quickly start from a template.
- Returning users can quickly find recent projects.
- Primary actions are visually obvious.
- The design remains usable on common laptop resolutions.

## Priority order

1. Surface `terminal_status` and `terminal_reason` in the Training Dashboard.
2. Surface best checkpoint metadata in the Training Dashboard.
3. Add validation metric events to the structured debug trace.
4. Add checkpoint and validation panels to Studio Debugger.
5. Move materialization, loader preparation, and heavy pre-execution work into generic background tasks with progress and cancellation.
6. Preserve graph view during training with read-only live overlays.
7. Add CPU/GPU fallback and pinned-memory warnings to dashboard/debugger surfaces.
8. Add UI-only metric smoothing.
9. Remove nested dashboard scroll regions.
10. Link CyxWiz semantic trace events to Tracy spans where practical.

## Acceptance criteria

- User can see what the engine is doing before epoch 1 starts.
- GUI does not appear frozen during heavy materialization.
- User can cancel materialization safely.
- User can see early-stop reason without opening raw JSON.
- User can see best checkpoint epoch, validation loss, validation accuracy, and path.
- User can compare runs using validation metrics.
- User can inspect the graph while training is running.
- User can identify CPU/GPU fallback and pinned-memory limitations from the UI.
- User can choose smoothed curves without losing raw metric data.
- Dashboard uses one clear page scroll instead of nested scroll panes for critical information.

### 2026-06-30 update: clustering/regression materializer visibility
- Added generic `PipelineOperatorProgressCallback` support to clustering operators: `KMeansCluster`, `DBSCANCluster`, `HierarchicalCluster`, and `GMMCluster`.
- Added generic `PipelineOperatorProgressCallback` support to regression operators: `LinearRegression` and `PolynomialRegression`.
- Added CyxWiz semantic profiling zones for these materializers so Tracy/profiler timelines show engine-level operation names instead of only low-level call stacks.
- Progress events now use the existing engine contract: `stage`, `message`, `progress`, `processed_items`, `total_items`, and `estimated_memory_bytes`.
- Release build passed for `cyxwiz-engine` and `test_pipeline_executor_operator_routing` after this slice.
- Focused routing test passed: `build\bin\Release\test_pipeline_executor_operator_routing.exe` exited with code `0`.

### 2026-06-30 update: remaining small pipeline operator visibility
- Added generic progress callback support and CyxWiz semantic profiling zones to the remaining small pipeline operators: `Identity`, `LogTransform`, `Differencing`, `TimeSeriesSplit`, and `SentimentAnalyzer`.
- These operators now report concise lifecycle stages such as read/plan/transform/write/finalize/complete using the existing `PipelineOperatorProgress` contract.
- Release build passed for `cyxwiz-engine` and `test_pipeline_executor_operator_routing` after this slice.
- Focused routing test passed: `build\bin\Release\test_pipeline_executor_operator_routing.exe` exited with code `0`.

### 2026-06-30 update: Training Dashboard terminal-state clarity
- Training Dashboard terminal state now renders stop reason as an explicit row instead of compressing it beside the epoch counter.
- Early-stopped runs now clearly show `Stop reason: early stopping triggered` while still showing the final epoch state, so `11 / 30` no longer reads like a normal completed run.
- Restored best-checkpoint information now renders under an `Active model state` section with checkpoint epoch, validation loss/accuracy when available, and checkpoint path.
- Preparation progress now uses full-width status/progress rows instead of fixed hard offsets.
- Release engine build passed after the dashboard terminal-state UI patch.

### 2026-06-30 update: graph visibility during training
- Node graph remains useful during training/preparation by surfacing the active training trace node directly on the graph.
- Active node badge now includes current trace stage and percent progress when available.
- Active node badge now has a small progress strip for materialization/preparation progress.
- Hovering the active node now shows task name, stage, message, progress, processed/total item counts, and estimated memory when available.
- Release engine build passed after the node graph training-visibility patch.

### 2026-06-30 update: Get Started screen redesign
- Refreshed the Get Started screen toward the planned dark developer-focused CyxWiz Engine launcher.
- Added a dark navy/black gradient background with subtle blue/teal glow accents.
- Left workspace panel now presents search, starter graphs, and recent projects in a stronger card-like layout.
- Starter graph rows now use clearer task/domain hierarchy and blue primary Open actions.
- Right launcher panel now has a clear `Launch workspace` heading, workflow lanes, domain starters, and file actions.
- Primary create-project action now uses a stronger electric-blue treatment while secondary actions use subdued dark cards.
- Release engine build passed after the Get Started screen redesign patch.

### 2026-06-30 validation run
- Initial combined multi-target build command timed out after 184 seconds and left a stale `MSBuild` process running; the stale process was stopped before retrying in smaller batches.
- Release build passed for `cyxwiz-engine` and `test_pipeline_executor_operator_routing`.
- Release build passed for focused validation targets: `test_text_arrow_materializer`, `test_computation_truth_tfidf_loss`, and `test_debugger_contracts`.
- Focused routing test passed: `build\bin\Release\test_pipeline_executor_operator_routing.exe` exited with code `0`.
- Text Arrow materializer test passed: `build\bin\Release\test_text_arrow_materializer.exe` exited with code `0`.
- TF-IDF computation truth test passed: `build\bin\Release\test_computation_truth_tfidf_loss.exe` exited with code `0`.
- Debugger contract test passed: `build\bin\Release\test_debugger_contracts.exe` exited with code `0`.
- Note: `test_pipeline_executor_operator_routing` prints expected negative validation errors for bad graph/operator cases; the executable completed successfully with exit code `0`.
