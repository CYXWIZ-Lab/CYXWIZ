# CyxWiz System Debugger Architecture

This document defines the system design for the CyxWiz Studio Debugger.

The debugger is not only a crash detector. It is the workflow that helps an engineer move from graph design to safe training:

1. compile the graph
2. validate data and preprocessing
3. run synthetic local debug
4. run a small real-data smoke test
5. decide what to adjust
6. only then run full training

## Purpose

The System Debugger should answer:

- did the graph compile correctly
- did the selected data load correctly
- did preprocessing produce the expected tensors
- did the model forward pass produce valid activations
- did the loss receive valid targets
- did backward pass produce gradients for trainable parameters
- did the optimizer step update trainable parameters
- is the model ready for full training
- if not, what should the engineer change next
- if the engine process crashes, what run crashed and what was the last known stage

For CyxWiz, this is a core product feature. Engineers should not have to train for hours to discover a graph, preprocessing path, or gradient path is broken.

## Terms

`Local Debug`

The current backend sanity check. It uses a synthetic batch and runs forward, loss, backward, and one optimizer step.

`Studio Debugger`

The user-facing debugger panel and workflow inside CyxWiz Studio.

`System Debugger`

The full architecture described here: session management, preflight, local debug, smoke run, trace collection, UI, persistence, and recommendations.

`Smoke Run`

A short real-data run on a deterministic subset, for example 100 training samples. It validates the real data path without committing to full training.

`Crash Envelope`

A persisted record written during training that survives process exit. It stores the active run id, graph hash, dataset, epoch, batch, last training stage, thread id, backend, and any imported Windows Error Reporting data.

## Workflow

The debugger workflow should be:

1. User edits graph.
2. User runs Compile.
3. System runs Preflight.
4. System runs Local Debug on synthetic data.
5. User runs Smoke Run on a small real-data subset.
6. Studio Debugger shows traces, warnings, metrics, and recommendations.
7. User adjusts graph or training parameters.
8. User starts full training only after debugger checks are clean.

This workflow prevents the current expensive loop:

1. change a small backend or graph setting
2. train for hours
3. discover the issue late
4. repeat

## High-Level Architecture

The System Debugger is composed of these subsystems:

- `DebugSessionManager`
- `PreflightValidator`
- `LocalDebugRunner`
- `SmokeRunRunner`
- `GraphTraceCollector`
- `PreprocessingTraceCollector`
- `ModelTraceCollector`
- `StudioEventTracer`
- `RecommendationEngine`
- `StudioDebuggerPanel`
- `DebugRunStore`
- `CrashRunRecorder`

The design principle is simple: each debug run has one immutable session snapshot and all trace records point back to that session.

## Implementation Progress - 2026-05-21

The first implementation pass is underway. The architecture is no longer only a
proposal; these components now exist in code:

- `DebugSessionManager`
- `PreflightValidator`
- `SmokeRunExecutor`
- `DebugRunStore`
- `DebugTraceRecord`
- `TrainingTraceCollector`
- `CrashRunRecorder`
- `DebugRecommendationEngine`
- async Studio Debugger execution through `AsyncTaskManager` / Task View
- first Graph Trace View based on a frozen graph snapshot
- per-node trace status aggregation for severity, trace count, issue count, and
  recommendation count
- plot panel lifecycle/read/write/trim event hooks for long-training crash
  diagnosis

The active implementation flow is:

1. Capture the graph snapshot on the UI thread.
2. Create a debug session from that immutable snapshot.
3. Run preflight, local debug, or smoke work in a background task.
4. Persist useful run metadata and trace summaries.
5. Render issues, recommendations, trace rows, and graph-node status in
   `StudioDebuggerPanel`.

The next slice should stay focused on the visual debugger:

1. improve graph trace status rendering
2. add selected-node inspector details
3. connect trace rows to graph-node selection
4. then add Windows crash import and richer tensor/value preview

## Debug Session

`DebugSession` is the root object for one debugger run.

It should capture:

- run id
- run mode: `Preflight`, `LocalDebug`, `SmokeRun`, `FullTrainTrace`
- graph snapshot
- graph hash
- compile result
- dataset name and source path
- selected sample id or sample set
- preprocessing config
- training config
- model layer config
- active Studio panel
- selected node ids
- timestamp
- engine version
- backend device information

The session must be immutable after the run starts. If the user edits the graph while the debugger is running, the edit belongs to the next session.

## Execution Tiers

### Tier 1: Preflight

Preflight runs before model execution.

It should check:

- graph has required nodes
- required pins are connected
- graph has no cycle
- dataset exists in registry
- text vocab file exists when configured
- tokenizer, vocabulary, and padding config are coherent
- model input shape is known
- output classes match loss configuration
- graph hash matches the active UI graph

Preflight output:

- validation issues
- dataset summary
- preprocessing summary
- expected model input and output shapes
- readiness status

Preflight should be fast and should not allocate a full model unless required.

### Tier 2: Local Debug

Local Debug uses the current `DebugExecutor` path.

It should:

- build model from `TrainingConfiguration`
- create synthetic batch from `SyntheticBatch`
- run forward pass layer by layer
- compute loss
- run backward pass
- run one optimizer step
- capture layer traces
- capture gradient norms
- detect NaN, Inf, shape mismatch, zero gradients, and missing gradients

Local Debug answers:

Does the compiled model graph work independently of the real dataset?

### Tier 3: Smoke Run

Smoke Run validates the real data path.

It should:

- select a deterministic subset of real training data
- use a fixed seed
- stratify by label when labels are available
- default to about 100 samples
- run one or a few batches
- use the real tokenizer, vocabulary, padding, batcher, model, loss, backward, and optimizer paths
- collect the same trace categories as Local Debug

Smoke Run answers:

Does the model work on real data well enough to justify full training?

Smoke Run should report:

- sample count
- class distribution
- unknown token ratio
- truncation ratio
- pad ratio
- input tensor shape
- target tensor shape
- loss finiteness
- first-batch loss
- short-run loss trend
- gradient coverage
- runtime path selection

### Tier 4: Full Train Trace

Full Train remains the long-running training path.

The debugger should not trace every tensor for every batch by default. Instead, it should record lightweight summaries:

- epoch metrics
- periodic batch metrics
- warnings
- runtime fallback events
- first failure trace
- first NaN or Inf trace
- checkpoint metadata
- final model metadata

For long runs, the debugger must also persist a crash-safe heartbeat. A silent process exit after hours of training is not acceptable if Studio cannot answer which run crashed, which batch was last completed, and which stage was executing.

## Crash And Last Run Diagnostics

The Studio Debugger needs a `Crash / Last Run` lens.

This is required because long training can fail outside normal C++ exception handling. Example observed failure:

- training reached epoch 7 batch 150 of 659
- last engine log showed normal loss and accuracy
- process closed without a normal shutdown, stop, cancellation, or exception log
- Windows reported `cyxwiz-engine.exe` APPCRASH
- fault module: `VCRUNTIME140.dll`
- exception code: `0xc0000005`
- timestamp: `2026-05-20 08:13:37`

This likely class of bug includes native access violations, UI/training-thread races, lifetime bugs, and shared state mutation while the UI reads.

### Crash Envelope Requirements

During training, the engine should persist a small local crash envelope after important state changes.

It should include:

- run id
- graph snapshot id or graph hash
- graph name
- dataset name
- sample count
- batch size
- epoch count
- current epoch
- current batch
- total batches
- active backend: CPU, CUDA, OpenCL, ArrayFire
- training thread id
- UI thread id
- last completed stage
- last event timestamp
- last user-visible log message
- whether training was running, paused, stopping, or complete

Training stages to record:

- `GetNextBatch`
- `Forward`
- `ComputeLoss`
- `Backward`
- `UpdateParameters`
- `BatchCallback`
- `UIPlotUpdate`
- `Validation`
- `Checkpoint`
- `Shutdown`

The crash envelope must be flushed often enough that process death still leaves useful evidence. It should be small and overwrite the same current-run file rather than append large logs every batch.

Suggested current-run file:

`workspace/.cyxwiz/debug_runs/current_run.json`

After a normal completion, rename or copy it into:

`workspace/.cyxwiz/debug_runs/<run_id>/session.json`

### Thread And Panel Events

The debugger should record lifecycle events for objects shared across training and UI code.

Required events:

- plot panel created
- plot panel visible
- plot panel hidden
- plot panel destroyed
- training thread started
- training thread stopped
- batch callback entered
- batch callback exited
- UI plot update entered
- UI plot update exited
- plot data read
- plot data write
- plot data trimmed

This is specifically needed for suspected races such as:

- UI viewport reads plot state through `HasData()`
- training thread mutates plot vectors through batch callbacks
- `TrimDataIfNeeded()` erases vectors after many batch points
- training thread holds a raw `TrainingPlotPanel*`

### Windows Error Reporting Import

On next startup, the engine should check for evidence of a previous abnormal exit.

The first implementation can support manual or local import of Windows crash information. Later, the engine can query Windows Error Reporting or Event Viewer data when available.

Fields to capture:

- executable name
- fault module
- exception code
- crash timestamp
- process id if available
- Windows report id if available

The imported crash marker should be attached to the last active crash envelope if timestamps are close.

### Crash Lens UI

The Studio Debugger should show a `Crash / Last Run` lens.

Example:

```text
Run #42 - Training Crash
Status: crashed
Last event: Epoch 7 batch 150/659
Last stage: BatchCallback -> TrainingPlotPanel update
Fault: 0xc0000005 in VCRUNTIME140.dll
Backend: CUDA / ArrayFire
Dataset: sentiment_mental_health
Graph: Stacked BiGRU Classifier
```

Timeline example:

```text
08:10:40  Epoch 7 started
08:10:41  Batch 1 complete
08:11:31  Batch 50 complete
08:12:21  Batch 100 complete
08:13:11  Batch 150 complete
08:13:37  Process crash detected by Windows Error Reporting
```

Warnings should include:

- potential race: UI thread read training plot data while training thread was mutating plot vectors
- potential lifetime issue: training thread holds a raw `TrainingPlotPanel*`
- potential missing crash envelope: last run ended without normal completion marker

### Acceptance Criteria For Crash Diagnostics

The debugger is useful for this class of bug when it can answer:

- what training run crashed
- which graph and dataset were active
- which epoch and batch were last completed
- which stage was executing after the last normal log
- whether the crash was likely in model compute, batch loading, callback/UI update, or shutdown
- which UI/backend objects were shared across threads
- whether Windows reported an APPCRASH
- which fault module and exception code Windows reported

## Trace Model

All trace records should share a common structure:

```cpp
struct DebugTraceRecord {
    std::string run_id;
    int node_id;
    std::string node_name;
    std::string node_type;
    std::string phase;
    std::string role;
    std::vector<size_t> input_shape;
    std::vector<size_t> output_shape;
    std::string dtype;
    float duration_ms;
    std::string status;
    std::vector<ValidationIssue> issues;
    nlohmann::json payload;
};
```

`payload` is node-specific. It should stay small by default and store previews, summaries, and statistics rather than full tensors.

## Trace Roles

Use consistent roles so the UI can filter and compare traces.

Recommended roles:

- `RawInput`
- `PreprocessingOutput`
- `FeatureTensor`
- `ModelInput`
- `Activation`
- `Parameter`
- `Gradient`
- `Prediction`
- `Target`
- `Loss`
- `OptimizerStep`
- `CompileArtifact`
- `GeneratedCode`
- `StudioEvent`
- `Warning`
- `Error`

## Preprocessing Trace

Preprocessing must be first-class. Most model failures start before the model layers.

### Text

Trace payload should include:

- raw text preview
- normalized text preview
- token list preview
- token id preview
- vocab hits
- vocab misses
- unknown token count
- unknown token ratio
- pad count
- pad ratio
- truncation flag
- final sequence length

### Image

Trace payload should include:

- source path
- original width and height
- decoded channels
- resize mode
- final width and height
- normalization min, max, mean
- augmentation flags

### Audio

Trace payload should include:

- source path
- sample rate
- waveform length
- feature extractor type
- feature output shape
- normalization summary

### Tabular

Trace payload should include:

- selected feature columns
- label column
- missing value summary
- normalization summary
- one-hot summary
- final feature count

## Model Trace

The model trace should extend the current `DebugExecutor` behavior.

For each model layer:

- predicted shape
- actual shape
- shape match status
- dtype
- forward time
- activation min, max, mean, std
- NaN count
- Inf count
- backend path: CPU, CUDA, OpenCL, fallback

For backward:

- parameter name
- parameter shape
- gradient shape
- gradient L2 norm
- zero-gradient flag
- missing-gradient flag
- NaN or Inf flag

## Studio Event Trace

The debugger must explain Studio state, not only backend math.

Record:

- graph loaded
- graph edited
- node selected
- data applied
- compile started and completed
- local debug started and completed
- smoke run started and completed
- train started
- export started
- selected sample changed
- active panel changed
- user-visible warning or error displayed

Each event should include:

- run id if tied to a run
- timestamp
- graph hash
- selected node id
- action name
- status
- short message

## Recommendation Engine

The debugger should produce practical recommendations from traces.

Initial rule set:

- high unknown-token ratio: adjust tokenizer, vocab file, vocab size, or `min_word_freq`
- high truncation ratio: increase `max_length`
- shape mismatch: inspect node wiring and layer config
- missing gradients: inspect module parameter/gradient bookkeeping
- all-zero gradients: inspect disconnected layer, dead activation, or loss target
- loss NaN or Inf: lower learning rate, inspect input ranges, inspect labels
- both train and validation flat in smoke/full run: increase model capacity or revise features
- train improves but validation worsens: increase regularization or fix class imbalance
- CPU fallback in expected GPU path: fix backend support before comparing model quality

Recommendations must be labeled as guidance, not absolute truth.

## UI Design

`StudioDebuggerPanel` should become the main user surface.

Main controls:

- run mode selector: `Preflight`, `Local Debug`, `Smoke Run`
- sample selector
- rerun button
- stop button
- freeze-on-error toggle
- lens selector
- run history selector

Main views:

- issue summary
- node timeline
- Studio event timeline
- selected trace inspector
- tensor preview table
- gradient table
- preprocessing summary
- recommendations panel

Lenses:

- `Overview`
- `Preprocessing`
- `Shapes`
- `Values`
- `Gradients`
- `Runtime`
- `Studio Events`
- `Recommendations`

The panel should focus nodes on the canvas when a trace row is selected.

## Persistence

Persist only useful debug artifacts.

Store:

- run metadata
- graph hash
- compile summary
- trace summaries
- issue list
- recommendations
- smoke sample ids
- scalar metrics

Do not store full tensors by default. Full tensor dumps should be opt-in and capped.

Suggested location:

`workspace/.cyxwiz/debug_runs/<run_id>/`

Suggested files:

- `session.json`
- `trace.json`
- `events.json`
- `recommendations.json`
- `metrics.json`

## Integration With Existing Code

Existing pieces to reuse:

- `GraphCompiler`
- `DebugExecutor`
- `SyntheticBatch`
- `TextDataset`
- `TextDatasetBatcher`
- `TrainingExecutor`
- `TrainingManager`
- `StudioDebuggerPanel`
- `ValidationIssue`

Likely new files:

- `cyxwiz-engine/src/core/debug_session.h`
- `cyxwiz-engine/src/core/debug_session_manager.h/.cpp`
- `cyxwiz-engine/src/core/preflight_validator.h/.cpp`
- `cyxwiz-engine/src/core/smoke_run_executor.h/.cpp`
- `cyxwiz-engine/src/core/debug_trace_record.h`
- `cyxwiz-engine/src/core/debug_run_store.h/.cpp`
- `cyxwiz-engine/src/core/debug_recommendation_engine.h/.cpp`
- `cyxwiz-engine/src/core/studio_event_tracer.h/.cpp`

## Implementation Phases

### Phase 1: Stabilize Current Debugger

- keep `DebugExecutor` green
- keep text, embedding, GRU, and BiGRU regression tests
- ensure params with gradients and missing gradients are correct
- ensure Studio panel shows current debug result cleanly

### Phase 2: Preflight Validator

- extract compile and data-readiness checks into a reusable validator
- show preflight result in Studio Debugger
- block smoke run when preflight has errors

### Phase 3: Text Preprocessing Trace

- add trace payloads for `TextTokenizer`, `TextVocabulary`, and `TextPadding`
- show raw text, tokens, ids, unknown ratio, and truncation ratio
- use sentiment graph as the first test case

### Phase 4: Smoke Run

- implement deterministic subset selection
- add class-stratified sampling where labels exist
- run one or a few real-data batches
- collect loss, gradient, and preprocessing summaries
- expose the result in Studio Debugger

### Phase 5: Recommendations

- add rule-based recommendations
- connect recommendations to trace rows
- explain which parameter or node should be adjusted next

### Phase 6: Run History

- persist debug runs
- compare Local Debug and Smoke Run results
- compare smoke runs before and after graph edits

## Acceptance Criteria

The first useful System Debugger is complete when:

- user can run Preflight from Studio Debugger
- user can run Local Debug from Studio Debugger
- user can run Smoke Run on a small real-data subset
- text preprocessing traces show raw text, tokens, ids, unknowns, padding, and truncation
- model traces show shape, timing, NaN/Inf, and gradient status
- the debugger recommends practical next changes
- full training is no longer the first place engine or graph errors are discovered

## Non-Goals

For the first implementation, do not build:

- a full TensorBoard replacement
- a full Netron replacement
- distributed tracing across machines
- full tensor storage for every batch
- automatic model architecture search
- automatic hyperparameter tuning

The first goal is a reliable debugging workflow for local Studio users.
