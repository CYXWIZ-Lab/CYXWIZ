# To Fix 9 - CyxWiz Studio Debugger Architecture

This note defines the debugger we need for CyxWiz Studio.

Short answer:
- `Local Debug` is a model sanity check.
- We need a separate Studio debugger for the whole Studio experience.
- The first major use case is node-by-node graph tracing.
- The debugger should show preprocessing, data flow, intermediate tensors, node outputs, UI actions, compile/export events, and runtime failures.
- It should help answer both "what happened in this graph?" and "what happened in Studio?"

## Implementation Progress - 2026-05-21

This section records what has already moved from design into code so future
work continues from the current implementation instead of restarting the
architecture discussion.

Built so far:
- `DebugSessionManager` with immutable graph snapshot ownership for debug runs.
- `PreflightValidator` for reusable compile/data-readiness checks.
- `SmokeRunExecutor` first slice for small real-data smoke checks.
- `DebugRunStore` and `DebugTraceRecord` for persisted run/trace records.
- `TrainingTraceCollector` for training-stage progress and crash diagnostics.
- `CrashRunRecorder` for the first crash-envelope layer.
- `DebugRecommendationEngine` for rule-based warnings and next-step guidance.
- `StudioDebuggerPanel` async execution through `AsyncTaskManager` and Task View
  so Run Debug should not block the whole app.
- First Graph Trace View in the debugger UI using the frozen graph snapshot.
- Per-node trace status aggregation: severity color, trace count, issue count,
  recommendation count, and tooltip summary.
- Training plot lifecycle/read/write/trim trace hooks to help diagnose the
  long-training crash pattern around plot vector mutation.
- Live training trace reload in the Runtime lens, including a Tracy-inspired
  per-thread event timeline and selected-event inspector.
- Per-layer timing capture for `ModelForward` and `ModelBackward`, including
  layer index, layer name, input shape, output shape, and duration.
- Runtime warning capture from backend debug hooks into the training trace.
- Lightweight memory snapshots on training trace events: CyxWiz CPU
  allocated/peak bytes, ArrayFire allocated/locked bytes, and ArrayFire buffer
  counts.
- First Memory Trace UI in the Runtime lens with a recent memory trend graph,
  latest memory bars, hover tooltips, and hints for CPU fallback, host copies,
  GPU pressure, and retained device buffers.
- Text loader async/shared-state cleanup and text batcher warning cleanup are
  included in the same stabilization pass.

Current flow:
1. User opens Studio Debugger.
2. Studio captures a graph snapshot on the UI thread.
3. Debug work runs as an `AsyncTaskManager` task visible in Task View.
4. Backend/debugger code operates on the frozen snapshot.
5. Results are returned to the panel and rendered as trace lists, graph status,
   issues, recommendations, and persisted debug artifacts.

Next implementation steps:
1. Make the Graph Trace View more useful with selected-node details and trace
   filtering.
2. Add a Node Inspector lens with inputs, outputs, shape, warnings,
   recommendations, and related training events for the selected node.
3. Add Windows crash import from WER/Event Viewer and match it to the most
   recent debug/training run by timestamp, process name, and run id when
   available.
4. Deepen CPU/GPU runtime tracing so the debugger can classify backend paths
   and fallback reasons per operation, not only record warnings.
5. Deepen memory tracing with per-node tensor allocation summaries, peak
   markers, and out-of-memory risk signals.
6. Add richer tensor/value previews for activations, gradients, token ids,
   unknown-token ratios, padding, and truncation.
7. Improve sample selection so Smoke Run can choose deterministic and
   stratified examples from dataset preview/labels.
8. Polish the debugger UI after the workflow is stable: layout, colors,
   search/filter controls, graph interactions, and Comgra-like trace navigation.

Still not complete:
- true node-by-node execution for every real graph node
- full tensor/value preview tables
- full CPU/GPU fallback classification and runtime path timeline
- per-node tensor allocation summaries and peak markers
- generated-code/export correlation
- deep graph edit and panel-event tracing
- robust WER/Event Viewer import
- professional UI polish for the debugger surface

## Why This Is Needed

The current `Local Debug` path is useful, but it is not enough for complex graphs or Studio-level troubleshooting.

What it does today:
- validates the graph
- runs a synthetic forward pass
- runs backward and optimizer step
- shows layer traces and gradient norms

What it does not do:
- trace preprocessing nodes as first-class steps
- show tokenization, vocab lookup, padding, or truncation
- inspect per-node values in a real graph execution path
- help debug multi-stage pipelines where the bug is before the model layers
- correlate Studio UI state, compile/export attempts, backend requests, and graph execution

For the current text pipeline, that gap matters:
- `TextTokenizer` transforms raw text into tokens
- `TextVocabulary` maps tokens to ids
- `TextPadding` forces fixed length
- `Embedding` and `GRU` only see the final numeric tensor

If the issue is in tokenization or vocab coverage, `Local Debug` cannot reveal it.
If the issue is stale Studio state, wrong sample selection, wrong compile target, or a mismatch between the UI and backend graph snapshot, `Local Debug` cannot explain that either.

## Studio Debugger Goal

The Studio debugger should answer two questions:

1. What happened at every node between input data and final output?
2. What happened across Studio while the user tried to build, compile, run, export, or inspect that graph?

For graph execution, the debugger must trace:
- node input
- node output
- node metadata
- shape changes
- sample values or previews
- errors and warnings

For Studio itself, the debugger must trace:
- graph edits that affect execution
- validation and compile attempts
- dataset/sample selection
- generated-code/export events
- panel/tool actions that change debugger state
- backend requests and responses relevant to the active run
- user-visible errors, warnings, and stale-state problems

The graph debugger is the first product slice. The architecture should still leave room for Studio-wide debugging so we do not end up with separate unconnected tools for graph tracing, compile debugging, export debugging, and UI state debugging.

## What The Debugger Must Help Us With

The real value of Studio Debugger in CyxWiz is not just finding crashes. It is helping us decide what to change next when a model trains but does not improve.

The debugger should make these questions answerable:
- did the data load correctly
- did tokenization and vocabulary build the expected ids
- did padding and truncation preserve the right sequence length
- did the graph compile to the same shape the model actually sees
- did the first training batch reach the loss with valid labels
- did gradients flow through every trainable layer
- did any layer silently fall back to a slower or weaker path
- are train and validation both flat, or is the model overfitting
- is the issue in preprocessing, model capacity, learning rate, or class imbalance

This matters for the current sentiment and text graphs because the failure mode is often not a hard crash. More often the graph runs, but the curve stalls, oscillates, or collapses after a few epochs. In that case the debugger should point us at the next change, not just say "Complete."

## Debugger Focus Areas For Training Work

When we build models, the debugger should focus on these points first:

1. Data preprocessing
   - raw sample preview
   - token list
   - vocabulary hits and misses
   - unknown token count
   - pad/truncation count
   - final tensor shape

2. Shape propagation
   - input shape at every node
   - output shape at every node
   - batch dimension handling
   - stacked recurrent shapes
   - embedding and dense compatibility

3. Gradient flow
   - parameter/gradient coverage
   - zero-gradient detection
   - missing-gradient detection
   - dead-layer or disconnected-layer warnings

4. Optimization behavior
   - loss trend per batch and per epoch
   - validation trend
   - learning-rate sensitivity
   - unstable updates versus underfitting

5. Runtime path selection
   - CPU fallback versus GPU path
   - backend selected by the run: CPU, CUDA, OpenCL, ArrayFire
   - backend actually used by each node/layer
   - fallback reason when a GPU-capable graph runs on CPU
   - bidirectional recurrent support
   - multi-layer recurrent support
   - batcher mode selection

6. Memory behavior
   - CPU RAM used by dataset loading and batching
   - GPU/device memory allocated, locked, cached, and peak usage
   - per-node tensor allocation summaries where available
   - warnings before out-of-memory failures
   - whether memory pressure caused smaller batches, fallback, or failed allocation

7. Studio state
   - stale graph snapshot
   - wrong dataset selected
   - wrong vocab file selected
   - compile/export mismatch
   - node edits not reflected in the next run

## How The Debugger Should Guide The Next Train

The debugger should not just report traces. It should help us tune the next run.

Recommended decision loop:
- if token coverage is poor, fix vocabulary or tokenizer first
- if shapes are wrong, fix preprocessing or layer wiring first
- if gradients are missing, fix module bookkeeping first
- if loss is noisy but valid, adjust learning rate and batch size
- if train improves but val stalls, reduce overfitting with dropout, class weights, or early stopping
- if both train and val flatten early, increase model capacity or change the sequence head
- if a faster or richer path exists but the model uses a fallback, fix the backend path before changing hyperparameters
- if memory pressure is high, reduce batch size or sequence length before changing the model itself

For the sentiment example, that means the debugger should make these tuning calls easy:
- vocab size and `min_word_freq`
- max sequence length
- embedding dimension
- GRU hidden size
- number of recurrent layers
- bidirectional on/off
- dropout rate
- learning rate
- class balance handling

## Smoke Test Mode

The debugger should have a fast smoke-test path before any full training run.

Suggested behavior:
- sample a small deterministic subset of the training data, for example 100 rows
- keep the sample stratified when labels are available
- run one or a few batches through the full path
- collect the same trace artifacts as Local Debug
- report whether the run is stable enough to justify full training

Why this matters:
- it gives engineers a quick read on whether the model is wired correctly
- it shows whether a change improved or worsened the current graph
- it exposes issues in preprocessing, class balance, sequence length, or model depth before a long train
- it lets the engineer iterate on model changes quickly instead of waiting hours for feedback

What the smoke test should surface:
- token coverage
- unknown token ratio
- truncation ratio
- loss finiteness
- gradient flow
- shape mismatches
- early signs of underfitting or overfitting

What the smoke test should not replace:
- full-dataset training
- proper validation metrics
- long-run convergence checks

## Recommended Architecture

### Architecture Aligned To Current Workflow

CyxWiz should not treat debugger design as a separate research project. The architecture has to match the workflow we already use in the engine:

1. Graph edit
2. Compile and validate
3. Preflight data checks
4. Local Debug on one synthetic batch
5. Smoke Run on a tiny deterministic real-data subset
6. Full training only after the first four steps are clean

That means the debugger architecture should be organized around one shared debug session object and four execution tiers.

#### Shared Session Layer

One `DebugSession` owns the run.

It should capture:
- graph snapshot
- node parameters
- dataset selection
- preprocessing config
- selected sample or sample set
- run mode
- compile hash
- timestamp
- active Studio UI state

This object is the source of truth for every trace, warning, and comparison view in the debugger.

#### Tier 1: Preflight Validator

This runs before any model execution.

Responsibilities:
- confirm data is loaded
- confirm vocab exists and can be loaded
- confirm text/image/audio preprocessing config is valid
- confirm required graph links exist
- confirm model input and output shapes are compatible
- confirm the selected run target matches the current graph snapshot

This is where most cheap failures should stop.

#### Tier 2: Local Debug

This is the current backend sanity check.

Responsibilities:
- build the model from the graph
- run one synthetic batch
- run forward, loss, backward, optimizer step
- capture layer traces and grad norms
- catch shape mismatch, NaN, Inf, or missing-gradient bugs

This tier answers: “Does the model graph itself work?”

#### Tier 3: Smoke Run

This is the missing bridge between Local Debug and long training.

Responsibilities:
- run 100 or so deterministic real samples
- stratify when labels exist
- use the real preprocessing path, real vocab, and real loss
- run only a few batches
- report whether the model is healthy enough for full training

This tier answers: “Does the model work on the actual data path?”

#### Tier 4: Full Train

This is the existing long training path.

Responsibilities:
- run the entire dataset
- update the training dashboard
- record epoch metrics
- preserve checkpoints and final metrics

This tier should only be used once the first three tiers are green.

### Studio Debugger Role

The Studio Debugger is the UI and decision layer over those tiers.

It should:
- choose which tier to run
- show the resulting trace timeline
- show preprocessing, shape, and gradient details
- compare runs
- explain what to adjust next
- keep Studio state, backend execution, and graph snapshot aligned

In practice:
- `Local Debug` is the backend engine for Tier 2
- `Smoke Run` reuses the same trace model but on real data for Tier 3
- `Studio Debugger` orchestrates all tiers and surfaces the inspection UI

### 1. Debug Session Manager

A session object owns one debug run.

Responsibilities:
- snapshot the current graph
- snapshot node parameters
- snapshot dataset selection and preprocessing config
- snapshot relevant Studio state, including active panel, selected nodes, selected sample, compile target, and export target
- assign a stable run id
- collect trace records
- collect Studio events linked to the run id
- manage cancellation and cleanup

Why:
- the debugger must run against a frozen graph state
- the user may edit the graph while the trace is running
- Studio actions must be correlated with the frozen graph state that produced the trace

### 2. Studio Event Tracer

A Studio-level tracer records the product events around a debug run.

Responsibilities:
- record graph edit events that affect execution
- record validation, compile, local debug, export, and run commands
- record selected dataset/sample changes
- record panel actions such as selecting a trace row, focusing a node, changing a debugger lens, or rerunning a sample
- record backend request ids and result statuses
- attach user-visible errors and warnings to the same run id

This tracer is not a telemetry product. It is a local debugging tool that explains why the Studio state and the backend trace may disagree.

### 3. Graph Execution Tracer

A tracing layer sits beside the normal executor.

Responsibilities:
- execute the graph node by node
- emit a trace event for each node
- record input and output shapes
- record runtime duration
- record warnings and exceptions

Trace record should contain:
- node id
- node name
- node type
- phase
- input summary
- output summary
- duration
- status
- dependencies
- dependents
- debug payload

### 4. Preprocessing Tracer

This is the critical part for text, image, audio, and tabular graphs.

For text, it should expose:
- raw text sample
- token list
- token ids
- unknown token count
- padding count
- truncation flag
- final tensor shape

For image, it should expose:
- original size
- resized size
- color format
- normalization summary

For audio, it should expose:
- waveform length
- feature extraction output shape
- spectrogram or MFCC summary

Why:
- most "it does not work" bugs in data science graphs happen before the model layer
- the current debug path skips those steps

### 5. Model Layer Tracer

This can reuse most of the existing `DebugExecutor` ideas.

Responsibilities:
- run forward pass through the compiled model
- capture per-layer output shape
- capture activation statistics
- capture gradient norms
- capture NaN / Inf checks

This layer should be a reusable backend service.

### 6. Lens And Selector Model

Comgra's strongest GUI idea is not its exact PyTorch recorder. It is the idea that the same recorded graph can be inspected through different lenses.

The Studio Debugger should support lenses such as:
- `Values`
- `Stats`
- `Shapes`
- `Warnings`
- `Gradients`
- `Preprocessing`
- `Studio Events`
- `Generated Code`

Selectors should include:
- debug run
- selected sample
- selected node
- phase: preprocess, forward, backward, compile, export, UI
- aggregation: selected sample, batch mean, batch min/max
- tensor role or debug payload role

Changing a selector should update the same inspector in place. It should not open new popups or create a separate disconnected view.

### 7. Tensor Roles And Debug Payloads

Comgra labels tensors by role. CyxWiz should do the same, but with Studio and preprocessing roles included.

Recommended roles:
- `Raw Input`
- `Preprocessing Output`
- `Feature Tensor`
- `Model Input`
- `Activation`
- `Parameter`
- `Gradient`
- `Prediction`
- `Target`
- `Loss`
- `Compile Artifact`
- `Generated Code`
- `Studio Event`
- `Warning`
- `Error`

Each node type should be able to publish a node-specific debug payload.

Examples:
- `TextTokenizer`: token boundaries, normalized text, token list
- `TextVocabulary`: token ids, unknown tokens, vocab hits and misses
- `TextPadding`: pad count, truncation flag, mask
- image resize: original size, resized size, crop details
- normalization: before and after value ranges
- model layer: activation stats, shape, NaN / Inf count
- loss node: per-sample loss
- code generation: emitted framework node, generated symbol name, validation warnings
- Studio UI: selected node, focused panel, command that triggered the run

This keeps the trace model generic while allowing each node and Studio subsystem to explain the details that matter for that step.

### 8. Inspector Panel in Studio

The UI needs a dedicated debugger panel.

It should show:
- a timeline of nodes
- a Studio events timeline for the active run
- expandable trace details per node
- tensor previews
- warnings and errors
- source node highlighting on the canvas
- upstream and downstream dependency highlighting
- role badges for tensors and Studio events

It should also let the user:
- step to next node
- step over a subtree
- rerun with a different sample
- freeze the trace at an error node
- switch lenses without rerunning the graph
- compare selected sample view with batch summary view
- jump from a Studio event to the graph node, generated code, or backend trace record it affected

Recommended layout:
- top toolbar: run, stop, step, rerun sample, freeze on error, run selector, sample selector, lens selector
- left/center: node timeline or graph-focused trace list with status badges and durations
- right inspector: selected node or Studio event details, input/output preview, shape/dtype/stats, warnings, custom debug payload
- bottom/details area: tensor table, raw trace JSON, or generated-code excerpt for advanced inspection

### Status update 2026-05-18

A first `StudioDebuggerPanel` has landed in the engine UI, but it is still a model-centric proof of concept rather than the full node-by-node and Studio-wide debugger described in this document.

Current behavior:
- shows a dedicated Studio Debugger panel instead of reusing the compile popup
- renders graph metadata, issue rows, and `DebugExecutor` trace data
- can focus the corresponding node on the canvas from a trace row

Still missing:
- preprocessing node tracing
- sample-driven graph execution
- per-node input/output previews for real pipeline steps
- step-over / rerun / freeze controls
- Studio event tracing
- lens and selector model
- node-specific debug payloads
- generated-code and export trace correlation

## Minimal MVP

The smallest useful version should support:
- one graph snapshot
- one selected input sample
- node-by-node execution
- text preprocessing trace
- layer output trace
- error stop on first failure
- a Studio run record that captures the command, selected sample, selected graph, validation result, and active panel state
- one inspector lens for values and one for warnings/stats

That MVP is enough to debug:
- tokenizer mistakes
- bad vocab paths
- padding mismatch
- shape mismatch
- broken layer wiring
- stale Studio state where the UI selection, compile target, or debug run target does not match what the backend executed

## What Should Be Reused

Reuse these existing pieces:
- `GraphCompiler` for graph snapshot and layer config
- `TextDataset` and `TextDatasetBatcher` for text preprocessing logic
- `DebugExecutor` for model-layer trace concepts
- compile popup rendering patterns for issue display

Do not reuse `Local Debug` as-is for the full debugger.
It is too synthetic and too model-centric.

## Open Source Components To Reuse

The Studio Debugger itself should stay custom, but some parts can reuse open source tools instead of being built from scratch:

- `Netron` for static model graph inspection and tensor metadata viewing
- `TensorBoard` for training curves, scalar tracking, and embeddings
- `OpenTelemetry` or Chrome-trace style events for structured runtime tracing
- `Comgra` as design inspiration for dependency-focused tensor inspection, tensor roles, selector-driven GUI, sample-vs-summary views, helper debug values, and anomaly-oriented recordings
- `Tracy` as an optional developer profiler and as design inspiration for runtime timelines, zones, thread lanes, memory plots, GPU events, and selected-event inspection

Recommended split:
- keep the Studio Debugger UI, node timeline, and trace selection in-house
- reuse OSS only for supporting views, exports, or instrumentation
- do not try to force a general model viewer to replace the debugger
- do not make Comgra's PyTorch training recorder the core architecture; CyxWiz needs a Studio-first debugger driven by Studio graph snapshots and Studio events
- do not embed Tracy's full profiler UI as the first Studio Debugger UI; use its profiling model and interaction patterns while keeping the user-facing debugger graph-aware and Studio-native

## What To Learn From Similar Tools

These tools are not a 1:1 match for CyxWiz, but they show patterns worth copying.

### Tracy

Tracy is a real-time instrumentation profiler with CPU zones, GPU profiling,
memory allocation tracking, lock/contention tracing, thread views, plots, and a
high-performance timeline UI. It fits CyxWiz because our hardest training
debugging problems are runtime problems: UI thread versus training thread,
batcher stalls, recurrent layer cost, GPU-to-CPU fallback, memory pressure, and
silent native crashes.

What to borrow:
- nested zones: `Training`, `Epoch`, `Batch`, `GetNextBatch`, `Forward`,
  `LayerForward:LSTM`, `Backward`, `LayerBackward:LSTM`, `UpdateParameters`,
  `BatchCallback`, `UIPlotUpdate`
- thread lanes for UI thread, training thread, data loader workers, and backend
  worker threads
- timeline-first runtime view instead of only a table
- warning/event markers over the timeline for fallback, NaN/Inf, OOM, shape
  mismatch, checkpoint save, graph edit, and crash envelope updates
- memory plots for process RAM, ArrayFire allocated bytes, locked bytes,
  allocation buffers, GPU total/available estimates, and peak markers
- selected-event inspector showing duration, thread, stage, node/layer, shapes,
  backend, memory summary, and related recommendations
- optional developer build instrumentation using compile flags such as
  `CYXWIZ_ENABLE_TRACY`

What not to borrow directly:
- replacing Studio Debugger with Tracy's standalone profiler UI
- exposing low-level profiler concepts before users can answer "what node or
  graph decision should I change next?"
- storing full profiler traces by default for normal users

Recommended CyxWiz split:
- Studio Debugger keeps the product UI: graph trace, node inspector, runtime
  lens, recommendations, crash lens, and run history
- Tracy integration is optional and developer-oriented: enable it for deep
  profiling sessions, CI perf investigations, and backend hot-path work
- Studio Debugger can export or correlate a run id with a Tracy capture later,
  but first it should implement a Tracy-inspired timeline from our own
  lightweight trace events

Implementation idea:

```cpp
#ifdef CYXWIZ_ENABLE_TRACY
#include <tracy/Tracy.hpp>
#define CYX_PROFILE_ZONE(name) ZoneScopedN(name)
#else
#define CYX_PROFILE_ZONE(name)
#endif
```

Initial zones to add later:
- `DataLoader/GetNextBatch`
- `Training/Forward`
- `Training/Forward/Embedding`
- `Training/Forward/Recurrent`
- `Training/Forward/DenseHead`
- `Training/Backward`
- `Training/Backward/Recurrent`
- `Training/UpdateParameters`
- `Studio/UIPlotRead`
- `Studio/UIPlotWrite`

### Comgra

Comgra is closest to the trace-inspection part of what we want. It records activations, builds a dependency graph, supports sample-focused inspection, compares early and late training states, and lets the user inspect gradient flow through the network. It also supports custom visualization layers and can keep loading new data while training is still running.

What to borrow:
- dependency-centered trace navigation
- sample selector plus summary-vs-individual views
- gradient-flow visualization
- custom debug payloads per recorded object
- live updates while training continues

What not to borrow directly:
- a PyTorch-only recorder
- a training-only mental model without Studio events

### Deepkit

Deepkit is the best example of a debugger as part of a full ML workspace. Its site describes experiment tracking, model debugging, computation management, experiment comparison, timeline insights, and real-time collaboration. The key lesson is that debugging is not isolated from execution, files, notes, or resource management.

What to borrow:
- experiment/run comparison
- artifact and file history
- notes and structured findings alongside runs
- integrated model debugger inside a broader workspace
- real-time UI state tied to the active run

What not to borrow directly:
- a workspace model that treats the debugger as a separate utility

### TensorWatch

TensorWatch is useful because it treats logs, sockets, consoles, and visualizers as streams. It also supports lazy logging, where the UI can query the live process instead of requiring everything to be pre-recorded.

What to borrow:
- stream-based trace transport
- live query or lazy logging style inspection
- many-to-many mapping between recorded data and visualizations
- custom visualization extensibility

What not to borrow directly:
- notebook-only interaction
- Python-only assumptions about how the debugger is consumed

### ClearML And ModelDB

ClearML and ModelDB are not debugger UIs in the same sense, but they are strong references for reproducibility. Both emphasize experiment history, model metadata, artifacts, metrics, and comparison across runs.

What to borrow:
- stable run identifiers
- reproducible experiment snapshots
- model/data/config lineage
- artifact retention for failing runs
- comparison between runs, not just within one run

### TensorBoard

TensorBoard remains the baseline for scalars, graphs, histograms, embeddings, and run comparison. Its logdir/run structure is a useful storage model even if CyxWiz does not copy the UI.

What to borrow:
- simple run organization
- scalar and histogram dashboards
- graph and embedding views
- easy comparison of multiple runs

### Netron

Netron is the right reference for static model inspection. It is not a runtime debugger, but it is a good companion tool for exported graphs and tensor metadata.

What to borrow:
- clean model-viewing UX
- static architecture inspection
- format-aware model parsing

### Net Result For CyxWiz

The clearest design rule is:
- use Comgra-like trace inspection for the active run
- use Deepkit-like workspace context for experiment history and notes
- use TensorWatch-like streams for live data transport
- use TensorBoard-like run organization for training summaries
- use Netron-like static inspection for exported models
- keep the actual Studio Debugger panel and graph-aware execution model custom

## What Must Be Added

Backend additions:
- trace event data structures
- Studio event trace data structures
- node-level execution dispatcher
- preprocessing trace hooks
- sample capture API
- debug-friendly tensor previews
- tensor role classification
- node-specific debug payload API
- anomaly and warning detectors
- framework-aware code generation adapters

UI additions:
- debugger toolbar
- step controls
- trace timeline
- Studio events timeline
- expandable node inspector
- sample selector
- lens selector
- role badges
- selected sample vs batch summary switch
- dependency highlighting on the canvas
- generated-code/export correlation view

Persistence additions:
- save debug run metadata
- save trace JSON for later replay
- optionally export a failing trace as a support artifact
- save Studio event records linked to the debug run
- save interesting samples, such as first failure, first NaN, high unknown-token ratio, or truncation cases

## Framework Code Generation Spec

Debugging only matters if the generated code matches the graph users built.

The code generator should support multiple ML frameworks with framework-native output:
- PyTorch
- TensorFlow
- Keras
- PyCyxWiz

What good generation means:
- the same graph exports valid code for each supported framework
- the generated code is idiomatic to that framework
- tensor shapes and preprocessing are preserved
- the code is runnable with minimal edits
- the output reflects the same behavior the debugger shows

The generator must understand framework differences such as:
- model class vs functional style
- input shape conventions
- activation and loss wiring
- optimizer setup
- training loop structure
- preprocessing and tokenization representation

The best implementation approach is:
- keep one canonical internal graph representation
- add framework adapters for code emission
- add per-framework validation rules
- add per-framework shape inference rules

Why this belongs in tofix9:
- the debugger explains what the graph does
- the generator emits code for what the graph means
- if those two disagree, users lose trust

Priority order for generation quality:
1. PyTorch
2. Keras
3. TensorFlow
4. PyCyxWiz

Failure modes to avoid:
- code compiles but is semantically wrong
- shapes are off by one dimension
- preprocessing is not reproduced
- framework-specific assumptions are ignored
- generated code is so unnatural that users rewrite it

## Important Design Rule

The debugger must not become a training mode.

It should:
- inspect
- trace
- explain
- reproduce
- correlate Studio actions with backend behavior

It should not:
- update weights
- mutate the active graph
- depend on random hidden state
- silently use a graph, dataset, compile target, or export target different from the one shown in Studio

## Relationship To Local Debug

Keep both.

`Local Debug` should remain:
- fast
- synthetic
- model-centric
- suitable for compile sanity checks

Studio Debugger should be:
- Studio-wide
- graph-centric
- sample-driven
- preprocessing-aware
- compile/export-aware
- UI-state-aware
- suitable for real troubleshooting

## Node Coverage Matrix

The debugger and Node Browser need a visible node-status map so users can
tell what is already supported, what is only a composition/preset, and what
is still planned.

Present today:
- `LSTM`
- `GRU`
- `RNN`
- `Bidirectional`
- `Embedding`

Composition-only labels / presets:
- `BiGRU` = `Bidirectional` + `GRU`
- `BiLSTM` = `Bidirectional` + `LSTM`
- these should appear as presets or patterns, not as separate backend node types

Missing or still to wire cleanly:
- explicit `BiGRU` and `BiLSTM` one-click presets in the Node Browser
- status badges in node search for `Implemented`, `Alias`, `Pattern`, `Planned`
- a node-coverage panel that shows what exists in backend, what is UI-only, and what is still deferred

Coming soon in the debugger UI:
- filters for `Implemented` vs `Missing` vs `Planned`
- search results grouped by canonical node and composition presets
- a "why is this not listed?" explanation for alias/preset nodes
- links from debugger warnings to the missing node or missing capability entry

## Why This Matters

This is important because the current product is moving toward more complex graphs and a more complete Studio workflow:
- text pipelines
- image pipelines
- audio pipelines
- multi-step preprocessing chains
- generated code and framework exports
- UI-driven dataset/sample selection
- backend compile and run services

Without a Studio debugger:
- users will guess where failures happen
- support will rely on logs only
- complex graphs will be hard to trust
- Studio UI/backend mismatches will be hard to reproduce

With a Studio debugger:
- users can see each node's effect
- preprocessing bugs become obvious
- graph behavior becomes explainable
- UI actions, compile attempts, export events, and backend traces become correlated
- the product feels like a real data/ML studio, not just a compiler

## Suggested First Implementation Order

1. Add a trace event model in the backend.
2. Add a Studio event model linked to the same run id.
3. Add text preprocessing tracing first.
4. Add node-by-node execution for one sample.
5. Render a simple trace list and Studio event list in Studio.
6. Add tensor previews, role badges, and node highlighting.
7. Add lens selection for values, stats, warnings, and Studio events.
8. Add generated-code/export correlation.
9. Extend to image and audio pipelines.

## Acceptance Criteria

The Studio debugger is good enough when it can show:
- the exact text sample selected
- token ids before `Embedding`
- unknown token count
- padding count
- output shape at each node
- the first node where a failure occurs
- the Studio command that started the run
- the graph snapshot/run id used by the backend
- whether the UI selection, compile target, and executed graph agree
- warnings tied to exact nodes or exact Studio events

At that point, the debugger becomes a real product feature and not just a log viewer.
