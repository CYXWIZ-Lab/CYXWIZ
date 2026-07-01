# To Fix 32 - CyxWiz Studio Debugger / Telemetry Spine

## Intent

Build the CyxWiz debugger into a first-class system lens, not a cosmetic panel.

The debugger should eventually be able to see, explain, and correlate the whole
Studio workflow:

- graph edits and selected path
- compile/preflight issues
- data loading and preprocessing transitions
- Arrow schema and shape changes
- model-layer forward/backward traces
- loss, metric, optimizer, and gradient behavior
- backend placement and CPU/GPU fallback decisions
- memory pressure and allocation deltas
- export/import/package activity
- crash/runtime traces
- support-bundle diagnostics
- stable error codes and recovery hints
- optional external profiler timelines such as Tracy

This ticket should not start by redesigning the UI. The UI is currently ahead
of the execution core. The next work must add real trace producers and a stable
explanation model underneath the panel.

## External Reference: Tracy

Reference: <https://github.com/wolfpld/tracy>

Tracy describes itself as a real-time, nanosecond-resolution, remote telemetry,
hybrid frame and sampling profiler. Its public README says it supports CPU
profiling, GPU profiling across major graphics/compute APIs, memory allocations,
locks, context switches, screenshots attributed to frames, and more.

What CyxWiz should learn from Tracy:

- use cheap scoped events for timing-sensitive paths,
- separate instrumentation from visualization,
- capture enough context to correlate events across threads/subsystems,
- make timeline data exportable/inspectable,
- support optional deep profiling without making normal Studio debugging depend
  on an external profiler GUI.

What CyxWiz should not do first:

- vendor or require Tracy before the internal debug trace contract is strong,
- build a second execution framework just for debugging,
- claim kernel-level or allocator-proven tracing before the runtime emits those
  facts truthfully.

Recommended Tracy relationship:

- Phase 1: internal CyxWiz debug trace spine only.
- Phase 2: optional compile-time Tracy instrumentation macros around known hot
  runtime scopes.
- Phase 3: export or bridge selected `DebugTraceRecord` timelines into
  Tracy-compatible workflows if the dependency/build story is acceptable.

## Current Code Truth

The Studio Debugger already has a serious foundation.

Already implemented:

- `StudioDebuggerPanel` renders debugger runs, lenses, trace filters, node
  details, runtime trace, memory trace, run history, and recommendations.
- `MainWindow::BuildStudioDebuggerSessionFromSnapshot` captures a frozen graph
  snapshot before async debugger execution starts.
- `DebugSession` stores run id, mode, graph hash, graph snapshot, selected
  sample, compiled configuration, preflight result, traces, and Studio events.
- `DebugTraceRecord` and `DebugNodeTraceContract` define the canonical trace
  record shape with schema `cyxwiz.debug.node_trace.v1`.
- `DebugGraphTraceExecutor` converts prepared `DebugGraphTraceStep` values into
  canonical `DebugTraceRecord` values.
- `DebugOperatorTraceAdapter` can build Arrow table input/output transition
  steps.
- `DebugRunStore` persists debugger runs, traces, Studio events, issues, and
  recommendations.
- `TextPreprocessingTracer` emits selected-sample text traces for registered
  legacy text datasets.
- `SmokeRunExecutor` runs a small real-data text smoke path and records batch,
  loss, and gradient traces.
- `TrainingTraceCollector` and `CrashRunRecorder` feed runtime/crash summaries.
- Existing debugger contracts cover canonical node traces, graph trace step
  conversion, backend classification, memory ownership estimates, export
  correlation traces, support-bundle redaction, Windows crash import parsing,
  text preprocessing traces, and node inspector summaries.

Important limitation:

- `DebugGraphTraceExecutor` does not walk or execute the real Studio graph.
- It only converts already-prepared trace steps.
- Normal Studio Debugger runs still lack real operator-backed graph execution
  traces for the active preprocessing/materialization path.

This means the UI can display traces well, but the execution core does not yet
produce enough truthful traces from real graph operators.

## tofix26 Dependency: Error Codes as Explanation Spine

`tofix26.md` defines the CyxWiz error-code system:

- `CW-C-*` compiler/graph validation
- `CW-R-*` runtime/pipeline execution
- `CW-G-*` GPU backend
- `CW-P-*` CPU/host backend
- `CW-D-*` data contract/schema
- `CW-F-*` file/IO
- `CW-M-*` memory/resource
- `CW-U-*` UI/workflow
- `CW-X-*` external integration
- `CW-S-*` serialization/artifact
- `CW-T-*` training loop

The debugger should consume this system directly.

Do not make debugger explanations depend only on ad-hoc text matching. Each
important trace issue should eventually carry:

- stable code,
- severity,
- subsystem,
- node id/name/type when available,
- human-readable message,
- diagnostic detail,
- recovery hint,
- correlation ids such as run id and graph hash.

First debugger-specific use:

- add optional `error_code` / `warning_code` fields to trace issue payloads, or
  attach codes through a structured `ErrorRecord` once `tofix26` is implemented.
- compiler/preflight issues should preserve existing readable text but expose
  `CW-C-*`, `CW-D-*`, or `CW-U-*` where obvious.
- operator execution failures should use `CW-R-*` or `CW-D-*` depending on
  whether the failure is runtime/operator execution or input schema mismatch.

## Structured Diagnostics Contract For Studio Debugger

The debugger must not depend on parsing plain log strings. The engine should
emit structured diagnostics that Studio Debugger can ingest, persist, filter,
and explain.

Recent training failures exposed the gap clearly:

- the backend knew exactly why materialization failed,
- the task system logged the failing task and node,
- the user-facing Studio Debugger did not surface the failure as a structured,
  actionable diagnostic,
- backend fallback warnings were visible in logs but not correlated to the
  model layer, backend decision, training phase, or graph run.

This is not only a logging problem. It is a debugger contract problem.

Each important compiler, materializer, runtime, training, backend, export, or
crash issue should eventually emit a diagnostic record with:

- stable `code`,
- `severity`,
- `phase`,
- `subsystem`,
- `component`,
- `node_id`,
- `node_name`,
- `node_type`,
- `dataset`,
- `task_id`,
- `run_id`,
- `graph_hash`,
- concise `message`,
- detailed `cause`,
- concrete `suggested_fix`,
- `impact`,
- `source_file`,
- `source_symbol`,
- optional `source_line`,
- original low-level error text as `raw_error`,
- timestamp.

Studio Debugger should show this in an issue inspector:

- what failed,
- where it failed,
- why CyxWiz believes that is the cause,
- what phase owns it,
- whether training stopped or continued,
- whether fallback occurred,
- what concrete change the engineer can try,
- which source component owns the fix.

### Example: TF-IDF materializer parameter failure

Observed log:

```text
Task 'Prepare graph training' failed:
Materializer failed for dataset 'sentiment_mental_health':
PipelineMaterializer: Configure failed for node 'TF-IDF 5000 min_df2':
TFIDFVectorizer: min_df values other than 1 are not supported by this operator
```

Debugger-grade diagnostic:

```text
code: CW-D-TFIDF-0003
severity: error
phase: materialization
subsystem: data
component: TFIDFVectorizer
node_name: TF-IDF 5000 min_df2
dataset: sentiment_mental_health
task: Prepare graph training
message: Unsupported TF-IDF parameter
cause: This engine build rejected min_df values other than 1.
suggested_fix: Set min_df=1 or use an engine build that supports min_df>=1.
impact: Training did not start because dataset materialization failed.
source_file: cyxwiz-engine/src/core/node_executors/tfidf_vectorizer_operator.cpp
source_symbol: cyxwiz::TFIDFVectorizerOperator::Configure
```

Why this helps:

- the issue is tied to the exact graph node,
- the failing phase is clear before training begins,
- the owning component is clear,
- the fix is user-actionable,
- the engine source location helps CyxWiz engineers narrow the code path.

### Example: ArrayFire LinearLayer CPU fallback

Observed warning:

```text
ArrayFire LinearLayer::Forward failed (reason=unsupported_dtype);
falling back to CPU. Training continues, but this path may be slower.
Context: backend=cuda; in=5000; out=96; batch=1; bias=true.
```

Debugger-grade diagnostic:

```text
code: CW-G-LINEAR-0002
severity: warning
phase: training_forward
subsystem: gpu_backend
component: LinearLayer
backend: cuda
message: GPU LinearLayer forward fell back to CPU
cause: ArrayFire matmul rejected the input dtype or layout.
impact: Training continued, but this layer ran slower on CPU fallback.
suggested_fix: Inspect dtype flow into LinearLayer and ensure Float32 tensor
               input reaches the backend.
source_file: cyxwiz-backend/src/algorithms/layers/linear.cpp
source_symbol: cyxwiz::LinearLayer::Forward
```

Why this helps:

- the warning is not lost as a plain log line,
- Studio can show that training continued,
- backend placement and fallback can be correlated with performance,
- model engineers can see which layer shape triggered the fallback,
- backend engineers can navigate directly to the relevant implementation.

### Diagnostic event routing

Diagnostic events should be routed to:

- task status UI,
- Studio Debugger issue timeline,
- selected node inspector,
- training dashboard warnings/errors,
- support bundle records,
- persistent debug run store.

The same diagnostic should not be reformatted independently by each subsystem.
The engine should create one structured record and each surface should render it
at the appropriate level of detail.

### Required debugger behavior

Studio Debugger should catch and display:

- compiler/preflight errors before execution,
- materializer configuration failures,
- operator apply failures,
- schema/shape/dtype contract failures,
- task failure reason and owning task,
- CPU/GPU backend fallback warnings,
- memory/resource warnings,
- early stopping and checkpoint decisions,
- crash/runtime summaries,
- export/import/package warnings.

Debugger issue visibility should not require the user to open raw logs.

### Implementation direction

Do not start with a huge catalog. Start with a small `DiagnosticRecord` or
`ErrorRecord` bridge that can attach to existing trace records and task events.

First useful fields:

- `code`,
- `severity`,
- `phase`,
- `component`,
- `node_id`,
- `node_name`,
- `message`,
- `cause`,
- `suggested_fix`,
- `source_file`,
- `source_symbol`,
- `raw_error`.

Then wire producers incrementally:

1. compiler/preflight diagnostics,
2. materializer/operator diagnostics,
3. training task diagnostics,
4. backend fallback diagnostics,
5. checkpoint/early-stop diagnostics,
6. crash/runtime diagnostics.

This should reuse the future `tofix26` error-code helpers, but `tofix32` should
reserve the debugger ingestion contract now so the UI and trace store are ready.

## Problem

CyxWiz failures are cross-cutting. A useful debugger must explain the chain, not
just show the last error.

Examples:

- A graph compiles but selected data columns are wrong.
- Arrow preprocessing changes table shape or schema unexpectedly.
- Tokenization creates columns the next node does not consume correctly.
- Pipeline materialization succeeds, but the debugger cannot show each operator
  transition.
- A backend placement warning happens far away from the UI node that caused it.
- Training later crashes, but the root cause was a data/schema/shape contract
  violation during setup.
- Export/import produces an artifact, but the debugger cannot correlate it to
  graph state, generated code, manifest contents, and runtime warnings.

The current debugger can explain compile, preflight, legacy text preprocessing,
synthetic Local Debug, text Smoke Run, runtime summaries, memory estimates, and
recommendations. It cannot yet prove operator-by-operator data flow through a
real Studio graph.

## Design Rule

Do not build a second execution framework.

The debugger must grow by wiring narrow trace producers into the existing
session and trace model:

- frozen Studio graph snapshot
- graph compiler and compiled graph plan
- `PipelineMaterializer` / preprocessing operators
- `PipelineExecutor` where runtime operators already exist
- `DebugOperatorTraceAdapter`
- `DebugGraphTraceExecutor`
- `DebugRunStore`
- `StudioDebuggerPanel`
- future `tofix26` error-code helpers

The debugger must stay an inspector:

- it can trace,
- it can explain,
- it can persist,
- it can recommend,
- it can export support bundles,
- it must not silently mutate the active graph,
- it must not become another training mode,
- it must not fake unsupported operator coverage.

## Target Architecture

### 1. Canonical Trace Spine

Keep `DebugTraceRecord` as the central event shape.

Each meaningful event should have:

- `run_id`,
- graph hash where available,
- node id/name/type where available,
- phase,
- role,
- input/output shape,
- dtype,
- backend,
- duration,
- status,
- structured payload,
- issues with stable error codes when available.

### 2. Trace Producers

Add producers one at a time. Each producer should emit canonical traces and be
usable by tests without the GUI.

Initial producers:

- graph snapshot producer: already present through session build,
- compile/preflight producer: present but should become code-aware,
- legacy text preprocessing producer: already present,
- smoke-run producer: already present,
- runtime training producer: already present through collector/recorder,
- operator-backed preprocessing producer: missing and should be first.

Future producers:

- generated-code/export correlation,
- import/package manifest trace,
- graph edit event trace,
- backend placement and fallback trace,
- memory pressure trace,
- kernel/timeline trace,
- optional Tracy bridge/export.

### 3. Explanation Engine

`DebugRecommendationEngine` should evolve into an explanation layer that
combines:

- trace status,
- issue severity,
- error code,
- node type,
- phase,
- backend status,
- shape/schema deltas,
- crash/runtime context,
- known recovery hints.

This should answer:

- What happened?
- Where did it happen?
- Why is CyxWiz confident?
- What changed from input to output?
- What should the user try next?
- Which subsystem owns the fix?

### 4. UI Surface

The current `StudioDebuggerPanel` is acceptable as a starting shell, but the
final debugger should become a command center.

Future UI direction:

- left rail: runs, graph snapshot, active lens, filters,
- center: timeline / graph-path trace view,
- right inspector: selected trace, node, payload, issues, recovery hints,
- bottom: raw payload/log/support-bundle details,
- lenses: Data, Compile, Runtime, Backend, Memory, Export, Crash, Studio Events,
- clear labels distinguishing real traces from synthetic/local/smoke traces.

Do not polish the UI before the trace producers are real.

### 5. Optional Tracy / External Profiler Layer

After internal traces are strong:

- add compile-time optional instrumentation around hot scopes,
- keep instrumentation macros no-op when disabled,
- start with CPU scopes around materialization, operator apply, compiler passes,
  training stages, batch creation, model forward/backward, export/import,
- only add GPU Tracy instrumentation when the backend can report it truthfully,
- consider exporting CyxWiz run timelines so external tooling can inspect them.

## Recommended Next Slice

Wire operator-backed Arrow preprocessing traces into normal Studio Debugger
runs.

Initial scope:

1. Detect when the active graph uses an Arrow dataset and supported
   preprocessing operators during a Studio Debugger run.
2. Execute only the preprocessing/materialization operator chain needed for the
   selected debug sample or bounded sample table.
3. For each supported operator transition, build a `DebugGraphTraceStep` with
   `DebugOperatorTraceAdapter`.
4. Convert those steps through `DebugGraphTraceExecutor`.
5. Append the resulting canonical traces to `session.traces`.
6. Persist them through `DebugRunStore`.
7. Keep existing legacy text traces and Smoke Run behavior intact.
8. Report unsupported operators as explicit warning traces, not silent gaps.
9. Where possible, attach first-pass error-code fields aligned with `tofix26`.

First useful operator:

- `TextTokenizerOperator`

Why start there:

- It exercises Arrow table input/output shape and schema tracing.
- It connects directly to current text pipeline debugging.
- `DebugOperatorTraceAdapter` already has contract coverage for Arrow table
  transitions.
- It avoids pretending every graph node is supported.

Possible helper:

- `DebugOperatorTraceProducer`

Possible responsibility:

- take frozen graph snapshot, links, dataset registry/materialization context,
  dataset name, selected sample/bounds, and run id,
- walk only supported preprocessing operators in selected graph order,
- execute through existing materializer/operator code,
- emit one canonical trace per supported operator,
- emit warning traces for unsupported operators,
- never mutate the active graph.

Likely files:

- `cyxwiz-engine/src/gui/main_window.cpp`
- `cyxwiz-engine/src/core/debug_operator_trace_adapter.h`
- `cyxwiz-engine/src/core/debug_operator_trace_adapter.cpp`
- `cyxwiz-engine/src/core/debug_graph_trace_executor.h`
- `cyxwiz-engine/src/core/debug_graph_trace_executor.cpp`
- new `cyxwiz-engine/src/core/debug_operator_trace_producer.*`
- `cyxwiz-engine/src/core/pipeline_materializer.*`
- `cyxwiz-engine/src/core/pipeline_executor.*`
- `cyxwiz-engine/src/core/debug_run_store.*`
- `cyxwiz-engine/tests/test_debugger_contracts.cpp`

## Acceptance Criteria For Next Slice

The next implementation slice is complete when:

- A Studio Debugger run on an Arrow text preprocessing graph emits an
  operator-backed trace for `TextTokenizerOperator`.
- The trace uses `cyxwiz.debug.node_trace.v1`.
- The trace includes input table shape, output table shape, input schema, output
  schema, operator name, backend, duration, status, run id, and graph/node
  context.
- The trace is visible through the existing Preprocessing/Shapes debugger
  lenses.
- The trace is persisted by `DebugRunStore`.
- Unsupported operator chains do not crash the debugger.
- Unsupported operator chains emit warning traces and do not claim full graph
  tracing.
- A deterministic test covers the path from debugger session construction to
  persisted/session trace record.
- First-pass issue payloads can carry or reserve error-code fields compatible
  with `tofix26`.

## Non-Goals For The Next Slice

Do not implement yet:

- full node-by-node execution for every graph node,
- full tensor value preview tables,
- image/audio/tabular/time-series trace producers,
- allocator-proven memory ownership,
- complete CPU/GPU kernel-level tracing,
- Tracy dependency/vendor integration,
- generated-code/export UI correlation,
- full Studio graph-edit event history,
- debugger UI redesign,
- automatic support upload.

## Follow-Up Queue

After the first operator-backed trace slice:

1. Add `tofix26` central error-code catalog and formatting helpers.
2. Attach structured error codes to compiler/preflight/debug trace issues.
3. Add more preprocessing operators one at a time.
4. Add backend placement enrichment to canonical node traces.
5. Add memory delta traces only where before/after snapshots are meaningful.
6. Integrate export/import correlation traces into debugger run history.
7. Integrate Windows crash import into the Runtime/Crash lens.
8. Add Studio event producers for graph edits, dataset selection, sample
   changes, export attempts, generated-code actions, and plugin actions.
9. Add optional internal timeline export for support bundles.
10. Evaluate optional Tracy compile-time instrumentation after the internal
    trace spine is stable.
11. Redesign the debugger UI around trace timeline, graph path, and explanation
    inspector once producers are real.

## Risk

The biggest risk is making the UI imply more capability than the engine has.

Avoid labels such as:

- `full graph trace`,
- `complete profiler`,
- `kernel profiler`,
- `memory ownership proven`,
- `Tracy integrated`,
- `understands everything`.

Use precise labels until coverage is real:

- `operator-backed preprocessing trace`,
- `legacy text sample trace`,
- `synthetic Local Debug trace`,
- `Smoke Run real-data trace`,
- `runtime training trace`,
- `estimated memory trace`,
- `backend placement trace`,
- `export correlation trace`,
- `optional external profiler export`.

## Bottom Line

CyxWiz needs a verbose debugger that can explain the whole system, but it must
be built from truthful trace producers, stable error codes, and persisted run
records.

The first implementation should not be UI polish or Tracy integration. It
should be a real `DebugOperatorTraceProducer` that emits canonical,
error-code-ready Arrow preprocessing traces for one supported operator path.

## Priority Clarification: Engine-First ML Debugger

The primary debugger mission is CyxWiz engine development and ML workflow
explainability.

The debugger should be optimized first for engineers who need to understand how
CyxWiz builds, validates, trains, and executes models. External profiler-style
features are useful later, but they are secondary. Tracy-style telemetry should
never displace the core ML/debugging view.

Primary focus areas:

- graph-to-training contract validation,
- selected training path and graph plan,
- data loading, schema, feature, label, and batch contracts,
- preprocessing transformations and shape/schema deltas,
- model layer construction and parameter counts,
- forward activation shapes and dtype flow,
- loss inputs, targets, reductions, class weights, label smoothing, and metrics,
- backward gradients, zero/NaN/Inf detection, and parameter update visibility,
- optimizer configuration and step behavior,
- CPU/GPU/backend placement and fallback reasons,
- recurrent/transformer path decisions and performance hotspots,
- memory pressure from tensors, batches, ArrayFire buffers, and model state,
- training loop stages, epoch/batch timing, throughput, and failures,
- checkpoint/export/import/package correlation where it affects training.

Secondary focus areas:

- Studio UI event history,
- support-bundle packaging,
- generated-code correlation,
- optional timeline export,
- optional Tracy/external profiler integration.

The debugger UI and trace model should answer ML-engine questions first:

- What graph path is actually being trained?
- What data shape/schema reached each stage?
- What model layers were built from the graph?
- Which tensors flowed through forward/backward?
- Which gradients were produced or missing?
- Which backend ran each stage and why?
- Which issue or error code explains the failure?
- What concrete change should the engineer try next?

This keeps `tofix32` centered on CyxWiz's core purpose: building and training
models truthfully and observably.

## Engine-First Debugger Extras

These additions are valuable after the first operator-backed trace producer is
working. They should stay focused on model building and training.

### 1. Training graph diff

Compare two debug runs and show what changed:

- graph nodes,
- graph links,
- node parameters,
- dataset selection,
- loss/metric/optimizer configuration,
- backend placement,
- generated training config.

Main question answered: why did this run behave differently from the previous
one?

### 2. Tensor lifecycle view

Track important tensors through forward/backward:

- origin node/layer,
- shape,
- dtype,
- backend/device,
- estimated bytes,
- consumers,
- retained/freed status where known,
- relation to batch/model/loss/gradient.

Do not claim allocator-proven lifecycle until allocator hooks exist.

### 3. Gradient health dashboard

Per trainable layer:

- gradient norm,
- zero-gradient flag,
- NaN/Inf flag,
- update magnitude,
- parameter norm,
- grad/parameter ratio,
- missing-gradient explanation.

Main question answered: is the model actually learning?

### 4. Loss and metric explainer

For each loss/metric node:

- expected prediction shape,
- expected target shape,
- actual shapes,
- class count,
- reduction mode,
- ignore index,
- label smoothing,
- class weights,
- metric threshold/top-k settings,
- why the result may be wrong or misleading.

### 5. Backend decision audit

For every node/layer/operator:

- intended backend,
- actual backend,
- fallback reason,
- estimated cost,
- unsupported reason,
- suggested fix.

This should reuse existing backend placement classification instead of creating
parallel logic.

### 6. Batch inspector

Show the first bounded debug batch:

- feature columns,
- label column,
- row count,
- tensor shape,
- sequence lengths,
- padding/mask summary,
- class balance,
- null/missing summary,
- dtype conversion summary.

### 7. Model construction trace

Show exactly how Studio graph nodes become engine layers:

- graph node -> model layer,
- parameter mapping,
- inferred input/output size,
- trainable parameter count,
- skipped/unsupported nodes,
- warnings from `ModelBuilder`.

### 8. Shape prophecy vs reality

Compare compiler/shape-inference predictions with runtime traces:

- predicted input/output shape,
- actual input/output shape,
- first mismatch node,
- upstream node that introduced the mismatch,
- likely recovery hint.

### 9. Slow path detector

Identify expensive stages:

- preprocessing operator time,
- batch creation time,
- data wait time,
- CPU fallback time,
- forward time,
- backward time,
- optimizer step time,
- export/import time where relevant.

### 10. Error-code timeline

Use `tofix26` codes to show when and where warnings/errors appeared:

- compile,
- preflight,
- data/materialization,
- model build,
- forward,
- loss,
- backward,
- optimizer,
- export/import,
- crash/runtime.

### 11. Run replay capsule

Persist enough metadata to reproduce or explain a run:

- graph snapshot,
- dataset reference,
- selected sample/batch bounds,
- compiled config,
- seed,
- backend selection,
- trace records,
- warnings/errors,
- environment summary.

The capsule should redact sensitive paths/data before support export.

### 12. Explain this node

Right-click any node and ask the debugger:

- what role it played,
- whether it was on the selected training path,
- what data reached it,
- what shape/dtype/backend it produced,
- what issues are attached,
- what to inspect next.

### 13. Training stall detector

Detect likely stall causes:

- loss not changing,
- gradients zero,
- learning rate too low/high,
- saturated activations,
- class imbalance,
- malformed labels,
- batcher not advancing,
- optimizer not updating parameters.

### 14. Numerics lens

Track bounded numeric summaries:

- activation min/max/mean,
- gradient min/max/mean,
- NaN/Inf,
- exploding values,
- dead ReLU/zero outputs,
- softmax saturation,
- logits/target mismatch.

Do not render full tensor dumps by default.

### 15. Export/import consistency check

After export/import/package actions, correlate:

- graph hash,
- manifest contents,
- tokenizer/assets,
- model parameters,
- expected input/output contract,
- package warnings,
- missing runtime/training assets.

## Priority Extras After First Trace Producer

Recommended order after `DebugOperatorTraceProducer`:

1. Model construction trace.
2. Shape prophecy vs reality.
3. Gradient health dashboard.
4. Backend decision audit.
5. Batch inspector.
6. Error-code timeline.
7. Slow path detector.

## Tracy / External Profiler UI-UX Boundary

CyxWiz should not reimplement Tracy-class CPU/GPU/memory profiling UI from
scratch.

Use CyxWiz debugger UI for ML semantics:

- graph node meaning,
- data/schema/shape contracts,
- model/loss/optimizer breakdown,
- gradients and training health,
- backend decisions and recovery hints,
- error-code explanations.

Use Tracy or Tracy-style external profiling for low-level timelines when the
user needs deep performance inspection:

- CPU zones,
- GPU zones,
- memory allocations,
- locks/contention,
- thread scheduling,
- frame/timeline navigation,
- hot-path flame/timeline analysis.

The UX should make this relationship clear:

- CyxWiz debugger explains ML-engine behavior.
- Tracy/external profiler explains low-level performance behavior.
- CyxWiz can link or export timing spans to external profiler views.
- CyxWiz should show summarized CPU/GPU/memory facts inline, but deep timeline
  exploration should use external profiler tooling where possible.

Possible UI affordances:

- `Open external profiler trace` button when a compatible trace exists.
- `Export timeline` action in a debug run.
- `Profiler correlation id` shown on CyxWiz traces and external spans.
- Inline `CPU/GPU/Memory summary` cards sourced from internal traces.
- Clear badge: `Internal ML trace` vs `External profiler timeline`.

Implementation guardrail:

- add optional instrumentation/export hooks only after internal canonical traces
  are stable,
- keep external profiler dependencies compile-time optional,
- keep no-op macros when disabled,
- never require Tracy to run the CyxWiz debugger,
- never show external profiler controls as if they are required for normal ML
  debugging.
