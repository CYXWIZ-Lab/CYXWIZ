# To Fix 32 - Studio Debugger Follow-Up

## Current Code Truth

The Studio Debugger has a real foundation, but it is not yet a complete
node-by-node graph debugger.

Already implemented:
- `StudioDebuggerPanel` renders debugger runs, lenses, trace filters, node
  details, runtime trace, memory trace, run history, and recommendations.
- `MainWindow::BuildStudioDebuggerSessionFromSnapshot` captures a frozen graph
  snapshot before async debugger execution starts.
- `DebugTraceRecord` and `DebugNodeTraceContract` define the canonical trace
  record shape.
- `DebugRunStore` persists debugger runs, traces, Studio events, issues, and
  recommendations.
- `TextPreprocessingTracer` emits selected-sample text traces for registered
  legacy text datasets.
- `SmokeRunExecutor` runs a small real-data text smoke path and records batch,
  loss, and gradient traces.
- `TrainingTraceCollector` and `CrashRunRecorder` feed the Runtime lens.
- Support-bundle redaction, backend classification helpers, memory ownership
  estimates, export correlation traces, Windows crash import parsing, and node
  inspector summaries have contract-level coverage.

Important limitation:
- `DebugGraphTraceExecutor` currently converts prepared
  `DebugGraphTraceStep` values into `DebugTraceRecord` values. It does not yet
  walk and execute the real Studio graph node by node.

This means the UI is ahead of the execution core. The next work should make
real operator-backed traces appear in normal Studio Debugger runs instead of
adding more UI polish.

## Problem

The debugger can explain compile, preflight, text preprocessing, synthetic
Local Debug, text Smoke Run, and live training/runtime summaries. It cannot yet
prove the exact operator-by-operator data flow through a real Studio graph.

That gap matters because many Studio failures happen before the model layer:
- Arrow table preprocessing changes shape or schema.
- Tokenization creates unexpected columns.
- Feature/label columns are selected incorrectly.
- Pipeline materialization succeeds, but the debugger does not show each
  operator transition.
- Runtime traces show symptoms, but not the exact graph node that transformed
  the data into the failing shape.

## Design Rule

Do not build a second execution framework.

The debugger should grow by wiring narrow trace producers into the existing
session and trace model:
- frozen Studio graph snapshot
- existing graph compiler and materializer
- existing preprocessing operators
- `DebugOperatorTraceAdapter`
- `DebugGraphTraceExecutor`
- existing `StudioDebuggerPanel`

The debugger must stay an inspector:
- it can trace
- it can explain
- it can persist
- it can recommend
- it must not silently mutate the active graph
- it must not become another training mode

## Recommended Next Slice

Wire operator-backed Arrow preprocessing traces into normal Studio Debugger
runs.

Initial scope:
1. Detect when the active graph uses an Arrow dataset and preprocessing
   operators during a Studio Debugger run.
2. Execute only the preprocessing operator chain needed for materialization.
3. For each operator transition, build a `DebugGraphTraceStep` with
   `DebugOperatorTraceAdapter`.
4. Convert those steps through `DebugGraphTraceExecutor`.
5. Append the resulting canonical traces to `session.traces`.
6. Keep existing text legacy traces and Smoke Run behavior intact.
7. Add deterministic contract coverage proving operator-backed traces appear in
   the saved/debugger session.

First useful operator:
- `TextTokenizerOperator`

Why start there:
- It already has contract coverage through `DebugOperatorTraceAdapter`.
- It exercises Arrow table input/output shape and schema tracing.
- It connects directly to the current text pipeline debugging need.
- It avoids pretending that all graph nodes are supported.

## Implementation Notes

Likely files:
- `cyxwiz-engine/src/gui/main_window.cpp`
- `cyxwiz-engine/src/core/debug_operator_trace_adapter.h`
- `cyxwiz-engine/src/core/debug_operator_trace_adapter.cpp`
- `cyxwiz-engine/src/core/debug_graph_trace_executor.h`
- `cyxwiz-engine/src/core/debug_graph_trace_executor.cpp`
- `cyxwiz-engine/src/core/pipeline_materializer.*`
- `cyxwiz-engine/tests/test_debugger_contracts.cpp`

Preferred approach:
- Keep `DebugGraphTraceExecutor` simple. It should remain the converter from
  trace steps to records.
- Add the real graph/operator walking logic outside that converter, near the
  existing Studio Debugger orchestration or a small dedicated producer.
- Do not duplicate preprocessing execution logic that already exists in
  `PipelineMaterializer`.
- Do not add a generic "execute every node" abstraction until one real operator
  chain is traced end-to-end.

Possible helper name:
- `DebugOperatorTraceProducer`

Possible responsibility:
- take frozen nodes, links, data registry, dataset name, and run id
- apply supported preprocessing operators in graph order
- emit one canonical trace per supported operator
- report unsupported operators honestly as warnings, not fake success traces

## Acceptance Criteria

The next slice is complete when:
- A Studio Debugger run on an Arrow text preprocessing graph emits an
  `OperatorTransform` trace for `TextTokenizerOperator`.
- The trace includes input table shape, output table shape, input schema,
  output schema, operator name, backend, duration, status, and run id.
- The trace uses `cyxwiz.debug.node_trace.v1`.
- The trace is visible through the existing Preprocessing and Shapes lenses.
- The trace is persisted by `DebugRunStore`.
- A deterministic test covers the full path from debugger session build to
  persisted/session trace record.
- Unsupported operator chains do not crash the debugger and do not claim full
  graph tracing.

## Non-Goals For This Slice

Do not implement yet:
- full node-by-node execution for every graph node
- image, audio, tabular, or time-series trace producers
- full tensor value preview tables
- per-node allocator-proven memory ownership
- complete CPU/GPU kernel-level tracing
- generated-code/export UI correlation
- full Studio graph-edit event history
- new debugger UI layout polish

## Follow-Up Queue

After the first operator-backed trace slice:
1. Add more preprocessing operators one at a time.
2. Add backend placement enrichment to canonical node traces.
3. Add memory delta traces only where before/after snapshots are meaningful.
4. Integrate export correlation traces into the Studio Debugger run history.
5. Integrate Windows crash import into the Runtime lens.
6. Add Studio event producers for graph edits, dataset selection, sample
   changes, export attempts, and generated-code actions.
7. Add a true selected-node inspector backed by canonical traces, not only UI
   rendering logic.

## Risk

The biggest risk is making the UI imply more capability than the engine has.

Avoid labels such as "full graph trace" until every real graph node in the
active path is being traced. Use precise labels instead:
- "operator-backed preprocessing trace"
- "legacy text sample trace"
- "synthetic Local Debug trace"
- "Smoke Run real-data trace"
- "runtime training trace"

This keeps the debugger trustworthy while the execution coverage expands.
