# 46) Studio Debugger trace visibility and filtering contracts

## 46.1 Scope and boundary

This section documents the UI-side contracts for trace visibility and operator interaction in the Studio Debugger panel.

Coverage:

- tracing lens taxonomy and predicate definitions
- attention-only mode and keyword search filtering semantics
- trace table rendering, selection, and focus interactions
- run history restore path for saved trace sessions
- assistant-context export contracts derived from current UI selection

This section documents presentation contracts only; trace payload production remains in core executor contracts and previous sections.

## 46.2 UI state model used by filter workflow

`StudioDebuggerPanel` owns these filter and selection fields in `studio_debugger_panel.h`:

- `StudioDebuggerLens active_lens_`
- `StudioDebuggerRunMode run_mode_`
- `char trace_search_[128]`
- `bool trace_attention_only_`
- `int selected_trace_index_`

`StudioDebuggerLens` enumeration values:

```text
Overview
Preprocessing
Shapes
Values
Gradients
Runtime
StudioEvents
Recommendations
```

`StudioDebuggerRunMode` values:

```text
FullWorkflow
Preflight
LocalDebug
SmokeRun
RuntimeTrace
```

The same panel also stores per-run snapshot state:

- `StudioDebuggerSnapshot session_`
- `std::vector<DebugTraceRecord> session_.traces`
- `std::vector<DebugTraceRecord> session_.traces` is the canonical input to the modern unified timeline contract

## 46.3 Trace schema preconditions and lens contracts

`DebugTraceRecord` is the filter input with fields:

- `phase`, `role`, `node_id`, `node_name`, `node_type`, `status`, `dtype`,
  `input_shape`, `output_shape`, `duration_ms`, `issues`, `payload`
- `role` is one of:
  - `RawInput`, `PreprocessingOutput`, `FeatureTensor`, `Activation`, `Gradient`,
    `Prediction`, `Target`, `Loss`, `Warning`, `Error`, and others.

Lens semantics in `StudioDebuggerPanel`:

```text
Overview       => all traces (no filtering)
Preprocessing  => trace has preprocessing roles OR phase contains text preprocessing markers
Shapes         => shape-bearing traces OR shape mismatch statuses OR shape fields in payload
Values         => activation/prediction/target/loss-like roles OR payload fields:
                 loss, average_loss, token_ids_preview
Gradients      => gradient role OR backward phase OR status is zero OR status is nan
Runtime        => duration > 0 OR phase mentions SmokeRun/Train OR status warning/failed
StudioEvents   => role == StudioEvent
Recommendations=> statuses/warning/error-like classes + warning/error roles
```

## 46.4 Attention contract

`IsAttentionTrace()` is the shared predicate for “attention mode”:

- status class attention: `failed`, `warning`, `blocked`, `shape_mismatch`, `zero`, `nan`
- role class attention: `Warning`, `Error`
- issue presence (`!issues.empty()`)
- payload attention counters:
  - numeric `warning_count > 0`
  - numeric `error_count > 0`

## 46.5 Workflow filter contract

`TraceMatchesWorkflowFilter(trace)` enforces the full filter stack in order:

1. `TraceMatchesActiveLens(trace)` must pass
2. if `trace_attention_only_ == true`, must also satisfy `IsAttentionTrace(trace)`
3. if search query is non-empty, case-insensitive substring match must pass against:
   - phase, role-name, node name, node type, status, dtype, and serialized payload JSON

`ContainsIgnoreCase(query="", haystack=anything)` is a deliberate identity true (empty query means no search constraint).

Filter counters in UI show:

- `%d/%d visible`
- `%d attention`
- where `visible_count` is lens+search(+attention) matches and `lens_count` is lens-only matches.

## 46.6 Toolbar and control contract

Render-time controls include:

- run mode combo (5 modes), run/cancel button state tied to `run_in_progress_`
- lens combo (8 lenses), which clears selected index (`selected_trace_index_ = -1`) on change
- sample input integer
- search box bound to `trace_search_` (`Get/Set` directly into fixed char array)
- “Attention only” toggle (`trace_attention_only_`)
- “Clear filters” resets both search and attention flag

## 46.7 Trace timeline and visibility contract

`RenderTraceTimeline()` has two rendering branches:

- Modern branch when `session_.traces` is non-empty:
  - render label: `"<phase>  <role>  <node_name_or_(graph)>"` plus optional `[node N]`
  - row style by status:
    - `ok/passed`: green
    - `warning/zero/shape_mismatch`: amber
    - `failed/nan`: red
    - default: neutral
  - rows are clickable; click sets `selected_trace_index_` and invokes focus callback if `node_id >= 0`
  - if no trace survives filter, render: `"No traces match the active lens and filters."`
- Legacy fallback when `session_.traces` is empty:
  - timeline shows `debug_result.layer_traces` with shape-mismatch/nan highlighting and tooltips.

## 46.8 Selection context and detail contracts

Selection contract:

- `GetSelectedTraceIdForAssistant()` returns `""` when no valid selected trace.
- otherwise returns `"<run_id>:<selected_trace_index>"`.

`RenderSelectedTraceDetails()` on modern traces shows:

- identity block: node, node id, type, phase, role, status, input/output shape, dtype, duration
- focus button when `node_id >= 0`
- issues list with error code and issue message
- related recommendations filtered to same `node_id`, or generic issues when status is not ok/passed and `node_id < 0`
- diagnosis block and optional raw payload JSON tree
- preprocessing payload inspector gated by preprocessing role predicate

## 46.9 Run-history contract and session reload path

Session initialization path:

- `SetSession`:
  - stamps active and “current” snapshots
  - loads crash heartbeat + latest training trace if available
  - loads recent runs via `DebugRunStore::ListRecent(8)`
  - sets selected index to first trace when `session_.traces` exists, else `-1`
- `LoadStoredRun(run_id)`:
  - loads `DebugRunStore::Load(run_id)` and replaces current session fields
  - repopulates issues/traces/studio events/recommendations/run history
  - resets selected trace index to first entry when traces exist
- `RenderRunHistory` renders each run label with:
  - pass/attention prefix, `(current)` marker, timestamp, and summary counts
  - click action calls `LoadStoredRun`

This creates a deterministic UI contract for recalling historical trace sets and continuing exploration from a persistent run record.

## 46.10 Assistant context contract

Assistant-facing context is generated from UI state (`schema v1`):

- `BuildAssistantDebuggerContextJson()`:
  - top-level fields: run metadata, counts, `selected_trace_index`, `active_lens`
  - optional selected trace block with canonical fields and up to 5 issue items
  - schema: `cyxwiz.assistant.debugger_context.v1`
- `BuildAssistantTrainingContextJson()`:
  - training summary fields (run/status/stage/epoch/batch/loss/accuracy)
  - falls back to `TrainingTraceCollector::LoadLastTrace()` if current snapshot unavailable
  - schema: `cyxwiz.assistant.training_context.v1`

## 46.11 Contracts in ASCII

```text
Modern trace UI pipeline

StudioDebuggerSnapshot(traces)
  -> active_lens_ / run mode / search / attention flag
  -> TraceMatchesActiveLens
  -> TraceMatchesWorkflowFilter
  -> RenderTraceTimeline (row list + selection)
  -> selected_trace_index_
  -> RenderSelectedTraceDetails + BuildAssistantDebuggerContextJson
```

```text
Run history recall path

Run history row selected
  -> LoadStoredRun(run_id)
  -> session_.traces / issues / studio_events / recommendations
  -> selected_trace_index_ reset
  -> timeline + selected detail contract recomputed
```

```text
Attention and search evaluation order

Trace + current lens
  => active lens predicate
     => optional attention predicate
        => optional containsIgnoreCase query predicate
           => visible row (if true) / hidden row (if false)
```

## 46.12 Evidence anchors

| Claim | Source |
|---|---|
| Lens enum, state fields, and method surface | `cyxwiz-engine/src/gui/panels/studio_debugger_panel.h:39-98` |
| Lens definitions and attention predicate semantics | `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:114-171` |
| Active lens dispatch and workflow filtering | `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:598-675` |
| Search box + attention checkbox + clear semantics | `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:662-688` |
| Trace and trace-color rendering + selection callback behavior | `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:1858-1899` |
| Legacy layer trace fallback timeline | `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:1900-1938` |
| Selected trace details, related recommendations, payload inspector | `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:2197-2291` |
| Assistant debugger context schema and selected-trace payload | `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:350-411` |
| Assistant training context schema and terminal extraction fallback | `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:413-443` |
| Session init, set-session selection init, run-history load/listing | `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:332-357`, `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:741-762`, `cyxwiz-engine/src/gui/panels/studio_debugger_panel.cpp:1080-1120` |
| Trace schema baseline (`DebugTraceRecord`, `DebugTraceRole`) used by UI filters | `cyxwiz-engine/src/core/debug_trace_record.h:12-55`, `cyxwiz-engine/src/core/debug_trace_record.h:155-175` |
| Run store summary shape used by run-history rows | `cyxwiz-engine/src/core/debug_run_store.h:13-22`, `cyxwiz-engine/src/core/debug_run_store.h:34-37` |
