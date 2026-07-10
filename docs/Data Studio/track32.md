# track32 - Studio Debugger Telemetry Spine

## Status

First slice implemented and resumed hardening pass completed.

## Ticket Read

`tofix32` is a large debugger/telemetry-spine ticket. The first implementation
slice should not redesign the UI or build a second executor. The codebase
already has:

- canonical `DebugTraceRecord` schema: `cyxwiz.debug.node_trace.v1`;
- `DebugGraphTraceExecutor` for converting prepared trace steps;
- `DebugOperatorTraceAdapter` for Arrow table input/output transitions;
- `DebugRunStore` persistence;
- existing Studio Debugger lenses that can render preprocessing and shape traces.

The main missing piece for the recommended slice is a real producer that runs a
bounded operator-backed preprocessing transition and emits canonical traces.

## Lean Scope

Implement only the first operator-backed preprocessing trace path:

- support `TextTokenizer` only;
- execute through existing `TextTokenizerOperator`;
- use existing `DebugOperatorTraceAdapter` and `DebugGraphTraceExecutor`;
- emit warning traces for unsupported operators or invalid inputs;
- avoid full graph execution, UI redesign, Tracy integration, and broad
  diagnostic catalogs.

## Planned Files

- `cyxwiz-engine/src/core/debug_operator_trace_producer.h`
- `cyxwiz-engine/src/core/debug_operator_trace_producer.cpp`
- `cyxwiz-engine/src/gui/main_window.cpp`
- `cyxwiz-engine/tests/test_debugger_contracts.cpp`
- CMake wiring for the new source if required.

## Acceptance For This Slice

- A `TextTokenizer` Arrow table transition produces a canonical node trace.
- The trace includes input/output shapes and schemas through the existing
  adapter.
- Unsupported operators emit warning traces instead of silent gaps.
- A focused debugger contract test covers the producer path.
- Existing debugger persistence remains unchanged and stores the new traces when
  they are appended to a session.

## Implemented Slice

- Added `DebugOperatorTraceProducer`.
- Executes real `TextTokenizerOperator` against a bounded Arrow table input.
- Converts the input/output table transition through `DebugOperatorTraceAdapter`
  and `DebugGraphTraceExecutor`.
- Appends operator-backed preprocessing traces to Studio Debugger sessions when
  the compiled dataset is available as an Arrow dataset.
- Emits warning traces for unsupported non-folded downstream operators, including a first-pass `error_code` payload.
- Added focused debugger contract coverage for success, unsupported warning, and
  persistence through `DebugRunStore`.

## Validation

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`

## Known Dirty Tree Boundary

Before this work, the tree already had unrelated dirty changes around
pre-existing ticket work, `track43.md`, and several docs renames. This track file
and the `tofix32` implementation should avoid changing those unrelated edits
except where CMake wiring must work with the current file state.

## Resume 2026-07-09 - Bounded Debugger Trace Hardening

Follow-up on the first slice:

- `DebugOperatorTraceProducer` now selects the `DataInput` / `DatasetInput` that
  matches the compiled `dataset_name` when Studio provides one, instead of
  blindly starting from the first data source in the graph.
- Studio Debugger now passes `config.dataset_name` and the selected sample index
  into the operator-backed preprocessing trace producer.
- Operator-backed debug tracing now runs on a bounded Arrow row window starting
  at the selected sample index, with a default cap of 32 rows. This avoids using
  the full registered table as a debugger input.
- Trace payloads now preserve the original source size and row-window metadata:
  `source_rows`, `source_columns`, `selected_sample_index`, `debug_row_offset`,
  `debug_row_count`, `debug_row_limit`, and `bounded_debug_table`.
- Contract coverage now proves bounded row-window tracing and named dataset
  source-node selection.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
## Resume 2026-07-09 - Folded Text Config Parity

Second follow-up on the first slice:

- `DebugOperatorTraceProducer` now folds `TextVocabulary` and `TextPadding`
  parameters with the same explicit mapping used by the Arrow table
  materializer path.
- `TextVocabulary.min_freq` now reaches `TextTokenizerOperator` as
  `min_word_freq` during debugger tracing.
- `TextPadding.max_length` and `pad_value` now shape the tokenizer-backed trace
  without emitting separate fake operator traces for folded config nodes.
- Contract coverage now proves folded vocabulary/padding nodes alter the single
  tokenizer trace instead of producing separate unsupported warnings.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
## Resume 2026-07-09 - Trace Graph Topology Guard

Third follow-up on the first slice:

- `DebugOperatorTraceProducer` now validates the supported trace topology before
  running `TextTokenizerOperator`.
- Cyclic paths reachable from the selected data source now emit a warning trace
  instead of silently producing partial traces.
- Branched paths with more than one supported tokenizer-backed trace branch now
  emit a warning trace instead of implying a truthful linear materialization
  trace.
- Unsupported non-folded downstream nodes still emit explicit warning traces;
  this guard only blocks shapes where a supported operator-backed trace would be
  misleading.
- Contract coverage now proves branch and cycle topologies are warning-only and
  do not claim `operator_backed` execution.

Additional validation:

- Passed on rerun after an initial timeout: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`

Build note:

- The full engine build re-ran CMake because `cyxwiz-engine/CMakeLists.txt` was
  newer than the generated stamp, and it compiled the unrelated dirty-tree
  `materialization_cache.cpp` file already present in the workspace.

## Resume 2026-07-09 - Graph Source Warning Traces

Fourth follow-up on the first slice:

- `DebugOperatorTraceProducer` no longer silently returns no traces when it
  cannot find a graph data source.
- Missing `DataInput` / `DatasetInput` now emits a graph-level canonical warning
  trace with node id `-1`.
- A compiled/requested dataset name that does not match any graph source now
  emits a graph-level warning trace preserving `source_dataset_name`.
- These warning traces carry the same `DebugOperatorTraceProducer`,
  `operator_backed=false`, `diagnostic_phase=graph_walk`, issue, and error-code
  payload shape as other unsupported debugger trace gaps.
- Contract coverage now proves missing and mismatched graph-source paths are
  warning-only instead of silent.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
## Resume 2026-07-09 - Supported-Trace Topology Scope

Fifth follow-up on the first slice:

- The trace topology guard now runs only when a supported tokenizer-backed trace
  operator is reachable from the selected source.
- Unsupported-only graph cycles are no longer masked by the topology guard; they
  continue to emit the normal unsupported-node warning trace.
- This keeps the guard aligned with its purpose: prevent misleading
  operator-backed traces, not suppress unsupported topology diagnostics.
- Contract coverage now proves an unsupported-only cycle still attaches the
  warning to the unsupported node and does not report a graph-walk topology
  failure.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`

## Resume 2026-07-09 - Failed Operator Stops Downstream Trace

Sixth follow-up on the first slice:

- `DebugOperatorTraceProducer` now treats a failed supported operator trace as
  a traversal boundary.
- When `TextTokenizerOperator` fails during configure/apply, the debugger emits
  the tokenizer warning trace and does not enqueue downstream nodes with a
  missing or fabricated output table.
- Downstream unsupported-node warnings still appear after successful tokenizer
  traces, preserving truthful partial coverage for `DataInput -> TextTokenizer
  -> unsupported` chains.
- Contract coverage now proves `DataInput -> invalid TextTokenizer -> Dense`
  emits only the tokenizer configure warning and does not invent a downstream
  Dense warning.

Additional validation:

- Initial build attempt timed out before completion: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed on rerun: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`

## Resume 2026-07-09 - Missing Arrow Source Table Warning

Seventh follow-up on the first slice:

- `DebugOperatorTraceProducer` now validates source-table availability after it
  resolves the graph `DataInput` / `DatasetInput` node.
- A resolved source node with no Arrow table now emits one graph-level warning
  trace instead of falling through to tokenizer execution.
- The warning uses `diagnostic_phase=data_source`, preserves
  `source_dataset_name`, and keeps the bounded debug table metadata at zero
  rows/columns when no table exists.
- Existing missing-source and mismatched-source graph-walk warnings keep their
  current `diagnostic_phase=graph_walk` behavior.
- Contract coverage now proves `DataInput -> TextTokenizer` with a null Arrow
  source emits only the source-table warning.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`

## Resume 2026-07-09 - Folded Config Without Tokenizer Warning

Eighth follow-up on the first slice:

- `DebugOperatorTraceProducer` now detects the case where traversal reaches
  downstream text config nodes but emits no traces because no `TextTokenizer`
  operator is present.
- `DataInput -> TextPadding` / folded-config-only shapes now emit one source-node
  warning instead of silently returning no operator-backed trace records.
- Data-only graphs remain quiet; the warning only appears when the selected
  source has downstream nodes but those nodes never produce a trace or
  unsupported-node warning.
- Existing folded config behavior remains unchanged when the config nodes can be
  folded into a reachable `TextTokenizer` trace.
- Contract coverage now proves a folded-text-config-only graph reports the
  missing tokenizer operator.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`

## Resume 2026-07-09 - Trace-Relevant Cycle Validation

Ninth follow-up on the first slice:

- The topology cycle guard now follows only branches that can reach a supported
  operator-backed trace operator.
- Unsupported-only side cycles no longer suppress a valid tokenizer trace when
  a supported tokenizer branch exists elsewhere from the selected source.
- Supported trace cycles still emit the graph-walk topology warning, preserving
  the guard against misleading cyclic tokenizer-backed traces.
- Normal unsupported-node warnings still attach to unsupported cyclic side
  branches through the traversal path.
- Contract coverage now proves `DataInput -> TextTokenizer` plus an unsupported
  self-cycling side branch emits the tokenizer trace and the unsupported-node
  warning, not a graph-level topology failure.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`

## Resume 2026-07-09 - Unsupported Operators Are Trace Boundaries

Tenth follow-up on the first slice:

- `DebugOperatorTraceProducer` now treats unsupported non-folded operators as
  traversal boundaries.
- A path such as `DataInput -> Dense -> TextTokenizer` now emits the Dense
  unsupported warning and does not run `TextTokenizer` against the unchanged
  upstream table.
- Supported-trace reachability now uses the same boundary rule, so unsupported
  side paths that eventually point at a tokenizer do not make valid direct
  tokenizer paths look like branched supported topology.
- Folded text config nodes still remain traversable because they do not execute
  independently and can legitimately feed tokenizer configuration.
- Contract coverage now proves unsupported-before-tokenizer paths stop at the
  unsupported node, while a valid direct tokenizer branch still traces when an
  unsupported side path also points at the tokenizer.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`

## Resume 2026-07-09 - Effective Tokenizer Config Payload

Eleventh follow-up on the first slice:

- `DebugOperatorTraceProducer` now attaches an
  `effective_text_tokenizer_config` payload to tokenizer-backed traces.
- The payload records the effective tokenizer fields after folded
  `TextVocabulary` / `TextPadding` params are applied, including `text_col`,
  `label_col`, `tokenizer_type`, `max_length`, `lowercase`, `min_word_freq`,
  `max_vocab_size`, `pad_value`, and `vocab_build_if_missing` when present.
- Tokenizer traces now also include `folded_text_config_applied` so the debugger
  can distinguish plain tokenizer params from graph-folded config.
- Configure/apply warning traces from `TextTokenizerOperator` also preserve the
  same effective config payload, which makes invalid config warnings easier to
  inspect.
- Raw `vocab_file` paths are not stored in the payload; the trace only records
  `vocab_file_configured` as a boolean.
- Contract coverage now proves plain tokenizer traces, folded config traces, and
  invalid tokenizer warnings expose the effective config used by the operator.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Initial engine build timed out once, then a rerun compiled the executable but
  failed in the post-build resource copy step.
- Passed on retry: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`

## Resume 2026-07-09 - Folded Config Provenance Payload

Twelfth follow-up on the first slice:

- `DebugOperatorTraceProducer` now records `folded_text_config_nodes` on
  tokenizer-backed traces.
- The provenance payload contains only folded config node id, name, and type,
  so the debugger can show which `TextVocabulary` / `TextPadding` nodes shaped
  the effective tokenizer config without copying raw parameter maps or paths.
- `folded_text_config_applied` now reflects actual folded-node provenance rather
  than only comparing parameter maps.
- Plain tokenizer traces and plain tokenizer configure warnings expose an empty
  folded provenance array.
- Contract coverage now proves folded tokenizer traces preserve the vocabulary
  and padding config node identities in traversal order.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`

## Resume 2026-07-09 - Folded Config Contributed Keys

Thirteenth follow-up on the first slice:

- `folded_text_config_nodes` now includes `contributed_keys` for each folded
  config node.
- `TextVocabulary` provenance reports effective keys such as `min_word_freq`,
  `max_vocab_size`, `vocab_file_configured`, and `vocab_build_if_missing` when
  those values are contributed.
- `TextPadding` provenance reports effective keys such as `max_length` and
  `pad_value`.
- This lets Studio Debugger explain not only which folded config nodes shaped a
  tokenizer trace, but which effective tokenizer settings each node supplied.
- Contract coverage now proves vocabulary and padding provenance includes the
  expected contributed effective keys.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`

## Resume 2026-07-09 - Source Node Provenance Payload

Fourteenth follow-up on the first slice:

- `DebugOperatorTraceProducer` now annotates resolved-source traces with source
  graph node provenance.
- Traces now include `source_node_id`, `source_node_name`, and
  `source_node_type` when a `DataInput` / `DatasetInput` source node has been
  selected.
- If the source node carries a dataset name, traces also include
  `source_node_dataset_name`.
- Missing-source graph warnings remain graph-level without source-node fields;
  null-table warnings after source resolution do include source-node
  provenance.
- Contract coverage now proves plain tokenizer traces, named dataset selection,
  and unavailable Arrow source warnings preserve the selected source node.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`

## Resume 2026-07-09 - Skip Empty Folded Config Provenance

Fifteenth follow-up on the first slice:

- `DebugOperatorTraceProducer` no longer records folded config provenance for
  `TextVocabulary` / `TextPadding` nodes that contribute no effective tokenizer
  settings.
- Empty folded config nodes still remain traversable, so meaningful downstream
  folded config can still be discovered.
- `folded_text_config_applied` now stays false when the tokenizer is followed
  only by config nodes with no effective parameters.
- Contract coverage now proves an empty `TextPadding` after `TextTokenizer`
  preserves the tokenizer trace, leaves output shape unchanged, and emits no
  folded config provenance.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`

## Resume 2026-07-09 - Readable Unsupported Node Type Names

Sixteenth follow-up on the first slice:

- `DebugOperatorTraceProducer` now maps `Dense` to a readable node type name in
  canonical warning traces.
- Unsupported Dense warnings no longer expose the numeric enum value as
  `node_type`.
- Contract coverage now proves both downstream unsupported Dense warnings and
  unsupported-before-tokenizer boundary warnings preserve `node_type=Dense`.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`

## Resume 2026-07-09 - Selected Sample Clamp Metadata

Seventeenth follow-up on the first slice:

- `DebugOperatorTraceProducer` now records `selected_sample_clamped` in trace
  payloads.
- In-range selected samples report `selected_sample_clamped=false`.
- Out-of-range selected samples preserve the requested `selected_sample_index`,
  expose the actual clamped `debug_row_offset`, and report
  `selected_sample_clamped=true`.
- Contract coverage now proves a request for sample `99` on a three-row table
  traces the final available row while preserving the requested sample index.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`

## Resume 2026-07-09 - Selected Sample Availability Metadata

Eighteenth follow-up on the first slice:

- `DebugOperatorTraceProducer` now records `selected_sample_available` in trace
  payloads.
- In-range selected samples report `selected_sample_available=true` and
  `selected_sample_clamped=false`.
- Out-of-range selected samples report `selected_sample_available=false` and
  `selected_sample_clamped=true` while still tracing the clamped row window.
- Empty source tables report `selected_sample_available=false` and
  `selected_sample_clamped=false`, because no source row exists to clamp to.
- Contract coverage now proves all three cases: in-range bounded row, clamped
  out-of-range row, and empty source table.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`

## Resume 2026-07-09 - Unsupported Operator Diagnostic Phase

Nineteenth follow-up on the first slice:

- Unsupported non-folded operator warnings now carry
  `diagnostic_phase=unsupported_operator`.
- The debugger contract no longer has to infer unsupported-node gaps only from
  warning message text.
- Contract coverage now proves both downstream unsupported Dense warnings and
  unsupported-before-tokenizer boundary warnings preserve the explicit phase.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`

## Resume 2026-07-09 - Topology Warning Phase Contracts

Twentieth follow-up on the first slice:

- Tightened debugger contract coverage for topology warning phases.
- Cyclic supported trace paths now explicitly prove
  `diagnostic_phase=graph_walk`.
- Unsupported-only and mixed unsupported cycles now explicitly prove
  `diagnostic_phase=unsupported_operator`.
- No production code change was needed; the producer already emitted the
  intended structured phase values.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`

## Resume 2026-07-09 - Source Availability Error Codes

Twenty-first follow-up on the first slice:

- Source availability warnings from `DebugOperatorTraceProducer` now use
  `CW-R-0301` / `Runtime::InputDatasetMissing` instead of the generic runtime
  execution failure code.
- The retagged warnings cover missing graph source nodes, requested dataset
  name mismatches, and resolved source nodes that do not provide an Arrow table.
- Payload `error_code` and the first trace issue `error_code` are kept in sync
  for each source availability warning.
- Contract coverage now proves all three source availability paths preserve the
  missing-input code while keeping their existing diagnostic phases.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`

## Resume 2026-07-09 - Unsupported Operator Error Codes

Twenty-second follow-up on the first slice:

- Unsupported non-folded operator warnings from `DebugOperatorTraceProducer` now
  use `CW-R-0201` / `Runtime::UnsupportedNode` instead of the generic runtime
  execution failure code.
- `BuildWarningTrace` now accepts a caller-provided error code while preserving
  `Runtime::ExecutionFailed` as the default for real operator configure/apply
  failures.
- Payload `error_code` and the first trace issue `error_code` are kept in sync
  for unsupported operator warnings.
- Contract coverage now proves unsupported-only cycles, mixed unsupported
  cycles, downstream unsupported Dense warnings, unsupported-before-tokenizer
  boundary warnings, and unsupported side paths preserve the unsupported-node
  code.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`

## Resume 2026-07-10 - Aggregate Issue Code Summary

Twenty-third follow-up on the first slice:

- Added `DebugNodeTraceContract::AttachIssueSummary` as a small reusable helper
  for aggregate debugger traces that carry multiple `ValidationIssue` records.
- Compile and preflight traces now preserve `issue_count`, severity counts,
  unique `issue_codes`, `primary_error_code`, and `primary_warning_code` in the
  trace payload while keeping the existing issue list unchanged.
- This gives Studio Debugger and support bundles a structured code summary for
  compiler/preflight diagnostics without parsing free-form messages or widening
  graph execution.
- Contract coverage now proves preflight issue summaries preserve total counts,
  severity counts, first error code, and unique first-seen code order.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Initial full engine build attempt timed out before completion.
- Passed on rerun: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`

## Resume 2026-07-10 - Aggregate Diagnostic Context Payload

Twenty-fourth follow-up on the first slice:

- Added `DebugNodeTraceContract::AttachDiagnosticContext` for aggregate traces
  that need structured owner metadata without creating a broad diagnostic
  framework.
- Compile and preflight traces now carry `diagnostic_phase`, `component`,
  `source_file`, and `source_symbol` payload fields while preserving their
  existing trace shape and issue lists.
- This makes compiler/preflight trace ownership inspectable by Studio Debugger
  and support bundles without parsing the trace name or free-form summary text.
- Contract coverage now proves the context helper preserves phase, component,
  source file, and source symbol.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`

## Resume 2026-07-10 - Text Preprocessing Diagnostic Context

Twenty-fifth follow-up on the first slice:

- Text preprocessing success traces now carry the same structured diagnostic
  context fields used by aggregate compiler/preflight diagnostics.
- Missing, empty, out-of-range, and materialization failure traces now identify
  their diagnostic phase, component, source file, and source symbol without
  adding a new trace producer or changing issue/error-code shape.
- Contract coverage now proves tokenizer output and missing-dataset failures
  expose the text preprocessing diagnostic context.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: scoped `git diff --check` for the tofix32 files.

## Resume 2026-07-10 - Text Preprocessing Issue Summary

Twenty-sixth follow-up on the first slice:

- Text preprocessing traces that already carry issues now attach the aggregate
  issue summary payload used by compiler/preflight diagnostics.
- Missing, empty, out-of-range, vocabulary-coverage, truncation, and
  materialization-failure traces now expose issue counts, severity counts,
  first-seen issue codes, and primary error/warning codes without changing the
  existing issue list.
- Contract coverage now proves missing-dataset errors and truncation warnings
  expose issue summary fields.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: scoped `git diff --check` for the tofix32 files.

## Resume 2026-07-10 - Operator Warning Issue Summary

Twenty-seventh follow-up on the first slice:

- Operator-backed debugger warning traces now attach the same aggregate issue
  summary payload as compiler/preflight and text preprocessing diagnostics.
- Graph-level warnings and node-level operator warnings now expose issue counts,
  severity counts, first-seen issue codes, and primary warning codes without
  changing their existing issue list or warning role.
- Contract coverage now proves missing Arrow source, unsupported operator, and
  failed tokenizer warnings expose issue summary fields.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: scoped `git diff --check` for the tofix32 files.

## Resume 2026-07-10 - Operator Warning Diagnostic Context

Twenty-eighth follow-up on the first slice:

- Operator-backed debugger warning traces now carry structured owner
  context alongside their diagnostic phase and issue summary payload.
- Graph-level and node-level warning builders now attach component, source
  file, and source symbol fields without changing existing warning roles,
  issue lists, or caller-specific diagnostic phases.
- Contract coverage now proves missing Arrow source, unsupported operator,
  and failed tokenizer warnings expose source context.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1` (MSVC reported incremental-link `.ilk` warnings, but produced the executable.)
- Passed: scoped `git diff --check` for the tofix32 files.

## Resume 2026-07-10 - Operator Success Diagnostic Context

Twenty-ninth follow-up on the first slice:

- Successful operator-backed TextTokenizer traces now carry the same structured
  owner context as operator warning traces.
- The success path attaches diagnostic phase, component, source file, and source
  symbol after `DebugGraphTraceExecutor` converts the step, keeping the generic
  executor unchanged.
- Contract coverage now proves operator-backed success traces expose source
  context while preserving the existing trace payload.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: scoped `git diff --check` for the tofix32 files.

## Resume 2026-07-10 - Graph Trace Step Issue Summary

Thirtieth follow-up on the first slice:

- `DebugGraphTraceExecutor` now attaches aggregate issue summaries when a
  converted graph trace step emits warnings or errors.
- Step traces now expose issue counts, severity counts, first-seen issue codes,
  and primary warning/error codes without changing no-issue traces or widening
  the step model.
- Contract coverage now proves warning-only and error graph trace steps expose
  issue summary fields.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: scoped `git diff --check` for the tofix32 files.

## Resume 2026-07-10 - Graph Trace Step Diagnostic Context

Thirty-first follow-up on the first slice:

- `DebugGraphTraceExecutor` now attaches default structured owner context to
  converted graph trace steps.
- The context is attached before custom step payload fields are copied, so
  specialized producers can still override diagnostic ownership without
  changing the step model.
- Contract coverage now proves no-issue graph trace steps expose executor
  diagnostic phase, component, source file, and source symbol fields.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: scoped `git diff --check` for the tofix32 files.

## Resume 2026-07-10 - Central Trace Issue Summary

Thirty-second follow-up on the first slice:

- `DebugNodeTraceContract::AddWarning` and `AddError` now attach aggregate
  issue summaries automatically whenever they append a trace issue.
- Helper-based trace producers no longer need local summary calls after using
  the canonical warning/error helpers; direct issue-list producers still attach
  summaries explicitly.
- Contract coverage now proves the low-level warning/error helpers maintain
  issue counts, primary codes, and issue-code summaries.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: scoped `git diff --check` for the tofix32 files.

## Resume 2026-07-10 - Auxiliary Trace Diagnostic Context

Thirty-third follow-up on the first slice:

- Memory ownership, export correlation, backend placement, and Windows crash
  import traces now attach diagnostic ownership metadata with stable phase,
  component, source file, and source symbol fields.
- The change reuses the central `DebugNodeTraceContract::AttachDiagnosticContext`
  helper and does not add new trace shapes or UI behavior.
- Contract coverage now proves these auxiliary trace producers expose diagnostic
  context and that warning/error payload summaries stay attached through the
  canonical warning/error helpers.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: scoped `git diff --check` for the tofix32 files.
- Reconfirmed in current workspace: `cmake --build D:\Dev\CyxWiz_Claude\build --config Debug --target test_debugger_contracts cyxwiz-engine`
- Reconfirmed in current workspace: `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_debugger_contracts.exe`

## Resume 2026-07-10 - Auxiliary Trace Issue Codes

Thirty-fourth follow-up on the first slice:

- Export correlation warnings for missing artifact paths now use the existing
  `CW-S-0101` serialization/artifact code instead of the generic runtime
  fallback.
- Windows crash import warnings now pass the importer's documented `CW-R-0501`
  code explicitly instead of relying on the default warning helper fallback.
- Memory and backend attention warnings intentionally stay on the existing
  generic runtime fallback until a narrower non-overstated diagnostic code is
  warranted.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: scoped `git diff --check` for the tofix32 files.

## Resume 2026-07-10 - Canonical Graph Snapshot Trace

Thirty-fifth follow-up on the first slice:

- `DebugSessionManager::BuildGraphSnapshotTrace` now creates the always-present
  graph snapshot through `DebugNodeTraceContract::Make` instead of manually
  filling a partial trace record.
- Graph snapshot traces now carry the canonical node trace schema plus stable
  diagnostic phase, component, source file, and source symbol metadata.
- Contract coverage now proves debugger sessions emit canonical graph snapshot
  traces while preserving the existing frozen graph payload.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: scoped `git diff --check` for the tofix32 files.

## Resume 2026-07-10 - Support Bundle Diagnostic Text Redaction

Thirty-sixth follow-up on the first slice:

- Support bundle serialization now redacts token-bearing free-form diagnostic
  issue node names and messages before writing run-level and trace-level issue
  records.
- Studio event messages and recommendation titles/details/actions in the debug
  run section now use the same local redaction helper as logs and training
  trace events.
- Contract coverage now proves stable error codes remain visible while sensitive
  diagnostic text is redacted from support-bundle records.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: scoped `git diff --check` for the tofix32 files.

## Resume 2026-07-10 - Support Bundle Metadata Redaction

Thirty-seventh follow-up on the first slice:

- Support bundle serialization now redacts token-bearing top-level request
  reason text, debug-run summary text, and debug trace node names.
- The change reuses the existing support-bundle string redactor and keeps stable
  identifiers, structured error codes, and trace payload schema fields intact.
- Contract coverage now proves support bundles redact these metadata fields while
  retaining the diagnostic fields engineers need for triage.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: scoped `git diff --check` for the tofix32 files.

## Resume 2026-07-10 - Smoke Run Canonical Trace Context

Thirty-eighth follow-up on the first slice:

- `SmokeRunExecutor` now creates smoke-run batch, loss, and backward traces
  through `DebugNodeTraceContract::Make` instead of hand-filling partial
  `DebugTraceRecord` fields.
- Smoke-run traces now carry canonical rank/numel shape payloads at construction
  time plus stable diagnostic phase, component, source file, and source symbol
  metadata.
- No-gradient smoke backward traces now attach the canonical warning issue summary
  on the emitted trace while preserving the existing run-level warning.
- Contract coverage now uses canonical smoke loss/backward fixtures for debugger
  recommendation checks and preserves the support-bundle redaction assertion
  formatting cleanup.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: scoped `git diff --check` for the Smoke Run trace files.

## Resume 2026-07-10 - Canonical Text Preprocessing Trace Records

Thirty-ninth follow-up on the first slice:

- `TextPreprocessingTracer` now creates successful tokenizer, vocabulary, and
  padding traces through `DebugNodeTraceContract::Make` instead of hand-filling
  partial `DebugTraceRecord` fields.
- Text preprocessing output shapes are passed at construction time so canonical
  rank and element-count payload fields match the final trace shapes.
- Missing, empty, out-of-range, and materialization error traces now use the
  same canonical node trace schema while preserving diagnostic phase metadata
  and structured issue summaries.
- Contract coverage now proves text preprocessing success traces and missing
  dataset error traces advertise the canonical schema and shape summaries.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: scoped `git diff --check` for the text preprocessing trace files.

## Resume 2026-07-10 - Canonical Auxiliary Trace Schema Marker

Fortieth follow-up on the first slice:

- `DebugNodeTraceContract::Make` now emits a dedicated `node_trace_schema`
  marker alongside the existing canonical `schema` value.
- `DebugNodeTraceContract::IsNodeTrace` remains backward-compatible with older
  canonical traces while also recognizing the dedicated node-trace marker.
- Export correlation and Windows crash import traces now use the canonical trace
  envelope without overwriting their domain-specific `payload["schema"]`
  contracts.
- Contract coverage now proves auxiliary export/crash traces are canonical node
  traces while preserving their export/crash schema payloads.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: scoped `git diff --check` for the auxiliary trace schema files.

## Resume 2026-07-10 - Canonical Compile Preflight Session Traces

Forty-first follow-up on the first slice:

- Studio Debugger compile traces now use `DebugNodeTraceContract::Make` instead
  of hand-filling the canonical trace envelope in `MainWindow`.
- Studio Debugger preflight traces now use the same canonical envelope while
  preserving their ready/blocked role, status, issues, diagnostic context, and
  summary payloads.
- The change keeps compile/preflight trace construction inside the existing
  session flow and does not add a parallel debugger execution path.

Additional validation:

- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: scoped `git diff --check` for the compile/preflight trace files.

## Resume 2026-07-10 - Canonical Local Debug Runtime Traces

Forty-second follow-up on the first slice:

- Studio Debugger Local Debug forward activation traces now use
  `DebugNodeTraceContract::Make` instead of hand-filling trace envelope fields
  in `MainWindow`.
- Local Debug gradient traces now use the same canonical envelope while
  preserving parameter name, layer index, L2 norm, NaN, and zero-gradient
  payloads.
- Forward and gradient traces now attach stable diagnostic phase, component,
  source file, and source symbol metadata pointing back to `DebugExecutor`.

Additional validation:

- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: scoped `git diff --check` for the Local Debug trace files.

## Resume 2026-07-10 - Local Debug Structured Issue Summaries

Forty-third follow-up on the first slice:

- Local Debug forward activation traces now attach structured warning issues
  when runtime shape differs from the compiler prediction.
- Local Debug forward activation traces now attach structured warning issues
  when the observed output contains NaN or Inf values.
- Local Debug gradient traces now attach a structured warning issue when a
  gradient norm is NaN.
- The issue summaries use existing `tofix26` codes and preserve the existing
  trace status values instead of flattening shape mismatch into a generic
  warning status.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Blocked: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1` is currently failing in unrelated dirty LM panel work: `language_model_generation_panel.cpp` references missing `NextTokenCandidate::logit`.
- Passed: scoped `git diff --check` for the Local Debug issue summary files.

## Resume 2026-07-10 - Local Debug Numerics Recommendations

Forty-fourth follow-up on the first slice:

- `DebugRecommendationEngine` now turns Local Debug forward traces with NaN/Inf
  activation payloads into a critical numerics recommendation.
- Local Debug gradient traces with NaN gradient norms now produce a critical
  gradient recommendation.
- Recommendation contract coverage now uses canonical Local Debug forward and
  gradient fixtures to prove these issue-summary traces produce actionable
  guidance.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: scoped `git diff --check` for the Local Debug recommendation files.

## Resume 2026-07-10 - Local Debug Zero-Gradient Recommendations

Forty-fifth follow-up on the first slice:

- `DebugRecommendationEngine` now turns Local Debug gradient traces with zero
  parameter norms into a warning recommendation.
- Recommendation contract coverage now includes a canonical Local Debug
  zero-gradient fixture so NaN and zero-gradient paths stay distinct.
- The change reuses existing Local Debug gradient payloads and does not add a
  parallel gradient-health model.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: scoped `git diff --check` for the Local Debug zero-gradient recommendation files.

## Resume 2026-07-10 - Export Crash Trace Recommendations

Forty-sixth follow-up on the first slice:

- `DebugRecommendationEngine` now turns failed export correlation traces into
  warning recommendations for export follow-up.
- Export traces missing artifact paths now produce an explicit warning
  recommendation instead of relying only on raw issue payloads.
- Windows crash import traces now produce recommendations for unavailable crash
  reports or imported reports that cannot be confidently matched to the run.
- Recommendation contract coverage now uses the existing export correlation and
  Windows crash import trace producers as fixtures.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: scoped `git diff --check` for the export/crash recommendation files.

## Resume 2026-07-10 - Smoke Run Zero-Gradient Trace Warnings

Forty-seventh follow-up on the first slice:

- `SmokeRunExecutor` now marks backward traces as warning traces when every
  captured gradient tensor is zero.
- The all-zero gradient trace now carries a structured warning issue using the
  existing Smoke Run training-execution error code.
- Recommendation contract coverage now proves all-zero Smoke Run gradients stay
  distinct from the no-gradient case.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: scoped `git diff --check` for the Smoke Run zero-gradient files.

## Resume 2026-07-10 - Trace Issue Recommendations

Forty-eighth follow-up on the first slice:

- `DebugRecommendationEngine` now turns structured warning/error issues stored
  on canonical trace records into recommendations.
- Recommendation issue handling now de-duplicates trace issues that were already
  supplied through the session-level issue list.
- Recommendation contract coverage now proves trace-only warnings are surfaced
  while duplicate trace/session warnings do not produce duplicate trace
  recommendations.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: scoped `git diff --check` for the trace issue recommendation files.

## Resume 2026-07-10 - Local Debug Zero-Gradient Issue Summaries

Forty-ninth follow-up on the first slice:

- Local Debug gradient traces now attach a structured warning issue when the
  gradient norm is zero.
- The zero-gradient trace keeps the existing `is_zero` payload and `zero` status
  while adding the same issue-summary payload fields used by NaN gradients.
- Recommendation contract coverage now proves the zero-gradient fixture exposes
  a warning issue summary in addition to the existing zero-gradient
  recommendation.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: scoped `git diff --check` for the Local Debug zero-gradient issue files.

## Resume 2026-07-10 - Local Debug Missing-Gradient Traces

Fiftieth follow-up on the first slice:

- `GradNormEntry` now preserves whether a gradient tensor was actually present,
  so missing gradients no longer collapse into ordinary zero-gradient traces.
- Local Debug gradient traces now emit `has_gradient=false`, a
  `missing_gradient` status, and a structured warning issue when a trainable
  parameter has no gradient tensor.
- `DebugRecommendationEngine` now emits a specific critical recommendation for
  Local Debug missing-gradient traces while keeping true zero-gradient traces on
  the existing zero-gradient rule.
- Contract coverage now keeps NaN, missing-gradient, and zero-gradient fixtures
  distinct, and `test_debug_executor` proves the golden path marks gradients as
  present.

Additional validation:

- Passed: `cmake --build build --target test_debugger_contracts --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debugger_contracts.exe`
- Passed: `cmake --build build --target test_debug_executor --config Debug -- /m:1`
- Passed: `build\\bin\\Debug\\test_debug_executor.exe`
- Passed: `cmake --build build --target cyxwiz-engine --config Debug -- /m:1`
- Passed: scoped `git diff --check` for the Local Debug missing-gradient files.

## Resume Pointer

Current stopping point:

- Last completed slice: `Local Debug Missing-Gradient Traces`
- Commit pushed: not yet for this Local Debug missing-gradient continuation; previous pushed repository checkpoint was `e9b0c883 Record local debug zero gradient issue checkpoint` on `origin/Nodes_Implementation`.
- Current uncommitted files for this continuation: `debug_executor.h`, `debug_executor.cpp`, `debug_recommendation_engine.cpp`, `main_window.cpp`, `test_debugger_contracts.cpp`, `test_debug_executor.cpp`, and `track32.md`.
- Next safe continuation point: continue from the current `tofix32` / `track32` trail and pick the next smallest debugger diagnostic gap after Local Debug missing-gradient traces, or switch back to the next tracked tofix item requested by the user.
