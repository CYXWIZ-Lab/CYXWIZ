# Data Studio To Fix Truth Audit

Created: 2026-06-07

Purpose: track the pass that reconciles active `tofix*.md` files with closed
`done*.md` archives and current code. Keep this file short and update it while
scanning so stale or duplicate work is not reintroduced.

## Rules

- `done*.md` files are closed archives. Do not move work back into them.
- Active `tofix*.md` files should describe only current gaps or explicitly
  marked historical context.
- If an active doc still lists work already completed in a pushed commit, mark
  that item done, move it to a completed note, or delete the stale claim.
- If multiple active docs describe the same gap, keep one source of truth and
  point the others to it.

## Inventory

| File | Initial classification | Audit decision |
| --- | --- | --- |
| `done.md` | closed archive | Keep closed. Migration boundary is already clear. |
| `done1.md` | closed archive | Keep closed. Follow-ups remain split into active `tofix16.md`, `tofix17.md`, and `tofix18.md`. |
| `done2.md` | closed archive | Keep closed. Use it as evidence for completed CPU fallback/residency work. |
| `done3.md` | closed archive | Keep closed. Frontend review archive; no active edits in this pass. |
| `done10.md` | closed archive | Keep closed. Memory/residency cleanup is completed for that pass. |
| `done13.md` | closed archive | Keep closed. Backend gap audit is complete; broader model gaps live in `tofix19.md`. |
| `done15.md` | closed archive | Keep closed. Tensor ArrayFire layout/residency contract marked complete. |
| `tofix4.md` | active/tracked | Updated. Historical fake-success wording now reflects current fail-closed legacy executor truth. |
| `done5.md` | closed archive | Closed 2026-06-08. Concrete executor/materializer correctness pass is complete; remaining broad architecture work moved to `tofix20.md`. |
| `tofix6.md` | active/untracked | Updated. Pipeline map now says audited placeholders fail closed and optimizer drift is no longer pending. |
| `tofix7.md` | active/untracked | Updated. Support matrix now reflects wired optimizers, blocked unsupported training layers, and legacy fail-closed nodes. |
| `tofix8.md` | active/untracked | Updated. LLM note now distinguishes TransformerEncoder classifiers from decoder/causal LLM gaps and points to `tofix19.md`. |
| `tofix9.md` | active/tracked | Keep active. Debugger CPU/GPU fallback classification remains pending as diagnostics/tooling, not backend fallback implementation. |
| `tofix11.md` | active/untracked | Keep active. Vocabulary dialog/workflow scope is distinct from model-family support. |
| `tofix12.md` | active/untracked | Updated. Sentiment fine-tuning note now points pretrained/generative gaps to `tofix19.md`. |
| `tofix14.md` | active/untracked | Updated. NER/Siamese note records the Dense-encoded NER compile guard and keeps real NER contracts open. |
| `tofix16.md` | active/tracked | Keep active. Loader throughput/performance work remains deferred and distinct. |
| `tofix17.md` | active/tracked | Updated. CPU fallback work from `done2.md`/`done10.md`/`done15.md` is no longer implied as undone. |
| `tofix18.md` | active/tracked | Keep active. Pipeline canvas placeholder naming cleanup remains a separate deferred UI task. |
| `tofix19.md` | active/tracked | Updated. Phase 1 truth guardrails now record Dense-encoded NER and multi-loss/multi-head compiler guards as started/completed slices. |
| `tofix20.md` | active/tracked | New source of truth for broad runtime architecture follow-ups carried out of `done5.md`; Priority 3 string-only alias guard and Priority 5 `DataInput.source_type` cleanup started. |

## Findings

- No active `tofix*.md` file was deleted in this pass. The active docs
  still have separate useful scopes.
- The stale docs problem was mainly wording drift, not duplicate files:
  old docs described placeholder fake-success paths as current even after
  the 2026-06-07 fail-closed pass.
- CPU fallback wording needed separation between completed backend
  fallback implementation slices and still-pending fallback diagnostics,
  policy, and GPU-first performance work.
- Model-family docs needed a boundary: `tofix19.md` is the broad
  unsupported-family map; `tofix8.md`, `tofix12.md`, and `tofix14.md`
  remain focused child notes.

## Patch Log

- 2026-06-08 follow-up: started `tofix20` Priority 5 by removing the stale
  late `DataInput.source_type=ml_dataset` executor branch after central
  validation already limited DataInput sources to `file` and `folder`.
- 2026-06-08 follow-up: started the parallel workflow and accepted a
  `tofix19` guardrail slice that rejects dataset-reachable multi-loss graphs
  until CyxWiz has a first-class loss aggregation or alternating-step
  training contract.
- 2026-06-08 follow-up: started `tofix20` Priority 3 by adding explicit
  compatibility reasons for remaining string-only legacy runtime aliases and
  pinning their exception list in metadata drift tests.
- 2026-06-08 closeout: closed `tofix5.md` as `done5.md` for the concrete
  executor/materializer correctness pass and created `tofix20.md` for the
  remaining broad runtime architecture follow-ups.
- Updated `tofix4.md`, `tofix5.md`, `tofix6.md`, and `tofix7.md` for
  fail-closed executor/compiler truth.
- Updated `tofix17.md` for completed CPU fallback/residency evidence from
  closed docs.
- Updated `tofix8.md`, `tofix12.md`, `tofix14.md`, and `tofix19.md` for
  model-family source-of-truth boundaries and the NER compile guard.
- 2026-06-07 follow-up: updated metadata truth for registered fail-closed
  nodes: unsupported classical ML/evaluation/DNN/utility/signal nodes are now
  template/blocked, while supported `LSTM` and `GRU` metadata is marked
  implemented. Legacy GUI-visible nodes without registry entries remain a
  separate visibility/registration cleanup.
- 2026-06-07 follow-up: registered the remaining known fail-closed
  CNN/pooling/upsampling and classic classifier nodes as template/blocked
  metadata entries, so the compiler and node browser now agree that they are
  visible but unsupported.
- 2026-06-07 follow-up: fixed `tofix5.md` Priority 0 item 5 by sharing
  Arrow/Parquet tabular input-size derivation, including the time-series
  GraphCompiler override.
- 2026-06-07 follow-up: updated `tofix4.md` Priority 5 to reflect the
  existing compile-time guard that rejects template/deferred metadata nodes
  on the selected training path.
- 2026-06-07 follow-up: routed exact registered operator-backed node names
  from legacy `PipelineExecutor` through `PipelineOperatorFactory`, added a
  `StandardScaler` executor regression test, and updated `tofix4.md` /
  `tofix5.md` to keep the remaining work scoped to canonical runtime
  convergence and dead branch cleanup.
- 2026-06-07 follow-up: removed the now-unreachable exact-name
  fail-closed dispatch branches for registered operator-backed nodes from
  `PipelineExecutor`; the remaining cleanup is dead placeholder helper
  bodies and a central capability owner.
- 2026-06-07 follow-up: removed dead placeholder executor declarations
  from `PipelineExecutor`'s header and quarantined their historical
  passthrough/fake-success helper bodies behind compile-time exclusion.
- 2026-06-07 follow-up: started `tofix5` capability centralization by
  moving exact operator-backed legacy runtime names into
  `pipeline_runtime_capabilities.{h,cpp}` and testing them against
  `PipelineOperatorFactory`.
- 2026-06-07 follow-up: extended `pipeline_runtime_capabilities` with
  known fail-closed legacy runtime names/reasons and routed
  `PipelineExecutor` hard-fail decisions through that table.
- 2026-06-07 follow-up: removed the remaining legacy `ExecutePCA`
  declaration from `PipelineExecutor` and quarantined its passthrough body
  behind compile-time exclusion.
- 2026-06-07 follow-up: added `ResolvePipelineRuntimeSupport()` so the
  executor branches on one explicit runtime-support mode instead of
  separate capability lookups.
- 2026-06-07 follow-up: deleted the historical compile-excluded
  `PipelineExecutor` placeholder helper block after support decisions were
  moved into `pipeline_runtime_capabilities`.
- 2026-06-07 follow-up: extended `PipelineExecutor::ValidatePipeline()`
  with required source/export parameter checks and preserved specific
  validation failure messages.
- 2026-06-07 follow-up: added active legacy-executor node names to
  `pipeline_runtime_capabilities` as `LegacyExecutor` support and used
  that resolver to reject unknown pipeline node types during validation.
- 2026-06-07 follow-up: refreshed the `tofix5` executive summary so
  routing, validation, and remaining canonical-runtime work match the
  current implementation.
- 2026-06-07 follow-up: moved legacy pipeline source-node role truth into
  `pipeline_runtime_capabilities` and marked standalone `ParquetInput` as
  known fail-closed; executable parquet loading remains `DataInput`
  with `type=parquet`.
- 2026-06-07 follow-up: moved fixed multi-input arity truth for legacy
  `Join` into `pipeline_runtime_capabilities`, replacing the local
  validator-only list.
- 2026-06-07 follow-up: added the first materializer capability dimension
  to `PipelineRuntimeSupport`: operator-backed nodes are marked
  Arrow-table materializer capable; legacy and fail-closed nodes are not.
- 2026-06-07 follow-up: added source integer parameter validation for
  `DataInput.skip_rows` and Excel `sheet_idx`, preventing late
  execution-time `std::stoi` failures.
- 2026-06-07 follow-up: added bounded integer validation for active
  legacy transform parameters such as `TSWindow.window_size`,
  `TSLag.lag_periods`, `PolynomialFeatures.degree`, `Binning.n_bins`,
  and table row helpers.
- 2026-06-07 follow-up: added validation-time checks for required active
  legacy transform parameters such as `FilterRows.condition`,
  `Join.on_column`, `GroupBy` fields, and `StringManipulation.column`.
- 2026-06-07 follow-up: made PipelineMaterializer source-scope truth
  explicit in `MaterializeResult`, including Arrow-table support and
  non-Arrow source skips for legacy text/parquet/image/audio-style paths.
- 2026-06-07 follow-up: moved active legacy transform required-parameter
  truth into `pipeline_runtime_capabilities` and added metadata drift
  coverage for those entries.
- 2026-06-07 follow-up: moved supported enum values and active legacy
  integer parameter validation rules into `pipeline_runtime_capabilities`,
  with drift coverage and a `DataOutput.format` validation regression.
- 2026-06-07 follow-up: strengthened the runtime capability metadata drift
  guard to reject duplicate entries within each capability list.
- 2026-06-07 follow-up: moved static source/export required parameters
  such as `FileInput.path` and `DataOutput.file_path` into
  `pipeline_runtime_capabilities`, with a `DataOutput.file_path`
  validation regression.
- 2026-06-07 follow-up: blocked the legacy `PolynomialFeatures`
  pass-through path by requiring `columns` at validation time.
- 2026-06-07 follow-up: moved pass-through table helper placeholders
  (`CellExtractor`, `CellUpdater`, `ColumnAppender`, `RowAppender`,
  `Unpivot`) from legacy-dispatched support to fail-closed capability
  entries.
- 2026-06-07 follow-up: wired legacy `ExportCSV` to the real registry
  CSV export path and moved fake-success legacy `ExportExcel`/`ExportJSON`
  branches to fail-closed runtime capability entries.
- 2026-06-07 follow-up: made pipeline parsing fail closed on dangling
  link endpoints and preserved the specific parse failure in
  `ExecutePipeline()` errors.
- 2026-06-07 follow-up: added validation-time rejection for disconnected
  pipeline graphs, with a regression that fails before either source is
  loaded.
- 2026-06-07 follow-up: made legacy `RenameColumns` perform a real Arrow
  schema rename from `mapping`/`rename_map` pairs and fail validation
  when no rename mapping is provided.
- 2026-06-07 follow-up: made legacy `RowToColumnNames` perform a real
  Arrow schema promotion, remove the promoted row, and validate
  `row_index` before execution.
- 2026-06-07 follow-up: tightened legacy `TableCropper` runtime bounds
  checks so invalid crop ranges fail with explicit errors instead of
  depending on Arrow slice behavior.
- 2026-06-07 follow-up: removed dead executor declarations, dispatch
  branches, and fake-success bodies for fail-closed legacy helpers
  (`ExportExcel`, `ExportJSON`, `CellExtractor`, `CellUpdater`,
  `ColumnAppender`, `RowAppender`, `Unpivot`).
- 2026-06-07 follow-up: repaired `DuckDBConnector` Arrow-table
  registration via a copied table/appender path, preserved basic
  numeric Arrow types in DuckDB query results, restored legacy
  `MathFormula` with required `formula` validation, and kept
  `RuleEngine` fail-closed because it ignored the advertised `rules`
  contract.
- 2026-06-07 follow-up: fixed `ArrowToTensor` ArrayFire dimension
  construction so Arrow tables convert to the intended `[rows, columns]`
  tensor shape; `test_arrow_integration` now passes end to end.
- 2026-06-07 follow-up: replaced legacy `FillMissing` statistic
  placeholder zero-fill with per-column DuckDB `mean`/`median`/`mode`
  expressions and centralized `strategy` validation.
- 2026-06-07 follow-up: made legacy `StringManipulation` execute its
  advertised `replace` and `substring` operations and centralized
  operation enum validation so unsupported operations fail before
  execution.
- 2026-06-07 follow-up: tightened legacy `Binning` to require one
  explicit column, validate supported methods centrally, quote SQL
  identifiers, and compute equal-width bins through a tested DuckDB
  expression.
- 2026-06-07 follow-up: tightened legacy `PolynomialFeatures` to one
  explicit column, degree `>= 2`, quoted SQL identifiers, and generated
  powers through the requested degree instead of retaining no-column or
  partial-degree behavior.
- 2026-06-07 follow-up: moved legacy `TableSplitter` to fail-closed
  support because Data Studio pipeline links do not preserve output-pin
  identity, so the executor cannot route its advertised `Top` and
  `Bottom` outputs truthfully.
- 2026-06-07 follow-up: aligned metadata status for fail-closed
  table/export helpers (`CellExtractor`, `CellUpdater`, `ColumnAppender`,
  `RowAppender`, `Unpivot`, `ExportExcel`, `ExportJSON`) so the node
  registry no longer marks runtime-blocked nodes as implemented.
- 2026-06-07 follow-up: made the runtime capability materializer
  dimension explicit as `ArrowTableOnly` versus `None`, matching the
  current `PipelineMaterializer` storage scope instead of implying
  general graph materialization.
- 2026-06-07 follow-up: tightened legacy `MathFormula` output-column SQL
  generation to use the shared identifier quoting helper and added a
  regression for output names containing double quotes.
- 2026-06-07 follow-up: moved GraphCompiler's unsupported
  sequential-model layer and unsupported training-control node truth into
  typed capability entries so compile-time training gaps are no longer
  duplicated between compiler code and tests.
- 2026-06-07 follow-up: tightened Parquet-backed training batcher setup
  to mirror Arrow time-series partition filtering, internal metadata
  column skipping, and regression label shape; extended
  `test_training_batcher_setup` with real Parquet-backed tabular and
  time-series parity coverage.
- 2026-06-07 follow-up: extended `test_training_batcher_setup` again
  with explicit multi-row-group Parquet tabular splitting and
  time-series partition-filtering coverage. `tofix5.md` now leaves only
  the broader Arrow/Parquet full training-loop parity under Ticket E.
- 2026-06-07 follow-up: started deeper Arrow/Parquet loop parity by
  extending `test_training_batcher_setup` with matching
  `BuildSequentialFromConfig()` train/validation model-step coverage for
  Arrow and multi-row-group Parquet batchers. Full `TrainingExecutor`
  end-to-end parity remains pending.
- 2026-06-07 follow-up: centralized materializer storage-backend
  availability in `pipeline_runtime_capabilities`; Arrow tables are the
  only supported materializer backend, with explicit unsupported reasons
  for Parquet-backed, image, audio, and text datasets. `PipelineMaterializer`
  now consults that registry before pass-through.
- 2026-06-07 follow-up: added loaded-table schema/type checks for active
  column transforms (`StringManipulation`, `Binning`, and
  `PolynomialFeatures`) so missing columns and text-vs-numeric mistakes
  fail with explicit errors before DuckDB query construction.
- 2026-06-07 follow-up: extended active legacy transform schema checks
  to `SelectColumns` and `SortRows`; both now validate requested loaded
  table columns and quote resolved column identifiers before building
  DuckDB SQL.
- 2026-06-07 follow-up: extended the same loaded-table structural-column
  checks to `Join.on_column` and `GroupBy.group_columns`; `GroupBy`
  aggregation expressions remain free-form SQL pending a separate
  expression policy.
- 2026-06-07 follow-up: added central enum validation and runtime SQL
  normalization for `SortRows.order`, legacy `SortRows.ascending`, and
  `Join.join_type`, including frontend-style `outer` to `FULL OUTER`.
- 2026-06-07 follow-up: replaced `GroupBy.aggregations` raw SQL
  concatenation for the supported simple aggregate subset with
  schema-checked aggregate parsing, identifier quoting, alias validation,
  and numeric-type checks for numeric-only aggregate functions.
- 2026-06-07 follow-up: added the first explicit runtime `fail_mode`
  axis to central runtime support (`real` for operator-backed and active
  legacy-dispatched paths, `hard_fail` for known unsupported paths) with
  drift coverage for stable fail-mode names.
- 2026-06-07 follow-up: centralized the first training backend support
  resolver through `PipelineTrainingBackendSupport`; graph compiler
  unsupported sequential-layer and training-control checks now share that
  typed truth, with drift coverage for allowed and unsupported modes.
- 2026-06-07 follow-up: moved source-node and required-input-count truth
  onto resolved `PipelineRuntimeSupport`, so `PipelineExecutor`
  validation consumes the same support object used for runtime routing.
- 2026-06-07 follow-up: moved required-parameter, allowed-value, and
  integer-parameter validation axes onto resolved `PipelineRuntimeSupport`
  as well, while preserving the existing specialized validation behavior.
- 2026-06-07 follow-up: added optional blocked metadata node identity to
  fail-closed runtime capability entries and made the metadata drift guard
  enforce that those browser-visible blocked nodes are not implemented.
- 2026-06-07 follow-up: carried the optional fail-closed metadata node
  identity through resolved `PipelineRuntimeSupport` so routing support
  truth and metadata status truth remain tied together.
- 2026-06-07 follow-up: added stable names and drift coverage for
  runtime support mode, materializer storage support scope, and training
  backend support mode.
