# To Fix 5 - Execution Architecture, Fail-Open Behavior, and Runtime Drift

This document focuses on the parts of CyxWiz that decide what actually
executes once a graph leaves the frontend.

The main issue is not one isolated bug. The current engine has multiple
execution surfaces with overlapping responsibility:

1. `PipelineExecutor`
2. `PipelineMaterializer`
3. `PipelineOperatorFactory` / `IPipelineOperator`
4. `TrainingExecutor`
5. `DebugExecutor`

Those paths do not agree on coverage, validation, or failure behavior.
That creates a dangerous class of bugs where:

- a node appears supported
- a graph compiles
- execution succeeds
- but the runtime either did nothing, used a placeholder path, or ran a
  weaker path than the system already has available

---

## Executive Summary

The highest-risk runtime issue in the engine today is `execution drift`.

The product surface suggests a graph-first platform with consistent node
execution. The implementation is still split between:

- a legacy string-dispatch executor that now fails closed for audited
  placeholder branches and routes exact operator-backed names through
  the operator framework, while still retaining legacy-only branches
- a newer operator framework with real implementations for a growing
  subset of nodes
- a separate training/debug model path that builds sequential models
  from compiled config

The result is inconsistent truthfulness.

Current truth after the 2026-06-07 cleanup:

1. Done: `PipelineExecutor::ExecuteParallel()` no longer shares one
   mutable context across async tasks without a merge boundary.
2. Done: cached nodes without a cached output dataset are no longer
   treated as completed successful nodes.
3. Done for current executor input binding: single-input nodes and the
   two-input `Join` path now resolve upstream datasets through one
   shared binding helper instead of silently selecting the first input
   edge.
4. Done for baseline structure/runtime support: `ValidatePipeline()`
   now rejects duplicate ids, missing/self links, invalid source/input
   shapes, unsupported multi-input paths, cycles/topology failures,
   missing source/export parameters, unsupported source/export parameter
   values, and node types unknown to the central runtime capability
   registry.
5. Partially fixed for truth: `PipelineMaterializer` only materializes
   in-memory Arrow graphs and explicitly skips Parquet-backed, image,
   audio, and text sources. Runtime capabilities now expose that
   operator-backed materializer support is `ArrowTableOnly`, not general
   storage-mode support.
6. Partially fixed: audited placeholder branches in `PipelineExecutor`
   now fail closed instead of returning fake success, and exact
   operator-backed names route through `PipelineOperatorFactory`.
   Remaining work is to choose the canonical executor/materializer
   ownership model for all Data Studio graph execution.
7. Done: `TrainingManager::StartTrainingArrow()` and
   `StartTrainingParquet()` now use the same shared input-size resolver,
   including the GraphCompiler time-series override. Parquet batcher
   setup now also mirrors Arrow time-series partition filtering,
   internal metadata-column skipping, and regression-label shape.

---

## Priority 0: Concrete Bugs

### 1. Data race in `PipelineExecutor::ExecuteParallel`

**Severity:** High

**Status 2026-06-07:** Fixed for the current legacy executor pass.
Parallel work now uses per-task result state and merges results after
task completion. `ExecuteNode` is serialized where it still depends on
shared executor/DuckDB state, so this is safe before a larger executor
ownership redesign.

Relevant files:

- `cyxwiz-engine/src/core/pipeline_executor.cpp:2209`

Problem:

- `ExecuteParallel()` creates one shared `ExecutionContext ctx`.
- It launches multiple `std::async` tasks.
- Each task receives `&ctx` and calls `ExecuteNode(*node, ctx)`.
- `ExecutionContext` contains mutable shared state, including
  `node_results`.
- There is no locking, no per-task context, and no merge step.

Observed code shape:

- `std::async(std::launch::async, [this, node, &ctx]() mutable { return ExecuteNode(*node, ctx); })`

Effect:

- concurrent writes to shared maps
- racy reads of upstream outputs
- nondeterministic execution bugs
- intermittent cache corruption or missing result propagation

Recommendation:

- keep the current guarded merge behavior until `PipelineExecutor` is
  replaced or split into executor instances with explicit resource
  ownership

---

### 2. Cached nodes are marked completed even when no cached output exists

**Severity:** High

**Status 2026-06-07:** Fixed. A cached node with no cached output is
treated as invalid for completion instead of silently unblocking
downstream nodes.

Relevant files:

- `cyxwiz-engine/src/core/pipeline_executor.cpp:2216`

Problem:

- In `ExecuteParallel()`, nodes with `needs_execution == false` are
  inserted into `completed`.
- `ctx.node_results[node.id]` is only populated when
  `cached_output_dataset` is non-empty.
- That means a node can be considered complete while downstream nodes
  still have no dataset name available to consume.

Effect:

- downstream nodes may become ready too early
- runtime can fail later with "no input dataset" errors
- cache behavior becomes stateful and misleading

Recommendation:

- keep adding tests around dirty/cached graph transitions as the
  executor is converged with the materializer/operator path

---

### 3. `GetInputDatasetName()` only uses the first input edge

**Severity:** High

**Status 2026-06-07:** Fixed for the current legacy helper. Multiple
inputs now fail closed instead of silently using the first edge.

**Status 2026-06-07 follow-up:** The executor now has a shared input
dataset binding helper. Single-input nodes require exactly one upstream
dataset, while `Join` uses the same helper to bind exactly two ordered
input datasets instead of manually indexing `node.inputs`. Named input
pins remain future work for graph formats that need pin-level semantics.

Relevant files:

- `cyxwiz-engine/src/core/pipeline_executor.cpp:1727`
- `cyxwiz-engine/src/core/pipeline_executor.cpp:2046`

Original problem:

- `GetInputDatasetName()` returned `node.inputs[0]` only.
- Any node that logically depends on more than one upstream input is not
  modeled correctly by this helper.

Original effect:

- multi-input nodes could not execute truthfully on this path
- future node additions could silently inherit broken semantics
- compile-time graph shape and runtime data flow could diverge

Recommendation:

- keep shared input binding as the only executor path for upstream
  dataset lookup
- model named input pins or typed input collections when the graph JSON
  preserves pin-level semantics
- keep failing closed for node types that require unsupported
  multi-input or multi-output semantics

---

### 4. `ValidatePipeline()` is too weak to protect execution

**Severity:** High

**Status 2026-06-07:** Partially fixed. Baseline structural validation
now covers duplicate ids, missing/self links, invalid source/input
shapes, disconnected graphs, unsupported multi-input paths, cycles, and required
source/export parameters such as `DataInput.file_path`,
`DataInput.folder_path`, `FileInput.path`, and output `file_path`
settings. It also rejects missing required legacy transform parameters
such as `FilterRows.condition`, `SelectColumns.columns`,
`SortRows.columns`, `Join.on_column`, `GroupBy` fields, and
`StringManipulation.column`; `RenameColumns` now requires a `mapping`
or legacy `rename_map` value. Validation rejects node types that are
unknown to the central runtime capability registry before execution
starts. Validation now rejects invalid `DataInput.skip_rows` and Excel
`sheet_idx` integer parameters before execution reaches `std::stoi`,
plus bounded integer parameters for active legacy transforms such as
`TSWindow`, `TSLag.lag_periods`, `PolynomialFeatures`, `Binning`, and
table row helpers, including `RowToColumnNames.row_index`. Dangling
links whose start or end node id is missing
now fail during parsing instead of being silently dropped. Disconnected
graphs now fail validation instead of running as independent islands.
Validation and parse errors now preserve the specific failed rule. Broader
schema/type-aware validation remains future work, but the active
column-transform path now rejects obvious loaded-table mismatches for
`StringManipulation`, `Binning`, and `PolynomialFeatures` before DuckDB
query construction. `SelectColumns`, `SortRows`, `Join.on_column`, and
`GroupBy.group_columns` now also validate loaded-table columns before
query construction and quote the resolved column identifiers instead of
passing raw structural column strings to SQL. `GroupBy.aggregations`
now accepts only a small schema-checked aggregate-expression policy
(`COUNT(*)`, `COUNT(column)`, `SUM`, `AVG`, `MIN`, `MAX`, `MEDIAN`, and
`MODE` over existing columns, with optional `AS` aliases) and rejects raw
SQL fragments before query construction. Numeric-only aggregates reject
text columns before DuckDB SQL is built. Active legacy enum parameters
now also reject unsupported values through the central capability
registry, including `SortRows.order`, legacy `SortRows.ascending`, and
`Join.join_type`; the executor normalizes those values before building
DuckDB SQL.

Relevant files:

- `cyxwiz-engine/src/core/pipeline_executor.cpp:157`

Problem:

- validation currently only checks the most basic shape of the node
  list
- code comments still mark disconnected-node checks, type checks, and
  parameter validation as TODOs

Effect:

- schema/type-invalid graphs can still reach runtime for node families
  without dedicated validation beyond the current source parameter,
  legacy scalar-integer baseline, and the first loaded-table column
  checks for string/numeric transform nodes
- some runtime correctness still depends on late execution-time failures

Recommendation:

- validation must at minimum reject:
  - broader disconnected-graph policy for intentionally separate jobs
  - cycles for DAG-only paths
  - missing required params beyond the current source/export and active
    legacy transform baseline
  - missing required inputs
  - unsupported node types for execution paths beyond the current legacy
    executor/runtime registry baseline
  - obvious type/schema mismatches beyond the current source, legacy
    scalar-integer/enum parameter baseline, and active loaded-table
    column/list/aggregation transform checks

---

### 5. Arrow and Parquet training paths are not logically equivalent

**Severity:** Medium-High

**Status 2026-06-07:** Fixed for tabular/time-series input-size
derivation and first batcher-shape parity. Arrow and Parquet training
start paths now both call `ResolveTabularTrainingInputSize()`, which
preserves the GraphCompiler time-series `input_size` override and
otherwise reserves one label column for normal multi-column tabular data.
`BuildParquetTrainingBatchers()` now mirrors Arrow time-series setup by
using `__partition__`, label `y`, and regression labels. A focused
`test_training_batcher_setup` regression now covers tabular fallback,
time-series override preservation, real Parquet-backed tabular batches,
multi-row-group tabular splitting, and Arrow/Parquet time-series
feature/label shape parity, including multi-row-group partition
filtering. The same regression now also drives a tiny
`BuildSequentialFromConfig()` train/validation model-step pass over
matching Arrow and multi-row-group Parquet batchers, proving that both
storage paths can feed forward, loss, backward, update, and validation
steps with finite losses.

Relevant files:

- `cyxwiz-engine/src/core/training_manager.cpp:127`
- `cyxwiz-engine/src/core/training_manager.cpp:189`

Problem before this pass:

- `StartTrainingArrow()` contains special handling for time-series
  graphs, trusting the `GraphCompiler` input-size override.
- `StartTrainingParquet()` does not mirror that logic.
- It always falls back to `num_cols - 1`.

Effect before this pass:

- time-series or materialized schema flows can behave differently based
  on storage mode
- Parquet-backed training can build a model with the wrong input size
  even if the Arrow path is correct

Recommendation:

- keep the shared resolver as the single schema-to-model input-size
  decision point for tabular Arrow/Parquet training paths
- continue auditing deeper Arrow/Parquet parity separately, especially
  full `TrainingExecutor` end-to-end behavior such as callbacks,
  checkpoint policy, and manager dispatch

---

## Priority 1: Architecture Drift

### 6. `PipelineExecutor` and `PipelineMaterializer` are competing execution systems

**Severity:** High

Relevant files:

- `cyxwiz-engine/src/core/pipeline_executor.cpp`
- `cyxwiz-engine/src/core/pipeline_materializer.cpp`
- `cyxwiz-engine/src/core/node_executors/pipeline_operator_factory.cpp`

Problem:

- `PipelineExecutor` is a legacy string-dispatch runtime over
  `node.type`.
- `PipelineMaterializer` is a newer operator-based path using
  `PipelineOperatorFactory` and `IPipelineOperator`.
- Both are graph execution mechanisms.
- They do not share one canonical notion of node support.

Effect:

- duplicated runtime logic
- duplicated node capability decisions
- placeholder behavior survives even after real operators exist
- engineers can add support in one path and still leave the user-facing
  path misleading

Recommendation:

- declare one canonical Data Studio execution path
- treat the other as compatibility scaffolding only
- create an explicit migration plan from legacy dispatcher to operator
  framework

---

### 7. `PipelineMaterializer` is still a narrow v1 path

**Severity:** Medium

**Status 2026-06-07:** Partially fixed for truth reporting. The
materializer still only transforms in-memory Arrow tables, but
`MaterializeResult` now reports the detected source kind and whether a
non-Arrow source was skipped as unsupported for materialization. Tests
cover Arrow-table materialization and legacy text pass-through. Parquet,
image, audio, and legacy text sources remain non-materialized paths.

Relevant files:

- `cyxwiz-engine/src/core/pipeline_materializer.h:34`
- `cyxwiz-engine/src/core/pipeline_materializer.cpp:52`
- `cyxwiz-engine/src/core/pipeline_materializer.cpp:100`

Problem:

- materialization only runs for in-memory Arrow datasets
- Parquet-backed, image, audio, and text datasets are explicitly
  skipped; this is now explicit in the materializer result instead of
  only implicit in a debug log
- traversal is a BFS from `DataInput`
- comments acknowledge that parallel preprocessing branches are a v1
  limitation

Effect:

- operator-based execution only helps a subset of graphs
- support depends on dataset storage mode
- graph semantics are not uniformly modeled across data types

Recommendation:

- document current scope in UI/runtime capability checks
- do not present materialization coverage as general graph execution
- plan the next stage around topological planning and typed dataset
  adapters, not BFS-only linear-chain assumptions

---

### 8. The runtime is still stringly typed

**Severity:** Medium

Relevant files:

- `cyxwiz-engine/src/core/pipeline_executor.cpp`

Problem:

- `PipelineExecutor::Node` stores node type as string
- dispatch is a long chain of string comparisons
- the engine already has `NodeType` enum-based systems elsewhere

Effect:

- drift between frontend node metadata, compile-time enums, runtime
  names, and operator registration
- higher bug risk during node additions or renames

Recommendation:

- move runtime dispatch to `NodeType`
- centralize type-to-capability mapping
- avoid multiple parallel registries of support truth

---

## Priority 2: Fail-Open and Placeholder Truthfulness

### 9. `PipelineExecutor` placeholder-success paths from the old legacy dispatch

**Severity:** High

**Status 2026-06-07:** Fixed for the audited legacy dispatch branches.
These nodes now return explicit unsupported execution errors through
`FailUnsupportedNode()` instead of passthrough/fake success. The old
placeholder helper bodies have been removed from the header contract and
the historical compile-excluded placeholder block has been deleted, so
they cannot be called from active dispatch.

Relevant files:

- `cyxwiz-engine/src/core/pipeline_executor.cpp`

Representative examples:

- `PCA`
- `t-SNE`
- classical ML nodes such as `DecisionTree`, `RandomForest`, `SVM`,
  `KNN`, `NaiveBayes`, `LogisticRegression`
- evaluation nodes such as `ConfusionMatrix`, `ROCCurve`,
  `LearningCurves`, `FeatureImportance`, `CrossValidation`
- utility/text nodes such as `TFIDF`, `CountVectorizer`,
  `SentimentAnalyzer`, `Regex`, `JSONPath`, `Calculator`
- dataset and augmentation nodes such as `ImageFolderDataset`,
  `MNISTDataset`, `CIFAR10Dataset`, `HuggingFaceDataset`,
  `KaggleDataset`, `AugmentationPreset`, `GeometricTransform`,
  `ColorTransform`, `MorphologyTransform`, `AdvancedAugment`
- legacy table helper placeholders such as `CellExtractor`,
  `CellUpdater`, `ColumnAppender`, `RowAppender`, and `Unpivot`
- legacy writer nodes such as `ExportExcel` and `ExportJSON`

Problem now:

- exact registered operator-backed node names now route through
  `PipelineOperatorFactory`
- known unsupported legacy node names fail closed through the central
  runtime capability registry
- audited table-helper pass-through placeholders now fail closed instead
  of forwarding their input dataset as if a transform occurred
- legacy `ExportCSV` now writes through `DataRegistry::ExportArrowToCSV`;
  legacy `ExportExcel` and `ExportJSON` fail closed instead of only
  logging and returning success
- legacy `RenameColumns` now rebuilds the Arrow schema with renamed
  fields instead of registering the input table unchanged
- legacy `RowToColumnNames` now promotes a row to Arrow schema names and
  removes the promoted row instead of registering the input table unchanged
- legacy `TableCropper` now validates crop bounds against the loaded
  Arrow table instead of relying on Arrow slice clamping behavior
- legacy `TableSplitter` now fails closed because the pipeline JSON
  links do not carry output-pin identity, so the executor cannot route
  the advertised `Top` and `Bottom` outputs truthfully
- legacy `MathFormula` now requires an explicit `formula` and runs
  through the repaired DuckDB registration path instead of silently
  creating a constant zero column when the expression is missing; its
  output column name now uses the shared SQL identifier quoting helper
  instead of raw double quotes
- legacy `RuleEngine` now fails closed because its old executor path
  ignored the `rules` parameter and only wrote the default value
- `DuckDBConnector` now registers Arrow input by copying supported scalar
  types into an in-memory DuckDB table and preserves basic numeric result
  types when converting query output back to Arrow; this restores SQL-backed
  transform behavior while true zero-copy Arrow scan support remains future
- `ArrowToTensor` now constructs ArrayFire dimensions with the intended
  `[rows, columns]` shape instead of collapsing to a one-dimensional comma
  expression
- legacy `FillMissing` now builds per-column fill expressions for
  `mean`, `median`, `mode`, and `constant` strategies instead of using
  a placeholder zero fill for statistic-based strategies
- legacy `Binning` now requires an explicit single column, validates
  `equal_width` / `equal_freq` methods centrally, quotes SQL identifiers,
  and computes equal-width bins without relying on an unverified placeholder
  column default
- legacy `PolynomialFeatures` now requires one explicit column and
  degree `>= 2`, quotes SQL identifiers, and generates all requested
  powers instead of retaining a no-column passthrough or partial
  degree coverage
- legacy `StringManipulation` now executes its advertised `replace` and
  `substring` operations, validates the operation enum centrally, and
  rejects unknown operations instead of returning a successful no-op
- fail-closed legacy helper bodies such as `ExportExcel`,
  `ExportJSON`, `CellExtractor`, `CellUpdater`, `ColumnAppender`,
  `RowAppender`, `Unpivot`, and `TableSplitter` have been removed from
  the active executor contract; the runtime capability registry now owns
  their unsupported behavior
- broader support truth still needs the next multi-axis capability matrix

Effect:

- user-facing fake success is fixed for the audited branches
- engineering truth is cleaner because fake-success helpers are no
  longer part of the compiled API or retained as dead TODO code

Recommendation:

- keep unsupported nodes failing closed by default

---

### 10. Real operators already exist for some node families, but the legacy runtime still masks that progress

**Severity:** High

**Status 2026-06-07:** Partially fixed. The legacy runtime no longer
masks these paths with fake success for the audited branches. Exact
node names with registered `PipelineOperatorFactory` implementations now
route through the operator framework from `PipelineExecutor`, and their
old unreachable fail-closed dispatch branches have been removed.
Remaining work is architectural: choose the canonical runtime, expand
coverage where storage-mode support is still narrow, and remove or
delete the quarantined historical helper block.

Relevant files:

- `cyxwiz-engine/src/core/node_executors/pipeline_operator_factory.cpp`
- `cyxwiz-engine/src/core/pipeline_executor.cpp`

Confirmed operator-backed families include:

- time-series: `TimeSeriesWindow`, `TimeSeriesSplit`,
  `TimeSeriesFeatures`, `LogTransform`, `Differencing`
- text: `TextTokenizer`, `TFIDFVectorizer`, `CountVectorizer`,
  `SentimentAnalyzer`
- dimensionality reduction / clustering: `PCANode`,
  `KMeansCluster`, `DBSCANCluster`, `HierarchicalCluster`,
  `GMMCluster`
- signal processing: `FFTNode`, `Convolution1D`, `FilterDesigner`
- regression: `LinearRegressionNode`, `PolynomialRegressionNode`
- preprocessing: `StandardScaler`, `MinMaxScaler`, `RobustScaler`,
  `LabelEncoder`, `OrdinalEncoder`, `TargetEncoder`,
  `OutlierDetector`
- time-series analysis: `TimeSeriesDecomposition`,
  `ARIMAForecaster`, `ExponentialSmoothing`

Problem now:

- the engine now exposes real operator implementations from the older
  `PipelineExecutor` for exact registered node names
- the placeholder-era historical helper block has been deleted instead
  of preserved as dead source

Effect now:

- real operator progress is surfaced in the legacy runtime for the
  registered exact node names
- support truth is still harder to reason about because executor
  ownership remains split
- node audits become more expensive because "implemented" depends on
  which path ran

Recommendation:

- add a capability matrix owned by runtime, not scattered comments
- continue converging support truth into runtime-owned capability data

---

## Priority 3: Design and Validation Improvements

### 11. Training and debug correctly share model-building, but that also means model-builder gaps hit both paths

**Severity:** Medium

Relevant files:

- `cyxwiz-engine/src/core/training_executor.cpp:79`
- `cyxwiz-engine/src/core/debug_executor.cpp:84`

Problem:

- both training and debug rely on `BuildSequentialFromConfig(config_)`
- this is good for consistency
- but it also means any missing layer support or build mismatch affects
  both product surfaces

Recommendation:

- keep the shared builder
- strengthen compile-time capability checks so unsupported layer graphs
  are blocked before either training or debug starts

---

### 12. Capability truth should be explicit and centralized

**Severity:** Medium

**Status 2026-06-07:** Started. Exact legacy runtime names that route
through `PipelineOperatorFactory`, plus known fail-closed legacy runtime
names and reasons, and active legacy-executor node names now live in
`pipeline_runtime_capabilities.{h,cpp}` instead of being embedded only in
`PipelineExecutor`. `PipelineExecutor` now asks that registry for one
explicit runtime-support mode before routing operator-backed nodes or
hard-failing known unsupported nodes. Validation also rejects unknown
runtime-support modes before execution, and source-node role truth now
lives in the same module. Fixed multi-input arity overrides, currently
`Join`, are also centralized there. Resolved runtime support now carries
those validation axes too: `source_node` and `required_input_count`
drive `PipelineExecutor` validation from the same support object used
for runtime routing. Runtime support now carries the first materializer
dimension: exact operator-backed nodes are marked
`ArrowTableOnly` materializer capable, while fail-closed and
legacy-dispatched nodes are marked `None`. Static source/export required
parameters and required legacy transform parameters for active runtime
validation are now centralized there as well, along with supported enum
values such as `DataInput.source_type` and `DataOutput.format` and
bounded integer parameter rules for active legacy transforms. Resolved
runtime support now carries those parameter-validation axes too, so
`PipelineExecutor` validates required parameters, enum values, and
integer bounds from the same support object used for routing. The
metadata drift test
verifies every listed operator runtime capability has a real factory
operator, that operator/fail-closed/legacy-dispatched names do not
overlap, and that validation capability entries resolve to known
runtime-supported names. Fail-closed runtime entries that correspond to
browser-visible blocked nodes now carry that metadata node identity in
the same capability registry and on resolved runtime support, and the
drift suite verifies those nodes are not marked implemented. The first
explicit `fail_mode` axis now lives
on resolved runtime support too: operator-backed and active
legacy-dispatched paths report `real`, while known unsupported paths
report `hard_fail`; stable fail-mode names are covered by the drift
test. Stable names now also exist for runtime support mode,
materializer storage support scope, and training backend support mode,
with drift coverage so frontend-facing support labels do not silently
change. Materializer storage-backend truth is now centralized too: Arrow
tables are the only supported materializer backend, while
Parquet-backed, image, audio, and text datasets carry explicit
unsupported reasons and `PipelineMaterializer` consults that registry
before pass-through. The first compile/training backend availability
axis is now explicit through `PipelineTrainingBackendSupport`.
Resolved runtime support now also carries an explicit
`pipeline_executor_supported` axis. Operator-backed and active
legacy-dispatched nodes are marked executable, fail-closed and unknown
nodes are not, and `PipelineExecutor` validation now rejects unsupported
nodes from that central axis with the registry fail-closed reason before
any fake execution branch can run. Browser-visible node metadata now
also consumes the fail-closed portion of that central truth: matching
nodes are forced to template status, carry a `Blocked` badge, and expose
the registry reason in help text. Remaining work is broader frontend
presentation of all support axes, not another separate runtime list.

**Status 2026-06-07 follow-up 4:** `NodeMetadataRegistry` now applies
fail-closed runtime capability status after built-in metadata
initialization. The add-node search and node browser already consume
that metadata, so unsupported runtime-backed nodes now inherit the
central blocked status and reason instead of relying on parallel
frontend assumptions. The drift suite verifies the badge and reason
alongside template status.

**Status 2026-06-07 follow-up 3:** The first training-support axis is now
centralized too: compiler-blocked sequential-model layers and
training-control nodes live in typed capability entries with explicit
reasons. The graph compiler now consumes unified
`PipelineTrainingBackendSupport` results for these failures, and the
drift suite verifies supported and unsupported training backend modes
instead of carrying separate hardcoded unsupported lists.

**Status 2026-06-07 follow-up:** The legacy `PolynomialFeatures`
branch no longer validates successfully without `columns`, because that
path was a pass-through rather than an "all numeric columns"
implementation.

**Status 2026-06-07 follow-up 2:** The legacy `PolynomialFeatures`
branch now executes only one explicit column, validates degree `>= 2`,
quotes SQL identifiers, and generates all requested powers through the
selected degree.

Problem:

- the system currently communicates support through a mix of:
  - frontend node visibility
  - metadata flags
  - compiler recognition
  - executor branches
  - operator registration
  - backend implementation availability

Effect:

- there is no single trustworthy answer to "is this node supported"

Recommendation:

- continue growing the central runtime capability registry with dimensions such
  as:
  - `frontend_visible`
  - `compile_supported`
  - `training_supported`
  - `pipeline_supported` - now explicit as `pipeline_executor_supported`
  - `materializer_supported`
  - `backend_available`
  - `fail_mode` (`hard_fail`, `simulated`, `passthrough`, `real`)

---

## Recommended Engineering Order

### Phase 1 - Stop correctness hazards

1. Fix `ExecuteParallel()` shared-context concurrency bug.
2. Fix cached-node completion semantics.
3. Strengthen `ValidatePipeline()`.
4. Hard-fail unsupported placeholder branches instead of silent success.

### Phase 2 - Make support truth explicit

1. Build a runtime capability matrix.
2. Block unsupported nodes at compile/run entry points.
3. Expose unsupported/simulated states in UI and logs.

### Phase 3 - Converge execution systems

1. Choose the canonical Data Studio execution path.
2. Route operator-backed nodes through that path first.
3. Decommission legacy duplicate branches as coverage migrates.

### Phase 4 - Normalize training/data semantics

1. Unify Arrow/Parquet input-size derivation. Done for tabular and
   time-series paths.
2. Move schema-to-model logic into shared helpers.
3. Add execution-path tests for storage-mode parity. Covered with
   tabular/time-series batcher-shape parity and multi-row-group
   Parquet batcher coverage, shared model-step coverage, and full
   `TrainingExecutor::Train()` Arrow/Parquet parity coverage.

---

## Good First Tickets

### Ticket A: Make `ExecuteParallel()` safe

Status: completed in this branch.

Scope:

- `cyxwiz-engine/src/core/pipeline_executor.cpp`

Deliverable:

- remove shared mutable `ExecutionContext` from async tasks
- add deterministic result merge
- add regression test for parallel node execution

### Ticket B: Cache validity cleanup

Status: completed in this branch.

Scope:

- `cyxwiz-engine/src/core/pipeline_executor.cpp`

Deliverable:

- do not mark non-executed nodes complete without a valid cached output
- add explicit cache-invalid behavior

### Ticket C: Replace placeholder success with fail-closed behavior

Status: partially completed in this branch. Known fake-success branches
listed above have been converted or implemented, and metadata now marks
the covered blocked helpers as `Template` / `Blocked`. Fail-closed
runtime entries now carry optional blocked metadata node identity through
resolved runtime support, and the drift suite checks those browser-visible
nodes are not marked implemented. Continue treating newly discovered
placeholder branches as defects, not supported features.

Scope:

- `cyxwiz-engine/src/core/pipeline_executor.cpp`

Deliverable:

- audit placeholder branches
- convert unsupported ones to explicit runtime errors
- preserve only intentionally simulated nodes with visible status

### Ticket D: Capability registry

Status: mostly completed in this branch. Runtime support now has centralized operator,
legacy, fail-closed, fail-mode, fail-closed metadata status,
required-parameter, enum, integer-validation, source-node, required-input-count,
compiler-blocked training, first training backend support mode,
Arrow-table materializer-scope truth, materializer storage-backend
availability, pipeline-executor availability, and stable names for the
exposed support axes. The graph compiler, PipelineExecutor validation,
PipelineExecutor routing, and PipelineMaterializer now consume the
central capability truth for their covered axes. Browser-visible node
metadata now consumes fail-closed runtime truth for matching nodes and
pushes the blocked badge/reason into the existing add-node search and
node browser metadata path. Remaining work is broader frontend
presentation of all support axes, not another parallel UI support list.

Scope:

- engine node metadata / runtime support layer

Deliverable:

- one authoritative support matrix used by frontend and runtime

### Ticket E: Arrow vs Parquet parity

Status: completed in this branch. The shared setup test now creates real
Parquet-backed datasets and verifies tabular batch shape plus
multi-row-group tabular split behavior, time-series Arrow/Parquet
partition and regression-label shape parity, and multi-row-group
time-series partition filtering. It also runs matching Arrow and
multi-row-group Parquet model train/validation steps through the shared
model builder. The dedicated `test_training_executor_arrow_parquet`
target now runs the real `TrainingExecutor::Train()` loop for both an
in-memory Arrow dataset and the same data written as multi-row-group
Parquet, verifying epoch callbacks, completion callbacks, finite
train/validation losses, batch counts, and metric history.

Scope:

- `cyxwiz-engine/src/core/training_manager.cpp`

Deliverable:

- shared input-size derivation helper
- parity tests for tabular and time-series flows

---

## Bottom Line

The main runtime problem in CyxWiz is not lack of features.

It is that multiple execution systems exist at once, and several of them
still fail open. That makes the platform harder to trust than it needs
to be.

The immediate engineering priority should be:

1. stop unsafe and misleading execution behavior
2. centralize support truth
3. converge on one canonical runtime path

That will improve correctness faster than adding more nodes on top of
the current drift.
