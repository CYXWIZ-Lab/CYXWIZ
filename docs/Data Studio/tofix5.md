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
   Runtime implementation ownership is now explicit in the capability
   registry, but the broader executor/materializer migration plan is
   still open.
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

**Status 2026-06-07 follow-up 2:** The remaining active legacy
single-input branches (`FilterRows`, `SelectColumns`,
`RemoveDuplicates`, `SaveDataset`, `FillMissing`, `SortRows`, `GroupBy`,
and `DeployToNodeEditor`) now use the shared binding helper instead of
duplicating first-edge `node.inputs[0]` lookups. The only direct
`ctx.node_results` lookup for input binding is now centralized in
`GetInputDatasetNames()`.

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

**Status 2026-06-07 follow-up:** `MathFormula` no longer passes raw
formula text straight into DuckDB SQL. The executor now rewrites a small
allowed expression language consisting of arithmetic operators, numeric
literals, parentheses, and existing numeric column names. Unknown
columns, text columns, function calls, quoted strings, semicolons, and
other raw SQL tokens fail before query construction.

**Status 2026-06-07 follow-up 2:** `SaveDataset` now matches its Data
Studio node contract when a `path` is supplied: it exports through
DataRegistry using supported `csv` and `parquet` formats while preserving
the legacy in-memory `name` alias behavior. Unsupported formats such as
`arrow` now fail validation through the central allowed parameter registry
instead of being advertised and ignored.

**Status 2026-06-08 follow-up 38:** `SaveDataset` now also publishes its
saved/aliased dataset name into the shared executor node-result binding.
Graphs that use `SaveDataset` as an intermediate node can feed downstream
transforms instead of losing the dataset after a successful save/export.

**Status 2026-06-08 follow-up 39:** `DataOutput.format=json` and
`SaveDataset.format=json` now fail validation instead of being advertised
as supported while reaching the `DataRegistry::ExportArrowToJSON()` stub.
Executable export formats are `csv` and `parquet` until native Arrow-table
JSON export is implemented.

**Status 2026-06-08 follow-up 58:** `DataOutput` now treats the UI-authored
`file_type` parameter as the same export-format contract as `format`.
`file_type=parquet` exports through `DataRegistry::ExportArrowToParquet`,
unsupported dialog choices such as `json` fail validation, and conflicting
`format` / `file_type` values fail closed instead of silently defaulting to
CSV.

**Status 2026-06-08 follow-up 59:** The DataOutput dialog, node defaults,
and browser metadata now advertise only the executable CSV and Parquet
formats. The UI no longer presents TSV, JSON, Excel, or HDF5 as selectable
DataOutput formats while the runtime lacks those exporters.

**Status 2026-06-08 follow-up 60:** Browser-visible DataInput metadata now
matches the PipelineExecutor-supported file formats. The DataInput node no
longer advertises JSON, Excel, or HDF5 in keywords or file filters; it lists
the supported CSV, TSV, Parquet, Feather, Arrow, and IPC formats instead,
with metadata drift coverage.

**Status 2026-06-08 follow-up 61:** The DataInput dialog's tabular format
picker now offers only executable Data Studio graph formats: Auto, CSV, TSV,
Parquet, Feather, and Arrow/IPC. If an unsupported file extension is
auto-detected, the dialog keeps the detected label but shows a warning
instead of rendering JSON, Excel, or HDF5-specific controls.

**Status 2026-06-08 follow-up 62:** The DataInput tabular loader validation
now fails closed before launching async work for unsupported detected file
types beyond JSON and Excel, including HDF5, TXT, and ARFF. TXT users are
directed to the Text source path instead of being treated as tabular CSV.

**Status 2026-06-08 follow-up 63:** `DataInput.file_type` is now a runtime
alias for `DataInput.type`, matching the DataInput node defaults and older
graphs that persisted `file_type`. Both aliases share the same executable
format list (`auto`, `csv`, `tsv`, `parquet`, `feather`, `arrow`, and
`ipc`), unsupported aliases such as `file_type=json` fail validation, and
graphs that specify contradictory `type` / `file_type` values fail closed.
`ExecuteDataInput()` now resolves the same alias before loading, so
`file_type=csv` does not silently fall back to auto-detect.

**Status 2026-06-08 follow-up 64:** The tabular async loader no longer keeps
dead support-era branches for JSON, Excel, TXT, or ARFF after validation has
already rejected those file types. Its Apply context comment and async branch
logic now match the supported tabular list, reducing the chance that future
work reuses stale loader code as if those formats were executable.

**Status 2026-06-08 follow-up 40:** `DeployToNodeEditor` now also
publishes its deployed dataset name into the shared executor node-result
binding. It still marks the graph deployment-ready, but can now be used
as an intermediate node without losing the downstream dataset binding.

**Status 2026-06-07 follow-up 3:** `ExportCSV` now accepts both the
legacy `file_path` parameter and the Data Studio registry's `path`
parameter. Validation treats either spelling as satisfying the required
output path, so UI-authored ExportCSV nodes no longer fail before
execution.

**Status 2026-06-07 follow-up 4:** Active legacy time-series SQL
branches (`TSWindow`, `TSFeatures`, `TSLag`, and `TSDiff`) now validate
that their source columns exist and are numeric before query
construction, and quote source/output identifiers instead of inserting
raw column names into DuckDB SQL.

**Status 2026-06-08 follow-up 17:** `TSWindow.stride` now fails closed
for values other than `1`. The legacy SQL branch still builds every-row
lag windows and does not implement strided window emission, so non-`1`
stride values are no longer accepted and ignored.

**Status 2026-06-07 follow-up 5:** Active legacy text SQL branches
(`TextClean`, `TextTokenize`, and `TextVectorize`) now validate that
their configured text columns exist and are string-typed before query
construction, and quote source/output identifiers before building
DuckDB SQL.

**Status 2026-06-07 follow-up 6:** `TextTokenize.method` and
`TextVectorize.method` now validate through the central allowed-parameter
registry. Unsupported values such as `ngram` for the legacy tokenizer or
`tfidf` for the legacy count-style vectorizer fail validation instead of
falling back to a weaker execution path.

**Status 2026-06-08 follow-up 16:** `TextClean.remove_stopwords=true`
now fails closed during executor validation instead of being accepted and
silently ignored. The implemented legacy text-clean path remains limited
to HTML removal, special-character removal, lowercasing, whitespace
normalization, and trimming until stopword removal has a real backend.

**Status 2026-06-08 follow-up 18:** Legacy `TextVectorize.max_features`
now fails closed instead of being accepted by the simple `text_length` /
`word_count` branch and ignored. Vocabulary-size capping belongs to the
operator-backed `CountVectorizer` and `TFIDFVectorizer` nodes.

**Status 2026-06-08 follow-up 19:** Operator-backed text vectorizers now
also fail closed for unsupported non-default knobs instead of ignoring
them: `CountVectorizer.binary=true`, non-default `CountVectorizer.ngram_range`,
and non-default TF-IDF n-gram / `min_df` settings now report explicit
configuration errors until those semantics are implemented.

**Status 2026-06-08 follow-up 20:** Text vectorizer boolean parameters now
validate strictly. Malformed `CountVectorizer.binary`, `TFIDFVectorizer.use_idf`,
and `TFIDFVectorizer.smooth_idf` values now fail with explicit errors
instead of being interpreted as `false`.

**Status 2026-06-08 follow-up 21:** Additional operator-backed boolean
parameters now validate strictly at configure time. Malformed
`TextTokenizer.lowercase`, `LinearRegressionNode.fit_intercept`, and
`ExponentialSmoothing.damped` values now fail with explicit errors
instead of being interpreted as `false`.

**Status 2026-06-08 follow-up 22:** Operator-backed PCA and scaler
boolean parameters now validate strictly as well. Malformed
`PCANode.center` / `scale`, `StandardScaler.with_mean` / `with_std`, and
`RobustScaler.with_centering` / `with_scaling` values now fail with
explicit errors, and reconfiguring scaler operators no longer leaves a
stale optional `label_col` behind.

**Status 2026-06-08 follow-up 23:** The central required-parameter axis
now covers static requirements for operator-backed text, time-series,
signal-processing, regression, and categorical-encoder nodes. Missing
parameters such as `TextTokenizer.text_col`,
`LinearRegressionNode.target_col`, `Convolution1D.kernel`, and
`LabelEncoder.column` now fail during executor validation instead of
after upstream source execution reaches operator configuration.

**Status 2026-06-08 follow-up 24:** The central allowed-parameter axis
now covers scalar enum choices for operator-backed vectorizers,
sentiment, clustering, filter design, decomposition, smoothing, and
encoder/outlier nodes. Unsupported values such as
`CountVectorizer.norm=cosine`, `KMeansCluster.init=forgy`, or
`ExponentialSmoothing.method=ets` now fail during executor validation.
`TimeSeriesDecomposition.period` is also part of the required-parameter
axis instead of falling through to a later local period-range error.

**Status 2026-06-08 follow-up 25:** The central integer-parameter axis
now covers operator-backed minimum bounds that match current runtime
semantics, including time-series window/features/differencing, text
tokenizer/vectorizers, PCA, clustering iteration/count parameters,
polynomial regression degree, filter order, and decomposition period.
Invalid values such as `TimeSeriesWindow.input_width=0`,
`TimeSeriesFeatures.lag_values=1,0`, `PCANode.n_components=0`,
`KMeansCluster.max_iter=0`, and `TimeSeriesDecomposition.period=1`
now fail before upstream source execution reaches operator
configuration. Cross-field rules and sentinel values such as `-1` auto
modes remain local because the current central axis intentionally models
only simple integer minimums.

**Status 2026-06-08 follow-up 26:** Audited operator-backed
`Configure()` implementations now reset optional fields and documented
defaults before parsing new params. Reusing an operator instance no
longer carries stale labels, column lists, enum choices, numeric bounds,
time-series analysis settings, regression degree, clustering defaults,
or signal-processing settings from a previous configuration. The
executor normally creates a fresh operator per node, but direct
operator/materializer tests and future pooling paths now get idempotent
configuration semantics.

**Status 2026-06-08 follow-up 27:** Added a direct
`test_operator_configure_resets` regression target for representative
operator reuse. It verifies `CountVectorizer` clears stale label and
`max_features` state, `TimeSeriesFeatures` clears stale lag lists, and
`TimeSeriesWindow` clears stale feature/time columns when the same
operator instance is configured again.

**Status 2026-06-08 follow-up 28:** Central enum validation and legacy
runtime execution now agree on case-insensitive values for active
branches that already advertise allowed values. `DataInput.source_type`,
`DataOutput.format`, `FillMissing.strategy`, `TextTokenize.method`,
`TextVectorize.method`, and `StringManipulation.operation` are normalized
before execution, so values accepted by `ValidatePipeline()` no longer
fail or take fallback behavior later because of casing.

**Status 2026-06-08 follow-up 29:** Active legacy boolean-like
parameters now validate and execute consistently for `DataInput` and
`TextClean`. Malformed `DataInput.has_header` / `json_lines` and
`TextClean.lowercase` / `remove_html` / `remove_special_chars` /
`remove_stopwords` values fail validation, while accepted casing such as
`TRUE` is normalized before execution. `TextClean.remove_stopwords=TRUE`
still fails closed because stopword removal is not implemented in the
legacy SQL branch.

**Status 2026-06-08 follow-up 30:** `StringManipulation.operation`
cross-parameter validation now uses the same normalized enum value as
execution. Case variants such as `REPLACE` and `SUBSTRING` no longer skip
the local `param1` / `param2` requirements and then fail only after an
upstream source executes.

**Status 2026-06-08 follow-up 31:** `DataInput.source_type` is now
normalized before branch-specific validation checks as well as during
execution. Case variants such as `FILE` now still enforce file-only
integer rules like `skip_rows` / Excel `sheet_idx` before loader
execution can reach `std::stoi`.

**Status 2026-06-08 follow-up 32:** Runtime support now has a central
float-parameter validation axis. `TimeSeriesSplit.train_ratio`,
`val_ratio`, and `test_ratio` are validated as numeric values in the
supported `0..1` range before upstream source execution reaches operator
configuration. The same axis now covers `RobustScaler.quantile_min` and
`quantile_max` in the implemented `0..100` range.

**Status 2026-06-08 follow-up 33:** `OutlierDetector.method` now
normalizes the centrally accepted enum value before operator
configuration. Case variants such as `IQR` no longer pass executor
validation and then fail after upstream source execution.

**Status 2026-06-08 follow-up 34:** Operator-backed clustering enum
parameters now normalize centrally accepted values before local operator
configuration. Case variants and padded values for `KMeansCluster.init`,
`DBSCANCluster.metric`, `HierarchicalCluster.linkage`,
`HierarchicalCluster.metric`, and `GMMCluster.covariance_type` no longer
pass executor validation and then fail or trip cross-field checks after
the source node executes.

**Status 2026-06-08 follow-up 35:** Operator-backed text enum
parameters now share a text-operator normalization helper before local
configuration. Case variants and padded values for `CountVectorizer.norm`,
`TFIDFVectorizer.norm`, and `SentimentAnalyzer.method` no longer pass
central validation and then fail when the text operator configures.

**Status 2026-06-08 follow-up 36:** Signal and time-series analysis enum
parameters now share a numeric-series normalization helper before local
configuration. Case variants and padded values for
`FilterDesigner.filter_type`, `TimeSeriesDecomposition.method`,
`TimeSeriesDecomposition.algorithm`, and `ExponentialSmoothing.method`
no longer pass central validation and then fail or take the wrong local
branch after the source node executes.

**Status 2026-06-08 follow-up 37:** `DataInput.source_type=ml_dataset`
now fails closed during executor validation instead of being advertised
as supported and then reaching the placeholder `LoadMLDatasetToArrow()`
path, which always returned no dataset. The executable DataInput source
types are `file` and `folder` until ML dataset Arrow loading has a real
backend bridge.

**Status 2026-06-07 follow-up 7:** Legacy text and time-series SQL
branches now require their explicit source-column selectors in the
central required-parameter registry. `TextClean`, `TextTokenize`, and
`TextVectorize` require `text_column`; `TSWindow` requires
`target_column`; and `TSFeatures`, `TSLag`, and `TSDiff` require
`columns`, so validation no longer falls through to implicit `text` or
`value` defaults.

**Status 2026-06-07 follow-up 8:** `RemoveDuplicates.columns` is now a
real, schema-checked selector instead of ignored metadata. When the
selector is supplied, the executor validates and quotes those columns,
deduplicates by that key, and preserves the original table schema. Missing
dedupe columns now fail before DuckDB query construction.

**Status 2026-06-07 follow-up 9:** `FillMissing` constant mode now
builds type-aware SQL constants instead of inserting the raw configured
value into DuckDB SQL. String constants are quoted as literals, numeric
columns reject nonnumeric constants before query construction, and
unsupported column types fail closed with a specific validation error.

**Status 2026-06-08 follow-up 41:** `FillMissing` numeric statistic
strategies now fail closed on nonnumeric table columns instead of
silently passing those columns through. `mean` and `median` require
numeric columns because the node has no per-column selector; `constant`
and `mode` remain the table-wide strategies for nonnumeric data.

**Status 2026-06-08 follow-up 42:** The central float-parameter
validation axis now supports optional/exclusive bounds and covers more
operator-backed scalar checks that previously failed only after source
execution reached operator configuration. `TargetEncoder.smoothing`
requires a nonnegative number, while `OutlierDetector.threshold`,
`DBSCANCluster.eps`, `FilterDesigner.cutoff`, and
`FilterDesigner.sample_rate` require positive finite numbers. Cross-field
checks such as band-filter `cutoff_high > cutoff` remain local operator
validation.

**Status 2026-06-08 follow-up 43:** The central integer-parameter
validation axis now supports explicit forbidden scalar values, covering
time-series analysis controls whose valid domain is an auto sentinel plus
a positive range. `ACFNode.max_lag`, `ACFNode.lags`,
`PACFNode.max_lag`, and `PACFNode.lags` now reject `0` before source
execution while still accepting `-1` auto mode and positive lag counts.
The same grouped pass moved `StationarityTest.max_lags`,
`SeasonalityDetector.min_period`, and `FFTNode.sample_rate` scalar bounds
into central executor validation.

**Status 2026-06-08 follow-up 44:** Operator-backed text analytics now
share an executor-bound input schema check before `Apply()`. `TextTokenizer`,
`CountVectorizer`, `TFIDFVectorizer`, and `SentimentAnalyzer` validate
that `text_col` exists and is `string`/`large_string`; optional
`label_col` now fails early if missing or if its type is not one of the
label types supported by the text operators. This keeps the real operator
checks as a backstop while moving common text schema rejection out of the
hot operator execution path.

**Status 2026-06-08 follow-up 45:** Operator-backed signal and
time-series source columns now share the same executor-bound schema
check. `FFTNode`, `Convolution1D`, `FilterDesigner`,
`TimeSeriesDecomposition`, `ACFNode`, `PACFNode`, `StationarityTest`,
`SeasonalityDetector`, `ARIMAForecaster`, and `ExponentialSmoothing`
validate `signal_col` as numeric before `Apply()`. `TimeSeriesWindow`,
`TimeSeriesFeatures`, `LogTransform`, and `Differencing` do the same for
`value_col`. Multi-feature `feature_cols`/`columns` auto-detect behavior
remains a separate schema-validation slice.

**Status 2026-06-08 follow-up 46:** Explicit multi-column selector schema
checks now cover the operator-backed analytics/preprocessing families.
`StandardScaler`, `MinMaxScaler`, `RobustScaler`, and `OutlierDetector`
validate explicit `columns` as numeric before `Apply()`. `PCANode` and
the clustering operators validate explicit `feature_cols` as numeric
while preserving their numeric auto-detect behavior when the selector is
empty. `LinearRegressionNode` validates numeric `feature_cols` plus
numeric `target_col`; `PolynomialRegressionNode` validates numeric
`feature_col` and `target_col`. `LabelEncoder`, `OrdinalEncoder`, and
`TargetEncoder` now reject non-string categorical selectors early, and
`TargetEncoder.target_col` must be numeric. The remaining schema/type
work is now narrower: node families with custom expression semantics,
storage-mode-specific checks, or less common selector roles still need
separate audits.

**Status 2026-06-08 follow-up 47:** Optional exclusion-only `label_col`
selectors are now schema-checked for the numeric auto-detect families
that use them to keep labels out of feature selection. `StandardScaler`,
`MinMaxScaler`, `RobustScaler`, `OutlierDetector`, and the clustering
operators fail early when a supplied `label_col` is missing instead of
silently including the intended label in auto-detected numeric features.
The same pass preserved `OutlierDetector.columns=all` as the documented
auto-detect spelling, including case variants accepted by central
validation, instead of treating `all` as a literal column name.

**Status 2026-06-08 follow-up 48:** `TimeSeriesWindow` now schema-checks
its optional multivariate and plotting selectors before `Apply()`.
`feature_cols` entries and `time_col` must exist and be numeric, matching
the operator's v1 contract for extra feature blocks and
`__window_start_time` metadata. This closes another less-common selector
role that previously depended on late operator failures.

**Status 2026-06-08 follow-up 49:** Cross-field and custom-list
validation now covers the remaining small operator-backed parameter
rules that were still local to `Configure()`. `TimeSeriesSplit` rejects
zero train splits and ratio totals that do not sum to 1.0,
`RobustScaler` rejects inverted quantile bounds, `Convolution1D.kernel`
must be a comma-separated finite-number list, and band filters require
`FilterDesigner.cutoff_high > cutoff` before source execution.

**Status 2026-06-08 follow-up 50:** The capability-metadata drift guard
now accepts open-ended float bounds and checks inclusive/exclusive bound
flags, matching the runtime capability model used by positive and
nonnegative scalar parameters. This keeps open-ended rules such as
`TargetEncoder.smoothing` and `FilterDesigner.cutoff_high` under the same
registry test instead of treating them as invalid metadata.

**Status 2026-06-08 follow-up 51:** The active Data Studio node catalog
now checks enum parameter values against the central runtime
allowed-parameter axis when one exists. `SaveDataset.format` no longer
advertises `json`; the catalog exposes only the executable `csv` and
`parquet` formats until JSON export has a real Arrow-table backend.

**Status 2026-06-08 follow-up 52:** Legacy `FileInput.format` is now a
real runtime parameter instead of ignored catalog metadata. The active
catalog no longer advertises JSON for `FileInput`, central validation
rejects unsupported format values, and explicit `csv` / `parquet`
choices route through the matching DataRegistry Arrow loaders while the
no-format path keeps existing auto-detect behavior.

**Status 2026-06-07 follow-up 10:** `FilterRows.condition` now uses a
small schema-checked condition language instead of appending raw text to
DuckDB SQL. The executor accepts column comparisons against numeric or
quoted string literals, with `AND`/`OR` and parentheses, quotes resolved
column identifiers, and rejects unknown columns, type-incompatible
literals, unsupported operators, and raw SQL tokens before query
construction.

**Status 2026-06-07 follow-up 11:** File-source `DataInput.type` is now
validated through the central allowed-parameter registry instead of
falling through to late loader failures for unsupported declared formats.
Supported declared types are `auto`, `csv`, `tsv`, `parquet`, `feather`,
`arrow`, and `ipc`. The legacy `Binning.method`
contract also accepts the documented `equal_frequency` alias and
normalizes it to the implemented `equal_freq` execution path.

**Status 2026-06-08 follow-up 53:** `DataInput.type=json` and
`DataInput.type=excel` now fail validation instead of being advertised as
PipelineExecutor-supported while reaching incomplete DataRegistry loader
paths. JSON loading still falls through to unsupported Arrow auto-detect,
and Excel loading still returns no dataset, so those formats stay outside
the executable PipelineExecutor type list until real Arrow-table loaders
exist.

**Status 2026-06-08 follow-up 54:** The direct tabular DataInput loader
now also fails closed for JSON and Excel before launching async load work,
with a worker-path backstop for callers that bypass precheck validation.
This keeps the GUI apply path aligned with the PipelineExecutor support
truth while preserving CSV, TSV, Parquet, Feather, Arrow, and IPC paths.

**Status 2026-06-08 follow-up 55:** The legacy `ExcelInput` source node
now shares the same support truth as `DataInput.type=excel`: it is
recognized as a source-shaped node but marked fail-closed in the central
runtime registry until a real Excel Arrow loader exists. Fail-closed
runtime support now takes priority over typed legacy dispatch, preventing
`ExcelFile` metadata from reaching the stale `ExecuteExcelInput()` path.

**Status 2026-06-08 follow-up 56:** `TableCropper` now validates the
`end_row >= start_row` cross-field rule centrally before execution when
`end_row` is not the `-1` tail sentinel. Invalid crop ranges no longer
load the upstream table before failing in the slice path.

**Status 2026-06-08 follow-up 57:** `DataInput.source_type=folder` now
fails validation for non-image `file_category` values on the
PipelineExecutor path. The executor's folder branch only implements
`LoadImageFolderToArrow()`, so audio/text folders no longer pass runtime
validation and then fall into the image-folder loader.

**Status 2026-06-08 follow-up 12:** Cycle validation is covered by
`ValidatePipeline()` through `TopologicalSort()`, and the executor routing
test now locks that behavior with a cyclic two-node graph. Disconnected
graphs, dangling links, missing required inputs, unsupported node types,
and required/allowed parameter baselines are already validated for the
covered runtime paths. Remaining validation work is broader schema/type
coverage for node families that still do not have loaded-table checks.

**Status 2026-06-08 follow-up 13:** `RenameColumns.mapping` is now part
of the central required-parameter capability registry instead of being
only a local validator special case. The executor still accepts the
legacy `rename_map` alias for compatibility, but runtime support truth
now advertises the canonical required parameter.

Relevant files:

- `cyxwiz-engine/src/core/pipeline_executor.cpp:157`

Problem:

- validation no longer only checks the most basic shape of the node list,
  but loaded-table schema/type coverage is still uneven across node
  families
- some historical documentation still treated already-covered graph-shape
  checks as pending

Effect:

- schema/type-invalid graphs can still reach runtime for node families
  without dedicated validation beyond the current source parameter,
  legacy scalar-integer baseline, and the first loaded-table column
  checks for string/numeric transform nodes
- some runtime correctness still depends on late execution-time failures

Recommendation:

- keep the current structural validation guarantees in place for
  disconnected graphs, dangling links, cycles, missing required inputs,
  unsupported node types, and required/allowed parameter baselines
- continue expanding loaded-table schema/type checks for node families
  that still depend on execution-time failures rather than pre-query or
  configure-time rejection

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

**Status 2026-06-07:** Partially fixed for support truth. Runtime
support, materializer scope, and implementation ownership now live in
`pipeline_runtime_capabilities.{h,cpp}` and are covered by drift tests.
The remaining broad issue is execution-path ownership and migration, with
drift guards now catching factory/runtime list gaps as they appear.

**Status 2026-06-08 follow-up:** The factory/runtime migration guard now
checks both directions: operator-backed runtime capabilities must have a
factory operator, and factory-registered operators must resolve through
central operator-backed runtime support. `Identity`, `ACFNode`,
`PACFNode`, `StationarityTest`, and `SeasonalityDetector` have been moved
out of the factory-only gap into central operator-backed capability
truth.

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
- They now share central support and owner truth, but still do not have
  one canonical graph execution owner.

Effect:

- duplicated runtime logic
- lower risk of duplicated node capability decisions than before, but
  support still depends on which execution path actually owns the graph
- operator-backed nodes can be available through `PipelineOperatorFactory`
  while legacy-only nodes still execute through `PipelineExecutor`
- future migrations need to move ownership, not add another support list

Recommendation:

- use `implementation_owner` as the migration source of truth
- migrate legacy-dispatched nodes to operator-backed ownership in small
  slices with tests
- keep `PipelineExecutor` compatibility paths explicit until their nodes
  move or are removed

---

### 7. `PipelineMaterializer` is still a narrow v1 path

**Severity:** Medium

**Status 2026-06-07:** Partially fixed for truth reporting. The
materializer still only transforms in-memory Arrow tables, but
`MaterializeResult` now reports the detected source kind and whether a
non-Arrow source was skipped as unsupported for materialization. Tests
cover Arrow-table materialization and legacy text pass-through. Parquet,
image, audio, and legacy text sources remain non-materialized paths.

**Status 2026-06-07 follow-up:** `MaterializeResult` now also carries
the central unsupported-source reason from the materializer storage
backend capability registry, and the graph training launcher logs that
reason when materialization passes through a non-Arrow source.

**Status 2026-06-07 follow-up 2:** `GraphTrainingLaunchResult` now
carries the materializer source kind, unsupported-source skip flag, and
central skip reason, so callers can surface the same materializer truth
without scraping logs.

**Status 2026-06-07 follow-up 3:** Arrow-table materialization now
fails closed for branched operator paths. The v1 materializer still only
supports a linear operator path from the selected data input, but it no
longer applies sibling operator branches sequentially to one table as if
that represented the graph faithfully.

**Status 2026-06-08 follow-up 4:** Arrow-table materialization now uses
central `PipelineRuntimeSupport` axes to decide whether a node is
materializable. `PipelineOperatorFactory` is still used to construct an
approved operator, but factory registration alone no longer expands
materializer coverage. A regression test advertises a factory-only
`SVMClassifier` operator and verifies the materializer ignores it because
runtime capability truth marks that node fail-closed.

**Status 2026-06-08 follow-up 5:** The v1 Arrow-table materializer now
fails closed on cycles reachable from the selected data input before its
linear graph walk runs. Upstream graph compilation already rejects
cycles for normal training launch, but `MaterializeTable()` no longer
silently suppresses revisits if it is called directly.

**Status 2026-06-08 follow-up 6:** Named-source Arrow-table
materialization now binds to a matching `DataInput`/`DatasetInput`
instead of falling back to the first data input. If callers provide a
source dataset name and the graph has no matching source node, the
materializer fails closed before applying operators, preventing stale
preprocessing paths from being applied to the active dataset.

Relevant files:

- `cyxwiz-engine/src/core/pipeline_materializer.h:34`
- `cyxwiz-engine/src/core/pipeline_materializer.cpp:52`
- `cyxwiz-engine/src/core/pipeline_materializer.cpp:100`
- `cyxwiz-engine/src/core/pipeline_table_materializer.cpp`

Problem:

- materialization only runs for in-memory Arrow datasets
- Parquet-backed, image, audio, and text datasets are explicitly
  skipped; this is now explicit in the materializer result instead of
  only implicit in a debug log
- traversal is a BFS from `DataInput`
- cycles and parallel preprocessing branches are explicit v1 fail-closed
  graph shapes
- named Arrow-table materialization now requires the selected graph source
  node to match the active source dataset

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

**Status 2026-06-07:** Partially fixed for capability truth.
`PipelineRuntimeSupport` now carries an optional `gui::NodeType`, and
`pipeline_runtime_capabilities.{h,cpp}` exposes lookup in both directions:
legacy runtime name to `NodeType`, and `NodeType` back to the canonical
runtime name. Operator-backed, fail-closed, and legacy-dispatched
capabilities are covered by drift tests. The original direct
`PipelineExecutor::ExecuteNode()` string dispatch chain has since been
removed; remaining string storage is compatibility state on parsed
legacy nodes, not the active dispatch mechanism.

**Status 2026-06-07 follow-up:** `PipelineExecutor::ParsePipeline()`
now resolves each parsed node to the central optional runtime
`gui::NodeType`, and validation/operator routing consume that typed
identity before falling back to the legacy string name. Ambiguous legacy
aliases remain string-only until their dispatch branches are migrated.

**Status 2026-06-07 follow-up 2:** The first exact legacy-dispatched
nodes now execute through typed `gui::NodeType` cases in
`PipelineExecutor`, including DataInput/DataOutput, core tabular
transforms, Join/GroupBy, ExportCSV, RowToColumnNames, TableCropper,
StringManipulation, MathFormula, and RenameColumns. Ambiguous or legacy
alias names such as FileInput, SaveDataset, ExcelInput, text aliases,
and old time-series aliases remain on string dispatch until their
canonical typed ownership is clarified.

**Status 2026-06-07 follow-up 3:** `FileInput` and `ExcelInput` now
resolve to `CSVFile` and `ExcelFile` runtime types respectively, and the
executor routes them through typed legacy dispatch instead of direct
string branches. The remaining direct string branches are now
`SaveDataset`, `DeployToNodeEditor`, text aliases, old time-series
aliases, `PolynomialFeatures`, and `Binning`.

**Status 2026-06-07 follow-up 4:** The remaining string-only legacy
aliases now carry an explicit `PipelineLegacyDispatchKind` in the
central runtime capability registry. `PipelineExecutor::ExecuteNode()`
routes those aliases through the resolved runtime support object instead
of comparing raw `node.type` strings, and the metadata drift guard now
requires every string-only legacy runtime entry to declare its dispatch
kind.

**Status 2026-06-08 follow-up 5:** Audited against the current executor:
`ExecuteNode()` no longer contains direct `node.type == ...` dispatch
comparisons. It first routes typed legacy nodes through
`ExecuteTypedLegacyNode()`, then operator-backed and fail-closed nodes
through `PipelineRuntimeSupport`, and finally string-only compatibility
aliases through `PipelineLegacyDispatchKind`.

Relevant files:

- `cyxwiz-engine/src/core/pipeline_executor.cpp`
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.h`
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp`

Problem:

- `PipelineExecutor::Node` still stores node type as string, but now also
  carries optional typed runtime identity from the central registry
- dispatch is no longer a raw string-comparison chain, but string-only
  compatibility aliases still exist for legacy names that do not yet map
  to a first-class `gui::NodeType`
- capability truth now bridges to `NodeType`, validation/operator routing
  consumes it, and remaining string-only aliases carry explicit dispatch
  kinds

Effect:

- lower drift risk between runtime support truth, enum metadata, and
  operator registration
- remaining bug risk is concentrated in ambiguous legacy aliases and
  runtime entries that still need first-class typed ownership

Recommendation:

- continue using parsed `PipelineExecutor::Node` runtime identity
- continue migrating string-only compatibility aliases to first-class
  typed nodes only when their ownership and UI metadata are clear
- keep `pipeline_runtime_capabilities` as the central type-to-capability
  bridge; do not add another runtime registry

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
Remaining work is architectural: choose the canonical runtime and expand
coverage where storage-mode support is still narrow. The quarantined
historical helper block has already been deleted.

Relevant files:

- `cyxwiz-engine/src/core/node_executors/pipeline_operator_factory.cpp`
- `cyxwiz-engine/src/core/pipeline_executor.cpp`

Confirmed operator-backed families include:

- utility: `Identity`
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
- time-series analysis: `TimeSeriesDecomposition`, `ACFNode`,
  `PACFNode`, `StationarityTest`, `SeasonalityDetector`,
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

**Status 2026-06-07:** Fixed for current compiler-known backend gaps.
Training and debug still share `BuildExecutableFromConfig`, but the
graph compiler now checks centralized `PipelineTrainingBackendSupport`
before either surface starts execution. Unsupported sequential-model
layers and unsupported training-control nodes are blocked during
compile with explicit registry reasons, and the graph compiler drift
test walks those capability lists to verify the block remains active.

Relevant files:

- `cyxwiz-engine/src/core/training_executor.cpp:79`
- `cyxwiz-engine/src/core/debug_executor.cpp:84`
- `cyxwiz-engine/src/core/graph_compiler.cpp`
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp`
- `cyxwiz-engine/tests/test_graph_compiler_deferred_nodes.cpp`

Problem:

- both training and debug rely on `BuildSequentialFromConfig(config_)`
- this is good for consistency
- missing layer support or build mismatch would affect both product
  surfaces if the compiler allowed those graphs through

Recommendation:

- keep the shared builder
- keep compile-time capability checks in front of training/debug
- add new backend gaps to `PipelineTrainingBackendSupport` when the
  compiler recognizes a node before the model builder can execute it

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
before pass-through. `MaterializeResult` now returns that unsupported
source reason to callers instead of leaving it only in debug logs, and
`GraphTrainingLaunchResult` carries the same source kind / skip reason
for frontend launch callers. The first compile/training backend
availability axis is now explicit through `PipelineTrainingBackendSupport`.
Resolved runtime support now also carries explicit
`pipeline_executor_supported` and implementation-owner axes.
Operator-backed and active legacy-dispatched nodes are marked
executable, fail-closed and unknown nodes are not, and
`PipelineExecutor` validation now rejects unsupported nodes from that
central axis with the registry fail-closed reason before any fake
execution branch can run. Browser-visible node metadata now also
consumes the fail-closed portion of that central truth: matching nodes
are forced to template status, carry a `Blocked` badge, and expose the
registry reason in help text. Remaining work is broader frontend
presentation of all support axes, not another separate runtime list.

**Status 2026-06-07 follow-up 4:** `NodeMetadataRegistry` now applies
fail-closed runtime capability status after built-in metadata
initialization. The add-node search and node browser already consume
that metadata, so unsupported runtime-backed nodes now inherit the
central blocked status and reason instead of relying on parallel
frontend assumptions. The drift suite verifies the badge and reason
alongside template status.

**Status 2026-06-07 follow-up 5:** Operator-backed node metadata now
also exposes the central runtime axes in help text: runtime mode,
fail mode, PipelineExecutor support, and materializer storage scope.
The drift suite verifies those labels use the same stable names as the
capability registry.

**Status 2026-06-07 follow-up 6:** Runtime support now carries an
explicit implementation-owner axis. Operator-backed nodes are owned by
`PipelineOperatorFactory`, legacy-dispatched nodes by `PipelineExecutor`,
and fail-closed nodes by no runtime owner. The drift suite verifies
owner values and stable owner labels, and operator-backed metadata
exposes the owner in its support summary.

**Status 2026-06-07 follow-up 7:** Runtime support now also carries an
optional `gui::NodeType`, and `pipeline_runtime_capabilities.{h,cpp}`
can resolve exact legacy runtime names to typed nodes and typed nodes
back to canonical runtime names. Operator-backed, fail-closed, and
clear legacy-dispatched mappings are covered by the drift suite.
Ambiguous legacy aliases remain string-only until they are migrated or
renamed to first-class typed nodes.

**Status 2026-06-07 follow-up 8:** Browser-visible metadata now consumes
the centralized training-backend support axis for registered unsupported
sequential-model layers and training-control nodes. Matching metadata
entries are forced to template/blocked state and expose the stable
training backend support mode plus the central reason in help text. The
drift suite verifies those labels and reasons.

**Status 2026-06-07 follow-up 9:** Training-control capability entries
now have matching metadata coverage for scheduler and regularization
nodes such as StepLR, ReduceOnPlateau, ExponentialLR, WarmupScheduler,
L1/L2 regularization, and ElasticNet. They register as blocked templates
and inherit the central training-backend support reason; the drift suite
now requires metadata for every unsupported training-control capability.

**Status 2026-06-07 follow-up 10:** Browser-visible metadata now carries
structured support axes in addition to the existing help-text summaries.
`NodeMetadataRegistry` populates runtime, fail-mode, pipeline-executor,
materializer, implementation-owner, compile, training, and training-backend
axes from the central capability registry, and the Node Info panel renders
those axes directly. The drift suite verifies the structured axes for
operator-backed, fail-closed, and training-blocked nodes, so the frontend
no longer has to parse prose or maintain a parallel support list for this
view.

**Status 2026-06-07 follow-up 11:** The active Data Studio node catalog now
fails closed against the same runtime capability registry before registering
built-in pipeline node IDs. Stale catalog entries such as `ArrowDataset`,
`Aggregate`, and `DetectOutliers` are no longer advertised as executable
pipeline nodes, and the catalog's `Join` parameter schema now uses the
runtime-supported `on_column` field instead of unsupported left/right key
names. The drift suite now checks every Data Studio catalog node resolves to
a PipelineExecutor-supported capability.

**Status 2026-06-08 follow-up 12:** `PipelineMaterializer` now consumes the
central runtime capability support axes for Arrow-table operator
applicability. The factory remains the construction mechanism, but
materializer support is no longer a second support list inferred from
`PipelineOperatorFactory::HasOperator()`.

**Status 2026-06-08 follow-up 13:** The central required-parameter axis
now includes `RenameColumns.mapping`. Compatibility handling for the
legacy `rename_map` spelling remains in the executor validator, but the
canonical Data Studio parameter is no longer absent from runtime support
truth.

**Status 2026-06-08 follow-up 14:** Browser-visible metadata now applies
central runtime support axes to typed legacy-executor nodes as well as
operator-backed nodes. Active metadata entries such as DataInput,
DataOutput, FilterRows, SelectColumns, table helpers, and RenameColumns
now expose runtime mode, fail mode, PipelineExecutor support,
materializer scope, and implementation owner through the same Node Info
support-axis path. String-only legacy aliases without `gui::NodeType`
metadata remain runtime-only entries until they become first-class typed
nodes.

**Status 2026-06-08 follow-up 15:** Fail-closed browser-visible metadata
now resolves the same metadata target as runtime support (`metadata_node_type`
when supplied, otherwise `node_type`) and applies the shared runtime
support-axis path. Blocked nodes now expose runtime mode, fail mode,
PipelineExecutor support, materializer scope, implementation owner, and
the central fail-closed reason through Node Info instead of only carrying
a smaller blocked-node subset.

**Status 2026-06-08 follow-up 16:** The Node Browser hover tooltip now
renders the same structured `support_axes` used by Node Info. Runtime
mode, fail mode, PipelineExecutor support, materializer scope,
implementation owner, and blocked reasons are visible from the add-node
browser without another UI-specific support list.

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
availability, pipeline-executor availability, implementation ownership,
and stable names for the
exposed support axes. The graph compiler, PipelineExecutor validation,
PipelineExecutor routing, and PipelineMaterializer now consume the
central capability truth for their covered axes. Browser-visible node
metadata now consumes fail-closed runtime truth for matching nodes and
pushes the blocked badge/reason into the existing add-node search and
node browser metadata path. Operator-backed metadata now also exposes
runtime mode, fail mode, PipelineExecutor support, and materializer
storage scope, and implementation owner from the same registry.
Registered unsupported training-backend nodes now expose the central
training support mode and reason through the same metadata path.
Remaining work is broader frontend presentation for capability axes that
still need richer filtering or badges, not another parallel UI support
list.

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
