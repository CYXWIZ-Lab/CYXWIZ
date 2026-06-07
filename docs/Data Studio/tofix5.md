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
3. Done for the legacy single-input path: `GetInputDatasetName()` now
   fails closed on multiple inputs instead of silently selecting the
   first input edge.
4. Done for baseline structure/runtime support: `ValidatePipeline()`
   now rejects duplicate ids, missing/self links, invalid source/input
   shapes, unsupported multi-input paths, cycles/topology failures,
   missing source/export parameters, unsupported source/export parameter
   values, and node types unknown to the central runtime capability
   registry.
5. Still pending: `PipelineMaterializer` only materializes in-memory Arrow graphs and
   explicitly skips Parquet-backed, image, audio, and text sources.
6. Partially fixed: audited placeholder branches in `PipelineExecutor`
   now fail closed instead of returning fake success, and exact
   operator-backed names route through `PipelineOperatorFactory`.
   Remaining work is to choose the canonical executor/materializer
   ownership model for all Data Studio graph execution.
7. Done: `TrainingManager::StartTrainingArrow()` and
   `StartTrainingParquet()` now use the same shared input-size resolver,
   including the GraphCompiler time-series override.

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
inputs now fail closed instead of silently using the first edge. A real
multi-input binding contract is still pending.

Relevant files:

- `cyxwiz-engine/src/core/pipeline_executor.cpp:1377`

Problem:

- `GetInputDatasetName()` returns `node.inputs[0]` only.
- Any node that logically depends on more than one upstream input is not
  modeled correctly by this helper.

Effect:

- multi-input nodes cannot execute truthfully on this path
- future node additions can silently inherit broken semantics
- compile-time graph shape and runtime data flow can diverge

Recommendation:

- replace first-input lookup with explicit per-node input binding
- model named input pins or typed input collections
- keep failing closed for node types that require multi-input semantics
  until the executor can support them

---

### 4. `ValidatePipeline()` is too weak to protect execution

**Severity:** High

**Status 2026-06-07:** Partially fixed. Baseline structural validation
now covers duplicate ids, missing/self links, invalid source/input
shapes, unsupported multi-input paths, cycles, and required
source/export parameters such as `DataInput.file_path`,
`DataInput.folder_path`, `FileInput.path`, and output `file_path`
settings. Validation also rejects node types that are unknown to the
central runtime capability registry before execution starts. Validation
now rejects invalid `DataInput.skip_rows` and Excel `sheet_idx` integer
parameters before execution reaches `std::stoi`. Validation errors now
preserve the specific failed rule. Broader schema/type-aware validation
remains future work.

Relevant files:

- `cyxwiz-engine/src/core/pipeline_executor.cpp:157`

Problem:

- validation currently only checks the most basic shape of the node
  list
- code comments still mark disconnected-node checks, type checks, and
  parameter validation as TODOs

Effect:

- schema/type-invalid graphs can still reach runtime for node families
  without dedicated validation beyond the current source parameter
  baseline
- some runtime correctness still depends on late execution-time failures

Recommendation:

- validation must at minimum reject:
  - disconnected graphs
  - cycles for DAG-only paths
  - missing required params beyond the current source/export baseline
  - missing required inputs
  - unsupported node types for execution paths beyond the current legacy
    executor/runtime registry baseline
  - obvious type/schema mismatches beyond the current source integer
    parameter baseline

---

### 5. Arrow and Parquet training paths are not logically equivalent

**Severity:** Medium-High

**Status 2026-06-07:** Fixed for tabular/time-series input-size
derivation. Arrow and Parquet training start paths now both call
`ResolveTabularTrainingInputSize()`, which preserves the GraphCompiler
time-series `input_size` override and otherwise reserves one label
column for normal multi-column tabular data.

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
- continue auditing the rest of Arrow/Parquet parity separately

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

Relevant files:

- `cyxwiz-engine/src/core/pipeline_materializer.h:34`
- `cyxwiz-engine/src/core/pipeline_materializer.cpp:52`
- `cyxwiz-engine/src/core/pipeline_materializer.cpp:100`

Problem:

- materialization only runs for in-memory Arrow datasets
- Parquet-backed, image, audio, and text datasets are explicitly
  skipped
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

Problem now:

- exact registered operator-backed node names now route through
  `PipelineOperatorFactory`
- known unsupported legacy node names fail closed through the central
  runtime capability registry
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
`Join`, are also centralized there. Runtime support now carries the
first materializer dimension: exact operator-backed nodes are marked
Arrow-table materializer capable, while fail-closed and legacy-dispatched
nodes are not. The metadata drift test verifies every listed operator
runtime capability has a real factory operator and that operator,
fail-closed, and legacy-dispatched names do not overlap. Remaining work
is to expand this into a fuller multi-axis capability matrix for compile,
training, materializer storage scope, and backend availability.

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
  - `pipeline_supported`
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

1. Unify Arrow/Parquet input-size derivation.
2. Move schema-to-model logic into shared helpers.
3. Add execution-path tests for storage-mode parity.

---

## Good First Tickets

### Ticket A: Make `ExecuteParallel()` safe

Scope:

- `cyxwiz-engine/src/core/pipeline_executor.cpp`

Deliverable:

- remove shared mutable `ExecutionContext` from async tasks
- add deterministic result merge
- add regression test for parallel node execution

### Ticket B: Cache validity cleanup

Scope:

- `cyxwiz-engine/src/core/pipeline_executor.cpp`

Deliverable:

- do not mark non-executed nodes complete without a valid cached output
- add explicit cache-invalid behavior

### Ticket C: Replace placeholder success with fail-closed behavior

Scope:

- `cyxwiz-engine/src/core/pipeline_executor.cpp`

Deliverable:

- audit placeholder branches
- convert unsupported ones to explicit runtime errors
- preserve only intentionally simulated nodes with visible status

### Ticket D: Capability registry

Scope:

- engine node metadata / runtime support layer

Deliverable:

- one authoritative support matrix used by frontend and runtime

### Ticket E: Arrow vs Parquet parity

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
