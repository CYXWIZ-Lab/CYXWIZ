# Done 22 - Runtime Architecture Future Work

**Created:** 2026-06-10
**Source:** Follow-up after completing `done20.md`.

## Boundary

`done20.md` closed the practical runtime-architecture cleanup slice:
central runtime support truth, explicit materializer scope, tested
compatibility aliases, support-axis presentation fixes, and broader
pre-execution schema validation coverage.

This file tracks future runtime architecture work that is larger than the
completed `done20` slice. These items are not regressions in `done20`.

## Runtime Truth 2026-06-24

This section is the working truth for future runtime work. Use it before adding
new runtime code so we do not create duplicate owners, orphan paths, or
inconsistent UI claims.

### Current Owners

- `PipelineRuntimeSupport` in `pipeline_runtime_capabilities.*` is the central
  support truth for Data Studio runtime nodes. It owns operator-backed,
  fail-closed, legacy-executor, source-node, required-parameter,
  enum/numeric-bound, materializer-scope, and implementation-owner claims.
- `NodeMetadataRegistry` consumes `PipelineRuntimeSupport` to populate
  `support_axes`. Node Browser and Node Info should keep reading those axes;
  they must not introduce a second support matrix.
- `PipelineOperatorFactory` owns Cat-1 Arrow-table operators that implement
  `IPipelineOperator`.
- `PipelineMaterializer` owns graph materialization through Cat-1 operators.
  It is intentionally in-memory Arrow-table-only today; non-Arrow storage
  backends pass through with central unsupported-source reasons.
- `PipelineExecutor` owns legacy Data Studio runtime dispatch for nodes listed
  in `PipelineLegacyRuntimeCapability`. This is real runtime support, but it is
  not the desired long-term owner for new operator-backed nodes.
- `CompiledGraphPlan` already exists and owns the selected model-training graph
  structure for `GraphExecutableModel`. It is not a Data Studio
  source/transform/sink/training-launch plan.
- `GraphCompiler`, `TrainingManager`, `TrainingExecutor`,
  `GraphExecutableModel`, and `graph_training_launcher.*` own training graph
  compilation and launch. Do not route general Data Studio materialization
  through these APIs unless the graph is actually entering training.
- Backend placement claims live in `backend_placement_capabilities.h` and
  `GraphCompiler` placement output. Runtime backend exceptions still mostly
  live in backend layer/loss implementations.

### Do Not Duplicate

- Do not add a second node support table in frontend code. Add or correct
  support facts in `pipeline_runtime_capabilities.*`, then let metadata/UI
  consume `support_axes`.
- Do not create another model graph plan to solve item 1. The missing piece is
  a typed Data Studio execution plan for source/transform/sink/training-launch
  steps, built after alias resolution and validation. It should coexist with,
  not replace, `CompiledGraphPlan`.
- Do not expand `PipelineMaterializer` by silently converting Parquet, image,
  audio, or legacy text datasets to Arrow. Add explicit adapters only when they
  preserve domain semantics and tests prove the pass-through/adapter behavior.
- Do not add new string aliases directly in `PipelineExecutor` without also
  adding central capability metadata, validation constraints, and routing
  tests.
- Do not claim GPU success or CPU fallback policy in UI/docs unless the claim
  is backed by backend placement metadata or a structured backend fallback
  record.

### Current Gaps

- There is no single typed Data Studio execution plan that covers sources,
  transforms, sinks, and training launch. Existing runtime paths still combine
  `PipelineExecutor`, `PipelineOperatorFactory`, `PipelineMaterializer`, and
  training launch code directly.
- `GetPipelineLegacyRuntimeCapabilities()` contains both canonical legacy
  executor nodes and compatibility alias spellings. The item 2 list below is a
  retirement-priority subset, not the full legacy capability table.
- Schema validation now has central parameter constraints plus a bad-schema
  routing coverage guard. New executable nodes must keep that guard updated
  before being considered runtime-supported.
- ArrayFire fallback now has a shared reason-code helper, broad eval-barrier
  coverage across dense/linear layers, recurrent paths, losses, data
  transforms, dimensionality reduction, linear algebra, clustering, dropout,
  embedding, activation, evaluation, and optimizer paths, and a deterministic
  Dense forward/backward forced-fallback test hook. A source scan now guards
  future raw fallback warning strings and ArrayFire exception handlers from
  bypassing the shared reason-coded, one-time logging contract.

## Remaining Work

### 1. Canonical Graph Execution Plan

`PipelineExecutor`, `PipelineOperatorFactory`, and `PipelineMaterializer` have
clearer ownership boundaries, and `CompiledGraphPlan` already exists for
selected model-training graph structure. The missing object is a typed Data
Studio execution plan for all executable Data Studio paths.

Implement:

- a typed execution-plan representation for validated Data Studio graphs,
- explicit source, transform, sink, and training-launch plan steps,
- compatibility alias resolution before execution-plan construction,
- drift tests that fail when a node has multiple competing runtime owners,
- an explicit bridge rule for when a Data Studio execution plan hands off to
  `CompiledGraphPlan` and training launch.

### 2. Legacy Alias Retirement Or Typed Migration

The remaining string-only aliases are pinned as compatibility exceptions in
`PipelineLegacyRuntimeCapability`.

For each remaining alias, choose one of:

- migrate to first-class typed metadata,
- map to an existing canonical typed node before execution,
- keep as a documented hidden compatibility alias,
- remove after proving no supported graph depends on it.

The retirement-priority alias list is:
`SaveDataset`, `DeployToNodeEditor`, `TextClean`, `TextTokenize`,
`TextVectorize`, `TSWindow`, `TSFeatures`, `TSLag`, `TSDiff`,
`PolynomialFeatures`, and `Binning`.

Do not treat that list as the whole legacy runtime table. The table also
contains canonical legacy-executor nodes such as `DataInput`, `FilterRows`,
`SelectColumns`, `Join`, export nodes, metrics nodes, and others. Future work
should retire or type aliases without accidentally deleting real supported
legacy executor nodes.

### 3. Materializer Beyond Arrow Tables

`PipelineMaterializer` is intentionally Arrow-table-only today. Parquet-backed,
image, audio, and legacy text datasets pass through unchanged with central
unsupported-source reasons.

Future work should only expand materialization when the adapter preserves the
domain semantics:

- Parquet row-group rewrite or streaming materialization,
- typed text dataset materialization that preserves sequence/classification
  contracts,
- explicit no-op/pass-through behavior for image and audio domain datasets,
- user-visible diagnostics for skipped materialization.

### 4. Frontend Support-Axis Polish

Node Browser and Node Info consume central support axes, including availability
filters. More UI polish can happen without creating a second support matrix.

Implement:

- clearer blocked-node affordances in graph-building workflows,
- compact reasons for unsupported runtime, compile, training, and materializer
  axes,
- tests or snapshots that prove frontend filters remain backed by
  `support_axes`.

### 5. Schema Validation Coverage Maintenance

`done20.md` added coverage for less-used PipelineOperatorFactory families and
kept broad routing tests green. Future node work should keep the same standard.

For every new executable Data Studio node:

- required parameters should validate before execution,
- enum and numeric bounds should live in central runtime capability tables when
  static,
- schema checks should fail before SQL/operator execution,
- routing tests should include at least one representative bad-schema case.

Current guard:

- central validation capability tables are drift-checked against runtime
  support,
- bad-schema routing coverage is explicitly registered for each node with
  static validation capabilities,
- invalid graph schemas are asserted to fail before `DataInput`, SQL, or
  operator execution starts.

### 6. ArrayFire CUDA JIT Fusion Overflow Policy

Recent sentiment/LSTM runs exposed a repeated CUDA backend warning:

`ArrayFire LSTMLayer::Forward failed ... NVRTC_ERROR_COMPILATION ... Formal parameter space overflowed (... bytes required, max 4096 bytes allowed) ... falling back to CPU`

This is not a graph validation error. It is an ArrayFire/CUDA JIT codegen
failure caused by large fused expressions. Dense and LSTM paths can build
compound expressions such as matmul + tiled bias + gate activations + state
updates. ArrayFire lazily fuses those into CUDA kernels; when the generated
kernel argument footprint exceeds the backend limit, NVRTC refuses to compile
the kernel and the engine falls back to CPU.

Target this as a cross-node backend policy, not a one-off LSTM issue:

- audit ArrayFire-first node paths for large lazy expression chains,
- add `eval()` barriers after matmul, bias add, joins/slices, gate activation,
  recurrent state updates, and other high-fan-in expressions,
- record a structured fallback reason such as `cuda_jit_param_overflow`,
  `arrayfire_jit_compile_failure`, or `gpu_backend_exception`,
- avoid logging the full NVRTC compiler dump every batch after the first
  occurrence for the same node/shape/backend,
- surface the user-facing message as performance fallback, not training
  failure: training continues on CPU but will be slower,
- add focused smoke tests for Dense, LSTM, GRU, attention, and loss paths that
  confirm GPU path success where available or a clean one-time fallback where
  not available,
- keep at least one deterministic forced or mocked backend-failure test for
  any new ArrayFire-first fallback policy surface,
- keep CPU fallback behavior correct and deterministic while GPU path fixes
  are incremental.

Known examples:

- Dense/Linear: use `eval()` barriers around matmul/bias, initialization, and
  backward-gradient boundaries, with shared reason-coded fallback logging.
- LSTM/GRU: forward and backward paths have recurrent placement checks,
  barriers, and structured fallback reporting for CUDA formal-parameter
  overflow and other backend failures.
- Attention, losses, clustering, data transforms, dimensionality reduction,
  linear algebra, activation, evaluation, dropout, embedding, and optimizer
  paths have been audited for the current high-risk ArrayFire expression
  chains. New ArrayFire-first code should follow the same helper and barrier
  pattern.

## Verification Targets

Future work should keep these checks green:

- `test_pipeline_operator_metadata`
- `test_pipeline_executor_operator_routing`
- `test_text_gui_training_launch`
- `test_text_loader_csv_preflight`
- `cyxwiz-engine` Debug build
