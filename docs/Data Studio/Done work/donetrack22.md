# Done Track 22 - Runtime Architecture Execution Plan

**Created:** 2026-06-24
**Source:** Working plan for `done22.md`.

## Purpose

This file tracks execution of `done22.md`. Use it as the active checklist
when changing runtime architecture so each change has one owner, one validation
path, and one test target.

`done22.md` remains the problem statement and runtime truth. This file is the
work plan.

## Operating Rules

- Update `pipeline_runtime_capabilities.*` before changing runtime support UI.
- Add no second frontend support matrix; Node Browser and Node Info must keep
  consuming `support_axes`.
- Add no second model graph plan. `CompiledGraphPlan` remains the model
  training graph plan. Track 22 only allows a Data Studio execution plan for
  source, transform, sink, and training-launch steps.
- Do not silently convert non-Arrow datasets in `PipelineMaterializer`.
- Do not add string-only aliases directly in `PipelineExecutor` without
  central capability metadata, validation constraints, and routing tests.
- Treat backend fallback as runtime behavior, not graph validation failure.

## Phase 0 - Baseline Lock

Goal: freeze the current truth before refactoring.

Tasks:

- [x] Add a runtime-owner drift test that proves every runtime-supported node
  resolves to exactly one implementation owner:
  `PipelineOperatorFactory`, `PipelineExecutor`, or `None`.
- [x] Add a support-axis drift test that proves node metadata runtime axes come
  from `PipelineRuntimeSupport`.
- [x] Add a materializer storage-scope test that proves Arrow is supported and
  Parquet/image/audio/text pass through with central reasons.
- [x] Record the current retirement-priority alias list in a focused test.

Status 2026-06-24:

- Complete. `test_pipeline_operator_metadata` now pins the Phase 0 runtime
  truth: each runtime support entry resolves to one owner, materializer storage
  scope is exactly Arrow plus four pass-through domains, and the Track 22
  retirement-priority aliases remain legacy-executor aliases with typed
  metadata resolution.
- Existing support-axis drift coverage remains in `test_pipeline_operator_metadata`
  for operator-backed, fail-closed, and legacy-executor metadata.
- Verified:
  `cmake --build build --config Debug --target test_pipeline_operator_metadata`,
  `build\bin\Debug\test_pipeline_operator_metadata.exe`,
  `cmake --build build --config Debug --target test_pipeline_executor_operator_routing`,
  `build\bin\Debug\test_pipeline_executor_operator_routing.exe`, and
  `cmake --build build --config Debug --target cyxwiz-engine`.

Acceptance:

- `test_pipeline_operator_metadata`
- `test_pipeline_executor_operator_routing`
- `cyxwiz-engine` Debug build

## Phase 1 - Typed Data Studio Execution Plan

Goal: introduce one typed Data Studio runtime plan without replacing
`CompiledGraphPlan`.

Target owner files:

- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.*`
- new `cyxwiz-engine/src/core/data_studio_execution_plan.*` if needed
- `cyxwiz-engine/src/core/pipeline_executor.*`
- `cyxwiz-engine/src/core/pipeline_materializer.*`
- `cyxwiz-engine/src/gui/graph_training_launcher.*`

Plan shape:

- `SourceStep`: dataset/file source and resolved storage kind.
- `TransformStep`: canonical node type, owner, validation facts, input/output
  dataset binding.
- `SinkStep`: export/save/deploy target.
- `TrainingLaunchStep`: explicit handoff to graph training launch and, when
  applicable, `CompiledGraphPlan`.

Tasks:

- [x] Define the smallest typed plan structs needed by current runtime paths.
- [x] Resolve compatibility aliases before plan construction.
- [x] Build plan construction from validated Data Studio nodes/links.
- [x] Keep existing execution paths working while the plan is introduced.
- [x] Add drift tests for duplicate runtime owners and invalid owner
  combinations.
- [x] Add at least one plan test covering source -> transform -> sink.
- [x] Add at least one plan test covering source -> transform ->
  training-launch handoff.

Status 2026-06-24:

- Complete as a passive contract. `DataStudioExecutionPlan` now models typed
  source, transform, sink, and explicit training-launch handoff steps without
  replacing `CompiledGraphPlan` or changing `PipelineExecutor` execution.
- Plan construction normalizes compatibility aliases through
  `PipelineRuntimeSupport`, preserves the original string type for audit, and
  fails unsupported/fail-closed nodes before execution with central reasons.
- `test_data_studio_execution_plan` covers source -> transform -> sink, alias
  normalization, explicit training-launch handoff, and fail-closed rejection.
- `test_pipeline_operator_metadata` now guards both exactly-one runtime owner
  and mode/owner compatibility for operator-backed, legacy-executor, and
  fail-closed runtime capability tables.
- Verified:
  `cmake --build build --config Debug --target test_data_studio_execution_plan`,
  `build\bin\Debug\test_data_studio_execution_plan.exe`,
  `cmake --build build --config Debug --target test_pipeline_operator_metadata`,
  `build\bin\Debug\test_pipeline_operator_metadata.exe`, and
  `cmake --build build --config Debug --target test_pipeline_executor_operator_routing`,
  `build\bin\Debug\test_pipeline_executor_operator_routing.exe`, and
  `cmake --build build --config Debug --target cyxwiz-engine`.

Acceptance:

- A supported Data Studio graph can be validated into a typed plan.
- Unsupported nodes fail before execution with central reasons.
- Training handoff is explicit and does not duplicate `CompiledGraphPlan`.

## Phase 2 - Legacy Alias Retirement

Goal: reduce string-only runtime aliases without breaking supported old graphs.

Retirement-priority aliases:

- `SaveDataset`
- `DeployToNodeEditor`
- `TextClean`
- `TextTokenize`
- `TextVectorize`
- `TSWindow`
- `TSFeatures`
- `TSLag`
- `TSDiff`
- `PolynomialFeatures`
- `Binning`

Allowed decisions for each alias:

- migrate to first-class typed metadata,
- normalize to an existing canonical node before execution,
- keep as a documented hidden compatibility alias,
- remove after proving no supported graph depends on it.

Tasks:

- [x] Create an alias decision table in `pipeline_runtime_capabilities.*` or a
  nearby test fixture.
- [x] For each alias, choose exactly one decision.
- [x] Add tests proving aliases normalize before plan construction.
- [x] Remove direct special-case alias handling only after tests prove the
  canonical path remains covered.

Status 2026-06-24:

- Complete, intentionally conservative. `PipelineLegacyAliasDecisionCapability`
  now records one decision for every retirement-priority alias.
- Normalized now: `TextClean -> TextCleanNode`,
  `TSLag -> TimeSeriesLag`, `PolynomialFeatures -> PolynomialFeaturesNode`,
  and `Binning -> BinningNode`.
- Kept as documented hidden compatibility aliases: `SaveDataset`,
  `DeployToNodeEditor`, `TextTokenize`, `TextVectorize`, `TSWindow`,
  `TSFeatures`, and `TSDiff`. These have different output behavior,
  parameter names, or runtime contracts from their typed metadata target.
- `DataStudioExecutionPlan` consumes the decision table. Normalized aliases
  plan under the canonical type; hidden compatibility aliases keep their
  legacy runtime support and parameter contract while still being explicitly
  marked as compatibility aliases.
- Removed the dead string-only legacy dispatch fallback
  (`PipelineLegacyDispatchKind` and `ExecuteLegacyDispatchKind`). All legacy
  runtime capability entries must now resolve typed metadata and execute
  through `runtime_type`.
- Hidden compatibility aliases still keep active typed-path compatibility
  branches where behavior differs from the canonical target, for example
  `SaveDataset` preserving in-memory publishing semantics that `DataOutput`
  does not have.
- Verified:
  `cmake --build build --config Debug --target test_data_studio_execution_plan`,
  `build\bin\Debug\test_data_studio_execution_plan.exe`,
  `cmake --build build --config Debug --target test_pipeline_operator_metadata`,
  `build\bin\Debug\test_pipeline_operator_metadata.exe`,
  `cmake --build build --config Debug --target test_pipeline_executor_operator_routing`,
  `build\bin\Debug\test_pipeline_executor_operator_routing.exe`, and
  `cmake --build build --config Debug --target cyxwiz-engine`.

Acceptance:

- No alias has two runtime meanings.
- No supported legacy graph silently changes behavior.
- `PipelineLegacyRuntimeCapability` contains only intentional legacy executor
  nodes and documented compatibility aliases.

## Phase 3 - Materializer Boundary

Goal: keep materialization explicit by storage domain.

Tasks:

- [x] Preserve Arrow-table materialization as the only default behavior.
- [x] Add user-visible diagnostics for skipped non-Arrow materialization.
- [x] Add a Parquet adapter only if it preserves row-group semantics or streams
  safely.
- [x] Add a typed text adapter only if it preserves sequence/classification
  contracts.
- [x] Keep image/audio as explicit no-op/pass-through unless a real domain
  adapter exists.

Status 2026-06-25:

- Complete as an explicit boundary. `PipelineMaterializer` still applies
  Cat-1 operators only to in-memory Arrow tables.
- Non-Arrow sources pass through unchanged and now expose a stable
  `diagnostic_message` built from the central storage-backend capability
  reason.
- `GraphTrainingLaunchResult` now carries
  `materializer_diagnostic_message`, and successful non-Arrow training
  launches set `status_title/status_detail` to a user-visible materializer
  skip diagnostic.
- Parquet and typed text adapters were intentionally not added. Parquet remains
  pass-through until row-group semantics or safe streaming are preserved; text
  remains pass-through until sequence/classification contracts are preserved.
- Image and audio remain explicit no-op/pass-through domains backed by central
  reasons.
- Verified:
  `cmake --build build --config Debug --target test_text_gui_training_launch`,
  `build\bin\Debug\test_text_gui_training_launch.exe`,
  `cmake --build build --config Debug --target test_pipeline_operator_metadata`,
  `build\bin\Debug\test_pipeline_operator_metadata.exe`, and
  `cmake --build build --config Debug --target cyxwiz-engine`.

Acceptance:

- `PipelineMaterializer` never silently converts storage domains.
- Tests cover Arrow apply, non-Arrow pass-through, and diagnostics.

## Phase 4 - Support-Axis UI Polish

Goal: improve UI clarity without creating another source of support truth.

Tasks:

- [x] Make blocked-node affordances clearer in graph-building workflows.
- [x] Show compact central reasons for runtime, compile, training, and
  materializer axes.
- [x] Add tests or snapshots proving filters and blocked states are backed by
  `support_axes`.

Status 2026-06-25:

- Complete. Node add search now copies blocked state and central support
  reasons from `NodeMetadata::support_axes`, visibly disables blocked nodes,
  and prevents keyboard/click insertion for centrally blocked nodes.
- Node Browser graph-building affordances now use the same support-axis gate
  for double-click, drag-drop, and context-menu Add to Canvas actions.
- Node Browser and Node Info now show compact support reasons from central
  axes instead of introducing a UI-side support matrix.
- `test_pipeline_operator_metadata` now has a frontend support-axis regression
  guard that recreates the graph-add blocked predicate from `support_axes` and
  checks real, UI-only partial, fail-closed, and training-blocked cases.

Acceptance:

- A node's UI support status matches `PipelineRuntimeSupport`.
- UI-only logic does not decide runtime support independently.

## Phase 5 - Schema Validation Maintenance

Goal: make validation expectations enforceable for every new executable node.

Tasks:

- [x] Document the required validation checklist in the runtime capability
  tests.
- [x] Ensure required parameters, enum values, integer bounds, and float bounds
  live in central capability tables when static.
- [x] Add bad-schema routing coverage for each new executable node.
- [x] Fail before SQL/operator execution when schema is invalid.

Status 2026-06-25:

- Complete as an enforceable maintenance guard.
  `test_pipeline_operator_metadata` now documents the Phase 5
  checklist directly beside the central runtime validation guard.
- Static required-parameter, enum-value, integer-bound, and float-bound tables
  now have drift checks proving each entry resolves through
  `PipelineRuntimeSupport` to an executable runtime owner, not an unknown or
  fail-closed node.
- `test_pipeline_executor_operator_routing` now has an explicit bad-schema
  routing coverage registry. Every node named in the central static validation
  tables must appear in that registry, and stale registry entries fail once the
  corresponding validation capability is removed.
- `test_pipeline_executor_operator_routing` now also asserts validation
  preflight behavior directly: an invalid downstream node fails before
  `DataInput` loads or SQL/operator execution begins.
- Verified:
  `cmake --build build --config Debug --target test_pipeline_executor_operator_routing`
  and `build\bin\Debug\test_pipeline_executor_operator_routing.exe`.

Acceptance:

- New executable nodes without central validation and at least one bad-schema
  test are considered incomplete.

## Phase 6 - ArrayFire Fallback Policy

Goal: make GPU JIT failures structured, one-time, and safe.

Target owner files:

- `cyxwiz-backend/src/algorithms/layers/*`
- `cyxwiz-backend/src/algorithms/losses/*`
- `cyxwiz-engine/src/core/backend_placement_capabilities.h`
- `cyxwiz-engine/src/core/debug_runtime_backend_classifier.*`
- relevant smoke tests

Tasks:

- [x] Add a small backend fallback reason model for runtime exceptions:
  `cuda_jit_param_overflow`, `arrayfire_jit_compile_failure`,
  `gpu_backend_exception`.
- [x] Add one-time logging per node/shape/backend failure class.
- [x] Audit `DenseLayer` and add explicit `eval()` barriers where needed.
- [x] Audit LSTM and GRU fallback reporting after existing eval barriers.
- [x] Audit attention, losses, and reduction-heavy ArrayFire paths.
- [x] Add smoke coverage for Dense, LSTM, GRU, attention, and loss paths that
  proves either GPU success or clean CPU fallback.
- [x] Add deterministic backend-failure coverage for at least one
  ArrayFire-first layer path.

Status 2026-06-25:

- Complete for the current Track 22 scope. `DenseLayer` now has explicit
  ArrayFire `eval()` barriers after
  forward matmul, forward bias add, backward weight-gradient matmul, bias
  reduction/reshape, and input-gradient matmul.
- Shared `arrayfire_backend_utils.*` now owns the backend fallback reason
  model, stable reason-code names, backend/shape context formatting,
  fallback-message construction, and a one-time log gate keyed by operation,
  reason, backend, and shape/context.
- `DenseLayer` and `LinearLayer` ArrayFire fallback warnings now use the shared
  one-time, reason-coded messages and suppress full compiler dumps for
  `cuda_jit_param_overflow`.
- Recurrent ArrayFire fallback reporting now classifies runtime failures as
  `cuda_jit_param_overflow`, `arrayfire_jit_compile_failure`, or
  `gpu_backend_exception`. LSTM/GRU forward fallback messages use those stable
  reason codes, keep CUDA formal-parameter overflow as a performance fallback,
  and avoid logging the full NVRTC dump on the overflow message path.
- LSTM/GRU non-overflow ArrayFire fallback warnings are now one-time per
  layer/reason-code pair, so repeated batches do not keep emitting the same
  backend exception text.
- Attention audit note: current `MultiHeadAttentionLayer` forward/backward is
  CPU tensor-loop based; transformer feed-forward ArrayFire exposure runs
  through `DenseLayer`, which now has barriers.
- Reduction-heavy layer audit expanded. `SoftmaxActivation` now has explicit
  barriers after max, max-subtraction, exp, sum-exp, final normalization,
  backward grad-softmax product, backward sum, and final gradient. Its fallback
  warnings now use the shared one-time reason-coded message path.
- Pooling audit started. MaxPool2D and AvgPool2D ArrayFire paths now force
  barriers around padding, unwrap patches, max/argmax or mean reductions,
  reshapes, final outputs, cached indices, and backward gradients. Their
  fallback warnings now use shared reason-coded one-time logging.
- Data transform audit expanded. `DataTransform` GPU stats, normalization,
  standardization, log, Box-Cox, robust-scale, max-abs-scale, and power
  transform paths now force barriers before host copies and route operation
  fallbacks through shared reason-coded one-time logging with value/column
  context.
- Dimensionality reduction audit expanded. PCA covariance and t-SNE/UMAP
  squared-distance GPU helpers now force barriers around matmul, reduction,
  tiling, clamp, and host-copy boundaries, and their operation fallbacks now
  use shared reason-coded one-time logging with sample/feature context.
- Tensor-first linear algebra audit started. Tensor multiply, transpose,
  inverse, Frobenius norm, solve, and least-squares ArrayFire paths now
  materialize ArrayFire expressions before tensor/host conversion and route
  fallbacks through shared reason-coded one-time logging with input tensor
  shape context.
- Matrix linear algebra audit expanded. Add, subtract, multiply,
  scalar-multiply, transpose, inverse, determinant, QR, Cholesky, LU, solve,
  and least-squares ArrayFire paths now materialize before host reads and use
  shared reason-coded one-time fallback logging with matrix shape context.
  Inverse, Cholesky, and solve now fall through to the existing CPU algorithms
  after backend exceptions so backend failure remains a performance fallback
  while true singular/non-positive-definite inputs are still reported by the
  CPU path.
- Clustering audit expanded. K-Means, DBSCAN, hierarchical clustering, GMM,
  and clustering-evaluation ArrayFire paths now materialize distance,
  centroid, responsibility, covariance, silhouette, and host-conversion
  boundaries. GPU-only clustering failures now emit classified one-time
  backend failure logs with data/parameter context without pretending a CPU
  fallback exists.
- Dropout and embedding audit expanded. `DropoutLayer` forward/backward and
  `EmbeddingLayer` forward/backward now force barriers before tensor
  conversion and use shared reason-coded one-time fallback logging. Embedding
  max-norm normalization now emits a classified one-time backend warning when
  the ArrayFire normalization step cannot run.
- Activation audit expanded beyond Softmax. ReLU, LeakyReLU, ELU, GELU,
  Swish/SiLU, Sigmoid, Tanh, Mish, Hardswish, SELU, and PReLU now materialize
  ArrayFire expressions before tensor conversion and use shared reason-coded
  one-time fallback logging.
- Model-evaluation audit started. ROC and precision-recall GPU paths now force
  barriers around sort, mask, cumulative-sum, precision/recall, and host-copy
  boundaries, and their GPU fallbacks now use shared reason-coded one-time
  logging before the CPU implementation runs.
- Optimizer audit expanded. Adam, AdamW weight decay, NAdam, RMSprop, AdaGrad,
  Adadelta, and LAMB GPU update paths now materialize moments, caches,
  updates, and parameters before `SetFromArray`, and route GPU-step fallbacks
  through shared reason-coded one-time logging keyed by optimizer operation and
  parameter shape.
- Recurrent backward audit expanded. LSTM and GRU ArrayFire backward paths now
  force barriers around recurrent gradient accumulation and final layer
  gradients, and backend exceptions now emit shared reason-coded one-time
  fallback messages before falling through to CPU BPTT.
- Loss audit expanded. Shared ArrayFire loss reduction/softmax utilities now
  force barriers after mean/sum, max-subtraction, exp, sum-exp, and final
  normalization. Cross-entropy, focal, cosine embedding, triplet, and
  contrastive backward paths now add barriers around one-hot/mask/tile/scale
  gradient chains. Regression and probability losses now add barriers around
  diff/abs/select/sigmoid/log/exp/clamp/reduction chains.
- All current loss-family ArrayFire catch sites now use a centralized
  reason-coded, one-time fallback logger keyed by operation, backend, and input
  shape.
- `test_arrayfire_backend_utils` covers reason classification, the one-time
  operation/reason/context log gate, compiler-dump suppression, and the
  Debug-only `CYXWIZ_TEST_FORCE_ARRAYFIRE_FALLBACK` parser used to induce
  backend fallback deterministically in tests. Release builds ignore the hook.
- `test_arrayfire_backend_smoke` now exercises Dense forward/backward, LSTM
  forward, GRU forward, attention forward/backward, and cross-entropy
  forward/backward on the active backend, proving those paths either run on the
  selected backend or return clean CPU fallback tensors.
- `test_arrayfire_backend_smoke` also forces Dense forward/backward through
  the shared backend fallback policy with deterministic parameters and proves
  the CPU fallback tensors and gradients match the known Dense results.
- Raw fallback scan cleanup completed for live ArrayFire operation/init sites.
  `LinearLayer::InitializeWeights` and
  `MultiHeadAttentionLayer::InitializeWeights` now materialize initialization
  arrays before tensor conversion and use classified one-time CPU-initialization
  fallback messages. Disabled/comment-only GRU/LSTM false positives were
  rewritten so the raw scan no longer reports dead legacy fallback text.
- `test_arrayfire_fallback_source_scan` now makes the raw fallback scan
  enforceable inside `cyxwiz-tests`: new raw CPU fallback warning strings under
  `cyxwiz-backend/src/algorithms` fail unless they are explicitly allowlisted
  as shared helper text, recurrent preflight text, NCCL fallback, or GPU
  availability probing.
- The source scan now also guards `catch (const af::exception...)` handlers:
  operation fallback handlers must call the shared reason-coded/one-time
  policy path or a narrowly allowed availability/preflight probe. This exposed
  and fixed silent CPU fallthroughs in legacy ReLU/Sigmoid/Tanh activation
  files and SGD. Those paths now materialize ArrayFire expressions before
  tensor/parameter writes and log through the shared fallback policy.
- LSTM and GRU forward backend fallbacks now use the same
  operation/reason/backend/context log gate as backward paths. The context
  includes input shape, hidden size, layer count, bidirectionality, and
  `batch_first`; CUDA formal-parameter overflow messages are also gated so
  repeated batches do not emit duplicate overflow warnings after recurrent
  CUDA placement has been disabled for the process.
- Current raw scan residuals are intentional exclusions from the operation
  fallback audit: canonical shared helper message strings, recurrent preflight
  text that deliberately describes CPU routing after CUDA formal-parameter
  overflow, the distributed NCCL fallback warning, and `LinearLayer` GPU
  availability probes.
- Remaining fallback work is future-maintenance only: apply the guarded
  helper/barrier pattern to any new ArrayFire-first path and add a forced or
  mocked backend-failure test when a new path introduces its own fallback
  policy surface.
- Verified:
  `cmake --build build --config Debug --target cyxwiz-tests`,
  `cmake --build build --config Debug --target cyxwiz-engine`,
  `build\bin\Debug\cyxwiz-tests.exe "[arrayfire][fallback]"`,
  `build\bin\Debug\cyxwiz-tests.exe "[loss]"`,
  `build\bin\Debug\cyxwiz-tests.exe "[dense]"`,
  `build\bin\Debug\cyxwiz-tests.exe "[activation]"`,
  `build\bin\Debug\cyxwiz-tests.exe "[pool]"`,
  `build\bin\Debug\cyxwiz-tests.exe "[linalg]"`,
  `build\bin\Debug\cyxwiz-tests.exe "[dropout]"`,
  `build\bin\Debug\cyxwiz-tests.exe "[embedding]"`,
  `build\bin\Debug\cyxwiz-tests.exe "[optimizer]"`,
  `build\bin\Debug\cyxwiz-tests.exe "[lstm]"`,
  `build\bin\Debug\cyxwiz-tests.exe "[gru]"`,
  `build\bin\Debug\cyxwiz-tests.exe "[linear]"`,
  `build\bin\Debug\cyxwiz-tests.exe "[attention]"`,
  `build\bin\Debug\cyxwiz-tests.exe "[source_scan]"`,
  `build\bin\Debug\cyxwiz-tests.exe "[arrayfire][backend_smoke]"`,
  `cmake --build build --config Debug --target test_recurrent_backend_placement`,
  `build\bin\Debug\test_recurrent_backend_placement.exe`,
  raw scan:
  `rg -n "falling back to CPU|using CPU|GPU initialization failed|ArrayFire init failed|ArrayFire GRULayer::Forward failed" cyxwiz-backend/src/algorithms`,
  `git diff --check`,
  `cmake --build build --config Debug --target test_pipeline_executor_operator_routing`,
  and `build\bin\Debug\test_pipeline_executor_operator_routing.exe`.
- Note: building `cyxwiz-tests` and `cyxwiz-engine` in parallel can collide on
  MSVC-generated backend objects/PDB state; rerunning the targets separately
  succeeded.
- Note: the broad `test_pipeline_executor_operator_routing.exe` executable was
  attempted after the clustering audit but timed out while still progressing
  through unrelated routing cases; no focused clustering backend test target
  currently exists.

Acceptance:

- Backend fallback is surfaced as performance fallback, not training failure.
- Repeated NVRTC dumps do not spam every batch.
- CPU fallback remains deterministic.

## Definition Of Done

Track 22 is done when:

- Data Studio runtime graph execution has one typed plan contract.
- Runtime owner, support status, validation, and execution path are derived
  from central capability truth.
- Legacy aliases are typed, normalized, documented, or removed.
- Materializer behavior is explicit by storage domain.
- Frontend support state is backed by `support_axes`.
- New executable nodes cannot bypass central validation expectations.
- ArrayFire backend fallback has structured reason codes and controlled
  logging.

## Verification Targets

Keep these green after every phase:

- `test_pipeline_operator_metadata`
- `test_pipeline_executor_operator_routing`
- `test_text_gui_training_launch`
- `test_text_loader_csv_preflight`
- `cyxwiz-engine` Debug build

Add focused tests beside the phase that introduces the behavior.
