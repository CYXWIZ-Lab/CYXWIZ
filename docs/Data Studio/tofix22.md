# To Fix 22 - Runtime Architecture Future Work

**Created:** 2026-06-10
**Source:** Follow-up after completing `done20.md`.

## Boundary

`done20.md` closed the practical runtime-architecture cleanup slice:
central runtime support truth, explicit materializer scope, tested
compatibility aliases, support-axis presentation fixes, and broader
pre-execution schema validation coverage.

This file tracks future runtime architecture work that is larger than the
completed `done20` slice. These items are not regressions in `done20`.

## Remaining Work

### 1. Canonical Graph Execution Plan

`PipelineExecutor`, `PipelineOperatorFactory`, and `PipelineMaterializer` now
have clearer ownership boundaries, but there is still no single typed graph
plan object for all executable Data Studio paths.

Implement:

- a typed execution-plan representation for validated Data Studio graphs,
- explicit source, transform, sink, and training-launch plan steps,
- compatibility alias resolution before execution-plan construction,
- drift tests that fail when a node has multiple competing runtime owners.

### 2. Legacy Alias Retirement Or Typed Migration

The remaining string-only aliases are pinned as compatibility exceptions in
`PipelineLegacyRuntimeCapability`.

For each remaining alias, choose one of:

- migrate to first-class typed metadata,
- map to an existing canonical typed node before execution,
- keep as a documented hidden compatibility alias,
- remove after proving no supported graph depends on it.

The current exception list is:
`SaveDataset`, `DeployToNodeEditor`, `TextClean`, `TextTokenize`,
`TextVectorize`, `TSWindow`, `TSFeatures`, `TSLag`, `TSDiff`,
`PolynomialFeatures`, and `Binning`.

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
- keep CPU fallback behavior correct and deterministic while GPU path fixes
  are incremental.

Known examples:

- Dense/Linear: fixed with `eval()` barriers around matmul/bias and backward
  gradients.
- LSTM forward: warning observed in the Release engine log during sentiment
  training; needs full audit across unidirectional and bidirectional paths.
- GRU/attention/losses: likely candidates because they also build large gate,
  join, reduction, or matmul expression graphs.

## Verification Targets

Future work should keep these checks green:

- `test_pipeline_operator_metadata`
- `test_pipeline_executor_operator_routing`
- `test_text_gui_training_launch`
- `test_text_loader_csv_preflight`
- `cyxwiz-engine` Debug build
