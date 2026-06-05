# To Fix 2 - CyxWiz Backend Review for Engineering Pickup

This document captures a broader backend-library review of
`cyxwiz-backend` with emphasis on architecture, API coherence,
correctness, memory behavior, and performance.

The goal is to give engineers a practical backlog they can pick up in
priority order.

---

## Executive Summary

The biggest issue in `cyxwiz-backend` is not one isolated bug. It is
architectural drift:

- overlapping abstractions for the same responsibilities
- incomplete or misleading public APIs
- a Tensor model that is still CPU-primary even when GPU paths exist
- memory accounting that does not represent live usage
- repeated host-device synchronization across training and linalg paths

The backend contains a lot of working functionality, but it behaves
more like a set of individually landed subsystems than a single clean
compute library.

### Current Audit Checkpoint - 2026-06-05

This backlog has been re-audited against the current backend after the
recent Tensor/runtime work. Several original findings are now stale or
partially addressed:

- `MemoryManager` now tracks allocation sizes and decrements live bytes
  on `Deallocate()`, so the old "never decrements" finding is no longer
  current.
- `Tensor` now has cached ArrayFire state plus host/device dirty flags;
  `GetArray()` no longer blindly creates a fresh device array on every
  call when cached native device state is current.
- `SetFromArray()` keeps device data resident until host data is
  requested, so optimizer/layer `SetFromArray()` calls are not the same
  immediate CPU materialization bug described in the original review.
- The raw `new af::array` leak pattern was not found in the current
  Tensor operator implementation.

Confirmed active issues for the first implementation slice were:

- `cyxwiz_tensor_matmul()` is public in the C API but still returns "not
  yet implemented".
- `src/core/device_1.cpp` is an accidental build-command artifact and
  should be removed from the source tree.
- Two different `cyxwiz::DataLoader` classes existed in separate
  headers before the 2026-06-05 third-slice fix.
- Tensor CPU random generation used raw `rand()` in the fallback path
  before the 2026-06-05 second-slice fix.
- CPU fallback policy remains inconsistent and should be audited
  operation group by operation group.

Implementation order should favor small, verified fixes first:

1. Clean stale/artifact files and public API honesty issues.
2. Add focused tests around repaired public surfaces.
3. Document or defer compatibility-breaking architecture migrations.
4. Only then start broader naming, fallback, and performance refactors.

First slice status:

- Completed: removed `src/core/device_1.cpp`.
- Completed: implemented `cyxwiz_tensor_matmul()` through the existing
  tensor-first `LinearAlgebra::Multiply()` path.
- Completed: added focused C API coverage for successful matmul and
  invalid-shape error reporting.
- Completed: fixed the tensor-first ArrayFire matmul path to use the
  row-major Tensor/ArrayFire adapters instead of interpreting row-major
  tensor memory as native ArrayFire layout.
- Completed: aligned the C API export macro with the generated backend
  export macro, removing the C API DLL-linkage warning storm.
- Verified: `cyxwiz-tests.exe "[c_api]"` and `cyxwiz-tests.exe
  "[tensor]"` pass.

Second slice status:

- Completed: clarified `engine.h` as an intentional compatibility
  header and kept lifecycle declarations in `cyxwiz.h`.
- Completed: changed broad public header comments from
  "GPU-accelerated" to "ArrayFire-backed where available" for algorithm
  groups whose fallback coverage varies by operation.
- Completed: updated backend README wording to describe optional
  ArrayFire acceleration, operation-specific coverage, no public
  `Tensor::ToDevice()` / `Tensor::ToCPU()`, and the process-global
  device/backend selection caveat.
- Completed: removed raw `rand()` usage from `Tensor::Random()` CPU
  fallback and replaced it with a private thread-local C++ RNG helper.
- Completed: added a focused Tensor random factory range/shape test.

Third slice status:

- Completed: kept the compiled DuckDB/file SQL loader as
  `cyxwiz::DataLoader`.
- Completed: renamed the dormant dataset batch iterator in
  `dataloader.h` to `cyxwiz::TrainingDataLoader`.
- Completed: compiled `src/algorithms/dataloader.cpp` and installed
  `include/cyxwiz/dataloader.h` through the backend target.
- Completed: renamed the tensor-backed helper to
  `CreateTrainingDataLoader()` so it no longer competes conceptually
  with the DuckDB loader.
- Completed: added tests proving `data_loader.h` and `dataloader.h` can
  be included together and that `TrainingDataLoader` batches synthetic
  and tensor-backed datasets.

Fourth slice status:

- Completed: documented `SequentialModel` + `Module` as the canonical
  model-facing training/runtime stack.
- Completed: documented direct `Layer` classes as low-level primitives
  that modules may wrap, not the preferred owner type for new
  model-facing integrations.
- Completed: documented `Model` as a minimal legacy compatibility base
  for code that owns raw `Layer` instances directly.

Fifth slice status:

- Completed: audited Tensor residency against the current implementation.
  The old note that `GetArray()` always creates a fresh ArrayFire array
  and `SetFromArray()` immediately copies back to CPU is stale.
- Completed: added a regression test that device-only ArrayFire Tensor
  arithmetic keeps host memory unmaterialized until explicit CPU data
  access.
- Remaining: reduce coarse host invalidation from non-const `Data()`
  call sites and profile operator/layer paths that still bounce through
  CPU fallback or inspection APIs.

Sixth slice status:

- Completed: audited `MemoryManager`; it now records allocation sizes,
  decrements live bytes on `Deallocate()`, and exposes peak reset.
- Completed: added a direct regression test for live-byte accounting and
  `ResetPeak()`.
- Remaining: document that MemoryManager counters are scoped to
  MemoryManager-routed host buffers and do not include ArrayFire or other
  third-party allocator state.

Seventh slice status:

- Completed: hardened the public C tensor factory boundary. Tensor
  creation now rejects null shape pointers for nonzero-rank tensors,
  rejects null data for `cyxwiz_tensor_create_with_data()`, and rejects
  invalid C data type enum values before constructing backend tensors.
- Completed: added C API regression coverage for invalid factory inputs
  and the valid zero-rank scalar creation path.
- Remaining: continue the broader public header audit by subsystem; do
  not batch large algorithm fallback work into the C API contract slice.

Eighth slice status:

- Completed: corrected clustering public header comments so they no
  longer claim unconditional GPU acceleration. The header now states that
  clustering is ArrayFire-backed when built with ArrayFire and that
  non-ArrayFire builds keep the API available but return
  unsuccessful/empty results with error messages.
- Remaining: keep auditing per-algorithm public headers as each subsystem
  is touched; broad product/marketing docs should be handled separately.

Ninth slice status:

- Completed: audited Tensor arithmetic ArrayFire exception paths. The
  original raw `new af::array`/manual delete leak pattern is stale in the
  current implementation: Tensor arithmetic uses stack `af::array`
  temporaries, and cached device state is held through
  `std::unique_ptr<af::array>`.
- Completed: no code change was needed for item 7; the existing tensor
  and full backend tests continue covering the current RAII path.
- Remaining: continue RAII audits in other ArrayFire-heavy subsystems
  when those files are touched, but do not mix that broader cleanup into
  the Tensor arithmetic item.

Tenth slice status:

- Completed: audited Tensor cached ArrayFire state. The original note
  that `GetArray()` ignores `af_array_` is stale: `GetArray()`,
  `GetArrayRowMajor2D()`, and `GetArrayRowMajor3D()` now reuse current
  compatible cached device arrays and track layout with
  `TensorDeviceLayout`.
- Completed: existing tests cover lazy host materialization for
  device-backed tensors, row-major device layout preservation across
  copy/move, and host mutation invalidating cached ArrayFire data.
- Remaining: performance work remains in higher-level operators and
  optimizers that call broad host access APIs, but that belongs to the
  Priority 3 host/device churn items rather than this cached-state item.

Eleventh slice status:

- Completed: audited optimizer GPU paths. The original claim that GPU
  updates copy parameters and optimizer state back to CPU every step is
  stale for the current Tensor residency contract: GPU paths write
  updated parameters and state back with `SetFromArray()`, which keeps
  ArrayFire data resident until explicit host access.
- Completed: extended optimizer ArrayFire residency regression coverage
  beyond SGD/Adam to AdamW, RMSprop, AdaGrad, NAdam, Adadelta, and LAMB.
- Remaining: optimizer hot paths still depend on broad `GetArray()` and
  fallback behavior; profiling and reducing accidental CPU fallback belongs
  to the larger Priority 3 performance pass.

Twelfth slice status:

- Completed: audited layer/activation/loss hot paths against the current
  Tensor residency contract. Many helpers now return device-backed
  tensors through `Tensor(arr)`, `SetFromArray()`, or row-major
  `FromArrayRowMajor*()` helpers instead of explicitly copying to host.
- Completed: added focused `LinearLayer` ArrayFire residency tests for
  forward output, backward input gradient, and parameter gradients. These
  tests cover the main standalone linear training path without changing
  layer math.
- Remaining: the large legacy `layer.cpp` still contains manual
  `af::array::host(...)` copies in complex layers such as attention and
  some convolution/backward paths. Those should be handled as targeted
  layer-group slices, not as one broad refactor.

Thirteenth slice status:

- Completed: audited Tensor factory helpers. The original claim that
  `Tensor::Zeros()`, `Tensor::Ones()`, and `Tensor::Random()` allocate on
  GPU only to return CPU-owned tensors is stale for ArrayFire builds:
  successful factory paths now return `Tensor(af::array)` and keep host
  memory unmaterialized until explicit CPU access.
- Completed: added ArrayFire regression coverage proving factory-created
  tensors do not allocate tracked host memory until `Data()` is read.
- Remaining: fallback behavior and operation-specific GPU failure paths
  should be handled under the CPU fallback and performance profiling
  items, not this factory ownership item.

Fourteenth slice status:

- Completed: made the CPU fallback policy explicit for the first core
  training slice. Core, shape-preserving float32 primitives should have
  CPU implementations; advanced GPU-heavy operators may remain
  ArrayFire-required, but that requirement must be documented or tracked
  per operator group.
- Completed: added CPU fallback implementations for factory
  `ReLUActivation`, `SigmoidActivation`, and `TanhActivation`, aligning
  the factory activation stack with the standalone activation helpers.
- Completed: added CPU fallback implementations for `MSELoss`,
  `L1Loss`, and `SmoothL1Loss`/`HuberLoss` across `None`, `Mean`, and
  `Sum` reductions.
- Completed: added focused activation/loss regression coverage so the
  basic factory APIs remain usable in CPU-only builds.
- Remaining: advanced activations, classification/embedding losses,
  legacy layers, and partial linalg fallbacks still need targeted
  group-by-group policy and implementation slices.

Fifteenth slice status:

- Completed: extended factory activation CPU fallback coverage to the
  shape-preserving elementwise group: `LeakyReLU`, `ELU`, `GELU`,
  `Swish`/`SiLU`, `Mish`, `Hardswish`, `SELU`, and shared-alpha `PReLU`.
- Completed: added CPU fallback implementations for `BCELoss` and
  `BCEWithLogitsLoss`, including reduction handling.
- Completed: added focused activation/loss regression coverage for the
  new fallback group and fixed expected BCE-with-logits stable-formula
  semantics in tests.
- Remaining: `Softmax`, `CrossEntropy`, `NLLLoss`, `KLDiv`, focal,
  metric-learning losses, per-channel `PReLU`, legacy layers, and linalg
  decompositions still need separate policy/shape-contract slices.

Sixteenth slice status:

- Completed: added CPU fallback for factory `SoftmaxActivation` forward
  and backward with row-major axis handling.
- Completed: fixed factory `SoftmaxActivation` ArrayFire 2D layout by
  using the semantic row-major Tensor/ArrayFire bridge instead of the
  generic native bridge.
- Completed: added focused 2D row-major Softmax forward/backward tests
  that validate independent row normalization and Jacobian-vector
  gradients.
- Remaining: `CrossEntropy`, `NLLLoss`, `KLDiv`, focal and
  metric-learning losses still need separate label/index shape-contract
  slices before CPU fallback is added.

Seventeenth slice status:

- Completed: added CPU fallback for `CrossEntropyLoss` and `NLLLoss`
  for the common 1D/2D class-axis contract: predictions as `[classes]`
  or `[batch, classes]`, class-index targets, and same-shape soft-label
  targets for CrossEntropy.
- Completed: fixed ArrayFire class-index gather for row-major 2D
  `CrossEntropyLoss` and `NLLLoss` by using column-major flat indices
  against semantic ArrayFire arrays.
- Completed: made `CrossEntropyLoss::Backward()` recompute softmax when
  called without a prior forward cache.
- Completed: removed stale debug logging from the CrossEntropy soft-label
  ArrayFire branch.
- Completed: added focused class-index CrossEntropy/NLL forward and
  backward tests.
- Remaining: `KLDiv`, focal and metric-learning losses still need
  separate fallback slices; legacy layer and linalg fallback work remains
  deferred.

Eighteenth slice status:

- Completed: added CPU fallback for `KLDivLoss` forward and backward,
  matching the existing default contract where predictions are log
  probabilities and non-positive probability targets contribute zero.
- Completed: added focused `KLDivLoss` forward and backward regression
  coverage.
- Remaining: focal and metric-learning losses still need separate
  fallback slices; legacy layer and linalg fallback work remains
  deferred.

Nineteenth slice status:

- Completed: added CPU fallback for `FocalLoss` forward and backward
  under the existing class-index classification contract.
- Completed: corrected the ArrayFire `FocalLoss` class-index gather and
  backward formula to match the standard softmax focal-loss derivative.
- Completed: added focused `FocalLoss` forward and backward regression
  coverage.
- Remaining: metric-learning losses still need separate fallback slices;
  legacy layer and linalg fallback work remains deferred.

Twentieth slice status:

- Completed: added CPU fallback for `CosineEmbeddingLoss` forward and
  backward under the existing `[batch, embedding_dim]` plus `SetLabels()`
  contract.
- Completed: added focused `CosineEmbeddingLoss` forward and backward
  regression coverage.
- Remaining at this checkpoint: `TripletLoss` and `ContrastiveLoss`
  still needed separate fallback slices; both are now handled in later
  slices.

Twenty-first slice status:

- Completed: added CPU fallback for `TripletLoss` forward and backward
  under the existing `[batch, embedding_dim]` plus `SetNegative()`
  contract. Backward preserves the current API surface by returning the
  anchor gradient.
- Completed: made the ArrayFire `TripletLoss::Backward()` path recompute
  anchor-positive and anchor-negative distances when called without a
  prior forward cache.
- Completed: added focused Euclidean `TripletLoss` forward and anchor
  gradient regression coverage.
- Remaining: `ContrastiveLoss` still needs a separate fallback slice;
  legacy layer and linalg fallback work remains deferred.

Twenty-second slice status:

- Completed: added CPU fallback for `ContrastiveLoss` forward and
  backward under the existing `[batch, embedding_dim]` plus
  `SetLabels()` contract where labels are `0` for similar pairs and `1`
  for dissimilar pairs. Backward preserves the current API surface by
  returning the `x1` gradient.
- Completed: made the ArrayFire `ContrastiveLoss::Backward()` path
  recompute distances when called without a prior forward cache.
- Completed: added focused `ContrastiveLoss` forward and `x1` gradient
  regression coverage.
- Remaining: CPU fallback Priority 12 now moves beyond core factory
  activations and standalone losses into legacy layer groups and linalg
  decomposition policy.

Twenty-third slice status:

- Completed: added CPU fallback for legacy `DenseLayer` forward and
  backward in `layer.cpp`, covering 1D single-sample and 2D batched
  Float32 inputs.
- Completed: kept the fallback aligned with the legacy ArrayFire
  contract: row-major `output = input @ weights^T + bias`, summed
  `grad_weights`/`grad_bias`, and `grad_input = grad_output @ weights`.
- Completed: added focused deterministic legacy `DenseLayer` forward,
  backward, and gradient-accumulator regression coverage.
- Remaining at this checkpoint: larger legacy layer families such as
  convolution, pooling, normalization, attention, upsample, pixel
  shuffle, and linalg decompositions still needed separate
  fallback/required-backend policy slices.

Twenty-fourth slice status:

- Completed: added CPU fallback for legacy `DropoutLayer` forward and
  backward in `layer.cpp`.
- Completed: preserved the existing contract: inference and `p == 0`
  pass through unchanged; training mode uses inverted dropout scaling
  and reuses the forward mask in backward.
- Completed: added focused legacy `DropoutLayer` eval pass-through and
  forward/backward mask-reuse regression coverage.
- Remaining at this checkpoint: pooling was the next small legacy layer
  family to consider; convolution, normalization, attention, upsample,
  pixel shuffle, and linalg decompositions remained larger follow-up
  slices.

Twenty-fifth slice status:

- Completed: added CPU fallbacks for legacy `MaxPool2DLayer` and
  `AvgPool2DLayer` forward/backward in `layer.cpp` under the existing
  `[H, W, C, N]` Float32 contract.
- Completed: added direct CPU implementation for legacy
  `GlobalAvgPool2DLayer` forward/backward. This intentionally avoids the
  legacy 4D ArrayFire bridge path, which still has row-major semantic
  issues for global spatial reductions.
- Completed: added focused deterministic pooling forward/backward
  regression coverage for max, average, and global average pooling.
- Remaining at this checkpoint: convolution, normalization, attention,
  upsample, pixel shuffle, and linalg decompositions remained larger
  follow-up slices.

Twenty-sixth slice status:

- Completed: added direct CPU implementation for legacy
  `Upsample2DLayer` nearest-neighbor forward/backward under the existing
  `[H, W, C, N]` Float32 contract.
- Completed: added direct CPU implementation for legacy
  `PixelShuffleLayer` forward/backward.
- Completed: added positive scale-factor validation for
  `Upsample2DLayer` and `PixelShuffleLayer`.
- Completed: added focused deterministic nearest upsample and
  pixel-shuffle forward/backward regression coverage.
- Remaining at this checkpoint: bilinear `Upsample2DLayer` CPU
  fallback was intentionally deferred as a separate
  interpolation-gradient slice. Convolution, normalization, attention,
  and linalg decompositions remained larger follow-up slices.

Twenty-seventh slice status:

- Completed: added direct CPU implementation for legacy
  `Upsample2DLayer` bilinear forward/backward using center-aligned
  interpolation under the existing `[H, W, C, N]` Float32 contract.
- Completed: added focused deterministic bilinear upsample
  forward/backward regression coverage.
- Remaining: convolution, normalization, attention, and linalg
  decompositions remain larger follow-up slices.

Twenty-eighth slice status:

- Completed: added direct CPU implementation for legacy `Conv2DLayer`
  forward/backward using the existing `[H, W, C, N]` Float32 contract,
  zero padding, stride, cross-correlation semantics, and gradients for
  input, weights, and bias.
- Completed: removed the old `Conv2DLayer` ArrayFire-first
  forward/backward branches after the regression test exposed a wrong
  backward shape under the GPU path.
- Completed: added focused deterministic `Conv2DLayer`
  forward/backward/parameter-gradient regression coverage.
- Follow-up found during this slice: `Tensor::Zeros()` can round-trip
  through ArrayFire and collapse trailing singleton dimensions. Conv2D
  now avoids it for rank-sensitive gradient buffers, but the Tensor
  factory should get a separate shape-preservation audit.
- Remaining: `Conv1DLayer`, `ConvTranspose2DLayer`, normalization,
  attention, and linalg decompositions remain follow-up slices.

Twenty-ninth slice status:

- Completed: added direct CPU implementation for legacy `Conv1DLayer`
  forward/backward using `[L, C, N]` Float32 inputs and the existing
  `[out_channels, in_channels, kernel]` weight storage contract.
- Completed: removed the old `Conv1DLayer` ArrayFire-first
  forward/backward branches because they used a conflicting implicit
  weight-axis interpretation and were not the canonical tested path.
- Completed: added focused deterministic `Conv1DLayer`
  forward/backward/parameter-gradient regression coverage over two
  batches.
- Remaining: `ConvTranspose2DLayer`, normalization, attention, and
  linalg decompositions remain follow-up slices.

Thirtieth slice status:

- Completed: added direct CPU implementation for legacy
  `ConvTranspose2DLayer` forward/backward using `[H, W, C, N]`
  Float32 inputs and the existing `[kernel, kernel, out_channels,
  in_channels]` weight storage contract.
- Completed: removed the old `ConvTranspose2DLayer` ArrayFire-first
  forward/backward branches so the tested row-major Tensor contract is
  canonical for the legacy convolution family.
- Completed: added focused deterministic `ConvTranspose2DLayer`
  forward/backward/parameter-gradient regression coverage.
- Remaining: normalization, attention, and linalg decompositions remain
  follow-up slices.

Thirty-first slice status:

- Completed: added direct CPU implementation for legacy
  `BatchNorm2DLayer` forward/backward using `[H, W, C, N]` Float32
  inputs, per-channel batch statistics, running statistics, and
  gamma/beta gradients.
- Completed: removed the old `BatchNorm2DLayer` ArrayFire-first
  forward/backward branches so training/eval behavior follows the
  tested row-major Tensor contract.
- Completed: added focused deterministic training-mode and eval-mode
  `BatchNorm2DLayer` regression coverage.
- Remaining: `LayerNormLayer`, `InstanceNorm2DLayer`, `GroupNormLayer`,
  attention, and linalg decompositions remain follow-up slices.

Thirty-second slice status:

- Completed: added direct CPU implementation for legacy
  `LayerNormLayer` forward/backward using a suffix normalized-shape
  contract over Float32 row-major tensors.
- Completed: removed the old `LayerNormLayer` ArrayFire-first
  forward/backward branches so affine gamma/beta gradients follow the
  tested CPU tensor contract.
- Completed: added focused deterministic `LayerNormLayer`
  forward/backward/parameter-gradient regression coverage.
- Remaining: `InstanceNorm2DLayer`, `GroupNormLayer`, attention, and
  linalg decompositions remain follow-up slices.

Thirty-third slice status:

- Completed: added direct CPU implementation for legacy
  `InstanceNorm2DLayer` forward/backward using the established
  `[H, W, C, N]` Float32 tensor layout and per-instance/per-channel
  spatial statistics.
- Completed: removed the old `InstanceNorm2DLayer` ArrayFire-first
  forward/backward branches so affine gamma/beta gradients follow the
  tested row-major Tensor contract.
- Completed: added focused deterministic `InstanceNorm2DLayer`
  forward/backward/parameter-gradient regression coverage that checks
  independent normalization across batch instances.
- Remaining: `GroupNormLayer`, attention, and linalg decompositions
  remain follow-up slices.

Thirty-fourth slice status:

- Completed: added direct CPU implementation for legacy
  `GroupNormLayer` forward/backward using the established `[H, W, C,
  N]` Float32 tensor layout and per-group/per-instance statistics.
- Completed: removed the old `GroupNormLayer` ArrayFire-first
  forward/backward branches so affine gamma/beta gradients follow the
  tested row-major Tensor contract.
- Completed: added focused deterministic `GroupNormLayer`
  forward/backward/parameter-gradient regression coverage that checks
  grouped statistics across channels and spatial positions.
- Completed: legacy normalization fallback group is now covered for
  `BatchNorm2DLayer`, `LayerNormLayer`, `InstanceNorm2DLayer`, and
  `GroupNormLayer`.
- Remaining: attention and linalg decompositions remain follow-up
  slices.

Thirty-fifth slice status:

- Completed: added deterministic CPU implementation for legacy
  `MultiHeadAttentionLayer` forward/backward over row-major `[batch,
  seq, embed]` Float32 tensors.
- Completed: removed the ArrayFire-only dependency from the core
  attention path for dropout-free execution and added strict projection,
  mask, cache, and parameter shape validation.
- Completed: corrected self-attention backward to combine query, key,
  and value gradients when Q/K/V are the same input tensor.
- Completed: added focused deterministic `MultiHeadAttentionLayer`
  regression coverage for forward weights, self-attention input
  gradients, and projection gradients.
- Remaining: CPU attention dropout is intentionally not implemented yet;
  cross-attention `Backward()` still returns only query gradients because
  the current `Layer::Backward` API cannot return key/value gradients.
  Linalg decompositions remain follow-up slices.

Thirty-sixth slice status:

- Completed: added deterministic CPU thin SVD for vector-matrix linalg
  using a local symmetric Jacobi eigensolver over `A^T A`.
- Completed: made thin SVD CPU-canonical instead of trying the
  ArrayFire SVD path first, because the GPU-first path can hang and
  prevents reliable CPU fallback behavior.
- Completed: unblocked SVD-dependent helpers: `Rank`,
  `ConditionNumber`, and `LowRankApproximation`.
- Completed: added focused linalg regression coverage for rectangular
  SVD reconstruction, rank, condition number, and rank-1 approximation.
- Remaining: `full_matrices=true` SVD is intentionally unsupported in
  the CPU fallback. General nonsymmetric full eigendecomposition remains
  a follow-up slice.

Thirty-seventh slice status:

- Completed: added CPU attention-dropout support for legacy
  `MultiHeadAttentionLayer` training mode.
- Completed: cached and reused the dropout mask across attention
  forward/backward so gradient flow matches the sampled attention path.
- Completed: added focused regression coverage for one-token attention
  dropout that validates mask-scaled forward output, cached softmax
  weights, and backward mask reuse without depending on a fixed RNG
  seed.
- Remaining: cross-attention `Backward()` still returns only query
  gradients because the current `Layer::Backward` API cannot return
  key/value gradients. `full_matrices=true` SVD and general
  nonsymmetric eigendecomposition also remain follow-up slices.

Thirty-eighth slice status:

- Completed: extended CPU SVD fallback to support `full_matrices=true`
  by completing the left-singular-vector basis with a local
  Gram-Schmidt pass.
- Completed: preserved the thin SVD reconstruction contract while
  returning full `U` and `Vt` basis shapes for full SVD callers.
- Completed: added focused linalg regression coverage for full
  rectangular SVD shape, reconstruction, and orthonormal `U` columns.
- Remaining: cross-attention `Backward()` still returns only query
  gradients because the current `Layer::Backward` API cannot return
  key/value gradients. General nonsymmetric eigendecomposition remains
  a follow-up slice.

---

## Priority 0: Core Architectural Problems

### 1. Overlapping abstractions for the same job

**Severity:** High

**Status 2026-06-05:** Partially fixed. The current code already uses
`SequentialModel` + `Module` as the de facto model-facing training path:
tests, Python bindings, distributed wrappers, interpretability, and
examples all target `SequentialModel`. Header and README documentation
now make that role explicit. Remaining work is to migrate or deprecate
duplicate primitive paths only when compatibility handling is planned.

The backend currently has multiple competing abstractions:

- `Layer` / `DenseLayer` / `Conv2DLayer` / etc. in
  `include/cyxwiz/layer.h`
- `Module` / `LinearModule` / `SequentialModel` in
  `include/cyxwiz/sequential.h`
- activation hierarchy in `include/cyxwiz/activation.h`
- separate activation implementations in `src/algorithms/activation.cpp`
  and `src/algorithms/activations/*.cpp`

This causes:

- duplicate implementation paths
- unclear ownership of training behavior
- weaker API ergonomics
- more places for shape/layout bugs to hide

**Observed examples:**

- `DenseLayer` exists in `layer.*`
- `LinearLayer` exists in `layers/linear.*`
- `LinearModule` wraps `LinearLayer`
- `Model` exists as a base class but actual training logic lives in
  `SequentialModel`

**Recommendation:**

Pick one canonical stack and deprecate the others.

Practical path:

1. Decide whether `Layer` or `Module` is the canonical training unit.
2. Map every existing model-facing API to that unit.
3. Freeze new feature work on the non-canonical path.
4. Migrate call sites incrementally.
5. Remove duplicate classes only after compatibility is handled.

---

### 2. Two `DataLoader` concepts share the same name

**Severity:** High

**Status 2026-06-05:** Fixed for the backend C++ surface. The compiled
DuckDB/file SQL loader remains `cyxwiz::DataLoader`; the dataset batch
iterator is now `cyxwiz::TrainingDataLoader` and is built into the
backend target with coverage that includes both headers together.

There are two distinct `DataLoader` concepts in the same namespace:

- `include/cyxwiz/data_loader.h`
  - DuckDB-backed file/query loader
- `include/cyxwiz/dataloader.h`
  - training batch iterator over datasets

This is a naming collision at the conceptual level.

It causes:

- confusion for maintainers
- API ambiguity
- documentation complexity
- harder binding generation and language interop

**Recommendation:**

Rename one or both classes.

Suggested split:

- `TabularDataLoader` or `DuckDBDataLoader`
- `BatchDataLoader` or `TrainingDataLoader`

Also update docs and Python bindings to use the new terminology.

---

### 3. Tensor is CPU-primary, not truly device-aware

**Severity:** High

**Status 2026-06-05:** Partially fixed. Tensor now carries a cached
ArrayFire array, host/device current flags, and layout metadata for native,
row-major 2D, and row-major 3D views. `SetFromArray()` and the ArrayFire
constructor keep device data resident and materialize host memory lazily.
`GetArray()` reuses a current compatible device array rather than always
rebuilding it. Remaining work is performance cleanup: non-const `Data()`
is still a coarse host mutation boundary, and some higher-level operators
and layers still force CPU materialization through fallback or inspection
paths.

Original audit behavior, now stale in part:

- Tensor was CPU-primary, with host data treated as the authoritative
  buffer.
- `GetArray()` rebuilt ArrayFire arrays instead of reusing current device
  state.
- `SetFromArray()` copied device results back to CPU immediately.
- factory helpers created GPU arrays and immediately materialized them on
  CPU.

The current implementation has moved to lazy host/device residency, but
the higher-level code still needs cleanup so chained training operations
avoid accidental CPU materialization.

**Consequences:**

- excessive host->device copies
- excessive device->host copies
- poor performance for chained training operations
- hard ceiling on GPU efficiency

**Recommendation:**

Redesign Tensor around explicit residency:

1. host buffer
2. device buffer
3. host dirty flag
4. device dirty flag
5. explicit sync boundaries

Target behavior:

- pure device pipelines stay on device
- host materialization happens only for serialization, UI inspection,
  Python conversion, or explicit CPU access

---

## Priority 1: Correctness and API Honesty

### 4. MemoryManager is not truthful

**Severity:** High

**Status 2026-06-05:** Fixed for MemoryManager-routed host
allocations. `Allocate()` records pointer sizes, `Deallocate()` subtracts
the recorded live bytes, and `ResetPeak()` resets peak tracking to the
current live count. Tensor host buffers use this path and have regression
coverage. The counters are not a whole-process memory profiler: ArrayFire,
DuckDB, STL, and direct third-party allocations remain outside this
accounting scope.

Original audit behavior:

- `Allocate()` increments the byte counter
- `Deallocate()` frees memory but does not decrement the counter
- hot allocation paths often bypass `MemoryManager` entirely

So:

- `GetAllocatedBytes()` is not current live memory
- leak investigations using these counters are unreliable

**Recommendation:**

Either:

- implement real size tracking per allocation

or:

- remove or clearly relabel these counters until they are accurate

Also route Tensor core allocations through a tracked allocator if
tracking remains part of the public surface.

---

### 5. Declared API methods are missing or incomplete

**Severity:** High

**Status 2026-06-05:** Partially fixed. `cyxwiz_tensor_matmul()` now
routes through `LinearAlgebra::Multiply(const Tensor&, const Tensor&)`
and has focused C API tests for success and invalid-shape error
reporting. C tensor factories now validate null shape/data pointers and
invalid dtype enum values before constructing backend tensors. Remaining
work in this item is the broader public API audit: unsupported APIs must
be implemented, removed, or explicitly marked deferred.

Examples:

- `Tensor::ToDevice(Device*)` / `Tensor::ToCPU()` were removed from the public Tensor API
- C API `cyxwiz_tensor_matmul()` returned "not yet implemented" before
  the 2026-06-05 first-slice fix
- `engine.h` is effectively empty

This creates a mismatch between:

- header surface
- backend implementation
- docs
- user expectations

**Recommendation:**

For every public API:

1. implement it
2. remove it
3. or mark it clearly unsupported/deferred

The backend surface must be honest.

---

### 6. README and public docs overstate backend capabilities

**Severity:** High

**Status 2026-06-05:** Partially fixed. The backend README and umbrella
header comments now describe acceleration as optional ArrayFire-backed
coverage rather than a uniformly transparent GPU path. Remaining work is
to continue auditing detailed per-algorithm docs and examples as each
operation group is hardened. The clustering public header has also been
corrected to describe its ArrayFire build requirement and non-ArrayFire
unsupported result behavior.

The docs present the backend as a coherent high-performance ML compute
library with GPU acceleration and device transfer semantics.

But the implementation currently has:

- missing `ToDevice` / `ToCPU`
- incomplete CPU fallback coverage
- duplicated model abstractions
- unimplemented C API operations

**Recommendation:**

Audit docs against code.

If the code is not ready to support a claim, the docs should not claim
it yet.

---

## Priority 2: Memory and Leak Risks

### 7. Tensor GPU operator exception paths can leak heap objects

**Severity:** High

**Status 2026-06-05:** Fixed/stale. The current Tensor arithmetic
implementation no longer allocates `af::array*` with raw `new` in
`operator+`, `operator-`, `operator*`, or `operator/`. These paths use
stack `af::array` temporaries and return device-resident `Tensor`
instances. Tensor cached ArrayFire state is owned by
`std::unique_ptr<af::array>`, so exception cleanup is handled by RAII.
Broader ArrayFire interop should still prefer stack objects or
`std::unique_ptr`, but the specific Tensor arithmetic leak risk described
below is no longer present.

In Tensor arithmetic operators:

- `operator+`
- `operator-`
- `operator*`
- `operator/`

the code allocates `af::array*` with `new` and deletes only on the
success path.

If an exception is thrown before cleanup, the pointers leak.

**Recommendation:**

Immediate fix:

- replace raw `af::array*` with stack `af::array`
- or use `std::unique_ptr`

Broader fix:

- remove manual cleanup patterns in Tensor core
- standardize RAII across ArrayFire interop

---

### 8. `af_array_` cached state is mostly vestigial

**Severity:** Medium

**Status 2026-06-05:** Fixed/stale. `af_array_` is now part of a real
host/device residency contract. `Tensor` tracks `host_current_`,
`device_current_`, and `TensorDeviceLayout`; `GetArray()` reuses a
current native ArrayFire cache, row-major 2D/3D accessors reuse their
compatible cached layout, and `SetFromArray()` keeps device data resident
until host data is explicitly requested. Host mutation marks the cached
device state stale. The remaining concern is no longer whether
`af_array_` is vestigial, but whether higher-level training paths avoid
unnecessary host materialization.

`Tensor` owns an `af_array_` pointer, but `GetArray()` ignores it and
creates a new array from CPU memory every time.

This means the code is carrying both:

- host ownership
- partial device ownership

without a clean contract.

**Recommendation:**

Choose one:

- real cached device-backed tensor state
- or remove cached state until the device model is redesigned

---

## Priority 3: Performance Problems

### 9. Optimizer GPU paths still copy state back every step

**Severity:** High

**Status 2026-06-05:** Partially fixed/stale. Optimizer GPU paths now
use `Tensor::SetFromArray()` for updated parameters and optimizer state
buffers, so successful ArrayFire steps keep those tensors device-resident
until explicit host access. Regression coverage checks this behavior for
SGD, Adam, AdamW, RMSprop, AdaGrad, NAdam, Adadelta, and LAMB. Remaining
work is performance-oriented: profile `GetArray()` cache misses,
operation-specific ArrayFire failures, scalar host reads, and CPU fallback
paths during real training loops.

SGD, Adam, AdamW, RMSprop, AdaGrad, and others create GPU arrays from
host parameter buffers, perform updates, then copy parameters and
optimizer state back to CPU every step.

This is expensive and undermines GPU gains.

**Recommendation:**

Keep:

- parameters
- optimizer moments / buffers
- intermediate update tensors

resident on device across steps.

Only sync at explicit boundaries.

---

### 10. Layer implementations repeatedly materialize host data

**Severity:** High

**Status 2026-06-05:** Partially fixed. The standalone activation
helpers, losses, sequential modules, and newer `layers/linear.cpp` paths
mostly return device-backed tensors through the current Tensor residency
contract. Focused tests now verify `LinearLayer` GPU forward/backward
outputs and gradients remain device-resident until explicit host access.
Remaining work is targeted cleanup in legacy `layer.cpp`, where some
complex layers still call `af::array::host(...)` into CPU tensors during
forward/backward.

Many forward/backward implementations:

- compute on ArrayFire
- immediately copy outputs back to CPU tensors
- later re-upload those tensors for the next operation

This is visible in:

- `layers/linear.cpp`
- `sequential.cpp`
- parts of `layer.cpp`
- parts of `loss.cpp`
- parts of `activation.cpp`

**Recommendation:**

Refactor training hot paths to keep:

- outputs
- gradients
- cached activations
- running stats

on device throughout the step.

---

### 11. Some "GPU helpers" do GPU work only to return CPU-owned tensors

**Severity:** Medium

**Status 2026-06-05:** Fixed/stale for Tensor factories. `Tensor::Zeros`,
`Tensor::Ones`, and `Tensor::Random` return device-backed tensors through
the ArrayFire constructor on successful ArrayFire paths, and host buffers
are allocated lazily only when CPU data is requested. Regression coverage
now checks this behavior for all three factories.

Examples:

- `Tensor::Zeros`
- `Tensor::Ones`
- `Tensor::Random`

These allocate device arrays and immediately copy them back to host.

**Recommendation:**

After Tensor redesign:

- device-backed constructors should stay on device
- host-backed constructors should build directly on CPU

Do not pay GPU setup cost if the result is immediately host-owned.

---

### 12. CPU fallback coverage is inconsistent

**Severity:** High

**Status 2026-06-05:** Partially fixed. The backend policy is now:
core, shape-preserving float32 training primitives should have CPU
fallbacks; advanced GPU-heavy operators may remain ArrayFire-required,
but they must say so honestly and be tracked by group. Factory
`ReLUActivation`, `SigmoidActivation`, `TanhActivation`, `Softmax`,
`LeakyReLU`, `ELU`, `GELU`, `Swish`/`SiLU`, `Mish`, `Hardswish`,
`SELU`, shared-alpha `PReLU`, `MSELoss`, `L1Loss`,
`SmoothL1Loss`/`HuberLoss`, `CrossEntropyLoss`, `NLLLoss`, `BCELoss`,
`BCEWithLogitsLoss`, `KLDivLoss`, `FocalLoss`,
`CosineEmbeddingLoss`, `TripletLoss`, `ContrastiveLoss`, legacy
`DenseLayer`, legacy `DropoutLayer`, legacy pooling layers
(`MaxPool2DLayer`, `AvgPool2DLayer`, `GlobalAvgPool2DLayer`),
`Upsample2DLayer` nearest and bilinear modes, `PixelShuffleLayer`, and
legacy `Conv2DLayer`, legacy `Conv1DLayer`, and legacy
`ConvTranspose2DLayer`, legacy `BatchNorm2DLayer`, and legacy
`LayerNormLayer`, legacy `InstanceNorm2DLayer`, and legacy
`GroupNormLayer`, and legacy deterministic `MultiHeadAttentionLayer`
now have CPU fallback coverage with tests. Vector-matrix thin SVD,
rank, condition number, and low-rank approximation also have CPU
fallback coverage with tests. Remaining fallback work should proceed by
operator group, starting with cross-attention gradient API limitations,
and general eigendecomposition.

The build can run without ArrayFire, but many public operations cannot:

- numerous activation functions
- many losses
- many layers
- partial linalg coverage

In several cases the code throws `"requires ArrayFire"` instead of
using a real CPU implementation.

**Recommendation:**

Choose one explicit policy:

Policy A:
- backend supports CPU for all public APIs

Policy B:
- backend is GPU-first and some APIs are GPU-required

But do not mix both stories implicitly.

---

## Priority 4: Build, Hygiene, and Maintainability

### 13. Large monolithic source files should be split

**Severity:** Medium

Particularly large files:

- `src/algorithms/layer.cpp`
- `src/algorithms/sequential.cpp`
- `src/algorithms/linear_algebra.cpp`
- `src/algorithms/time_series.cpp`
- `src/algorithms/text_processing.cpp`

**Why this matters:**

- harder code review
- harder targeted testing
- more hidden coupling
- higher merge-conflict risk

**Recommendation:**

Split by domain responsibility.

Example for `layer.cpp`:

- dense / conv / pooling
- normalization
- recurrent
- attention / transformer
- utility reshape / upsample / pixel shuffle

---

### 14. Repository hygiene issue: stray `device_1.cpp`

**Severity:** Medium

**Status 2026-06-05:** Fixed. The stray `src/core/device_1.cpp` build
artifact was removed after confirming it was not part of the CMake
backend source list.

`src/core/device_1.cpp` appears to contain build-command garbage rather
than valid source.

Even if it is not compiled, it should not remain in the source tree.

**Recommendation:**

- remove it after confirming it is not required
- add a cleanup pass for accidental build artifacts in tracked files

---

### 15. Empty or placeholder surfaces should be cleaned up

**Severity:** Low to Medium

Examples:

- `engine.h` has no meaningful surface
- `model.cpp` is effectively placeholder

**Recommendation:**

Either:

- remove these until needed

or:

- define the intended role clearly and implement accordingly

---

## Priority 5: Determinism, Thread Safety, and Global State

### 16. Thread-safety story is weaker than documentation suggests

**Severity:** Medium

Global state exists in several places:

- current active device singleton
- backend initialization flag
- default distributed process group
- global stopword and lexicon caches

This is not inherently wrong, but it means the backend is not simply
"thread-safe for independent tensors" in any broad sense.

**Recommendation:**

Document which APIs are:

- process-global
- thread-local
- instance-local

and which are not safe for concurrent mutation.

---

### 17. Reproducibility is inconsistent

**Severity:** Medium

**Status 2026-06-05:** Partially fixed. Core `Tensor::Random()` CPU
fallback no longer uses raw `rand()` or shared C RNG state. A full
reproducibility API is still deferred: seeded Tensor random factories and
cross-subsystem RNG policy should be designed together.

Different subsystems use different random sources:

- raw `rand()` in Tensor random generation before the 2026-06-05
  second-slice fix
- `mt19937` in some algorithms
- `random_device` in others

This makes seeded reproducibility inconsistent across the backend.

**Recommendation:**

Adopt one RNG strategy:

- explicit seedable engine
- consistent API for random state
- no raw `rand()` in core tensor paths

---

## Priority 6: Distributed Subsystem Maturity

### 18. Distributed training code is promising but still experimental

**Severity:** Medium

Strengths:

- structured process group abstraction
- CPU and NCCL backends
- default process group plumbing

Weaknesses:

- NCCL path still contains TODOs in critical areas
- global singleton process-group model increases coupling
- robustness needs more stress validation

**Recommendation:**

Treat distributed training as a focused subsystem with its own
hardening plan:

1. dtype/device validation
2. proper average-reduction handling
3. lifecycle and error-path testing
4. multi-rank stress tests

---

## Suggested Engineering Roadmap

### Phase A - Foundation Cleanup

1. Remove/repair misleading public APIs
2. Fix MemoryManager honesty
3. Remove raw ArrayFire heap leaks
4. Remove stray files and placeholders
5. Align docs with reality

### Phase B - Canonical Architecture Decision

1. Choose one NN abstraction stack
2. Rename conflicting `DataLoader` classes
3. Define the intended long-term `Tensor` contract

### Phase C - Tensor Residency Refactor

1. Add device-backed state
2. Add host/device dirty flags
3. Stop rebuilding AF arrays from host each call
4. Convert hot paths to persistent device execution

### Phase D - Training Performance Pass

1. optimizer state stays on device
2. layer caches stay on device
3. minimize `host(...)` calls in forward/backward
4. validate throughput improvements with profiling

### Phase E - Fallback and API Completion

1. either complete CPU fallback or mark GPU-required APIs
2. implement missing C API pieces
3. remove dead/duplicate surfaces

---

## Best First Tasks for Engineers

If picking this up incrementally, start here:

### Task 1
Replace exception-prone `new af::array` usage in `Tensor` ops with RAII.

### Task 2
Make `MemoryManager` truthful or remove exposed counters temporarily.

### Task 3
Write an RFC deciding:

- `Layer` vs `Module`
- `DataLoader` naming split
- target Tensor residency model

### Task 4
Add a profiling harness for:

- Tensor ops
- optimizer step
- linear layer forward/backward
- simple sequential training loop

### Task 5
Audit README and public headers against actual backend behavior.

---

## Final Note

The backend is already feature-rich, but it needs consolidation more
than it needs more surface area.

The highest-value engineering work now is:

- reduce abstraction overlap
- make the API honest
- make Tensor/device behavior explicit
- remove unnecessary host-device churn

That work will improve:

- maintainability
- correctness
- debuggability
- training speed
