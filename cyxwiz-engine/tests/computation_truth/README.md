# CyxWiz Engine Computation Truth Tests

This folder is for deterministic numerical parity tests.

Purpose:

- Compare CyxWiz engine computations against stable references.
- Catch silent math drift in preprocessing, model forward passes, losses, gradients, optimizers, and training lifecycle.
- Keep correctness tests independent of the GUI.

Reference stack:

- PyTorch for tensor/model/loss/optimizer parity.
- scikit-learn for classification/regression metrics and TF-IDF parity.
- Small in-repo reference implementations when external Python packages are unavailable.

PyTorch is a reference-generation dependency, not an Engine runtime
dependency. Small generated fixtures are checked in under `fixtures/`, copied
beside their test executable at build time, and consumed by the C++ tests.

`coverage_inventory.json` is the machine-readable Plan 39 Tier-0/Tier-1
baseline. It records ownership, semantic contracts, oracle choice, current
tests, execution path, honest coverage status, and remaining evidence gaps.

Optional accelerator coverage is evaluated over locally qualified routes.
Discovery alone does not make a route eligible: platform compatibility,
installed runtime/provider support, exact activation, and execution validation
must authorize it. An unavailable or incompatible CUDA/OpenCL/oneAPI route is
an explicit skip, not a global computation-coverage blocker. A qualified route
must pass with exact requested/effective identity and may never pass by
silently selecting another backend.

Initial target cases:

- TF-IDF bounded materialization values and shape.
- Dense forward parity.
- CrossEntropy loss parity.
- Regression-loss forward/gradient parity across reductions and thresholds.
- Multi-step Adam, AdamW, and PyTorch-scheduled NAdam parity.
- Learning-rate scheduler sequence and boundary parity.
- Classification decisions, reports, ROC/AUC, PR/AP, and threshold selection.
- Training lifecycle: configured epochs vs completed/stopped/cancelled reason.

The broad tracking ticket is:

`docs/Data Studio/tofix39.md`

## Current tests

`test_computation_truth_tfidf_loss`

- Builds an Arrow text table in memory.
- Runs `TFIDFVectorizerOperator`.
- Verifies bounded output width: `max_features` columns plus optional `y`.
- Verifies deterministic TF-IDF values against a hand reference.
- Verifies deterministic label encoding.
- Verifies MSE, L1, SmoothL1, and Huber forward values and prediction gradients
  against generated PyTorch fixtures across `none`/`mean`/`sum`. The matrix
  distinguishes SmoothL1 `beta` from Huber `delta`, covers exact threshold
  edges, and proves SmoothL1 `beta=0` is L1 rather than a division-by-zero path.
- Verifies `CrossEntropyLoss` against generated PyTorch `cross_entropy`
  fixtures for class-last rank-1/2/3 logits, index and soft targets,
  `none`/`sum`/`mean`, class weights, ignored targets, label smoothing,
  extreme logits, all-ignored mean behavior, forward loss, and full logit
  gradients.
- Verifies `NLLLoss` against generated PyTorch `nll_loss` fixtures for
  class-last rank-1/2/3 log probabilities, Int64 class indices,
  `none`/`sum`/`mean`, ignored and all-ignored targets, forward loss, and full
  prediction gradients.
- Verifies `FocalLoss` against an explicit PyTorch-autograd focal equation for
  class-last rank-1/2/3 logits, `none`/`sum`/`mean`, gamma zero, extreme logits,
  forward loss, and full prediction gradients. Alpha and gamma reject
  non-finite or negative values, and backward recomputes from its supplied
  logits rather than a stale same-shaped forward cache. The stable
  extreme-logit equation and ordinary class-index mean denominator are also
  exercised directly in a no-ArrayFire native CPU build.
- Verifies BCE, BCEWithLogits, and KLDiv against generated PyTorch fixtures
  across `none`/`sum`/`mean`, scalar through rank-4 tensors, fractional and exact-boundary
  probability targets, weighted and extreme logits, zero KL targets, and
  negative log targets. BCE follows PyTorch's `-100` log cap and default
  `1e-12` derivative floor while retaining its explicit configurable floor;
  BCEWithLogits validates positive finite `pos_weight`; KLDiv preserves
  PyTorch `log_target` and zero-target semantics.
- Verifies SoftDice, Tversky, and Jaccard against explicit PyTorch tensor
  equations and autograd across `none`/`sum`/`mean`, scalar through rank-4
  tensors, configurable smoothing, Tversky alpha/beta, all-zero degenerate
  masks with positive smoothing, forward values, and full prediction
  gradients. Scalar and rank-1 inputs are one sample; rank-2 through rank-4
  use dimension 0 as the batch and reduce all remaining dimensions per sample.
  A separate two-batch SoftDice-to-Linear-to-SGD sequence proves the loss
  gradient reaches parameter updates. Forced overlap fallback proves strict
  rejection and compatible, attributed native execution.
- Verifies CosineEmbedding, Contrastive, and Euclidean/smoothed-cosine Triplet
  forward values and every embedding-branch gradient against generated
  PyTorch 2.10 fixtures across `none`/`mean`/`sum`, including zero vectors,
  coincident embeddings, inactive margins, and exact label conventions. A
  two-batch shared-Linear Contrastive-to-SGD sequence proves device-resident
  gradient accumulation and parameter updates. Strict ArrayFire CPU and every
  locally qualified CUDA/OpenCL route run with exact identity and zero native
  fallback; forced failures separately prove strict rejection and compatible,
  attributed native CPU execution. Incompatible oneAPI remains an explicit
  device-selection skip rather than a global blocker.
- Verifies `LinearLayer` forward output against a generated PyTorch linear fixture.
- Verifies `LinearLayer` backward values:
  - gradient with respect to input
  - summed weight gradients, matching PyTorch autograd
  - summed bias gradients, matching PyTorch autograd
- Verifies Adam, AdamW, and NAdam parameters, moments, per-parameter update
  counts, missing-gradient behavior, `zero_grad` preservation, and exact resume
  across three steps against generated PyTorch fixtures. AdamW includes
  decoupled decay; NAdam includes PyTorch's momentum schedule and product state.
- Verifies `SGDOptimizer` parameter updates and persistent momentum-buffer
  state across three steps against a generated PyTorch SGD fixture.
- Verifies RMSprop, AdaGrad, and Adadelta parameter and accumulator state across
  three steps, `zero_grad` preservation, and exact resume from exported state
  against generated PyTorch fixtures.
- Verifies LAMB parameters, moments, bias correction, weight decay, trust-ratio
  zero-norm edges, `zero_grad` preservation, and exact resume across three steps
  against an independent PyTorch tensor-equation fixture.
- Activates ArrayFire CPU exactly, binds an immutable execution context, and
  runs the fixture-backed core checks with native CPU fallback forbidden.
- Repeats the complete Adam/AdamW/NAdam, SGD, RMSprop/AdaGrad/Adadelta, and
  LAMB multi-step/state/resume fixture matrix on every selectable installed
  CUDA/OpenCL route. Each route is activated exactly, forbids native fallback,
  and permits only bounded attributed verification readbacks. Locally
  incompatible oneAPI routes are recorded by device qualification and are not
  selected by this in-process matrix.
- Proves Linear forward/backward for biased and unbiased rank-1/rank-2 inputs,
  PyTorch sum-over-batch parameter gradients, and two sequential variable-size
  batch updates through SGD. The complete Linear matrix runs on strict
  ArrayFire CPU and every selectable installed CUDA/OpenCL route with exact
  identity, no native fallback, and only bounded attributed verification
  readback. Locally incompatible oneAPI routes remain quarantined by device
  qualification and do not block this computation contract.
- Proves the transformer-owned `DenseLayer` affine primitive against the same
  generated PyTorch Linear oracle for biased and unbiased rank-1/rank-2
  forward/input-gradient/parameter-gradient cases and two variable-size SGD
  updates. Invalid construction, dtype, shape, parameter, and
  backward-before-forward state fail before compute; supplied parameters are
  validated atomically. The matrix runs under the same strict ArrayFire CPU
  and qualified CUDA/OpenCL residency contract. GUI Dense nodes are a separate
  ownership boundary: ModelBuilder emits `LinearModule`, and activation is an
  explicit following graph module rather than a hidden DenseLayer property.
- Rejects native fallback attempts and undeclared host synchronization; only
  bounded, attributed test-output readbacks are allowed.
- Proves regression, probability, overlap, NLL, Focal, and every supported CrossEntropy rank use strict
  ArrayFire paths with no hidden native CPU computation on CPU and every
  locally qualified CUDA/OpenCL route. Forced regression fallback is
  separately tested for exact rejection/compatibility policy, reason code,
  numerical parity, and `loss_cpu_path` host-sync attribution. GUI/compiled
  Huber configuration constructs the distinct PyTorch-style `HuberLoss`; it
  no longer aliases the differently scaled `SmoothL1Loss` equation.
- Proves every supported CrossEntropy rank uses the same strict device path;
  rank-3 is reshaped on device and is not a hidden native CPU variant.
  Incompatible optional providers remain explicit device-selection skips.

The elementwise-activation, Linear, regression-loss, probability-loss,
overlap-loss, CrossEntropy, NLL, Focal,
Adam-family, SGD, adaptive-optimizer, LAMB, and weighted-sampler expectations come from the checked-in
`fixtures/training_core_pytorch.json` fixture. Regenerate it with:

```powershell
python cyxwiz-engine/tests/computation_truth/reference/generate_training_core_fixtures.py
```

The pinned reference package is listed in
`reference/requirements.txt`. Regenerating fixtures is an explicit developer
action; normal configuration, builds, and test runs do not invoke Python or
require network access.

The elementwise activation matrix covers ReLU, LeakyReLU, ELU, tanh-approximate
GELU, Swish/SiLU, Sigmoid, Tanh, Mish, Hardswish, and SELU forward and input
gradient parity. Rank-2 inputs include zero, branch boundaries, values adjacent
to Hardswish's `-3`/`3` boundaries, and finite extremes through `+/-100`.
Strict execution on ArrayFire CPU and every selectable installed CUDA/OpenCL
route permits no native fallback or compute-time host synchronization. Mish
uses a stable softplus/sigmoid formulation and SELU bounds the inactive
exponential branch so finite extremes remain finite on accelerators. The
backend `[activation][pytorch]` test consumes the same generated matrix in
Debug, Release, and the configured no-ArrayFire build, qualifying the exact
native CPU implementations used by compatibility fallback. Locally
incompatible optional accelerators remain explicit route-selection skips.

The Softmax matrix uses PyTorch forward and autograd Jacobian-vector products
for rank-1 through rank-4 inputs; axis `0`, axis `1`, axis `2`, and negative
last-axis selection; zero-element axes; and stable finite extremes through
`+/-100`. It verifies both the
standalone `SoftmaxActivation` and the production `SoftmaxModule` used by graph
models. Standalone backward derives its result from the supplied input and does
not trust a same-shaped stale forward cache. The module normalizes its configured
positive or negative axis instead of hard-coding the rank-2 class axis, and
rejects invalid dtype, axis, shape, and backward-before-forward state. The same
strict qualified-route residency rules apply. The backend
`[activation][pytorch][softmax]` test consumes the same generated matrix in
Debug, Release, and the configured no-ArrayFire build, qualifying both native
CPU compatibility implementations without adding a second oracle.

The PReLU matrix covers shared alpha plus per-channel rank-2 and rank-4 forms
against PyTorch forward, input-gradient, and alpha-gradient truth. Channel
parameters map to semantic dimension 1, parameter count/shape/dtype are
validated, and the exact ArrayFire CPU/CUDA/OpenCL matrix forbids fallback and
undeclared host synchronization. The backend
`[activation][pytorch][prelu]` matrix runs those same fixtures in the configured
no-ArrayFire build, including native per-channel forward and both gradients.
`PReLUModule` exposes `alpha` and its gradient through `SequentialModel`; the
fixture-driven rank-4 case proves discovery, SGD update, and write-back against
the PyTorch alpha gradient. Locally incompatible optional accelerators remain
device-selection skips.

`cyxwiz-tests "[regression_metrics]"` consumes the independent scikit-learn
1.8 fixture generated by
`reference/generate_regression_metrics_fixtures.py`. It proves the public
`ModelEvaluation::ComputeRegressionMetrics` API for MSE, RMSE, MAE, R-squared,
relative MAPE, and max error over ordinary, perfect/imperfect constant, zero,
signed, and single-sample targets. Constant-target R-squared follows sklearn's
finite policy (`1.0` for perfect and `0.0` for imperfect predictions), a
single sample returns undefined R-squared, and zero-target MAPE uses Float64
epsilon rather than silently dropping observations. Empty, mismatched, and
non-finite inputs fail explicitly.

`cyxwiz-tests "[classification_metrics]"` consumes the checked-in
scikit-learn 1.8/PyTorch 2.10 fixture generated by
`reference/generate_classification_metrics_fixtures.py`. It proves sorted-union
confusion matrices and classification reports, `score >= threshold` binary
decisions, single-class balanced accuracy, MCC, tied-score ROC/AUC and PR/AP,
multiclass one-vs-rest curves and thresholds, ascending-candidate threshold
selection, and monotonic-axis AUC behavior. The matrix includes a 1,005-sample
case that prevents input size from changing execution or semantics. Empty,
ragged, out-of-range, non-finite, single-class curve, and invalid-criterion
inputs fail explicitly.

These public vector APIs are bounded native C++ reporting operations: they do
not upload already-host-owned observations to the currently selected device
and do not hide a size-dependent fallback. Training-time Tensor classification
decisions remain ArrayFire-resident and perform only bounded reporting readback.
The source-policy test enforces both this execution boundary and the canonical
backend-owned public header used by the GUI.

Regenerate the classification fixture explicitly with:

```powershell
python cyxwiz-engine/tests/computation_truth/reference/generate_classification_metrics_fixtures.py
```

`cyxwiz-tests "[tensor_ownership]"` covers Tensor construction, metadata,
copy, move, clone, and independent mutation for Float32, Float64, Int32, Int64,
and UInt8. The ArrayFire CPU matrix starts from device-only semantic tensors
and proves copy/move/clone allocate no host-owned Tensor bytes and perform zero
host synchronization. Explicit `MutableData` then materializes only the
mutated copy and preserves the source value. Checked metadata cases cover
dimension-product overflow, dtype-sized byte overflow, invalid dtypes, and
PyTorch-compatible zero-element shapes: a later zero dimension short-circuits
otherwise overflowing preceding dimensions to `NumElements() == 0` and
`NumBytes() == 0`.

`cyxwiz-tests "[tensor_host_access]"` proves the explicit host boundary for
all five Tensor dtypes on ArrayFire CPU. A first `ReadData` records one exact
payload readback and preserves the current semantic device array; repeated
reads do not synchronize again. `MutableData` records the required readback,
invalidates the old device value, and the next semantic device access rebuilds
the changed value without another device-to-host transfer. The source scan now
rejects typed and untyped compatibility `Data` calls in its guarded training
manifest, including the sequence training step. Dataset ingress, selected CPU
collectives, host bucket transport, checkpoint/model serialization, model
import/export, language output selection, metric output/label validation, and
sequence input canonicalization also use explicit `ReadData` or `MutableData`
ownership. The remaining exact inventory is 29 files and 435 calls: 28
compatibility-compute owners (429 calls) plus the six-call NCCL transport owner.
NCCL remains separate because it needs an actual device-pointer contract after
`EnsureOnDevice`; changing it to a host accessor would preserve the wrong
transport semantics. Those residuals remain named rather than claimed complete.

`cyxwiz-tests "[tensor_layout]"` proves semantic ArrayFire layout round trips
for ranks 1 through 4 and all five Tensor dtypes. The matrix checks logical
shape and value preservation while a host-sync observer requires zero
device-to-host transfers. Rank-3 and rank-4 native-to-semantic and
semantic-to-native conversion use device-side reshape/reorder operations,
matching the existing rank-2 device-resident contract.

`cyxwiz-tests "[tensor_shape]"` consumes generated PyTorch 2.10 fixtures for
reshape, view, squeeze, unsqueeze, and flatten across ordinary, scalar, and
zero-element shapes. It also checks invalid dimensions/products and exercises
every Tensor dtype under strict ArrayFire CPU execution, requiring zero native
fallbacks and zero device-to-host transfers for supported device-resident
rank transitions. Explicit `Squeeze(dim)` follows PyTorch: negative dimensions
are normalized, non-singleton dimensions are a no-op, and rank-0 squeeze stays
rank 0; parameterless `Squeeze()` removes all singleton dimensions.

`cyxwiz-tests "[tensor_permute]"` consumes generated PyTorch 2.10 fixtures for
rank-0 through rank-4 identity, positive/negative-axis, general, and
zero-element permutations. An independent row-major index oracle covers every
Tensor dtype for ranks 2, 3, and 4 under strict ArrayFire CPU execution with
zero native fallback and zero device-to-host synchronization. Invalid, missing,
duplicate, and out-of-range dimensions fail before compute. Ranks above four
remain a declared native CPU compatibility path: compatibility mode records one
fallback, while strict mode records and rejects it before native computation.
Both `Transpose()` and `Transpose(dim0, dim1)` validate their public contracts
and delegate to this same canonical permutation path, with all-dtype rank-2/3/4
residency coverage.

`cyxwiz-tests "[tensor_indexing]"` consumes generated PyTorch 2.10 fixtures for
positive and negative dimensions, stepped and clamped slices, empty outputs,
repeated selections, and zero-element tensors. An independent row-major oracle
checks Slice and IndexSelect for every Tensor dtype at ranks 1 through 4 under
strict ArrayFire CPU execution, requiring zero native fallback and zero
device-to-host synchronization. PyTorch rejects raw negative `index_select`
indices; CyxWiz retains its documented negative-index extension and compares
the result with the equivalent normalized positive PyTorch indices. Ranks above
four are a declared native CPU compatibility path and are rejected before work
in strict mode. Bounded `At` and `Set` remain explicit, observable host
boundaries.

`cyxwiz-tests "[tensor_concat]"` consumes generated PyTorch 2.10 fixtures for
Cat, Stack, both Split overloads, and Chunk. The matrix covers positive and
negative axes, ranks 0 through 4 where applicable, scalar Stack, PyTorch's
one-dimensional empty Cat identity, zero-sized split sections, empty
dimensions, uneven final partitions, and requests for more chunks than
elements. An independent row-major oracle validates Cat and Stack for every
Tensor dtype, while direct observers prove Cat, Stack, Split, and Chunk perform
zero native fallback and zero device-to-host synchronization throughout the
supported ArrayFire CPU domain. Cat outputs and Stack outputs above rank four
are declared native CPU compatibility paths; strict mode records and rejects
them before native work. `test_graph_executable_model` covers graph-built
Concatenate forward, cached output, Split-based backward, and residency.

`cyxwiz-tests "[flatten]"` consumes generated PyTorch 2.10
`torch.nn.Flatten` forward/autograd fixtures for rank-2/3/4 inputs, positive
and negative configured `start_dim`, and exact value ordering. It exercises
both the public `FlattenLayer` and the production `FlattenModule` used by
graph-built `SequentialModel` instances. The contract also rejects backward
calls before a successful forward and rejects gradient shape/dtype drift,
preserves all five Tensor dtypes byte-for-byte, and proves rank-3-to-rank-2
forward plus rank-2-to-rank-3 backward remain ArrayFire-resident with zero
native fallback and zero host synchronization on ArrayFire CPU and every
installed CUDA/OpenCL device. `cyxwiz-route-probe` now owns the equivalent
strict Float32 forward/backward contract as a released, process-isolated route
operation. Installed oneAPI device 1 passes with exact backend/device identity,
zero fallback, and zero compute-time host synchronization. Installed oneAPI
device 0 is rejected safely: Intel `igc64.dll` raises access violation
`0xC0000005` during device input creation. The probe suppresses Windows fault
dialogs, and the route-qualification parent classifies crash/timeout outcomes;
the known-bad route is never executed inside monolithic `cyxwiz-tests`.

`cyxwiz-tests "[dropout]"` consumes generated PyTorch 2.10 Dropout fixtures
for evaluation identity, exact `p=0`/`p=1` boundaries, training
forward/backward distributions at `p=0.25`, `0.5`, and `0.9`, and explicit
cross-RNG semantics. `DropoutLayer` and the production `DropoutModule` share
one implementation, reject invalid probabilities and backward state/shape/
dtype drift, preserve the mask owned by the successful training forward, and
provide same-ArrayFire-backend seed replay without claiming bitwise equality
with PyTorch's different RNG. The released
`cyxwiz_dropout_forward_backward` route operation proves strict Float32
forward/backward residency with zero native fallback and zero compute-time host
synchronization on CPU device 0, CUDA device 0, OpenCL devices 0/1/2, and
oneAPI device 1. oneAPI device 0 remains rejected by its already-recorded Intel
`igc64.dll` access violation; Preferences and run preflight require certified
exact-route evidence, so an enumerated but incompatible route cannot be
committed silently for training.

`test_training_batcher_setup` consumes the weighted-sampler case from the same
fixture. It verifies inverse-class-frequency replacement sampling, fixed epoch
length, and class-probability parity with PyTorch over 4,096 draws. Exact draw
indices are intentionally not compared because CyxWiz and PyTorch use different
RNG implementations.

`cyxwiz-tests "[scheduler]"` consumes seven scheduler sequences and three
optimizer-warmup sequences from the same fixture. It verifies StepLR,
ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau in `min`/`max` modes,
LambdaLR-equivalent linear warmup, and the default two-phase cosine OneCycleLR
policy. The optimizer-owning `LRWarmup` wrapper additionally checks linear,
cosine, and disabled warmup learning rates together with the parameter value at
every SGD update. Initial/reset rates, patience, absolute thresholds, the
minimum LR floor, warmup progress, OneCycle final-divisor semantics, and
PyTorch-compatible overstep rejection are checked. These schedulers are backend
host-control primitives. `TrainingSchedulerController` now gives
`TrainingExecutor` a typed opt-in boundary that owns their real runtime cadence:
StepLR, ExponentialLR, CosineAnnealingLR, and LinearWarmupLR advance after a
fully completed epoch; ReduceLROnPlateau advances only after a completed epoch
that produced a finite validation loss from at least one validation sample;
OneCycleLR advances only after an actual optimizer update, including a forced
final partial accumulation flush.
Graph scheduler nodes remain blocked until GraphCompiler and the node-owning
work bind saved generic property values into this boundary.

Each `LRScheduler` also exports/imports a typed, transactional state envelope.
The scheduler tests resume every PyTorch LR sequence from a midpoint and reject
schema, type, configuration, and non-finite state drift without mutating the
active scheduler. Checkpoint v2 can persist that envelope as its reserved
`scheduler_state` payload and verifies the archive hash before import.

The executor lifecycle boundary keeps absolute completed-epoch and
optimizer-update cursors beside that backend envelope, rejects incomplete or
mismatched resume state, and restores an in-memory scheduler snapshot when the
same executor restores its best model. `test_training_scheduler_lifecycle`
checks every scheduler-family binding, exact StepLR continuation through a
hash-verified v2 scheduler payload, validation-only plateau cadence, OneCycleLR
cadence after partial accumulation flushes, trace/metric reconciliation, and
best-checkpoint cursor restoration on strict ArrayFire CPU with no native
fallback. The legacy version 1 best-model checkpoint remains a warm-start
artifact and is not reclassified as an exact persisted training resume.

The optimizer-owning `LRWarmup` wrapper persists its warmup configuration,
step cursor, and complete wrapped optimizer envelope together. Its v2
`scheduler_state` archive includes nested optimizer tensors and rejects wrapper
or optimizer configuration drift before mutation. SGD provides the required
typed learning-rate, step-count, momentum, and velocity state; its native CPU
fallback applies the same momentum equation, and `ZeroGrad()` preserves that
persistent optimizer state because gradients are caller-owned maps.

`test_debug_executor --checkpoint-serialization-only` proves the version 1.0
model-parameter payload contract for every supported Tensor dtype: Float32,
Float64, Int32, Int64, and UInt8. Shape, dtype, and exact payload bytes must
round-trip, and the file must contain exactly the declared payload size.
Unsupported dtype headers and truncated payloads must fail closed
without mutating the active model. Checkpoint reads and writes remain explicit
bounded host I/O; this is serialization evidence, not native CPU fallback.

`test_training_executor_arrow_parquet --uneven-epoch-metrics-only` runs the
focused full-epoch aggregation contract. It compares `{4, 2}` Train and Dev
metrics against evaluating the same unchanged model over each six-row role as
one batch, and covers classification loss/accuracy plus two-output regression
loss/MAE/RMSE under strict zero-native-fallback execution. Classification also
covers weighted, label-smoothed CrossEntropy with unequal class composition in
the `{4, 2}` batches.

`test_training_executor_arrow_parquet --gradient-accumulation-parity-only`
consumes four generated PyTorch effective-batch fixtures. The mean-reduction
matrix checks an uneven `{3, 2}` microbatch window, weighted and ignored targets,
and a three-microbatch window followed by a forced one-microbatch tail. A
weighted, ignored, label-smoothed sum-reduction case proves that microbatch
gradients add without mean normalization and that the reported epoch loss is
the exact effective-batch sum. Bias parameters are compared at every SGD
boundary, final parameters and optimizer-step counts must match, terminal
lifecycle truth is exact, and native CPU fallback is forbidden.

`test_training_executor_arrow_parquet --paused-control-matrix-only` exercises
resume and cancellation while paused across Arrow, Parquet, external
`IBatcher`, and sequence `ISequenceBatcher` runs. It proves that paused work
does not advance, resumed work consumes every remaining batch exactly once,
and cancellation cannot release the pause wait into another batch. Arrow,
Parquet, and external tabular cases require strict zero-fallback ArrayFire
execution. Sequence exposes its declared compatibility policy because token
accuracy is not yet a strict-resident metric.

`test_training_executor_dataset_handle` exercises the same production
`TrainingExecutor`/`DatasetBatcher` control loop for the legacy
`DatasetHandle` compatibility path. Its barrier-controlled resume case proves
three batches and three optimizer updates occur exactly once; its cancellation
case proves a paused run closes after the first batch without another update.
Both cases commit CPU device 0 through the pending Preferences selection,
repeat exact-route authorization in run preflight, and require certified
`arrayfire_cpu:0` execution with strict zero native fallback. Target-only seams
for unrelated optional preprocessing and annotation services fail loudly if
that out-of-scope behavior is entered.

Run:

```powershell
cmake --build build --config Release --target test_computation_truth_tfidf_loss -- /m:4 /v:minimal
build\bin\Release\test_computation_truth_tfidf_loss.exe

cmake --build build --config Debug --target test_training_batcher_setup -- /m:4 /v:minimal
build\bin\Debug\test_training_batcher_setup.exe

cmake --build build --config Debug --target cyxwiz-tests -- /m:4 /v:minimal
build\bin\Debug\cyxwiz-tests.exe "[scheduler]"

build\bin\Debug\cyxwiz-tests.exe "[classification_metrics]"

build\bin\Debug\cyxwiz-tests.exe "[tensor_ownership]"

build\bin\Debug\cyxwiz-tests.exe "[tensor_host_access]"

build\bin\Debug\cyxwiz-tests.exe "[tensor_layout]"

build\bin\Debug\cyxwiz-tests.exe "[tensor_shape]"

build\bin\Debug\cyxwiz-tests.exe "[tensor_permute]"

build\bin\Debug\cyxwiz-tests.exe "[tensor_indexing]"

build\bin\Debug\cyxwiz-tests.exe "[tensor_concat]"

build\bin\Debug\cyxwiz-tests.exe "[flatten]"

build\bin\Debug\cyxwiz-tests.exe "[dropout]"

cmake --build build --config Debug --target test_training_executor_arrow_parquet -- /m:4 /v:minimal
build\bin\Debug\test_training_executor_arrow_parquet.exe --uneven-epoch-metrics-only

build\bin\Debug\test_training_executor_arrow_parquet.exe --gradient-accumulation-parity-only

build\bin\Debug\test_training_executor_arrow_parquet.exe --paused-control-matrix-only

cmake --build build --config Debug --target test_training_executor_dataset_handle -- /m:4 /v:minimal
build\bin\Debug\test_training_executor_dataset_handle.exe
```

These checks work when `PyTorch/LibTorch: OFF`: PyTorch is used only to
generate the checked-in fixture, while the C++ test has no LibTorch runtime
dependency. If LibTorch is enabled for other computation-truth tests, keep it
outside the engine runtime boundary.

Latest observed run also exercised `LinearLayer` while ArrayFire GPU was active.

`test_computation_truth_transformer_primitives`

- Verifies embedding, positional encoding, attention, layer norm, encoder, and
  decoder primitive parity.
- Verifies single-block TransformerEncoder and causal TransformerDecoder
  backward parity for input gradients plus representative attention, layer norm,
  and feed-forward parameter gradients against PyTorch-derived fixtures.
- Verifies masked MultiHeadAttention backward parity for input and projection
  gradients against PyTorch-derived fixtures.
- Verifies two-block TransformerEncoder and causal TransformerDecoder stack
  backward parity with layer-indexed gradient checks for both blocks.
- Verifies tiny causal-LM vocabulary logits and loss against PyTorch linear and
  cross-entropy semantics.
- Verifies a tiny transformer token-classification training step using
  token-shaped CrossEntropy mean reduction, ignore_index, label smoothing,
  backward propagation, and SGD loss decrease.
- Verifies BERT-style CLS extraction, sequence-classifier logits, and
  token-classifier logits against PyTorch indexing and linear-head semantics.
- Verifies GPT-style generation candidate probabilities for temperature,
  top-k, top-p, and greedy selection against PyTorch softmax/top-k reference
  behavior, with hard-coded PyTorch-derived constants when LibTorch is not
  enabled.
- Verifies deterministic multinomial replay over the PyTorch-verified
  candidate distribution. CyxWiz does not require its C++ RNG stream to match
  `torch.multinomial` exactly.

Run:

```powershell
cmake --build build --config Debug --target test_computation_truth_transformer_primitives -- /m:4 /v:minimal
build\bin\Debug\test_computation_truth_transformer_primitives.exe

cmake --build build --config Release --target test_computation_truth_transformer_primitives -- /m:4 /v:minimal
$env:PATH = "<local-libtorch>\lib;$env:PATH"
build\bin\Release\test_computation_truth_transformer_primitives.exe
```
