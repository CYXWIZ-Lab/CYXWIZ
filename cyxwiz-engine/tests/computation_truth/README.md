# CyxWiz Engine Computation Truth Tests

This folder is for deterministic numerical parity tests.

Purpose:

- Compare CyxWiz engine computations against stable references.
- Catch silent math drift in preprocessing, model forward passes, losses, gradients, optimizers, and training lifecycle.
- Keep correctness tests independent of the GUI.

Reference stack:

- PyTorch for tensor/model/loss/optimizer parity.
- scikit-learn for TF-IDF parity when available.
- Small in-repo reference implementations when external Python packages are unavailable.

PyTorch is a reference-generation dependency, not an Engine runtime
dependency. Small generated fixtures are checked in under `fixtures/`, copied
beside their test executable at build time, and consumed by the C++ tests.

`coverage_inventory.json` is the machine-readable Plan 39 Tier-0/Tier-1
baseline. It records ownership, semantic contracts, oracle choice, current
tests, execution path, honest coverage status, and remaining evidence gaps.

Initial target cases:

- TF-IDF bounded materialization values and shape.
- Dense forward parity.
- CrossEntropy loss parity.
- One-step Adam/AdamW parity.
- Learning-rate scheduler sequence and boundary parity.
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
- Verifies `CrossEntropyLoss` against generated PyTorch `cross_entropy`
  fixtures for class-last rank-1/2/3 logits, index and soft targets,
  `none`/`sum`/`mean`, class weights, ignored targets, label smoothing,
  extreme logits, all-ignored mean behavior, forward loss, and full logit
  gradients.
- Verifies `LinearLayer` forward output against a generated PyTorch linear fixture.
- Verifies `LinearLayer` backward values:
  - gradient with respect to input
  - summed weight gradients, matching PyTorch autograd
  - summed bias gradients, matching PyTorch autograd
- Verifies `AdamWOptimizer` first-step parameter update with decoupled weight
  decay against a generated PyTorch AdamW fixture.
- Activates ArrayFire CPU exactly, binds an immutable execution context, and
  runs the fixture-backed core checks with native CPU fallback forbidden.
- Rejects native fallback attempts and undeclared host synchronization; only
  bounded, attributed test-output readbacks are allowed.
- Proves every supported CrossEntropy rank uses the same strict ArrayFire CPU
  path; rank-3 is reshaped on device and is not a hidden native CPU variant.

The Linear, CrossEntropy, AdamW, and weighted-sampler expectations come from the checked-in
`fixtures/training_core_pytorch.json` fixture. Regenerate it with:

```powershell
python cyxwiz-engine/tests/computation_truth/reference/generate_training_core_fixtures.py
```

The pinned reference package is listed in
`reference/requirements.txt`. Regenerating fixtures is an explicit developer
action; normal configuration, builds, and test runs do not invoke Python or
require network access.

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
host-control primitives; graph scheduler nodes remain blocked until
TrainingExecutor owns their update cadence, run state, and checkpoint
restoration.

Each `LRScheduler` also exports/imports a typed, transactional state envelope.
The scheduler tests resume every PyTorch LR sequence from a midpoint and reject
schema, type, configuration, and non-finite state drift without mutating the
active scheduler. Checkpoint v2 can persist that envelope as its reserved
`scheduler_state` payload and verifies the archive hash before import.

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

Run:

```powershell
cmake --build build --config Release --target test_computation_truth_tfidf_loss -- /m:4 /v:minimal
build\bin\Release\test_computation_truth_tfidf_loss.exe

cmake --build build --config Debug --target test_training_batcher_setup -- /m:4 /v:minimal
build\bin\Debug\test_training_batcher_setup.exe

cmake --build build --config Debug --target cyxwiz-tests -- /m:4 /v:minimal
build\bin\Debug\cyxwiz-tests.exe "[scheduler]"

build\bin\Debug\cyxwiz-tests.exe "[flatten]"

build\bin\Debug\cyxwiz-tests.exe "[dropout]"

cmake --build build --config Debug --target test_training_executor_arrow_parquet -- /m:4 /v:minimal
build\bin\Debug\test_training_executor_arrow_parquet.exe --uneven-epoch-metrics-only

build\bin\Debug\test_training_executor_arrow_parquet.exe --gradient-accumulation-parity-only

build\bin\Debug\test_training_executor_arrow_parquet.exe --paused-control-matrix-only
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
