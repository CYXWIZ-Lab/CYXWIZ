# CyxWiz Backend Validation Matrix

## Contents

1. Test principles
2. Change matrix
3. Backend and fallback matrix
4. Performance acceptance
5. Pre-commit gate

## 1. Test Principles

Test behavior and evidence together. A numerically correct result may still
hide host copies or native fallback; a zero-fallback trace may still compute
the wrong result.

Use deterministic fixtures and explicit tolerances. Cover edge cases relevant
to the operation, including empty/partial batches, shape mismatch, dtype,
reduction, weighting, smoothing, ignored targets, and finite extreme inputs.

Cover checked shape/byte overflow, invalid indices, deterministic stochastic
behavior, ownership lifetime, and persisted API/config compatibility when the
change can affect them.

Keep focused tests fast enough for iteration. Scale validation with blast
radius: shared Tensor, device, fallback, loss, optimizer, or TrainingExecutor
changes require broader gates than a leaf operator.

## 2. Change Matrix

### Tensor Ownership, Layout, or Host Access

Require:

- numerical and logical-shape round-trip tests;
- device-to-device layout conversion with zero host-sync evidence;
- read-only host access preserving device state;
- mutable host access invalidating/rebuilding device state;
- source guards against forbidden hot-path accessors;
- full core backend suite.

### Layer, Activation, Loss, Metric, or Optimizer

Require:

- forward numerical parity;
- backward/gradient numerical parity where applicable;
- all supported reductions and target forms;
- rank, shape, and dtype validation;
- strict residency test with device-current inputs;
- declared fallback behavior for unsupported cases;
- multi-batch integration through optimizer update.

For weighted or smoothed classification, test weighting denominator and
gradient semantics, not only finite output.

### Device Discovery, Selection, Context, or Placement

Require:

- CPU -> accelerator -> CPU switching in one process;
- requested versus effective identity assertions;
- one bind per run before compute/fallback events;
- no active-run mutation;
- placement, preflight, trace, and GUI agreement;
- unavailable optional backend behavior without relabeling another backend.

### Fallback or Exception Handling

Require both:

- compatibility policy: fallback completes and records complete evidence;
- strict policy: attempt is recorded and fails before native compute.

Run source-contract scans for unclassified ArrayFire catches and ambiguous
fallback wording.

### Training Execution or Reporting Cadence

Require:

- complete forward/loss/metric/backward/optimizer execution;
- exact callback and metric-sampling cadence;
- validation, early stopping, terminal metrics, and checkpoint behavior;
- trace group event/byte totals reconciling with run totals;
- no unknown or large repeated hot-path materialization;
- checkpoint/export reads classified as bounded outputs;
- strict zero-native-fallback coverage for the supported vertical.

### GUI Execution Truth

Require:

- focused state/render contract tests where available;
- rebuilt Release Engine;
- live run proving selected, requested, and effective backend/device;
- Dashboard/viewport/preference agreement;
- fallback, synchronization, reporting cadence, and residency verdict truth;
- responsive task/progress behavior during preparation and training.

## 3. Backend and Fallback Matrix

Always exercise ArrayFire CPU. Exercise every installed backend affected by the
change:

- CUDA;
- OpenCL;
- oneAPI.

Do not make optional hardware mandatory for all developer machines. Detect and
skip unavailable backends explicitly, while keeping their discovery and
selection branches covered by focused tests where practical.

For each available backend, verify:

- exact effective backend and device identity;
- numerical parity within justified tolerance;
- expected Tensor shape/layout;
- zero undeclared native fallback;
- expected host-sync categories and bytes.

Do not treat successful ArrayFire CPU execution as proof that the native C++
fallback works, or vice versa.

## 4. Performance Acceptance

Capture pre-change and post-change Release runs with identical:

- graph and tensor dimensions;
- dataset/fixture and seed;
- batch size and batch count;
- backend/device;
- fallback policy;
- reporting cadence;
- warmup and measurement method.

Report multiple samples when practical and prefer median throughput. Include:

- batches/sec or samples/sec;
- synchronization events and bytes per batch;
- fallback count;
- relevant stage timings;
- trace path and grouped boundaries;
- utilization source when available.

Reject an optimization when it changes results, shapes, reductions, update
count, device identity, or fallback policy. Do not improve metrics by hiding
trace events or reducing required work.

## 5. Pre-Commit Gate

Run this sequence before committing backend computation work:

1. Inspect `git status --short` and the complete diff. Leave unrelated user
   changes untouched.
2. Run `git diff --check`.
3. Build every affected target in the configuration used by focused tests.
4. Run focused numerical, shape, error, and residency tests for the change.
5. Run relevant standalone integration executables, especially
   `test_training_executor_arrow_parquet` for training-path changes.
6. Run the full `cyxwiz-tests` suite for shared Tensor/backend contracts or
   other broad changes.
7. Build the Release Engine for Engine runtime, GUI, public DLL, or linkage
   changes.
8. Run a live Release trace for device/fallback/GUI/materialization behavior.
9. Run and record fixed-fixture before/after benchmarks for performance claims.
10. Check hot-loop logging/allocation volume and persisted contract
    compatibility when touched.
11. Reinspect the final diff and stage only intended source, test, and required
    documentation files.

Adapt exact CMake target names to the current build tree; do not assume stale
binaries represent modified source. If a required gate cannot run, stop short
of commit and report the missing validation and residual risk.

Minimum completion evidence must name the commands or executables run and
their pass/fail totals. For performance work, include measured runtime facts
rather than only saying the run looked faster.
