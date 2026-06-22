# To Fix 30 - Missing ML Algorithm Parity, Accuracy Metrics, and CPU/GPU Test Coverage

## Purpose

Capture the remaining ML algorithm gaps in the engine, using PyTorch as the
implementation reference, and define the required test work for correctness,
accuracy, and backend support (CPU + GPU) across all implemented ML
components.

## Why This Exists

The sentiment pipeline currently uses available loss/optimization paths but is
lacking parity with common PyTorch classification and segmentation practices. We
need to close these gaps and add deterministic accuracy checks so model quality
regressions are caught early. Coverage should also verify every implemented
component works on both CPU and GPU backends.

## Missing / Incomplete Pieces (PyTorch Reference)

### 1) Losses and objective functions

- Add Soft Dice style loss support
  - Reference behavior: common implementation built from probability outputs and
    one-hot/interpreted targets in PyTorch custom functions (`dice_loss` style).
  - Current gap: no `SoftDiceLoss` node/metadata/path in loss builder.
- Add class-balancing support in `CrossEntropyLoss` parity
  - Reference: `torch.nn.CrossEntropyLoss(weight=..., label_smoothing=...)`.
  - Current gap: full constructor params are not reflected in graph schema and
    runtime options.
- Add Focal Loss style objective (classification hard-example focus)
  - Reference: typical PyTorch custom implementation using
    `-alpha * (1-p_t)**gamma * log(p_t)` around CE logits/probs.
  - Current gap: no node or executor support.
- Add (optional) Tversky / Focal-Tversky family support
  - Reference: common in segmentation PyTorch repos for false-positive/false-negative
    control with `alpha`, `beta`, `gamma`.
  - Current gap: absent.
- Add Jaccard/IoU or Dice-CE hybrid options (segmentation/class-imbalanced tasks)
  - Reference: PyTorch ecosystem custom losses combining CE + Dice/IoU terms.
  - Current gap: no combined/compound criterion path.

### 2) Accuracy and evaluation metrics

- Add first-class accuracy metric
  - Reference: `torchmetrics`-style `Accuracy`; supports multiclass,
    binary, top-k variants.
  - Current gap: no explicit accuracy metric node and no standard benchmark
    output in training/eval flows.
- Add standard classification metrics
  - Reference: `Precision`, `Recall`, `F1` patterns from PyTorch ecosystem.
  - Current gap: missing metric primitives and report formatting.
- Add confusion-matrix / sample-level diagnostics
  - Reference: standard ML workflows in PyTorch docs/examples via
    `sklearn.metrics` or custom tensor counting.
  - Current gap: no per-class summary available from run outputs.

### 3) Training/evaluation flow improvements

- Expose reproducible train/eval split and deterministic metric logging
  - Reference: PyTorch training loops with consistent eval mode and separate
    train/validation/test passes.
  - Current gap: inference/eval metrics are not consistently separated from
    training-loss output.
- Add full dataset pipeline parity checks around class imbalance and label formats
  - Reference: robust label preprocessing before loss/metric calculations.
  - Current gap: validation around target dtype/shape expectations for each loss type.

## Test Plan (Must Have)

### A) Accuracy correctness

- Add tests for sentiment classifier and any equivalent classification example:
  - verify train-time and post-train accuracy computation;
  - verify expected baseline threshold behavior (binary) and `argmax` behavior
    (multiclass);
  - assert metric updates and final reporting are deterministic for fixed seed.
- Add tests for each implemented metric family
  - Accuracy baseline sanity cases (all-correct, all-wrong, empty batch, mixed
    top-k behavior if supported).

### B) Backend support tests (all backends for all implemented nodes)

For every implemented and tested loss/metric/operator in ML flow, run:

1. CPU-only execution pass (train/eval).
2. GPU execution pass (train/eval) where device available.

If GPU is unavailable, tests must report skip (not fail) and preserve explicit
coverage notes.

Items to include in the matrix:

- Classification losses:
  - `CrossEntropyLoss`
  - `BCEWithLogitsLoss`
  - `BCELoss`
- Regression losses:
  - `MSELoss`
  - `L1Loss`
  - `SmoothL1Loss`
- Missing parity items once added:
  - `SoftDiceLoss`
  - `FocalLoss`
  - `Tversky` family (if implemented)
- Metrics once added:
  - `Accuracy`
  - `Precision`, `Recall`, `F1` (or equivalent grouped validation metric)

For each test item:

- validate forward/backward numerics,
- validate device placement is coherent across model, input, target, and outputs,
- validate no CPU-only assumptions in kernels/pipeline.

## Acceptance Criteria

- Soft Dice/focal-style losses are added as engine-level nodes with documentation
  and parser/builder coverage.
- `CrossEntropyLoss` parity options (`weight`, `label_smoothing`) are supported in
  schema and runtime.
- Accuracy metric is available as a first-class evaluation output with reproducible
  values.
- Sentiment and related classification examples can report accuracy.
- Every implemented ML training/eval primitive has both CPU and GPU test coverage,
  including shape checks and deterministic metric assertions.
- Regression is blocked until all to-fix items above are implemented and tested.
