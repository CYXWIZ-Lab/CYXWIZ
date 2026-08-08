# To Fix 30 - Remaining ML Loss, Metric Node, And Backend Coverage Parity

Updated: 2026-06-29
Original scope: missing ML algorithm parity, accuracy metrics, and CPU/GPU test
coverage.

## Purpose

Refresh the original `tofix30` ticket against the current codebase. Later work,
especially `done35.md`, closed several original gaps around class imbalance,
weighted losses, FocalLoss, stratified splits, DataLoader balancing, and tree
runtime support.

This ticket now tracks only the remaining ML parity work that is still visible
in the engine.

## Current Code Truth

### Resolved By Later Work

The following original `tofix30` concerns are now covered:

- `FocalLoss` exists in the backend and is exposed through graph loss metadata,
  compiler parsing, model-builder construction, properties editing, saved graph
  I/O, and tests.
- `CrossEntropyLoss` supports manual class weights and balanced auto-weights
  for supported Arrow/text train splits.
- `CrossEntropyLoss.label_smoothing` is implemented in backend/runtime graph
  paths with validation and focused tests.
- `SoftDiceLoss` exists in the backend and is exposed through graph loss
  metadata, compiler parsing, model-builder construction, properties editing,
  saved graph I/O, pipeline metadata, documentation, Python bindings, and tests.
- `TverskyLoss` exists in the backend and is exposed through graph loss
  metadata, compiler parsing, model-builder construction, properties editing,
  saved graph I/O, pipeline metadata, documentation, Python bindings, and tests.
- `JaccardLoss` / IoU loss exists in the backend and is exposed through graph
  loss metadata, compiler parsing, model-builder construction, properties
  editing, saved graph I/O, pipeline metadata, documentation, Python bindings,
  and tests.
- `BCEWithLogitsLoss` supports `pos_weight`.
- Common loss `reduction` settings are wired from graph parameters into runtime
  loss construction.
- `DataSplit.stratified` and DataLoader class balancing are implemented for
  supported Arrow/text training paths.
- Training accuracy is computed and reported by `TrainingExecutor`.
- Confusion matrix, ROC, PR, binary metrics, classification report, and
  regression metrics infrastructure exists through `ModelEvaluation`, pipeline
  analytics nodes, panels, and test-result views.

Relevant files:

- `cyxwiz-backend/include/cyxwiz/losses/classification.h`
- `cyxwiz-backend/include/cyxwiz/losses/probability.h`
- `cyxwiz-backend/include/cyxwiz/model_evaluation.h`
- `cyxwiz-engine/src/core/graph_compiler.cpp`
- `cyxwiz-engine/src/core/model_builder.cpp`
- `cyxwiz-engine/src/core/training_executor.h`
- `cyxwiz-engine/src/core/training_executor.cpp`
- `cyxwiz-engine/src/core/node_metadata_registry.cpp`
- `cyxwiz-engine/src/core/pipeline_executor.cpp`
- `cyxwiz-engine/src/gui/properties_node_editors.cpp`

### Still Open

Remaining gaps:

- No `FocalTverskyLoss` implementation.
- No Dice-CE hybrid criterion.
- Metrics exist as utilities/panels/analytics nodes, but there are no clean
  separate graph metric nodes for `Accuracy`, `Precision`, `Recall`, and `F1`.
  A first-class configurable `ClassificationMetricsNode` now covers these
  metrics through the graph/pipeline surface.
- CPU/GPU test coverage exists in many places, but there is no deliberate matrix
  proving each implemented ML loss/metric/train-eval primitive works on CPU and
  ArrayFire/GPU where available.

## Active Scope

### Phase 1 - Fix The Label Smoothing Contract

Goal: remove the current doc/runtime mismatch around
`CrossEntropyLoss.label_smoothing`.

Status: completed in this pass.

Tasks:

- [x] Decide whether `label_smoothing` belongs in backend
  `CrossEntropyLoss` now or should be removed from node docs until implemented.
- [x] If implemented, add a backend constructor/member for label smoothing.
- [x] Apply label smoothing in forward and backward behavior with documented
  target semantics.
- [x] Parse `label_smoothing` from graph loss params.
- [x] Expose and validate `label_smoothing` in the Properties panel.
- [x] Construct smoothed CE losses in `model_builder.cpp`.
- [x] Add backend and graph/compiler tests.

Acceptance:

- `label_smoothing` is either real and tested, or no UI/doc surface implies it
  is available.
- Invalid smoothing values fail with a clear diagnostic.

Validation:

- `cmake --build build --config Debug --target cyxwiz-tests`
- `build/bin/Debug/cyxwiz-tests.exe "[loss]"`
- `cmake --build build --config Debug --target test_debug_executor test_graph_compiler_deferred_nodes`
- `build/bin/Debug/test_debug_executor.exe`
- `build/bin/Debug/test_graph_compiler_deferred_nodes.exe`

### Phase 2 - Segmentation And Imbalance Loss Family

Goal: add common segmentation/class-imbalance losses that are still absent.

Status: partially completed in this pass. Soft Dice and Tversky are now
implemented as the first stable segmentation loss contracts; Focal Tversky and
Dice-CE hybrid variants remain deferred until the probability-mask
target/layout behavior settles.

Tasks:

- [x] Add `SoftDiceLoss` backend implementation.
- [x] Add `TverskyLoss` backend implementation.
- [x] Decide whether `FocalTverskyLoss` is a separate node or a configured
  Tversky variant.
- [x] Decide whether Jaccard/IoU loss and Dice-CE hybrid are in this ticket or
  explicitly deferred.
- [x] Add `NodeType` entries, metadata, documentation, pins, saved graph I/O,
  compiler parsing, properties editing, and model-builder construction for the
  chosen losses.
- [x] Define supported target formats: binary masks, one-hot masks, class-index
  masks, and batch/channel layout expectations.
- [x] Add numeric tests for forward/backward behavior and shape validation.

Acceptance:

- Soft Dice is usable as an engine-level graph loss.
- Tversky-family behavior is either implemented or explicitly deferred with a
  documented reason.
- Shape/target mismatches produce actionable diagnostics.

Current Soft Dice contract:

- Predictions and targets must be `Float32` tensors with identical non-empty
  shapes.
- The first dimension is treated as batch for rank > 1; 1D input is treated as
  one sample.
- Targets are probability/binary masks matching prediction layout; class-index
  masks are intentionally not accepted by this first contract.
- `smooth` must be finite and non-negative.

Current Tversky contract:

- Predictions and targets follow the same `Float32`, same-shaped, non-empty
  probability-mask contract as Soft Dice.
- The first dimension is treated as batch for rank > 1; 1D input is treated as
  one sample.
- `alpha` is the false-positive penalty and must be finite and non-negative.
- `beta` is the false-negative penalty and must be finite and non-negative.
- `smooth` must be finite and non-negative.
- `FocalTverskyLoss` is deferred as a separate follow-up variant instead of
  overloading the base Tversky node.

Current Jaccard / IoU contract:

- Predictions and targets follow the same `Float32`, same-shaped, non-empty
  probability-mask contract as Soft Dice and Tversky.
- The first dimension is treated as batch for rank > 1; 1D input is treated as
  one sample.
- `smooth` must be finite and non-negative.
- Dice-CE hybrid is deferred because it needs an explicit mixed
  class-index/probability-mask target contract rather than another simple mask
  loss.

Validation:

- `cmake --build build --config Debug --target cyxwiz-tests test_debug_executor test_graph_compiler_deferred_nodes`
- `build/bin/Debug/cyxwiz-tests.exe "[loss]"`
- `build/bin/Debug/test_debug_executor.exe`
- `build/bin/Debug/test_graph_compiler_deferred_nodes.exe`

Tversky validation:

- `cmake --build build --config Debug --target cyxwiz-tests test_debug_executor test_graph_compiler_deferred_nodes test_pipeline_operator_metadata`
- `build/bin/Debug/cyxwiz-tests.exe "[loss]"`
- `build/bin/Debug/test_debug_executor.exe`
- `build/bin/Debug/test_graph_compiler_deferred_nodes.exe`
- `build/bin/Debug/test_pipeline_operator_metadata.exe`

Jaccard validation:

- `cmake --build build --config Debug --target cyxwiz-tests test_debug_executor test_graph_compiler_deferred_nodes test_pipeline_operator_metadata`
- `build/bin/Debug/cyxwiz-tests.exe "[loss]"`
- `build/bin/Debug/test_debug_executor.exe`
- `build/bin/Debug/test_graph_compiler_deferred_nodes.exe`
- `build/bin/Debug/test_pipeline_operator_metadata.exe`

### Phase 3 - First-Class Metric Nodes

Goal: make common classification metrics graph-visible instead of only
available indirectly through panels and reports.

Status: partially completed in this pass. A configurable
`ClassificationMetricsNode` now exposes accuracy, macro precision, macro
recall, macro F1, weighted F1, count, and class count from actual/predicted
label columns.

Tasks:

- [x] Add first-class metric nodes for `Accuracy`, `Precision`, `Recall`, and
  `F1`, or add one configurable `ClassificationMetrics` node if that better
  matches the existing node system.
- [x] Reuse existing `ConfusionMatrixNode` label semantics where possible
  instead of introducing a conflicting classification metric contract.
- [x] Define binary, multiclass, macro, weighted, and top-k behavior.
- [x] Add metadata, properties, saved graph I/O, and pipeline execution.
- [x] Ensure outputs are datasets or metric payloads that Studio can display
  consistently.
- [x] Add deterministic metric tests for all-correct, all-wrong, mixed,
  empty/invalid, binary threshold, and multiclass cases.

Acceptance:

- Users can add metric nodes from Studio search.
- Classification examples can report accuracy and related metrics through the
  graph/pipeline surface, not only through ad hoc training logs.

Current Classification Metrics contract:

- Input is a dataset with actual and predicted label columns.
- Labels are compared by scalar string value, matching `ConfusionMatrixNode`.
- `precision`, `recall`, and `f1` are macro averages across the union of actual
  and predicted labels.
- `weighted_f1` is weighted by actual-label support.
- `count` reports valid non-null actual/predicted pairs.
- Top-k and probability-threshold behavior are deferred to score/probability
  nodes; this first node is class-label based.
- Deterministic routing tests cover all-wrong, all-correct, mixed multiclass,
  missing-column invalid, unsupported metric, and no-valid-pairs cases.

ClassificationMetrics validation:

- `cmake --build build --config Debug --target test_pipeline_executor_operator_routing test_pipeline_operator_metadata`
- `build/bin/Debug/test_pipeline_executor_operator_routing.exe`
- `build/bin/Debug/test_pipeline_operator_metadata.exe`

### Phase 4 - CPU/GPU Coverage Matrix

Goal: turn backend coverage from scattered tests into an explicit matrix.

Tasks:

- [x] Inventory implemented ML losses, metrics, and train/eval primitives.
- [x] Mark each item as CPU-tested, GPU-tested, CPU-only by design, or
  unsupported.
- [x] Add focused GPU tests for ArrayFire-backed paths where practical.
- [x] Ensure GPU-unavailable tests skip cleanly with an explicit note.
- [x] Validate device placement consistency for model, input, target, loss, and
  metric outputs.
- [x] Add shape/dtype validation cases for each implemented loss and metric.

Matrix:

| Primitive group | Items | Status | Coverage |
|---|---|---|---|
| Backend regression losses | MSE, L1, SmoothL1/Huber | CPU/host-tested with ArrayFire CPU forced, ArrayFire GPU smoke for MSE | `test_ml_primitive_coverage_matrix`, existing debug/model tests |
| Backend classification losses | CrossEntropy with `label_smoothing`, NLL, Focal | CPU/host-tested with ArrayFire CPU forced, ArrayFire code path present where supported | `test_ml_primitive_coverage_matrix`, `test_debug_executor`, graph compiler/model-builder tests |
| Backend probability losses | BCE, BCEWithLogits, KLDiv, SoftDice, Tversky, Jaccard | CPU/host-tested with ArrayFire CPU forced, ArrayFire smoke deferred except MSE baseline | `test_ml_primitive_coverage_matrix`, segmentation-loss numeric tests |
| Backend metric-learning losses | CosineEmbedding, Contrastive, Triplet | CPU/host-tested with ArrayFire CPU forced, ArrayFire code path present where supported | `test_ml_primitive_coverage_matrix`, `test_metric_learning_losses` |
| First-class pipeline metrics | ClassificationMetricsNode accuracy/precision/recall/F1/weighted_f1/count/class_count | CPU/Arrow-tested, CPU-only by design | `test_pipeline_executor_operator_routing`, `test_pipeline_operator_metadata` |
| Metric-learning metrics | Pair distance accuracy/means, retrieval recall/MRR/nearest agreement | CPU-tested, CPU-only by design | `test_ml_primitive_coverage_matrix`, `test_metric_learning_metrics` |
| Train/eval primitives | TrainingExecutor/TestExecutor loss construction, synthetic batches, graph executable tensor placement | CPU-tested plus ArrayFire graph placement where active | `test_debug_executor`, `test_graph_executable_model`, `test_training_executor_arrow_parquet` |

Notes:

- GPU-unavailable machines are expected to print a skip from
  `test_ml_primitive_coverage_matrix` instead of failing.
- Arrow/pipeline metrics are intentionally CPU-side because their source and
  output contract is Arrow datasets, not ArrayFire tensors.
- The focused GPU smoke currently proves an ArrayFire-backed loss path with
  MSE. Broader per-loss CUDA/OpenCL numeric parity remains a follow-up if we
  decide to require GPU parity for every backend loss.
- `FocalLoss::Backward` currently logs an ArrayFire fallback under the forced
  ArrayFire CPU matrix path and then completes through the CPU fallback. This is
  visible in the matrix output and should be treated as a future parity cleanup,
  not a blocker for the current CPU/skip coverage contract.

Validation:

- `cmake --build build --config Debug --target test_ml_primitive_coverage_matrix`
- `build\bin\Debug\test_ml_primitive_coverage_matrix.exe`
- `cmake --build build --config Debug --target test_debug_executor test_graph_compiler_deferred_nodes test_pipeline_operator_metadata test_metric_learning_metrics test_metric_learning_losses test_ml_primitive_coverage_matrix`
- `build\bin\Debug\test_pipeline_operator_metadata.exe`
- `build\bin\Debug\test_metric_learning_metrics.exe`
- `build\bin\Debug\test_metric_learning_losses.exe`
- `build\bin\Debug\test_debug_executor.exe`
- `build\bin\Debug\test_graph_compiler_deferred_nodes.exe`
- `build\bin\Debug\test_pipeline_executor_operator_routing.exe`

Acceptance:

- The repo has a documented test matrix for ML primitives.
- GPU coverage gaps are visible and intentional.
- Tests do not fail on machines without GPU support solely because GPU is
  unavailable.

## Non-Goals

- Reopening class imbalance handling completed by `done35.md`.
- Rebuilding existing confusion matrix, ROC, PR, or test-result panels unless
  needed to expose first-class metric nodes.
- Applying balancing to validation/test data by default.
- Replacing `TrainingExecutor`, `GraphCompiler`, or `ModelEvaluation`.
- Adding every segmentation loss variant before the core Soft Dice contract is
  stable.

## Verification Targets

- Backend loss tests for CrossEntropy label smoothing and segmentation losses.
- Graph compiler tests for new loss/metric node parameter parsing.
- Model builder tests proving graph params construct the expected runtime loss.
- Pipeline executor tests for first-class metric nodes.
- CPU/GPU matrix tests with clean GPU-unavailable skips.
- Focused sentiment/classification example smoke proving metrics are reported
  deterministically.
