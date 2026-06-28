# To Fix 35 - Class Imbalance, Weighted Loss, And Tree Runtime Gaps

Created: 2026-06-27
Source: Analysis request on whether the CyxWiz engine supports weighted loss,
decision trees, and DataLoader class balancing for imbalanced sentiment
training examples.

## Purpose

The sentiment examples can train on imbalanced multi-class text datasets, but
the current engine does not provide first-class imbalance handling in the
normal CyxGraph training path. This document records the current code truth and
turns the missing pieces into explicit implementation targets.

## Progress Update

Implemented so far:

- Studio loss search now exposes the supported training losses, including
  `FocalLoss`.
- `DataSplit.stratified` is compiled and honored for Arrow/text-backed
  training splits.
- DataLoader class balancing parameters are compiled, runtime-applied for
  Arrow/text train batchers, and editable from the DataLoader Open Dialog.
- The DataLoader Properties panel is summary-only and points users to Open
  Dialog for complex settings.
- CrossEntropy manual class weights, CrossEntropy balanced auto-weights for
  supported Arrow/text train splits, BCEWithLogits positive weighting,
  loss `reduction`, SmoothL1/Huber `beta`, and FocalLoss `alpha`/`gamma` are
  wired from graph parameters into runtime loss construction.
- Simple loss nodes are controlled from the Properties panel, with no separate
  Open Dialog.

## Current Code Truth

### Weighted Loss

Current state:

- `CrossEntropyLoss` exists and supports `reduction` plus `ignore_index`.
- `BCELoss` exists and supports `reduction` plus `eps`.
- `BCEWithLogitsLoss` exists and supports `reduction`.
- `FocalLoss(alpha, gamma)` exists in the backend loss library.
- Normal CyxGraph training recognizes these loss node types:
  `MSELoss`, `CrossEntropyLoss`, `BCELoss`, `BCEWithLogits`, `L1Loss`,
  `SmoothL1Loss`, `HuberLoss`, and `NLLLoss`.

Resolved in this pass:

- `CrossEntropyLoss` supports manual per-class weights and balanced
  auto-weights for supported Arrow/text train splits.
- `BCEWithLogitsLoss` supports `pos_weight`.
- Graph/compiler loss parameter wiring covers supported weight parameters,
  `reduction`, `ignore_index`, SmoothL1/Huber `beta`, and FocalLoss
  `alpha`/`gamma`.
- `FocalLoss` is graph-exposed, discoverable, and runtime-constructed.

Still pending / explicitly out of scope:

- No generic `sample_weight` path in standard graph training loss execution.
- `BCELoss` remains unweighted by decision; use `BCEWithLogits.pos_weight`,
  CrossEntropy class weights, or DataLoader balancing for imbalance handling.

Affected files:

- `cyxwiz-backend/include/cyxwiz/losses/classification.h`
- `cyxwiz-backend/include/cyxwiz/losses/probability.h`
- `cyxwiz-backend/src/algorithms/loss.cpp`
- `cyxwiz-engine/src/core/graph_compiler.cpp`
- `cyxwiz-engine/src/core/model_builder.cpp`
- `cyxwiz-engine/src/core/graph_compiler.h`
- `cyxwiz-engine/src/core/node_metadata_registry.cpp`
- `cyxwiz-engine/src/gui/properties_node_editors.cpp`

### DataLoader Balanced Resampling

Current state:

- `DataLoader` configuration owns training-loop and iteration parameters:
  `batch_size`, `epochs`, `shuffle`, `drop_last`, `num_workers`,
  `prefetch_factor`, `log_interval`, `validation_freq`, `seed`,
  `grad_accum_steps`, checkpoint flags, and early stopping.
- `DatasetBatcher` builds split indices, optionally shuffles them, then reads
  batches sequentially.
- `ArrowDatasetBatcher` builds train/val/test index lists by split ratio or by
  `__partition__`, optionally shuffles, then reads batches sequentially.
- `TextDatasetBatcher` tokenizes text, adds a partition column, and delegates
  batching to `ArrowDatasetBatcher`.

Resolved in this pass:

- `balance_classes`, `balance_mode`, `balance_target`, and `balance_seed` are
  DataLoader parameters.
- Arrow/text train batchers support deterministic oversampling,
  undersampling, and weighted sampling while validation/test remain unchanged.
- Label-frequency scans and original/effective class-distribution logs are
  attached to the train split.
- Studio users configure imbalance settings from the DataLoader Open Dialog.
- The DataLoader Properties panel is summary-only and hands users to Open
  Dialog for complex settings.

Still pending / explicitly out of scope:

- Parquet/time-series/unsupported dataset paths remain conservative and do not
  silently invent balancing behavior where labels or partition semantics are
  not available.

Affected files:

- `cyxwiz-engine/src/core/graph_compiler.h`
- `cyxwiz-engine/src/core/graph_compiler.cpp`
- `cyxwiz-engine/src/core/training_batcher_setup.cpp`
- `cyxwiz-engine/src/core/dataset_batcher.cpp`
- `cyxwiz-engine/src/core/dataset_batcher.h`
- `cyxwiz-engine/src/core/arrow_dataset_batcher.cpp`
- `cyxwiz-engine/src/core/dataset_batcher.h`
- `cyxwiz-engine/src/core/text_dataset_batcher.cpp`
- `cyxwiz-engine/src/core/parquet_arrow_batcher.cpp`
- `cyxwiz-engine/src/gui/node_config_dialog.cpp`
- `cyxwiz-engine/src/gui/data_io_dialogs.cpp`
- `cyxwiz-engine/src/gui/properties_node_editors.cpp`
- `cyxwiz-engine/src/core/node_metadata_registry.cpp`

### DataSplit Stratification

Current state:

- Some example graphs set `DataSplit` parameter `stratified=true`.
- The compiler reads `train_ratio`, `val_ratio`, `test_ratio`, and `seed`.
- The compiler does not read or propagate `stratified`.
- `TextDatasetBatcher::AddSplitPartitionColumn` shuffles all rows globally and
  slices by ratio. It does not preserve class proportions.

Resolved in this pass:

- `TrainingConfiguration` has `stratified`.
- `GraphCompiler` extracts `DataSplit.stratified`.
- Arrow/text-backed datasets use label-aware stratified partition assignment.
- Unsupported/invalid stratification paths produce diagnostics instead of
  silent ignores.
- Tests cover stratified train/val/test behavior.

Affected files:

- `cyxwiz-engine/src/core/graph_compiler.h`
- `cyxwiz-engine/src/core/graph_compiler.cpp`
- `cyxwiz-engine/src/core/text_dataset_batcher.cpp`
- `cyxwiz-engine/src/core/arrow_dataset_batcher.cpp`
- `cyxwiz-engine/src/core/parquet_arrow_batcher.cpp`
- `examples/cyxgraph/Sentiment analysis/sentiment_analysis_gru_classifier.cyxgraph`

### Tree Classifier Runtime

Current state:

- `DecisionTreeClassifier`, `RandomForestClassifier`, and
  `GradientBoostingClassifier` node enum values exist.
- The Studio node metadata registers these nodes.
- `DecisionTreeClassifier` is implemented as a PipelineOperatorFactory-backed
  table-path classifier. It reads numeric feature columns plus `target_col`,
  fits a deterministic binary CART-style tree, and appends a configurable
  prediction column.
- `RandomForestClassifier` is implemented as a PipelineOperatorFactory-backed
  table-path ensemble. It reuses the tree training path, trains deterministic
  bootstrap feature-subset trees, majority-votes predictions, and appends a
  configurable prediction column.
- `GradientBoostingClassifier` is implemented as a
  PipelineOperatorFactory-backed table-path classifier. It trains deterministic
  one-vs-rest boosted regression trees and appends a configurable prediction
  column.
- `DecisionTreeClassifier` exposes GUI metadata for `target_col`,
  `feature_cols`, `prediction_col`, `max_depth`, `min_samples_split`,
  `min_samples_leaf`, and `criterion`.
- `RandomForestClassifier` exposes GUI metadata for `target_col`,
  `feature_cols`, `prediction_col`, `n_estimators`, `max_depth`,
  `min_samples_split`, `min_samples_leaf`, `criterion`, `max_features`, and
  `seed`.
- `GradientBoostingClassifier` exposes GUI metadata for `target_col`,
  `feature_cols`, `prediction_col`, `n_estimators`, `learning_rate`,
  `max_depth`, `min_samples_split`, and `min_samples_leaf`.
- `DecisionTreeClassifier`, `RandomForestClassifier`, and
  `GradientBoostingClassifier` are assigned to the `classic_ml` workflow lane.
- Runtime capability now treats `DecisionTreeClassifier`,
  `RandomForestClassifier`, and `GradientBoostingClassifier` as
  operator-backed.

Still pending:

- No saved tree model artifact or cross-session model serialization yet.
- No separate training-graph compiler path that exports persisted
  DecisionTree/RandomForest/GradientBoosting model artifacts; the implemented
  route is in-pipeline fit-and-predict over a table.

Affected files:

- `cyxwiz-engine/CMakeLists.txt`
- `cyxwiz-engine/src/gui/node_editor.h`
- `cyxwiz-engine/src/gui/node_editor_nodes.cpp`
- `cyxwiz-engine/src/core/node_executors/decision_tree_model.h`
- `cyxwiz-engine/src/core/node_executors/decision_tree_model.cpp`
- `cyxwiz-engine/src/core/node_executors/decision_tree_trainer.h`
- `cyxwiz-engine/src/core/node_executors/decision_tree_trainer.cpp`
- `cyxwiz-engine/src/core/node_executors/decision_tree_operator.h`
- `cyxwiz-engine/src/core/node_executors/decision_tree_operator.cpp`
- `cyxwiz-engine/src/core/node_executors/tree_classification_utils.h`
- `cyxwiz-engine/src/core/node_executors/tree_classification_utils.cpp`
- `cyxwiz-engine/src/core/node_executors/random_forest_model.h`
- `cyxwiz-engine/src/core/node_executors/random_forest_model.cpp`
- `cyxwiz-engine/src/core/node_executors/random_forest_trainer.h`
- `cyxwiz-engine/src/core/node_executors/random_forest_trainer.cpp`
- `cyxwiz-engine/src/core/node_executors/random_forest_operator.h`
- `cyxwiz-engine/src/core/node_executors/random_forest_operator.cpp`
- `cyxwiz-engine/src/core/node_executors/gradient_boosting_model.h`
- `cyxwiz-engine/src/core/node_executors/gradient_boosting_model.cpp`
- `cyxwiz-engine/src/core/node_executors/gradient_boosting_trainer.h`
- `cyxwiz-engine/src/core/node_executors/gradient_boosting_trainer.cpp`
- `cyxwiz-engine/src/core/node_executors/gradient_boosting_operator.h`
- `cyxwiz-engine/src/core/node_executors/gradient_boosting_operator.cpp`
- `cyxwiz-engine/src/core/node_executors/pipeline_operator_factory.cpp`
- `cyxwiz-engine/src/core/node_metadata_registry.cpp`
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp`
- `cyxwiz-engine/tests/test_decision_tree_operator.cpp`
- `cyxwiz-engine/tests/test_random_forest_operator.cpp`
- `cyxwiz-engine/tests/test_gradient_boosting_operator.cpp`
- `cyxwiz-engine/tests/test_pipeline_executor_operator_routing.cpp`
- `cyxwiz-engine/tests/test_pipeline_operator_metadata.cpp`

### Studio Node Search And Loss Catalog Drift

Current state:

- Studio has multiple add-node entry points:
  right-click canvas context menu,
  top-right canvas node-add search,
  and the Nodes panel search.
- These surfaces are already intended to share one source:
  `NodeMetadataRegistry`.
- `NodeEditor::InitializeSearchableNodes()` builds the canvas search list from
  `NodeMetadataRegistry`.
- `NodeBrowserPanel` uses `NodeMetadataRegistry::Search()` and
  `GetByCategory()`.
- The older hardcoded context-menu list was removed/commented in favor of the
  registry-driven path.
- Loss node enum values exist for `MSELoss`, `CrossEntropyLoss`, `BCELoss`,
  `BCEWithLogits`, `L1Loss`, `SmoothL1Loss`, `HuberLoss`, and `NLLLoss`.
- `NodeEditor::CreateNode` / node pin setup supports the additional loss nodes.
- Saved graph I/O maps these loss names to node types.
- `GraphCompiler` and `ModelBuilder` support these loss types for training.

Resolved catalog gap:

- `NodeMetadataRegistry::InitializeTrainingNodes()` registers the supported
  graph-training loss nodes, including `BCELoss`, `BCEWithLogits`, `L1Loss`,
  `SmoothL1Loss`, `HuberLoss`, `NLLLoss`, and `FocalLoss`.
- Visible Studio add-node searches use `NodeMetadataRegistry`, so supported
  loss nodes are discoverable from the shared catalog.

Resolved UX naming gap:

- Training losses and optimizers are under `NodeCategory::Training`, not an
  `Optimization` category.
- `InitializeOptimizationNodes()` is for numerical optimization/tool nodes
  such as gradient-descent visualization, convexity analysis, LP/QP solver,
  differentiation, and integration. It is not where training loss functions or
  neural-network optimizers live.
- Search aliases/keywords now make optimization/objective/criterion queries
  land on training losses and optimizers.

Affected files:

- `cyxwiz-engine/src/core/node_metadata_registry.cpp`
- `cyxwiz-engine/src/core/node_metadata_registry.h`
- `cyxwiz-engine/src/gui/node_editor_add_search.cpp`
- `cyxwiz-engine/src/gui/node_editor_context_menu.cpp`
- `cyxwiz-engine/src/gui/panels/node_browser_panel.cpp`
- `cyxwiz-engine/src/gui/node_editor_nodes.cpp`
- `cyxwiz-engine/src/gui/node_editor_io.cpp`

## Example Impact

The sentiment graph at
`examples/cyxgraph/Sentiment analysis/sentiment_analysis_gru_classifier.cyxgraph`
uses a multi-class CrossEntropy training path over the `status` label column.
For an imbalanced sentiment dataset after this pass:

- `CrossEntropyLoss` can use manual or supported balanced class weights.
- `BCEWithLogits` can use positive-class weights.
- `DataLoader` can rebalance Arrow/text train batches.
- `DataSplit.stratified=true` is honored for supported Arrow/text splits.
- `FocalLoss` can be selected as a supported visual graph loss.

Remaining caveat: `DecisionTreeClassifier`, `RandomForestClassifier`, and
`GradientBoostingClassifier` are executable as table-path
PipelineOperatorFactory nodes, but they are not yet persisted model artifacts.

## Recommended Implementation Order

### Phase 1 - Compiler Truth And Diagnostics

Goal: stop silently accepting imbalance-related parameters that are not wired.

Tasks:

- [x] Add `stratified`, `balance_classes`, `balance_mode`,
  `balance_target`, and loss-weight parameters to the relevant metadata only
  when implementation is ready, or explicitly warn when present but ignored.
- [x] Extend compile diagnostics to report ignored `DataSplit.stratified`.
- [x] Extend compile diagnostics to report unsupported `DataLoader`
  class-balancing parameters if users hand-edit graph JSON.
- [x] Extend compile diagnostics to report unsupported loss parameters such as
  `weight`, `class_weight`, `sample_weight`, and `pos_weight`.

Acceptance:

- A graph containing imbalance-related parameters either uses them or emits a
  clear warning/error saying they are not implemented.
- No example graph should imply stratification or balancing unless the runtime
  honors it.

### Phase 1B - Studio Loss Catalog Unification

Goal: make every compiler-supported loss discoverable from every Studio
add-node surface.

Tasks:

- [x] Register `BCELoss`, `BCEWithLogits`, `L1Loss`, `SmoothL1Loss`,
  `HuberLoss`, and `NLLLoss` in `NodeMetadataRegistry::InitializeTrainingNodes`.
- [x] Add search keywords/aliases:
  `binary`, `binary cross entropy`, `bce`, `logits`, `mae`, `l1`,
  `huber`, `smooth l1`, `nll`, `negative log likelihood`,
  `criterion`, `objective`, `optimization`, and `loss`.
- [x] Confirm right-click canvas search, top-right canvas search, and Nodes
  panel search all pull from `NodeMetadataRegistry`.
- [x] Make category naming clear: either keep `Training` and add aliases, or
  rename/display it as `Training / Optimization`.
- [x] Ensure blocked/template handling is consistent across the right-click
  menu, top-right search, and Nodes panel.

Acceptance:

- Searching `bce` shows `BCE Loss` and `BCE with Logits`.
- Searching `binary` shows binary classification loss nodes.
- Searching `optimization` shows training optimizers/losses or points users to
  the Training category.
- A loss node supported by `GraphCompiler` is discoverable in all add-node
  surfaces unless deliberately hidden with a documented reason.

### Phase 2 - Stratified DataSplit

Goal: preserve label distribution across train/val/test splits when requested.

Tasks:

- [x] Add `bool stratified` to `TrainingConfiguration`.
- [x] Parse `DataSplit.stratified` in `GraphCompiler`.
- [x] Add label-aware partition assignment for Arrow-backed tabular/text
  datasets.
- [x] Keep deterministic behavior under `split_seed` or `dataloader_seed`.
- [x] Add class-count logging for train/val/test partitions.
- [x] Reject stratified split when no label column is available.

Acceptance:

- With `stratified=true`, each split preserves class proportions within a
  documented tolerance.
- With `stratified=false`, current random split behavior remains available.

### Phase 3 - Balanced Training Sampler

Goal: rebalance only the training batches without altering validation/test
distributions.

Tasks:

- [x] Add DataLoader options:
  `balance_classes=false|true`,
  `balance_mode=none|oversample|undersample|weighted_sampler`,
  `balance_target=max|median|min|number`,
  and `balance_seed`.
- [x] Build label-frequency maps from the train split.
- [x] Implement deterministic oversampling and undersampling index generation.
- [x] Implement weighted random sampling with replacement if selected.
- [x] Ensure validation and test batchers never rebalance by default.
- [x] Report original and effective class distributions in logs.
- [x] Add Studio GUI controls in the DataLoader Open Dialog for imbalance
  settings: `balance_classes`, `balance_mode`, `balance_target`, and
  `balance_seed`.
- [x] Validate GUI-entered balancing options and write them into
  `node.parameters` so saved graphs round-trip without manual JSON edits.
- [x] Keep the DataLoader Properties panel summary-only, with users directed
  to the Open Dialog for editing.
- [x] Add compile coverage proving GUI-created DataLoader imbalance parameters
  reach `TrainingConfiguration`.

Acceptance:

- Training split can be balanced per epoch while validation/test remain
  untouched.
- Balanced sampling is deterministic with a fixed seed.
- Small minority classes are handled without empty-batch or divide-by-zero
  behavior.
- Studio users can enable/disable DataLoader balancing, choose the balancing
  mode/target, and set the balancing seed from the DataLoader Open Dialog.
- A graph saved after using the GUI reloads with the same DataLoader imbalance
  controls and compiles to the same training configuration.

### Phase 4 - Weighted Loss API

Goal: support standard class-weighted loss behavior for classification.

Tasks:

- [x] Extend `CrossEntropyLoss` to accept optional per-class weights.
- [x] Apply class weights in both forward and backward passes.
- [x] Define reduction semantics precisely for weighted mean.
- [x] Extend `BCEWithLogitsLoss` to support optional `pos_weight`.
- [x] Decide whether `BCELoss` should support element weights or remain
  unweighted.
- [x] Add backend tests for weighted forward and backward behavior.

Acceptance:

- Weighted CrossEntropy changes both loss value and gradients as expected.
- `pos_weight` affects positive BCEWithLogits examples as expected.
- Existing unweighted loss behavior remains unchanged.
- `BCELoss` remains unweighted; use `BCEWithLogits.pos_weight`,
  CrossEntropy class weights, or DataLoader balancing for imbalance handling.

### Phase 5 - Graph-Configured Loss Weights

Goal: expose weighted losses safely through Studio/CyxGraph.

Tasks:

- [x] Add loss-node metadata for supported weight parameters.
- [x] Add actual properties-panel editing in `properties_node_editors.cpp` for
  simple controllable loss parameters, not a separate Open Dialog:
  `CrossEntropyLoss.reduction`, `CrossEntropyLoss.ignore_index`,
  `CrossEntropyLoss.class_weight`, `CrossEntropyLoss.class_weights`,
  `BCEWithLogits.reduction`, `BCEWithLogits.pos_weight`, and common
  `reduction` controls for supported standard losses.
- [x] Add validation UI for manual class-weight vectors, including expected
  class count and parse errors.
- [x] Parse weights from graph params into `TrainingConfiguration.loss_params`.
- [x] Construct weighted loss objects in `model_builder.cpp`.
- [x] Wire parsed `reduction` values into runtime loss construction instead of
  defaulting every constructed loss to mean reduction.
- [x] Add optional auto-compute class weights from the training split:
  `class_weight=none|balanced|manual`.
- [x] Make manual weights validate against `output_size` / `num_classes`.

Acceptance:

- A visual `CrossEntropyLoss` node can use manual or balanced class weights.
- Selecting a supported loss node exposes its controllable properties in the
  Studio Properties panel.
- Edited loss properties persist in `node.parameters`, saved graph JSON, and
  the constructed runtime loss object.
- Invalid weight vector length fails compile with a clear diagnostic.
- Saved graph JSON round-trips the weight configuration.
- Users do not need to hand-edit `.cyxgraph` JSON to configure supported loss
  properties.

### Phase 6 - Graph-Exposed FocalLoss

Goal: make the existing backend focal loss usable in normal graph training.

Tasks:

- [x] Add `FocalLoss` to `NodeType` if no existing enum value is present.
- [x] Register `FocalLoss` metadata and node pins.
- [x] Add `FocalLoss` to graph compiler loss-node detection.
- [x] Parse `alpha` and `gamma`.
- [x] Build `FocalLoss(alpha, gamma)` in `model_builder.cpp`.
- [x] Add GUI property editing for `alpha`, `gamma`, and `reduction`.
- [x] Add graph compiler and training smoke coverage.

Acceptance:

- A saved graph can train with `FocalLoss`.
- `alpha`, `gamma`, and `reduction` are controllable from the Properties panel
  and honored by the constructed backend loss.
- Existing CrossEntropy graphs are unaffected.

### Phase 7 - Tree Classifier Runtime

Goal: make the tree classifier family real table-path runtime nodes without
pretending they are persisted model artifacts.

Decision:

- Implement `DecisionTreeClassifier` as a modular Cat-1 table operator with
  separate model, trainer, and operator translation units.
- Implement `RandomForestClassifier` as a modular Cat-1 table operator that
  reuses the native tree model/trainer and keeps ensemble code in separate
  model, trainer, and operator translation units.
- Implement `GradientBoostingClassifier` as a modular Cat-1 table operator
  with separate boosted model, trainer, and operator translation units.
- Keep model persistence/export as a separate future phase for all tree
  classifiers.

Tasks:

- [x] Pick implementation route: table-path fit-and-predict operator.
- [x] Confirm there is no existing native tree model or backend wrapper to
  wire safely.
- [x] Add a native deterministic decision-tree model representation.
- [x] Add a native trainer with `gini` and `entropy` split criteria.
- [x] Add a `DecisionTreeClassifier` operator that reads numeric features,
  learns from `target_col`, and appends `prediction_col`.
- [x] Register the operator in `PipelineOperatorFactory`.
- [x] Move `DecisionTreeClassifier` from fail-closed runtime support to
  operator-backed runtime support.
- [x] Expose controllable classifier properties in GUI metadata.
- [x] Add standalone fit/predict tests and routed pipeline tests.
- [x] Add shared classification-label/prediction helpers so tree-family
  operators do not duplicate target parsing and output writing.
- [x] Add a native deterministic random-forest model representation.
- [x] Add a native random-forest trainer with bootstrap row sampling,
  deterministic seeds, and `sqrt` / `log2` / `all` feature subsets.
- [x] Add a `RandomForestClassifier` operator that reads numeric features,
  learns from `target_col`, majority-votes, and appends `prediction_col`.
- [x] Register the random-forest operator in `PipelineOperatorFactory`.
- [x] Move `RandomForestClassifier` from fail-closed runtime support to
  operator-backed runtime support.
- [x] Expose controllable random-forest properties in GUI metadata.
- [x] Add standalone random-forest fit/predict tests and routed pipeline tests.
- [x] Add a native deterministic gradient-boosting model representation.
- [x] Add a native one-vs-rest boosted regression-tree trainer.
- [x] Add a `GradientBoostingClassifier` operator that reads numeric features,
  learns from `target_col`, and appends `prediction_col`.
- [x] Register the gradient-boosting operator in `PipelineOperatorFactory`.
- [x] Move `GradientBoostingClassifier` from fail-closed runtime support to
  operator-backed runtime support.
- [x] Expose controllable gradient-boosting properties in GUI metadata.
- [x] Add standalone gradient-boosting fit/predict tests and routed pipeline
  tests.

Acceptance:

- `DecisionTreeClassifier` is executable through PipelineExecutor and appends
  a prediction column for numeric or string labels.
- `RandomForestClassifier` is executable through PipelineExecutor and appends
  a prediction column for numeric or string labels.
- `GradientBoostingClassifier` is executable through PipelineExecutor and
  appends a prediction column for numeric or string labels.
- `DecisionTreeClassifier`, `RandomForestClassifier`, and
  `GradientBoostingClassifier` metadata are implemented, operator-backed, and
  expose controllable properties in the Properties panel.

## Non-Goals

- Changing sentiment model architecture as a substitute for class balancing.
- Replacing `TrainingExecutor` or `GraphCompiler`.
- Applying balancing to validation/test data by default.
- Pretending tree nodes are supported through pass-through behavior.
- Using weighted F1 metrics as a replacement for weighted training loss.

## Verification Targets

Recommended coverage when implementing from this document:

- `test_graph_compiler_deferred_nodes`
- `test_text_arrow_training_launch`
- `test_text_dataset_batcher_arrow`
- `test_training_batcher_setup`
- `test_training_executor_arrow_parquet`
- New weighted CrossEntropy backend tests
- New BCEWithLogits `pos_weight` backend tests
- New stratified split tests
- New balanced sampler tests for Arrow/text datasets
- New DataLoader GUI imbalance-control regression or compile test
- New graph compile tests for ignored or unsupported imbalance params
- New loss Properties panel regression for editable loss parameters
- New `FocalLoss` graph training smoke if Phase 6 is implemented
- New decision-tree fit/predict tests if Phase 7 is implemented
- New random-forest fit/predict tests if Phase 7 is implemented
- New gradient-boosting fit/predict tests if Phase 7 is implemented
- `cyxwiz-engine` Debug build
- `git diff --check`
