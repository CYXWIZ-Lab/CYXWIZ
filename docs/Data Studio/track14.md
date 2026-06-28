# Track 14 - NER And Metric-Learning Execution Plan

**Created:** 2026-06-25
**Source:** Working plan for archived `done14.md`.

## Purpose

Track the remaining implementation work from `done14.md` against the current
codebase truth.

`done14.md` started as a design-gap document. Since then, several NER
sequence-tagging primitives have landed. This tracker separates what is now
real from what remains target design, so the next work does not reimplement
existing contracts or claim unsupported graph behavior.

## Operating Rules

- Do not add a second training graph abstraction. Use `GraphCompiler`,
  `TrainingConfiguration`, `ModelBuilder`, and `TrainingExecutor`.
- Do not overload the single-input `Batch` contract for sequence or
  metric-learning tasks. Use typed batch contracts.
- Do not mark a node fully supported in metadata until graph compile, launch,
  runtime execution, and tests prove the path.
- Do not encode target-design NER or Siamese nodes as fake `Dense` nodes.
- Keep `CompiledGraphPlan` and Track 22 runtime-owner rules intact.
- Prefer one narrow executable slice over broad node-surface expansion.

## Current Truth

### NER / Sequence Tagging

Implemented:

- `SequenceBatch` and `ISequenceBatcher` exist beside the old single-input
  `Batch` contract.
- `SequenceBatcher` pads word IDs, optional POS IDs, tag IDs, causal targets,
  and attention masks.
- `SequenceVocabulary` builds token, POS, and tag vocabularies.
- `NERSequenceBuilder` builds encoded sequence samples from sentence rows.
- `BuildSequenceBatcherFromArrowDataset` bridges Arrow sequence rows to an
  `ISequenceBatcher`.
- `GraphCompiler` captures `SequenceBatchConfig` from first-class
  `NERSequenceBuilder` nodes.
- `TimeDistributedDenseModule` exists and `ModelBuilder` builds it from
  `TimeDistributed`.
- `TrainingExecutor` has a `SequenceExternal` mode and can train token-tagging
  batches through `StartTrainingSequence`.
- `CrossEntropyLoss` uses sequence `ignore_index` for token-level targets.
- Sequence tag metrics and sequence inference decode helpers exist.
- Saved NER graph assets exist under `examples/cyxgraph/NER`.
- Explicit sequence feature fusion exists for token taggers through
  `SequenceFeatureFusionModule` and opt-in `sequence_feature_fusion=true`
  configs. It combines word IDs and POS IDs without adding a generic
  multi-input graph executor.

Known limitations:

- `FeatureConcat` is not a first-class runtime graph node.
- Attention masks are carried by the batch contract but not consumed by the
  model path.
- `SequenceTagOutput` is now a first-class Studio output node for
  token-level tagger logits and BIO decode metadata.
- Generic tabular `TrainingExecutor` dispatch still correctly fails closed for
  sequence graphs; Studio graph launch must build the `ISequenceBatcher`
  bridge before calling `StartTrainingSequence`.

### Siamese / Metric Learning

Implemented:

- Backend losses exist and are tested:
  `CosineEmbeddingLoss`, `TripletLoss`, and `ContrastiveLoss`.
- `GraphCompiler` rejects metric-learning/Siamese sketches in the selected
  training path instead of pretending the single-input runtime can train them.

Known limitations:

- No typed pair/triplet batch contract.
- No shared encoder ownership model.
- No branch-aware backward pass through shared weights.
- No first-class graph nodes for pair/triplet builders, shared encoders,
  metric-learning losses, pair metrics, retrieval metrics, or embedding
  output.

## Phase 0 - Refresh The Truth

Status: completed 2026-06-25.

Goal: align `done14.md` with the current implementation state.

Tasks:

- [x] Mark implemented NER primitives as current truth.
- [x] Mark stale NER missing-node bullets as completed, partial, or still
  missing.
- [x] Record that Siamese remains backend-loss-only plus compiler guardrails.
- [x] Keep the old tensor exposure progress as closed history.

Acceptance:

- `done14.md` no longer implies `NERSequenceBuilder`, sequence batches,
  TimeDistributed, or sequence metrics are entirely missing.
- Remaining work points to concrete code gaps.

## Phase 1 - NER End-To-End Smoke

Status: completed 2026-06-26 by
`test_saved_ner_sequence_smoke`.

Goal: prove the smallest real NER training path works from saved graph assets
through sequence batcher construction, sequence training, asset packaging, and
decode.

Target scope:

- Load `examples/cyxgraph/NER/ner_bilstm_sequence_tagger.cyxgraph`.
- Load `examples/cyxgraph/NER/generated/ner_sentences.csv`.
- Compile the graph and assert `SequenceBatchConfig` is populated.
- Build the Arrow-backed sequence batcher.
- Normalize embedding vocabulary size and TimeDistributed tag width from the
  sequence build result.
- Train one tiny deterministic epoch or bounded smoke pass.
- Assert sequence metrics are finite and token counts are non-zero.
- Package sequence vocabulary assets into `.cyxmodel`.
- Decode sample sequence logits through the existing inference decode helper.

Non-goals:

- Generic `FeatureConcat` graph fan-in execution.
- Full production NER accuracy.
- New UI node types.
- Real `SequenceTagOutput` Studio node.
- Rewriting generic `TrainingExecutor` dispatch.

Acceptance:

- [x] One focused test target covers saved NER graph compile -> sequence batcher ->
  tiny training -> sequence asset/decode proof.
- [x] The test does not depend on the external full NER dataset path.
- [x] Existing guardrails still reject fake Dense-encoded NER placeholders.

Notes:

- The saved graph still reports the current `FeatureConcat` limitation during
  compile. The Phase 2 implementation adds explicit sequence feature fusion
  for the training config instead of broadening `FeatureConcat` into generic
  graph fan-in execution.

## Phase 2 - Sequence Feature Fusion

Status: completed 2026-06-26 by `SequenceFeatureFusionModule` and the
POS-fused `test_saved_ner_sequence_smoke` path.

Goal: consume optional POS IDs without broadening the runtime into an arbitrary
multi-input graph executor.

Possible smallest design:

- Add a focused sequence feature module that combines word embedding and POS
  embedding for token-tagging graphs.
- Keep the module explicit and sequence-only.
- Validate matching `[batch, seq]` payloads before training starts.

Acceptance:

- [x] POS IDs affect the model path only when the graph declares POS usage.
- [x] Shape mismatches fail before training.
- [x] Existing word-only NER path remains supported.

Notes:

- `BuildSequenceModelInput` packs `[batch, seq]` word/POS ID tensors into
  `[batch, seq, 2]` only when the first layer declares
  `sequence_feature_fusion=true`.
- `SequenceFeatureFusionModule` owns separate word/POS embeddings and emits
  `[batch, seq, word_dim + pos_dim]` for the normal sequential model path.
- The saved NER smoke asserts missing/mismatched POS IDs fail before forward
  and that changing POS IDs changes model logits.

## Phase 3 - Sequence Output Surface

Status: completed 2026-06-26.

Goal: make sequence-tagging inference/output visible as a first-class Studio
contract after the runtime path is proven.

Tasks:

- [x] Decide whether `SequenceTagOutput` should be a real node or an
  export/infer option on Output.
- [x] Persist tag vocabulary and max-length metadata in exported models.
- [x] Ensure local inference can return token/tag pairs for sequence-tagging
  models.
- [x] Reuse `ModelBuilder` when importing packaged graph models so deployed
  inference understands the same sequence modules as training.
- [x] Pack local inference `word_ids` + `pos_ids` for POS-fused sequence
  models.
- [x] Add tests for invalid/missing sequence vocab assets.
- [x] Add first-class `SequenceTagOutput` Studio node metadata, creation,
  save/load, pattern, CyxQL, shape, properties, icon, codegen, and compiler
  terminal handling.
- [x] Update the saved NER graph to use `SequenceTagOutput` instead of a
  generic `Output` node.
- [x] Add compiler/template tests proving `SequenceTagOutput.num_tags` drives
  token-class CrossEntropy validation and that the saved graph uses the
  first-class node type.

Acceptance:

- A deployed sequence tagger can return readable token/tag predictions.
- Missing sequence metadata fails with a clear error.

Notes:

- `LocalInferenceServer` now preserves `input.pos_ids` and packs named
  sequence inputs for models whose first module is
  `SequenceFeatureFusionModule`.
- `ModelImporter` now delegates graph architecture rebuilds to
  `BuildSequentialFromConfig` instead of maintaining a stale duplicate layer
  builder.
- Local model loading now fails closed when a package declares sequence tag
  vocabulary metadata but the tag vocabulary is missing or empty.
- `SequenceTagOutput` is implemented as a terminal graph/output contract, not
  a trainable layer or PipelineExecutor table operator.
- The saved NER graph now serializes node 18 as `SequenceTagOutput`.

## Phase 4 - Siamese Backend Smoke Only

Status: completed 2026-06-26 by `test_siamese_backend_smoke`.

Goal: prove the backend metric-learning pieces can support a minimal manual
training loop before adding visual graph nodes.

Target scope:

- Small code or Python example with a manually shared encoder.
- Toy pair or triplet dataset.
- One metric-learning loss.
- Proof that similar/dissimilar or anchor/positive/negative distances move in
  the expected direction.

Non-goals:

- Studio graph node support.
- Shared encoder visual semantics.
- Pair/triplet batchers in the main training runtime.

Acceptance:

- [x] Backend example/test proves the losses are useful in a controlled training
  loop.
- [x] GraphCompiler continues to reject visual Siamese sketches.

Notes:

- `test_siamese_backend_smoke` trains a tiny manually shared linear encoder
  with `ContrastiveLoss`, proving positive pairs stay close while negative
  pairs move out to the margin.
- This does not add Studio graph nodes, pair/triplet batchers, or shared
  encoder graph semantics.

## Phase 5 - Siamese Graph Runtime Design

Status: completed 2026-06-26 as a design contract only.

Goal: design the first visual graph contract for metric learning only after the
backend smoke path is proven.

Required contracts:

- typed `PairBatch` / `TripletBatch`,
- shared encoder ownership,
- branch-aware gradient accumulation,
- pair/triplet loss wiring,
- pair and retrieval metrics,
- embedding or pair-score inference outputs.

Runtime contract:

- `PairBatch` and `TripletBatch` are new typed training payloads beside
  `Batch` and `SequenceBatch`; metric-learning launch must not pack multiple
  samples into `Batch.data`.
- `PairBatch` carries named tensors for `input_a`, `input_b`, `pair_label`,
  and optional `sample_id_a`, `sample_id_b`, `class_id_a`, and `class_id_b`.
- `TripletBatch` carries named tensors for `anchor`, `positive`, `negative`,
  plus optional sample/class IDs for retrieval metrics and export.
- `ContrastiveLoss` labels use the backend convention proven by
  `test_siamese_backend_smoke`: `0 = similar`, `1 = dissimilar`.
- `CosineEmbeddingLoss` labels remain `1 = similar`, `-1 = dissimilar`; graph
  metadata must show this explicitly because it differs from contrastive loss.
- `TripletLoss` consumes anchor/positive/negative embeddings directly and does
  not require pair labels.

Shared encoder contract:

- `SharedEncoder` owns exactly one encoder module/parameter set.
- Branch nodes reference the shared encoder by ID/name; they do not clone the
  encoder layers.
- The executor runs the same encoder object for each branch, accumulates
  branch gradients into the same parameters, and lets the optimizer update
  those parameters once per training step.
- The compiler rejects duplicated branch encoders, mismatched embedding
  dimensions, missing labels/negative samples, ambiguous label conventions,
  and optimizers that cannot reach the metric loss.

Inference contract:

- `EmbeddingOutput` is the single-sample export path and returns embeddings
  with optional sample/class metadata.
- `PairScoreOutput` is the pair-inference path and returns distance or
  similarity scores; it must not masquerade as class probabilities.
- Pair metrics are distance-threshold based first: pair accuracy, positive and
  negative distance means, and optional ROC/AUC.
- Retrieval metrics are embedding-space based: recall@k, mean reciprocal rank,
  and nearest-neighbor class agreement.

Implementation slices:

1. Add internal `PairBatch` / `TripletBatch` contracts and shape/label tests.
   Completed 2026-06-26 by `metric_learning_batch.h` and
   `test_metric_learning_batch_contracts`.
2. Add pair/triplet dataset builders and batchers without visual graph runtime
   execution. Completed 2026-06-26 by `PairDatasetBuilder`,
   `TripletDatasetBuilder`, `PairBatcher`, and `TripletBatcher`.
3. Add Studio node metadata and compiler guards for metric-learning nodes,
   marked unsupported until the executor path exists. Completed 2026-06-26
   for blocked `Template` metadata, save/load aliases, pattern/CyxQL/icon
   coverage, default pins, and type-based compiler rejection.
4. Add shared-encoder compile representation and a runtime harness that reuses
   one encoder object across branches. Runtime harness completed 2026-06-26
   by `SharedEncoderRuntime` and
   `test_metric_learning_shared_encoder_contracts`; passive compiler graph
   representation completed 2026-06-26 by `MetricLearningGraphContract` and
   `test_graph_compiler_deferred_nodes`. Executable compiler/runtime support
   remains open. Deterministic `SequentialModel` pair/triplet backward is now
   supported by branch replay plus accumulated gradient maps; training-mode
   stateful modules still require activation snapshots.
5. Wire `ContrastiveLoss`, `CosineEmbeddingLoss`, and `TripletLoss` through
   the graph executor. Internal loss adapter contracts completed 2026-06-26
   by `metric_learning_losses.h` and `test_metric_learning_losses`; internal
   training-step contracts completed 2026-06-26 by
   `metric_learning_training_step.h` and
   `test_metric_learning_training_step`. Visual graph executor routing remains
   open.
6. Add `PairMetrics`, `RetrievalMetrics`, `EmbeddingOutput`, and
   `PairScoreOutput` after training is executable. Internal pair/retrieval
   metric helpers completed 2026-06-26 by `metric_learning_metrics.h` and
   `test_metric_learning_metrics`; internal inference response contracts
   completed 2026-06-26 by `metric_learning_inference_outputs.h` and
   `test_metric_learning_inference_outputs`, including JSON packaging and
   pair-score mode parsing. Local-inference input parsing contracts completed
   2026-06-26 by `metric_learning_inference_input.h/.cpp` and
   `test_metric_learning_inference_input`. Embedded local inference routes
   completed 2026-06-26 by `/v1/embeddings` and `/v1/pair-score` handlers.
   Graph nodes and visual runtime wiring remain open.

Acceptance:

- [x] A graph-level design can be implemented without overloading `Batch.data`
  or duplicating encoder weights.
- [x] The next implementation slice is narrowed to typed batch contracts and
  tests, not broad Studio node support.
- [x] The first internal contract slice has typed pair/triplet payloads,
  validation helpers, label-convention helpers, and focused tests.
- [x] In-memory pair/triplet sample batchers can emit aligned tensors, preserve
  optional sample/class metadata IDs, reject invalid labels/shapes, and handle
  partial/drop-last batches.
- [x] Internal pair/triplet dataset builders can validate row shape/metadata
  contracts, derive pair labels from class IDs when configured, validate triplet
  class semantics, and create the in-memory batchers.
- [x] Internal shared-encoder runtime harness owns exactly one executable
  encoder, routes pair/triplet branches through that object, accumulates branch
  gradients into the same encoder, and performs one parameter update per batch.
- [x] Internal pair/retrieval metric helpers compute distance-threshold pair
  accuracy, positive/negative distance means, recall@k, mean reciprocal rank,
  and nearest-neighbor class agreement without enabling graph execution.
- [x] Internal embedding and pair-score response helpers preserve embedding
  shape/metadata and expose distance or similarity scores without
  masquerading as class probabilities.
- [x] Metric-learning inference output helpers serialize stable JSON payloads
  for embedding records and pair-score records, with explicit score-mode
  parsing.
- [x] Metric-learning local-inference input helpers parse embedding and
  pair-score request payloads into typed tensors, validate branch shape
  agreement, and preserve batch-aligned sample/class metadata without adding
  HTTP routes.
- [x] Embedded local inference exposes `/v1/embeddings` and `/v1/pair-score`
  routes that run a loaded embedding model and serialize the metric-learning
  response contracts without accepting visual Siamese graph execution.
- [x] Internal metric-loss adapters validate pair/triplet contracts and return
  branch gradients for contrastive, cosine-embedding, and Euclidean triplet
  loss helpers without enabling graph execution.
- [x] Internal metric-learning training-step helpers route typed pair/triplet
  batches through shared encoder forward, metric loss, branch backward, and an
  optional single optimizer update without enabling visual graph execution.
- [x] Explicit metric-learning Studio node types are registered as blocked
  templates, can round-trip through the Studio metadata surfaces, and are
  rejected by the compiler by node type even when renamed.
- [x] Compiler output exposes a passive metric-learning graph contract with
  detected node IDs, inferred pair/triplet/output intent, and structured
  blockers while keeping execution disabled.
- [x] Shared-encoder runtime supports deterministic real `SequentialModel`
  multi-branch backward by replaying cached branch inputs and summing
  parameter-gradient maps.
- [ ] Graph compiler/runtime support for visual shared-encoder Siamese graphs
  is still not accepted.
- [ ] Training-mode stateful `SequentialModel` modules still need branch
  activation snapshots before they can be used safely for multi-branch graph
  execution.
- [ ] Visual graph/runtime wiring for `EmbeddingOutput` and `PairScoreOutput`
  is still open.

## Verification Targets

Keep these green while working from Track 14:

- `test_graph_compiler_deferred_nodes`
- `test_training_executor_arrow_parquet`
- `test_text_gui_training_launch`
- `test_cyxmodel_sequence_assets`
- `test_saved_ner_sequence_smoke`
- `test_pattern_template_guard`
- `test_pipeline_operator_metadata`
- `test_siamese_backend_smoke`
- `test_metric_learning_batch_contracts`
- `test_metric_learning_shared_encoder_contracts`
- `test_metric_learning_losses`
- `test_metric_learning_training_step`
- `test_metric_learning_metrics`
- `test_metric_learning_inference_input`
- `test_metric_learning_inference_outputs`
- `test_recurrent_backend_placement`
- `cyxwiz-tests` filters: `[loss]`, `[sequence]` or focused sequence filters
  touched by the change
- `cyxwiz-engine` Debug build
- `git diff --check`
