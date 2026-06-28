# To Fix 34 - Track 14 Follow-Up: Visual Metric Learning And Sequence Runtime Gaps

Created: 2026-06-27
Source: Follow-up from `done14.md` and `track14.md` after commit
`135d7993 Track 14 NER and metric learning execution contracts`.

## Purpose

`done14.md` closed the main Track 14 implementation slice. The engine now has
real NER sequence-tagging execution contracts and internal metric-learning
runtime contracts. This document keeps the remaining work out of the completed
Track 14 archive and makes the next targets explicit.

## Current Code Truth

### NER / Sequence Tagging

Implemented:

- Saved NER graph compile and focused smoke coverage.
- Typed `SequenceBatch` / `ISequenceBatcher` path.
- `NERSequenceBuilder`, `SequenceVocabulary`, sequence padding, and Arrow
  sequence batcher bridge.
- `SequenceExternal` training through `TrainingExecutor`.
- `TimeDistributedDenseModule` for token-level logits.
- Sequence-aware `CrossEntropyLoss` with ignored padding labels.
- Token/entity sequence tag metrics and decode helpers.
- POS feature fusion through the explicit sequence-only module.
- First-class `SequenceTagOutput` Studio output node and terminal compiler
  handling.
- Local sequence inference/importer support for named `word_ids` and `pos_ids`
  on packaged sequence models.

Still open:

- Validate `examples/cyxgraph/NER/ner_inference.py` against a real deployed
  model endpoint.
- Consume `attention_mask` inside the sequence model path instead of only
  carrying it through the batch contract.
- Decide whether generic `FeatureConcat` should become a real runtime graph
  node or stay out of scope in favor of task-specific feature fusion.

### Siamese / Metric Learning

Implemented:

- Typed `PairBatch` and `TripletBatch` contracts.
- Pair/triplet in-memory dataset builders and batchers.
- Metric-learning loss adapters for contrastive, cosine-embedding, and triplet
  workflows.
- Internal pair/triplet training-step helpers.
- `SharedEncoderRuntime` with one shared encoder object, branch routing, branch
  gradient accumulation, and one optional optimizer update.
- Deterministic real `SequentialModel` multi-branch backward by replaying
  cached branch inputs and summing parameter-gradient maps.
- Pair/retrieval metric helpers.
- Embedding and pair-score inference response contracts and JSON packaging.
- Local inference input parsing for embedding and pair-score payloads.
- Embedded local inference routes:
  - `POST /v1/embeddings`
  - `POST /v1/pair-score`
- Blocked Studio metric-learning node metadata, aliases, icons, properties,
  pattern/CyxQL coverage, and type-based compiler rejection.
- Passive compiler graph contract that records detected metric-learning intent
  while keeping execution disabled.

Still open:

- Visual graph compiler/runtime support for executable shared-encoder Siamese
  graphs.
- Visual graph executor routing for `SharedEncoder`, branch nodes,
  metric-learning losses, `EmbeddingOutput`, and `PairScoreOutput`.
- Activation snapshots for training-mode stateful modules such as Dropout and
  BatchNorm before they can be used safely in multi-branch shared encoders.

## Recommended Implementation Order

### Phase 1 - Deployed NER Inference Proof

Goal: prove the packaged sequence model can be served and queried through the
real helper script.

Tasks:

- [ ] Build or reuse a tiny packaged NER model artifact from the saved graph
  smoke path.
- [ ] Start the embedded/local inference server with that model.
- [ ] Run `examples/cyxgraph/NER/ner_inference.py` without `--dry-run`.
- [ ] Verify returned token/tag pairs preserve max length, vocabulary order,
  and optional POS input behavior.
- [ ] Add a focused automated smoke if the server lifecycle can be made stable
  in CI.

Acceptance:

- `ner_inference.py` can query a real local/deployed model endpoint and decode
  readable BIO tags.
- Missing or mismatched sequence vocab metadata still fails with a clear error.

### Phase 2 - Attention Mask Consumption

Goal: make sequence models respect the existing attention mask contract where
that mask is semantically required.

Tasks:

- [ ] Audit recurrent and sequence-head modules for where masked timesteps
  should be ignored.
- [ ] Keep padding-label ignore behavior in the loss path.
- [ ] Add model-side attention-mask behavior only where it changes forward or
  metric semantics.
- [ ] Add regression coverage proving masked padded timesteps do not affect
  sequence outputs or metrics beyond the already ignored loss positions.

Acceptance:

- The sequence batch `attention_mask` is not just transported; it is consumed by
  supported sequence model paths where appropriate.

### Phase 3 - Visual Metric-Learning Compiler Slice

Goal: turn the passive metric-learning graph contract into a narrow executable
compiler/runtime plan without broad graph rewrites.

Tasks:

- [ ] Define the exact selected-path shape for one pair-training graph:
  pair dataset builder -> shared encoder -> pair branches -> metric loss ->
  optimizer.
- [ ] Map visual node IDs to the internal `PairBatch`, `SharedEncoderRuntime`,
  and metric-loss adapter contracts.
- [ ] Keep unsupported triplet, retrieval, or pair-score paths rejected until
  pair training is proven.
- [ ] Add compiler tests showing the pair-training graph moves from structured
  blocker to executable plan only for the supported shape.

Acceptance:

- One minimal visual pair-training graph compiles into an executable internal
  metric-learning plan.
- Unsupported metric-learning sketches still fail closed with structured
  blockers.

### Phase 4 - Visual Pair Training Runtime

Goal: execute the narrow pair-training plan through the existing internal
metric-learning helpers.

Tasks:

- [ ] Build the selected visual pair dataset into a `PairBatcher`.
- [ ] Instantiate one shared encoder object from the visual encoder subtree.
- [ ] Route pair branches through `SharedEncoderRuntime`.
- [ ] Run the selected metric loss and one optimizer update per batch.
- [ ] Report pair metrics through the training result surface.
- [ ] Add an end-to-end saved-graph smoke for tiny visual pair training.

Acceptance:

- A tiny saved visual pair-training graph runs a bounded training pass and
  proves distance movement on similar/dissimilar pairs.

### Phase 5 - Metric-Learning Outputs

Goal: wire the visual output nodes to the already implemented inference
contracts.

Tasks:

- [ ] Wire `EmbeddingOutput` visual graphs to `/v1/embeddings`.
- [ ] Wire `PairScoreOutput` visual graphs to `/v1/pair-score`.
- [ ] Preserve sample/class metadata in responses.
- [ ] Keep score modes explicit: distance and similarity must not masquerade as
  class probabilities.

Acceptance:

- Visual metric-learning inference graphs can return stable embedding and
  pair-score JSON responses using the internal contracts added in Track 14.

### Phase 6 - Stateful Shared-Encoder Snapshots

Goal: safely support training-mode stateful modules in multi-branch shared
encoders.

Tasks:

- [ ] Identify modules whose backward pass depends on branch-specific forward
  state, including Dropout and BatchNorm.
- [ ] Add branch activation/state snapshots or a backend-level replay contract
  that preserves equivalent semantics.
- [ ] Keep the current guard rejecting unsupported stateful training paths until
  snapshot coverage is proven.
- [ ] Add regression tests for pair and triplet branches with stateful modules.

Acceptance:

- Training-mode stateful `SequentialModel` encoders are either safely supported
  with snapshots or still rejected with a clear diagnostic.

## Non-Goals

- Reopening completed Track 14 NER sequence training work.
- Replacing `TrainingExecutor`, `GraphCompiler`, or `ModelBuilder` with a new
  graph abstraction.
- Overloading the old single-input `Batch` contract for pair/triplet data.
- Marking Studio metric-learning nodes fully supported before compile, launch,
  runtime execution, inference, and tests prove the path.

## Verification Targets

Keep these green while working from this document:

- `test_saved_ner_sequence_smoke`
- `test_cyxmodel_sequence_assets`
- `test_graph_compiler_deferred_nodes`
- `test_pipeline_operator_metadata`
- `test_pattern_template_guard`
- `test_siamese_backend_smoke`
- `test_metric_learning_batch_contracts`
- `test_metric_learning_shared_encoder_contracts`
- `test_metric_learning_losses`
- `test_metric_learning_training_step`
- `test_metric_learning_metrics`
- `test_metric_learning_inference_input`
- `test_metric_learning_inference_outputs`
- `cyxwiz-engine` Debug build
- `git diff --check`
