# To Fix 58 - Data Studio Sequence And Metric-Learning Follow-Up After done34

Created: 2026-07-07
Source: Follow-up from `done34.md` after the attention-mask consumption slice.

## Purpose

`done34.md` mixed one sequence-runtime gap with several larger visual
metric-learning runtime gaps. The focused 2026-07-07 slice closed attention-mask
consumption for sequence model inputs in training and packaged inference. This
document carries forward the remaining work so `done34.md` can stay archived
without pretending visual metric-learning execution is complete.

## Completed Slice From done34

- Sequence model input construction now consumes `attention_mask`.
- Masked word and POS ids are normalized to the configured pad ids before
  forward.
- Attention-mask shape mismatches fail before model execution.
- Sequence pad ids are threaded from the Arrow sequence batcher build result
  into `TrainingConfiguration::sequence_batch`.
- Local packaged inference now uses the shared sequence input builder instead
  of parsing `attention_mask` and discarding it.
- Regression coverage lives in `test_cyxmodel_sequence_assets`.

Verified:

- `cmake --build build --config Debug --target test_cyxmodel_sequence_assets test_saved_ner_sequence_smoke -- /m:1 /v:minimal`
- `cmake --build build --config Debug --target cyxwiz-engine -- /m:1 /v:minimal`
- `build\bin\Debug\test_cyxmodel_sequence_assets.exe`
- `build\bin\Debug\test_saved_ner_sequence_smoke.exe`
- `git diff --check`

## Remaining Follow-Ups

### Phase 1 - Deployed NER Inference Proof

Goal: prove the saved sequence model path works through the real helper script
and a real model endpoint.

Tasks:

- [ ] Build or reuse a tiny packaged NER `.cyxmodel` artifact from the saved
  graph smoke path.
- [ ] Start the embedded/local inference server with that model.
- [ ] Run `examples/cyxgraph/NER/ner_inference.py` without `--dry-run`.
- [ ] Verify returned token/tag pairs preserve vocabulary order, sequence
  lengths, optional POS inputs, and attention-mask behavior.
- [ ] Add a focused automated smoke if the server lifecycle can be made stable
  in CI.

Acceptance:

- `ner_inference.py` can query a real local/deployed model endpoint and decode
  readable BIO tags.
- Missing or mismatched sequence vocabulary metadata still fails with a clear
  error.

### Phase 2 - FeatureConcat Scope Decision

Goal: decide whether generic visual `FeatureConcat` should become an executable
runtime graph node for sequence feature fusion, or whether sequence-specific
fusion remains the supported contract.

Tasks:

- [ ] Audit current `Concatenate` / `FeatureConcat` graph compiler handling.
- [ ] Compare the generic graph-node route against the existing
  `SequenceFeatureFusionModule` route.
- [ ] Keep generic `FeatureConcat` rejected if it cannot preserve named
  sequence payload semantics cleanly.
- [ ] Document the decision in node metadata and compiler diagnostics.

Acceptance:

- Users get one clear supported route for word/POS sequence fusion.
- Unsupported generic concat shapes fail closed with actionable diagnostics.

### Phase 3 - Visual Metric-Learning Compiler Slice

Goal: turn the passive metric-learning graph contract into one narrow
executable compiler/runtime plan without broad graph rewrites.

Tasks:

- [ ] Define the exact selected-path shape for one pair-training graph:
  pair dataset builder -> shared encoder -> pair branches -> metric loss ->
  optimizer.
- [ ] Map visual node IDs to the internal `PairBatch`,
  `SharedEncoderRuntime`, and metric-loss adapter contracts.
- [ ] Keep unsupported triplet, retrieval, and pair-score paths rejected until
  pair training is proven.
- [ ] Add compiler tests showing the supported pair-training graph moves from
  structured blocker to executable plan.

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

Goal: wire visual output nodes to the already implemented inference contracts.

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

## Guardrails

- Do not mark Studio metric-learning nodes executable until compile, launch,
  runtime execution, inference, and tests prove the path.
- Do not overload the single-input `Batch` contract for pair/triplet data.
- Prefer one narrow executable visual metric-learning shape before expanding to
  triplet, retrieval, or pair-score workflows.
- Keep generic graph-runtime work separate from the proven sequence-specific
  input builder unless a shared contract removes real duplication.

## Verification Targets

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
