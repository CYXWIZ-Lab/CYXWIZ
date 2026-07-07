# Done 34 - Track 14 Follow-Up: Visual Metric Learning And Sequence Runtime Gaps

Created: 2026-06-27
Source: Follow-up from `done14.md` and `track14.md` after commit
`135d7993 Track 14 NER and metric learning execution contracts`.

Archived: 2026-07-07 after the attention-mask consumption slice was
implemented for sequence training and packaged inference. Remaining deployed
NER proof and visual metric-learning runtime work moved to `tofix58.md`.

## Purpose

`done14.md` closed the main Track 14 implementation slice. `tofix34.md` was a
mixed follow-up document: it contained one narrow sequence-runtime gap and a
larger set of visual metric-learning runtime phases. This archive records the
slice completed from that document and points the unfinished queue at
`tofix58.md`.

## Archive Resolution

Done here:

- Phase 2 attention-mask consumption for sequence model input construction.
- Training path now consumes `attention_mask` through `BuildSequenceModelInput`.
- Packaged local inference now uses the same sequence input builder instead of
  parsing `attention_mask` and discarding it.
- Masked word/POS positions are normalized to configured pad ids before forward.
- Mask shape mismatches fail before model execution.
- Regression coverage was added in `test_cyxmodel_sequence_assets`.

Moved to `tofix58.md`:

- Phase 1 deployed `examples/cyxgraph/NER/ner_inference.py` endpoint proof.
- Generic `FeatureConcat` scope decision.
- Phase 3 visual metric-learning compiler slice.
- Phase 4 visual pair-training runtime.
- Phase 5 visual metric-learning outputs.
- Phase 6 stateful shared-encoder snapshots.

Verification completed before archive:

- `cmake --build build --config Debug --target test_cyxmodel_sequence_assets test_saved_ner_sequence_smoke -- /m:1 /v:minimal`
- `cmake --build build --config Debug --target cyxwiz-engine -- /m:1 /v:minimal`
- `cmake --build build --config Release --target cyxwiz-engine -- /m:1 /v:minimal`
- `build\bin\Debug\test_cyxmodel_sequence_assets.exe`
- `build\bin\Debug\test_saved_ner_sequence_smoke.exe`
- `git diff --check`

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

Moved to `tofix58.md`:

- Validate `examples/cyxgraph/NER/ner_inference.py` against a real deployed
  model endpoint.
- Decide whether generic `FeatureConcat` should become a real runtime graph
  node or stay out of scope in favor of task-specific feature fusion.

Moved to done in this archive:

- Consume `attention_mask` inside the sequence model path instead of only
  carrying it through the batch contract.

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

Moved to `tofix58.md`:

- Visual graph compiler/runtime support for executable shared-encoder Siamese
  graphs.
- Visual graph executor routing for `SharedEncoder`, branch nodes,
  metric-learning losses, `EmbeddingOutput`, and `PairScoreOutput`.
- Activation snapshots for training-mode stateful modules such as Dropout and
  BatchNorm before they can be used safely in multi-branch shared encoders.

## Remaining Work

No open checklist is tracked in this archive. Continue from `tofix58.md`, which
contains the deployed NER proof, `FeatureConcat` decision, visual
metric-learning compiler/runtime phases, metric-learning outputs, and stateful
shared-encoder snapshot work.

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
