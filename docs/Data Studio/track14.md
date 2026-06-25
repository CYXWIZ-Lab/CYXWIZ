# Track 14 - NER And Metric-Learning Execution Plan

**Created:** 2026-06-25
**Source:** Working plan for `tofix14.md`.

## Purpose

Track the remaining implementation work from `tofix14.md` against the current
codebase truth.

`tofix14.md` started as a design-gap document. Since then, several NER
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

Known limitations:

- The sequence model currently forwards only `word_ids`; POS IDs and
  attention masks are carried by the batch contract but not consumed by the
  model.
- `FeatureConcat` is not a first-class runtime graph node.
- POS embedding fusion is not implemented end-to-end.
- `SequenceTagOutput` exists as inference decode behavior, not as a
  first-class Studio output node.
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

Goal: align `tofix14.md` with the current implementation state.

Tasks:

- [ ] Mark implemented NER primitives as current truth.
- [ ] Mark stale NER missing-node bullets as completed, partial, or still
  missing.
- [ ] Record that Siamese remains backend-loss-only plus compiler guardrails.
- [ ] Keep the old tensor exposure progress as closed history.

Acceptance:

- `tofix14.md` no longer implies `NERSequenceBuilder`, sequence batches,
  TimeDistributed, or sequence metrics are entirely missing.
- Remaining work points to concrete code gaps.

## Phase 1 - NER End-To-End Smoke

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

- Full POS embedding fusion.
- Full production NER accuracy.
- New UI node types.
- Real `SequenceTagOutput` Studio node.
- Rewriting generic `TrainingExecutor` dispatch.

Acceptance:

- One focused test target covers saved NER graph compile -> sequence batcher ->
  tiny training -> sequence asset/decode proof.
- The test does not depend on the external full NER dataset path.
- Existing guardrails still reject fake Dense-encoded NER placeholders.

## Phase 2 - Sequence Feature Fusion

Goal: consume optional POS IDs without broadening the runtime into an arbitrary
multi-input graph executor.

Possible smallest design:

- Add a focused sequence feature module that combines word embedding and POS
  embedding for token-tagging graphs.
- Keep the module explicit and sequence-only.
- Validate matching `[batch, seq]` payloads before training starts.

Acceptance:

- POS IDs affect the model path only when the graph declares POS usage.
- Shape mismatches fail before training.
- Existing word-only NER path remains supported.

## Phase 3 - Sequence Output Surface

Goal: make sequence-tagging inference/output visible as a first-class Studio
contract after the runtime path is proven.

Tasks:

- Decide whether `SequenceTagOutput` should be a real node or an export/infer
  option on Output.
- Persist tag vocabulary and max-length metadata in exported models.
- Ensure local inference returns token/tag pairs for sequence-tagging models.
- Add tests for invalid/missing sequence vocab assets.

Acceptance:

- A deployed sequence tagger can return readable token/tag predictions.
- Missing sequence metadata fails with a clear error.

## Phase 4 - Siamese Backend Smoke Only

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

- Backend example/test proves the losses are useful in a controlled training
  loop.
- GraphCompiler continues to reject visual Siamese sketches.

## Phase 5 - Siamese Graph Runtime Design

Goal: design the first visual graph contract for metric learning only after the
backend smoke path is proven.

Required contracts:

- typed `PairBatch` / `TripletBatch`,
- shared encoder ownership,
- branch-aware gradient accumulation,
- pair/triplet loss wiring,
- pair and retrieval metrics,
- embedding or pair-score inference outputs.

Acceptance:

- A graph-level design can be implemented without overloading `Batch.data` or
  duplicating encoder weights.

## Verification Targets

Keep these green while working from Track 14:

- `test_graph_compiler_deferred_nodes`
- `test_training_executor_arrow_parquet`
- `test_text_gui_training_launch`
- `test_cyxmodel_sequence_assets`
- `test_recurrent_backend_placement`
- `cyxwiz-tests` filters: `[loss]`, `[sequence]` or focused sequence filters
  touched by the change
- `cyxwiz-engine` Debug build
- `git diff --check`

