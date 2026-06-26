# To Fix 14 - NER Sequence Tagging and Siamese Engine Gaps

This note documents the gap between advanced graph designs and what the
current CyxWiz engine can train today.

The two immediate advanced graph families are:

- NER sequence tagging
- Siamese / metric-learning networks

Both are useful because they expose the same underlying limitation: the
backend has several low-level neural-network pieces, but the Studio graph
runtime still assumes a mostly single-input sequential training path.

## Current Code Truth 2026-06-25

This document is older than several sequence-tagging implementation slices.
Before adding work from this file, use this current truth:

- `SequenceBatch`, `ISequenceBatcher`, `SequenceBatcher`,
  `SequenceVocabulary`, and `NERSequenceBuilder` exist.
- `BuildSequenceBatcherFromArrowDataset` can bridge Arrow sequence rows into a
  prebuilt `ISequenceBatcher`.
- `GraphCompiler` captures first-class `NERSequenceBuilder` nodes into
  `SequenceBatchConfig`.
- `TimeDistributed` is implemented as `TimeDistributedDenseModule` and is
  built by `ModelBuilder`.
- `TrainingExecutor` has `SequenceExternal` mode and can train token-tagging
  batches when launched with a prebuilt sequence batcher.
- `CrossEntropyLoss` uses sequence `ignore_index` for token-level targets; the
  current implementation does not use a separate first-class
  `TokenCrossEntropyLoss` node.
- Sequence tag metrics and sequence inference decode helpers exist.
- Saved NER graph assets exist under `examples/cyxgraph/NER`.
- Optional `pos_ids` can be consumed by the explicit sequence feature-fusion
  path when the training config declares `sequence_feature_fusion=true`.
  Attention masks are carried by the batch contract but not yet consumed by
  the model path.
- Generic `FeatureConcat` is still not a first-class runtime graph node; POS
  embedding fusion is implemented as a sequence-only module instead.
- `SequenceTagOutput` is now a first-class Studio output node for
  token-level logits and BIO decode metadata. It is a terminal graph/output
  contract, not a trainable layer or PipelineExecutor table operator.
- Siamese support remains backend-loss-only plus compiler guardrails. Track 14
  now records the first graph-runtime design contract, but the visual graph
  runtime still lacks typed pair/triplet batches, shared encoder ownership,
  branch-aware backward pass, and metric-learning output contracts.

Use `track14.md` as the active execution plan. The next narrow target is an
end-to-end saved NER graph smoke, not broad Siamese graph support.

## Carryover From To Fix 13

`tofix13.md` closed the in-repo C++ tensor-operation parity gap through
tensor slices 1-13. The backend now has the core tensor API needed for
the next Studio-facing work, including shape operations, indexing,
concat/split, reductions, broadcasting, elementwise math, comparisons,
logical masks, `Dot`, and `BatchMatMul`.

Do not reopen that tensor parity thread unless a regression appears. The
remaining backend work should move forward here in two lanes:

- immediate exposure lane: Python bindings, Studio graph/runtime
  integration, and frontend surfaces for the newly migrated tensor API
- broader backend lane: layer file/folder split, metrics, callbacks,
  serialization, checkpointing, ONNX, quantization, model lifecycle,
  model registry/hub, and documentation/testing hardening

The immediate next step is the exposure lane. Keep it narrow: bind and
surface the completed tensor API first, add smoke tests, then return to
the larger backend architecture items after Studio can actually use the
new primitives.

## Backend Modularity Follow-up: Split Layer Translation Units

The backend still has a weak layer boundary: many unrelated neural-network
layers live in the large monolithic `cyxwiz-backend/src/algorithms/layer.cpp`
and shared `cyxwiz-backend/include/cyxwiz/layer.h` surface. This makes
attention, recurrent, normalization, embedding, convolution, and transformer
work harder to review, test, and evolve independently.

This should stay tracked here as part of the broader backend lane. Do not
mix a full file-structure rewrite into focused feature work such as
`tofix8`, but when touching layer families, prefer extracting them into
focused translation units and headers.

Target direction:

- keep public compatibility through `cyxwiz/layer.h` during migration
- move implementation by family into `src/algorithms/layers/`
- examples: `embedding_layer.cpp`, `attention_layers.cpp`,
  `transformer_layers.cpp`, `recurrent_layers.cpp`,
  `normalization_layers.cpp`, `convolution_layers.cpp`
- keep shared private helpers in `src/algorithms/layers/layer_utils.*`
- avoid adding new layer families to the monolithic `layer.cpp`
- move tests toward family-specific fixtures as files are split

Done criteria:

- new backend layer work has a clear translation-unit home
- `layer.cpp` no longer grows for unrelated layer families
- one family split compiles and passes existing tests before continuing
  broader migration

## Shared Root Gap

The current graph/training path is strongest for:

```text
one input tensor -> sequential model -> one prediction tensor -> one loss
```

NER needs:

```text
one token sequence -> per-token predictions -> mask-aware token loss
```

Siamese networks need:

```text
two or three related inputs -> shared encoder weights -> pair/triplet loss
```

So the reusable engine work should avoid adding one-off graph hacks. The
smallest useful design is:

- typed multi-input batch contracts
- explicit shared-module support
- mask-aware and pair-aware losses
- graph compiler validation for non-linear model topology
- inference outputs that preserve structure instead of flattening every
  model result into one classification vector

---

# NER Sequence Tagging

The NER example lives here:

```text
examples/cyxgraph/NER/
```

The graph file is:

```text
examples/cyxgraph/NER/ner_bilstm_sequence_tagger.cyxgraph
```

The source dataset is:

```text
D:\Dev\DataSet_List\NER\NER dataset.csv
```

## Short Version

The NER graph is a target design, not fully supported by the engine yet.

Current CyxWiz text examples are whole-text classifiers:

```text
Text -> one label
```

NER is token-level sequence tagging:

```text
Token 1 -> tag 1
Token 2 -> tag 2
Token 3 -> tag 3
...
```

That means NER needs model output shaped like:

```text
[batch_size, sequence_length, num_tags]
```

and labels shaped like:

```text
[batch_size, sequence_length]
```

The current graph/training path is not yet complete for this.

## Dataset Shape

The raw dataset is CoNLL-style CSV:

```text
Sentence #,Word,POS,Tag
Sentence: 1,Thousands,NNS,O
,of,IN,O
,demonstrators,NNS,O
,have,VBP,O
,marched,VBN,O
,through,IN,O
,London,NNP,B-geo
```

Each row is one token. The `Tag` column is the NER label for that token.

The prep script converts it into sentence-level rows:

```text
sentence_id,tokens,pos_tags,ner_tags
1,"Thousands of demonstrators ...","NNS IN NNS ...","O O O ..."
```

Validation run against the dataset found:

```text
sentences: 47,959
tokens: 1,048,575
max source sentence length: 104
POS vocab size: 44
NER tag vocab size: 19
```

The tag vocabulary is:

```text
[PAD], [UNK], O,
B-geo, B-gpe, B-per, I-geo, B-org, I-org, B-tim,
B-art, I-art, I-per, I-gpe, I-tim, B-nat, B-eve, I-eve, I-nat
```

## Intended NER Pipeline

The target NER graph is:

```text
NER CSV
  -> NERSequenceBuilder
  -> TokenVocabulary
  -> POSVocabulary
  -> NERTagVocabulary
  -> SequencePadding
  -> DataSplit
  -> DataLoader
  -> Word Embedding
  -> POS Embedding
  -> FeatureConcat
  -> BiLSTM return_sequences=true
  -> Dropout
  -> TimeDistributed Dense
  -> TokenCrossEntropy
  -> Adam
  -> SequenceTagMetrics
  -> SequenceTagOutput
```

The model predicts one BIO tag for each visible token.

## What The Engine Already Has

The current engine already has pieces we can reuse:

- `DataInput`
- `DataSplit`
- `DataLoader`
- `Embedding`
- `LSTM`
- `GRU`
- `Dropout`
- `Dense`
- `Adam`
- basic `CrossEntropyLoss`
- text tokenizer/vocabulary/padding path for whole-text classification

These are enough for sentiment-style text classification:

```text
TextTokenizer -> TextVocabulary -> TextPadding
  -> Embedding -> GRU/LSTM -> Dense -> CrossEntropy
```

They are not enough for NER because NER needs aligned token labels and
per-token output.

## Missing Or Incomplete Nodes

### 1. `NERSequenceBuilder`

Current status:

- Implemented as `NERSequenceBuilder` / `BuildNERSequenceData`.
- Also executable through `PipelineExecutor::ExecuteNERSequenceBuilder` for
  Arrow-table materialization.
- Still useful to keep in this document because future work must connect it to
  richer sequence feature fusion and output contracts.

Purpose:

- read sentence-level rows
- split `tokens`, `pos_tags`, and `ner_tags`
- verify all three sequences have the same length
- preserve sentence boundaries
- produce sequence samples

Why needed:

The current text path treats a row as one text sample with one label.
NER rows contain multiple labels, one per token.

### 2. `TokenVocabulary`

Current status:

- Implemented through `SequenceVocabulary` with token vocabulary mode.

Purpose:

- map words to IDs
- support `[PAD]` and `[UNK]`
- save/load vocab files
- optionally lowercase tokens

Why needed:

NER needs a word vocabulary independent of POS and tag vocabularies.

### 3. `POSVocabulary`

Current status:

- Implemented through `SequenceVocabulary` with POS vocabulary mode.
- POS IDs are batchable and can be consumed by the explicit
  sequence-feature-fusion model path.

Purpose:

- map POS tags like `NNP`, `VBD`, `IN` to IDs
- support `[PAD]` and `[UNK]`

Why needed:

The graph optionally uses POS embeddings as extra token features.

### 4. `NERTagVocabulary`

Current status:

- Implemented through `SequenceVocabulary` with tag vocabulary mode.
- Tag vocabulary values are used by sequence metrics and inference decode.

Purpose:

- map BIO tags to class IDs
- reserve `0` for `[PAD]`
- keep `O` as the outside/non-entity class
- save/load tag order for inference decoding

Why needed:

The loss and output decoder must agree on the exact tag index order.

### 5. `SequencePadding`

Current status:

- Implemented inside `SequenceBatcher`.
- It pads/truncates word IDs, optional POS IDs, tag IDs, and attention masks.

Purpose:

- pad/truncate word IDs
- pad/truncate POS IDs
- pad/truncate tag IDs
- create attention/loss mask

Expected outputs:

```text
word_ids:       [batch, max_length]
pos_ids:        [batch, max_length]
tag_ids:        [batch, max_length]
attention_mask: [batch, max_length]
```

Why needed:

Existing `TextPadding` only covers simple text-token padding. NER needs
labels and masks padded in alignment with tokens.

### 6. `FeatureConcat`

Current status:

- Still missing as a first-class runtime graph node.
- Word/POS fusion for sequence taggers is implemented separately by
  `SequenceFeatureFusionModule`, which avoids turning the training runtime
  into a generic multi-input graph executor.

Purpose:

- concatenate word embeddings and POS embeddings per token

Expected input:

```text
word_embedding: [batch, seq_len, word_dim]
pos_embedding:  [batch, seq_len, pos_dim]
```

Expected output:

```text
[batch, seq_len, word_dim + pos_dim]
```

Why needed:

The model should be able to consume multiple token-level feature streams.

### 7. `LSTM return_sequences=true` Graph Support

Current status:

- Implemented for the supported sequential recurrent wrappers and covered by
  TimeDistributed sequence-head tests.
- Future work should keep validating output/label shape compatibility for
  token classification graphs.

Purpose:

- preserve output for every timestep

Expected output:

```text
[batch, seq_len, hidden_size * num_directions]
```

Why needed:

Sentiment classification can use the last hidden state. NER needs one
hidden vector per token.

Backend LSTM/GRU code exists, but the graph compiler/runtime must verify
that `return_sequences=true` is preserved through the model path.

### 8. `TimeDistributedDense`

Purpose:

- apply the same Dense classifier to each token

Expected input:

```text
[batch, seq_len, hidden]
```

Expected output:

```text
[batch, seq_len, num_tags]
```

Why needed:

A normal `Dense` layer may collapse or assume a 2D input. NER needs a
per-timestep classifier.

Current engine truth:

- `TimeDistributed` is already visible and implemented as a graph node.
- `GraphCompiler` recognizes it as a model layer and requires sequence-shaped
  input.
- `ModelBuilder` builds it as `TimeDistributedDenseModule`.
- Backend placement now classifies it as an explicit sequence wrapper rather
  than direct ArrayFire tensor-capable work.
- Token-label batching, ignore-index-aware loss, token metrics, and inference
  decode exist.
- Full NER support still needs feature fusion for optional POS embeddings and
  a first-class sequence output/export surface.

Follow-up work:

- keep backend placement honest: it is a sequence wrapper, not direct GPU
  tensor execution
- keep validating output/label shapes for token classification graphs
- keep the saved NER graph end-to-end smoke green before claiming broader NER
  work
  support
- keep sequence-only POS feature fusion separate from any future generic
  multi-input graph executor

### 9. `TokenCrossEntropyLoss`

Current status:

- Implemented as sequence-aware `CrossEntropyLoss` with `ignore_index`, plus
  `TokenCrossEntropyLossCpu` helper coverage.
- There is no separate first-class Studio `TokenCrossEntropyLoss` node today.

Purpose:

- compute cross entropy for every non-padding token
- ignore padded labels using `ignore_index=0`

Expected inputs:

```text
predictions: [batch, seq_len, num_tags]
targets:     [batch, seq_len]
mask:        [batch, seq_len]
```

Why needed:

Standard classification loss is not enough. Padding positions must not
contribute to the loss.

### 10. `SequenceTagMetrics`

Current status:

- Implemented by `ComputeSequenceTagMetricsFromLogits` and consumed by
  sequence training.

Purpose:

- token accuracy ignoring padding
- entity precision
- entity recall
- entity F1
- optional per-tag counts

Why needed:

Token accuracy alone can be misleading because the `O` tag dominates.
Entity-level F1 is the real NER metric.

### 11. `SequenceTagOutput`

Current status:

- Implemented as sequence inference decode helpers, local inference response
  formatting, and a first-class Studio output node.
- The saved NER graph now uses `SequenceTagOutput` instead of overloading the
  generic `Output` node.

Purpose:

- decode predicted tag IDs to BIO labels
- optionally repair invalid BIO transitions
- expose token/tag pairs for inference

Expected output:

```text
London -> B-geo
Iraq   -> B-geo
British -> B-gpe
```

Why needed:

The model output is numeric logits. The user needs readable entity tags.

## Inference Gap

The NER folder includes:

```text
examples/cyxgraph/NER/ner_inference.py
```

The helper can already:

- load metadata
- load word/POS/tag vocabularies
- encode and pad a sentence
- print a dry-run payload
- call the embedded inference endpoint
- decode token-level logits into BIO tags

But the backend inference server must support sequence-tagging payloads:

```json
{
  "inputs": {
    "word_ids": [...],
    "pos_ids": [...],
    "attention_mask": [...]
  }
}
```

and return either:

```text
[max_length, num_tags]
```

or:

```text
max_length * num_tags
```

Until the model/runtime supports that output contract, the helper is most
useful in `--dry-run` mode.

## Implementation Order

Use `track14.md` for active execution. The historical order below is retained
as design context, but several early items are now implemented.

### Phase 1 - Data preparation and schema

Status: partially complete.

1. Keep `prepare_ner_demo.py` as the dataset converter. Implemented.
2. Generate these assets. Implemented under `examples/cyxgraph/NER/generated`:
   - `ner_sentences.csv`
   - `ner_word_vocab.txt`
   - `ner_pos_vocab.txt`
   - `ner_tag_vocab.txt`
   - `ner_metadata.json`
3. Add graph/runtime support for sentence-level sequence samples. Implemented
   for Arrow-backed sequence rows and prebuilt `ISequenceBatcher` launch.

### Phase 2 - Sequence tensors

Status: mostly complete.

1. Implement `NERSequenceBuilder`. Implemented.
2. Implement or adapt `TokenVocabulary`. Implemented through
   `SequenceVocabulary`.
3. Implement `NERTagVocabulary`. Implemented through `SequenceVocabulary`.
4. Implement `SequencePadding`. Implemented through `SequenceBatcher`.
5. Add `attention_mask` and `loss_mask`. Attention masks and tag
   `ignore_index` are present; model-side attention-mask consumption remains
   future work.

### Phase 3 - Sequence model path

Status: partially complete.

1. Verify `Embedding` accepts `[batch, seq_len]`. Implemented for the current
   sequence training path.
2. Verify LSTM/GRU can return `[batch, seq_len, hidden]`. Implemented for
   supported `return_sequences=true` wrappers.
3. Add generic `FeatureConcat`. Still missing as a first-class graph-runtime
   node; sequence-only word/POS fusion is implemented separately.
4. Add `TimeDistributedDense`. Implemented.

### Phase 4 - Token-level training

Status: mostly complete.

1. Add token-level cross entropy. Implemented through sequence-aware
   `CrossEntropyLoss` / `ignore_index`.
2. Add mask-aware reduction. Implemented through ignored tag IDs.
3. Add token-level accuracy. Implemented.
4. Add entity-level precision/recall/F1. Implemented.
5. Add saved NER graph end-to-end smoke. Implemented by
   `test_saved_ner_sequence_smoke`; keep it as the narrow regression gate for
   compile, sequence batcher construction, tiny training, asset packaging, and
   decode.

### Phase 5 - Inference and export

Status: mostly complete for the local/deployed sequence tagger surface.

1. Save vocab and metadata with `.cyxmodel`. Implemented at the sequence asset
   packaging level.
2. Add `SequenceTagOutput`. Implemented as a first-class Studio node plus
   decode/response behavior.
3. Make embedded inference accept named sequence inputs. Implemented for
   word IDs and POS IDs, including POS-fused packing.
4. Make embedded inference return token logits/decode. Decode response exists
   for sequence logits.
5. Verify `ner_inference.py` against a deployed model. Still missing.

## Definition Of Done

NER support is done when:

- `ner_bilstm_sequence_tagger.cyxgraph` loads without unknown-node errors.
- the graph compiles and populates `SequenceBatchConfig`.
- the generated data prep files are consumed through the graph launch path.
- batches contain aligned word IDs, optional POS IDs, tag IDs, and masks.
- BiLSTM/GRU sequence models can return per-token hidden states.
- the token classifier outputs `[batch, seq_len, num_tags]`.
- loss ignores padding / ignored tag IDs.
- metrics report token accuracy and entity F1.
- a focused saved-graph smoke trains a tiny bounded run successfully.
- checkpoint/export includes graph, weights, vocabularies, metadata, and max
  length.
- deployed/local inference returns readable token/tag predictions for a
  sentence.
- optional POS IDs are either consumed through an explicit feature-fusion path
  or clearly reported as unused.

## Why This Matters

NER is the first real sequence-labeling use case for CyxWiz.

It forces the backend to handle:

- multiple aligned input sequences
- one label per token
- padding masks
- recurrent/transformer sequence outputs
- per-token loss
- entity-level metrics
- sequence-aware inference output

These capabilities will also help future tasks such as:

- part-of-speech tagging
- chunking
- slot filling
- token classification
- sequence labeling for logs/code/text
- transformer token classification

---

# Siamese / Metric-Learning Networks

This section documents the gap between a Siamese CyxGraph design and what
the current CyxWiz engine can train today.

## Short Version

The CyxWiz backend already has several pieces needed for Siamese
networks, including embeddings, recurrent layers, dense layers, optimizers,
and metric-learning losses.

The Studio graph path is not yet complete for Siamese training because it
does not yet model:

- two-input or three-input training batches
- shared-weight encoder branches
- pairwise distance/merge operations in the executable training path
- contrastive, cosine embedding, or triplet loss nodes in the visual graph
- pair/triplet metrics and inference output

Current status 2026-06-25:

- Backend `CosineEmbeddingLoss`, `TripletLoss`, and `ContrastiveLoss` exist
  and have focused unit coverage.
- The graph compiler deliberately rejects selected-path metric-learning
  sketches such as `SharedEncoder` instead of pretending the existing
  single-input runtime can train them.
- No visual graph node should be marked runtime-supported until typed
  pair/triplet batches, shared encoder ownership, branch-aware backward pass,
  and embedding/pair-score output contracts exist.

## Intended Siamese Pipeline

For pair-based contrastive learning:

```text
Pair Dataset
  -> PairBatcher
  -> Encoder Input A
  -> Shared Encoder
  -> Embedding A

Pair Dataset
  -> PairBatcher
  -> Encoder Input B
  -> Shared Encoder
  -> Embedding B

Embedding A + Embedding B + pair label
  -> ContrastiveLoss or CosineEmbeddingLoss
  -> Adam
  -> PairMetrics
  -> EmbeddingOutput
```

For triplet learning:

```text
Triplet Dataset
  -> TripletBatcher
  -> Anchor   -> Shared Encoder -> Anchor Embedding
  -> Positive -> Shared Encoder -> Positive Embedding
  -> Negative -> Shared Encoder -> Negative Embedding
  -> TripletLoss
  -> Adam
  -> RetrievalMetrics
  -> EmbeddingOutput
```

## What The Engine Already Has

The current backend already has reusable components:

- `Tensor`
- `SequentialModel`
- `Dense`
- `Embedding`
- `LSTM`
- `GRU`
- `Dropout`
- activations
- `Adam`
- `SGD`
- `CosineEmbeddingLoss`
- `TripletLoss`
- `ContrastiveLoss`

These are enough to prototype a Siamese model in backend code or Python
bindings with custom training logic.

They are not enough for a first-class `.cyxgraph` Siamese workflow because
the visual graph compiler and training executor still build a mostly
single-path sequential model.

## Missing Or Incomplete Nodes

### 1. `PairDatasetBuilder`

Purpose:

- build `(sample_a, sample_b, label)` rows
- support positive and negative pair generation
- preserve original class IDs for evaluation
- optionally balance positive/negative pairs

Expected labels:

```text
0 = similar
1 = dissimilar
```

This matches the current backend `ContrastiveLoss` convention.

### 2. `TripletDatasetBuilder`

Purpose:

- build `(anchor, positive, negative)` samples
- guarantee anchor and positive share the same class
- guarantee negative comes from a different class
- support simple hard-negative hooks later

This should be separate from `PairDatasetBuilder` instead of overloading
one node with too many modes.

### 3. `PairBatcher`

Purpose:

- emit two aligned input tensors and one pair-label tensor

Expected output:

```text
input_a: [batch, ...]
input_b: [batch, ...]
labels:  [batch]
```

Why needed:

The current batch path expects one `Batch.data` tensor and one
`Batch.labels` tensor.

### 4. `TripletBatcher`

Purpose:

- emit anchor, positive, and negative tensors

Expected output:

```text
anchor:   [batch, ...]
positive: [batch, ...]
negative: [batch, ...]
```

Why needed:

Triplet loss cannot be represented cleanly as a normal single-input
classification batch.

### 5. `SharedEncoder`

Purpose:

- define one encoder module whose weights are reused by multiple branches
- prevent accidental branch weight divergence
- expose one parameter set to the optimizer

Why needed:

Duplicating the same layers in the graph creates two independent networks.
A Siamese model requires the same encoder weights for both branches.

### 6. `SiameseBranch`

Purpose:

- mark branch inputs that feed the same shared encoder
- make graph validation explicit
- keep branch wiring readable in the node editor

This can be a lightweight graph/runtime marker. It should not introduce a
second model abstraction unless the compiler needs it.

### 7. `PairDistance`

Purpose:

- compute Euclidean distance, squared Euclidean distance, cosine distance,
  or cosine similarity between embeddings

Expected input:

```text
embedding_a: [batch, embed_dim]
embedding_b: [batch, embed_dim]
```

Expected output:

```text
distance: [batch]
```

Why needed:

Some Siamese workflows train directly with a distance output and BCE/MSE,
while others feed embeddings directly to a metric loss. Both should be
possible.

### 8. `AbsDiff`

Purpose:

- compute absolute difference between two embeddings

Expected output:

```text
abs(embedding_a - embedding_b): [batch, embed_dim]
```

Why needed:

This supports common verification heads:

```text
AbsDiff -> Dense -> Sigmoid/BCE
```

### 9. `ContrastiveLoss`

Purpose:

- expose backend `ContrastiveLoss` as a graph training node
- support configurable margin
- validate that labels use the expected convention

Expected inputs:

```text
embedding_a: [batch, embed_dim]
embedding_b: [batch, embed_dim]
labels:      [batch]
```

### 10. `CosineEmbeddingLoss`

Purpose:

- expose backend `CosineEmbeddingLoss` as a graph training node
- support configurable margin

Expected labels:

```text
1  = similar
-1 = dissimilar
```

The graph compiler should make the label convention visible, because it
differs from `ContrastiveLoss`.

### 11. `TripletLoss`

Purpose:

- expose backend `TripletLoss` as a graph training node
- support Euclidean and cosine distance
- support configurable margin

Expected inputs:

```text
anchor_embedding:   [batch, embed_dim]
positive_embedding: [batch, embed_dim]
negative_embedding: [batch, embed_dim]
```

### 12. `PairMetrics`

Purpose:

- pair accuracy at a distance threshold
- ROC/AUC where possible
- positive/negative distance means
- distance histogram for debugging collapse

Why needed:

Loss alone does not tell whether embeddings are useful.

### 13. `RetrievalMetrics`

Purpose:

- recall@k
- mean reciprocal rank
- nearest-neighbor class agreement

Why needed:

Siamese embeddings are often used for retrieval, not only binary
verification.

### 14. `EmbeddingOutput`

Purpose:

- export learned embeddings
- expose embedding vectors for inference
- optionally include source sample IDs and class IDs

Expected output:

```text
sample_id, class_id, embedding[]
```

Why needed:

The useful product of Siamese training is often the embedding space, not a
single classification label.

## Runtime / Compiler Gaps

Track 14 Phase 5 now fixes the target contract for the next implementation
slices. Keep the current compiler fail-closed behavior until the typed
batchers and shared-encoder executor path exist.

### 1. Multi-input batch contract

The current training path should grow a typed batch representation for
advanced models instead of overloading `Batch.data` with packed tensors.

Chosen direction:

```text
PairBatch:
  input_a: Tensor
  input_b: Tensor
  pair_label: Tensor
  optional sample/class IDs

TripletBatch:
  anchor: Tensor
  positive: Tensor
  negative: Tensor
  optional sample/class IDs
```

Do not route pair or triplet training through the existing single-input
`Batch.data` field.

### 2. Shared-weight execution

The runtime must call the same encoder object for multiple inputs and
accumulate gradients into the same parameter set.

The optimizer must update shared parameters once per training step.

`SharedEncoder` should own the encoder module/parameter set. Branch nodes
reference that shared encoder; they do not clone layers or independently own
weights.

### 3. Branch-aware backward pass

For contrastive/cosine losses, gradients must flow back through both
encoder calls:

```text
loss gradient -> embedding_a -> shared encoder
loss gradient -> embedding_b -> shared encoder
```

For triplet loss, gradients must flow through anchor, positive, and
negative branches.

The branch gradients are accumulated onto the shared encoder parameters before
the optimizer step. The optimizer sees one shared parameter set, not one set
per branch.

### 4. Graph validation

Compile should reject Siamese graphs when:

- branch encoders are duplicated instead of shared
- embedding dimensions do not match
- pair labels are missing
- triplet negative input is missing
- loss label convention is ambiguous
- optimizer cannot reach the metric loss
- a metric-learning output is forced through a class-probability `Output`
  node

### 5. Inference contract

Siamese inference should support at least two modes:

```text
single sample -> embedding
sample pair   -> distance/similarity score
```

This should not be forced through a class-probability output.

Use `EmbeddingOutput` for single-sample embeddings and `PairScoreOutput` for
distance/similarity scoring. Pair metrics should report thresholded pair
accuracy and positive/negative distance means first; retrieval metrics can add
recall@k, mean reciprocal rank, and nearest-neighbor class agreement after
embedding export is available.

## Implementation Order

### Phase 1 - Backend smoke path

Status: completed 2026-06-26 by `test_siamese_backend_smoke`.

1. Add a small C++ or Python Siamese example using existing backend
   losses. Implemented as a focused C++ test.
2. Build a shared encoder manually in code.
3. Train on a toy pair dataset.
4. Verify similar pairs stay close and dissimilar pairs move apart.

### Phase 2 - Graph node surface

Status: completed for blocked Studio metadata and compiler rejection guards on
2026-06-26. Runtime execution and metric-loss graph wiring are still not
accepted.

1. Add `ContrastiveLoss`, `CosineEmbeddingLoss`, and `TripletLoss` node
   types. Implemented as blocked Studio node types.
2. Register node metadata and documentation. Implemented with
   `Template` / `Blocked` metadata.
3. Add graph load/save aliases for those node types. Implemented for the
   blocked node surface.
4. Add compiler loss mapping for metric losses. Still open; the compiler
   currently rejects metric-learning graph sketches instead of executing them.

Guardrail:

- keep metric-learning nodes marked unsupported until `PairBatch` /
  `TripletBatch` and shared-encoder execution are implemented
- graph compile should still reject selected Siamese training paths with a
  clear typed-batch/shared-weight contract error

### Phase 3 - Pair and triplet batching

Status: completed for internal row/sample builders and in-memory batchers.
Studio graph execution is still closed.

1. Add internal `PairBatch` and `TripletBatch` structs. Implemented in
   `metric_learning_batch.h`.
2. Add shape and label-convention tests independent of Studio nodes.
   Implemented by `test_metric_learning_batch_contracts`.
3. Add `PairSample`, `TripletSample`, `PairBatcher`, and `TripletBatcher`.
   Implemented in `metric_learning_batcher.h`.
4. Add in-memory batcher tests for aligned branch tensors, label conventions,
   metadata IDs, partial batches, drop-last behavior, and invalid shapes.
   Implemented by `test_metric_learning_batch_contracts`.
5. Add `PairDatasetBuilder`. Implemented in
   `metric_learning_dataset_builder.h`.
6. Add `TripletDatasetBuilder`. Implemented in
   `metric_learning_dataset_builder.h`.
7. Add tests for label conventions and shape contracts. Implemented by
   `test_metric_learning_batch_contracts`.

### Phase 4 - Shared encoder graph execution

Status: started. Internal shared-encoder runtime harness, passive compiler
contract, deterministic SequentialModel branch replay/backward, and unsafe
stateful SequentialModel replay guard completed 2026-06-26; visual graph
execution and full branch activation snapshots remain open.

1. Add a minimal `SharedEncoder` compiler/runtime representation.
   `SharedEncoderRuntime` now covers the runtime ownership harness.
   `MetricLearningGraphContract` now records the selected compiler graph
   contract and blockers without accepting execution.
2. Execute the same encoder object for every branch. Covered for internal
   pair/triplet payloads by `SharedEncoderRuntime`.
3. Accumulate branch gradients into shared parameters. Covered by the runtime
   harness and `test_metric_learning_shared_encoder_contracts`.
4. Update shared parameters once per batch. Covered by the runtime harness.
5. Add branch activation snapshots before using training-mode stateful
   `SequentialModel` modules for multi-branch graph training. Still open;
   deterministic `SequentialModel` encoders now replay cached branch inputs and
   accumulate parameter-gradient maps.

### Phase 5 - Metrics and inference

Status: started. Internal metric-loss adapters, pair/retrieval metric helpers,
inference output response/JSON contracts, local-inference input parsing
contracts, minimal embedded embedding/pair-score routes, and blocked Studio
metadata completed 2026-06-26; graph/runtime integration remains open.

1. Add `PairMetrics`. Implemented internally by
   `ComputePairDistanceMetrics` in `metric_learning_metrics.h`.
2. Add `RetrievalMetrics`. Implemented internally by
   `ComputeRetrievalMetrics` in `metric_learning_metrics.h`.
3. Add `EmbeddingOutput`. Implemented internally by
   `BuildEmbeddingOutputResponse` in
   `metric_learning_inference_outputs.h`; JSON packaging implemented by
   `EmbeddingOutputResponseToJson`.
4. Add inference mode for embedding extraction. Internal response contract is
   implemented. Request payload parsing is implemented by
   `ParseMetricEmbeddingInferenceInput`; embedded local inference route
   integration is implemented by `LocalInferenceServer::HandleEmbeddings`.
5. Add inference mode for pair scoring. Internal response contract is
   implemented by `BuildPairScoreOutputResponse`; score-mode parsing and JSON
   packaging are implemented by `ParsePairScoreMode` and
   `PairScoreOutputResponseToJson`. Request payload parsing is implemented by
   `ParseMetricPairScoreInferenceInput`; embedded local inference route
   integration is implemented by `LocalInferenceServer::HandlePairScore`.

Metric-loss graph-executor adapter:

- `ComputePairMetricLoss` wraps backend `ContrastiveLoss` and
  `CosineEmbeddingLoss`, validates label conventions, and returns branch
  gradients for both embeddings
- `ComputeEuclideanTripletMetricLoss` wraps backend `TripletLoss` loss/anchor
  gradient and computes positive/negative branch gradients for the Euclidean
  triplet contract
- `RunPairMetricTrainingStep` and `RunTripletMetricTrainingStep` provide the
  internal executor-facing call surface for batch -> shared encoder -> metric
  loss -> branch backward -> optional update
- visual graph executor wiring is still open

## Definition Of Done

Siamese support is done when:

- a pair-based `.cyxgraph` loads without unknown-node errors
- the graph compiler accepts a shared encoder with two branches
- pair batches contain aligned `input_a`, `input_b`, and labels
- both branches reuse the same encoder parameters
- `ContrastiveLoss` trains without custom code outside the graph runtime
- a triplet `.cyxgraph` can train with anchor/positive/negative inputs
- metrics show positive and negative distance separation
- checkpoint includes graph, shared encoder weights, and embedding metadata
- inference can return embeddings for single samples
- inference can return a distance/similarity score for sample pairs

## Why This Matters

Siamese support is the first real metric-learning use case for CyxWiz.

It forces the backend and graph runtime to handle:

- multi-input training batches
- shared module reuse
- non-linear graph topology
- branch-aware gradients
- pairwise and triplet losses
- embedding-centric inference
- retrieval-oriented metrics

These capabilities will also help future tasks such as:

- face or signature verification
- duplicate detection
- semantic similarity
- few-shot classification
- recommendation embeddings
- retrieval and nearest-neighbor search

## Progress Log

### 2026-06-26 - Metric-learning batch contracts

Started the Siamese graph-runtime implementation path without enabling visual
graph execution:

- added `PairBatch` and `TripletBatch` as typed payloads separate from
  `Batch` and `SequenceBatch`
- added validation helpers for branch shape agreement, batch dimensions,
  pair-label shape, and optional sample/class metadata IDs
- recorded label-convention helpers for contrastive, cosine-embedding, and
  triplet losses
- added `IPairBatcher` and `ITripletBatcher` interfaces for the future builder
  slice
- added `test_metric_learning_batch_contracts`

### 2026-06-26 - Metric-learning in-memory batchers

Continued the same internal-only path:

- added `PairSample` and `TripletSample`
- added `PairBatcher` and `TripletBatcher` as in-memory emitters for the typed
  pair/triplet payloads
- batchers infer flat feature shape or accept an explicit shape, preserve
  optional sample/class metadata IDs, and support shuffle plus drop-last
  semantics
- constructors reject mismatched branch feature widths, mixed metadata
  presence, and pair labels that do not match the configured loss convention
- expanded `test_metric_learning_batch_contracts` to cover aligned tensor
  output, partial batches, drop-last behavior, metadata preservation, and
  invalid input rejection

This does not add `PairDatasetBuilder`, `TripletDatasetBuilder`, visual nodes,
or shared-encoder execution yet.

### 2026-06-26 - Metric-learning dataset builders

Closed the internal Phase 3 builder/batcher slice:

- added `PairDatasetRow` and `TripletDatasetRow`
- added `PairDatasetBuilder` and `TripletDatasetBuilder`
- pair builder validates feature widths, metadata consistency, label presence,
  and loss-specific label conventions
- pair builder can derive contrastive or cosine-embedding pair labels from
  class IDs when configured
- triplet builder validates anchor/positive/negative feature widths,
  metadata consistency, and optional class semantics
- build results create the in-memory pair/triplet batchers without touching
  Studio graph execution
- expanded `test_metric_learning_batch_contracts` to cover explicit labels,
  derived labels, triplet class validation, and build-result batcher handoff

This still does not add visual nodes, compiler acceptance, graph loss wiring,
or shared-encoder execution.

### 2026-06-26 - Metric-learning shared encoder runtime harness

Started Phase 4 without accepting visual Siamese graph execution:

- added `SharedEncoderRuntime`
- runtime owns exactly one `IExecutableModel`
- pair and triplet forwards route every branch through that same encoder object
- pair and triplet backward calls accumulate branch gradients into the same
  encoder
- parameter updates are delegated once per training batch
- added `test_metric_learning_shared_encoder_contracts`

This was still not graph acceptance. Later slices added the graph contract,
metric-loss adapters, output contracts, and deterministic `SequentialModel`
branch replay. Visual Siamese graph execution remains closed until graph
runtime routing exists.

### 2026-06-26 - Metric-learning metrics

Started Phase 5 without adding inference outputs or Studio nodes:

- added `metric_learning_metrics.h`
- added distance-threshold pair metrics for pair accuracy, positive distance
  mean, and negative distance mean
- pair metrics honor the contrastive convention (`0 = similar`,
  `1 = dissimilar`) and cosine-embedding convention (`1 = similar`,
  `-1 = dissimilar`)
- added retrieval metrics for recall@k, mean reciprocal rank, and nearest
  neighbor class agreement
- retrieval metrics consume embedding tensors plus class IDs and exclude the
  query sample from its own neighbor set
- added `test_metric_learning_metrics`

Remaining Phase 5 work after this slice was graph/runtime integration,
inference routing, and export metadata. Local embedding/pair-score routes were
added later in Track 14.

### 2026-06-26 - Metric-learning loss adapter contracts

Continued Phase 5 without wiring metric losses into Studio graph execution:

- added `metric_learning_losses.h`
- added `ComputePairMetricLoss` as the graph-executor-facing adapter for
  backend `ContrastiveLoss` and `CosineEmbeddingLoss`
- pair loss adapter validates contrastive labels (`0 = similar`,
  `1 = dissimilar`) separately from cosine-embedding labels (`1 = similar`,
  `-1 = dissimilar`)
- pair loss adapter returns loss plus gradients for both embedding branches
- added `ComputeEuclideanTripletMetricLoss` for loss, anchor gradient,
  positive gradient, and negative gradient in the Euclidean triplet contract
- added `test_metric_learning_losses`

This still does not accept visual metric-learning graphs. The graph executor
does not yet route branch embeddings into these helpers. Later slices added
deterministic sequential shared-encoder branch replay; stateful training-mode
modules still require activation snapshots.

### 2026-06-26 - Metric-learning inference output contracts

Continued Phase 5 without adding Studio nodes or local inference routes:

- added `metric_learning_inference_outputs.h`
- added `EmbeddingOutputResponse` and `EmbeddingOutputRecord`
- embedding output preserves per-sample embedding shape and optional
  sample/class metadata IDs
- added `PairScoreOutputResponse` and `PairScoreRecord`
- pair scoring supports Euclidean distance, negative Euclidean distance, and
  cosine similarity while always exposing raw distance separately
- pair score output preserves optional paired sample/class metadata IDs
- added score-mode parsing for `distance`, negative-distance, and
  cosine-similarity pair scoring
- added JSON packaging for `EmbeddingOutputResponse` and
  `PairScoreOutputResponse`
- added `test_metric_learning_inference_outputs`

This is still not end-to-end inference. The helpers define response contracts
and JSON payloads that graph/runtime or local-inference routes can call once
Siamese graph execution is accepted.

### 2026-06-26 - Metric-learning local inference input contracts

Continued Phase 5 without adding HTTP routes or accepting Studio graph
execution:

- added `metric_learning_inference_input.h/.cpp`
- added `ParseMetricEmbeddingInferenceInput` for embedding extraction payloads
- added `ParseMetricPairScoreInferenceInput` for pair-scoring payloads
- input tensors accept 1D single-sample arrays or 2D batch arrays
- optional sample/class metadata is parsed as batch-aligned `Int64` tensors
- pair scoring validates matched branch shapes, paired metadata presence, and
  explicit score-mode parsing
- added `test_metric_learning_inference_input`

This parser slice did not add an embedded endpoint by itself. It provided the
strict request contract used by the later local-inference route slice before
invoking `EmbeddingOutput` or `PairScoreOutput`.

### 2026-06-26 - Metric-learning embedded inference routes

Wired the internal metric-learning inference contracts into the embedded local
server without accepting visual Siamese graph execution:

- added `POST /v1/embeddings`
- added `POST /v1/pair-score`
- `/v1/embeddings` parses an embedding payload, runs the loaded model once,
  and serializes `EmbeddingOutputResponse`
- `/v1/pair-score` parses paired payloads, runs the loaded model once per side,
  and serializes `PairScoreOutputResponse`
- both routes preserve sample/class metadata and return latency
- `cyxwiz-engine` Debug build verifies the route integration compiles

This still does not make Studio Siamese graphs executable. The routes assume a
loaded model that already produces embeddings; graph compiler/runtime support
for building and training that model from visual Siamese nodes remains open.

### 2026-06-26 - Metric-learning training step contracts

Added the first internal graph-executor-facing metric-learning training step
adapter without accepting visual Siamese graph execution:

- added `metric_learning_training_step.h`
- added `RunPairMetricTrainingStep`
- added `RunTripletMetricTrainingStep`
- pair steps run a valid `PairBatch` through `SharedEncoderRuntime`, compute
  contrastive or cosine-embedding loss, backpropagate both branch gradients,
  and optionally perform exactly one parameter update
- triplet steps run a valid `TripletBatch`, compute Euclidean triplet loss,
  backpropagate anchor/positive/negative branch gradients, and optionally
  update once
- updates require an explicit optimizer
- added `test_metric_learning_training_step`

This is still an internal runtime contract. The Studio graph executor does not
yet route visual `SharedEncoder`, branch, and metric-loss nodes into these
helpers. Deterministic `SequentialModel` shared-branch backward is supported by
branch replay and accumulated gradient maps; training-mode stateful modules
still require activation snapshots.

### 2026-06-26 - Metric-learning passive compiler graph contract

Advanced Phase 4 without enabling visual Siamese graph execution:

- added `MetricLearningGraphContract`
- `TrainingConfiguration` now exposes a passive `metric_learning_graph`
  contract when the selected compiler path contains metric-learning nodes,
  legacy sketch names, or pair/triplet column parameters
- the contract records detected pair/triplet/shared-encoder/output node IDs,
  infers pair-training, triplet-training, embedding-export, or pair-scoring
  intent, and keeps `executable=false`
- blockers are now machine-readable for missing shared encoder ownership,
  missing pair/triplet branches, missing visual output routing, missing
  visual metric-loss executor wiring, and missing visual shared-encoder graph
  execution
- `test_graph_compiler_deferred_nodes` now verifies the compiler output for
  selected shared-encoder, pair-score, and triplet sketches

This does not accept metric-learning training. The compiler still rejects the
selected graph path until visual branch-aware execution and metric-loss wiring
are implemented.

### 2026-06-26 - SequentialModel deterministic branch replay

Opened real `SequentialModel` shared-branch backward for deterministic
encoders without accepting visual Siamese graph execution:

- `SharedEncoderRuntime` caches pair/triplet branch inputs during forward
- sequential pair/triplet backward replays each branch forward immediately
  before that branch's backward pass
- parameter-gradient maps from each branch are summed in the runtime so
  backend modules that reset gradients per backward still contribute all
  branches
- `GetGradients` exposes the accumulated branch gradients
- `UpdateParameters` applies one optimizer step with the accumulated gradient
  map and clears it
- training-mode stateful modules such as `Dropout` and `BatchNorm` remain
  rejected until branch activation snapshots exist
- `test_metric_learning_shared_encoder_contracts` verifies accumulated
  sequential linear gradients and the stateful replay guard

This still does not make visual Siamese graphs executable. It removes the
blanket real-`SequentialModel` blocker for deterministic internal shared
encoders and keeps the snapshot requirement for stateful/stochastic training
paths.

### 2026-06-26 - SequentialModel branch-backward guard

Added a runtime safety guard for the remaining activation-snapshot gap:

- `SharedEncoderRuntime` still allows generic executable test doubles to prove
  branch routing and gradient accumulation
- when the shared encoder wraps a real `SequentialModel`,
  `BackwardPair` / `BackwardTriplet` threw before using overwritten module
  activation caches
- `test_metric_learning_shared_encoder_contracts` verified that pair forward
  still worked for a sequential encoder, but pair backward required branch
  activation snapshots

This did not implement snapshots. It prevented silent wrong-gradient behavior
until deterministic replay narrowed the guard; stateful/stochastic sequential
modules still require real per-branch forward-state snapshots.

### 2026-06-26 - Metric-learning blocked Studio node metadata

Added explicit Studio node types for the target Siamese / metric-learning
surface without enabling graph execution:

- added blocked `Template` metadata for `PairDatasetBuilder`,
  `TripletDatasetBuilder`, `SharedEncoder`, `SiameseBranch`,
  `ContrastiveLoss`, `CosineEmbeddingLoss`, `TripletLoss`, `PairMetrics`,
  `RetrievalMetrics`, `EmbeddingOutput`, and `PairScoreOutput`
- added save/load, pattern-library, CyxQL, icon, and default-pin coverage for
  the blocked node surface
- extended the graph compiler guard to reject these concrete node types even
  if their display names are changed
- kept the metric-learning graph executor, metric-loss wiring, and local
  inference routes closed at that point
- added regression coverage in `test_pipeline_operator_metadata` and
  `test_graph_compiler_deferred_nodes`

This closes the metadata/compiler-rejection slice only. Visual shared-encoder
Siamese graph execution is still blocked on visual branch-aware graph
compilation and metric-loss execution. Local embedding/pair-score inference
routes were added later.

### 2026-06-26 - Siamese graph runtime contract

Closed Track 14 Phase 5 as a design-only step after the backend contrastive
smoke passed:

- chose typed `PairBatch` and `TripletBatch` payloads instead of overloading
  `Batch.data`
- recorded label conventions for contrastive, cosine-embedding, and triplet
  losses
- defined `SharedEncoder` ownership as one encoder module/parameter set
  referenced by branch nodes
- required branch gradients to accumulate into that shared parameter set before
  a single optimizer step
- separated embedding inference (`EmbeddingOutput`) from pair scoring
  (`PairScoreOutput`)
- narrowed the next implementation slice to internal batch contracts and tests
  before Studio node/runtime expansion

The compiler should continue rejecting selected Siamese graph sketches until
those typed batch and shared-weight execution contracts are implemented.

### 2026-06-26 - Siamese backend smoke

Added the first narrow metric-learning proof without opening Studio graph
support:

- added `test_siamese_backend_smoke`
- trains a manually shared linear encoder on a tiny pair dataset
- uses backend `ContrastiveLoss` with the existing label convention
  (`0 = similar`, `1 = dissimilar`)
- verifies finite loss, strong loss reduction, positive pairs staying close,
  and negative pairs moving out to the configured margin

The remaining Siamese work is still graph/runtime design: typed pair/triplet
batches, shared encoder ownership, branch-aware gradients, pair/triplet graph
loss nodes, metrics, and embedding/pair-score inference outputs.

### 2026-06-26 - First-class SequenceTagOutput surface

Closed the remaining Track 14 Phase 3 NER output-node work:

- added `SequenceTagOutput` as a real Studio node type with metadata, pins,
  defaults, icon/category, property editor, shape handling, save/load,
  pattern import, CyxQL naming, and codegen marker behavior
- treated `SequenceTagOutput` as a terminal output contract in the compiler,
  model analyzer, model builder, and test executor instead of adding a
  trainable layer
- updated CrossEntropy class-count inference to read
  `SequenceTagOutput.num_tags`
- updated the saved NER graph so node 18 is serialized as
  `SequenceTagOutput`
- added regression coverage in the graph compiler, saved graph guard, and
  metadata tests

Remaining NER follow-up: validate `examples/cyxgraph/NER/ner_inference.py`
against a real deployed model endpoint. The core Track 14 remainder now moves
to the Siamese backend smoke/design phases.

### 2026-06-26 - Sequence POS feature fusion

Added the Track 14 Phase 2 POS-fusion path without broadening the graph
runtime into arbitrary multi-input execution:

- added backend `SequenceFeatureFusionModule`
- added `BuildSequenceModelInput` to pack word/POS IDs only for configs that
  declare `sequence_feature_fusion=true`
- updated sequence training and validation loops to use the packed input when
  fusion is enabled
- carried POS vocabulary size through the Arrow sequence batcher build result
- updated the saved NER smoke to train the POS-fused path
- added smoke assertions that missing/mismatched POS IDs fail before forward
  and that changing POS IDs changes model logits

Generic `FeatureConcat` is still not a first-class runtime graph node, and
attention-mask consumption remains future sequence work.

### 2026-06-26 - Sequence local inference/importer surface

Started Track 14 Phase 3 by tightening the deployed/local sequence inference
path:

- `ModelImporter` now reuses `BuildSequentialFromConfig` when rebuilding a
  packaged graph model, so import understands the same sequence modules as
  training instead of a stale duplicate layer switch.
- `LocalInferenceServer` now preserves named `input.pos_ids`.
- POS-fused sequence models pack `word_ids` and `pos_ids` before local
  inference forward, matching the training path.
- Sequence asset coverage now checks the shared packing helper and
  `SequenceFeatureFusionModule` detection.
- Local model loading now fails clearly when a package declares sequence tag
  vocabulary metadata but the tag vocabulary is missing or empty.

Still open: decide the first-class `SequenceTagOutput` Studio surface.

### 2026-06-26 - Saved NER sequence smoke

Added `test_saved_ner_sequence_smoke` as the Track 14 Phase 1 proof:

- loads `examples/cyxgraph/NER/ner_bilstm_sequence_tagger.cyxgraph`
- compiles the graph and verifies the sequence batch contract
- preserves saved sequence padding and token-loss ignore settings
- loads `examples/cyxgraph/NER/generated/ner_sentences.csv`
- builds the Arrow-backed `ISequenceBatcher`
- normalizes embedding vocabulary size and TimeDistributed tag width
- trains one tiny sequence pass through `TrainingExecutor`
- packages sequence vocab assets into `.cyxmodel`
- decodes sample sequence logits through the inference helper

This was originally word-ID-only. Track 14 Phase 2 later updated the smoke to
train through explicit word/POS sequence feature fusion.


### 2026-06-07 - NER target-design compile guard

Added the first truth guard from `tofix19.md`: the graph compiler now
rejects selected training paths where intended NER/sequence task nodes
are encoded as generic `Dense` nodes with custom target-design
parameters. This prevents the NER example from being silently interpreted
as a trainable Dense graph.

This does not implement NER training. The missing sequence contracts in
this file remain open: typed sequence batches, token/POS/tag
vocabularies, sequence padding, token classifier head,
`ignore_index`-aware token cross-entropy, and BIO/entity metrics.

### 2026-06-02 - Python tensor exposure slice 1

Started the exposure lane after `tofix13` completed backend C++ tensor
parity.

First, split the Python binding surface so tensor bindings no longer
grow the already-large `python/bindings.cpp` file:

- added `python/bindings_tensor.h`
- added `python/bindings_tensor.cpp`
- changed `python/bindings.cpp` to call `BindTensor(m)`
- added `python/bindings_tensor.cpp` to the `pycyxwiz` CMake target

Then exposed the migrated tensor API through `pycyxwiz.Tensor`,
including shape operations, indexing/slicing, concat/split/chunk,
reductions with optional `dim` and `keepdim`, scalar/tensor math,
comparison and logical masks, broadcasting, `dot`, `batch_matmul`, and
`range_n`.

Added `examples/python_tests/test_tensor_parity_bindings.py` as a
focused smoke test for the new Python tensor surface. Verification:

- `pycyxwiz` Debug target builds
- `pycyxwiz` Release target builds for normal Python 3.12 import
- Python tensor parity smoke test passes with Python 3.12
- C++ tensor test suite passes with 52/52 tests
- encoding scan is clean for touched Python binding and tofix files

### 2026-06-02 - Studio tensor frontend exposure slice 1

Added the first Studio-facing frontend catalog for the tensor parity
work without pretending the graph runtime can execute every operation
yet.

The Node Browser/search template loader now preserves template category
names beyond the original cloud/database/reporting set and parses JSON
pin `"type"` values into real Studio pin types. This matters for tensor
templates because `"Tensor"` ports now show as tensor ports instead of
being forced to dataset ports.

Added small operation-family template catalogs under
`cyxwiz-engine/resources/node_templates`:

- `tensor_shape_nodes.json`
- `tensor_reduction_nodes.json`
- `tensor_math_nodes.json`
- `tensor_linalg_mask_nodes.json`

These templates are marked `Python Ready` because the PyCyxWiz tensor
API is exposed and smoke-tested. They remain Studio templates until the
next runtime slice adds concrete graph execution nodes/adapters.

Also registered the existing concrete Studio shape/merge node types in
`NodeMetadataRegistry` so they are discoverable through the normal
implemented-node add menus, not only through templates:

- `Reshape`, `View`, `Permute`, `Squeeze`, `Unsqueeze`, `Split`
- `Concatenate`, `Add`, `Multiply`, `Average`

The template catalogs now avoid duplicating those concrete shape nodes;
template-only coverage starts with operations that do not yet have real
Studio node types, such as `Tensor Index Select`, reductions, broadcast,
elementwise math, linalg, comparisons, and logical masks.

### 2026-06-02 - PyCyxWiz tensor code export slice 1

Tightened the Studio PyCyxWiz exporter for the concrete tensor shape and
merge nodes that were exposed in the registry.

Shape operations now export actual `pycyxwiz.Tensor` method calls in the
generated `forward()` path:

- `Flatten` -> `x.flatten()`
- `Reshape` / `View` -> `x.view([...])`
- `Permute` -> `x.permute([...])`
- `Squeeze` -> `x.squeeze()` or `x.squeeze(dim)`
- `Unsqueeze` -> `x.unsqueeze(dim)`

The exporter no longer emits fake layer objects such as `cx.Flatten()`,
`cx.Reshape()`, `cx.Add()`, or `cx.Concatenate()` for tensor operations
that are not layer classes in the Python binding. Multi-output and
multi-input tensor graph export (`Split`, `Concatenate`, `Add`,
`Multiply`, `Average`) is now left as an explicit generated-code comment
until the graph adapter supports those shapes.

### 2026-06-02 - PyCyxWiz tensor graph export slice 2

Refined the PyCyxWiz `forward()` generator from a mostly sequential
`x = ...` path into a small graph-aware variable emitter for the
concrete tensor operations already registered in Studio.

The exporter now names intermediate node outputs, resolves incoming
links by input pin, and tracks output pins so multi-output nodes can be
referenced precisely. This enables generated PyCyxWiz code for:

- `Split` through `tensor.split(split_size, dim)` with output-pin-aware
  downstream references
- `Concatenate` through `cx.Tensor.cat([...], dim)`
- `Add` through chained `+`
- `Multiply` through chained `*`
- `Average` through summed inputs divided by input count

This is still code export, not the local C++ graph executor. The local
runtime path should remain a separate adapter slice so Studio execution,
code export, and backend tensor APIs do not get mixed together.

### 2026-06-02 - Studio tensor reduction nodes slice 1

Promoted tensor reductions from template-only catalog entries to real
Studio node types because they are single-input/single-output and fit
the graph-aware PyCyxWiz exporter added above.

Added concrete nodes:

- `TensorSum`
- `TensorMean`
- `TensorMax`
- `TensorMin`
- `TensorProd`
- `TensorVar`
- `TensorStd`

Each node has tensor input/output pins plus `dim` and `keepdim`
parameters. PyCyxWiz export now emits the matching `Tensor` method calls
and converts Studio boolean parameters into valid Python `True`/`False`
literals. The old reduction template catalog was removed to avoid
duplicate template/implemented entries in the frontend.

### 2026-06-02 - Studio tensor unary/broadcast nodes slice 1

Promoted the remaining single-input tensor math and broadcast templates
to concrete Studio node types:

- `TensorBroadcastTo`
- `TensorExpand`
- `TensorPow`
- `TensorSqrt`
- `TensorExp`
- `TensorLog`
- `TensorAbs`
- `TensorSign`
- `TensorClip`

Each node has tensor input/output pins, registry metadata, graph load
mapping, and PyCyxWiz export through the corresponding `Tensor` method.
The old overloaded math template catalog was removed so users see real
implemented nodes instead of duplicate template entries.

### 2026-06-02 - Studio tensor linalg/index/mask nodes slice 1

Promoted the final tensor template catalog entries to concrete Studio
node types:

- `TensorDot`
- `TensorBatchMatMul`
- `TensorCompare`
- `TensorLogicalMask`
- `TensorIndexSelect`

These nodes now have pins, default parameters, registry metadata, graph
load mapping, and PyCyxWiz export. `TensorCompare` supports tensor or
scalar comparison through the existing Python comparison operators.
`TensorLogicalMask` exports `&`, `|`, or `~` depending on the selected
operator. `TensorIndexSelect` exports `index_select(dim, indices)`.

After this slice, the tensor operation frontend is no longer exposed as
template-only JSON catalogs. The remaining Studio gap is local graph
runtime execution in C++ for these tensor operation nodes; code export
and node palette exposure are now separate from that runtime adapter.

### 2026-06-02 - Tensor frontend exposure closure

Audited the promoted tensor nodes after removing the template catalogs.
Added the missing non-runtime mappings so tensor operation nodes can
round-trip through pattern import, show stable names in CyxQL, and land
in the documentation/category panels as Tensor Operations.

The tensor exposure lane from `tofix13` is now complete for:

- C++ backend tensor API
- Python binding exposure
- Python smoke coverage
- Studio node palette exposure
- Studio graph save/load mapping
- Studio pattern/CyxQL naming
- PyCyxWiz graph-aware code export

Do not mix the remaining local Studio runtime work into this completed
exposure thread. The local C++ training/debug runtime is still built
around `TrainingConfiguration -> SequentialModel`; it does not execute
arbitrary tensor graph nodes. A future runtime adapter should be tracked
as separate backend work in this file, with explicit scope for tensor
node materialization, multi-input/multi-output value routing, and
runtime validation.
