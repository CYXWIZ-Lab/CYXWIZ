# To Fix 14 - NER Sequence Tagging and Siamese Engine Gaps

This note documents the gap between advanced graph designs and what the
current CyxWiz engine can train today.

The two immediate advanced graph families are:

- NER sequence tagging
- Siamese / metric-learning networks

Both are useful because they expose the same underlying limitation: the
backend has several low-level neural-network pieces, but the Studio graph
runtime still assumes a mostly single-input sequential training path.

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

Purpose:

- map words to IDs
- support `[PAD]` and `[UNK]`
- save/load vocab files
- optionally lowercase tokens

Why needed:

NER needs a word vocabulary independent of POS and tag vocabularies.

### 3. `POSVocabulary`

Purpose:

- map POS tags like `NNP`, `VBD`, `IN` to IDs
- support `[PAD]` and `[UNK]`

Why needed:

The graph optionally uses POS embeddings as extra token features.

### 4. `NERTagVocabulary`

Purpose:

- map BIO tags to class IDs
- reserve `0` for `[PAD]`
- keep `O` as the outside/non-entity class
- save/load tag order for inference decoding

Why needed:

The loss and output decoder must agree on the exact tag index order.

### 5. `SequencePadding`

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

### 9. `TokenCrossEntropyLoss`

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

### Phase 1 - Data preparation and schema

1. Keep `prepare_ner_demo.py` as the dataset converter.
2. Generate:
   - `ner_sentences.csv`
   - `ner_word_vocab.txt`
   - `ner_pos_vocab.txt`
   - `ner_tag_vocab.txt`
   - `ner_metadata.json`
3. Add graph/runtime support for sentence-level sequence samples.

### Phase 2 - Sequence tensors

1. Implement `NERSequenceBuilder`.
2. Implement or adapt `TokenVocabulary`.
3. Implement `NERTagVocabulary`.
4. Implement `SequencePadding`.
5. Add `attention_mask` and `loss_mask`.

### Phase 3 - Sequence model path

1. Verify `Embedding` accepts `[batch, seq_len]`.
2. Verify LSTM/GRU can return `[batch, seq_len, hidden]`.
3. Add `FeatureConcat`.
4. Add `TimeDistributedDense`.

### Phase 4 - Token-level training

1. Add `TokenCrossEntropyLoss`.
2. Add mask-aware reduction.
3. Add token-level accuracy.
4. Add entity-level precision/recall/F1.

### Phase 5 - Inference and export

1. Save vocab and metadata with `.cyxmodel`.
2. Add `SequenceTagOutput`.
3. Make embedded inference accept named sequence inputs.
4. Make embedded inference return token logits.
5. Verify `ner_inference.py` against a deployed model.

## Definition Of Done

NER support is done when:

- `ner_bilstm_sequence_tagger.cyxgraph` loads without unknown-node errors
- the graph compiles
- the data prep files are consumed by the graph
- batches contain aligned word IDs, POS IDs, tag IDs, and masks
- BiLSTM returns per-token hidden states
- token classifier outputs `[batch, seq_len, num_tags]`
- loss ignores padding
- metrics report token accuracy and entity F1
- checkpoint includes graph, weights, vocabularies, metadata, and max length
- `ner_inference.py` returns readable token/tag predictions for a sentence

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

### 1. Multi-input batch contract

The current training path should grow a typed batch representation for
advanced models instead of overloading `Batch.data` with packed tensors.

Minimum useful shape:

```text
Batch.inputs: map<string, Tensor>
Batch.labels: map<string, Tensor>
```

or an equivalent small typed struct for pair/triplet training.

### 2. Shared-weight execution

The runtime must call the same encoder object for multiple inputs and
accumulate gradients into the same parameter set.

The optimizer must update shared parameters once per training step.

### 3. Branch-aware backward pass

For contrastive/cosine losses, gradients must flow back through both
encoder calls:

```text
loss gradient -> embedding_a -> shared encoder
loss gradient -> embedding_b -> shared encoder
```

For triplet loss, gradients must flow through anchor, positive, and
negative branches.

### 4. Graph validation

Compile should reject Siamese graphs when:

- branch encoders are duplicated instead of shared
- embedding dimensions do not match
- pair labels are missing
- triplet negative input is missing
- loss label convention is ambiguous
- optimizer cannot reach the metric loss

### 5. Inference contract

Siamese inference should support at least two modes:

```text
single sample -> embedding
sample pair   -> distance/similarity score
```

This should not be forced through a class-probability output.

## Implementation Order

### Phase 1 - Backend smoke path

1. Add a small C++ or Python Siamese example using existing backend
   losses.
2. Build a shared encoder manually in code.
3. Train on a toy pair dataset.
4. Verify similar pairs move closer and dissimilar pairs move apart.

### Phase 2 - Graph node surface

1. Add `ContrastiveLoss`, `CosineEmbeddingLoss`, and `TripletLoss` node
   types.
2. Register node metadata and documentation.
3. Add graph load/save aliases for those node types.
4. Add compiler loss mapping for metric losses.

### Phase 3 - Pair and triplet batching

1. Add `PairDatasetBuilder`.
2. Add `PairBatcher`.
3. Add `TripletDatasetBuilder`.
4. Add `TripletBatcher`.
5. Add tests for label conventions and shape contracts.

### Phase 4 - Shared encoder graph execution

1. Add a minimal `SharedEncoder` compiler/runtime representation.
2. Execute the same encoder object for every branch.
3. Accumulate branch gradients into shared parameters.
4. Update shared parameters once per batch.

### Phase 5 - Metrics and inference

1. Add `PairMetrics`.
2. Add `RetrievalMetrics`.
3. Add `EmbeddingOutput`.
4. Add inference mode for embedding extraction.
5. Add inference mode for pair scoring.

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
