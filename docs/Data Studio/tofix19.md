# Done 19 - Sequence Tagging Training Path

**Created:** 2026-06-07
**Completed:** 2026-06-10
**Source:** Follow-up from the NER example audit in
`examples/cyxgraph/NER/ner_bilstm_sequence_tagger.cyxgraph`.

## Status

Done. The `tofix19` planned slice has been implemented and pushed in
commit `c543e504`.

## Boundary

This file documents model families that PyTorch can train but the current
CyxWiz engine cannot train truthfully end to end yet.

This is not a request to copy PyTorch wholesale. The smallest useful goal is
to make CyxWiz honest and extensible:

- keep supported graphs trainable and tested,
- fail closed for unsupported model families,
- add missing contracts one at a time,
- avoid adding frontend nodes before the backend can execute them.

This overlaps with `tofix14.md` for NER/Siamese details, `tofix16.md` for
performance/deferred transformer work, and `tofix17.md` for GPU execution.
Use this file as the broader model-family map.

---

## What CyxWiz Can Train Today

The current training path is strongest for:

```text
one input tensor -> mostly sequential model -> one prediction tensor -> one loss
```

Confirmed or directionally supported pieces:

- dense MLP-style regression and classification,
- tabular/image/audio/text datasets once materialized into one feature tensor
  and one target tensor,
- whole-text classification through tokenization/vocabulary/padding,
- `Embedding -> GRU/LSTM -> Dense` text classifiers,
- `Embedding -> TransformerEncoder -> Dense` text classifiers,
- time-series windowing into fixed-width supervised samples,
- basic activations, dropout, batch norm, shape ops, reductions, and some
  graph fan-in tensor ops,
- losses such as MSE, cross-entropy, BCE, BCE-with-logits, L1, SmoothL1/Huber,
  and NLL,
- optimizers including SGD, Adam, AdamW, RMSprop, Adagrad, and NAdam.

Important limitation:

The compiler and metadata expose more than this. A node being present in the
UI or metadata does not mean that the training path can execute it end to end.

---

## PyTorch Baseline

PyTorch can train these families because it gives the user:

- arbitrary `nn.Module.forward(...)` signatures,
- multiple named inputs and outputs,
- dict/list/tuple batch payloads from `Dataset` and `DataLoader`,
- custom losses over arbitrary tensor shapes,
- shared modules and tied weights,
- dynamic control flow in `forward`,
- direct tensor reshaping/masking/indexing inside the model,
- pretrained model import and fine-tuning through ordinary modules.

CyxWiz currently has lower-level tensor and layer pieces, but the Studio
training contract is narrower. Most unsupported families below need contracts,
not just individual layer classes.

---

## Unsupported Or Not Truthfully Trainable Families

### 1. Token Classification / NER / Sequence Tagging

PyTorch pattern:

```text
Dataset -> word_ids, optional pos_ids, attention_mask, tag_ids
Embedding -> BiLSTM/TransformerEncoder -> Linear per token
CrossEntropyLoss(ignore_index=pad_tag)
```

CyxWiz gap:

- first-class `NERSequenceBuilder` now exists as a training contract node,
  and the compiler extracts its sequence batch contract,
- typed sequence batches with `word_ids`, optional `pos_ids`,
  `attention_mask`, and `tag_ids` are materialized by the tabular
  Arrow/Parquet runtime path,
- `TimeDistributedDense` token classifier support exists in the backend,
  ModelBuilder, compiler validation, and executor path,
- token-shaped CrossEntropy/NLL with `ignore_index` is wired for sequence
  training, including the CPU backward path that avoids the ArrayFire 3D
  CrossEntropy warning,
- token accuracy and BIO entity F1 are produced by the sequence executor and
  surfaced in the Studio training plot panel,
- `TokenVocabulary`, `POSVocabulary`, and `NERTagVocabulary` are still
  contract/documentation nodes, not executable graph runtime nodes,
- placeholder sequence designs encoded as Dense nodes are rejected at import
  or compile time rather than silently training as ordinary Dense layers.

Concrete example:

`examples/cyxgraph/NER/ner_bilstm_sequence_tagger.cyxgraph` has been updated
away from stale serialized node ids and Dense-encoded sequence placeholders.
It now uses first-class sequence contract nodes where they exist. The remaining
non-executable vocabulary nodes are still kept out of the selected training
path until graph-runtime vocabulary execution is wired.

### 2. Sequence-To-Sequence And Generative Text

PyTorch pattern:

```text
Encoder/decoder or decoder-only transformer
causal mask
shifted token targets
teacher forcing
token-level loss
sampling/generation loop
```

CyxWiz gap:

- no trainable `TransformerDecoder` path,
- no causal language-model objective,
- no shifted-token target materializer,
- no attention mask contract in training,
- no generation/sampling inference loop,
- tokenizer and vocabulary are not packaged as first-class deployment assets
  for generative text.

Affected model families:

- GPT-style language models,
- seq2seq translation,
- summarization,
- instruction-tuning,
- text generation from prompts.

### 3. CNN Vision Training

PyTorch pattern:

```text
Conv2d -> BatchNorm2d -> ReLU -> Pool -> Conv blocks -> classifier
```

CyxWiz gap:

- CNN/pooling/upsampling nodes are intentionally blocked in training graphs
  until `ModelBuilder` and `SequentialModel` can build them,
- no canonical CNN module wrappers in the current train path,
- no verified image tensor shape contract across `[N,C,H,W]` / `[N,H,W,C]`,
- no end-to-end tests for convolutional training.

Affected model families:

- MNIST/CIFAR CNNs,
- ResNet/VGG/MobileNet-style classifiers,
- image autoencoders,
- image segmentation backbones.

### 4. Object Detection And Instance/semantic Segmentation

PyTorch pattern:

```text
image -> boxes/classes/masks
multi-head model outputs
classification + box regression + mask losses
custom target dicts per sample
NMS/evaluation metrics
```

CyxWiz gap:

- DNN inference nodes are not a trainable detection path,
- no target schema for boxes, masks, areas, image ids, or variable object
  counts,
- no multi-head loss aggregation,
- no detection metrics such as mAP,
- no trainable detection heads or anchor/proposal contracts.

Affected model families:

- YOLO training,
- Faster R-CNN,
- Mask R-CNN,
- U-Net/DeepLab-style segmentation.

### 5. Multi-Input / Shared-Weight Metric Learning

PyTorch pattern:

```text
shared_encoder(anchor), shared_encoder(positive), shared_encoder(negative)
TripletLoss or ContrastiveLoss
```

CyxWiz gap:

- no typed multi-input batch contract for anchor/positive/negative or pair
  inputs,
- no explicit shared-module/tied-weight graph representation,
- low-level triplet/contrastive losses exist, but Studio training does not
  expose the full pair/triplet data flow,
- no pair/triplet mining or sampling path.

Affected model families:

- Siamese networks,
- face/person re-identification,
- metric learning,
- contrastive representation learning.

### 6. Multi-Task And Multi-Head Models

PyTorch pattern:

```text
shared backbone -> head A + head B + ...
loss = weighted_sum(loss_A, loss_B, ...)
```

CyxWiz gap:

- current training path expects one prediction tensor and one loss,
- graph executable support is limited to a small set of fan-in tensor ops,
- no loss aggregation node with named outputs,
- no per-head metrics or output packaging.

Affected model families:

- classification + regression heads,
- joint sentiment + topic models,
- detection-style multi-loss models,
- multitask tabular/text models.

### 7. Autoencoders, VAEs, GANs, And Diffusion Models

PyTorch pattern:

```text
custom forward path
multiple losses or alternating optimizers
sampling/noise schedule
encoder-decoder reconstruction targets
```

CyxWiz gap:

- no alternating optimizer/training-step contract,
- no latent distribution/KL-loss contract for VAEs,
- no generator/discriminator training orchestration,
- no diffusion scheduler/noise-prediction training loop,
- no image decoder/generation output path.

Affected model families:

- autoencoders,
- variational autoencoders,
- GANs,
- diffusion models.

### 8. Graph Neural Networks

PyTorch pattern:

```text
node_features, edge_index, edge_features, graph_batch
message passing layers
node/edge/graph-level losses
```

CyxWiz gap:

- no graph dataset schema,
- no sparse adjacency / edge-index tensor contract,
- no message-passing layer family,
- no node/edge/graph-level batching semantics.

Affected model families:

- GCN/GAT/GraphSAGE,
- molecular graph models,
- fraud/network graph models.

### 9. Reinforcement Learning

PyTorch pattern:

```text
environment loop
policy/value networks
rollout buffer
policy loss + value loss + entropy
target networks or replay buffers
```

CyxWiz gap:

- RL node metadata exists, but the training executor is supervised-learning
  oriented,
- no environment stepping loop as the primary training driver,
- no rollout/replay sampling contract,
- no RL-specific metrics, checkpoints, or evaluation loop.

Affected model families:

- DQN,
- PPO,
- actor-critic,
- policy-gradient agents.

### 10. Pretrained Model Import And Fine-Tuning

PyTorch pattern:

```text
load pretrained checkpoint
map parameters into module graph
freeze/unfreeze layers
attach task head
fine-tune with compatible tokenizer/preprocessor
```

CyxWiz gap:

- no first-class pretrained transformer import,
- no ONNX/checkpoint-to-training-graph conversion contract,
- no layer freezing controls in Studio training,
- no tokenizer compatibility layer for pretrained NLP models,
- no parameter mapping/shape validation workflow.

Affected model families:

- BERT/RoBERTa/DistilBERT fine-tuning,
- pretrained vision backbones,
- transfer learning classifiers,
- embedding-model fine-tuning.

---

## Cross-Cutting Missing Contracts

These are the shared blockers behind most unsupported families:

1. **Typed multi-input batches**
   - Need named tensors, not only `Batch.data` and `Batch.labels`.

2. **Structured outputs**
   - Need model outputs that can be dicts/tuples or multiple named tensors.

3. **Loss shape contracts**
   - Need token, detection, segmentation, pair/triplet, and multi-head losses
     with explicit target schemas.

4. **Module graph execution**
   - Need non-linear module graphs, shared modules, tied weights, and multiple
     inputs/outputs beyond the current mostly sequential path.

5. **Dataset schemas**
   - Need schemas for sequence labels, boxes, masks, graph edges, pairs,
     triplets, and RL transitions.

6. **Metrics and debug traces**
   - Need metrics that match the task, not generic accuracy for everything.

7. **Deployment packaging**
   - Need vocab/tokenizer/preprocessor/label metadata saved with the model.

---

## Missing Algorithm And Backend Implementation Backlog

This section is the implementation map. It separates missing algorithms from
the surrounding contracts needed to train them truthfully. A layer or loss
implemented in isolation does not make the model family supported until the
batch schema, compiler path, executor path, metrics, and packaging are tested
end to end.

### Sequence Tagging / NER

Implement:

- first-class `TokenVocabulary`, `POSVocabulary`, and `NERTagVocabulary`
  metadata/runtime nodes,
- `NERSequenceBuilder` that materializes `word_ids`, optional `pos_ids`,
  `attention_mask`, and `tag_ids`,
- label padding and `ignore_index` preservation from graph config into the
  loss builder,
- `TimeDistributedDense` per-token projection over `[batch, seq, hidden]`
  is implemented as a backend/module and `ModelBuilder` layer,
- token cross-entropy over `[batch, seq, tags]`,
- token accuracy plus BIO entity precision/recall/F1,
- public `TrainingExecutor` support for prebuilt `ISequenceBatcher` payloads,
- compiler and executor tests that train one tiny NER graph end to end through
  the public Studio launch path.

### Decoder / Generative Text

Implement:

- trainable `TransformerDecoder` module integration in `ModelBuilder`,
- causal mask construction and propagation through the training batch,
- shifted-token target materializer for decoder-only and seq2seq objectives,
- token-level language-model loss with padding ignore support,
- generation/sampling inference loop separate from supervised training,
- tokenizer/vocabulary packaging in exported model artifacts,
- smoke tests for a tiny causal LM and a tiny encoder-decoder graph.

### CNN Vision

Implement:

- trainable `Conv1D`, `Conv2D`, and eventually `Conv3D` module wrappers,
- pooling/global-pooling wrappers and verified flatten/global-pool handoff,
- image tensor layout contract with explicit `[N,C,H,W]` or `[N,H,W,C]`
  conversion rules,
- shape inference for convolution, pooling, padding, stride, dilation, and
  channel order,
- CNN optimizer/loss smoke tests for MNIST/CIFAR-sized graphs.

### Detection And Segmentation

Implement:

- target schemas for boxes, classes, masks, areas, image ids, and variable
  object counts,
- detection/segmentation heads with named outputs,
- loss aggregation for classification, box regression, objectness, and mask
  losses,
- NMS and mAP/IoU metrics as task-specific evaluation paths,
- dataset adapters for COCO/YOLO/VOC-style annotations,
- tiny detection/segmentation compile and training smoke tests.

### Multi-Input / Shared-Weight / Metric Learning

Implement:

- named multi-input batch payloads for pairs and triplets,
- graph representation for shared modules and tied weights,
- selected-path compiler support for one module reused across multiple inputs,
- pair/triplet sampling or mining contracts,
- contrastive/triplet loss shape validation connected to named inputs,
- smoke tests for Siamese and triplet graphs.

### Multi-Task / Multi-Head

Implement:

- structured model outputs with named tensors,
- loss aggregation node with explicit weights and per-loss target schema,
- per-head metric routing and logging,
- graph-plan support for multiple prediction edges feeding multiple losses,
- tests proving two-head graphs optimize both heads instead of silently using
  one loss.

### Autoencoders / VAEs / GANs / Diffusion

Implement:

- reconstruction-target routing for autoencoders,
- latent mean/logvar outputs and KL-loss contract for VAEs,
- alternating optimizer/training-step orchestration for GANs,
- diffusion noise scheduler, timestep conditioning, and noise-prediction loss,
- generation/decoder output packaging for image synthesis,
- tests for one tiny autoencoder before attempting VAE/GAN/diffusion work.

### Graph Neural Networks

Implement:

- graph dataset schema with node features, edge indices, edge features, and
  graph batch ids,
- sparse adjacency/edge-index tensor handling,
- message-passing module family such as GCN/GAT/GraphSAGE,
- node-level, edge-level, and graph-level loss/metric routing,
- batching tests for multiple small graphs in one batch.

### Reinforcement Learning

Implement:

- environment stepping loop as the primary training driver,
- rollout/replay buffer schemas,
- policy/value model output contracts,
- RL losses such as policy gradient, value loss, entropy bonus, and target
  network updates,
- checkpoint/evaluation metrics for episodic returns,
- tiny DQN/PPO-style tests once the driver contract exists.

### Pretrained Import / Fine-Tuning

Implement:

- model-import to training-graph conversion for supported formats only,
- parameter-name and shape mapping into trainable modules,
- freeze/unfreeze ownership at module/parameter granularity,
- optimizer-state compatibility checks for resumed training,
- tokenizer/preprocessor compatibility and artifact packaging,
- tests that import a tiny saved model, attach a head, freeze the base, and
  fine-tune only the head.

---

## Recommended Engineering Order

### Phase 1 - Truth And Import Guardrails

Status 2026-06-07: started.

Completed so far:

- `GraphCompiler` now rejects selected training paths where a node is
  encoded as `Dense` but carries NER/sequence target-design parameters
  such as `bio_scheme`, token/tag columns, vocabulary markers, sequence
  padding markers, `ignore_index`, or decode metadata.
- Focused compiler regression coverage verifies that connected
  Dense-encoded NER target-design nodes fail closed, while structurally
  valid side branches outside the selected training path do not block a
  normal Dense graph.
- `GraphCompiler` now rejects graphs with more than one dataset-reachable
  loss node. The current compiled plan and executor have a single
  `loss_node_id` and no loss aggregation or alternating-step contract, so
  multi-head/multi-task, detection-style, VAE/GAN, and similar sketches
  fail closed instead of silently training only one selected loss.
- `GraphCompiler` now rejects selected training paths where more than one
  `DataInput`/`DatasetInput` source feeds the selected loss. The current
  compiled plan and batcher contract expose one `data_node_id`, one data pin,
  and one label pin, so Siamese/pair/triplet or other multi-input sketches need
  a typed named-batch contract before they can compile truthfully.
- CNN/pooling/upsampling training layers are already blocked through the
  central unsupported sequential-layer capability list and table-driven
  compiler tests. `Conv2D`, pooling/global-pooling, transposed convolution,
  upsample, and pixel-shuffle nodes remain visible design/model-analysis
  surfaces, but they cannot compile as trainable Studio layers until
  `ModelBuilder`/`SequentialModel` support and image batch-shape contracts are
  implemented.
- `GraphCompiler` now rejects selected training paths that sketch
  decoder/generative training with `TransformerDecoder`, explicit causal LM
  flags, shifted-token target parameters, prompt/completion columns, or
  teacher-forcing markers. The current trainer has no trainable decoder path,
  causal mask contract, shifted-token target materializer, token-level
  sequence loss, or generation packaging, so these graphs fail closed instead
  of implying GPT/seq2seq training works.
- `GraphCompiler` now rejects selected training paths that sketch
  imported/pretrained fine-tuning with pretrained shortcut nodes, DNN model-load
  nodes, explicit fine-tune/transfer-learning flags, model/checkpoint/weights
  paths, freeze/unfreeze controls, optimizer-state resume markers, shape
  mismatch markers, or adapter/LoRA paths. The current trainer has no
  import-to-training-graph contract for parameter mapping, shape validation,
  freeze ownership, optimizer-state compatibility, or tokenizer/preprocessor
  packaging.
- `GraphCompiler` now rejects selected training paths that sketch
  reinforcement-learning training with RL nodes, policy/value nodes, or explicit
  RL markers such as rollout/replay, reward/action/state columns, actor-critic,
  policy-gradient, target-network, or episodic settings. The current trainer is
  a supervised single-batch executor and lacks an environment stepping loop,
  rollout/replay buffer schema, policy/value loss contracts, target-network
  handling, and episodic metrics.
- `GraphCompiler` now rejects selected training paths that sketch
  detection/segmentation training with DNN detection/postprocessing nodes or
  explicit object, box, mask, anchor, IoU, NMS, mAP, or segmentation markers.
  The current trainer has no detection target schema, variable-object batching,
  detection heads, multi-head loss aggregation, NMS/evaluation metrics, or
  detection output packaging.
- `GraphCompiler` now rejects selected training paths that sketch
  per-timestep/per-token heads with explicit per-token head markers.
  First-class `TimeDistributed` is no longer blocked by identity alone: it
  compiles only when its input shape is sequence-like and `ModelBuilder` builds
  a `TimeDistributedDenseModule`. The sequence executor path now consumes
  prebuilt `ISequenceBatcher` payloads and reports token/BIO metrics; Studio
  graph materialization is still pending.
- `GraphCompiler` now rejects selected training paths that sketch
  autoencoder/VAE/GAN/diffusion training with exact design names or explicit
  reconstruction, latent, KL, generator/discriminator, gradient penalty,
  diffusion scheduler, timestep, or noise-prediction markers. The current
  trainer has one supervised optimizer step and lacks reconstruction target
  routing, latent KL-loss contracts, alternating/adversarial-step orchestration,
  diffusion noise schedules, and generation output packaging.
- `GraphCompiler` now rejects selected training paths that sketch
  metric-learning/Siamese training with exact pair/triplet/shared-encoder/loss
  node names or explicit anchor/positive/negative, pair/triplet id, pair label,
  shared encoder, or tied-weight markers. The current trainer has one selected
  input tensor and lacks typed pair/triplet batch payloads, shared encoder
  ownership, pair/triplet loss wiring, mining/sampling rules, and embedding
  output packaging.
- `GraphCompiler` now rejects selected training paths that sketch
  graph-neural-network/GNN training with exact graph-convolution/message-passing
  node names or explicit edge-index, edge-attribute, node-feature, adjacency,
  node-classification, link-prediction, or graph-classification markers. The
  current trainer has no graph batch schema, edge-index/adjacency routing,
  message-passing kernels, node/edge/graph target contracts, neighborhood
  batching, or graph-level output packaging.
- Pattern and saved-graph imports now fail closed for unknown pattern node type
  strings and invalid/`Unknown` serialized node type ids instead of silently
  creating generic Dense-like nodes. This keeps unsupported placeholder
  identity from being erased before compile-time guardrails can run.
- Pattern and saved-graph imports also reject exact NER/sequence placeholder
  names when they are encoded as `Dense`, covering name-only placeholders such
  as `NERSequenceBuilder`, vocabulary builders, sequence padding, token loss,
  metrics, and sequence outputs before they can masquerade as trainable Dense
  layers. The shared import-guard predicate is covered by
  `test_pattern_template_guard`.
- The same shared guard is now used by `PatternLibrary` instantiation, including
  the direct creator path. Dense-encoded NER placeholder names and target-design
  parameter markers are rejected before pattern insertion can create a fake
  trainable Dense layer.
- `GraphCompiler` now rejects selected training paths that sketch sequence/NER
  batch materialization through `DataInput` category/column markers or
  sequence-style `DataLoader` payload markers. The current trainer still has a
  single tensor/label batch contract and cannot materialize named sequence
  payloads such as `word_ids`, optional `pos_ids`, `attention_mask`, and
  `tag_ids`.
- `TrainingConfiguration` now has an explicit `SequenceBatchConfig` and the
  compiler populates it from selected-path NER/sequence `DataInput` and
  `DataLoader` markers. This is the first typed contract surface for Phase 2,
  but runtime sequence batching/execution remains blocked until the batcher and
  token-level trainer exist.
- Runtime launch paths now also fail closed when a sequence batch contract is
  present. `StartGraphTrainingFromCompiledConfig`, `SmokeRunExecutor`, and
  `TrainingExecutor::Initialize` reject `SequenceBatchConfig` with a named
  payload message instead of falling through to the tabular/text batcher path.
- The core batch interface now has a dedicated `SequenceBatch` and
  `ISequenceBatcher` contract for token-level payloads (`word_ids`, optional
  `pos_ids`, `attention_mask`, and training `tag_ids`). No implementation is
  wired into training yet; this only fixes the missing typed payload shape that
  the real NER batcher/executor slice will consume.
- A standalone `SequenceBatcher` can now batch already-tokenized sequence
  samples into that payload contract, including truncation, padding,
  `attention_mask`, optional POS ids, and `ignore_index` padding for token tags.
  It is intentionally not connected to `TrainingExecutor` yet because token
  cross-entropy and token-level model outputs are still missing.
- `CrossEntropyLoss` construction now preserves `ignore_index` from loss params
  and from `SequenceBatchConfig`; the default is aligned to `-100`. CPU
  CrossEntropy/NLL ignore handling now skips negative ignore values as well.
- CPU CrossEntropy/NLL now accept native token class axes shaped
  `[batch, seq, tags]` with tag targets shaped `[batch, seq]`, preserving
  unreduced `[batch, seq]` losses, excluding ignored/padded tokens from mean
  loss/grad normalization, and zeroing ignored token gradients.
  Training-executor wiring and token-level metrics are still pending.
- Attention and normalization layer concepts that exist as node types or model
  analysis concepts but are not trainable backend layers now fail closed through
  the central training support table (`LayerNorm`, `GroupNorm`, `InstanceNorm`,
  `MultiHeadAttention`, `SelfAttention`, `CrossAttention`, and
  `LinearAttention`). Metadata marks them blocked instead of letting selected
  training paths silently drop them before `ModelBuilder`.
- A standalone sequence-tag metric helper now computes argmax token predictions,
  token accuracy over non-ignored labels, and exact-match BIO entity
  precision/recall/F1. It is covered with NER-style padding and partial-entity
  tests, but these metrics are not yet surfaced by `TrainingExecutor`.
- A standalone sequence vocabulary helper now builds deterministic token/POS/tag
  id maps over already-tokenized sequence fields. Token/POS vocabularies can
  reserve PAD/UNK ids and apply frequency caps; tag vocabularies keep `O` at id
  zero when present and reject unknown labels. First-class vocabulary nodes are
  still pending.
- A standalone `NERSequenceBuilder` now composes sequence vocabularies with the
  sequence batcher over already-tokenized token/POS/tag rows. It validates row
  shape alignment, encodes `SequenceSample` fixtures, preserves PAD/UNK and
  `ignore_index` settings, and can create a `SequenceBatcher` for focused
  contract tests. It is intentionally not wired into `TrainingExecutor` yet.
- First-class NER contract node identities now exist for
  `NERSequenceBuilder`, `TokenVocabulary`, `POSVocabulary`, and
  `NERTagVocabulary`. They import from pattern/saved graph type strings,
  create default pins/parameters on the canvas, show blocked metadata in the
  node registry, and compile-fail selected training paths through the sequence
  payload guard while preserving `SequenceBatchConfig` details. Runtime
  sequence training remains blocked.
- `TimeDistributedDenseModule` now applies one dense projection over
  `[batch, seq, hidden]` and returns `[batch, seq, units]`, reusing the normal
  linear parameter/gradient path. `ModelBuilder` wires `TimeDistributed` to
  that module, the node has editable `units` metadata, and compiler validation
  rejects `TimeDistributed` over non-sequence tensors.
- A focused `TrainSequenceTaggerEpoch` helper now builds the executable model,
  consumes `ISequenceBatcher` named payloads, runs token-shaped CrossEntropy
  with `ignore_index`, applies optimizer updates, and returns token/BIO
  metrics. This proves the narrow sequence batch/model/loss step.
- Public `TrainingExecutor::Train` now has a sequence mode for prebuilt
  `ISequenceBatcher` payloads. It runs the normal lifecycle callbacks,
  optimizer step, validation pass, token accuracy, and BIO entity F1. The
  Arrow/Parquet/Studio graph launch paths still fail closed for
  `SequenceBatchConfig` until Studio can materialize real sequence batchers
  from graph/runtime data.

### Next Session Handoff - 2026-06-09

Last pushed commit: `4fa72a1a Add sequence vocabulary helper`.

Verified before shutdown:

- `cmake --build build --config Debug --target test_training_executor_arrow_parquet`
- `build\bin\Debug\test_training_executor_arrow_parquet.exe`
- `cmake --build build --config Debug --target cyxwiz-engine`

Current Phase 2 state:

- The low-level NER primitives and first metadata surface exist: typed sequence
  payload contract, standalone sequence batcher, token-shaped CrossEntropy/NLL
  with `ignore_index`, sequence tag metrics, sequence vocabularies, standalone
  `NERSequenceBuilder`, and blocked first-class NER vocabulary/builder node
  identities.
- Runtime sequence training is supported when the launcher can route the graph
  through the tabular loader path and provide a prebuilt `ISequenceBatcher`.
  Sequence runs now expose token accuracy and BIO F1 in the training dashboard
  with sequence-specific labels and plot presentation.
- The token classifier head backend slice, focused sequence-training step, and
  public sequence executor mode are in place. The next practical slice is
  validating the shipped NER example graph end to end and tightening
  import-time guards for placeholder/custom-task node sketches.

1. Continue hardening import-time guards for graphs that use placeholder node
   types or Dense nodes as fake custom task nodes.
2. Mark unsupported model families as blocked at compile time with precise
   messages.
3. Keep `tofix7.md` as the support matrix and update it after every real
   capability change.

### Phase 2 - Sequence Tagging As The First Structured-Output Slice

NER is the best first target because it is concrete and already has an example.

Implement:

1. `NERSequenceBuilder`,
2. word/POS/tag vocabulary node types and executor wiring,
3. sequence padding for tokens and labels,
4. sequence batcher with `word_ids`, optional `pos_ids`, `attention_mask`,
   and `tag_ids`,
5. `TimeDistributedDense` or equivalent token classifier head,
6. token cross-entropy with `ignore_index`,
7. executor surfacing for token accuracy and BIO entity F1.

### Phase 3 - Generalize To Multi-Input / Multi-Output

After NER works, add:

1. named tensor batch payloads,
2. structured model outputs,
3. loss aggregation,
4. shared-module graph support.

This unlocks Siamese, multitask, and some detection-style workflows.

### Phase 4 - CNN Training

Implement CNN wrappers only after the input shape contract is explicit:

1. `Conv2D`,
2. pooling,
3. flatten/global pooling,
4. image batch shape parity,
5. MNIST/CIFAR CNN smoke tests.

### Phase 5 - Decoder / Generative / Pretrained Work

Defer until the structured training contracts are stable:

1. `TransformerDecoder`,
2. causal LM objective,
3. generation loop,
4. pretrained checkpoint import,
5. tokenizer/model package compatibility.

---

## Bottom Line

CyxWiz is currently a supervised training engine with a mostly sequential
single-input/single-output contract. PyTorch can train broader model families
because its core contract is arbitrary module code plus arbitrary batch
structures.

The next backend growth should not be "add every PyTorch layer." It should be:

```text
truthful capability gates
-> typed batch contracts
-> structured outputs/losses
-> non-linear module graphs
-> task-specific families
```

NER should be the first serious structured-output implementation because the
example graph already shows the missing contracts clearly.

## Follow-up Carried Forward From done12

The sentiment fine-tuning work in `done12.md` confirmed that current fine
tuning means training an existing CyxWiz graph from scratch or from an engine
checkpoint. Full pretrained-model fine tuning remains under this broader
model-family/import scope, not the lightweight run-comparison scope.

Future pretrained/fine-tuning work should cover:

- pretrained transformer import and fine tuning,
- freeze/unfreeze controls at module or parameter-group granularity,
- per-layer learning-rate schedules,
- user-selected checkpoint fine tuning and optimizer-state resume semantics,
- tokenizer/vocabulary/preprocessor compatibility for imported models,
- shape and dtype validation before any checkpoint weights are accepted.

Guardrails:

- Do not present pretrained fine tuning as supported until import, mapping,
  tokenizer packaging, freeze state, optimizer grouping, and at least one
  minimal reference-style smoke test exist.
- Keep this separate from `TrainingRunComparisonRecord`; run comparison can
  record what checkpoint was used, but it should not own model import or
  optimizer parameter-group semantics.
