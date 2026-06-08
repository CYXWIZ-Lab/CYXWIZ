# To Fix 19 - Model Families CyxWiz Cannot Train Yet

**Created:** 2026-06-07
**Source:** Follow-up from the NER example audit in
`examples/cyxgraph/NER/ner_bilstm_sequence_tagger.cyxgraph`.

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

- no first-class `NERSequenceBuilder`,
- no typed batch with `word_ids`, `pos_ids`, `attention_mask`, and `tag_ids`,
- no real `TokenVocabulary`, `POSVocabulary`, and `NERTagVocabulary` node types,
- no sequence padding contract for labels,
- no `TimeDistributedDense` or equivalent token classifier head,
- current CPU cross-entropy supports 1D/2D predictions, not native
  `[batch, seq, tags]`,
- loss builder does not preserve graph-level `ignore_index`,
- no token-level or entity-level BIO metrics.

Concrete example:

`examples/cyxgraph/NER/ner_bilstm_sequence_tagger.cyxgraph` is a target design,
not a trainable graph yet. Several intended NER nodes are represented as
generic Dense-typed nodes with custom names, so the engine would misinterpret
them without new node types or import-time guards.

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
- `GraphCompiler` now rejects selected training paths that sketch
  decoder/generative training with `TransformerDecoder`, explicit causal LM
  flags, shifted-token target parameters, prompt/completion columns, or
  teacher-forcing markers. The current trainer has no trainable decoder path,
  causal mask contract, shifted-token target materializer, token-level
  sequence loss, or generation packaging, so these graphs fail closed instead
  of implying GPT/seq2seq training works.

1. Add import-time guards for graphs that use placeholder node types or Dense
   nodes as fake custom task nodes.
2. Mark unsupported model families as blocked at compile time with precise
   messages.
3. Keep `tofix7.md` as the support matrix and update it after every real
   capability change.

### Phase 2 - Sequence Tagging As The First Structured-Output Slice

NER is the best first target because it is concrete and already has an example.

Implement:

1. `NERSequenceBuilder`,
2. word/POS/tag vocabulary node types,
3. sequence padding for tokens and labels,
4. sequence batcher with `word_ids`, optional `pos_ids`, `attention_mask`,
   and `tag_ids`,
5. `TimeDistributedDense` or equivalent token classifier head,
6. token cross-entropy with `ignore_index`,
7. token accuracy and BIO entity F1.

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
