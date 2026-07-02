# Transformer and NLP Blocks

This page describes what CyxWiz Engine can truthfully use today for
transformer-style NLP graphs.

## Supported building blocks

- `Embedding`: token-id lookup from `[batch, seq_len]` to
  `[batch, seq_len, embedding_dim]`.
- `PositionalEncoding`: sinusoidal positional encoding over
  `[batch, seq_len, d_model]`.
- `LayerNorm`: CPU-backed model layer with forward/backward correctness
  coverage.
- `MultiHeadAttention`: supported as single-input self-attention in
  `SequentialModel`.
- `TransformerEncoder`: supported as a CPU-backed model layer with forward
  computation-truth coverage.
- `TransformerDecoder`: supported as a CPU-backed decoder-only causal
  self-attention model layer with forward computation-truth and `.cyxmodel`
  graph export/import coverage.
- `Linear`: usable as a vocabulary/logit head after transformer hidden states.
- `CrossEntropyLoss`: supports class-index targets for logits shaped
  `[batch, classes]` and `[batch, seq_len, classes]`.

## Current graph guidance

For BERT-like encoder experiments, use this shape:

```text
Token ids -> Embedding -> PositionalEncoding -> TransformerEncoder -> Flatten/Pooling -> Dense -> Loss
```

For GPT-like decoder experiments, use this shape:

```text
Token ids -> Embedding -> PositionalEncoding -> TransformerDecoder -> Linear vocab head -> CrossEntropy
```

The current decoder path is decoder-only causal self-attention. Full
encoder-decoder cross-attention is not yet a first-class training graph
contract.

## Sidebar patterns

The Pattern Browser includes a `Causal Decoder Starter (CPU-backed)` pattern in
the Transformer category. It follows the currently proven decoder inference
shape:

```text
Token ids -> Embedding -> PositionalEncoding -> TransformerDecoder -> LayerNorm -> Flatten -> Dense logit head -> Output
```

Use it as a small model starter or architecture reference. It is not a full
GPT product claim yet: generation controls, token-by-token decoding UX, and
full language-model training contracts remain follow-up work.

The Pattern Browser also includes `Causal LM Generation Starter`. It keeps the
sequence dimension through a `TimeDistributed` token head:

```text
Token ids -> Embedding -> PositionalEncoding -> TransformerDecoder -> LayerNorm -> TimeDistributed token head -> Output
```

That pattern is the better starting point for prompt-generation runtime
experiments because it produces sequence logits shaped `[batch, seq, vocab]`.
Do not add `Softmax` before generation or `CrossEntropyLoss`; both expect raw
logits.

## Generation controls

CyxWiz has a reusable generation-control contract for decoder-style models:

- `max_new_tokens`: maximum number of generated tokens.
- `temperature`: scales logits before sampling; must be greater than zero.
- `top_k`: keeps only the highest-k candidate tokens when non-zero.
- `top_p`: nucleus sampling threshold in the range `(0, 1]`.
- `eos_token_id`: optional token that stops generation when emitted.
- `sampling_mode`: greedy or seeded multinomial sampling.

This is the control-plane contract. The core runtime path and Studio tool can
now use the generation config against decoder models that return logits shaped
`[1, seq, vocab]`.

Studio exposes this through:

```text
Tools -> Machine Learning -> Text Processing -> Language Model Generation
```

The panel supports two prompt modes:

- `Text prompt`: loads a CyxWiz vocabulary file, encodes text through the
  existing tokenizer, generates token IDs, then decodes generated IDs with the
  same vocabulary. It can also load packaged tokenizer assets directly from a
  `.cyxmodel` containing `tokenizer/config.json` and `tokenizer/vocab.txt`.
- `Raw token IDs`: accepts space/comma/newline-separated token IDs for debugging
  model/runtime behavior directly.

The panel can use either:

- the last trained model held by `TrainingManager`, or
- a model imported directly from a selected `.cyxmodel` package.

Use `Check compatibility` before generating to confirm the active model returns
`Float32 [1, seq, vocab]` logits for the current prompt. This catches classifier
or flattened decoder heads before generation is attempted.

When packaged tokenizer assets are loaded, the panel reports vocabulary size,
max prompt length, and EOS token ID, and uses those packaged values as safer
defaults for generation.

If a `.cyxmodel` does not contain tokenizer assets, use manual vocabulary-file
mode for text prompts or raw token-ID mode for runtime debugging.

For deployable text-generation packages, include both tokenizer assets:

- `tokenizer/config.json`
- `tokenizer/vocab.txt`

This lets Studio load the model and tokenizer together, encode text prompts to
token IDs, and decode generated token IDs back to text with the same vocabulary
used during training/export.

`.cyxmodel` packages can also declare generation intent in their manifest:

- `model.family = causal_lm`
- `model.supports_generation = true`
- `model.generation_output_contract = Float32[1,seq,vocab]`

This metadata is a package declaration, not a substitute for runtime proof. Use
the Studio `Check compatibility` action to verify the active imported model
actually returns `Float32 [1, seq, vocab]` logits for the current prompt.

## Causal LM graph shape convention

For generation-shaped causal language-model graphs, `DatasetInput` should
describe the per-token feature width in `shape` and the sequence length
separately:

- `shape = [features]`
- `create_causal_lm_targets = true`
- `max_sequence_length = seq_len`

The compiler treats this as a sequence tensor contract:

- compiler `input_shape = [seq_len, features]`
- compiler `input_size = features`
- `TransformerDecoder` preserves `[seq_len, features]`
- `TimeDistributed` projects `[seq_len, features]` to `[seq_len, vocab]`

For token-ID graphs that start with `Embedding`, the per-token feature width is
one scalar token ID, so use:

- `shape = [1]`
- `max_sequence_length = seq_len`

The Embedding node then converts the sequence to `[seq_len, d_model]` before
the transformer stack.

Do not encode the same graph as `shape = [seq_len, features]` unless the
loader/materializer contract explicitly owns that full tensor shape. Otherwise
the compiler may treat the input as flattened feature width for non-sequence
graphs.

## Important limitations

- Connected `Key`, `Value`, or `Context` pins on standalone
  `MultiHeadAttention` are intentionally blocked for training until the graph
  runtime, export/import, debugger, and inference contracts agree.
- `TransformerEncoder` and `TransformerDecoder` are currently treated as
  CPU-backed model layers for placement truth.
- Forward correctness is stronger than full training correctness for
  transformer stacks. Backward parity, masking coverage, and full inference
  contracts are still being expanded.
- Do not treat current support as full BERT, GPT, T5, or ELMo support yet.
  Those labels require complete model-family contracts, not only primitive
  layers.

## GPU status and roadmap

Current transformer-family layers are CPU-backed by design:

- `LayerNorm`
- `MultiHeadAttention`
- `TransformerEncoder`
- `TransformerDecoder`

This means GPU kernels are not yet implemented or claimed for those layers. It
does not mean the engine tried GPU and failed. A real GPU fallback is reported
separately when a GPU-capable layer, such as `Dense`, cannot run on the active
backend for a specific dtype, shape, or runtime condition.

Roadmap:

1. Finish CPU truth first: math correctness, graph compile behavior,
   checkpointing, export/import, debugger placement, and inference contracts.
2. Add focused ArrayFire/CUDA kernels second, after the CPU contract is stable.
3. Claim GPU support only after parity, residency, fallback, and performance
   tests prove the GPU implementation.

Rule of thumb: no GPU transformer claim before CPU correctness is complete and
tested.

## What is tested now

- Embedding lookup and gradient accumulation against PyTorch semantics.
- Positional encoding against sinusoidal PyTorch-style values.
- Scaled dot-product attention, causal masks, additive masks, and cross
  attention primitive math.
- Multi-head attention forward/backward primitive parity.
- LayerNorm forward/backward parity.
- TransformerEncoder forward parity for self-attention, residuals,
  post-layernorm, ReLU feed-forward, and final post-layernorm.
- TransformerDecoder decoder-only causal forward parity.
- Tiny causal language-model vocabulary-head logits and cross-entropy loss.
- Generation-control validation plus greedy, top-k, top-p, and seeded
  multinomial next-token selection.
- Sidebar causal-LM generation starter pattern that preserves
  `[batch, seq, vocab]` logits through a `TimeDistributed` token head.
- `.cyxmodel` graph export/import roundtrip for `TransformerEncoder` and
  `TransformerDecoder` stacks with Dense heads.
- Source-vs-imported inference-output roundtrip for a decoder stack shaped
  `TransformerDecoder -> Flatten -> Dense`.
- Checkpoint parameter roundtrip for a `TransformerDecoder` stack with a Dense
  head.

## Practical recommendation

Use transformer blocks for small controlled experiments first. For production
training graphs, prefer models where the graph compiler, debugger, checkpoint,
export/import, and computation-truth tests all describe the same behavior.
