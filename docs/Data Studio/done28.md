# To Fix 28 - LLM/Text Generation Follow-Up Work

This document tracks follow-up work split out from `done8.md` after the focused
`tofix8` implementation completed.

`done8.md` established a minimal, native, reference-tested LLM/text-generation
path in CyxWiz. This file is not a restart of that roadmap. It is for production
hardening and broader capability work that should stay separate from the closed
minimal path.

---

## Goal

Harden and extend the tested text-generation path without bloating the engine or
claiming production-scale LLM support before the required pieces are real.

The essential rule remains:

- keep native CyxWiz components small and auditable
- use reference-tested fixtures before broad enablement
- keep ArrayFire/GPU as first-class compute where it is proven stable
- preserve CPU fallback and capability truth where GPU/runtime support is not
  proven
- do not grow already-large implementation files; add focused translation units
  and narrow headers for new transformer/NLP primitives, builders, tests, and
  runtime contracts

---

## Follow-Up Areas

### 0. Model family capability roadmap

Scope:

- `CBOW` / shallow embedding models as the smallest production NLP baseline
- `ELMo`-style contextual embedding path only after recurrent sequence
  correctness and export contracts are proven
- `BERT` encoder-only transformer path for classification, embedding, and
  masked-language-model style workloads
- `GPT` decoder-only causal-LM path for generation and instruction-style
  completion
- `T5` encoder-decoder path for sequence-to-sequence workloads such as
  summarization, translation, and text-to-text transformation

Implementation direction:

- build these from shared primitive layers instead of one model-specific
  monolith
- make reusable primitives first: embeddings, positional encodings, layer norm,
  multi-head attention, feed-forward blocks, residual/dropout behavior, masks,
  token losses, and tokenizer/vocab contracts
- keep each model family behind truthful capability metadata until compile,
  train, export, load, and inference paths are tested

Non-goal:

- do not claim full BERT/GPT/T5 production support because one small fixture
  trains or generates text.

### 0a. PyTorch numerical reference and open-source oracle policy

Scope:

- use PyTorch as the primary numerical reference for transformer/NLP primitives
  and model-family fixtures
- compare CyxWiz CPU and ArrayFire/GPU outputs against PyTorch tensors for:
  forward pass values, loss values, gradients where supported, mask behavior,
  generation logits, and deterministic sampling
- keep PyTorch reference tests outside the core runtime boundary as test
  harnesses, fixtures, or generated expected-output artifacts
- use well-known open-source implementations such as PyTorch examples and
  HuggingFace-style reference behavior to validate contracts, but port only the
  minimal algorithmic behavior needed by CyxWiz

Reference-test requirements:

- every new primitive must have tiny deterministic PyTorch parity fixtures
- tolerances must be explicit per backend and dtype
- CPU parity must pass before GPU support is marked real
- GPU parity must record backend placement and fallback reason when ArrayFire
  cannot execute a kernel
- fixtures must be small enough to run quickly and avoid hiding numerical bugs
  behind memory pressure

Non-goal:

- do not embed PyTorch as a required engine runtime dependency.
- do not copy a full open-source framework into CyxWiz; use references to prove
  behavior, then implement the smallest compatible native primitive.

### 1. Production dataset ingestion

Scope:

- real HuggingFace or HF-style dataset loading beyond deterministic row mapping
- streaming or chunked ingestion for large instruction datasets
- split handling for train/validation/test
- robust schema detection for prompt/query/answer/response variants

Non-goal:

- do not add a large dataset framework inside the core engine unless the boundary
  is narrow and testable.

### 2. Generation controls

Scope:

- temperature sampling
- top-k sampling
- top-p / nucleus sampling
- EOS stopping behavior
- max-new-token validation
- deterministic seeded generation tests

Non-goal:

- do not expose UI claims for sampling modes until server/runtime tests prove
  the behavior.

### 3. Standalone attention graph contract

Scope:

- define whether `MultiHeadAttention` is a real standalone graph node or only an
  internal layer used by encoder/decoder modules
- align compiler, model builder, runtime, metadata, and UI behavior
- add tests before enabling standalone graph usage

Non-goal:

- do not silently unblock standalone attention just because decoder-internal
  attention is now tested.

### 4. Larger model and runtime stability

Scope:

- stress tests for longer sequence lengths and larger vocab/logit shapes
- memory behavior under ArrayFire CUDA
- clear GPU-to-CPU fallback diagnostics
- avoid blanket `eval()` calls unless a specific kernel/materialization issue is
  proven

Non-goal:

- do not hide CUDA kernel failures behind vague success states.

### 5. Performance and backend placement

Scope:

- profile minimal causal-LM forward/generation paths
- identify hotspots in attention, token loss, embedding, and time-distributed
  dense operations
- preserve backend placement truth for unsupported or CPU-routed paths

Non-goal:

- do not optimize by adding broad abstractions or duplicating algorithm paths.

### 6. Artifact and serving hardening

Scope:

- richer `.cyxmodel` text metadata validation
- backward-compatible tokenizer/vocab manifest versioning
- clearer `/v1/generate` error codes and response contracts
- model capability reporting that separates classification from generation

Non-goal:

- do not merge classification and generation response shapes into one ambiguous
  contract.

### 7. Transformer primitive implementation order

Recommended implementation order:

1. `Embedding` and tied output projection contracts
2. positional encoding and learned position embeddings
3. `LayerNorm` parity against PyTorch
4. attention masks: padding mask, causal mask, and combined masks
5. scaled dot-product attention
6. `MultiHeadAttention` as a reusable primitive
7. transformer feed-forward block
8. encoder block and decoder block
9. token-level cross entropy and masked loss contracts
10. tiny model fixtures for CBOW, BERT-like encoder, GPT-like decoder, and
    T5-like encoder-decoder

Guardrail:

- each step must compile, run, and pass PyTorch parity before the next model
  family is exposed in UI metadata.

### 8. Translation-unit modularization guardrail

Current risk:

- transformer/NLP support touches areas that are already large, especially
  model building, graph compilation, node metadata, and backend layer code
- adding BERT/GPT/T5 support directly into existing monolithic files will make
  correctness and debugging harder

Required direction:

- split new model-family logic into focused files, for example:
  `text_model_family_contracts.*`, `transformer_primitive_contracts.*`,
  `language_model_generation_config.*`, and small model-family builders
- keep `model_builder.cpp` as routing/glue only; move detailed NLP/transformer
  construction helpers into separate translation units before adding more
  model-family behavior
- keep PyTorch/reference fixtures in computation-truth tests or generated
  fixture files, not embedded in production runtime code
- do not add broad helper classes unless at least two concrete primitives use
  them immediately

Done rule:

- any implementation batch that makes a large existing file larger must either
  justify why it is glue-only or first extract focused code into a new
  translation unit.

---

## Relationship To Other Docs

- `done8.md`: closed minimal LLM/text-generation implementation roadmap.
- `tracker8_done.md`: implementation history and validation record for `done8.md`.
- `done14.md`: structural/modularity debt such as splitting oversized layer
  implementation files into cleaner translation units.
- `done27.md`: Data Studio runtime/materializer parity follow-up from
  `done7.md`, not an LLM follow-up.

---

## Done Criteria

This follow-up is done only when each enabled production feature has:

- a narrow design boundary
- reference-driven tests or deterministic fixtures
- CPU correctness coverage where applicable
- ArrayFire/GPU coverage where support is claimed
- truthful compiler/runtime/UI capability metadata
- clear unsupported states for anything not implemented

---

## Close Summary - 2026-07-02

`tofix28` is closeable.

The ticket delivered the transformer/NLP foundation needed for truthful small
language-model work in CyxWiz. The engine now has CPU-backed transformer-family
building blocks, generation controls, `.cyxmodel` generation metadata, packaged
tokenizer support, Studio generation tooling, and validated causal-LM shaped
export/import behavior.

### Implemented capabilities

- First-class transformer/NLP blocks:
  - `Embedding`
  - `PositionalEncoding`
  - `LayerNorm`
  - `MultiHeadAttention`
  - `TransformerEncoder`
  - `TransformerDecoder`
  - `TimeDistributedDense`
- Reusable language-model generation controls:
  - greedy decoding
  - multinomial sampling
  - temperature
  - top-k
  - top-p
  - EOS stopping
  - seed-controlled sampling
- Studio Language Model Generation panel:
  - raw token-ID prompt mode
  - text prompt mode
  - packaged tokenizer loading from `.cyxmodel`
  - imported `.cyxmodel` model loading
  - compatibility check for `Float32 [1, seq, vocab]`
- `.cyxmodel` generation package metadata:
  - `model_family = causal_lm`
  - `supports_generation = true`
  - `generation_output_contract = Float32[1,seq,vocab]`
- Packaged tokenizer support:
  - `tokenizer/config.json`
  - `tokenizer/vocab.txt`
  - tokenizer extraction
  - text prompt encoding
  - generated token decoding
- Causal-LM graph/export/import support:
  - `TransformerDecoder -> TimeDistributedDense`
  - parameter roundtrip
  - imported model output shaped `[batch, seq, vocab]`
- Graph compiler causal-LM shape truth:
  - `shape = [features]`
  - `max_sequence_length = seq_len`
  - compiler input shape becomes `[seq_len, features]`
  - compiler `input_size = features`
  - token-ID plus Embedding graphs use `shape = [1]`
- Sidebar starter patterns:
  - conservative decoder starter
  - causal-LM generation starter with causal target metadata, max sequence
    length, token-ID shape, and `TimeDistributed` vocab head.

### What the engine can do now

- Build and run small CPU-backed transformer encoder/decoder stacks.
- Represent decoder-only causal-LM style graphs truthfully.
- Export and import generation-shaped `.cyxmodel` packages.
- Package tokenizer assets with a model.
- Load packaged model/tokenizer assets for generation experiments.
- Validate generation compatibility before generation.
- Generate token IDs with configurable sampling controls.
- Decode generated token IDs back to text when tokenizer assets are present.

### Validated targets

Focused validation passed on 2026-07-02:

- `cyxwiz-engine`
- `test_language_model_generation`
- `test_cyxmodel_generation_metadata`
- `test_cyxmodel_exporter_generation_metadata`
- `test_graph_compiler_causal_lm_shape`
- `test_cyxmodel_causal_lm_generation_roundtrip`
- `test_pattern_template_guard`

### Not claimed by this ticket

The following are explicitly not claimed as complete:

- Full GPT training support.
- Full BERT model-family support.
- T5 / encoder-decoder support.
- ELMo support.
- Transformer GPU kernels.
- Full transformer backward parity across stacked models.
- Real trained causal-LM launcher coverage from dataset to exported package.

### Follow-up tickets

Follow-ups were documented separately because `tofix46` through `tofix48`
already exist:

- `done49` - Full LM-stack inference contract completed.
- `tofix50` - GPT-style generation controls and UX.
- `tofix51` - BERT-style encoder graph/head/inference coverage.
- `tofix52` - Transformer backward parity and training correctness.
- `tofix53` - Cross-attention and encoder-decoder contract.
- `tofix54` - CBOW first-class small model family.
- `tofix55` - Deferred ELMo and T5 readiness gate.
