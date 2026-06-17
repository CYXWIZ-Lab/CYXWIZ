# Tracker 8 - Reference-Tested LLM Implementation Roadmap

This tracker belongs to `tofix8.md`.

`tofix8.md` defines the missing LLM/text-generation capability. This tracker
keeps implementation progress measurable without turning the problem statement
into a long work log.

Core engineering rule:

Do not port a full LLM framework into CyxWiz. Use trusted references to create
small deterministic fixtures, then implement native CyxWiz components with
ArrayFire as the first-class compute path and CPU fallback for correctness and
portability.

---

## Reference-Test-Driven Engineering

For each computational feature:

1. Identify the trusted reference.
2. Create a tiny deterministic fixture.
3. Record expected outputs or invariants.
4. Implement the CyxWiz ArrayFire path.
5. Implement or preserve CPU fallback.
6. Assert GPU/CPU/reference agreement within tolerance.
7. Update capability metadata only after tests prove support.

References should be used for math and expected behavior, not copied blindly.
Licenses must be respected.

---

## Candidate References

| Area | Reference type | Notes |
|---|---|---|
| Transformer math | Papers | `Attention Is All You Need`, causal LM literature |
| Expected tensor behavior | PyTorch / HuggingFace | Use for deterministic fixtures and shape semantics |
| Minimal causal LM structure | minGPT / nanoGPT-style educational implementations | Use for design understanding, not direct copy |
| Inference/runtime ideas | llama.cpp / ggml-style projects | Useful for deployment ideas; not the training backend target |

---

## Roadmap Phases

| Phase | Status | Commit | Tests | Notes |
|---|---|---|---|---|
| Audit current text stack | Not started | | | Inventory tokenizer, vocab, dataset, batcher, embedding, transformer encoder, save/load, inference server. |
| Reference fixtures | Not started | | | Add deterministic fixtures for attention, masks, token loss, shifting, positional encoding. |
| Tokenizer/vocab packaging | Not started | | | Store tokenizer config and vocab in `.cyxmodel`. |
| Raw text inference | Not started | | | Server accepts raw text and tokenizes internally for existing text classifiers first. |
| Token-level training primitives | Not started | | | Shifted targets, padding ignore index, token-level cross entropy. |
| Causal attention | Not started | | | Masked scaled dot-product attention, ArrayFire-first with CPU fallback. |
| Minimal causal LM | Not started | | | Tiny decoder-only model path that can overfit a toy corpus. |
| Generation | Not started | | | Greedy decoding first; temperature/top-k later. |
| UI/capability truth | Not started | | | Expose only tested supported nodes; keep unsupported LLM nodes blocked/experimental. |

---

## Phase Details

### 1. Audit current text stack

Scope:

- tokenizer and vocabulary code
- `TextDataset` and text batchers
- `Embedding`
- `TransformerEncoder`
- text-classification training path
- `.cyxmodel` save/load
- embedded HTTP inference server
- graph compiler and metadata exposure for transformer/LLM nodes

Done criteria:

- exact current capability map is documented
- duplicate or stale implementation paths are identified
- first implementation batch is chosen from current engine truth

### 2. Reference fixtures

Scope:

- embedding lookup
- positional encoding
- padding mask
- causal mask
- scaled dot-product attention
- token-level cross entropy
- shifted next-token targets

Done criteria:

- deterministic expected outputs exist
- tests are small enough to run in the normal suite
- reference provenance is documented

### 3. Tokenizer/vocab packaging

Scope:

- tokenizer settings in `.cyxmodel`
- vocabulary/token map in `.cyxmodel`
- deployment manifest metadata
- load-time reconstruction

Done criteria:

- text classifier artifact is self-contained
- inference does not require external tokenizer/vocab files

### 4. Raw text inference

Scope:

- raw text request support
- internal tokenization for text models
- classifier inference first
- no generation mode until causal LM exists

Done criteria:

- client can send text directly
- numeric tensor inference remains supported
- server reports clear errors when tokenizer metadata is missing

### 5. Token-level training primitives

Scope:

- shifted input/target construction
- padding ignore index
- token-level cross entropy
- shape validation for sequence logits and labels

Done criteria:

- toy loss matches reference fixture
- CPU and ArrayFire paths agree within tolerance

### 6. Causal attention

Scope:

- masked scaled dot-product attention
- stable softmax under mask
- batch/head/sequence shape validation
- ArrayFire-first compute path
- CPU fallback or clear unsupported status

Done criteria:

- attention output matches reference fixture
- mask behavior is tested
- backend placement truth is surfaced

### 7. Minimal causal LM

Scope:

- token embedding
- positional encoding
- causal self-attention block
- feed-forward block
- LM head
- next-token training loop

Done criteria:

- tiny model overfits a toy corpus
- no unsupported LLM nodes are marked real before this passes

### 8. Generation

Scope:

- greedy generation
- prompt tokenization
- EOS/max-token stopping
- later: temperature and top-k

Done criteria:

- deterministic prompt-to-token generation test passes
- raw-text generation response is separate from classification response

### 9. UI/capability truth

Scope:

- compiler support
- builder support
- runtime support
- metadata support axes
- blocked/experimental labels

Done criteria:

- UI, compiler, builder, runtime, and docs tell the same story
- unsupported transformer/LLM nodes fail closed or are hidden

---

## Batch Rule

Every batch should include:

- a narrow implementation target
- reference source or provenance
- tests before or with implementation
- build/test result
- tracker update
- commit and push

Do not combine unrelated phases in one batch.
