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
| Audit current text stack | Complete | | Read-only audit; no build/test needed | Current engine supports supervised text classification and encoder text paths, but not causal LM/generation. |
| Reference fixtures | Complete | | Unit build passed; tests passed: 1534 assertions in 213 test cases | Added deterministic fixtures for shifted targets, causal/padding masks, token loss/gradient, scaled dot-product attention, positional encoding, greedy generation, decoder semantics, and capability metadata. |
| Tokenizer/vocab packaging | Complete | | Build passed; tests passed: 1340 assertions in 172 test cases | `.cyxmodel` has typed tokenizer/vocab manifest flags, package asset slots, direct extraction API, formatter coverage, and reusable tokenizer package loading for inference. |
| Raw text inference | Complete | | Unit build passed; tests passed: 1538 assertions in 215 test cases; `cyxwiz-engine` target compiles raw-text inference/generation files but remains blocked by unrelated `node_metadata_registry.cpp`/`pipeline_executor.cpp` errors | Local `/v1/predict` accepts string input when a packaged tokenizer/vocab is present; numeric tensor input remains supported; local `/v1/generate` exposes greedy text generation for packaged text models and returns decoded text plus prompt/generated token ids. |
| Token-level training primitives | Complete | | Unit build passed; tests passed: 1565 assertions in 223 test cases; broader engine build compiled `model_builder.cpp`/`training_executor.cpp` before unrelated existing `node_metadata_registry.cpp`/`pipeline_executor.cpp` failures stopped the target | Added shifted next-token batch construction, token-stream window packing, padding attention mask, ignore-index targets, CPU reference token cross-entropy loss/gradient, backend CrossEntropy token-logit agreement fixture, opt-in causal-LM target materialization through the existing SequenceBatcher contract, and sequence-training routing for `target_ids`. |
| Causal attention | Complete | | Unit build passed; tests passed: 1623 assertions in 224 test cases | Added causal attention mask, batched key-padding mask, single/batched/multi-head CPU reference scaled dot-product attention, and deterministic causal masked `MultiHeadAttentionLayer` agreement coverage against the CPU scaled-dot-product reference; standalone attention remains blocked until the pipeline contract is explicit. |
| Decoder pipeline readiness | Complete | | Unit build passed; tests passed: 1534 assertions in 213 test cases; broader engine build compiled `pipeline_runtime_capabilities.cpp`, `graph_compiler.cpp`, and `model_builder.cpp` before unrelated existing `node_metadata_registry.cpp`/`pipeline_executor.cpp` failures stopped the target | `TransformerDecoder` is now gated by shifted targets, causal masks, token loss, tokenizer packaging, raw-text contract, generation primitive coverage, and capability truth; generic LLM/generation sketches remain blocked while the explicit tested decoder stack is no longer fail-closed as unsupported. |
| Minimal causal LM | Complete | | Unit build passed; tests passed: 1623 assertions in 224 test cases; broader engine build compiled decoder/model-builder changes before unrelated existing `node_metadata_registry.cpp`/`pipeline_executor.cpp` failures stopped the target | Added `TransformerDecoderModule`, `PositionalEncodingModule`, model-builder routing for `TransformerDecoder` and positional encoding, decoder FFN 3D flatten/restore handling, decoder module shape/backward fixture, decoder-only single-input semantics that skip cross-attention, Embedding -> PositionalEncoding -> TransformerDecoder -> TimeDistributed LM-head stack fixture with token cross-entropy gradients, and a deterministic toy next-token training-step fixture covering finite loss, backward, gradients, and parameter update. |
| Generation | Complete | | Unit build passed; tests passed: 1538 assertions in 215 test cases; broader engine build compiled `local_inference_server.cpp`/`text_inference_input.cpp` before unrelated existing `node_metadata_registry.cpp`/`pipeline_executor.cpp` failures stopped the target | Added a narrow greedy token-generation primitive for `SequentialModel` logits, deterministic argmax/validation fixtures, a real sequential LM-head append-loop fixture, generation-safe text token-id helpers, and local `/v1/generate` greedy server integration; temperature/top-k remain future work. |
| UI/capability truth | Complete | | Unit build passed; tests passed: 1534 assertions in 213 test cases; broader engine build compiled capability/compiler/model-builder changes before unrelated existing `node_metadata_registry.cpp`/`pipeline_executor.cpp` failures stopped the target | Training capability registry now exposes tested causal-LM building blocks (`Embedding`, `PositionalEncoding`, `TransformerEncoder`, `TransformerDecoder`, `TimeDistributed`) while standalone attention nodes remain blocked; real `TransformerDecoder` is no longer treated as a generic unsupported generation sketch. |
| HuggingFace / instruction dataset mapping | Complete | | Unit build passed; tests passed: 1557 assertions in 222 test cases | Added deterministic instruction-record formatting, strict table/HF-style column mapping for query/answer or prompt/response corpora, optional system-column support, empty-row handling, and causal-LM window packing; real remote HuggingFace streaming/loading remains future work. |

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

Audit result:

- real today: `TextDataset`, `TextDatasetBatcher`, tokenizer operator, supervised classification, embedding/recurrent/encoder text paths, token-shaped loss support, sequence tagging primitives, and numeric HTTP inference
- not real today: causal LM training, shifted next-token batch construction, causal mask execution, decoder generation, tokenizer/vocab packaging in `.cyxmodel`, and raw-text inference
- design decision: keep `TransformerDecoder` fail-closed until the causal pipeline readiness gate is satisfied

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

### 7. Decoder pipeline readiness

Scope:

- shifted-target batcher contract
- causal mask contract
- token-level loss contract
- tokenizer/vocab artifact contract
- raw-text request/response contract
- capability metadata contract

Done criteria:

- compiler, builder, runtime, inference, and docs agree on decoder prerequisites
- unsupported decoder paths remain blocked with clear errors
- `TransformerDecoder` is not marked supported until these contracts are tested

### 8. Minimal causal LM

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

### 9. Generation

Scope:

- greedy generation
- prompt tokenization
- EOS/max-token stopping
- later: temperature and top-k

Done criteria:

- deterministic prompt-to-token generation test passes
- raw-text generation response is separate from classification response

### 10. UI/capability truth

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
