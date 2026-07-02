# Track 28 - NLP/LLM Capability Tracker

Implementation tracker for `tofix28.md`.

## Status

Closed on 2026-07-02.

`tofix28` delivered the CPU-backed transformer/NLP foundation, generation
controls, `.cyxmodel` generation metadata, packaged tokenizer support,
Studio generation tooling, causal-LM compiler shape truth, and
generation-shaped export/import validation.

Remaining GPT/BERT/T5/ELMo/CBOW expansion work is tracked separately in
`tofix49` through `tofix55`.

## Scope

Take CyxWiz toward stronger native NLP/model-building capability while keeping
the engine truthful, modular, and reference-tested.

Target model families:

- `CBOW`
- `ELMo`-style contextual embeddings
- `BERT`-style encoder models
- `GPT`-style causal decoder models
- `T5`-style encoder-decoder models

## Current Engine Truth

- Backend already contains native sequence primitives:
  `EmbeddingModule`, `PositionalEncodingModule`,
  `TransformerEncoderModule`, `TransformerDecoderModule`, and
  `MultiHeadAttentionLayer`.
- Studio training support marks `Embedding`, `PositionalEncoding`,
  `TransformerEncoder`, `TransformerDecoder`, and `TimeDistributed` as
  supported model-layer nodes.
- Standalone Studio `MultiHeadAttention` remains blocked for training/model
  building even though the backend layer exists.
- Minimal causal language-model work exists from `done8.md` and
  `tracker8_done.md`.
- Computation-truth infrastructure already exists under
  `cyxwiz-engine/tests/computation_truth`.
- Existing unit tests under `tests/unit` cover deterministic attention and
  sequence-layer behavior, but `tofix28` needs explicit PyTorch-oracle parity
  fixtures before broader model-family claims.

## PyTorch Reference Policy

- PyTorch is the numerical oracle for transformer/NLP primitive correctness.
- PyTorch is not a required CyxWiz runtime dependency.
- Reference fixtures should be tiny, deterministic, and version-tolerant.
- Compare CPU first, then ArrayFire/GPU where backend support is claimed.
- Tolerances must be explicit by primitive, dtype, and backend.

## Modularization Guardrails

- Do not add more transformer/model-family logic directly into already-large
  implementation files unless it is glue-only.
- Keep `model_builder.cpp` as routing/glue; move detailed NLP/transformer
  construction into focused translation units.
- Keep graph-compiler support truth separate from backend primitive
  implementation.
- Keep PyTorch fixture generation/testing separate from production runtime code.
- Prefer small translation units with narrow headers over broad new registries.

## CPU First, GPU Second Roadmap

CyxWiz should not claim GPU support for transformer/NLP model-family layers
until CPU correctness is complete and tested.

Roadmap:

1. CPU truth first: prove forward/backward math, graph compile behavior,
   checkpointing, `.cyxmodel` export/import, debugger placement, and inference
   contracts on the CPU-backed path.
2. GPU implementation second: add focused ArrayFire/CUDA kernels only after the
   CPU contract is stable.
3. GPU claim last: mark a transformer/NLP layer as GPU-capable only after
   parity, residency, fallback, and performance tests prove the GPU path.

Current CPU-backed transformer-family placement is intentional. It is not a
runtime fallback unless a placement entry explicitly says a GPU-capable layer
fell back due to a backend/runtime reason.

Candidate extraction points:

- `transformer_primitive_contracts.*`
- `language_model_generation_config.*`
- `text_model_family_contracts.*`
- `model_builder_transformer.*`
- `model_builder_language_models.*`
- `computation_truth_transformer_primitives.*`

## Capability Matrix

| Area | Current truth | Next action |
|---|---|---|
| Embedding | Native module exists and Studio training supports it | Initial LibTorch oracle parity implemented for non-Debug builds; Debug uses baked PyTorch-derived fixture because Windows Debug + release LibTorch crashes; padding/pretrained/freeze coverage still pending |
| PositionalEncoding | Native module exists and Studio training supports it | Initial LibTorch oracle parity implemented for non-Debug builds; Debug uses baked PyTorch-derived sinusoidal fixture |
| MultiHeadAttention backend | Native attention math exists with tests | Scaled dot-product attention, causal mask parity, projected `MultiHeadAttentionLayer` forward/per-head weights parity, additive mask parity, self-attention backward parity, cross-attention forward/backward parity, and key-padding-style mask parity are covered; export/load still required before exposing standalone Studio training |
| MultiHeadAttention Studio node | Single-input self-attention is supported through `MultiHeadAttentionModule`; connected Key/Value/Context cross-attention remains blocked; `.cyxmodel` graph-backed export/import and debugger CPU-backed capability truth are covered | Keep multi-input/cross-attention blocked until a graph/runtime/export/debugger contract exists |
| TransformerEncoder | Native module exists and Studio training supports it; forward PyTorch-oracle parity, `.cyxmodel` graph-backed export/import, and CPU-backed placement truth are covered | Add mask/backward parity before claiming broader BERT-style encoder correctness |
| TransformerDecoder | Native causal decoder module exists; decoder-only causal forward parity, `.cyxmodel` graph-backed export/import with inference-output roundtrip, checkpoint roundtrip, CPU-backed placement truth, and reusable generation controls are covered | Wire generation controls into prompt-driven runtime/UI before broader GPT-style claims |
| LayerNorm | Lower-level backend layer exists and Studio metadata exists | Added focused CPU forward/backward contract with PyTorch parity; added `LayerNormModule` SequentialModel wrapper, ModelBuilder support, CPU-backed backend placement truth, checkpoint parameter round-trip coverage, and `.cyxmodel` graph-backed export/import regression; GPU coverage remains pending |
| CBOW | Not first-class as a model family | Implement as a small embedding-pooling-classifier/reference fixture before larger families |
| ELMo | Not first-class | Wait for recurrent sequence correctness/export contracts |
| BERT | Not first-class | Build only after encoder/mask/loss parity is proven |
| GPT | Minimal causal path exists; decoder-only causal transformer block forward parity, tiny LM-head logits/loss parity, decoder graph export/import with inference-output roundtrip, decoder checkpoint coverage, and generation-control sampling contract are covered | Wire prompt-driven generation runtime/UI and full LM-stack training coverage |
| T5 | Not first-class | Defer until encoder-decoder memory/cross-attention contract is real |

## First Implementation Slice

1. Completed: add a computation-truth target for transformer primitives.
2. Completed: start with `EmbeddingModule` and `PositionalEncodingModule` because they are
   small, deterministic, and already in the training path.
3. Completed: use live LibTorch oracle values when LibTorch is available in
   non-Debug builds, with baked PyTorch-derived fallback values for Debug builds
   and builds without LibTorch.
4. Completed: keep the test target separate from large existing unit binaries.
5. Completed: do not expose new BERT/GPT/T5 UI capability from this slice.
6. Completed: add scaled dot-product attention parity coverage for unmasked and
   causal-masked attention using the existing language-model attention helper.
7. Completed: add focused `LayerNormForwardCpu` and `LayerNormBackwardCpu`
   primitive contracts with PyTorch parity coverage for forward, grad_input,
   grad_gamma, and grad_beta.
8. Completed: connect Studio/compiled `LayerNorm` model-layer configs to a real
   `LayerNormModule` in `SequentialModel` through ModelBuilder.
9. Completed: classify LayerNorm as a supported CPU-backed model layer in
   backend placement reporting so Studio/debugger truth does not overclaim GPU
   residency.
10. Completed: add checkpoint save/load regression for LayerNorm gamma/beta
    parameters through the public `CheckpointManager` API.
11. Completed: add `.cyxmodel` graph-backed export/import regression coverage
    for LayerNorm architecture rebuild and gamma/beta parameter round-trip.
12. Completed: add deterministic `MultiHeadAttentionLayer` forward parity for
    explicit Q/K/V/output projections and per-head attention weights.
13. Completed: add additive-mask parity for `MultiHeadAttentionLayer` output
    and per-head attention weights.
14. Completed: add deterministic self-attention backward parity for
    `MultiHeadAttentionLayer` input gradients plus Q/K/V/output projection
    weight and bias gradients.
15. Completed: add deterministic cross-attention parity for different
    query/key-value sequence lengths, per-head weights, and separate
    query/key/value gradients.
16. Completed: add key-padding-style additive mask parity for cross-attention
    output and per-head weights.
17. Completed: add explicit graph-compiler fail-closed regression for
    standalone `MultiHeadAttention`, including unsupported backend placement
    truth and stable unsupported-node error code.
18. Completed: name attention-family nodes explicitly in backend placement
    reporting instead of falling back to generic `Layer`.
19. Completed: add a real `MultiHeadAttentionModule` SequentialModel wrapper
    for single-input self-attention, ModelBuilder routing, CPU-backed placement
    truth, and DebugExecutor forward/backward gradient exposure coverage.
20. Completed: keep connected Key/Value/Context `MultiHeadAttention` graphs
    fail-closed until graph-level cross-attention contracts exist.
21. Completed: add `.cyxmodel` graph-backed export/import regression coverage
    for single-input `MultiHeadAttention` architecture rebuild and Q/K/V/O
    projection weight/bias parameter round-trip.
22. Completed: add debugger backend-classification contract coverage for
    CPU-backed single-input `MultiHeadAttention`, including proven CPU status,
    `graph_runtime_cpu_backed` reason, attention-worthy performance warning,
    and non-failing trace status.
23. Completed: add `.cyxmodel` graph-backed export/import regression coverage
    for `TransformerEncoder` architecture rebuild and full parameter
    round-trip, and classify `TransformerEncoder` as CPU-backed placement truth.
24. Completed: add deterministic `TransformerEncoderModule` forward
    computation-truth coverage against PyTorch primitive composition for
    self-attention, residuals, post-layernorm, ReLU feed-forward, and final
    post-layernorm.
25. Completed: add deterministic decoder-only causal
    `TransformerDecoderModule` forward computation-truth coverage against
    PyTorch primitive composition, including causal self-attention masking,
    residuals, post-layernorm, ReLU feed-forward, and final post-layernorm.
26. Completed: add tiny causal language-model logits/loss computation-truth
    coverage for verified decoder hidden states, vocabulary-head logits, and
    `CrossEntropyLoss` class-index targets against PyTorch semantics.
27. Completed: add `.cyxmodel` graph-backed export/import regression coverage
    for `TransformerDecoder` architecture rebuild and full parameter
    round-trip, and classify `TransformerDecoder` as CPU-backed placement
    truth.
28. Completed: add `CheckpointManager` parameter round-trip coverage for a
    `TransformerDecoder` + Dense-head stack and update graph-compiler
    decoder-only placement expectations from unclassified to CPU-backed.
29. Completed: extend `TransformerDecoder` `.cyxmodel` roundtrip to a valid
    decoder inference stack (`TransformerDecoder -> Flatten -> Dense`) and
    verify imported output matches source output for a deterministic input.
30. Completed: align the sidebar Transformer decoder pattern with the proven
    CPU-backed decoder starter stack by adding the explicit `Flatten -> Dense`
    logit head path and removing broad full-generation wording from the
    pattern description.
31. Completed: add a focused language-model generation-control contract for
    decoder-style models, including config validation, temperature scaling,
    top-k filtering, top-p/nucleus filtering, greedy selection, and seeded
    multinomial next-token sampling.
32. Completed: add a core `SequentialModel` runtime helper that applies the
    generation-control contract to prompt token IDs when the model returns
    `Float32 [1, seq, vocab]` logits.
33. Completed: add a separate sidebar `Causal LM Generation Starter` pattern
    that uses `TimeDistributed` as the token-logit head so the graph preserves
    `[batch, seq, vocab]` logits for prompt-generation runtime experiments.
34. Completed: add a dedicated Studio `Language Model Generation` panel under
    Tools > Machine Learning > Text Processing. The first UI slice accepts raw
    prompt token IDs, applies the reusable generation controls to the last
    trained model, and displays generated token IDs.
35. Completed: extend the Studio language-model generation panel with text
    prompt mode backed by the existing CyxWiz `Tokenizer` and vocabulary file
    format, including text-to-token encoding and generated-token decoding.
36. Completed: allow the Studio language-model generation panel to load
    packaged tokenizer assets directly from `.cyxmodel` packages via
    `tokenizer/config.json` and `tokenizer/vocab.txt`, with manual vocabulary
    file mode retained as fallback.
37. Completed: allow the Studio language-model generation panel to import a
    `.cyxmodel` model into a local generation slot and run generation from that
    imported model instead of only using `TrainingManager`'s last trained model.
38. Completed: add a Studio generation compatibility check that probes the
    active model with the current prompt and reports whether it returns
    `Float32 [1, seq, vocab]` logits before generation is attempted.
39. Completed: improve the Studio language-model generation panel with active
    model status, packaged tokenizer summary, and safer defaults from packaged
    tokenizer assets, including EOS token ID and max prompt length.
40. Completed: add `.cyxmodel` model-family generation metadata to manifests,
    probe results, and the Studio generation panel so decoder packages can
    declare `causal_lm` support and the expected `Float32[1,seq,vocab]`
    contract before runtime probing.
41. Completed: add a focused `.cyxmodel` generation metadata roundtrip test
    proving `model.family`, `supports_generation`, and
    `generation_output_contract` survive package creation and probe.
42. Completed: add an exporter-path `.cyxmodel` generation metadata test
    proving `ModelExporter` marks explicit causal-LM sequence exports as
    `causal_lm` generation packages.
43. Completed: add a causal-LM generation-shaped `.cyxmodel` export/import
    roundtrip test proving `TransformerDecoder -> TimeDistributedDense`
    packages preserve parameters and return `[batch, seq, vocab]` logits after
    import.
44. Completed: add focused graph-compiler coverage for causal-LM DatasetInput
    shape handling, proving `shape=[features]` plus `max_sequence_length`
    compiles as `[seq, features]` and allows `TransformerDecoder ->
    TimeDistributed` without a false shape error.
45. Completed: extend the causal-LM `.cyxmodel` generation roundtrip with
    packaged tokenizer assets, extraction, `LoadTextTokenizerPackage`, prompt
    token encoding, and generated-token decoding coverage.
46. Completed: update the sidebar causal-LM generation starter pattern with
    explicit causal-LM DatasetInput metadata: scalar token-id shape `[1]`,
    `create_causal_lm_targets=true`, `max_sequence_length`, and output classes
    tied to `vocab_size`.
47. Completed: refresh pattern-template guard expectations now that
    `MultiHeadAttention` is implemented and keep the guard passing for the
    updated causal-LM starter/template behavior.

## Next Implementation Slice

1. Do not broaden the legacy binary importer beyond its current Linear/activation
   fallback unless a real use case requires it.
2. Add graph-training launcher coverage once a small fixture can produce a
   trained causal-LM package with tokenizer assets without long runtime.
3. Follow-up tickets now start at `tofix49` because `tofix46` through
   `tofix48` already exist for unrelated work.
4. Use `tofix49` through `tofix55` for LM-stack inference, GPT-style generation
   UX, BERT-style coverage, transformer backward parity, cross-attention/T5
   readiness, CBOW, and deferred ELMo/T5 gates.
3. Add LayerNorm GPU coverage only after an ArrayFire implementation and
   residency/parity tests exist.
4. Keep connected Key/Value/Context attention blocked until compiler, runtime,
   export/load, and debugger capability truth all agree.
5. After LayerNorm and attention are proven end to end, move toward BERT-like
   encoder claims.
6. The tiny causal-LM fixture now exercises the real `LinearModule` vocabulary
   head with deterministic `weight`/`bias` parameters; the earlier logits
   mismatch was a fixture parameter-name error, not proven Dense GPU drift.

Validation target:

```powershell
cmake --build build --config Debug --target test_computation_truth_transformer_primitives -- /m:1 /v:minimal
build\bin\Debug\test_computation_truth_transformer_primitives.exe
cmake --build build --config Debug --target test_debug_executor -- /m:1 /v:minimal
build\bin\Debug\test_debug_executor.exe
cmake --build build --config Debug --target test_graph_compiler_deferred_nodes -- /m:1 /v:minimal
build\bin\Debug\test_graph_compiler_deferred_nodes.exe
cmake --build build --config Debug --target test_recurrent_backend_placement -- /m:1 /v:minimal
build\bin\Debug\test_recurrent_backend_placement.exe
cmake --build build --config Debug --target test_cyxmodel_layernorm_roundtrip -- /m:1 /v:minimal
build\bin\Debug\test_cyxmodel_layernorm_roundtrip.exe
cmake --build build --config Debug --target test_cyxmodel_mha_roundtrip -- /m:1 /v:minimal
build\bin\Debug\test_cyxmodel_mha_roundtrip.exe
cmake --build build --config Debug --target test_debugger_contracts -- /m:1 /v:minimal
build\bin\Debug\test_debugger_contracts.exe
cmake --build build --config Debug --target test_cyxmodel_transformer_encoder_roundtrip -- /m:1 /v:minimal
build\bin\Debug\test_cyxmodel_transformer_encoder_roundtrip.exe
cmake --build build --config Debug --target test_cyxmodel_transformer_decoder_roundtrip -- /m:1 /v:minimal
build\bin\Debug\test_cyxmodel_transformer_decoder_roundtrip.exe
cmake --build build --config Debug --target test_language_model_generation -- /m:1 /v:minimal
build\bin\Debug\test_language_model_generation.exe
cmake --build build --config Debug --target test_cyxmodel_generation_metadata -- /m:1 /v:minimal
build\bin\Debug\test_cyxmodel_generation_metadata.exe
cmake --build build --config Debug --target test_cyxmodel_exporter_generation_metadata -- /m:1 /v:minimal
build\bin\Debug\test_cyxmodel_exporter_generation_metadata.exe
cmake --build build --config Debug --target test_cyxmodel_causal_lm_generation_roundtrip -- /m:1 /v:minimal
build\bin\Debug\test_cyxmodel_causal_lm_generation_roundtrip.exe
cmake --build build --config Debug --target test_graph_compiler_causal_lm_shape -- /m:1 /v:minimal
build\bin\Debug\test_graph_compiler_causal_lm_shape.exe
cmake --build build --config Debug --target test_pattern_template_guard -- /m:1 /v:minimal
build\bin\Debug\test_pattern_template_guard.exe
```

`test_cyxmodel_causal_lm_generation_roundtrip` also covers packaged tokenizer
assets (`tokenizer/config.json`, `tokenizer/vocab.txt`), extraction,
`LoadTextTokenizerPackage`, prompt encoding, and generated-token decoding.

Latest focused validation: passed on 2026-07-02.

Validated targets:

- `cyxwiz-engine`
- `test_language_model_generation`
- `test_cyxmodel_generation_metadata`
- `test_cyxmodel_exporter_generation_metadata`
- `test_graph_compiler_causal_lm_shape`
- `test_cyxmodel_causal_lm_generation_roundtrip`
- `test_pattern_template_guard`

## Done Criteria

- Primitive parity tests exist for the shared transformer/NLP building blocks.
- New model-family code is split into focused translation units.
- Existing large files are not used as dumping grounds.
- Single-input standalone attention is supported only where compiler, runtime,
  tests, and placement truth agree; multi-input attention remains blocked.
- TransformerEncoder is graph/export/import truthful as a CPU-backed model
  layer with forward PyTorch parity, but BERT-style claims still require mask,
  backward, pooling/head, and inference-contract coverage.
- TransformerDecoder has decoder-only causal forward PyTorch parity plus
  source-vs-imported inference-output roundtrip, but GPT-style claims still
  require generation controls and full LM-stack training coverage.
- The sidebar decoder pattern is available as a truthful CPU-backed starter,
  not as a full GPT-style generation claim.
- Generation controls are implemented as a reusable contract with core
  prompt-token runtime wiring and a first Studio panel for raw token-ID
  generation plus vocabulary-backed and packaged-tokenizer text prompt
  encode/decode.
- The Studio panel can generate from either the last trained model or an
  imported `.cyxmodel` loaded into the panel.
- The Studio panel can preflight the active generation model for the required
  `Float32 [1, seq, vocab]` output contract.
- Packaged tokenizer summaries and defaults are surfaced in the Studio panel.
- `.cyxmodel` generation metadata has focused package/probe roundtrip coverage.
- Generation-shaped causal-LM `.cyxmodel` packages have export/import inference
  roundtrip coverage for `[batch, seq, vocab]` logits.
- Causal-LM DatasetInput sequence-shape compilation has focused regression
  coverage.
- Causal-LM `.cyxmodel` packages have tokenizer config/vocabulary extraction
  and generation prompt encode/decode coverage.
- The causal-LM generation starter declares the sequence shape and causal target
  metadata needed by compiler/export truth.
- The sidebar now has both a conservative Flatten-based decoder starter and a
  generation-shaped causal-LM starter with a `TimeDistributed` token head.
- Each model-family claim has compiler, runtime, training, export/load, and
  inference truth.
