# Track 49 - Full LM-Stack Inference Contract

## Phase 1 - Contract Boundary

Status: completed.

Lean scope:

- Keep the first phase as a reusable inference-contract boundary.
- Do not wire Studio or server callers until the contract helper is proven by focused tests.
- Reuse existing tokenizer package loading, prompt encoding/decoding, and generation runtime primitives.

Implementation target:

- Add `src/inference/language_model_inference_contract.{h,cpp}`.
- Validate `.cyxmodel` probe metadata for causal-LM generation:
  - `model_family == causal_lm`
  - `supports_generation == true`
  - `generation_output_contract == Float32[1,seq,vocab]`
  - tokenizer config and vocabulary assets present
- Surface tokenizer vocabulary size, max sequence length, and EOS token ID from loaded tokenizer assets.
- Validate prompt token IDs as `[1, seq]` compatible.
- Validate runtime logits as `Float32 [1, seq, vocab]`.
- Treat model vocab as compatible when it is at least as large as tokenizer vocabulary size.

Tests:

- `test_language_model_inference_contract`
  - accepts causal-LM package metadata with tokenizer assets
  - rejects missing tokenizer assets
  - rejects classifier/non-generation metadata
  - validates prompt token IDs
  - accepts runtime logits with compatible vocab width
  - rejects classifier-shaped runtime output
  - rejects model vocab smaller than tokenizer vocab

Verification:

- `cmake --build D:\Dev\CyxWiz_Claude\build --config Debug --target test_language_model_inference_contract`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_language_model_inference_contract.exe`

## Phase 2 - Local Inference Server Wiring

Status: completed.

Implementation:

- `LocalInferenceServer` caches the shared language-model package contract after `.cyxmodel` load.
- `/v1/model` now reports causal-LM generation capability, tokenizer details, EOS ID, max sequence length, and the shared contract result.
- `/v1/generate` now rejects incompatible packages before generation, validates prompt token IDs, validates preflight runtime logits, honors the contract EOS by default, and calls `GenerateTokenIdsWithConfig`.
- Unloading the model clears the cached language-model contract.

Verification:

- `cmake --build D:\Dev\CyxWiz_Claude\build --config Debug --target cyxwiz-engine`

## Phase 3 - Studio Generation Panel Wiring

Status: completed.

Implementation:

- The Language Model Generation panel now validates package compatibility through the shared contract helper.
- Prompt token IDs and preflight runtime logits use the same shared validators as the server path.
- Local Studio generation now uses `GenerateTokenIdsWithConfig`.
- The panel compatibility summary now includes model family, generation output contract, tokenizer vocabulary size, max sequence length, and EOS ID.

Verification:

- `cmake --build D:\Dev\CyxWiz_Claude\build --config Debug --target cyxwiz-engine`

## Phase 4 - Package Roundtrip Generation Test

Status: completed.

Implementation:

- `test_cyxmodel_causal_lm_generation_roundtrip` now builds a real token-ID causal-LM graph: `Embedding -> TransformerDecoder -> TimeDistributedDense`.
- The exported `.cyxmodel` roundtrip validates package metadata, tokenizer assets, runtime logits, and generated token IDs through the shared contract.
- The test covers generation from an imported `.cyxmodel` instead of only classifier-shaped forward compatibility.

Verification:

- `cmake --build D:\Dev\CyxWiz_Claude\build --config Debug --target test_cyxmodel_causal_lm_generation_roundtrip`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_cyxmodel_causal_lm_generation_roundtrip.exe`

## Release Readiness Check - 2026-07-10

Status: completed.

Review:

- Ran the CR skill against the full uncommitted tree.
- No concrete blocker was found in the changed engine or docs surfaces.
- The generic CR skill references `cyxtrade/CLAUDE.md`-style docs, which do not exist in this CyxWiz repository; review used the root README, local docs, changed code, and adjacent tests instead.

Debug verification:

- `cmake --build D:\Dev\CyxWiz_Claude\build --config Debug --target test_debugger_contracts test_language_model_inference_contract test_cyxmodel_causal_lm_generation_roundtrip`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_debugger_contracts.exe`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_language_model_inference_contract.exe`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_cyxmodel_causal_lm_generation_roundtrip.exe`

Release verification:

- `cmake --build D:\Dev\CyxWiz_Claude\build --config Release --target cyxwiz-engine`
- `cmake --build D:\Dev\CyxWiz_Claude\build --config Release --target test_debugger_contracts test_language_model_inference_contract test_cyxmodel_causal_lm_generation_roundtrip`
- First combined Release test build was interrupted by a command timeout and then hit transient `.obj` permission errors; retrying the failed `test_cyxmodel_causal_lm_generation_roundtrip` target by itself passed.
- `D:\Dev\CyxWiz_Claude\build\bin\Release\test_debugger_contracts.exe`
- `D:\Dev\CyxWiz_Claude\build\bin\Release\test_language_model_inference_contract.exe`
- `D:\Dev\CyxWiz_Claude\build\bin\Release\test_cyxmodel_causal_lm_generation_roundtrip.exe`

Commit state:

- Committed as `aca3f02e Complete track49 LM inference contract`.
- Pushed to `origin/Nodes_Implementation`.
- Track49 implementation and verification are closed in that pushed checkpoint.
- Later dirty work in the workspace belongs to the track32 debugger diagnostic continuation, not track49.

## Next Pickup

- Optional: add a direct HTTP integration test for `/v1/model` and `/v1/generate` using a generated causal-LM package.
- Optional: extend the Studio panel test surface if a GUI harness becomes available.
- Optional: run the full engine regression suite if a broader release gate is required beyond the focused Debug/Release tests above.
