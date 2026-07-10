# Track 50 - GPT-Style Generation Controls And UX

## Reference Policy

`done28.md` section `0a. PyTorch numerical reference and open-source oracle policy`
is the compute reference for this ticket. PyTorch remains the numerical oracle
for transformer/NLP primitives, generation logits, probability behavior, and
deterministic sampling. Reference checks must stay outside the core runtime as
tests, fixtures, or generated expected-output artifacts.

This first slice does not add a runtime PyTorch dependency. It extends the
native generation contract and Studio surface, while the focused C++ tests keep
tiny deterministic fixtures for the report behavior. Broader PyTorch parity
fixtures remain the follow-up path when LibTorch/Python fixture plumbing is
available for this target.

## Phase 1 - Generation Run Report Contract

Status: complete for the first implementation slice.

Implemented scope:

- Added `LanguageModelGenerationResult` with generated IDs, new IDs, per-token
  steps, stop reason, prompt length, max-new-token budget, remaining budget, and
  include-prompt flag.
- Added stable stop reason names for Studio and tests.
- Kept `GenerateTokenIdsWithConfig` as the compatibility wrapper over the new
  report API.
- Added max-context prompt validation before inference.
- Surfaced Studio run metadata: prompt length, max context length, remaining
  generation budget, stop reason, sampling settings, generated IDs/text, and
  last-token candidate diagnostics.

Validated:

- `cmake --build D:\Dev\CyxWiz_Claude\build --config Debug --target test_language_model_generation cyxwiz-engine`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_language_model_generation.exe`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_language_model_inference_contract.exe`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_cyxmodel_causal_lm_generation_roundtrip.exe`

## Phase 2 - PyTorch Reference Sampling Fixture

Status: complete for the PyTorch reference sampling slice.

Scope:

- Extended computation-truth coverage for GPT-style generation probability
  behavior without adding PyTorch to the runtime boundary.
- Uses LibTorch as the direct reference when `CYXWIZ_HAS_PYTORCH` is enabled
  outside Debug; otherwise uses tiny PyTorch-derived constants in the C++ test.
- Covers temperature scaling, top-k, top-p, candidate ordering,
  renormalization, and greedy candidate selection.

Validated:

- `cmake --build D:\Dev\CyxWiz_Claude\build --config Debug --target test_computation_truth_transformer_primitives -- /m:4 /v:minimal`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_computation_truth_transformer_primitives.exe`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_language_model_generation.exe`
- `cmake --build D:\Dev\CyxWiz_Claude\build --config Release --target test_computation_truth_transformer_primitives -- /m:4 /v:minimal`
- `$env:PATH = 'D:\tmp\libtorch-cpu-2.7.0\libtorch\lib;' + $env:PATH; D:\Dev\CyxWiz_Claude\build\bin\Release\test_computation_truth_transformer_primitives.exe`

## Next Pickup

Add Studio-facing smoke coverage around the panel metadata path if a practical
harness is available. Remaining numerical follow-up is deterministic
multinomial parity policy: keep any PyTorch oracle in tests/fixtures and do not
force PyTorch into the engine runtime boundary.
