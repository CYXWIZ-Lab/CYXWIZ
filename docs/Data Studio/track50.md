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

## Next Pickup

Add Studio-facing smoke coverage around the panel metadata path if a practical
harness is available. The next numerical slice should add PyTorch-backed or
generated PyTorch expected-output fixtures for deterministic sampling/logit
parity without bringing PyTorch into the engine runtime boundary.