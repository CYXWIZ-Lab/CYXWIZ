# tofix64 - LM Inference Endpoint and Studio Harness Follow-up

## Status

Open.

## Background

`tofix49` / `track49` completed the full LM-stack inference contract:

- causal-LM `.cyxmodel` packages can be loaded and validated,
- packaged tokenizer assets are loaded from the same package,
- model-family metadata and `Float32 [1, seq, vocab]` runtime logits are checked,
- prompt text can be encoded through the packaged tokenizer,
- imported causal-LM models can generate token IDs through the normal generation path,
- generated token IDs can be decoded back to text,
- Studio and the local inference server surface contract failures before generation.

The remaining work is follow-up hardening, not core `tofix49` completion work.
Keep this ticket focused on proving the existing contract through boundary tests
and release checks.

## Problem

The shared LM inference contract is covered by focused engine tests and a package
roundtrip test. The local HTTP server wiring is implemented, but `/v1/model` and
`/v1/generate` do not yet have a direct endpoint-level integration test using a
generated causal-LM package.

The Studio generation panel is compile-validated through the engine build, but it
has no direct GUI-harness test for imported package contract behavior. That test
should only be added if an existing GUI harness can exercise the panel without
creating a broad new UI test framework.

## Required Scope

1. Add a direct local inference server integration test for a generated causal-LM
   `.cyxmodel` package.
2. The test should load the package, call `/v1/model`, and verify:
   - model family is `causal_lm`,
   - generation support is reported,
   - tokenizer vocabulary size is present,
   - max sequence length is present,
   - EOS token ID is visible,
   - `language_model_contract.compatible` is true.
3. The test should call `/v1/generate` with text input and verify:
   - prompt token IDs are returned,
   - generated token IDs are returned,
   - decoded text is returned,
   - EOS/default generation settings are visible,
   - incompatible contract errors return structured JSON instead of falling
     through to generation.
4. Add Studio panel test coverage only if an existing lightweight GUI harness can
   exercise imported package contract rejection without new framework work.
5. Run a broader release gate only if the endpoint integration test changes
   runtime/server behavior.

## Validation

- New endpoint test builds and passes in Debug.
- Existing focused LM tests still pass:
  - `test_language_model_inference_contract.exe`
  - `test_cyxmodel_causal_lm_generation_roundtrip.exe`
- `cyxwiz-engine` Debug build still passes.
- If server code changes, run the relevant local inference server tests and a
  Release `cyxwiz-engine` build.
- If Studio panel behavior changes, run `cyxwiz-engine` Debug build and any
  existing GUI harness test that already covers the panel.

## Non-goals

- Do not reopen `tofix49`.
- Do not add a new model family.
- Do not add GPT, BERT, encoder-decoder, cross-attention, or training features.
- Do not build a new GUI test framework just for this ticket.
- Do not change HTTP API shape unless the integration test exposes a real
  contract bug.
- Do not optimize generation kernels.

## Recommended First Slice

Add the endpoint integration test only. Reuse the existing causal-LM package
fixture path from `test_cyxmodel_causal_lm_generation_roundtrip` where practical,
but avoid copying large setup blocks if a small helper can keep the test readable.

## Completion Criteria

- `/v1/model` and `/v1/generate` are directly tested against a generated
  causal-LM package.
- The test proves success and at least one structured contract-failure path.
- Existing `tofix49` contract and package-roundtrip tests remain green.
- `track49.md` continues to mark the main LM inference contract as complete and
  points optional hardening work to this ticket.