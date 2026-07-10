# To Fix 65 - BERT Encoder Studio Harness Follow-up

## Status

Open.

## Background

`done51` / `donetrack51` completed the tested BERT-style encoder scope:

- explicit sequence-classifier and token-classifier graph contracts,
- fail-closed unsupported token_type/segment IDs,
- `.cyxmodel` BERT encoder metadata export/probe/import,
- backend/debugger placement truth for supported BERT graph paths,
- PyTorch-derived oracle coverage for CLS extraction and classifier logits,
- packaged text inference contracts for BERT encoder `.cyxmodel` packages,
- local inference raw-text routing through token IDs and optional attention masks.

The remaining work is not core BERT runtime support. It is a Studio/debugger
harness follow-up that should only proceed when an existing lightweight harness
can exercise the behavior without creating broad UI test infrastructure.

## Problem

The engine exposes BERT placement and inference metadata through focused tests
and `/v1/model` contract fields. There is not yet a Studio-level render/export
or support-bundle smoke over the same metadata.

That gap should be closed only if a true debugger support-bundle or headless UI
harness exists. Until then, adding UI framework work would be broader than the
BERT contract being validated.

## Required Scope

1. Add a Studio-level smoke that loads or constructs a BERT encoder graph with
   explicit sequence or token classification metadata.
2. Verify the debugger/support-bundle surface includes:
   - model family `bert_encoder`,
   - BERT task,
   - input kind `token_ids`,
   - output contract,
   - attention-mask support,
   - fail-closed token_type/segment ID status,
   - backend placement reasons for `TransformerEncoder` and `TimeDistributed`.
3. If endpoint behavior is touched, verify `/v1/model` still reports the BERT
   inference contract fields added in `done51`.
4. Keep the test on an existing harness path. Do not create a broad new UI test
   framework for this ticket.

## Validation

- New Studio/support-bundle/headless harness test builds and passes in Debug.
- Existing BERT contract tests still pass:
  - `test_bert_encoder_contracts.exe`
  - `test_cyxmodel_transformer_encoder_roundtrip.exe`
  - `test_language_model_inference_contract.exe`
- `cyxwiz-engine` Debug build still passes.
- Run endpoint or release builds only if this ticket changes server/runtime
  behavior.

## Non-goals

- Do not reopen `done51`.
- Do not add pretrained BERT checkpoint compatibility.
- Do not add HuggingFace loading.
- Do not add GPU transformer kernels.
- Do not add token_type/segment ID support.
- Do not build a new GUI test framework solely for this follow-up.

## Recommended First Slice

Locate the existing lightest Studio/debugger harness that can inspect exported
or rendered support-bundle metadata. If no such harness exists, leave this ticket
open with that blocker instead of expanding the UI test surface.
