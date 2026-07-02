# To Fix 55 - Deferred ELMo And T5 Readiness Gate

## Purpose

Prevent premature ELMo/T5 claims.

ELMo and T5 are important model families, but both depend on contracts that are
not yet complete. This ticket records the readiness gate so future work does not
overstate engine capability.

## ELMo readiness blockers

ELMo should remain deferred until:

- Recurrent sequence contracts are stronger.
- Bidirectional recurrent export/import is fully covered.
- Character/subword input handling is defined.
- Sequence output and pooling behavior are tested.
- Training and inference contracts are documented.

## T5 readiness blockers

T5 should remain deferred until:

- Cross-attention contract is implemented.
- Encoder-decoder graph support is tested.
- Source and target tokenization contracts exist.
- Decoder shifted targets are materialized.
- Text-to-text generation inference is covered.
- Export/import preserves encoder-decoder topology.

## Policy

Until blockers are complete:

- Do not expose ELMo or T5 as implemented patterns.
- Do not label generic transformer blocks as ELMo/T5 support.
- If shown in UI, mark as planned/deferred with reason.
- Docs must point to the missing readiness gates.

## Completion criteria

- ELMo and T5 are either hidden or clearly marked deferred.
- Any future implementation ticket references the blockers above.
- Studio and docs do not imply these families are currently supported.
