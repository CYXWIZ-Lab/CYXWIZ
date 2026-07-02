# To Fix 51 - BERT-Style Encoder Graph, Head, And Inference Coverage

## Purpose

Move from transformer encoder primitives to truthful BERT-style encoder model
contracts.

`tofix28` proved transformer encoder primitives and export/import basics, but
that is not enough to claim BERT-style support.

## Scope

Support first-class encoder use cases:

- Sequence classification.
- Token classification.
- Optional pooled or CLS-style output.
- Attention mask handling.
- Export/import.
- Inference input contract.
- Debugger shape and placement truth.

## Required graph contracts

BERT-style graphs need explicit contracts for:

- Token IDs.
- Segment/token-type IDs if supported.
- Attention mask.
- Positional encoding.
- Encoder stack.
- Pooling or CLS extraction.
- Classification head.
- Loss and metrics.

## Inference contracts

Support at least:

- Text input with tokenizer assets.
- Encoded tensor input for advanced users.
- Classifier output: `[batch, classes]`.
- Token-classifier output: `[batch, seq, classes]`.

The engine must fail closed when a graph requires unsupported segment IDs,
mask shape, pooling behavior, or export behavior.

## Tests

Add tests for:

- Encoder classifier graph compile.
- Token classification graph compile.
- Attention mask shape validation.
- Export/import parameter roundtrip.
- Inference output shape.
- Debugger placement report.
- Failure for unsupported BERT features.

## Non-goals

- Do not claim pretrained BERT checkpoint compatibility yet.
- Do not implement HuggingFace model loading here.
- Do not add GPU transformer kernels here.

## Completion criteria

- BERT-style encoder claims are limited to tested graph/head/inference paths.
- Unsupported BERT features fail with clear messages.
- Documentation describes exactly what is and is not supported.
