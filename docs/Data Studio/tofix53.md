# To Fix 53 - Cross-Attention And Encoder-Decoder Contract

## Purpose

Define the cross-attention and encoder-decoder contract required before
T5-style models can be claimed.

Current transformer work focuses on encoder-only and decoder-only paths.
Encoder-decoder models need stronger multi-input graph, runtime, export/import,
and debugger truth.

## Scope

- Encoder memory tensor contract.
- Decoder query input contract.
- Connected key/value/context inputs.
- Cross-attention mask contract.
- Encoder attention mask.
- Decoder causal mask.
- Export/import of multi-input attention graphs.
- Inference contract for encoder-decoder models.

## Required graph model

The graph must represent:

- Source token input.
- Target/decoder token input.
- Encoder stack.
- Decoder stack.
- Cross-attention connection from encoder memory to decoder.
- Token head.
- Loss target shifting.

## Compiler requirements

- Multi-input attention pins must be validated.
- Shape compatibility must be checked before training.
- Unsupported connected key/value/context paths must fail closed.
- Export/import must preserve the multi-input graph.
- Debugger must show encoder memory and decoder input shapes.

## Tests

Add tests for:

- Valid encoder-decoder compile.
- Invalid memory shape.
- Invalid mask shape.
- Export/import graph roundtrip.
- Inference shape.
- Clear failure for unsupported cross-attention runtime.

## Non-goals

- Do not claim T5 support until this contract is implemented and tested.
- Do not add text-to-text dataset materialization in this ticket unless required
  for the smallest smoke test.

## Completion criteria

- Cross-attention support is truthful across compiler, runtime, export/import,
  debugger, and docs.
- T5-style support remains blocked until this ticket is complete.
