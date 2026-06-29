# tofix31b - TransformerDecoder runtime contract

## Scope

This is the second small ticket extracted from `tofix31.md`.

The goal is not to implement full seq2seq or pretrained generation support yet.
The goal is to make the current TransformerDecoder boundary explicit and
truthful across compiler behavior, tests, and Studio UI documentation.

## Current engine contract

- Supported now: decoder-only causal self-attention stacks for tested
  language-model style training paths.
- Rejected now: selected TransformerDecoder paths with connected `Memory`
  input, because Studio does not yet have a first-class seq2seq/cross-attention
  graph contract.
- Rejected now: fine-tuning/generative training sketches that imply pretrained
  transformer import or generation-loop behavior without trainable weight import
  and an explicit generation runtime.
- Backend status: lower-level `TransformerDecoderLayer` has decoder-only and
  memory overloads, but Studio graph compilation intentionally exposes only the
  tested decoder-only contract.

## Code locations checked

- `cyxwiz-engine/src/core/graph_compiler.cpp`
  - Rejects selected TransformerDecoder Memory-input paths.
  - Rejects decoder/generative fine-tuning sketches without a real contract.
- `cyxwiz-engine/tests/test_graph_compiler_deferred_nodes.cpp`
  - Covers Memory-input rejection.
  - Covers decoder-only compile allowance.
  - Covers side/unselected decoder nodes not blocking compile.
- `cyxwiz-backend/src/algorithms/layers/transformer_layers.cpp`
  - Contains lower-level decoder-only and memory overload implementations.
- `cyxwiz-engine/src/gui/node_editor_nodes.cpp`
  - Memory pin text already states that connected Memory requires a future
    seq2seq contract.
- `cyxwiz-engine/src/gui/node_documentation.cpp`
  - Updated to avoid implying first-class cross-attention or generation-loop
    support.
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp`
  - Already describes TransformerDecoder as tested causal language-model stack
    support.

## Changes made

- Studio node documentation now describes TransformerDecoder as currently
  supporting tested decoder-only causal self-attention stacks.
- Studio node documentation now explicitly marks encoder-decoder Memory paths
  and autoregressive generation loops as future contract work.
- TransformerDecoder output-pin help now points decoder-only LM users to a
  projection head and avoids suggesting current seq2seq Memory wiring.

## Not done in this ticket

- No seq2seq training contract.
- No autoregressive generation runtime.
- No pretrained transformer weight import or fine-tuning contract.
- No change to lower-level backend decoder overloads.

## Acceptance check

- Existing compiler behavior remains the source of truth.
- UI documentation no longer over-promises cross-attention or generation.
- Future work can now be split into concrete implementation tickets:
  - first-class seq2seq graph contract,
  - generation loop runtime,
  - pretrained transformer import/fine-tuning.
