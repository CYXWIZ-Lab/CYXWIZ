# Track 51 - BERT-Style Encoder Graph, Head, And Inference Coverage

## Reference Policy

`done28.md` section `0a. PyTorch numerical reference and open-source oracle
policy` remains the compute reference for transformer/NLP primitives and
model-family fixtures. PyTorch reference checks stay in tests, fixtures, or
generated expected-output artifacts; the core runtime must not gain a PyTorch
dependency for this ticket.

## Phase 1 - Encoder Contract Boundary

Status: complete for the first contract slice.

Implemented scope:

- Added a small BERT-style encoder contract boundary in
  `src/core/bert_encoder_contract.h`.
- Added a passive `BertEncoderGraphContract` to `TrainingConfiguration`.
- Wired `GraphCompiler` to populate the contract from the selected graph plan
  and surface blocker messages as normal compile errors only when a graph
  declares or clearly forms a BERT-style encoder path.
- Covered explicit CLS sequence classification:
  `Embedding -> PositionalEncoding -> TransformerEncoder -> TensorIndexSelect`
  `-> Squeeze -> Dense`.
- Covered token classification:
  `Embedding -> PositionalEncoding -> TransformerEncoder -> TimeDistributed`.
- Added token input validation for `[batch, seq]` token IDs, attention-mask
  shape matching, max-sequence bounds, and fail-closed token_type/segment IDs.
- Added runtime output validation for sequence classifier logits
  `[batch, classes]` and token classifier logits `[batch, seq, classes]`.

Guardrails:

- Generic `TransformerEncoder -> Dense` graphs are not reclassified as
  BERT-style unless they declare the BERT-style contract or use an explicit
  CLS/pooling/token-head path.
- Token_type/segment IDs are not supported yet and now fail closed when a
  declared BERT-style graph requires them.
- This slice does not claim pretrained BERT checkpoint compatibility,
  HuggingFace loading, GPU transformer kernels, or full attention-bias mask
  parity.

Validated:

- `cmake --build D:\Dev\CyxWiz_Claude\build --config Debug --target test_bert_encoder_contracts -- /m:4 /v:minimal`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_bert_encoder_contracts.exe`
- `cmake --build D:\Dev\CyxWiz_Claude\build --config Debug --target test_graph_compiler_deferred_nodes test_graph_compiler_causal_lm_shape test_cyxmodel_transformer_encoder_roundtrip -- /m:4 /v:minimal`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_graph_compiler_deferred_nodes.exe`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_graph_compiler_causal_lm_shape.exe`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_cyxmodel_transformer_encoder_roundtrip.exe`

## Next Pickup

Continue with export/import and debugger coverage:

- Add `.cyxmodel` metadata/probe support for the BERT-style encoder family,
  including explicit sequence-classifier and token-classifier output contracts.
- Add a focused roundtrip that proves encoder head parameters and contract
  metadata survive export/import.
- Add debugger placement/report coverage for supported BERT-style encoder
  paths, including CPU-backed `TransformerEncoder` and `TimeDistributed`
  warning semantics.
- Add PyTorch oracle fixtures for CLS pooling/token-head logits once the
  metadata/export boundary is in place.
