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

## Phase 2 - CyxModel Metadata And Placement Coverage

Status: complete for the CyxModel metadata and placement slice.

Scope:

- Added typed `.cyxmodel` manifest/probe/import metadata for BERT-style encoder
  contracts.
- Export now infers supported BERT-style encoder metadata from graph JSON for
  explicit sequence-classifier and token-classifier paths.
- Extended the TransformerEncoder `.cyxmodel` roundtrip to prove BERT sequence
  and token classifier metadata plus head parameters survive export/import.
- Extended BERT contract tests to pin debugger/backend placement reporting for
  CPU-backed `TransformerEncoder` and explicit `TimeDistributed` wrapper warning
  semantics.

Validated:

- `cmake --build D:\Dev\CyxWiz_Claude\build --config Debug --target test_bert_encoder_contracts test_cyxmodel_transformer_encoder_roundtrip -- /m:4 /v:minimal`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_bert_encoder_contracts.exe`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_cyxmodel_transformer_encoder_roundtrip.exe`
- `cmake --build D:\Dev\CyxWiz_Claude\build --config Debug --target test_cyxmodel_generation_metadata test_cyxmodel_exporter_generation_metadata test_cyxmodel_sequence_assets -- /m:4 /v:minimal`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_cyxmodel_generation_metadata.exe`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_cyxmodel_exporter_generation_metadata.exe`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_cyxmodel_sequence_assets.exe`

## Phase 3 - PyTorch Encoder Head Oracle Fixtures

Status: complete for the PyTorch encoder-head oracle slice.

Scope:

- Added computation-truth coverage for BERT-style CLS extraction,
  sequence-classifier logits, and token-classifier logits.
- Uses LibTorch as the direct oracle when `CYXWIZ_HAS_PYTORCH` is enabled
  outside Debug; otherwise uses tiny PyTorch-derived constants in the C++ test.
- Keeps PyTorch oracle coverage in tests only and does not add a runtime
  dependency.

Validated:

- `cmake --build D:\Dev\CyxWiz_Claude\build --config Debug --target test_computation_truth_transformer_primitives -- /m:4 /v:minimal`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_computation_truth_transformer_primitives.exe`
- `cmake --build D:\Dev\CyxWiz_Claude\build --config Release --target test_computation_truth_transformer_primitives -- /m:4 /v:minimal`
- `$env:PATH = 'D:\tmp\libtorch-cpu-2.7.0\libtorch\lib;' + $env:PATH; D:\Dev\CyxWiz_Claude\build\bin\Release\test_computation_truth_transformer_primitives.exe`

## Phase 4 - Packaged Text Inference Contract

Status: complete for the BERT packaged-text inference boundary.

Scope:

- Added a BERT encoder inference contract helper for `.cyxmodel` packages with
  tokenizer/vocabulary assets.
- Validates BERT package metadata for `bert_encoder`, `token_ids` input,
  sequence/token classifier output contracts, and fail-closed token_type/segment
  IDs.
- Local inference now routes raw string `/v1/predict` input for BERT encoder
  packages through token-id sequence input instead of the legacy float-vector
  text path.
- Builds a BERT attention mask from the packaged tokenizer pad ID when the
  model declares attention-mask support.
- Validates BERT sequence/token classifier runtime logits before formatting the
  prediction response.
- Exposes BERT inference contract details in `/v1/model`.

Validated:

- `cmake --build D:\Dev\CyxWiz_Claude\build --config Debug --target test_language_model_inference_contract -- /m:4 /v:minimal`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_language_model_inference_contract.exe`
- `cmake --build D:\Dev\CyxWiz_Claude\build --config Debug --target cyxwiz-engine -- /m:4 /v:minimal`
- `cmake --build D:\Dev\CyxWiz_Claude\build --config Debug --target test_bert_encoder_contracts test_cyxmodel_transformer_encoder_roundtrip test_cyxmodel_sequence_assets -- /m:4 /v:minimal`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_bert_encoder_contracts.exe`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_cyxmodel_transformer_encoder_roundtrip.exe`
- `D:\Dev\CyxWiz_Claude\build\bin\Debug\test_cyxmodel_sequence_assets.exe`

## Next Pickup

No current implementation work is pending for the tested tofix51 scope.
Future follow-up moved to `tofix65.md`: add a Studio-level debugger support-bundle/headless UI smoke over the same BERT placement metadata when that harness exists.
