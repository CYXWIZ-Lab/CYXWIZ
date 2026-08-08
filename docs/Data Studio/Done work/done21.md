# Done 21 - NER And Sequence Graph Work

**Created:** 2026-06-10
**Source:** Follow-up after completing `done19.md`.

## Boundary

`done19.md` completed the practical sequence tagging training slice:
`TimeDistributedDense`, prebuilt `ISequenceBatcher` training, token-shaped
loss, token accuracy, BIO F1, sequence metrics in Studio, and guards for the
shipped NER graph.

This file tracks the NER/sequence follow-up work that happened after that
slice. Items marked complete describe the current supported scope rather than
an unlimited NER product surface.

## Tracked Work

## Status 2026-06-22

Started example-asset portability work for item 2:

- added a tiny checked-in CoNLL-style `sample_ner.csv` for smoke/demo use,
- changed the NER prep and inference helpers to default to paths inside
  `examples/cyxgraph/NER`,
- changed generated metadata to store paths relative to the metadata file so
  generated files can move with the example directory,
- changed the saved graph's dataset and vocabulary references from local
  absolute Windows paths to repository-relative example paths,
- updated the README commands and proof-run notes to avoid machine-specific
  paths.
- added a saved-graph regression guard in `test_pattern_template_guard` so the
  shipped NER graph keeps first-class sequence nodes and repository-relative
  dataset/vocabulary paths instead of regressing to local absolute paths.
- corrected stale serialized node type ids in the shipped NER graph for the
  data input, sequence builder, vocabulary nodes, and token padding node.
- ignored generated NER demo outputs locally so smoke runs do not create
  accidental untracked example artifacts.

At this point, the example was still a target design until the full Studio
graph launch path could materialize sequence batches from saved graph data.

Runtime progress:

- `TokenVocabulary`, `POSVocabulary`, and `NERTagVocabulary` now execute as
  narrow `PipelineExecutor` table transforms over prepared sentence-level
  string columns.
- The vocabulary nodes reuse the existing deterministic `SequenceVocabulary`
  helper and output `value,id` Arrow tables for downstream graph/runtime use.
- Focused routing coverage now proves deterministic token/POS/tag vocabulary
  output and schema failure on a missing vocabulary source column.
- `NERSequenceBuilder` now executes as a narrow `PipelineExecutor`
  materialization transform over prepared token/POS/tag string columns. It
  reuses the existing `NERSequenceBuilder` core helper and emits encoded
  `word_ids`, `pos_ids`, `tag_ids`, `attention_mask`, and `sequence_length`
  columns.
- Studio graph training can now build a `SequenceBatcher` directly from a
  registered Arrow sentence table and route sequence/NER graphs through
  `TrainingManager::StartTrainingSequence`, reusing the same
  `NERSequenceBuilder` core helper instead of duplicating token/tag encoding.
- The compiled sequence contract now carries `max_sequence_length` from
  first-class NER/DataLoader nodes into the Arrow sequence batcher, so Studio
  training and runtime materialization agree on sequence padding length.
- Sequence graph launch now normalizes the training config from the runtime
  vocabularies: input sequence length, embedding vocabulary size, tag output
  size, and final `TimeDistributed` token-head width.
- At this point, full NER graph launch still needed the higher-level model
  launch and inference packaging that consumes these prepared sequence
  artifacts.
- Added `validate_ner_graph_assets.py` for a repo-portable smoke check that
  verifies saved-graph path references and generated asset existence.
- Added a dedicated saved-graph launch regression in
  `test_text_gui_training_launch.cpp` that exercises Studio launch against the
  checked-in NER sequence fixture table, including materializer path, batcher
  config passthrough, and callback lifecycle checks.

Status 2026-06-23:

- Portable NER smoke validation now passes:
  `python examples\cyxgraph\NER\validate_ner_graph_assets.py`
- Duplicate path-prefix regression fixed in the validator (graph-reported
  `examples/cyxgraph/NER/...` paths now resolve correctly without duplicating
  the example directory).
- Item 3 end-to-end label/vocab consistency now has decode-level assertions in
  the saved-NER launch regression test: sentence split behavior, tag vocab size,
  and sequence prediction decode all flow from materialized data through batcher
  and model logits.
- Item 4 sequence inference packaging is partially wired: CyxModel manifests
  now carry sequence metadata/vocabulary paths, local model load probes and
  extracts sequence vocabularies, `/v1/model` exposes sequence decode
  capabilities, `/v1/predict` accepts named sequence tensors with
  `sequence_lengths`, and the NER helper prefers decoded `sequence.tag_labels`
  before falling back to logits.
- The portable NER smoke validator now builds a tokenized inference payload
  from generated metadata and checks the sequence tensor contract locally.
- Added focused C++ coverage for `.cyxmodel` sequence packaging:
  `test_cyxmodel_sequence_assets` creates a package with token/POS/tag
  vocabularies, probes sequence metadata, and extracts the vocabulary assets
  back out.
- Item 5 UI polish started: the node browser metadata for Token Vocabulary,
  POS Vocabulary, and NER Tag Vocabulary now marks the nodes as implemented
  runtime nodes instead of stale templates, with descriptions aligned to the
  executable `value,id` vocabulary table output.
- Sequence graph launch now performs a compact pre-dispatch schema check after
  materialization and reports missing token/tag/POS/sentence-id columns with
  the dataset name instead of falling through to a generic launch failure.
- Token classifier/loss UI labels are now task-aware: TimeDistributed appears
  as a dense token head, and CrossEntropy metadata exposes token CE usage plus
  the padding `ignore_index` control.
- The properties panel now has sequence-specific editors for
  `NERSequenceBuilder`, `TokenVocabulary`, `POSVocabulary`, and
  `NERTagVocabulary`, using the runtime parameter keys directly and showing
  compact guidance for required columns, padding, vocabulary limits, and BIO
  tag behavior.
- Sequence inference responses now decode tag IDs through a shared
  length-aware helper. When `/v1/predict` receives `sequence_lengths`, the
  returned `sequence.tag_ids` and `sequence.tag_labels` are clipped to the
  non-padding token lengths and include `sequence.effective_lengths`.
- `test_cyxmodel_sequence_assets` now covers `.cyxmodel` sequence asset
  round-trip plus length-aware BIO decode behavior, including a guard for
  mismatched `sequence_lengths`.
- Verified:
  `cmake --build build --config Debug --target test_cyxmodel_sequence_assets`,
  `cmake --build build --config Debug --target cyxwiz-engine`,
  `build\bin\Debug\test_cyxmodel_sequence_assets.exe`, and
  `python examples\cyxgraph\NER\validate_ner_graph_assets.py`.

### 1. Executable Vocabulary Nodes

`TokenVocabulary`, `POSVocabulary`, and `NERTagVocabulary` started as
first-class contract/documentation nodes without executable graph runtime
behavior.

Implement:

- graph-runtime execution for token/POS/tag vocabulary construction,
- vocabulary persistence and reload behavior,
- deterministic ID assignment for train/inference parity,
- metadata/status updates once execution is wired,
- tests that prove vocab nodes are either executable or clearly fail closed.

Status 2026-06-10:

- Started. `TokenVocabulary`, `POSVocabulary`, and `NERTagVocabulary` remain
  first-class sequence-training contract nodes rather than executable
  PipelineExecutor transforms. `test_pipeline_executor_operator_routing` now
  pins all three nodes as fail-closed through central runtime capability
  reasons, giving future executable vocabulary work a guarded baseline.

Status 2026-06-24:

- Complete for the current graph runtime scope. `TokenVocabulary`,
  `POSVocabulary`, and `NERTagVocabulary` execute as narrow
  `PipelineExecutor` transforms over prepared sentence-level string columns.
- The vocabulary nodes reuse the deterministic `SequenceVocabulary` helper and
  emit `value,id` Arrow tables for downstream runtime use.
- Runtime capability metadata and node-browser descriptions now mark the
  vocabulary nodes as implemented runtime nodes instead of stale templates.
- Focused routing coverage proves deterministic token/POS/tag vocabulary
  output and specific schema failure for missing source columns.

### 2. Full NER Example Assets

The saved NER graph now uses current first-class node IDs and avoids Dense
placeholders. The surrounding example assets needed to be made complete and
portable.

Implement:

- repo-safe demo data or a documented data preparation flow,
- checked-in README/inference helper review,
- no absolute local paths in the final public example graph,
- import/open smoke test for the saved graph schema,
- one minimal trainable NER demo with small data.

Status 2026-06-24:

- Complete for the checked-in example asset scope. The NER example includes a
  tiny `sample_ner.csv`, repo-relative graph/vocabulary/data references,
  README updates, generated metadata that resolves relative to the example
  directory, and ignored local generated outputs.
- `test_pattern_template_guard` protects the shipped graph from regressing to
  Dense placeholders, stale serialized node IDs, or machine-local absolute
  paths.
- `validate_ner_graph_assets.py` provides the portable smoke check for saved
  graph references, generated asset existence, and tokenized inference payload
  shape.

### 3. End-To-End Studio Graph Training

Sequence training works when Studio can provide a prebuilt `ISequenceBatcher`.
The remaining work was proving the saved graph launch path with real NER rows.

Implement:

- graph launch from saved NER graph into tabular sequence materialization,
- train/validation split behavior at sentence level,
- label/vocab consistency from data load through prediction decode,
- compact end-to-end test that runs a tiny NER graph through the public Studio
  launch path.

Status 2026-06-24:

- Complete for the current public Studio launch helper path. Saved-NER launch
  coverage in `test_text_gui_training_launch` exercises the checked-in NER
  sequence fixture table through graph launch, materializer/batcher config
  passthrough, sentence-level train/validation/test split behavior, tag
  vocabulary sizing, decode/metrics assertions, and callback lifecycle checks.
- Studio graph launch now builds a `SequenceBatcher` directly from registered
  Arrow sentence tables, routes sequence/NER configs through
  `TrainingManager::StartTrainingSequence`, and normalizes model dimensions
  from runtime vocabularies and sequence length.
- The GUI launch regression intentionally avoids running a full model forward;
  model runtime/deployment decode coverage lives in the sequence inference and
  `.cyxmodel` packaging checks.

### 4. Sequence Inference And Packaging

Training metrics existed, but deployment needed sequence-aware assets.

Implement:

- package token/POS/tag vocabularies with trained models,
- expose decode from tag IDs back to BIO labels,
- save/load sequence metadata such as `ignore_index`, `max_sequence_length`,
  padding IDs, and vocab file paths,
- inference helper tests for tokenized sequence inputs.

Status 2026-06-23:

- Complete for the current `.cyxmodel` and embedded `/v1/predict` deployment
  path. Sequence packages carry token/POS/tag vocabularies and sequence
  metadata, local model loading probes and extracts those assets, model info
  advertises sequence decode capability, prediction accepts named sequence
  tensors with `sequence_lengths`, and decoded BIO labels are returned through
  a length-aware response contract.

### 5. UI Polish For Sequence Graphs

Sequence metrics were visible in the training plot panel. The graph and node UI
needed task-specific polish.

Implement:

- clearer property editors for `NERSequenceBuilder` and vocabulary nodes,
- warnings for missing token/tag columns before launch,
- task-aware labels for token classifier and token loss nodes,
- compact status messaging when sequence graph materialization is blocked.

Status 2026-06-24:

- Complete for the current sequence graph launch path. Blocked training
  launches now carry compact `status_title` and `status_detail` fields, and
  `MainWindow` surfaces those details through the existing blocked-training
  popup instead of leaving sequence materialization failures silent.
- The sequence preflight path reports missing token/tag/POS/sentence columns
  before dispatch, including the affected dataset name.
- `test_text_gui_training_launch` keeps the saved-NER launch, split, batcher,
  vocabulary, decode, and callback regression coverage without running a full
  sequence model forward inside the GUI launch helper test.
- Verified:
  `cmake --build build --config Debug --target test_graph_training_sequence_preflight`,
  `build\bin\Debug\test_graph_training_sequence_preflight.exe`,
  `cmake --build build --config Debug --target test_text_gui_training_launch`,
  and `build\bin\Debug\test_text_gui_training_launch.exe`.

## Verification Targets

Keep these checks green:

- `test_pattern_template_guard`
- `test_graph_compiler_deferred_nodes`
- `test_text_gui_training_launch`
- `test_training_executor_arrow_parquet`
- `cyxwiz-engine` Debug build
