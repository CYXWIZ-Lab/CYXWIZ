# To Fix 21 - Future NER And Sequence Graph Work

**Created:** 2026-06-10
**Source:** Follow-up after completing `done19.md`.

## Boundary

`done19.md` completed the practical sequence tagging training slice:
`TimeDistributedDense`, prebuilt `ISequenceBatcher` training, token-shaped
loss, token accuracy, BIO F1, sequence metrics in Studio, and guards for the
shipped NER graph.

This file tracks the remaining NER/sequence work that should happen after that
slice. These items are future work, not regressions in `done19`.

## Remaining Work

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

The example remains a target design until the full Studio graph launch path can
materialize sequence batches from saved graph data.

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
- This does not complete full NER graph launch yet: Studio still needs the
  higher-level model launch/inference packaging that consumes these prepared
  sequence artifacts.
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

### 1. Executable Vocabulary Nodes

`TokenVocabulary`, `POSVocabulary`, and `NERTagVocabulary` are currently
first-class contract/documentation nodes, but they are not executable graph
runtime nodes.

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

### 2. Full NER Example Assets

The saved NER graph now uses current first-class node IDs and avoids Dense
placeholders. The surrounding example assets still need to be made complete and
portable.

Implement:

- repo-safe demo data or a documented data preparation flow,
- checked-in README/inference helper review,
- no absolute local paths in the final public example graph,
- import/open smoke test for the saved graph schema,
- one minimal trainable NER demo with small data.

### 3. End-To-End Studio Graph Training

Sequence training works when Studio can provide a prebuilt `ISequenceBatcher`.
The remaining work is proving the complete saved graph launch path with real
NER rows.

Implement:

- graph launch from saved NER graph into tabular sequence materialization,
- train/validation split behavior at sentence level,
- label/vocab consistency from data load through prediction decode,
- compact end-to-end test that runs a tiny NER graph through the public Studio
  launch path.

### 4. Sequence Inference And Packaging

Training metrics exist, but deployment still needs sequence-aware assets.

Implement:

- package token/POS/tag vocabularies with trained models,
- expose decode from tag IDs back to BIO labels,
- save/load sequence metadata such as `ignore_index`, `max_sequence_length`,
  padding IDs, and vocab file paths,
- inference helper tests for tokenized sequence inputs.

### 5. UI Polish For Sequence Graphs

Sequence metrics are visible in the training plot panel. The graph and node UI
still needs task-specific polish.

Implement:

- clearer property editors for `NERSequenceBuilder` and vocabulary nodes,
- warnings for missing token/tag columns before launch,
- task-aware labels for token classifier and token loss nodes,
- compact status messaging when sequence graph materialization is blocked.

## Verification Targets

Future work should keep these checks green:

- `test_pattern_template_guard`
- `test_graph_compiler_deferred_nodes`
- `test_text_gui_training_launch`
- `test_training_executor_arrow_parquet`
- `cyxwiz-engine` Debug build

Add a tiny NER graph end-to-end test once the full saved graph launch path is
portable.
