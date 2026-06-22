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

The example remains a target design until the full Studio graph launch path can
materialize sequence batches from saved graph data.

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
