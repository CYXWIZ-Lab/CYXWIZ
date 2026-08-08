# track38 - Properties Panel Truth Work Log

## Goal

Implement `tofix38` in lean slices: add a shared effective-property truth layer,
render a concise Properties summary, and keep complex editors in dialogs.

## Slice 1 - Done

- Add a small typed resolver for concrete truth failures from `tofix38`.
- Cover DataInput text label aliases, TF-IDF effective feature width, and
  Output/CrossEntropy versus Dense class count.
- Render the resolved truth before raw/detail sections in the Properties panel.
- Add focused non-GUI tests for the resolver.

Implemented:

- `cyxwiz-engine/src/gui/properties_truth.{h,cpp}` resolves effective
  properties and raw parameter mappings.
- `Properties` now renders a compact `Truth Summary` after `General`, including
  dialog-backed nodes.
- Quick edits write canonical keys; DataInput label edits also write the
  compatibility label alias.
- `test_properties_truth` covers DataInput alias fallback/conflict, TF-IDF
  default width, Output/Dense/CrossEntropy mismatch, and alias writes.

Verified:

- `cmake --build build --target test_properties_truth --config Debug`
- `build\bin\Debug\test_properties_truth.exe`
- `cmake --build build --target cyxwiz-engine --config Debug`

## Slice 2 - Done

- Added `DatasetTruthFact` to the resolver context so the GUI can pass loaded
  dataset facts without making the resolver depend on global registry state.
- DataInput truth now surfaces:
  - loaded dataset backing store,
  - label column missing from Arrow/Parquet columns,
  - label-column conflicts with text/audio registry metadata,
  - labeled datasets that registered `0` classes.
- `Properties` now gathers facts from Text/Image/Audio registries, Arrow,
  Parquet, and generic Dataset handles.
- Extended `test_properties_truth` with zero-class text dataset and missing
  tabular label-column cases.

Verified:

- `cmake --build build --target test_properties_truth --config Debug`
- `build\bin\Debug\test_properties_truth.exe`
- `cmake --build build --target cyxwiz-engine --config Debug`

## Slice 3 - Done

- Added TextTokenizer compiled-shape truth for `max_length`.
- The truth row labels `max_length` as the compiled token width and marks it
  as dialog-backed for full tokenizer configuration.
- Extended `test_properties_truth` with a TextTokenizer max-length case.

Verified:

- `cmake --build build --target test_properties_truth --config Debug`
- `build\bin\Debug\test_properties_truth.exe`
- `cmake --build build --target cyxwiz-engine --config Debug`

## Slice 4 - Done

- Add compile-report-backed recurrent placement truth for LSTM/GRU.
- Keep placement ingestion passive: `Properties` receives the latest compile
  report facts instead of compiling on every render.
- `MainWindow::BuildCompileResult` now pushes backend placement facts into the
  Properties panel.
- LSTM/GRU truth shows expected backend, status badge, reason code,
  explanation, and suggested action when a compile report exists.
- Without a compile report, LSTM/GRU truth shows a compile-report prompt
  instead of guessing.
- Extended `test_properties_truth` with a GRU compile-placement fact case.

Verified:

- `cmake --build build --target test_properties_truth --config Debug`
- `build\bin\Debug\test_properties_truth.exe`
- `cmake --build build --target cyxwiz-engine --config Debug`

## Slice 5 - Done

- Added explicit cleanup affordances to the raw parameter inspector for keys
  the resolver can prove are redundant.
- Kept cleanup conservative: this slice only marks DataInput's legacy
  `dataset` key removable when `dataset_name` is already set.
- Tightened raw alias mapping so label aliases are only mapped for label truth
  rows.
- Extended `test_properties_truth` with cleanup-safe and legacy-only dataset
  parameter cases.

Verified:

- `cmake --build build --target test_properties_truth --config Debug`
- `build\bin\Debug\test_properties_truth.exe`
- `cmake --build build --target cyxwiz-engine --config Debug`

## Slice 6 - Done

- Added recurrent configuration truth for LSTM/GRU `hidden_size` and
  `return_sequences`.
- Hidden-size truth now warns when a recurrent node name ends with a stale
  numeric width, for example `GRU 32` with `hidden_size=8`.
- Kept name handling read-only: the truth row reports the conflict but does
  not rename user nodes automatically.
- Extended `test_properties_truth` with recurrent hidden-size/name mismatch
  and return-sequence cases.

Verified:

- `cmake --build build --target test_properties_truth --config Debug`
- `build\bin\Debug\test_properties_truth.exe`
- `cmake --build build --target cyxwiz-engine --config Debug`

## Slice 7 - Done

- Added Output-node class truth against loaded dataset class counts.
- `Properties` now passes graph-wide DataInput dataset facts into the truth
  resolver, so Output can compare `num_classes` with the loaded dataset class
  count without the resolver reading global registry state.
- Output class truth reports a conflict when one loaded dataset class count
  disagrees with `num_classes`, and when multiple loaded datasets expose
  different class counts.
- Extended `test_properties_truth` with Output/dataset mismatch and ambiguous
  dataset class-count cases.

Verified:

- `cmake --build build --target test_properties_truth --config Debug`
- `build\bin\Debug\test_properties_truth.exe`
- `cmake --build build --target cyxwiz-engine --config Debug`

## Slice 8 - Done

- Normalized Output class-count truth across `num_classes` and legacy
  `classes`.
- Output truth now treats `classes` as an alias for canonical `num_classes`,
  reports alias conflicts, and quick edits write both keys for compatibility.
- The custom Output properties editor now writes canonical `num_classes` plus
  the compatibility `classes` alias.
- GraphCompiler CrossEntropy class-count extraction now accepts `num_classes`
  when `classes` is absent, preserving old graphs while making the canonical
  key executable.
- Extended `test_properties_truth` with Output alias fallback/conflict and
  alias-write cases.
- Extended `test_graph_compiler_deferred_nodes` with an Output
  `num_classes` fallback compile case.

Verified:

- `cmake --build build --target test_properties_truth --config Debug`
- `build\bin\Debug\test_properties_truth.exe`
- `cmake --build build --target test_graph_compiler_deferred_nodes --config Debug`
- `build\bin\Debug\test_graph_compiler_deferred_nodes.exe`
- `cmake --build build --target cyxwiz-engine --config Debug`

## Slice 9 - Done

- Added DataLoader truth for `pin_memory=true`.
- The truth row marks pinned host memory as unsupported and explains that the
  current batchers ignore it until a pinned host-memory transfer backend
  exists.
- Added DataLoader class-balancing truth for `balance_classes`.
- The class-balancing row explains the train-only sampler behavior: training
  batchers may balance classes, while validation and test batchers keep their
  natural evaluation distribution.
- Kept DataLoader truth rows dialog-backed instead of adding another inline
  DataLoader form to Properties.
- Extended `test_properties_truth` with unsupported `pin_memory=true` and
  train-only class-balancing cases.

Verified:

- `cmake --build build --target test_properties_truth --config Debug`
- `build\bin\Debug\test_properties_truth.exe`
- `cmake --build build --target cyxwiz-engine --config Debug`

## Slice 10 - Done

- Aligned Properties dataset truth with GraphCompiler's saved-graph
  compatibility rule: `dataset_name` wins, and legacy `dataset` is used only
  when `dataset_name` is empty.
- Graph-wide dataset facts now work for legacy-only DataInput nodes, so old
  saved graphs can still surface dataset class and label truth.
- Raw parameter mapping now reports legacy-only `dataset` as an alias of
  canonical `dataset_name`; duplicate `dataset` remains cleanup-safe only when
  `dataset_name` is already set.
- Extended `test_properties_truth` with legacy-only dataset alias mapping and
  dataset fact resolution.

Verified:

- `cmake --build build --target test_properties_truth --config Debug`
- `build\bin\Debug\test_properties_truth.exe`
- `cmake --build build --target cyxwiz-engine --config Debug`

## Slice 11 - Done

- Added positive-width validation to TF-IDF `max_features` truth.
- The Properties truth row now matches the TFIDFVectorizer operator contract:
  configured `max_features` must be at least `1`.
- Invalid `max_features` values are marked as a truth issue instead of showing
  as `OK`.
- Extended `test_properties_truth` with an invalid `max_features=0` case.

Verified:

- `cmake --build build --target test_properties_truth --config Debug`
- `build\bin\Debug\test_properties_truth.exe`
- `cmake --build build --target cyxwiz-engine --config Debug`

## Remaining follow-ups

- Broaden the cleanup-safe stale-key catalog as more node parameter truth
  schemas are added.

## Guardrails

- Do not rewrite `NodeMetadataRegistry` into a broad schema on the first pass.
- Do not duplicate large editor forms inside Properties.
- Keep stale/raw parameter visibility separate from effective truth.
- Preserve old graph compatibility by resolving aliases lazily before mutation.
