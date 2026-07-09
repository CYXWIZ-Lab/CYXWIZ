# Track 48 - Graph Node Property Truth

## Goal

Make the Properties panel truthful for every editable graph node without adding
a broad replacement framework. The first pass should inventory the current
contract surface and extend truth coverage one node family at a time.

## Lean guardrails

- Compose around `NodeMetadataRegistry` and `properties_truth`; do not replace
  them with a second full schema unless the inventory proves it is necessary.
- Keep editable fields backed by real compiler, runtime, loader, materializer,
  or exporter support.
- Prefer focused dialogs for large contracts and truth summaries for effective
  runtime/compiler state.
- Add tests that lock representative families before expanding coverage.
- List unsupported or planned nodes explicitly instead of making placeholder
  properties editable.

## Current repo anchors

- Metadata parameters and support axes: `cyxwiz-engine/src/core/node_metadata.h`.
- Metadata-driven parameter rendering:
  `cyxwiz-engine/src/gui/properties_metadata_editor.cpp`.
- Side-panel contract path classification:
  `cyxwiz-engine/src/gui/properties_contract.*`.
- Effective property truth and raw parameter mapping:
  `cyxwiz-engine/src/gui/properties_truth.*`.
- Properties panel truth rendering:
  `cyxwiz-engine/src/gui/properties.cpp`.

## Specialized truth coverage baseline

These node types currently have specialized effective-property truth beyond the
generic metadata renderer:

- `DataInput`
- `DataOutput`
- `DataConvert`
- `DeployToNodeEditorNode`
- `DataLoader`
- `DataProfiler`
- `StandardScaler`
- `MinMaxScaler`
- `RobustScaler`
- `LabelEncoder`
- `OrdinalEncoder`
- `TargetEncoder`
- `OutlierDetector`
- `TFIDFVectorizer`
- `CountVectorizer`
- `TextTokenizer`
- `RegressionMetricsNode`
- `ClassificationMetricsNode`
- `ConfusionMatrixNode`
- `ROCCurveNode`
- `PRCurveNode`
- `Dense`
- `TimeDistributed`
- `Dropout`
- `BatchNorm`
- `LayerNorm`
- `ReLU`
- `Sigmoid`
- `Softmax`
- `GELU`
- `Tanh`
- `LeakyReLU`
- `Flatten`
- `Reshape`
- `View`
- `Permute`
- `Squeeze`
- `Unsqueeze`
- `LSTM`
- `GRU`
- `NERSequenceBuilder`
- `TokenVocabulary`
- `POSVocabulary`
- `NERTagVocabulary`
- `SequenceTagOutput`
- `MSELoss`
- `FocalLoss`
- `BCELoss`
- `BCEWithLogits`
- `L1Loss`
- `SmoothL1Loss`
- `HuberLoss`
- `NLLLoss`
- `SoftDiceLoss`
- `TverskyLoss`
- `JaccardLoss`
- `Adam`
- `SGD`
- `AdamW`
- `RMSprop`
- `Adagrad`
- `NAdam`
- `Output`
- `CrossEntropyLoss`
- `ExportCSV`
- `ExportParquet`
- `ExportJSON`
- `TreeModelPredictor`

## First slice

- [x] Record the tofix48 working tracker.
- [x] Make the specialized truth coverage list machine-readable.
- [x] Add focused tests that lock the baseline coverage list.
- [x] Add a metadata-level inventory guard that visits every registered node,
      validates editable parameter schema shape, and ensures specialized truth
      coverage maps to implemented metadata.
- [x] Add missing `DataLoader` metadata using the same parameter names written
      by `DataLoaderDialog`.
- [x] Extract side-panel contract routing into `properties_contract.*` so UI
      rendering and tests share the same dialog-only/custom/metadata/fallback
      classification.
- [x] Extend metadata drift tests to prove every registered metadata node is
      visited and implemented nodes classify into a known side-panel path.
- [x] Surface `NodeMetadata` implementation status and support axes in the
      Properties panel General section so runtime/training/support truth is
      visible when selecting a node.
- [x] Add a metadata-to-runtime parameter drift guard for metadata-rendered
      nodes using the existing pipeline runtime capability tables.
- [x] Align runtime-backed metadata parameters for `SortRows`, `JoinTables`,
      `GroupByAggregate`, `SentimentAnalyzer`, clustering nodes, and ACF/PACF
      compatibility lag fields.
- [x] Extend the runtime drift guard to validate editable enum/bool renderer
      shape, runtime allowed values, defaults, and integer/float renderer
      types for metadata-rendered nodes.
- [x] Fix deeper drift found by the guard: `OrdinalEncoder.categories` now
      renders as the runtime-backed `auto` enum, and `RandomForestClassifier`
      `seed` is marked as a single integer runtime parameter.
- [x] Build the deeper editable-node inventory from dialog-only paths, custom
      editors, compiler/runtime consumers, and truth coverage.

## Editable contract inventory snapshot

The focused metadata test now validates every runtime-registered metadata node
is reachable through one known Properties-panel contract path:

- `dialog_only`: large or specialized contracts that should stay in focused
  dialogs: `DataInput`, `DataOutput`, `DataConvert`, `TextTokenizer`,
  `TextVocabulary`, `TextPadding`, `Embedding`.
- `custom_sequence_editor`: sequence/NLP builders with custom UI contracts:
  `NERSequenceBuilder`, `TokenVocabulary`, `POSVocabulary`,
  `NERTagVocabulary`, `SequenceTagOutput`.
- `metadata_renderer`: implemented parameterized nodes rendered directly from
  `NodeMetadata`. Runtime-backed nodes are audited against
  `pipeline_runtime_capabilities` for required names, enum/bool allowed
  values, defaults, and integer/float renderer types.
- `custom_fallback_editor`: implemented parameterless nodes that expose
  read-only status/support truth instead of fake editable fields.
- `specialized_truth`: nodes with effective property truth beyond generic
  metadata: `DataInput`, `DataLoader`, `TFIDFVectorizer`, `TextTokenizer`,
  `LSTM`, `GRU`, `Output`, `CrossEntropyLoss`, preprocessing materializers
  (`StandardScaler`, `MinMaxScaler`, `RobustScaler`, `LabelEncoder`,
  `OrdinalEncoder`, `TargetEncoder`, `OutlierDetector`), and
  `CountVectorizer`, metrics nodes, supported losses, and supported
  optimizers, plus core model layers, activations, normalization, shape ops,
  sequence-tagging builder/vocabulary/output nodes, and implemented table
  exporters/conversion/deploy/profiling/inference nodes.

Current planned/blocked/template inventory stays explicit in metadata rather
than editable in the panel. The largest remaining groups are:

- data source/import gaps: `ExcelFile`, `JSONFile`, `SQLQuery`, `HDF5Dataset`,
  `RESTAPISource`, planned SQL/Excel export;
- analytics/classical ML gaps: `TSNENode`, `SVMClassifier`, `KNNClassifier`,
  `NaiveBayesClassifier`, `LogisticRegressionNode`, decomposition/test/solver
  nodes;
- model/layer gaps: `Conv2D`, pooling, upsampling, attention variants,
  `GroupNorm`, `InstanceNorm`, `RNN`, `Bidirectional`;
- training gaps: schedulers, regularization nodes, pair/triplet builders,
  siamese/shared encoder nodes, contrastive/triplet losses, pair/retrieval
  metrics and embedding outputs;
- signal/text/visualization gaps: `IFFTNode`, `WaveletTransform`,
  `WordFrequencyNode`, `TokenizerNode`, `WordEmbeddings`,
  `NamedEntityRecognizer`, `GradCAMNode`, `SaliencyMapNode`,
  `GradientDescentViz`.

## Next implementation slices

1. [x] Preprocessing/materializer nodes: `StandardScaler`, `MinMaxScaler`,
   `RobustScaler`, `LabelEncoder`, `OrdinalEncoder`, `TargetEncoder`,
   `OutlierDetector`, `TFIDFVectorizer`, `CountVectorizer`.
2. [x] Loss, optimizer, and metric nodes: align metadata params with compiler and
   training runtime consumption.
3. [x] Core model layers: `Dense`, activations, normalization, pooling, and shape
   nodes.
4. [x] Recurrent and sequence nodes: extend the existing `LSTM`/`GRU` truth to
   sequence-tagging nodes and recurrent wrappers.
5. [x] Export/import/inference/debugger nodes: expose read-only support truth
   and reason codes before adding editable fields.

## Open follow-up

### Completed preprocessing/materializer slice

- Added effective materializer truth rows for scaler defaults, encoder required
  columns, target-encoder smoothing, outlier detector method/action support,
  vectorizer dense output width, vectorizer required text columns, and n-gram
  runtime limits.
- Kept editable fields backed by existing `NodeMetadata`; the truth resolver
  only explains effective defaults, unsupported values, and validation issues.
- Locked the slice in `test_properties_truth` and the metadata inventory guard.

### Completed loss/optimizer/metric slice

- Added effective truth for metric required columns, supported loss reductions
  and loss-specific runtime constraints, optimizer learning-rate aliases, and
  unsupported optimizer knobs still present on older graph nodes.
- Aligned optimizer metadata to canonical `learning_rate` and stopped exposing
  editable `momentum`/`weight_decay` fields until optimizer construction
  applies them.
- Locked the slice in `test_properties_truth` and the metadata inventory guard.

### Completed core model layer slice

- Added effective truth for `Dense`/`TimeDistributed` widths, unsupported
  inline Dense activation parameters, `Dropout` rate bounds, BatchNorm/LayerNorm
  epsilon aliases, LeakyReLU slope, activation shape preservation, and shape-op
  compiler parameters.
- Aligned `BatchNorm` metadata to expose compiler-consumed `eps` instead of
  relying only on legacy `epsilon` defaults from older graph nodes.
- Kept blocked pooling/conv/upsampling/template nodes listed as planned
  metadata rather than making their properties appear executable.
- Locked the slice in `test_properties_truth` and the metadata inventory guard.

### Completed recurrent/sequence slice

- Kept existing `LSTM`/`GRU` backend-placement and hidden-size truth, and added
  specialized truth coverage for `NERSequenceBuilder`, `TokenVocabulary`,
  `POSVocabulary`, `NERTagVocabulary`, and `SequenceTagOutput`.
- Surfaced compiler/materializer aliases for sequence columns, vocabulary
  thresholds, vocabulary caps, max sequence length, attention masks, ignore
  indices, tag vocabulary files, and BIO decode metadata.
- Marked drift explicitly where the runtime does not honor broader UI input:
  custom `outside_tag` values and non-`BIO` sequence tag schemes are reported
  as unsupported instead of silently appearing executable.
- Locked the slice in `test_properties_truth` and the metadata inventory guard.

### Completed export-path slice

- Added specialized truth coverage for implemented table exporters:
  `DataOutput`, `ExportCSV`, `ExportParquet`, and `ExportJSON`.
- Surfaced required output file paths, legacy `path` aliases, `DataOutput`
  `format`/`file_type` aliases, fixed exporter formats, and runtime result
  behavior (`ctx.output_dataset` becomes the output path while the input
  dataset remains available downstream).
- Kept blocked exporters (`ExportSQL`, `ExportExcel`) in template metadata
  rather than adding specialized truth for non-executable paths.
- Locked the slice in `test_properties_truth` and the metadata inventory guard.

### Completed import/inference/debug-support slice

- Added specialized truth coverage for `DataConvert`,
  `DeployToNodeEditorNode`, `DataProfiler`, and `TreeModelPredictor`.
- Surfaced `DataConvert` input source rules, required output paths, format
  defaults, and the registered `ds_dataconvert_<node id>` runtime result.
- Surfaced deployment handoff truth for the Node Editor path, including the
  default `deployed_<node id>` dataset name and the context fields set by the
  executor.
- Removed editable `DataProfiler.minimal` metadata because the executor
  currently emits the same per-column report schema regardless of that flag;
  stale saved graphs carrying `minimal=true` are still surfaced as unsupported
  truth.
- Surfaced `TreeModelPredictor` required artifact path, artifact-driven feature
  order default, prediction column default, and operator-factory inference
  route.
- Kept unresolved/template debugger and source gaps listed in metadata/follow-up
  inventory rather than creating fake editable contracts.
- Locked the slice in `test_properties_truth` and the metadata inventory guard.

Create a full inventory report that classifies every editable metadata node as
one of:

- specialized truth covered;
- metadata-rendered;
- dialog-only;
- custom editor;
- template/planned/unsupported;
- missing contract audit.
