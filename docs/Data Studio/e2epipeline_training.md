# E2E Training Pipeline Code Dive

Created: 2026-06-27

This document describes the current end-to-end training flow in CyxWiz, from DataInput loading through registry storage, optional Arrow pipeline materialization, batching, model/loss/optimizer construction, forward/backward computation, and metric reporting.

It is written for developers who need to understand what is already coded, what is pass-through, and where missing pieces should be implemented next.

## High-level map

```text
Studio canvas
    |
    | user configures DataInput and clicks Apply
    v
DataInputDialog::Apply
    |
    | category dispatch through gui/loaders
    v
DataLoader::LaunchAsyncLoad
    |
    | async worker scans/loads/probes data
    v
DataRegistry
    |
    | graph training button
    v
GraphCompiler::Compile
    |
    | produces TrainingConfiguration
    v
StartGraphTrainingFromCompiledConfig
    |
    | optional Arrow-only materialization
    v
PipelineMaterializer
    |
    | dispatch by registered storage kind
    v
DataLoader::LaunchTraining / TrainingManager::StartTraining*
    |
    | builds batcher + TrainingExecutor
    v
TrainingExecutor::Initialize
    |
    | ModelBuilder builds model/loss/optimizer
    v
TrainingExecutor::Train
    |
    | epochs -> batches -> forward -> loss -> backward -> optimizer step
    v
TrainingMetrics / TrainingPlotPanel / preserved trained model
```

## Core runtime contracts

### `gui::MLNode` and links

Canvas nodes are stored as `MLNode` objects with:

- `type`: strongly typed `gui::NodeType`.
- `parameters`: string key/value map used by dialogs, graph compiler, materializer operators, and JSON save/load.
- `description`: UI-facing state summary.
- links connect node output pins to downstream input pins.

The important constraint is that node parameters are stringly typed. Every compiler extractor, loader, materializer operator, and training builder must parse and validate its own keys.

### `TrainingConfiguration`

`GraphCompiler::Compile` converts graph nodes and links into `TrainingConfiguration`.

Current responsibilities include:

- dataset identity: `dataset_name`, label column, domain.
- split and data loader knobs: `train_ratio`, `val_ratio`, `test_ratio`, `batch_size`, `epochs`, `shuffle`, `drop_last`, `num_workers`, `prefetch_factor`, `dataloader_seed`, gradient accumulation.
- model structure: ordered layer configs, graph execution plan when graph-op nodes are present.
- preprocessing configs: tabular/image/audio/text/sequence/time-series.
- loss and optimizer config.
- checkpointing and early stopping.

Important limitation: `DataSplit.stratified` can be parsed at the graph level, but the current batchers do not implement stratified splitting or class-balanced resampling.

### `DataRegistry`

`DataRegistry` is the shared registry between DataInput Apply and training launch.

It stores multiple dataset contracts:

- Legacy `DatasetHandle` map for old in-memory dataset implementations.
- `ArrowDataset` map for in-memory tabular/text Arrow tables.
- `ParquetBackedDataset` map for disk-backed tabular CSVs.
- `ImageDatasetEntry` metadata for image folders.
- `AudioDatasetEntry` metadata for audio folders.
- `TextDatasetEntry` metadata for text sources.

Image/audio/text entries are usually lightweight metadata. Pixel decode, audio feature extraction, and text tokenization are deferred to training-time batchers unless an Arrow text materializer path is selected.

### `Batch`

Normal training uses `Batch`.

```text
Batch
  data   : Tensor [batch, ...input_shape]
  labels : Tensor [batch] or [batch, classes] or [batch, 1]
  size   : actual rows in this batch
```

All non-sequence batchers implement `IBatcher`.

### `SequenceBatch`

Sequence/NER training uses a different contract.

```text
SequenceBatch
  word_ids       : Tensor [batch, seq]
  pos_ids        : Tensor [batch, seq], optional
  attention_mask : Tensor [batch, seq]
  tag_ids        : Tensor [batch, seq], supervised tagging target
  target_ids     : Tensor [batch, seq], causal LM target
  size
  sequence_length
```

Sequence batchers implement `ISequenceBatcher`, not `IBatcher`.

### `IPipelineOperator`

`IPipelineOperator` is the Arrow table materializer contract.

```text
input  : shared_ptr<arrow::Table>
output : shared_ptr<arrow::Table>
```

Operators are configured from `MLNode::parameters` and must transform Arrow tables without UI interaction.

Only `PipelineRuntimeSupportMode::OperatorBacked` nodes can materialize into Arrow before training. Loss nodes, optimizers, neural layers, DataLoader nodes, and UI/executor-only nodes are not materializer operators.

## DataInput Apply flow

```text
DataInputDialog::Apply
    |
    | persist common params:
    | source_type, file_category, file_path, folder_path,
    | dataset_name, force_disk_backed, etc.
    |
    | prune stale category-specific params
    v
category-specific branch
    |
    | build loaders::ApplyContext
    v
loader->ValidateApplyContext(ctx)
    |
    | create AsyncLoadState
    v
loader->LaunchAsyncLoad(ctx, state)
    |
    | AsyncTaskManager worker
    v
DataRegistry registration
    |
    | UI thread polls completion
    v
DataInputDialog::PollAsyncLoadResult
    |
    | updates node params:
    | data_loaded=true/false, loaded_rows, loaded_cols,
    | memory_bytes, dataset_name, audit state
    v
canvas node ready for compile
```

### Tabular and time-series loading

File category: `Tabular` or `TimeSeries`.

Loader: `TabularLoader`.

Path:

```text
DataInputDialog::Apply
    -> TabularLoader::LaunchAsyncLoad
        -> DataRegistry::LoadTabularCSV for CSV/TSV
            -> ArrowDataset in memory, or ParquetBackedDataset on disk
        -> DataRegistry::LoadParquetToArrow / LoadArrowTable for Arrow-native files
        -> DatasetAudit::AuditTabular or AuditParquet
        -> AsyncLoadState backend:
             1 = Arrow in-memory
             2 = Parquet disk-backed
```

Notes:

- CSV/TSV can auto-pick in-memory Arrow or disk-backed Parquet.
- `force_disk_backed` forces the Parquet path.
- Time-series uses the same load path as tabular, but graph compilation and batching treat it as `PreprocessingDomain::TimeSeries`.
- Time-series materializer operators can create a `__partition__` column used by Arrow/Parquet batchers to choose train/val/test rows.

### Text loading

File category: `Text`.

Loader: `TextLoader`.

Current text load path now registers both:

- raw Arrow backing when possible, under the dataset name.
- `TextDatasetEntry` metadata under the same name.

Path:

```text
DataInputDialog::Apply
    -> TextLoader::LaunchAsyncLoad
        -> TextDataset probe for CSV/JSON/TXT/corpus stats and vocab
        -> Arrow table registration:
             CSV/TSV: Arrow CSV reader preserves original columns
             Arrow-native: ArrowDataset::FromFile
             JSON/TXT/corpus: BuildRawTextArrowTable adapter
        -> DataRegistry::RegisterArrowTable(name)
        -> DataRegistry::RegisterTextDataset(name, TextDatasetEntry)
        -> DatasetAudit::AuditText
        -> AsyncLoadState backend:
             5 = text metadata/lazy text
```

Important routing detail:

- If a text graph uses a Cat-1 Arrow text operator like `TextTokenizer`, materialization can produce `name__materialized`, and training routes as Arrow.
- If no materialized Cat-1 text table is selected, `TextLoader::LaunchTraining` intentionally uses legacy `TextDatasetBatcher`, even if raw Arrow backing exists.

### Image loading

File category: `Image`.

Loader: `ImageLoader`.

Path:

```text
DataInputDialog::Apply
    -> ImageLoader::LaunchAsyncLoad
        -> DataRegistry::LoadImageFolder or LoadImageCSV for scan/probe
        -> DataRegistry::RegisterImageDataset(name, ImageDatasetEntry)
        -> DatasetAudit::AuditImage
        -> AsyncLoadState backend:
             3 = image folder metadata
```

Important details:

- Single image files are refused for training. Image training expects a folder.
- Class-subdirectory layout is supported.
- Flat folder plus labels CSV is supported.
- Actual image pixels are not loaded into Arrow for training. Training constructs an `ImageDatasetBatcher` later.
- Memory shown in the UI is an estimate based on target dimensions and channel count.

### Audio loading

File category: `Audio`.

Loader: `AudioLoader`.

Path:

```text
DataInputDialog::Apply
    -> AudioLoader::LaunchAsyncLoad
        -> AudioDataset probe
             scans folder or flat folder + CSV
             probes feature shape
        -> DataRegistry::RegisterAudioDataset(name, AudioDatasetEntry)
        -> DatasetAudit::AuditAudio
        -> AsyncLoadState backend:
             4 = audio folder metadata
```

Important details:

- Class-subdirectory layout is supported.
- Flat folder plus labels CSV is supported.
- Audio features are extracted lazily during training.
- The Apply-time probe discovers feature rows/cols so the graph knows input size.

## Graph compile flow

```text
GraphCompiler::Compile(nodes, links)
    |
    | find DataInput / DatasetInput
    | resolve dataset_name and label_column
    | infer preprocessing domain
    |
    | extract DataSplit
    | extract DataLoader
    | extract preprocessing nodes
    | extract layers / graph execution plan
    | extract loss node
    | extract optimizer node
    |
    v
TrainingConfiguration
```

Current coded behavior:

- `DataLoader` node controls batch size, epochs, shuffle, drop_last, workers, prefetch, seed, log interval, validation frequency, gradient accumulation, checkpoint/early stop knobs.
- `DataSplit` controls ratios and seed. The code records split config, but stratified/class-balanced behavior is not implemented in batchers.
- Loss node types recognized by model building include MSE, CrossEntropy, BCE, BCEWithLogits, L1, SmoothL1/Huber, and NLL.
- CrossEntropy compile/build path handles ignore index and one-hot/class-count behavior.
- `FocalLoss` exists in the backend loss library, but normal graph training does not currently instantiate it from the compiled graph.
- Decision tree/classical tree training is not part of `TrainingExecutor`/`ModelBuilder`.

## Launch and materialization flow

Training launch from the graph enters `StartGraphTrainingFromCompiledConfig`.

```text
StartGraphTrainingFromCompiledConfig(config, nodes, links)
    |
    | resolve dataset_name and label_column
    | validate sequence dataset requirements
    | apply deprecated optimizer epochs/batch_size fallback only if no DataLoader
    |
    v
PipelineMaterializer::Materialize(nodes, links, registry, dataset_name)
    |
    | if source backend is ArrowTable:
    |     materialize supported Cat-1 operators
    |     register effective_dataset_name = dataset_name__materialized
    |
    | if source backend is Parquet/Image/Audio/Text:
    |     skip materializer and pass through dataset_name
    |
    v
DataLoader::LaunchTraining / TrainingManager::StartTraining*
```

### PipelineMaterializer source support

```text
Storage backend       Materializer support
---------------------------------------------------------------
ArrowTable            supported
ParquetBacked         not materialized; pass-through
ImageDataset          not materialized; pass-through
AudioDataset          not materialized; pass-through
TextDataset           not materialized as TextDataset; pass-through
```

Important nuance:

- Text can still materialize if the dataset name is also registered as an `ArrowDataset` and the training launch selects the Arrow materialized result.
- The `TextDataset` metadata backend itself is marked unsupported for `PipelineMaterializer`; legacy text training uses `TextDatasetBatcher`.

### Arrow materializer operator path

`PipelineMaterializer::MaterializeTable`:

```text
source Arrow table
    |
    | find DataInput/DatasetInput node
    | walk reachable graph path
    | reject reachable cycles
    | reject branched materializer operator paths
    |
    | for each materializable operator:
    |     ResolvePipelineRuntimeSupport(node.type)
    |     require mode == OperatorBacked
    |     PipelineOperatorFactory::Create(node.type)
    |     operator->Configure(node.parameters)
    |     table = operator->Apply(table)
    |
    v
materialized Arrow table
```

Current registered `IPipelineOperator` examples include:

- Identity.
- Time-series: window, split, log transform, differencing, features.
- Text: TextTokenizer, TFIDFVectorizer, CountVectorizer, SentimentAnalyzer.
- Preprocessing: scalers, encoders, outlier detector.
- PCA.
- Clustering annotations.
- Signal processing: FFT, 1D convolution, filter designer.
- Classical regression operators: linear regression and polynomial regression as table operators.
- Time-series analysis operators.

Important restrictions:

- Only linear materializer paths are supported. Branched operator paths are rejected.
- `TextVocabulary` and `TextPadding` are folded into `TextTokenizer` parameters. They are not standalone Arrow transforms.
- Non-operator-backed nodes are skipped or fail closed depending on runtime capability metadata.
- Parquet/image/audio/text metadata backends are not transformed by `PipelineMaterializer`.

## Training dispatch by domain/backend

```text
registered dataset name
    |
    +-- ArrowDataset -----------------> TrainingManager::StartTrainingArrow
    |
    +-- ParquetBackedDataset ---------> TrainingManager::StartTrainingParquet
    |
    +-- ImageDatasetEntry ------------> TrainingManager::StartTrainingImage
    |
    +-- AudioDatasetEntry ------------> TrainingManager::StartTrainingAudio
    |
    +-- TextDatasetEntry -------------> TrainingManager::StartTrainingText
    |
    +-- sequence config enabled ------> TrainingManager::StartTrainingSequence
```

`TrainingManager` starts a single async training session:

```text
TrainingManager::StartTraining*
    |
    | build domain batcher or pass Arrow/Parquet dataset
    | adjust config.input_size / input_shape / output_size
    | construct TrainingExecutor
    |
    v
StartTrainingCommon
    |
    | mark training active
    | create async task/thread
    v
TrainingThreadFunc
    |
    | executor.Initialize
    | executor.Train(callbacks)
    | preserve trained model/optimizer
```

## Batching paths

### ArrowDatasetBatcher

Used by in-memory Arrow tabular data and Arrow-materialized text/time-series tables.

```text
ArrowDatasetBatcher(dataset, label_column, batch_size, split config)
    |
    | InitializeColumns:
    |   label column -> label_col_idx
    |   numeric/non-internal columns -> feature_cols
    |   skip "__*" internal columns and partition column
    |
    | build indices:
    |   normal tabular: ratio slicing by row index
    |   time-series: filter __partition__ == 0/1/2
    |
    | Reset:
    |   shuffle indices for train when shuffle=true
    |
    | GetNextBatch:
    |   read selected rows/feature columns
    |   convert numeric values to float32
    |   read labels as int for classification or float for regression
    |   normalize if configured
    |   one-hot if configured
    v
Batch(data=[batch, features], labels)
```

Missing:

- No stratified split.
- No class-balanced sampler.
- No weighted sampling.
- `drop_last` requested from DataLoader is logged as unsupported for Arrow.

### ParquetArrowBatcher

Used by disk-backed tabular data.

```text
ParquetArrowBatcher(ParquetBackedDataset, label_column, batch_size)
    |
    | split row groups by train/val/test ratio
    | optionally filter rows by __partition__ for time-series
    |
    | Reset:
    |   shuffle row group order if train shuffle=true
    |
    | LoadNextRowGroup:
    |   read one row group into Arrow table
    |   build row index list
    |   shuffle rows inside row group
    |
    | GetNextBatch:
    |   fill batch across row groups
    |   convert numeric columns to float32
    |   normalize and one-hot/regression labels as needed
    v
Batch(data=[batch, features], labels)
```

Missing:

- Split is by row group, not row-level stratified/class-balanced split.
- No class-balanced sampler.
- Multi-chunk fallback is incomplete; unsupported/multi-chunk feature cases can leave zeros.
- `drop_last` is logged as unsupported for Parquet.

### PrefetchBatcher

`PrefetchBatcher` wraps an existing `IBatcher` when `prefetch_factor > 0`.

```text
source IBatcher
    |
    | background worker calls source.GetNextBatch
    | bounded queue depth = prefetch_factor
    v
TrainingExecutor consumes prefetched batches
```

Important:

- Prefetch changes I/O scheduling only.
- It does not rebalance, reshuffle beyond source behavior, stratify, or change class distribution.

### ImageDatasetBatcher

Used by image training.

```text
ImageDatasetEntry
    |
    | construct dataset:
    |   layout 0: ImageFolderDataset
    |   layout 1: ImageCSVDataset
    |
    | target dimensions:
    |   from Resize preprocessing config if present
    |   fallback 224x224
    |
    | transform:
    |   resize happens in dataset decode path
    |   ImageTransform handles augmentation/blur/enhancement
    |
    | split:
    |   build all_indices
    |   shuffle all_indices
    |   first train_ratio -> train_indices
    |   remainder -> val_indices
    |
    | GetNextBatch:
    |   load images, optionally in worker threads
    |   apply transform
    |   normalize if configured
    |   one-hot labels if configured
    |   emit [batch,H,W,C] or flattened [batch,H*W*C]
    v
Batch
```

Important:

- The split is randomized, which avoids alphabetic class-subdir leakage, but it is not stratified.
- There is no class-balanced resampling.
- `flatten_` defaults false in the batcher so the graph's `Flatten` node can handle it, but `TrainingManager` may set flatten depending on compiled config.
- Image materialization to Arrow is not used for training.

### AudioDatasetBatcher

Used by audio training.

```text
AudioDatasetEntry
    |
    | build AudioDatasetConfig:
    |   start from dialog/default entry values
    |   override feature node params when graph has Spectrogram/Mel/MFCC
    |   apply AudioAugmentation flags
    |
    | construct AudioDataset
    |   scan class subdirs or flat folder + CSV
    |   lazy decode audio per GetItem
    |   pad/truncate to max_duration
    |   compute Spectrogram/MelSpectrogram/MFCC
    |
    | probe feature shape
    |
    | split:
    |   shuffle all indices
    |   first train_ratio -> train
    |   remainder -> val
    |
    | GetNextBatch:
    |   load/extract features, optionally in worker threads
    |   zero-fill bad/silent samples
    |   per-sample z-score normalization
    |   optional explicit normalization
    |   one-hot labels if configured
    |   emit [batch, rows*cols] by default or [batch, rows, cols]
    v
Batch
```

Important:

- Audio batcher explicitly fixed the old sequential split leakage by shuffling before splitting.
- It still does not implement true stratified split.
- It does not implement class-balanced resampling.
- Feature extraction is lazy and can be expensive during training.

### TextDatasetBatcher

Used by legacy/lazy text training when no Arrow materialized text table is selected.

```text
TextDatasetEntry
    |
    | build TextDatasetConfig from registry entry + compiled overrides
    |
    | construct TextDataset
    |   CSV/TSV/JSON/JSONL/TXT/corpus support
    |   maps string labels to int ids
    |   builds tokenizer vocabulary
    |
    | BuildRawTextArrowTable
    |
    | TextTokenizerOperator
    |   text + label -> tok_0..tok_N,y
    |
    | add __partition__ by shuffled split
    |
    | wrap as ArrowDataset
    |
    | delegate train/val/test to ArrowDatasetBatcher
    v
Batch(data=[batch,max_length], labels)
```

Important:

- Legacy text training internally converts to Arrow and delegates to ArrowDatasetBatcher after tokenization.
- Split is shuffled ratio split, not stratified.
- No class-balanced resampling.
- Text sentiment datasets with imbalanced labels will train on the original skew unless developers add sampler/weights.

### Sequence / NER batcher

Sequence training is enabled by `config.sequence_batch.enabled`.

```text
Arrow/Parquet source table
    |
    | read token column, tag column, optional POS column
    | split cell strings by whitespace
    | optional sentence_id grouping for split units
    |
    | BuildNERSequenceData
    |   build token vocabulary
    |   build POS vocabulary
    |   build tag vocabulary
    |   pad/truncate to sequence length
    |
    | SequenceBatcher
    v
SequenceBatch(word_ids, pos_ids, attention_mask, tag_ids/target_ids)
```

Important:

- If sentence ID is configured, split units are sentence groups, not individual rows.
- Splitting is still ratio-based, not stratified.
- `attention_mask` is built in the sequence batch contract. Developers should verify every sequence layer/loss path that needs masks actually consumes it.

## Model, loss, optimizer, and computation

`TrainingExecutor::Initialize` builds computation objects from `TrainingConfiguration`.

```text
TrainingExecutor::Initialize
    |
    v
BuildExecutableFromConfig(config)
    |
    +-- graph_op_node_ids non-empty -> GraphExecutableModel
    |
    +-- otherwise -> BuildSequentialFromConfig
            |
            | Dense, Embedding, LSTM, GRU, TransformerEncoder,
            | activations, Dropout, Flatten, tensor shape ops,
            | BatchNorm, Output markers
            |
            | CNN layers currently warn/fail as unsupported in SequentialModel
            v
        SequentialModel
    |
    v
BuildLossFromConfig(config.loss_type, config.loss_params)
    |
    v
BuildOptimizerFromConfig(...)
```

Normal training loop:

```text
for epoch in 1..epochs:
    train_batcher.Reset()

    while !train_batcher.IsEpochComplete():
        batch = train_batcher.GetNextBatch()
        if !batch.IsValid(): break

        predictions = model.Forward(batch.data)
        loss = loss_fn.Forward(predictions, batch.labels)
        accuracy = ComputeAccuracy(predictions, batch.labels)

        loss_gradient = loss_fn.Backward(predictions, batch.labels)
        model.Backward(loss_gradient)

        AccumulateGradientsAndMaybeStep(...)
            -> optimizer.Step(model parameters)
            -> zero/clear accumulated grads as needed

        batch callback -> UI metrics

    validation if configured:
        val_batcher.Reset()
        forward-only eval over validation batches

    checkpoint / early stopping logic

after training:
    optional test eval
    complete callback
    TrainingManager preserves trained model and optimizer
```

Sequence training uses the same high-level epoch pattern, but consumes `SequenceBatch` and token-level metrics.

## Domain-specific diagrams

### Tabular classification/regression

```text
CSV/Parquet/Arrow file
    |
    v
TabularLoader
    |
    +-- small/Arrow-native -> ArrowDataset
    |
    +-- large/forced CSV -> ParquetBackedDataset
    |
    v
GraphCompiler -> TrainingConfiguration
    |
    v
PipelineMaterializer
    |
    +-- Arrow only: Cat-1 transforms produce dataset__materialized
    |
    +-- Parquet: pass-through
    |
    v
TrainingManager
    |
    +-- ArrowDatasetBatcher
    |
    +-- ParquetArrowBatcher
    |
    v
TrainingExecutor -> model/loss/optimizer -> metrics
```

### Text sentiment/classification

```text
CSV/TSV/JSON/TXT/corpus
    |
    v
TextLoader
    |
    +-- RegisterArrowTable(raw text table)
    |
    +-- RegisterTextDataset(metadata)
    |
    v
Graph training launch
    |
    +-- if TextTokenizer/TFIDF/CountVectorizer materializes:
    |       raw Arrow text -> tok_*/tfidf_*/count_* + y
    |       -> ArrowDatasetBatcher
    |
    +-- otherwise:
            TextDatasetEntry -> TextDatasetBatcher
            -> TextDataset -> BuildRawTextArrowTable
            -> TextTokenizerOperator
            -> ArrowDatasetBatcher
    |
    v
TrainingExecutor
```

Missing for imbalanced sentiment:

- No balanced sampler.
- No class-weighted loss wiring.
- No stratified split.
- No FocalLoss graph path despite backend class existing.

### Image classification

```text
folder/class/*.jpg or folder + labels.csv
    |
    v
ImageLoader scan/probe
    |
    v
DataRegistry::ImageDatasetEntry
    |
    v
TrainingManager::StartTrainingImage
    |
    v
ImageDatasetBatcher
    |
    | lazy image decode/resize/transform/normalize
    v
Batch [B,H,W,C] or [B,H*W*C]
    |
    v
TrainingExecutor
```

Not coded:

- Arrow image materialization for training.
- Stratified/balanced class sampler.
- Full CNN layer support in the current sequential model path.

### Audio classification

```text
folder/class/*.wav or folder + labels.csv
    |
    v
AudioLoader probe
    |
    v
DataRegistry::AudioDatasetEntry
    |
    v
TrainingManager::StartTrainingAudio
    |
    v
AudioDatasetBatcher
    |
    | lazy decode
    | pad/truncate
    | spectrogram/mel/MFCC
    | z-score normalization
    v
Batch [B,rows*cols] or [B,rows,cols]
    |
    v
TrainingExecutor
```

Not coded:

- Arrow audio materialization for training.
- Stratified/balanced sampler.
- Rich augmentation parameter ranges beyond current flags/default ranges.

### Time-series

```text
CSV/Arrow/Parquet table
    |
    v
TabularLoader with TimeSeries domain
    |
    v
PipelineMaterializer on Arrow
    |
    | TimeSeriesWindow / TimeSeriesSplit / feature operators
    | may emit __partition__ and y
    v
ArrowDatasetBatcher or ParquetArrowBatcher
    |
    | if __partition__ present:
    |   train = rows where __partition__ == 0
    |   val   = rows where __partition__ == 1
    |   test  = rows where __partition__ == 2
    |
    v
TrainingExecutor with regression-style labels
```

Not coded:

- Future-row forecasting schema beyond current materialized table style is limited.
- Class balancing is irrelevant for many regression time-series cases but no weighted regression sampler exists either.

## Data Studio `PipelineExecutor` versus training `PipelineMaterializer`

There are two separate pipeline systems.

### `PipelineMaterializer`

Purpose:

- Runs automatically during graph training launch.
- Only transforms Arrow tables.
- Uses `IPipelineOperator`.
- Produces a materialized dataset for training.

### `PipelineExecutor`

Purpose:

- Executes Data Studio pipelines from JSON/execution plans.
- Has broader node execution support through DuckDB/Arrow and legacy executor modes.
- Handles validation, topological ordering, and executor-style nodes.
- It is not the normal neural training loop.

Developer implication:

- Adding a node to `PipelineExecutor` does not automatically make it train-time materializable.
- To affect the training dataset before `TrainingExecutor`, a node must be registered as an operator-backed `IPipelineOperator`, have runtime capabilities metadata, and work against Arrow tables.

## Missing and incomplete pieces to target

### Class imbalance support

Current state:

- Train/val split is ratio-based.
- Arrow and Parquet split by row order or row groups.
- Image/audio/text shuffle before splitting, but do not stratify.
- No batcher implements class-balanced resampling.
- No batcher implements weighted random sampling.
- No DataLoader/DataSplit node exposes a coded sampler mode.
- Prefetch does not change sampling.

Needed:

- Add a sampler contract, likely independent from `IBatcher`, for class-aware index generation.
- Support at least:
  - sequential/ratio split.
  - shuffled split.
  - stratified split.
  - balanced oversampling.
  - optional undersampling.
  - weighted random sampling.
- Implement for Arrow, Parquet, image, audio, text, and sequence where applicable.
- Decide how sampler behavior interacts with `drop_last`, `prefetch_factor`, `num_workers`, and deterministic seeds.

### Weighted loss

Current state:

- Backend loss classes exist for common losses.
- Graph training currently builds common losses, but class weights are not wired into CrossEntropy/BCE-style training from the graph.
- FocalLoss exists in the backend loss library but is not instantiated by normal graph training.

Needed:

- Add graph node/UI parameters for `class_weight`, `pos_weight`, or automatic inverse-frequency weighting.
- Extend `TrainingConfiguration::loss_params` handling into concrete loss constructors.
- Decide whether weights are computed at load time, split time, or training start.
- Add a clear policy for labels:
  - integer labels.
  - one-hot labels.
  - binary BCE labels.
  - multi-label BCE.
  - ignored labels/ignore_index.

### Decision tree / classical model training

Current state:

- Some classical operators exist as Arrow table pipeline operators or executor-style nodes.
- `TrainingExecutor` and `ModelBuilder` are neural-network oriented.
- Decision tree training is not part of the normal compiled graph training backend.

Needed:

- Separate classical model training contract from neural `SequentialModel`.
- Define model artifact interface for tree models.
- Define prediction/evaluation path.
- Decide whether classical models are Data Studio pipeline operators, trainable graph models, or a separate backend mode.

### UI graph discoverability

Current state:

- Nodes may exist in backend code but not be visible/searchable in both Studio node search locations.
- Operator-backed materializer nodes, training backend nodes, and executor-only nodes are not unified under one source of truth.

Needed:

- Use a single node registry source for:
  - canvas right-click search.
  - node panel search.
  - node metadata.
  - runtime capability.
  - graph compiler support.
  - materializer support.
- Make category naming explicit. Loss nodes should likely live under a training/loss/optimization category, but they are not optimizers.
- Make unsupported/fail-closed nodes visible with clear status rather than silently absent.

### Materializer backend coverage

Current state:

- Arrow table only.
- Parquet/image/audio/text metadata are pass-through.

Needed:

- If developers want preprocessing nodes to affect Parquet without loading the whole table, implement Parquet materialization or streaming transforms.
- If developers want image/audio graph preprocessing to materialize, define tensor/file-domain operator contracts separate from Arrow table operators.
- If developers want legacy text metadata to materialize directly, route text through raw Arrow consistently and retire the duplicate legacy text batcher path.

### Batching edge cases

Current state:

- Arrow/Parquet `drop_last` is logged as unsupported.
- Parquet row-level stratification is not available.
- Parquet multi-chunk feature fallback is incomplete.
- Image/audio validation split is randomized but not stratified.
- Text legacy path converts to Arrow internally but still lacks balanced splits.

Needed:

- Implement `drop_last` consistently.
- Centralize split/index generation.
- Add label statistics extraction per backend.
- Add deterministic split manifests for reproducibility.

## Recommended developer target architecture

The current code duplicates split/sampling logic across batchers. The clean target is:

```text
Dataset label/index introspection
    |
    v
SplitPlanner
    |
    | input:
    |   num_samples
    |   labels optional
    |   groups optional
    |   ratios
    |   seed
    |   strategy = sequential/shuffle/stratified
    |
    v
Sampler
    |
    | input:
    |   train indices
    |   labels optional
    |   mode = normal/balanced/weighted/undersample
    |
    v
BatchIndexProvider
    |
    | consumed by Arrow/Parquet/Image/Audio/Text/Sequence batchers
    v
IBatcher / ISequenceBatcher
```

This would remove the need to implement stratification and balancing separately in every batcher.

## Quick answers for prior questions

### Does the engine currently support weighted loss?

Not end-to-end from Studio graph training. Common losses exist and are built, but class/positive weights are not exposed and wired through the normal compile/build path.

### Does the engine currently support decision tree training?

Not in the normal neural `TrainingExecutor` graph training path. Some classical operations exist as pipeline/executor nodes, but decision tree model training is not a compiled training backend.

### Does the DataLoader resample or balance class imbalance?

No. Current behavior is shuffled splitting for some domains and simple ratio/row-group splitting for Arrow/Parquet. No class-balanced resampling exists.

### Are MSE and BCE the same?

No.

- MSE is mean squared error, commonly used for regression.
- BCE is binary cross entropy, commonly used for binary or multi-label classification.
- BCEWithLogits combines sigmoid and BCE in a numerically safer form.

### Is loss an optimization function?

No. A loss function measures error. An optimizer updates parameters using gradients from that loss.

In graph/UI terms, loss nodes should be grouped near training/optimization because they are part of the training objective, but they are not optimizer nodes.

