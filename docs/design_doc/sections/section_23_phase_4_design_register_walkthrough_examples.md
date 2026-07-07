## 23) Phase 4 design register (walkthrough examples)

Goal:
- convert abstract contracts into concrete, linearly executable end-to-end stories.
- bind each story to launch path, materialization policy, and execution mode.

### 23.1 Baseline tabular image-classification path (legacy image domain)

```text
Nodes:
DataInput -> Resize -> Normalize -> Dense -> CrossEntropyLoss -> Adam
```

Flow:

1) Compile (`GraphCompiler::Compile`)
- resolves source as image domain (`PreprocessingDomain::Image`),
- populates `TrainingConfiguration` preprocessing from graph resize/normalize nodes.

2) Prepare/launch (`StartGraphTrainingFromCompiledConfig`)
- `FindDatasetName` resolves dataset key from `DataInput`.
- materializer invoked on source kind:
  - `ImageDataset` => `PipelineMaterializerStorageSupport::None`
  - result is `skipped_unsupported_source = true`, dataset unchanged.
- no sequence validation block when `sequence_batch.enabled == false`.

3) Dispatch (`dispatch` lambda in `MainWindow`)
- `dispatch_config.sequence_batch.enabled == false` so loader path used.
- `GetByRegisteredDataset` resolves `ImageLoader`.
- `ImageLoader::LaunchTraining` -> `TrainingManager::StartTrainingImage`.

4) Runtime
- `StartTrainingImage` builds `ImageDatasetBatcher`.
- `TrainingExecutor` constructed with `DatasetMode::External`.
- loop runs through `external_batcher_` path.

Acceptance:
- no hard fail from materializer,
- image batcher successfully created,
- executor mode external image.

### 23.2 Text classification path (text domain)

```text
Nodes:
DataInput -> TextTokenizer -> TextPadding -> Embedding -> Dense -> CrossEntropyLoss -> Adam
```

Flow:

1) Compile
- resolves text preprocessing config (`TextTokenizer`, padding settings, vocab defaults).

2) Launch preflight
- materializer attempts source lookup:
  - if source is `TextDataset`, cat-1 materialization is skipped.

3) Dispatch
- loader path selected by registry:
  - `TextLoader::LaunchTraining` receives `dataset_name` and `label_column`.
  - calls `TrainingManager::StartTrainingText`.

4) Runtime
- `TextDatasetBatcher` tokenizes/batches with compiled config.
- `TrainingExecutor` mode `External` and `config.input_size = max_length`.

Acceptance:
- text domain remains external, no operator rewrite on non-Arrow sources,
- input size and shapes derived from batcher output.

### 23.3 Sequence/NER-like path (Arrow-backed sequence + token batcher)

```text
Nodes:
DataInput -> NERSequenceBuilder -> TimeSeriesWindow/TimeSeriesFeatures -> LSTM -> SequenceTagOutput -> NLLLoss
```

Flow:

1) Compile
- graph marks `config.sequence_batch.enabled = true`,
- sequence columns inferred from sequence contract nodes.

2) Launch guard (`StartGraphTrainingFromCompiledConfig`)
- sequence branch first checks source in registry is Arrow or Parquet.
- materializer runs:
  - if source is Arrow and sequence ops are operator-backed, materialization may produce `<source>__materialized`.

3) Sequence validation
- `ValidateSequenceLaunchColumns` checks required `token` and `tag` columns, optional `pos` and `sentence_id`.
- failure here blocks before executor creation.

4) Dispatch
- sequence branch builds a dedicated batcher:
  - `BuildSequenceBatcherFromArrowDataset`
  - applies `ApplySequenceBatcherBuildResultToTrainingConfig`
  - calls `TrainingManager::StartTrainingSequence`

5) Runtime
- executor mode `SequenceExternal`.
- sequence metrics and metrics panel path used.

Acceptance:
- sequence path selected only when enabled,
- sequence columns pass validation,
- no fallback to legacy table training.

### 23.4 Tabular Arrow/Parquet path with materializer operators

```text
Nodes:
DataInput -> TimeSeriesWindow -> Normalize -> Dense -> CrossEntropyLoss -> Adam
```

Flow:

1) Compile
- domain likely tabular/time-series.

2) Materialization
- source kind can be `ArrowTable` or `ParquetBacked`.
- Arrow:
  - materializer executes supported operator chain and may register
    `dataset__materialized`.
- Parquet:
  - materializer is skipped as unsupported source kind.

3) Dispatch
- Arrow: loader route may call `StartTrainingArrow` (direct arrow constructor).
- Parquet: loader route calls `StartTrainingParquet`.

4) Runtime
- Arrow/Parquet constructors apply `ResolveTabularTrainingInputSize` and build
  dedicated row-group/table batchers.

Acceptance:
- Arrow source: operator materialization visible in task trace.
- Parquet source: no rewrite, direct parquet batch path.

### 23.5 Legacy or blocked node path

```text
Nodes:
DataInput -> TSNENode -> Dense -> CrossEntropyLoss -> Adam
```

Flow:

1) Compile/runtime support query
- `TSNENode` appears in fail-closed table with hard reason.

2) Launch behavior
- launch should not proceed into executor creation.
- failure surfaced via `GraphTrainingLaunchResult` with blocked title + detail.

Acceptance:
- training does not start,
- user receives explicit blocker text and recovery direction.

### 23.6 Cross-cutting execution branch matrix

```text
Input source kind      Preprocessor/domain                 Launch dispatcher branch             Training mode
-------------------   ----------------------------------  ----------------------------------  --------------------
ImageDataset          Image preprocessing                  DataLoader::LaunchTraining (Image)    External
AudioDataset          Audio preprocessing                  DataLoader::LaunchTraining (Audio)    External
TextDataset           Text preprocessing                   DataLoader::LaunchTraining (Text)     External
ArrowTable            tabular/domain inference             DataLoader::LaunchTraining           Arrow/Parquet sequence*
ParquetBacked         tabular/time-series                 DataLoader::LaunchTraining           Parquet / SequenceExternal*
legacy DatasetHandle  generic tabular/no loader            Direct StartTraining (legacy)         Legacy

* Sequence depends on config.sequence_batch.enabled and BuildSequenceBatcherFromArrowDataset.
```

### 23.7 Walkthrough tracepoints to verify on each run
At minimum capture:
- launch result `started`, `effective_dataset_name`, `label_column`,
- materializer:
  - `skipped_unsupported_source`,
  - `operators_applied`,
  - source kind and diagnostic.
- dispatch branch and dataset mode.
- sequence validation decision and batcher type.

Use these values to prove the exact execution branch for each design case.
