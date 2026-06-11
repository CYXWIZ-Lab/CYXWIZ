# CyxWiz Materializer Internals

This document explains how the CyxWiz Engine turns a visual data pipeline into
the concrete dataset used by training. The materializer is important because the
graph shown in the UI is not the same thing as the data object consumed by the
training loop.

## Short Version

The materializer runs before training starts. It walks forward from the graph's
`DataInput` or `DatasetInput` node, applies supported table operators to the
loaded Arrow table, and registers the transformed table in `DataRegistry` under
a generated name:

```text
<source_dataset_name>__materialized
```

Training then uses that materialized dataset name instead of the original input
dataset name.

Example:

```text
sentiment_mental_health
  -> TextTokenizer
  -> LSTM
  -> Dense

materializes to:

sentiment_mental_health__materialized
```

The training executor then opens `sentiment_mental_health__materialized` through
the Arrow batcher.

## Why The Materializer Exists

The engine has two different kinds of graph work:

- Data transformation work: tokenization, vectorization, scaling, time-series
  windowing, feature engineering, and similar table operations.
- Model training work: layers, loss, optimizer, and DataLoader configuration.

Training cannot consume UI nodes directly. It needs a concrete batch source:
usually an Arrow table, a Parquet-backed dataset, or a specialized image/audio/
text batcher. The materializer bridges that gap for supported table pipelines.

Without the materializer, a training graph could compile a model while silently
ignoring the preprocessing nodes in front of it. With the materializer, supported
preprocessing nodes produce a real transformed table before training starts.

## Main Components

Key files:

- `cyxwiz-engine/src/gui/graph_training_launcher.cpp`
- `cyxwiz-engine/src/core/pipeline_materializer.h`
- `cyxwiz-engine/src/core/pipeline_materializer.cpp`
- `cyxwiz-engine/src/core/pipeline_table_materializer.cpp`
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp`
- `cyxwiz-engine/src/core/node_executors/pipeline_operator.h`
- `cyxwiz-engine/src/core/data_registry_utils.cpp`

Key runtime objects:

- `GraphTrainingLauncher`: starts training from a compiled graph.
- `PipelineMaterializer`: decides whether materialization is needed and stores
  the resulting dataset.
- `IPipelineOperator`: table-transform interface implemented by materializable
  nodes.
- `PipelineOperatorFactory`: creates the correct operator implementation for a
  graph node.
- `DataRegistry`: owns the loaded and generated datasets.
- `ArrowDataset`: wrapper around an in-memory `arrow::Table`.
- `ArrowDatasetBatcher`: turns the materialized Arrow table into training
  batches.

## Training Launch Flow

When the user clicks Train, the engine does roughly this:

```text
StartGraphTrainingFromCompiledConfig
  1. Validate compiled TrainingConfiguration.
  2. Resolve the source dataset name from config or DataInput.
  3. Resolve the label column.
  4. Apply legacy loop params if old optimizer nodes still carry them.
  5. Call PipelineMaterializer::Materialize(...).
  6. If materialization produced a new dataset, replace config.dataset_name.
  7. Resolve the runtime label column again on the materialized table.
  8. Dispatch TrainingManager / TrainingExecutor.
```

The important handoff is step 6. The graph may start with `dataset_name =
"sentiment_mental_health"`, but after materialization the training config points
at `sentiment_mental_health__materialized`.

## What Materialize() Does

`PipelineMaterializer::Materialize(...)` is the registry-aware wrapper.

It:

1. Starts with `effective_dataset_name = source_dataset_name`.
2. Checks that the source dataset name is not empty.
3. Resolves the source storage kind:
   - `ArrowTable`
   - `ParquetBacked`
   - `ImageDataset`
   - `AudioDataset`
   - `TextDataset`
   - `Unknown`
4. Checks whether that storage kind is supported by the materializer.
5. For unsupported storage kinds, skips materialization and leaves the original
   dataset name unchanged.
6. For supported Arrow sources, fetches the `ArrowDataset` from `DataRegistry`.
7. Calls `MaterializeTable(...)` to apply table operators.
8. If no operators were applied, returns the original dataset.
9. If operators were applied, registers the resulting table as:

```text
<source_dataset_name>__materialized
```

10. Returns that name as `effective_dataset_name`.

## What MaterializeTable() Does

`PipelineMaterializer::MaterializeTable(...)` is the table transformation pass.
It does not know about training. It only receives:

- graph nodes
- graph links
- the source `arrow::Table`
- the source dataset name

It then:

1. Finds the matching `DataInput` or `DatasetInput` node.
2. Checks whether any node in the graph is an Arrow-table materializer operator.
3. Validates that the materializer path is linear enough to execute safely.
4. Walks the graph forward from the data input.
5. For each materializable node:
   - resolves the operator type,
   - creates an `IPipelineOperator` through `PipelineOperatorFactory`,
   - builds the parameter map from the node,
   - calls `Configure(params, err)`,
   - calls `Apply(current_table)`,
   - replaces `current_table` with the returned table.
6. Returns the final table and the number of operators applied.

The materializer is therefore a table pipeline:

```text
current_table = source_table
current_table = TextTokenizer.Apply(current_table)
current_table = TFIDFVectorizer.Apply(current_table)
current_table = TimeSeriesSplit.Apply(current_table)
...
```

## Supported Source Scope

Current materialization scope is intentionally narrow.

Supported:

- In-memory Arrow table datasets.

Pass-through / unsupported for materialization:

- Parquet-backed datasets.
- Image datasets.
- Audio datasets.
- Legacy text datasets that are not registered as Arrow tables.
- Unknown storage backends.

Unsupported does not necessarily mean training cannot run. It means the
materializer does not transform that source. The training launcher may still use
another batcher path.

The source-scope decision is deliberate. It prevents the engine from pretending
that a preprocessing node ran when that storage backend does not have a truthful
operator implementation yet.

## Materializable Operators

Materializable nodes are nodes with an `IPipelineOperator` implementation and
runtime capability metadata. Examples include:

- `TextTokenizer`
- `TFIDFVectorizer`
- `CountVectorizer`
- `TimeSeriesWindow`
- `TimeSeriesSplit`
- `TimeSeriesFeatures`
- `StandardScaler`
- `MinMaxScaler`
- `RobustScaler`
- `LabelEncoder`
- `OrdinalEncoder`
- `TargetEncoder`
- `OutlierDetector`
- `PCANode`
- clustering and selected time-series/signal operators

The capability list lives in `pipeline_runtime_capabilities.cpp`. Operator
implementations live under `cyxwiz-engine/src/core/node_executors/`.

## Text Pipeline Example

For sentiment analysis, the graph may look like:

```text
DataInput(sentiment_mental_health.csv)
  -> TextTokenizer(max_length=128, vocab settings...)
  -> LSTM
  -> Dense
  -> CrossEntropyLoss
  -> Adam
```

The materializer sees `TextTokenizer` as a table operator. It applies tokenizer
logic before training, producing a wide numeric Arrow table:

```text
token_0, token_1, ..., token_127, y
```

Then it registers:

```text
sentiment_mental_health__materialized
```

The model builder sees the LSTM layer. The training batcher sees only the
materialized numeric dataset.

## Label Column Resolution

After materialization, the label column may not have exactly the same name the
graph originally requested. For example, text operators commonly produce `y` as
the training label column.

Because of that, `StartGraphTrainingFromCompiledConfig` resolves the label column
again after materialization:

```text
ResolveRuntimeArrowLabelColumn(...)
```

If the requested label is missing, the engine falls back to common label names.
This prevents training from failing simply because an operator normalized the
label column name.

## Why You See "__materialized"

`__materialized` means the dataset was generated by graph execution, not loaded
directly from disk.

The suffix is:

```cpp
PipelineMaterializer::kMaterializedSuffix
```

The generated dataset name is deterministic:

```text
<source_dataset_name>__materialized
```

This makes logs, debugging, and training reproducible. Rerunning the same graph
uses the same generated name instead of creating unlimited temporary dataset
names.

## "Already Exists, Overwriting"

You may see:

```text
Arrow dataset 'sentiment_mental_health__materialized' already exists, overwriting
```

This happens when the same graph is trained or materialized again. The engine
already has a generated Arrow dataset with the deterministic materialized name,
so `DataRegistry::RegisterArrowTable(...)` replaces it with the newly generated
table.

For graph-generated materialized datasets, this is normally expected. It is not
a training error. It simply means:

```text
old materialized table -> replaced by new materialized table
```

This is useful because:

- rerunning the graph picks up changed tokenizer/vectorizer/scaler settings;
- the registry does not accumulate stale generated datasets;
- training always uses the latest table produced by the graph.

The log level should ideally be `info` for expected materialized reruns, and
`warning` only when overwriting a user-loaded dataset or a generated dataset with
an incompatible schema.

## Relationship To Arrow And Parquet

Arrow is the materializer's active table representation. Operators receive and
return `std::shared_ptr<arrow::Table>`.

Parquet is a disk storage format. CyxWiz can load and train from Parquet-backed
datasets, but the current materializer does not rewrite Parquet row groups
through the operator pipeline. Parquet-backed sources are passed through unless
they are first loaded or converted into an in-memory Arrow table path.

The practical rule:

```text
Materializer works on Arrow tables today.
Training can work on Arrow or Parquet-backed datasets.
```

## Error Handling

Materialization returns structured results:

- `success`
- `error_message`
- `operators_applied`
- `effective_dataset_name`
- `source_kind`
- `skipped_unsupported_source`
- `unsupported_source_reason`

Hard failures include:

- empty source dataset name;
- source dataset not found;
- null source Arrow table;
- invalid graph shape for materializer execution;
- operator factory failure;
- operator `Configure()` failure;
- operator `Apply()` failure;
- registry storage failure.

Unsupported storage backends are not hard failures. They produce a pass-through
result with `skipped_unsupported_source=true`.

## Debugging Logs

Useful log messages:

```text
PipelineMaterializer: applied operator '<operator>' on node '<node>'
PipelineMaterializer: materialized '<source>' -> '<source>__materialized' (N operators applied)
StartTrainingFromGraph: materialized '<source>' -> '<source>__materialized' (N Cat-1 ops)
StartTrainingFromGraph: resolved Arrow label column '<old>' -> '<new>' for materialized dataset '<name>'
TrainingExecutor: Arrow split samples train=<n> val=<n> test=<n>
```

These messages tell you:

- which operators actually ran;
- whether a generated dataset was created;
- which dataset training will consume;
- what label column training will use;
- how the materialized table was split for training, validation, and test.

## Current Limits

The materializer is not a full general graph executor. Important limits:

- It currently targets Arrow-table transformations.
- It expects a safe, mostly linear data path from the source input.
- It does not run model layers.
- It does not run loss or optimizer nodes.
- It does not materialize unsupported image/audio/Parquet-backed sources.
- It does not automatically prove that every visible UI node has changed the
  data; only registered materializer operators do.

These limits are intentional. They keep training truthful while more node
families are moved from placeholder/legacy behavior into real operator-backed
execution.

## Design Rule

The materializer should prefer a truthful pass-through over a fake transform.

If a node or storage backend is unsupported, the engine should say so or skip it
explicitly. It should not silently claim that transformed training data exists
when no real table was produced.
