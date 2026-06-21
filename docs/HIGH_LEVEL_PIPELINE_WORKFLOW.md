# CyxWiz High-Level Data Pipeline Workflow

This document explains how data should flow through CyxWiz from dataset
loading to algorithm consumption and training.

The key idea is to keep four concepts separate:

```text
storage format -> query/table format -> tensor format -> training batch
```

A Parquet file, Arrow table, DuckDB query result, and CyxWiz `Tensor` are
not the same thing. They are different stages of the pipeline.

## Pipeline Overview Diagram

```text
+--------------------+
| Dataset Storage    |
| CSV / JSON /       |
| Parquet / Arrow    |
+--------------------+
          |
          v
+--------------------+       +----------------------+
| Scanner / Query    |<----->| Schema / Metadata    |
| DuckDB / Arrow     |       | columns, dtypes,     |
| projection/filter  |       | nullability, labels  |
+--------------------+       +----------------------+
          |
          v
+--------------------+
| Tabular Batches    |
| chunks, columns,   |
| null masks         |
+--------------------+
          |
          v
+--------------------+       +----------------------+
| Feature Builder    |<----->| Transform State      |
| encode labels,     |       | means, stds, vocab,  |
| select columns     |       | category mapping     |
+--------------------+       +----------------------+
          |
          v
+--------------------+
| CyxWiz Tensors     |
| X features         |
| y labels           |
+--------------------+
          |
          v
+--------------------+
| DataLoader         |
| shuffle, batch,    |
| prefetch, collate  |
+--------------------+
          |
          v
+--------------------+
| DataBatch          |
| Tensor data        |
| Tensor labels      |
+--------------------+
          |
          v
+--------------------+
| Training Step      |
| forward, loss,     |
| backward, optimize |
+--------------------+
          |
          v
+--------------------+
| Outputs            |
| metrics, logs,     |
| checkpoint, model  |
+--------------------+
```

## Under-The-Hood Compute Diagram

```text
                 APPLICATION / STUDIO LAYER

  +--------------------------------------------------------------+
  | User picks dataset, feature columns, label column, model,    |
  | transforms, split policy, batch size, optimizer, metrics     |
  +--------------------------------------------------------------+
                                  |
                                  v

                 DATA ACCESS / QUERY LAYER

  +-------------------+     +-------------------+     +----------------+
  | File path / URI   | --> | DuckDB SQL plan   | --> | DuckDB chunks  |
  | parquet/csv/json  |     | projection/filter |     | typed vectors  |
  +-------------------+     +-------------------+     +----------------+
                                                            |
                                                            v

                 TABLE / FEATURE ENGINEERING LAYER

  +-------------------+     +-------------------+     +----------------+
  | Schema inspector  | --> | Feature selector  | --> | Label encoder  |
  | names and types   |     | numeric columns   |     | string -> int  |
  +-------------------+     +-------------------+     +----------------+
                                                            |
                                                            v

                 TENSOR CONVERSION LAYER

  +-------------------+     +-------------------+     +----------------+
  | Null policy       | --> | dtype conversion  | --> | row-major or   |
  | fill/drop/error   |     | float/int tensors |     | column bridge  |
  +-------------------+     +-------------------+     +----------------+
                                                            |
                                                            v

                 TRAINING DATA LAYER

  +-------------------+     +-------------------+     +----------------+
  | Tensor features X | --> | DataBatch         | --> | Async queue /  |
  | Tensor labels y   |     | data + labels     |     | prefetch       |
  +-------------------+     +-------------------+     +----------------+
                                                            |
                                                            v

                 MODEL COMPUTE LAYER

  +-------------------+     +-------------------+     +----------------+
  | model.forward(X)  | --> | loss(pred, y)     | --> | backward grads |
  | layer kernels     |     | scalar loss       |     | parameter grad |
  +-------------------+     +-------------------+     +----------------+
                                                            |
                                                            v

                 OPTIMIZATION / REPORTING LAYER

  +-------------------+     +-------------------+     +----------------+
  | optimizer.step()  | --> | metrics/update    | --> | checkpoint +   |
  | update weights    |     | accuracy/loss     |     | debugger event |
  +-------------------+     +-------------------+     +----------------+
```

## Current vs Target Data Path Diagram

Current DuckDB path:

```text
CSV / JSON / Parquet
        |
        v
DuckDB query
        |
        v
duckdb_result chunks
        |
        v
std::vector<float> row-major copy
        |
        v
cyxwiz::Tensor [rows, cols]
```

Current training dataset path:

```text
DatasetBase
        |
        v
GetItem(index)
        |
        v
std::vector<float> sample + int label
        |
        v
DataLoader collects batch
        |
        v
Tensor data + Tensor labels
        |
        v
DataBatch
```

Target unified path:

```text
CSV / JSON / Parquet / Arrow
        |
        v
DuckDB or Arrow scanner
        |
        v
RecordBatch / TabularBatch
        |
        v
Feature columns + label column
        |
        v
TransformPipeline
        |
        v
TensorBatch { features, labels, metadata }
        |
        v
AsyncDataLoader / prefetch queue
        |
        v
Trainer / Evaluator / Inference
```

## 1. Dataset Loading Pipeline

High-level flow:

```text
Dataset files
    |
    |  CSV / JSON / Parquet / future Arrow dataset
    v
File scanner / query engine
    |
    |  DuckDB SQL, Arrow scan, projection, filters, joins
    v
Tabular batch
    |
    |  typed columns, chunks, validity/null masks
    v
Feature/label extraction
    |
    |  numeric conversion, categorical encoding, text tokenization
    v
CyxWiz tensors
    |
    |  feature tensor X, label tensor y
    v
Dataset / DataLoader / BatchIterator
    |
    |  shuffle, batch, normalize, transform, prefetch
    v
Algorithm input batch
    |
    |  model.forward(X_batch)
    v
Loss / metrics / backward / optimizer
```

Current backend has two related but different data paths:

```text
Path A: DuckDB file/query loader

CSV / JSON / Parquet
    -> DuckDB in-memory connection
    -> duckdb_result / data chunks
    -> row-major float buffer
    -> cyxwiz::Tensor [rows, cols]
```

```text
Path B: Dataset/DataLoader training path

DatasetBase
    -> GetItem(index)
    -> std::vector<float> sample + int label
    -> DataBatch
    -> Tensor data + Tensor labels
```

These paths should be unified over time so file/query datasets can feed
the training `DataLoader` without unnecessary full materialization.

## 2. Parquet, Arrow, and DuckDB Roles

### Parquet

Parquet is a storage format.

It is good for:

- large tabular datasets
- column pruning
- compression
- predicate pushdown
- efficient scan of selected columns

Parquet should not be treated as the training representation. It should
be scanned into batches, then converted into tensors only at the point
where algorithms need numeric arrays.

### Arrow

Arrow is an in-memory columnar format.

It is good for:

- zero-copy column exchange between tools
- typed arrays
- validity/null bitmaps
- batch-oriented processing
- avoiding repeated row-wise parsing

CyxWiz does not currently have a first-class Arrow table layer in this
backend. The target design should add one or at least support Arrow-style
column batches as an internal bridge.

Target role:

```text
Parquet scan / DuckDB query
    -> Arrow-style RecordBatch
    -> selected columns
    -> encoded numeric buffers
    -> Tensor batch
```

### DuckDB

DuckDB is the query engine.

It is good for:

- SQL over CSV, JSON, and Parquet
- projection: selecting only needed columns
- filtering rows before training
- joins
- aggregations
- label encoding with SQL `CASE`
- train/validation/test split queries

Current backend path:

```text
DataLoader::LoadParquet(path)
    -> SELECT columns FROM read_parquet(path)
    -> duckdb_query(...)
    -> duckdb_result
    -> ResultToTensor(...)
    -> Tensor [rows, columns]
```

Current batch path:

```text
DataLoader::CreateBatchIterator(sql, batch_size)
    -> repeated SQL query with LIMIT/OFFSET style iteration
    -> duckdb_result
    -> Tensor batch
```

Target improvement:

```text
DuckDB/Arrow stream
    -> chunked RecordBatch
    -> TensorBatch
```

This avoids materializing an entire result when only a mini-batch is
needed.

## 3. Internal Data Representation

The backend should use different internal representations at different
stages.

```text
+------------------+-----------------------+----------------------------+
| Stage            | Representation        | Owner                      |
+------------------+-----------------------+----------------------------+
| Storage          | Parquet/CSV/JSON      | filesystem                 |
| Query/table      | DuckDB result/chunk   | DataLoader query layer     |
| Columnar bridge  | Arrow-style batch     | future table layer         |
| Numeric compute  | cyxwiz::Tensor        | tensor/runtime layer       |
| Training sample  | feature + label       | DatasetBase                |
| Training batch   | DataBatch             | DataLoader                 |
| Model input      | Tensor X, Tensor y    | model/training loop        |
+------------------+-----------------------+----------------------------+
```

Current `cyxwiz::DataLoader` from `data_loader.h` returns:

```text
Tensor [rows, columns]
```

Current `cyxwiz::DataLoader` from `dataloader.h` returns:

```text
DataBatch {
    Tensor data;
    Tensor labels;
    size_t size;
}
```

Target design should make the relationship explicit:

```text
TabularDataset
    -> feature columns
    -> label column
    -> transforms
    -> DataBatch { data, labels }
```

## 4. Transform and Preprocessing Stage

Transforms should happen before algorithm consumption, but not all
transforms need to happen in the same place.

### Query-level transforms

Best for operations DuckDB can do efficiently:

- column selection
- row filtering
- joins
- simple arithmetic
- label encoding with `CASE`
- train/validation/test split
- aggregations for statistics

Example:

```sql
SELECT
    sepal_length,
    sepal_width,
    petal_length,
    petal_width,
    CASE species
        WHEN 'setosa' THEN 0
        WHEN 'versicolor' THEN 1
        WHEN 'virginica' THEN 2
    END AS label
FROM 'iris.parquet'
WHERE sepal_length IS NOT NULL
```

### Table/column transforms

Best for columnar preprocessing:

- missing-value handling
- categorical dictionary encoding
- date/time feature extraction
- text tokenization setup
- normalization statistics

### Tensor transforms

Best for numeric compute:

- normalization
- standardization
- reshape
- one-hot encoding
- image normalization
- tensor augmentations
- masking

Current backend has `DataTransform` functions for:

- normalization
- standardization
- log transform
- Box-Cox
- Yeo-Johnson
- robust scaling
- max-abs scaling
- quantile transform
- power transform
- outlier detection

Current training `DataLoader` also supports:

- one-hot labels
- simple normalization
- shuffling
- batching

Target design:

```text
TransformPipeline
    |
    +-- SQL transforms
    +-- column transforms
    +-- tensor transforms
```

The pipeline should store fitted transform parameters so validation,
test, and inference use the same transformations as training.

## 5. Batch Creation and DataLoader Flow

Training algorithms should consume batches, not raw files.

Target batch lifecycle:

```text
Dataset
    |
    |  knows where data lives and how to read one row or chunk
    v
Sampler
    |
    |  chooses sample indices or row ranges
    v
BatchSampler
    |
    |  groups indices into mini-batches
    v
Collate
    |
    |  converts samples/columns into tensors
    v
TransformPipeline
    |
    |  applies feature/label transforms
    v
DataBatch
    |
    |  Tensor data, Tensor labels, metadata
    v
Training step
```

Current `dataloader.h` flow:

```text
DatasetBase::GetItem(index)
    -> pair<vector<float>, int>
    -> DataLoader collects N samples
    -> optional normalization
    -> VectorToTensor(flat_data, shape)
    -> LabelsToTensor or LabelsToOneHot
    -> DataBatch
```

Current DuckDB flow:

```text
SQL query
    -> ResultToTensor
    -> Tensor batch
```

Gap:

The DuckDB path currently returns a single tensor where feature columns
and label columns may be mixed. The training path expects separate
`Tensor data` and `Tensor labels`.

Needed bridge:

```text
TabularBatch {
    Tensor features;
    Tensor labels;
    column_names;
    row_ids;
    split;
}
```

## 6. Training Loop Flow

High-level training loop:

```text
for epoch in epochs:
    dataloader.reset()

    while dataloader.has_next():
        batch = dataloader.next()

        X = batch.data
        y = batch.labels

        y_pred = model.forward(X)
        loss = loss_fn.forward(y_pred, y)

        grad = loss_fn.backward(y_pred, y)
        model.backward(grad)

        optimizer.step(model.parameters)
        optimizer.zero_grad()

        metrics.update(y_pred, y)

    validation_metrics = evaluate(validation_loader)
    scheduler.step(validation_metrics or epoch)
    checkpoint.maybe_save(model, optimizer, metrics)
    debugger.record(epoch, metrics, timings, errors)
```

Under the hood:

```text
DataBatch
    -> Tensor data
        -> maybe CPU tensor first
        -> maybe ArrayFire/GPU array when used by layer
    -> Model layers
        -> forward cache saved for backward
    -> Loss
        -> scalar loss tensor
    -> Backward
        -> layer gradients
    -> Optimizer
        -> parameter tensors updated
```

The algorithm should not know whether the original dataset was CSV,
Parquet, Arrow, or SQL. By training time, it should only see tensors and
metadata.

## 7. Evaluation, Checkpoint, and Debugger Flow

Validation and test use the same data path but different mode:

```text
validation/test dataset
    -> same fitted transforms
    -> DataBatch
    -> model.eval()
    -> forward only
    -> loss + metrics
    -> debugger/report
```

Important rule:

```text
fit transforms on training data only
reuse fitted transform state on validation/test/inference
```

Checkpoint flow:

```text
model parameters
optimizer state
scheduler state
transform state
label mapping
feature schema
training config
metrics history
```

All of these should be saved together. A model checkpoint without schema
and transform state is not enough for reliable inference.

Debugger flow:

```text
batch timings
data-loader timings
CPU/GPU transfer timings
forward timings
backward timings
loss/metric values
fallback warnings
shape/type errors
checkpoint decisions
```

## 8. Bottleneck Map

Common bottlenecks by stage:

```text
+----------------------+-----------------------------------------------+
| Stage                | Bottlenecks                                   |
+----------------------+-----------------------------------------------+
| File scan            | disk I/O, remote storage latency, compression |
| Parquet decode       | decompression, row-group size, type decoding  |
| DuckDB query         | joins, sort, aggregation, LIMIT/OFFSET scans  |
| Arrow/table bridge   | null handling, string/categorical encoding    |
| Tensor conversion    | row-major copy, dtype cast, unsupported types |
| CPU preprocessing    | normalization, tokenization, image transforms |
| Dataloader           | no prefetch, too few workers, shuffle cost    |
| CPU->GPU transfer    | small batches, repeated copies, pinned memory |
| Model forward        | slow layer kernels, CPU fallback, shape layout|
| Backward             | activation caches, gradient memory, BPTT cost |
| Optimizer            | many small tensors, optimizer state memory    |
| Metrics/debugger     | sync points, frequent polling, logging volume |
+----------------------+-----------------------------------------------+
```

Current specific bottlenecks:

- DuckDB results are converted into a row-major `std::vector<float>`
  before becoming a tensor.
- Unsupported column types are silently converted to zero in some paths.
- String labels need explicit encoding before tensor conversion.
- Full query materialization can exceed memory limits.
- Batch iteration still converts each result batch into a tensor copy.
- Training `DataLoader` builds batches through vectors before tensor
  construction.
- CPU-to-GPU movement is implicit and can be repeated if tensors are not
  cached well.
- UI/debugger polling can become a sync bottleneck if it forces frequent
  metric or tensor inspection.

Target mitigations:

- use projection/filter pushdown before tensor conversion
- stream record batches instead of materializing full results
- add Arrow-style column batch bridge
- add typed feature encoders
- separate features and labels before training
- prefetch batches
- use pinned CPU memory for GPU transfer
- cache GPU arrays when tensor CPU data has not changed
- record timing per stage

## 9. Target Architecture

Target data architecture:

```text
storage/
    ParquetDataset
    CSVDataset
    JSONDataset
    ArrowDataset
    DuckDBDataset

table/
    Schema
    Column
    RecordBatch
    NullMask
    DictionaryEncoding

transforms/
    FitTransform
    TransformState
    Normalize
    Standardize
    EncodeCategorical
    TokenizeText
    ImageTransform

data/
    Dataset
    Sampler
    BatchSampler
    DataLoader
    AsyncDataLoader
    Collate
    DataBatch

tensor/
    Tensor
    TensorBatch
    DeviceTransfer

training/
    Trainer
    Evaluator
    Metrics
    Checkpoint
    DebuggerEvents
```

Target flow:

```text
Parquet/CSV/JSON
    -> DuckDB or Arrow scanner
    -> RecordBatch
    -> TransformPipeline
    -> TensorBatch { features, labels }
    -> AsyncDataLoader queue
    -> Trainer
    -> Model / Loss / Optimizer / Metrics
    -> Checkpoint + Debugger
```

## 10. Current Missing Pieces

Current backend has useful pieces, but the full pipeline is not complete.

Missing or weak:

- first-class Arrow/RecordBatch representation
- unified dataset abstraction for file/query data
- explicit `features` vs `labels` result type for tabular data
- typed categorical encoders
- transform fit/transform state persistence
- async/prefetch dataloader for general tabular data
- collate functions
- train/validation/test split object
- schema attached to tensors
- feature metadata attached to training batches
- robust null/missing-value policies
- string/text pipeline integration with tensor training
- clear CPU/GPU transfer policy
- per-stage performance tracing
- checkpoint persistence for transforms and schema

## Proof Use Case: Iris-Style Tabular Classification

Use case:

Train a small classifier from a Parquet or CSV table with four numeric
feature columns and one string label column.

Input table:

```text
sepal_length | sepal_width | petal_length | petal_width | species
```

Target pipeline:

```text
1. Load schema
       DataLoader::GetSchema(path)

2. Build query
       SELECT
           sepal_length,
           sepal_width,
           petal_length,
           petal_width,
           CASE species
               WHEN 'setosa' THEN 0
               WHEN 'versicolor' THEN 1
               WHEN 'virginica' THEN 2
           END AS label
       FROM path

3. Split rows
       train query
       validation query
       test query

4. Fit transforms on train only
       mean/std for four feature columns

5. Stream training batches
       DuckDB/Arrow chunk -> TensorBatch {features, labels}

6. Train
       y_pred = model.forward(features)
       loss = CrossEntropy(y_pred, labels)
       model.backward(...)
       optimizer.step()

7. Validate
       same transform state
       forward only
       metrics: accuracy, loss

8. Save checkpoint
       model weights
       optimizer state
       label mapping
       feature schema
       normalization parameters

9. Test
       load best checkpoint
       run held-out test query

10. Inference
       one new row/table
       same schema + transforms
       model.forward
       decode predicted class label
```

Current backend can do parts of this now:

- load CSV/Parquet/JSON through DuckDB
- inspect schema
- execute SQL label encoding
- create tensor batches from SQL queries
- perform basic normalization in training dataloader

But the proof use case should drive the next implementation work:

- add `TabularDataset`
- add `TensorBatch {features, labels}`
- add transform state
- add split handling
- add tabular dataloader that streams batches into training
- add metrics/checkpoint/debugger integration
