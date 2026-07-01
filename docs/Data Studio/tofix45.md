# tofix45 - Sparse Text Feature Path

## Status

Open.

## Problem

`TFIDFVectorizer` and `CountVectorizer` now support better dense text features:

- n-grams
- stop-word mode
- binary count features
- dense materialization warnings
- explicit `output_format=dense`

However, large text models still create wide dense Arrow tables. Dense memory
cost grows as:

```text
rows * max_features * 4 bytes
```

This is acceptable for small and medium experiments, but it does not scale to
large text datasets or high-dimensional vocabularies.

The engine currently has this dense path:

```text
Vectorizer
  -> dense Arrow float columns
  -> ArrowDataset
  -> ArrowDatasetBatcher
  -> dense Tensor
  -> Dense/Linear layer
```

Sparse support cannot be implemented only inside the vectorizer. It needs a
real sparse data path through materialization, batching, model input, and at
least the first linear layer.

## Goal

Add a first-class sparse text feature path that avoids dense `rows * features`
RAM blowups while staying explicit and debuggable.

## Non-goals

- Do not silently densify huge sparse matrices.
- Do not make `output_format=sparse` pretend to work until training supports it.
- Do not introduce a broad generic sparse tensor framework before the text
  vectorizer use case proves the shape of the abstraction.

## Proposed Engine Contract

Vectorizer parameter:

```text
output_format=dense | sparse
```

Current behavior:

```text
dense  supported
sparse fail-closed
```

Target behavior:

```text
dense  dense Arrow table, current behavior
sparse sparse text feature dataset
```

## Sparse Representation

Start with CSR-like row storage:

```text
row_offsets: int64[num_rows + 1]
col_indices: int32[nnz]
values: float32[nnz]
num_rows: int64
num_features: int64
feature_names: optional string[num_features]
label: optional int32[num_rows]
```

This is enough for TF-IDF and CountVectorizer because each document is naturally
a sparse row.

## Phase 1 - Sparse Dataset Object

Add a minimal sparse dataset type:

- name
- row count
- feature count
- nnz
- density
- memory estimate
- labels
- row offsets
- column indices
- values

Add registry support:

- register sparse dataset
- fetch sparse dataset
- list sparse dataset in debugger/profile views

Acceptance:

- sparse dataset can be registered and inspected
- no training integration yet
- no silent dense conversion

## Phase 2 - Sparse Vectorizer Output

Enable:

```text
TFIDFVectorizer output_format=sparse
CountVectorizer output_format=sparse
```

Output should register a sparse dataset instead of a dense Arrow table.

Acceptance:

- sparse TF-IDF materializes `row_offsets`, `col_indices`, `values`
- sparse CountVectorizer materializes the same structure
- materializer reports `nnz`, density, and estimated sparse memory
- dense output remains unchanged

## Phase 3 - Sparse Batcher

Add a sparse batcher that can produce mini-batches without building the full
dense matrix.

Initial contract:

- either sparse mini-batch object
- or controlled batch-local densification

Batch-local densification is acceptable as a first implementation if:

- it only densifies one batch at a time
- it logs the batch dense memory estimate
- it refuses unsafe batch sizes

Acceptance:

- sparse dataset can train with batch-local densification
- memory is bounded by `batch_size * num_features`, not `rows * num_features`

## Phase 4 - Sparse-aware First Linear Layer

Add a sparse-aware first linear operation:

```text
sparse_batch @ dense_weights + bias
```

This avoids even batch-local full densification.

Acceptance:

- first Dense/Linear layer can consume sparse batches
- backward path accumulates gradients correctly
- CPU path works first
- GPU path can remain follow-up if not safe yet

## Phase 5 - Debugger and UX

Debugger should show:

- sparse vs dense output
- rows
- features
- nnz
- density
- dense memory estimate
- sparse memory estimate
- densification points, if any

Properties panel should explain:

- dense is simpler and currently fastest for small/medium data
- sparse is for large high-dimensional text features
- sparse may limit downstream model choices until more layers support it

## Tests

Required tests:

- sparse dataset registration
- sparse TF-IDF correctness on tiny corpus
- sparse CountVectorizer correctness on tiny corpus
- dense and sparse outputs agree numerically when densified for comparison
- sparse batch-local densification shape correctness
- sparse training smoke test once batcher exists

## Acceptance Criteria

- `output_format=sparse` is supported for TF-IDF and CountVectorizer.
- Dense output remains unchanged.
- Sparse materialization does not allocate `rows * max_features` floats.
- Debugger reports sparse materialization truth.
- Training can consume sparse text features without full-dataset densification.

## Relationship to Other Tickets

- `tofix44`: dense text-feature quality and explicit dense-only contract.
- `tofix43`: memory guardrails and fail-fast budgeting.
- `tofix39`: numerical correctness tests should eventually compare sparse and
  dense vectorizer outputs.
