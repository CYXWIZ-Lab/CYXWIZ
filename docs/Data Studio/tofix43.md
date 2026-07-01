# tofix43 - Materialization Memory Safety and Large Dataset Guardrails

## Status

Open.

## Problem

TF-IDF exposed a broader engine issue: materialization can allocate more memory
than the user's machine can safely provide.

TF-IDF is only a small case study. Users can build much larger preprocessing
pipelines and models where materialized intermediate data may require many GBs
of RAM, for example 20GB or more. CyxWiz must not wait until the OS starts
thrashing, the GUI freezes, allocation fails, or the process crashes.

The engine needs a materialization memory-safety system before training starts.

## Current Risk

Materialization may currently:

- read large source data into memory
- build temporary token/statistics maps
- allocate dense feature matrices
- allocate Arrow builders and Arrow arrays
- keep intermediate and final datasets alive together
- block the GUI if preparation is heavy
- fail late with `bad allocation`
- make the engine appear frozen while the OS pages memory to disk

For TF-IDF:

```text
rows * features * 4 bytes
```

Example:

```text
52,681 rows * 8,000 features * 4 bytes ~= 1.68 GB raw float data
```

But raw feature memory is not peak memory. Peak memory can be much higher due
to CSV/text buffers, token maps, vocabulary maps, Arrow builders, Arrow arrays,
metadata, and train/validation/test loader state.

## Goal

Before materialization starts, CyxWiz should estimate memory risk and either:

- allow materialization safely
- warn the user clearly
- require confirmation for risky materialization
- automatically suggest safer graph settings
- block materialization when estimated memory is likely to exceed safe limits

The engine should fail gracefully with actionable diagnostics, not freeze or
crash.

## Required Capabilities

### 1. Materialization Memory Estimator

Add a generic estimator for materializer operators.

Each operator should report:

- expected input rows
- expected output rows
- expected output columns/features
- expected output dtype
- estimated raw output bytes
- estimated temporary bytes
- estimated Arrow builder/array overhead
- estimated peak bytes
- confidence level

The estimator should work before full materialization where possible.

### 2. System Memory Budget

Add runtime memory-budget detection:

- total system RAM
- currently available RAM
- optional user-configured max materialization memory
- safety margin
- OS paging risk threshold

Example policy:

```text
safe:     estimated_peak < 40% available RAM
warning:  estimated_peak 40-70% available RAM
risky:    estimated_peak 70-90% available RAM
blocked:  estimated_peak > 90% available RAM or > configured hard limit
```

Exact thresholds can be tuned later.

### 3. Preflight Integration

Compiler/preflight should surface materialization memory risk before training:

- estimated peak memory
- source operator/node
- output shape estimate
- risk level
- reason
- recommended changes

Examples:

- reduce TF-IDF `max_features`
- reduce row count/sample first
- use sparse output when available
- use disk-backed materialization
- use chunked materialization
- avoid high-cardinality one-hot expansion

### 4. Runtime Guardrails

During materialization:

- track actual memory growth
- compare actual memory against estimate
- stop gracefully before exhausting memory
- emit structured error if guardrail trips
- clean up partial intermediates
- keep GUI responsive

The runtime should produce a clear error such as:

```text
Materialization blocked: estimated peak memory 21.4GB exceeds available safe budget 9.8GB.
Node: TF-IDF 20000
Suggestion: reduce max_features to 5000, enable sparse output, or use chunked materialization.
```

### 5. Studio Debugger and Dashboard Visibility

Training Dashboard and Studio Debugger should show:

- memory estimate before materialization
- risk level
- active stage memory estimate
- actual memory snapshots
- guardrail decision
- recommended graph changes

This should connect to the materialization breakdown added in the training trace.

### 6. Sparse and Chunked Follow-up Path

Dense materialization is not enough for large datasets.

Future paths:

- sparse TF-IDF output
- sparse model input support
- chunked materialization
- disk-backed Arrow datasets
- streaming DataLoader from materialized chunks
- partial-fit/statistics-first operators where applicable

These do not all need to be implemented in the first pass, but the memory-safety
system should point users toward them.

## Operator Cases to Audit

Materialization memory safety must cover more than TF-IDF.

Audit at least:

- TFIDFVectorizer
- CountVectorizer
- TextTokenizer
- OneHotEncoder / categorical expansion operators
- TimeSeriesWindow
- TimeSeriesFeatures
- PCA
- FFT / signal operators
- image/video/audio materializers if present
- tree/classical model materializers that create native model artifacts
- any operator that expands rows or columns

## UI Requirements

### Before Training

When user clicks Train:

- show `Preparing materialization plan...`
- estimate memory
- show risk level
- if warning/risky, show decision text before heavy allocation
- if blocked, do not start materialization

### During Materialization

Show:

- current stage
- estimated memory
- actual memory snapshot
- processed rows/items
- output dataset name
- operator/node responsible

### On Failure or Block

Show:

- what was blocked or failed
- estimated vs available memory
- node responsible
- exact property causing expansion where possible
- recommended graph changes

## Acceptance Criteria

- Large materialization requests are detected before major allocation.
- Risky requests produce visible warnings, not silent freezes.
- Clearly impossible requests are blocked before training starts.
- Materializer memory errors are structured and actionable.
- Studio Debugger records the decision and responsible node.
- Training Dashboard shows memory estimate and risk level.
- Existing small materialization graphs continue to work.
- TF-IDF 5000/8000 graphs show estimated memory before materialization starts.

## First Implementation Slice

Start with TF-IDF because it exposed the issue:

- estimate dense TF-IDF output bytes from rows and `max_features`
- estimate peak memory with a conservative multiplier
- compare against available RAM
- warn/block based on thresholds
- surface estimate in Training Dashboard and Studio Debugger
- produce actionable suggestions

Then generalize the estimator interface to other materializers.

## Notes

This is separate from GPU memory. TF-IDF materialization currently runs on CPU
and uses system RAM. GPU memory becomes relevant later when batches are moved to
GPU for model computation.

Pinned memory does not solve materialization memory pressure. Pinned memory only
helps CPU-to-GPU transfer after batches already exist.
