---
title: "Decision 0002: Arrow is the tabular data path"
description: Why tables are Apache Arrow in memory and Parquet on disk, behind one registry entry point.
type: explanation
area: internals
audience: contributor
status: stable
since: 1.0.7
owner: internals
last_reviewed: 2026-11-25
related:
  - README.md
  - ../../concepts/data-pipeline.md
  - ../../concepts/data-bigger-than-memory.md
---

**Status:** accepted.

## Context

Training data ranges from a few kilobytes to more than the machine's
memory. An older stack of dataset handles, streaming datasets and an LRU
cache gave several half-working paths, settings that did nothing, and slow
column access.

## Decision

- Tabular data is **Apache Arrow**. `DataRegistry::LoadTabularCSV` is the
  one entry point, and it picks the back end itself:
  - in memory: an `ArrowDataset`, with integer columns narrowed where the
    values fit;
  - on disk, when the table is larger than the data budget or the user
    forces it: a Snappy-compressed Parquet cache, memory-mapped by
    `ParquetBackedDataset` and read page by page while training.
- Both register under the same name. Training picks the matching batcher
  (`ArrowDatasetBatcher` or `ParquetArrowBatcher`) with the same model,
  loss and optimizer.
- The older lazy-loading stack was removed, not reconnected.

## Consequences

- Speed and memory come from Arrow and good defaults, not from user knobs.
- Data steps are Arrow-table operations (`IPipelineOperator::Apply`), so
  their results can be cached and shared.
- Parquet caches are pruned by size and age; a cache file in use is skipped
  and retried later.
