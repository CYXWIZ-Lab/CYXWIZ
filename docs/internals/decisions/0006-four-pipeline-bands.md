---
title: "Decision 0006: The four pipeline bands"
description: Why every training graph is read as four bands, and what each band may and may not do.
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
  - ../add-a-node.md
---

**Status:** accepted (2026-04-16).

## Context

Data steps, splits, batching and the model were mixed in one flow. It was
unclear what could be cached, which steps saw the train or validation
phase, and where data leakage could happen.

## Decision

Every training graph flows through four bands, in order:

| Band | Nodes | Rule |
| --- | --- | --- |
| 1. Data preparation | Data Input; text, time-series and image steps; vectorizers | Stateless and deterministic; the same on every split; cacheable |
| 2. Partitioning | Data Split, Time Series Split | Assigns rows to train, validation and test from a seed; cacheable |
| 3. Iteration | Data Loader, augmentation, MixUp, CutMix | Reads the phase (train or validation); runs every epoch; not cacheable |
| 4. Model and optimisation | Layers, losses, optimizers, Output | Forward, loss, backward, update |

In code, a data step declares its band with `IPipelineOperator::GetBand()`
(`PipelineBand::DataPrep`, `Partition` or `PhaseAware`,
`cyxwiz-engine/src/core/node_executors/pipeline_operator.h`).

## Consequences

- The "prepare once, train many" cache boundary is between bands 2 and 3.
- A phase-aware node can never move into band 1.
- A step that learns from data (a scaler, a vocabulary) learns from the
  train rows only; otherwise it leaks the other splits into training.
- Inspection tools read any band without being part of the flow.

See [The data pipeline](../../concepts/data-pipeline.md).
