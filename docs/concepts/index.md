---
title: How CyxWiz works
description: Explanations of how the Engine turns graphs and data into trained models, where it computes, and how its parts connect.
type: landing
area: general
audience: user
status: stable
since: 1.0.7
owner: general
last_reviewed: 2026-11-25
related:
  - ecosystem.md
  - how-graphs-run.md
  - data-pipeline.md
  - compute-back-ends.md
  - ../internals/index.md
---

These pages explain how CyxWiz works and why it works that way. They hold no
steps: each links to the guides that do the task. Read them when a result
surprises you, before a long run, or before you extend the Engine.

## The big picture

```mermaid
flowchart LR
    D[Data Input] --> P[Data steps] --> S[Data Split] --> L[Data Loader]
    L --> M[Model layers] --> O[Loss and optimizer]
    O --> R[Trained model]
    R --> T[Test] --> E[Export]
```

You build a graph, the Engine checks it (compile), prepares the data, and
trains the model on the device you chose. The trained model is tested on
rows it never saw and saved as a model file.

## Pages in this section

How the Engine runs your work:

- [The CyxWiz ecosystem](ecosystem.md): the Engine, the backend library,
  plugins, the Server Node, the central server and CyxCloud, and how they
  connect.
- [How graphs run](how-graphs-run.md): compile, the three levels of
  findings, run modes, and what runs a graph's data steps and its training.
- [The data pipeline](data-pipeline.md): the four bands every training graph
  goes through, where results are cached, and how leakage happens.
- [Seeds and reproducibility](seeds-and-reproducibility.md): which seed
  decides which random draw, and how exactly a run repeats.

Data and memory:

- [Data lifecycle](data-lifecycle.md): who registers a dataset, who
  releases it, and how memory is shared.
- [Loaded data](loaded-data.md): what the Engine holds now and how to see
  it.
- [Data bigger than memory](data-bigger-than-memory.md): the data budget
  and tables kept on disk.
- [Disk caches](disk-caches.md): the files the Engine keeps so data loads
  fast next time.
- [Steps learn from the Train rows](steps-learn-from-train-rows.md): how
  fitted steps avoid seeing validation and test rows.

Devices and numbers:

- [Compute back ends](compute-back-ends.md): ArrayFire on the CPU, CUDA,
  OpenCL and oneAPI, GPU packs, and the CPU fallback.
- [What runs where on a GPU](gpu-providers.md): CyxWiz's own GPU providers
  beside ArrayFire.
- [Numerical accuracy](numerical-accuracy.md): float32 and float64, device
  differences, and how results are checked against PyTorch.

Models:

- [Model formats](model-formats.md): the `.cyxmodel` package, checkpoints,
  weight files, export formats and code export.

## For contributors

The same topics at code level, with the class names and threading rules,
are in [Engine internals](../internals/index.md).

Terms used on these pages are defined in the [glossary](../glossary.md).
