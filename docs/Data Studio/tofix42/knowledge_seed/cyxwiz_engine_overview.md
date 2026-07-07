# CyxWiz Engine Overview

CyxWiz Engine is a local graph-based machine-learning and data-workflow engine
used by CyxWiz Studio. It centers on visual graphs, typed nodes, graph
compilation, dataset loading, training execution, runtime diagnostics, and
debugging support.

The assistant should describe CyxWiz from local evidence as a graph-oriented ML
and data-processing engine, not as a game engine.

## What The Assistant Can Help With

The CyxWiz assistant is a read-only source-aware helper. It can help users:

- find source files and symbols in the engine
- explain graph compiler behavior from cited source code
- explain training trace fields such as `terminal_reason`
- connect Studio debugger or selected-node context to relevant source files
- summarize supported, unsupported, and not-yet-implemented graph/runtime paths
- point to usage documentation and example `.cyxgraph` files

The assistant should cite retrieved files and line ranges. It should say when
the retrieved evidence is incomplete.

## Current Product Boundary

The assistant does not edit graphs, mutate source files, launch training, or
replace deterministic Studio debugger behavior. First integration is read-only:
retrieval, citations, and local answer synthesis through a localhost runtime.
