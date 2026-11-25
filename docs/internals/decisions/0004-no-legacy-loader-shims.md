---
title: "Decision 0004: No legacy loader shims"
description: Why old graphs are rebuilt on the current node contracts instead of being translated by the loader.
type: explanation
area: internals
audience: contributor
status: stable
since: 1.0.7
owner: internals
last_reviewed: 2026-11-25
related:
  - README.md
  - ../add-a-node.md
---

**Status:** accepted (2026-09-28).

## Context

Node contracts change while the Engine is in alpha: nodes are merged (the
separate plot nodes became one Plot node), replaced (Dataset Input became
Data Input) or given new pins. Translating every old form inside the
loader grows a second, untested implementation of each node and hides
which contract a graph really uses.

## Decision

- Graphs, patterns and examples are rebuilt on the current contracts.
- The loader does not translate retired nodes. It refuses them with a
  message that names the replacement, for example "node '...' is a Dataset
  Input node, which the Data Input node replaced: remove it from the file
  or rebuild the graph with a Data Input node"
  (`cyxwiz-engine/src/core/graph_document.cpp`).
- `NodeType` values are append only, so a valid saved node never changes
  meaning.

## Consequences

- An old graph may need rebuilding after an update; the error says what to
  rebuild.
- A few parameter-name aliases and the marked data-boundary pin contract for
  unmigrated graphs predate this rule and remain. New code adds none.
