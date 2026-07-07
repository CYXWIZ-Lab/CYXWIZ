# tofix42 Retrieval and Answer Algorithm

## Overview

The first assistant is a retrieval-augmented local agent:

```text
workspace files + runtime records
        -> incremental index
        -> query analysis
        -> hybrid retrieval
        -> context builder
        -> local answer model
        -> cited answer or draft action
```

The model should not be treated as the source of CyxWiz truth. It is the
language layer over retrieved evidence.

## Indexing Pipeline

### 1. Discover Sources

Scan configured roots:

- source roots
- docs roots
- graph examples
- usage guides
- current workspace graph
- persisted debug runs
- latest training trace
- selected logs

Each source is assigned a type and trust rank.

### 2. Parse and Chunk

Chunk by structure where possible:

- C++ headers/source: class, function, enum, struct, and nearby comments
- markdown: heading section
- JSON graph files: graph summary, node chunks, link chunks, parameter chunks
- debug traces: one trace record per chunk plus run summary chunk
- training traces: run summary, terminal event, warning list, materialization
  events, checkpoint events, and recent stage windows
- logs: severity/timestamp windows

Avoid fixed-size chunking as the primary method for source and graph files. Use
fixed-size overflow only when a single structured section is too large.

### 3. Enrich Metadata

Add tags and references:

- subsystem
- node type
- file path
- symbol
- line range
- graph name
- run id
- graph hash
- node id/name/type
- phase
- role
- backend
- status
- terminal reason
- warning/error code when available

### 4. Embed and Store

Store:

- raw chunk text
- normalized searchable text
- metadata
- embedding vector
- content hash

The vector index should live inside the local workspace or a user-selected local
cache. It must be rebuildable from source files and traces.

## Retrieval Pipeline

### 1. Query Classification

Classify the request into one or more intents:

- source explanation
- graph property explanation
- graph failure diagnosis
- training run explanation
- debugger trace explanation
- graph draft generation
- Python scripting help
- source ownership lookup
- unsupported capability check

Intent controls the preferred sources and answer contract.

### 2. Evidence Routing

Route retrieval by intent:

- source questions: source chunks first, docs second
- graph questions: active graph and graph examples first, node definitions next
- debugger questions: selected trace/run first, source/docs next
- training questions: latest trace and checkpoint/runtime records first
- graph generation: supported node/property source plus validated examples
- scripting questions: usage docs, examples, and source APIs

### 3. Hybrid Retrieval

Use both:

- lexical search for exact symbols, node names, error text, and paths
- vector search for conceptual similarity

Exact matches should boost:

- file paths
- class/function names
- node types
- graph node names
- warning/error text
- diagnostic codes
- run ids
- `terminal_reason`
- backend names such as `cuda`, `cpu`, or `ArrayFire`

### 4. Reranking

Rerank using:

- trust rank
- recency
- active context match
- exact symbol/path match
- graph/run/node correlation
- source-vs-doc conflict rules
- chunk diversity

The final context should include enough diversity to avoid one-source tunnel
vision. For example, a materialization failure answer should include the failing
trace/log, active graph node, relevant operator source, and relevant docs if
available.

## Context Builder

The context builder creates a compact evidence packet:

- user question
- active Studio state
- selected graph node and params
- selected trace or latest run summary
- retrieved source snippets
- retrieved docs/examples
- conflicts or missing evidence notes
- required answer format

Context must preserve citations:

- file path and line range
- graph path and node id/name
- run id and trace phase/role
- log timestamp and source

## Answer Synthesis

The local model receives a strict instruction:

- answer from evidence
- cite evidence
- separate fact from inference
- do not invent unsupported engine behavior
- state what could not be verified

For runtime failures, answer:

- what happened
- where it happened
- why the evidence suggests that cause
- what the user can change in the graph
- what engine source may own the issue
- what follow-up trace/preflight action would improve confidence

For graph generation, answer:

- graph draft
- supported nodes used
- key properties
- assumptions
- preflight risks
- why this design fits the task

## Conflict Handling

Use this precedence:

1. active runtime context and structured traces
2. current source code
3. current active graph
4. current docs and usage guides
5. older tofix/done docs
6. raw logs
7. future fine-tuning memory

If docs and source conflict, cite both and say the source appears to be the
current implementation truth.

## Hallucination Guards

The assistant should refuse or qualify:

- unsupported node types
- unsupported node properties
- unsupported backend capabilities
- missing debugger producers
- complete graph tracing if only partial tracing exists
- pinned-memory behavior not proven by runtime diagnostics
- fine-tuning claims without a collected dataset

Phrase examples:

- `I found design intent for this, but not current source support.`
- `The trace proves training stopped because ..., but it does not prove the root
  cause.`
- `This graph draft should be run through compiler/preflight before training.`

## Evaluation

Track quality with local tests:

- retrieval recall for known source symbols
- answer citation coverage
- graph draft preflight pass rate
- unsupported feature refusal rate
- debugger explanation accuracy on saved runs
- latency by corpus size
- index rebuild time

Collect failed answers as candidates for future supervised data, but do not
fine-tune until reviewed examples exist.

## Algorithm Acceptance Criteria

- Queries retrieve source, docs, graph, and trace evidence by intent.
- Answers cite specific evidence.
- Active graph/run context outranks generic examples.
- Graph drafts use supported nodes/properties from retrieved truth.
- Runtime answers distinguish structured trace facts from inference.
- The index can be rebuilt and incrementally updated locally.
