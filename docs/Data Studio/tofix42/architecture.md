# tofix42 Architecture

## Role

The tofix42 assistant is an optional local explanation layer for CyxWiz Studio
and engine workflows. It should help users and engineers understand source code,
graphs, training behavior, debugger traces, logs, and usage docs.

The first architecture is read-only:

- no source edits
- no graph edits
- no training launches
- no fine-tuning
- no network dependency

## High-Level Components

```text
Studio UI surfaces
        |
Assistant Orchestrator
        |
Context Builder
        |
Retrieval Service ---- Local Index Store
        |                    |
        |              Indexer / Watcher
        |
Local Model Runtime
        |
Cited answer / draft recommendation
```

## Components

### Assistant Orchestrator

Responsibilities:

- receive requests from Studio surfaces
- gather active context
- classify intent
- call retrieval
- call local model runtime
- return cited answer blocks
- enforce read-only action boundaries

The orchestrator should be independent from ImGui so it can be tested without
the GUI.

### Indexer

Responsibilities:

- scan configured local roots
- parse source/docs/graphs/traces/logs
- chunk content
- compute hashes
- update embeddings
- write local index metadata
- support rebuild from Studio

The indexer should run incrementally and be cancelable. Indexing must not block
training.

### Local Index Store

Stores:

- chunk text
- metadata
- embeddings
- content hashes
- source timestamps

The first implementation can use a simple local vector store plus lexical
lookup tables. It should be replaceable later.

### Retrieval Service

Responsibilities:

- hybrid lexical/vector retrieval
- intent-aware source routing
- reranking
- conflict handling
- citation packet construction

The retrieval service should expose deterministic tests over small fixtures.

### Context Builder

Merges retrieved context with live Studio context:

- active graph
- selected node
- selected trace
- compiler/preflight output
- latest training trace
- latest materialization events
- latest warnings/errors
- current run id and graph hash

It should also attach a source precedence note so the model knows which evidence
is strongest.

### Local Model Runtime

Responsibilities:

- load configured answer model
- run local generation
- enforce prompt contract
- stream answer text if supported
- report model unavailable states cleanly

The runtime should be behind a narrow interface, with backends such as
llama.cpp-style inference, ONNX Runtime, or future CyxWiz-native inference.

## Studio Entry Points

### Studio Debugger

Initial entry point:

- `Explain selected trace`
- `Explain selected warning`
- `Why did this run stop?`
- `What source owns this failure?`

Context:

- `DebugTraceRecord`
- `DebugRunStoreRecord`
- trace issues
- node id/name/type
- graph hash
- selected lens
- latest training trace when relevant

### Training Dashboard

Entry points:

- `Explain run summary`
- `Explain early stop`
- `Explain validation plateau`
- `Explain backend warning`
- `Explain checkpoint choice`

Context:

- `TrainingTraceSummary`
- `terminal_reason`
- recent events
- materialization events
- warnings
- checkpoint events

### Node Properties Panel

Entry points:

- `Explain this property`
- `What values are valid?`
- `Why is this property not taking effect?`
- `What does this node feed?`

Context:

- selected node type
- selected property
- graph params
- node definition/source
- compiler/preflight output
- graph examples using same node type

### Graph Editor

Entry points:

- `Suggest graph fix`
- `Draft graph for this dataset/task`
- `Explain selected path`
- `Find disconnected or suspicious pins`

Context:

- active graph JSON
- selected nodes/links
- compiler/preflight issues
- graph examples
- supported node/property truth

### Error Dialogs and Logs

Entry points:

- `Explain this error`
- `Show likely source file`
- `What can I try next?`

Context:

- structured diagnostic if available
- raw log text
- task id/name
- node id/name
- current graph
- source search results for exact error text

## Data Flow for a Debugger Question

```text
User clicks "Explain selected trace"
        |
Studio passes selected DebugTraceRecord + run id
        |
Context Builder loads DebugRunStore record and latest training trace
        |
Retrieval finds matching source, docs, graph nodes, and related traces
        |
Model answers with evidence, inference, recommendation, and unknowns
```

## Data Flow for Graph Drafting

```text
User asks for graph draft
        |
Orchestrator classifies graph_generation
        |
Retrieval loads supported nodes/properties and similar graph examples
        |
Model drafts graph JSON or structured graph plan
        |
Optional later phase runs compiler/preflight
        |
User approves before graph mutation
```

The first milestone stops at a draft. Applying the graph is later work.

## Storage and Privacy

Default behavior:

- local-only index
- no network calls
- no private dataset rows indexed by default
- no binary weights indexed
- redaction hooks for support bundles and logs
- user-visible rebuild/delete index action

If future network models are ever allowed, that should be a separate explicit
mode. It is not part of tofix42 first implementation.

## Integration Boundaries

The assistant should depend on stable contracts, not raw UI state:

- graph snapshots
- compiler/preflight results
- `DebugTraceRecord`
- `DebugRunStore`
- `TrainingTraceCollector`
- structured diagnostics when available
- graph/node definition registries

Avoid parsing ImGui-rendered text as assistant input when structured records are
available.

## Architecture Acceptance Criteria

- Assistant is optional and local.
- Engine training does not depend on assistant startup or model loading.
- First UI integration can explain a selected Studio Debugger warning/error.
- Retrieval uses source/docs/graphs/traces/logs with citations.
- The assistant cannot mutate source or graphs in the first milestone.
- Runtime/model backend can be replaced without rewriting retrieval/indexing.
