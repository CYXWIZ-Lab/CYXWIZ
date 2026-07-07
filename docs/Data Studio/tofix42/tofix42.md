# tofix42 - Local CyxWiz Source-Aware LLM Agent

## Status

Open.

## Idea

Add a local CyxWiz-aware assistant inside the engine/studio that understands the
project source code, documentation, graph format, training logs, debugger traces,
and usage docs.

Long-term vision: CyxWiz should be able to build its own assistant. The engine
is the mother system that trains/builds models; the assistant is the child model
born from CyxWiz source, docs, graphs, traces, and workflows. Over time, that
assistant should help take care of the mother system by explaining failures,
generating valid graphs, guiding users, finding missing engine pieces, and
helping engineers improve CyxWiz itself.

The intended loop is:

```text
CyxWiz Engine builds models
        ↓
CyxWiz builds a local CyxWiz assistant model
        ↓
Assistant understands CyxWiz source/docs/graphs/traces
        ↓
Assistant helps users debug, explain, generate, and improve workflows
        ↓
Assistant helps engineers improve CyxWiz itself
```

The assistant should help engineers and users ask questions such as:

- Why did this graph fail?
- What does this node property mean?
- How do I build a sentiment-analysis graph?
- Generate a CyxGraph for this dataset/task.
- Explain this training curve.
- Suggest a Python scripting snippet.
- Point me to the source code responsible for this error.
- Explain whether a warning is GPU, memory, materialization, data-loader, or
  model-architecture related.

## Recommendation

Do not start with fine-tuning.

Start with a local retrieval-augmented assistant:

- Index CyxWiz source code.
- Index `docs/`, `examples/`, tofix/done tickets, usage docs, and graph examples.
- Index runtime traces and training/debugger logs.
- Retrieve the most relevant snippets at question time.
- Use a small CPU-capable model to answer from retrieved context.

Fine-tuning can come later, after we have:

- stable instruction examples
- source/documentation QA pairs
- graph-generation examples
- debugger/error explanation examples
- user feedback on bad answers

This avoids training a model that becomes stale every time the source code
changes.

## Why This Fits CyxWiz

CyxWiz is becoming a graph-based ML engine with:

- visual graph authoring
- training dashboard
- Studio Debugger
- materialization traces
- CPU/GPU runtime paths
- Python scripting
- graph export/import
- model packaging

That means users need help understanding both ML concepts and engine-specific
behavior. A source-aware local agent can become the explanation layer for the
whole system.

## Proposed Architecture

### 1. Knowledge Index

Build an index from:

- `cyxwiz-engine/src`
- `cyxwiz-backend/src`
- graph node definitions
- compiler/preflight code
- materializer/operator implementations
- training/debugger code
- `docs/`
- `examples/`
- `*.cyxgraph`
- recent task logs and training traces

Each indexed chunk should store:

- file path
- symbol/function/class when available
- line range
- content hash
- document type
- last modified timestamp
- tags such as `gpu`, `materializer`, `dataloader`, `debugger`, `graph`,
  `optimizer`, `loss`, `python`, `export`

### 2. Retrieval Layer

Given a question, retrieve relevant context from:

- source code
- docs
- graph files
- active graph
- current training trace
- current logs
- selected node properties
- Studio Debugger records

The answer must cite file paths or UI/runtime sources when possible.

### 3. Local Model Runtime

Use a small local model first.

Requirements:

- CPU-capable
- quantized runtime support
- low memory footprint
- works offline
- can answer from retrieved context

Possible runtime paths:

- llama.cpp-style backend
- ONNX Runtime text-generation backend
- future CyxWiz-native inference wrapper

The engine should treat this as an optional assistant module, not a hard
dependency for training.

### 4. Tool-Constrained Actions

The assistant should not blindly edit files or graphs.

It should produce proposed actions first:

- explain
- recommend
- generate graph draft
- generate Python snippet
- identify likely source file
- propose debugger follow-up
- propose graph change

Later, approved actions can be applied through existing CyxWiz systems.

### 5. Studio Integration

Add assistant entry points in:

- Studio Debugger
- Training Dashboard
- Node Properties panel
- Graph editor
- Error dialogs
- Usage/help panel

Examples:

- `Explain this warning`
- `Explain this materialization step`
- `Why did training early stop?`
- `Suggest graph changes for better validation accuracy`
- `Generate a graph for tabular binary classification`
- `Show source code responsible for this error`

## Required Engine Capabilities

### Source and Doc Indexer

- Incremental indexing.
- Detect changed files by hash/timestamp.
- Chunk source by symbol where possible.
- Chunk docs by heading.
- Chunk graph files by node/link/parameter sections.

### Embeddings

- Add an embedding model path or backend.
- Store embeddings in a local vector index.
- Keep index local to the workspace.
- Allow rebuild from Studio.

### Context Builder

- Merge retrieved snippets with:
  - active graph
  - selected node
  - latest training trace
  - latest materialization events
  - latest warnings/errors
  - compiler/preflight output

### Answer Contract

Answers should:

- say when they are using retrieved source context
- cite relevant files or runtime traces
- separate fact from recommendation
- avoid pretending unsupported engine features exist
- prefer engine truth over generic ML advice

### Graph Generation Contract

When generating a graph, the assistant must:

- use only supported CyxWiz nodes
- use valid node properties
- respect compiler/preflight constraints
- explain why the graph design fits the task
- optionally run preflight before recommending training

### Debugger Contract

For runtime issues, the assistant should inspect:

- current training trace
- materialization breakdown
- warnings
- checkpoint metadata
- terminal reason
- active graph
- relevant source files

Then it should answer:

- what happened
- where it happened
- why it likely happened
- what user can change in the graph
- what engine code may need fixing

## Fine-Tuning Later

Fine-tuning is a later phase, not the first implementation.

Potential fine-tuning data:

- CyxWiz graph examples
- source-code Q&A
- debugger explanations
- error-to-fix examples
- node property explanations
- training-curve diagnosis examples
- Python scripting examples

Fine-tuning should not replace retrieval. The source changes too often, so even
a fine-tuned model still needs live retrieval.

## Risks

- Hallucinated engine capabilities.
- Stale source knowledge.
- Slow CPU inference.
- Large memory usage if model/index are too big.
- Security risk if assistant can modify files without approval.
- Poor answers if graph schema and node property truth are not indexed well.

## Acceptance Criteria

- A local assistant can answer questions using CyxWiz source/docs context.
- Answers cite the source file, doc, graph, trace, or log used.
- Studio Debugger can ask the assistant to explain a selected warning/error.
- Training Dashboard can ask the assistant to explain a run summary.
- Node Properties can ask the assistant to explain a selected property.
- The assistant can draft a CyxGraph using only supported nodes/properties.
- The assistant can generate Python scripting snippets with clear assumptions.
- No file or graph mutation happens without explicit user approval.

## First Milestone

Build a read-only assistant:

- index docs and selected source folders
- index graph examples
- read latest training trace
- answer questions in Studio Debugger
- cite retrieved files/traces
- no graph editing
- no source editing
- no fine-tuning

## Second Milestone

Add graph-aware help:

- selected node explanation
- property truth explanation
- graph improvement suggestions
- graph draft generation
- preflight-aware validation

## Third Milestone

Evaluate local models and optional fine-tuning:

- compare CPU latency
- compare answer quality
- collect failed answers
- build CyxWiz-specific instruction examples
- decide whether fine-tuning is worth the maintenance cost
