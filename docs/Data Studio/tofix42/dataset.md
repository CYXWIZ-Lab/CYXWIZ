# tofix42 Dataset Design

## Purpose

The first assistant dataset is not a fine-tuning corpus. It is a local retrieval
corpus built from current CyxWiz truth sources: source code, docs, graph files,
debugger traces, training traces, logs, and usage examples.

Fine-tuning examples can be collected later from reviewed assistant sessions,
but the first implementation must answer from live indexed context so it does
not become stale when CyxWiz changes.

## First-Phase Retrieval Corpus

Index these sources first:

- `cyxwiz-engine/src`
- `cyxwiz-backend/src`
- `docs/`
- `docs/usage/`
- `docs/Data Studio/`
- `examples/cyxgraph/`
- exported `.cyxmodel` metadata where it is text or JSON
- latest persisted training trace from `TrainingTraceCollector`
- latest debugger runs from `DebugRunStore`
- Studio Debugger records using `cyxwiz.debug.node_trace.v1`
- current graph, selected node, compiler/preflight output, and active logs when
  invoked from Studio

Do not index binary model weights. For packaged models, index only text
metadata such as `manifest.json`, `config.json`, `history.json`, and
`graph.cyxgraph`.

## Corpus Types

### Source Code

Source chunks should preserve:

- path
- language
- class/function/symbol if available
- line range
- include/import context where useful
- subsystem tag such as `compiler`, `materializer`, `debugger`, `training`,
  `backend`, `ui`, `export`, `data`, `graph`, or `runtime`

Important initial source areas:

- graph compiler and preflight code
- node definitions and node property handling
- materializer and operator implementations
- `DebugTraceRecord`, `DebugRunStore`, and debugger trace producers
- `TrainingTraceCollector`, `CrashRunRecorder`, and training executor
- Studio Debugger and Training Dashboard panels
- backend layer implementations and CPU/GPU fallback paths

### Documentation

Docs should be chunked by heading and retain:

- heading path
- document path
- ticket/status when present
- related tofix/done identifier
- acceptance criteria
- non-goals
- current limitations

The retrieval layer should treat recent tofix documents as design intent, not
necessarily implemented behavior. Engine source and runtime traces outrank docs
when they conflict.

### Graph Files

Graph chunks should preserve:

- graph path
- graph name and description
- node id, type, name, and params
- link source/target pins
- editable graph parameters
- dataset path and label/text/feature columns where present
- task domain tags such as `text`, `audio`, `vision`, `timeseries`, `ner`, or
  `tabular`

The graph index should support two retrieval modes:

- whole-graph examples for generation and workflow questions
- node-level chunks for property and routing questions

### Training Traces

Training trace chunks should preserve:

- run id
- status
- latest stage
- epoch and batch position
- loss, accuracy, validation loss, validation accuracy
- materialization events
- warnings
- checkpoint events
- terminal status and `terminal_reason`
- task progress records with node id/name when available

These are high-value evidence for questions such as why training stopped,
whether early stopping occurred, whether materialization failed, or whether a
warning affected training.

### Debugger Traces

Debugger trace chunks should preserve:

- run id
- graph hash
- trace schema, especially `cyxwiz.debug.node_trace.v1`
- node id/name/type
- phase and role
- input/output shapes
- dtype and backend payload fields
- status
- issues
- structured payload
- Studio events and recommendations

The assistant should prefer structured trace fields over raw log text.

### Logs

Logs are useful, but lower-trust than structured records. Index logs with:

- timestamp
- source/component when available
- severity
- raw message
- nearby correlation ids such as run id, task id, graph hash, node name, and
  dataset name

Log answers must label log-derived claims as log evidence unless confirmed by a
structured trace or source file.

## Chunk Metadata

Every chunk should include:

- `id`
- `source_type`
- `path_or_source`
- `line_start`
- `line_end`
- `symbol`
- `heading`
- `content_hash`
- `last_modified`
- `tags`
- `trust_rank`
- `workspace_root`

Recommended trust rank:

1. active runtime context and structured traces
2. current source code
3. current graph file
4. current docs and usage guides
5. older tofix/done docs
6. raw logs
7. future curated QA examples

## Future Fine-Tuning Corpus

Fine-tuning should wait until the assistant has produced enough reviewed local
sessions. Candidate examples:

- source-code Q&A with cited file paths
- node property explanations grounded in node definitions
- graph-generation prompts paired with validated `.cyxgraph` outputs
- debugger explanations paired with structured trace evidence
- training-curve diagnosis examples
- error-to-fix examples with stable diagnostic codes
- Python scripting examples that run against current CyxWiz APIs
- refusal examples where the assistant says a requested feature is unsupported

Every future training example should include source citations or validation
evidence. Do not train on unreviewed assistant guesses.

## Dataset Acceptance Criteria

- The corpus can be rebuilt locally from the workspace.
- Changed files are detected by hash or timestamp.
- Source answers cite file paths and line ranges when possible.
- Runtime answers cite traces, run ids, node names, or task records.
- Graph answers use current node types and properties from indexed graphs/source.
- Logs are never treated as stronger evidence than structured records.
- No binary model weights or private dataset rows are indexed by default.
