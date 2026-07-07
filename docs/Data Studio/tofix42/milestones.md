# tofix42 Milestones

## Milestone 0: Design and Scope

Status: current documentation milestone.

Deliverables:

- define corpus and dataset rules
- define local model/runtime constraints
- define retrieval and context-building algorithm
- define Studio/engine architecture
- define implementation milestones
- define risk controls

Exit criteria:

- design docs exist under `docs/Data Studio/tofix42/`
- first implementation is clearly RAG-first, not fine-tuning-first
- engine code remains untouched unless explicitly requested

## Milestone 1: Read-Only Local Assistant Prototype

Goal: answer questions from local docs/source/graphs.

Scope:

- index `docs/`, `docs/usage/`, and selected source folders
- index `examples/cyxgraph/`
- local embedding model
- simple local vector/lexical index
- local answer model runtime
- command or internal test harness for assistant queries
- cited answers with file paths

Non-goals:

- no Studio UI polish
- no graph edits
- no source edits
- no fine-tuning
- no runtime trace diagnosis beyond loaded files

Exit criteria:

- asks "what does this node/property/source file do?" and gets cited answers
- asks for graph examples and retrieves relevant `.cyxgraph` files
- stale index detects changed files by hash or timestamp

## Milestone 2: Studio Debugger Explanation Slice

Goal: explain selected debugger warnings/errors from structured context.

Scope:

- integrate read-only assistant entry point in Studio Debugger
- pass selected `DebugTraceRecord` and run id
- retrieve related source/docs/graph chunks
- load `DebugRunStore` run context when available
- answer with evidence/inference/recommendation separation

Initial questions:

- why did this warning/error happen?
- what node or phase owns it?
- what source file is likely responsible?
- what can the user change in the graph?
- what evidence is missing?

Exit criteria:

- selected trace explanation cites trace fields and source/docs
- unsupported trace coverage is called out explicitly
- no debugger explanation depends only on parsing rendered UI text

## Milestone 3: Training Dashboard and Trace Summary

Goal: explain training runs and stop reasons.

Scope:

- ingest latest `TrainingTraceSummary`
- include `terminal_reason`, recent events, warnings, materialization events,
  and checkpoint records
- support "Why did training stop?", "Why did validation plateau?", and "What
  warnings mattered?"

Exit criteria:

- early stop vs completed vs cancelled vs failed is explained from trace data
- materialization failures are linked to graph node/source where possible
- backend warnings are labeled as warnings unless structured diagnostics prove
  stronger facts

## Milestone 4: Graph-Aware Help

Goal: help users understand and improve graphs without mutating them.

Scope:

- selected node/property explanation
- valid property/source lookup
- graph path explanation
- graph improvement suggestions
- graph draft generation from examples and node truth
- optional preflight call after draft generation, still user-approved

Exit criteria:

- generated graph drafts use known node types and properties
- answer explains assumptions and preflight risks
- no graph is applied without explicit user approval

## Milestone 5: Structured Diagnostics Integration

Goal: make assistant answers stronger by consuming stable diagnostics.

Depends on:

- tofix26-style error codes or equivalent structured diagnostics
- tofix32 debugger diagnostic routing

Scope:

- index diagnostic records
- route exact codes to source/docs
- explain diagnostic cause, impact, and suggested fix
- correlate diagnostic with graph node, run id, task id, and source component

Exit criteria:

- diagnostic-code answers no longer rely on raw log matching
- assistant can explain examples like materializer failures and CPU/GPU fallback
  warnings from structured fields

## Milestone 6: Evaluation Harness

Goal: measure answer quality before considering fine-tuning.

Scope:

- curated retrieval tests for known symbols/files
- debugger explanation fixtures
- graph-generation fixtures
- unsupported capability refusal tests
- latency and memory measurements
- bad-answer capture workflow

Exit criteria:

- quality failures are reproducible
- reviewed examples are stored as future supervised data candidates
- fine-tuning decision is based on evidence, not assumption

## Milestone 7: Optional Fine-Tuning Data Collection

Goal: collect CyxWiz-specific examples after RAG is working.

Scope:

- reviewed source-code QA
- reviewed graph-generation examples
- reviewed debugger explanation examples
- reviewed scripting examples
- refusal examples for unsupported features

Non-goals:

- no training on unreviewed assistant outputs
- no replacement of retrieval with model memory

Exit criteria:

- enough validated examples exist to justify an experiment
- RAG remains mandatory after any fine-tune

## Milestone 8: Optional Fine-Tuning Experiment

Goal: test whether a CyxWiz-specific fine-tune improves behavior.

Scope:

- train or adapt a small local model on reviewed examples
- compare against base model plus retrieval
- evaluate graph draft validity, citation quality, and unsupported-feature
  refusal
- measure maintenance cost when source changes

Exit criteria:

- fine-tune only ships if it improves measured quality without weakening source
  grounding
- retrieval remains the truth layer
