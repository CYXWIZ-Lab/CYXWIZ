# tofix42 Agent Handoff

## Role for the Side Agent

You are helping design `tofix42`: the local CyxWiz source-aware assistant.
Do not modify engine code unless explicitly asked. Start by analyzing and
documenting the architecture, data requirements, model/runtime options, and
implementation slices.

## Primary Goal

Design an assistant that can run locally and answer questions using CyxWiz
source code, docs, graph files, training traces, debugger traces, logs, and
usage documentation.

The assistant should eventually help users and engineers:

- explain graph failures
- explain node properties
- explain training curves
- inspect materialization/debugger traces
- identify likely source files behind errors
- generate valid CyxGraph drafts
- generate Python scripting snippets
- suggest graph/model improvements
- identify missing engine capabilities

## Core Product Vision

CyxWiz should eventually build its own assistant:

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

This is the "mother gives birth to child, child later helps mother" vision.

## Important Boundary

Do not start with fine-tuning.

The first implementation should be:

```text
local small model + retrieval over CyxWiz source/docs/graphs/traces
```

Fine-tuning is a later phase after we collect enough CyxWiz-specific examples.
The source code changes too often, so retrieval must remain the truth layer.

## Files to Read First

- `docs/Data Studio/tofix42/tofix42.md`
- `docs/Data Studio/ux.md`
- `docs/Data Studio/tofix32.md`
- `docs/Data Studio/tofix39.md`
- `docs/Data Studio/tofix41.md`
- CyxGraph examples under `examples/cyxgraph`
- Training/debugger trace code under `cyxwiz-engine/src/core`
- Studio Debugger UI under `cyxwiz-engine/src/gui/panels/studio_debugger_panel.*`

## Expected Outputs from Side Agent

Produce documents under `docs/Data Studio/tofix42/`, for example:

- `dataset.md`: what data is needed for RAG/future fine-tuning
- `model.md`: local model/runtime options and constraints
- `algorithm.md`: retrieval/indexing/context-building approach
- `architecture.md`: engine/studio integration design
- `milestones.md`: implementation phases
- `risks.md`: hallucination, stale source, security, performance risks

## Engine Truth Requirements

The assistant must not hallucinate unsupported CyxWiz features.

Answers should cite:

- source files
- docs
- graph files
- selected node properties
- current training trace
- materialization breakdown
- debugger trace
- logs/checkpoints

Separate:

- fact from source
- inference from logs/traces
- recommendation
- unsupported/missing engine capability

## First Milestone Scope

Read-only assistant only:

- index docs and selected source folders
- index graph examples
- read latest training trace/debugger trace
- answer questions in Studio Debugger
- cite retrieved files/traces
- no graph editing
- no source editing
- no fine-tuning

## Do Not Mix With Current Main Work

The main thread may continue building/testing the engine and sentiment graph.
This side agent should stay focused on `tofix42` design and documentation until
explicitly asked to implement.
