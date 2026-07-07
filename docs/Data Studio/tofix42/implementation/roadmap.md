# CyxWiz Assistant Roadmap

## Purpose

Control the direction of the CyxWiz assistant work so implementation does not
drift into disconnected experiments.

This roadmap is the coordination document. Other design docs define details;
this document defines order, gates, and what not to do yet.

## Current Position

The project has a working local RAG proof, a generated knowledge pack, and an
experimental assistant plugin integrated into the Release engine.

## Phase Checkpoints

- [x] Phase A: planning and contracts are defined
- [x] Phase B: knowledge pack builder/loader exists and is used by the plugin
- [x] Phase C: backend retrieval/runtime contract is implemented
- [x] Phase D: Assistant panel is wired and working
- [x] Phase E: first read-only context actions exist
- [x] Phase F: first async Command Window slash-command path exists
- [ ] Phase G: fine-tuning experiment is actually run and validated

Completed proof work:

- local retrieval prototype
- answer packet and prompt adapter
- localhost model/proxy validation path
- probe/evaluation harness
- question-quality guidance
- plugin deployment and boundary design
- knowledge-pack design
- Assistant panel design
- backend runtime contract design

Initial code scaffold:

- experimental assistant panel plugin scaffold exists
- it loads the panel contract shape
- it performs knowledge-pack retrieval inside the plugin
- it can call a configured localhost model runtime
- it receives a first engine context snapshot with project, graph, and selected
  node identity
- it routes explicit Command Window slash commands to the assistant provider
- assistant Command Window work runs in the background so the UI remains
  responsive
- assistant output can be copied from the Command Window

## Product Direction

The assistant should become:

```text
optional CyxWiz plugin/module
  -> Assistant panel first
  -> retrieval-backed answers
  -> citations
  -> debugger/training context later
  -> Command Window slash commands later
```

It should not become:

- a hard engine-core dependency
- a general chatbot first
- a graph/source mutation tool first
- a replacement for deterministic debugger/training behavior

## Controlling Principles

1. Retrieval is the truth layer.
2. The model is only the answer writer.
3. The assistant must stay read-only in the first product version.
4. The engine must work when the assistant is absent.
5. UI integration must use explicit boundaries.
6. Fine-tuning is optional and later.
7. Broad help questions require curated help/capability docs, not model guessing.

## Phase Order

### Phase A: Stabilize Planning and Contracts

Status: mostly done.

Required docs:

- [deployment_model.md](</D:/Dev/CyxWiz_Claude/docs/Data Studio/tofix42/implementation/deployment_model.md>)
- [plugin_command_window_integration.md](</D:/Dev/CyxWiz_Claude/docs/Data Studio/tofix42/implementation/plugin_command_window_integration.md>)
- [plugin_boundary.md](</D:/Dev/CyxWiz_Claude/docs/Data Studio/tofix42/implementation/plugin_boundary.md>)
- [command_window_integration.md](</D:/Dev/CyxWiz_Claude/docs/Data Studio/tofix42/implementation/command_window_integration.md>)
- [knowledge_pack.md](</D:/Dev/CyxWiz_Claude/docs/Data Studio/tofix42/implementation/knowledge_pack.md>)
- [assistant_panel.md](</D:/Dev/CyxWiz_Claude/docs/Data Studio/tofix42/implementation/assistant_panel.md>)
- [backend_runtime_contract.md](</D:/Dev/CyxWiz_Claude/docs/Data Studio/tofix42/implementation/backend_runtime_contract.md>)
- [fine_tuning_position.md](</D:/Dev/CyxWiz_Claude/docs/Data Studio/tofix42/implementation/fine_tuning_position.md>)

Exit criteria:

- docs agree on one architecture
- first implementation target is clear
- fine-tuning is explicitly not blocking first integration

### Phase B: Knowledge Pack Builder and Loader

Status: initial implementation exists and is used by the plugin.

Goal:

Turn the current dev-oriented retrieval index into a versioned knowledge-pack
asset.

Required work:

- define manifest schema in code
- write pack builder from current indexed corpus
- write pack validator
- write pack loader
- expose retrieval-only query API against the pack

Exit criteria:

- pack can be built from repo/docs/examples: done
- pack can be loaded without rescanning the repo: done
- retrieval-only queries work from the pack: done
- pack version mismatch can be detected: first manifest validation exists

Current implementation:

- [phase8_knowledge_pack.py](</D:/Dev/CyxWiz_Claude/docs/Data Studio/tofix42/phase8_knowledge_pack.py>)
- [phase8_knowledge_pack.md](</D:/Dev/CyxWiz_Claude/docs/Data Studio/tofix42/phase8_knowledge_pack.md>)
- default pack directory:
  [knowledge_pack](</D:/Dev/CyxWiz_Claude/docs/Data Studio/tofix42/knowledge_pack>)

Do not do yet:

- fine-tuning
- make the pack the shipped production asset until format/versioning is
  finalized
- add embeddings before lexical pack behavior is stable

### Phase C: Backend Runtime Library

Status: retrieval-only backend path and local runtime adapter exist and are
called by the plugin.

Goal:

Create a backend object that implements the documented
`AssistantRequest -> AssistantResponse` contract.

Required work:

- retrieval-only backend path
- localhost runtime adapter path
- parse/validate four answer sections
- stable error codes
- diagnostics suitable for UI

Current implementation:

- C++ backend contract:
  [assistant_backend_contract.h](</D:/Dev/CyxWiz_Claude/plugins/assistant/cyxwiz_assistant/assistant_backend_contract.h>)
- C++ knowledge-pack backend:
  [knowledge_pack_backend.h](</D:/Dev/CyxWiz_Claude/plugins/assistant/cyxwiz_assistant/knowledge_pack_backend.h>)
  [knowledge_pack_backend.cpp](</D:/Dev/CyxWiz_Claude/plugins/assistant/cyxwiz_assistant/knowledge_pack_backend.cpp>)
- Assistant panel now calls the backend.
- Retrieval-only mode searches the knowledge pack and returns citations.
- Full answer mode posts an evidence prompt to
  `http://127.0.0.1:8768/completion`.
- The panel exposes the runtime endpoint, restricted to localhost URLs.
- The panel exposes the knowledge-pack path and can reload the pack.
- Trace and training modes include bounded engine context in retrieval and
  prompt construction.

Exit criteria:

- backend can answer retrieval-only requests from a knowledge pack: done
- backend can call a configured localhost runtime: done
- backend returns citations and parse status: done
- backend can be tested outside the UI: done through the tofix42 scripts
- backend diagnostics suitable for in-app debugging: in progress

### Phase D: Assistant Panel Integration

Status: assistant panel is wired to the backend and first context bridge.

Goal:

Make the assistant panel useful with retrieval-only first, then full answer
mode.

Required work:

- wire panel to backend contract: done
- display retrieval hits and citations: done
- display structured answer sections: done
- show backend status and failure states: done
- keep request stateless per question: done
- show current graph/project/selected-node context: done
- add debugger/training-specific selected context: done, first bounded version

Exit criteria:

- user can open Assistant panel: done
- user can ask `What source file defines DebugTraceRecord?`: done
- panel shows retrieved citation: done
- panel shows structured answer when runtime is configured: done
- panel degrades cleanly when runtime is unavailable: done

### Phase E: Context-Aware Product Actions

Status: first assistant panel context actions exist.

Goal:

Add bounded assistant actions using actual engine context.

Required actions:

- publish graph/project/selected-node context to the Assistant plugin: done
- publish selected debugger trace context to the Assistant plugin: done
- publish training terminal context to the Assistant plugin: done
- explain selected debugger trace as a polished panel action: done
- explain training terminal reason as a polished panel action: done
- find related source for selected graph/node: done
- refine output quality against real local model responses

Exit criteria:

- context objects are bounded and read-only: first version done
- assistant output cites source/docs/traces: source/docs citations done, trace
  citation quality still needs QA
- no graph/source mutation is exposed: done
- missing context states are shown clearly: first version done
- answer quality against real local model responses: in progress

### Phase F: Command Window Slash Commands

Status: first implementation exists.

Goal:

Add explicit assistant routing to the existing Command Window without breaking
current console/Python behavior.

Allowed commands:

- `/ask <question>`
- `/find-source <query>`
- `/explain-trace`
- `/explain-training`

Exit criteria:

- plain input keeps current Command Window behavior: done
- slash commands route to assistant provider: done
- unknown slash commands fail clearly: done
- assistant output is visually distinct: done
- assistant work does not block the UI thread: done
- in-progress state is visible while assistant work runs: done
- assistant output is copyable: done
- real-runtime QA from the in-app Command Window: in progress

### Phase G: Fine-Tuning Experiment

Status: optional; dataset prep exists; not part of product integration yet.

Goal:

Only after the product/backend path stabilizes, test whether fine-tuning
improves answer discipline.

Allowed targets:

- section-format compliance
- citation copying
- concise grounded answers
- honest unknowns
- unsupported-feature refusal

Not allowed:

- replacing retrieval
- training on raw repo text alone
- hiding retrieval failures with model memory

Exit criteria:

- tuned model passes the same Phase 2/6/7 gates
- citation behavior does not regress
- unsupported behavior does not become more speculative

## Current Next Step

The next controlled engineering step should be:

```text
Phase B/E/F quality pass: improve knowledge-pack ranking and real-runtime QA
from the in-app Assistant panel and Command Window.
```

Reason:

- the pack, backend, panel, and slash-command path now work end to end
- broad questions still retrieve overly narrow implementation chunks
- the real model answers are structured, but answer quality still depends on
  better retrieval and curated overview/capability chunks
- runtime transport diagnostics were just improved and need one more in-app
  verification pass

## Do Not Start Yet

Do not start these until earlier gates pass:

- graph/source mutation
- automatic code or graph edits from assistant answers
- production shipping of the pack format
- fine-tuning run as a replacement for retrieval
- remote/non-local model calls from the plugin

These have already moved from planning to experimental implementation:

- Command Window slash routing
- debugger UI actions
- training UI actions

Keep them read-only until quality gates pass.

## Immediate Checklist

- [ ] Re-run source-specific command-window probes:
  `/ask What source file defines DebugTraceRecord?`
  `/find-source TrainingTraceEvent terminal_reason field`
- [ ] Re-run one broad help probe:
  `/ask what is cyxwiz engine?`
- [x] Improve broad-query retrieval with curated overview/capability chunks.
- [ ] Keep the local runtime endpoint at `http://127.0.0.1:8768/completion`.
- [ ] Keep the upstream model server separate from the engine for now.
- [ ] Rebuild Release engine and assistant plugin after each integration change.
- [ ] Capture bad answers for later fine-tuning only after retrieval quality is
  acceptable.

## Status Ladder

```text
Phase A planning/contracts       mostly done
Phase B knowledge pack           initial implementation used by plugin
Phase C backend runtime library  retrieval + local runtime path exists
Phase D assistant panel wiring   retrieval + local runtime path wired
Phase E debugger/training help   first read-only version exists
Phase F command window commands  first async slash-command version exists
Phase G fine-tuning experiment   optional later, dataset prep exists
```

## Recommendation

Use this roadmap as the controlling order:

```text
knowledge pack -> backend -> panel -> context actions -> command window -> optional tuning
```

That keeps the implementation aligned with the proven RAG architecture and
prevents the assistant from becoming an unbounded chatbot inside the engine.

Current practical focus:

```text
quality of retrieval -> quality of cited answers -> packaging boundary -> later tuning
```
