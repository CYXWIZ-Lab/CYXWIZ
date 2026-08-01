# To Fix 74 - CyxWiz AI Engineering Copilot, MCP Gateway, and Governed Research Loop

## Status

Open - research and architecture ticket. No product implementation is claimed
by this document.

## Decision statement

Implement this as a hybrid architecture:

- CyxWiz core owns the authoritative, typed automation contract, permission
  checks, approvals, task lifecycle, audit trail, and access to graph, dataset,
  training, testing, checkpoint, run, and artifact services.
- The AI assistant remains an optional plugin.
- MCP is a protocol adapter, preferably a separate local sidecar for external
  clients, rather than a mandatory dependency or an unrestricted service
  embedded in the engine.
- RAG, model-provider selection, conversational memory, planning policy, and
  external MCP connections belong to the assistant/plugin layer.
- Existing CyxWiz scripting remains a first-class execution plane. Script
  drafting, inspection, execution, and artifacts must use explicit
  engine-owned scripting contracts rather than an unrestricted MCP code tool.

The assistant may propose and explain work. The existing compiler, data audit,
task manager, training manager, test manager, and artifact systems remain the
source of truth and execute approved operations.

This boundary keeps the engine useful without an LLM, allows multiple
automation clients to reuse the same safe commands, and prevents MCP or one
model provider from becoming the engine architecture.

## Why this ticket exists

The APS failure-classification exercise demonstrated a practical engineering
loop:

1. understand the business and ML problem;
2. inspect the supplied data;
3. identify metadata rows, missing values, label semantics, class imbalance,
   and leakage risks;
4. design cleaning and preprocessing;
5. build and validate CyxGraphs;
6. train through the engine;
7. diagnose runtime, loader, loss, split, checkpoint, and test defects;
8. compare results and preserve reproducible artifacts.

An AI assistant can make this workflow substantially more approachable, but
only if it operates through engine-owned capabilities and exposes what it is
doing. APS is an acceptance example, not a special-case implementation. The
same workflow must work for other tabular, image, text, time-series, supervised,
unsupervised, and reinforcement-learning projects as their engine capabilities
become available.

## Relationship to existing CyxWiz work

This ticket extends, and must not duplicate, `tofix42 - Local CyxWiz
Source-Aware LLM Agent`.

Verified existing foundations include:

- the optional `com.cyxwiz.assistant` plugin and assistant panel;
- `IAssistantProvider` and assistant command routing;
- bounded engine, workspace, graph, run, node, debugger, and training context;
- the local knowledge-pack retrieval backend with source citations;
- local loopback model-runtime support;
- plugin permissions and per-version permission decisions;
- graph compilation and preflight;
- Data Studio profiling, auditing, preview, and materialization;
- asynchronous tasks, training, testing, checkpoints, run comparison, and
  artifact management.
- the Script Editor, project scripts, generated Python/PyCyxWiz code,
  `ScriptingEngine`, `PythonSandbox`, and `pycyxwiz` bindings;
- synchronous and asynchronous script/command execution, streaming output,
  plot capture, cancellation, timeouts, and the Python-driven RL route.

Important current limitations:

- there is no CyxWiz MCP server or client implementation;
- the current assistant is primarily read-only and retrieval-oriented;
- the knowledge pack is a useful RAG foundation, not yet a semantic retrieval
  and governed tool-use system;
- plugin callback isolation is not a security sandbox for hostile generated
  code, and the current embedded Python restrictions must not be treated as
  equivalent to an out-of-process hostile-code boundary;
- some older assistant documentation is behind the current panel and command
  implementation.

`tofix42` continues to own the local source-aware assistant and RAG foundation.
This ticket owns the next product layer: an ML engineering copilot, a governed
research loop, and MCP interoperability.

## OctOpus research summary - 2026-07-27

The public OctOpus material describes a workflow that starts with data and a
business objective, profiles the data, produces a research plan, runs and
compares experiments, evaluates the winner on a held-out set, and produces a
deployable model and report.

Useful product ideas:

- define a baseline, success metric, and stopping threshold before training;
- provide distinct ask/explore, clean/transform, and train-model workflows;
- make the research plan editable and require approval before execution;
- expose the proposed target, task, metric, split, features, leakage
  guardrails, and experiment budget;
- show planner decisions, stages, experiments, token/cost information, and
  run comparisons rather than presenting an idle interface;
- isolate the final holdout from the experiment loop;
- retain local artifacts, logs, models, commits, and an audit trail;
- support local, private, and enterprise deployment boundaries;
- connect external developer tools through MCP.

These are product-behavior observations from public pages, not a verified view
of OctOpus internals. CyxWiz should use them as design inspiration and must not
claim or infer undocumented implementation details.

### What CyxWiz should redesign rather than copy

CyxWiz is a typed graph engine. Its assistant should generate and operate on
CyxGraphs, node properties, dataset roles, preprocessing-state artifacts,
checkpoints, runs, and engine reports. It should not make arbitrary
LLM-authored `train.py` files or an opaque model file the primary contract.

The CyxWiz equivalent is:

```text
Problem goal
  -> bounded data inspection
  -> editable research plan
  -> graph draft
  -> compiler/preflight
  -> approved task execution
  -> immutable run evidence
  -> sealed holdout evaluation
  -> checkpoint/model artifact
  -> reproducible engineering report
```

Model-generated code can be an optional advanced capability later. It must run
out of process in a real restricted environment with no implicit secrets or
filesystem/network authority.

## Product scope

The copilot should help a user:

- turn a problem statement and dataset into an explicit ML goal;
- explain dataset roles, labels, splits, preprocessing, losses, metrics, and
  suitable model families;
- profile and audit data through bounded engine calls;
- propose an editable cleaning, feature, training, and evaluation plan;
- draft valid CyxGraphs using the live node/capability registry;
- explain, draft, review, and test project scripts using the supported
  `pycyxwiz` and scripting capability truth;
- compile and explain errors before execution;
- launch approved, bounded experiments as visible engine tasks;
- diagnose failures using logs, traces, graph context, source-aware RAG, and
  known support truth;
- compare runs against an explicit baseline and success threshold;
- protect a sealed test set from experimentation;
- load checkpoints, run final tests, and export artifacts only with approval;
- produce a reproducibility and decision report with citations and provenance.

This is an engineering copilot, not an autonomous replacement for the user.
It must distinguish evidence, inference, proposal, executed result, and unknown
state.

## Architecture

### 1. Core Automation API

Add a narrow, versioned engine API for domain operations. It must be reusable
from the GUI, tests, plugins, scripting bindings, and the MCP adapter.

The core owns:

- stable request and result schemas;
- live capability truth;
- opaque project, dataset, graph, task, run, checkpoint, and artifact handles;
- bounded read-only snapshots;
- graph validation and compilation;
- data profile/audit requests;
- asynchronous task creation, progress, cancellation, and terminal results;
- training, testing, checkpoint, and artifact lifecycle;
- approval requirements and permission evaluation;
- idempotency and correlation identifiers;
- provenance and audit events;
- safe error codes and user-facing remediation.

The API must not expose internal pointers, widgets, mutable global state, or
unbounded table contents. Core commands must remain useful and testable with no
assistant or MCP process installed.

### 2. Scripting execution plane

Scripting is an existing CyxWiz capability and must be represented explicitly.
The current engine has three separate model/workflow planes:

```text
Native graph
  -> compiler/materializer
  -> C++ TrainingExecutor
  -> native run/checkpoint/artifact

Generated Python/PyCyxWiz code
  -> Script Editor or exported file
  -> design-time artifact until the user explicitly executes it

Executed project/RL script
  -> ScriptingEngine
  -> embedded project Python environment + pycyxwiz
  -> streamed output/plots/status and script-owned artifacts
```

The copilot must not silently replace native graph execution with a script.
Every proposed plan must state which plane owns execution and what artifact
contract it produces.

The Core Automation API should eventually expose a typed scripting service for:

- scripting runtime diagnostics and capability discovery;
- listing and reading project-scoped scripts through bounded handles;
- syntax/static validation;
- creating a script draft in memory;
- producing a patch/diff for an existing script;
- explicit save with version/fingerprint checks;
- approved asynchronous execution through an engine task;
- streaming output and plot events;
- stop/cancel, timeout, resource status, and completion result;
- recording the script hash, environment identity, dependencies, inputs,
  outputs, and produced artifacts.

The first MCP version exposes only script metadata, bounded reads, explanation,
and validation. Script saving or execution is a later, separately approved
capability.

The existing embedded `PythonSandbox` remains useful for trusted user/project
scripts and compatibility. Before AI-generated or externally supplied code can
run, CyxWiz needs an out-of-process restricted script-worker profile. The
worker must receive explicit mounted inputs/outputs, no inherited secrets,
network denied by default, an environment lock/fingerprint, and hard resource
and cancellation controls.

### 3. Assistant plugin

The optional assistant plugin owns:

- assistant UI and conversation state;
- local or remote model-provider configuration;
- source-aware and project-aware RAG;
- prompt and plan construction;
- explanation and recommendation policy;
- research-plan editing;
- progressive discovery of relevant capabilities;
- presentation of approvals, evidence, unknowns, tasks, and results;
- optional connections to approved external MCP servers.

The in-engine assistant can call the Core Automation API directly. It does not
need to serialize every internal call through MCP.

### 4. CyxWiz MCP adapter

Provide a separate `cyxwiz-mcp-server` sidecar when external tools such as
Claude, Cursor, Codex, or an IDE need to operate CyxWiz.

Initial transport:

- local `stdio`;
- explicit launch by the client or CyxWiz;
- narrow authenticated local IPC from the sidecar to a running engine;
- no public listener and no unauthenticated localhost HTTP endpoint.

The sidecar maps MCP resources, tools, and prompts onto the Core Automation
API. It holds no independent engine truth and must not bypass permissions,
approvals, the compiler, task tracking, or audit logging.

An optional remote transport can be evaluated later only with authentication,
TLS, origin controls, tenancy, rate limits, and deployment-specific threat
modeling.

### 5. External MCP client support

The assistant may later consume approved external MCP servers for data
catalogs, warehouses, source control, experiment tracking, or documentation.

External tools must be:

- disabled by default;
- explicitly connected and scoped;
- discovered progressively rather than loading all tools into every prompt;
- treated as untrusted input that may contain prompt injection;
- brokered through CyxWiz permission and data-policy checks;
- visible in the task and audit UI.

External MCP results must never silently become executable graph changes.

## Proposed MCP surface

Names are illustrative and must be finalized through a versioned schema review.

### Read-only resources

```text
cyxwiz://project/current
cyxwiz://graph/active
cyxwiz://graph/compile-report
cyxwiz://capabilities/nodes
cyxwiz://capabilities/training
cyxwiz://datasets
cyxwiz://dataset/{id}/schema
cyxwiz://dataset/{id}/profile
cyxwiz://tasks/{id}
cyxwiz://runs/{id}
cyxwiz://checkpoints/{id}
cyxwiz://artifacts/{id}/metadata
cyxwiz://scripts
cyxwiz://script/{id}/metadata
cyxwiz://scripting/runtime
cyxwiz://docs/search
```

Dataset resources return bounded schema, profile, and policy-filtered samples;
they do not stream a whole dataset into an LLM context.

### Phase-one read-only tools

- `list_projects`
- `list_datasets`
- `inspect_dataset`
- `audit_dataset`
- `list_engine_capabilities`
- `compile_graph`
- `explain_compile_report`
- `list_tasks`
- `get_task_status`
- `list_runs`
- `compare_runs`
- `explain_trace`
- `list_scripts`
- `inspect_script`
- `validate_script`
- `get_scripting_runtime_diagnostics`
- `search_cyxwiz_knowledge`

### Proposal tools

- `propose_research_plan`
- `validate_research_plan`
- `propose_cleaning_plan`
- `propose_cyxgraph`
- `explain_cyxgraph`
- `propose_script`
- `propose_script_patch`
- `explain_script`

Proposal tools return drafts. They do not mutate a project.

### Later, approval-gated mutation tools

- `save_graph_draft`
- `apply_transform_plan`
- `start_training`
- `cancel_task`
- `load_checkpoint_for_test`
- `run_test`
- `save_script_draft`
- `run_project_script`
- `cancel_script_task`
- `export_model`
- `export_report`

Every mutation must support:

- a dry-run or preview where meaningful;
- an idempotency key;
- the expected source version/fingerprint;
- the exact requested scope;
- an approval record;
- a task or result handle;
- a complete audit event.

The first releases must not expose arbitrary shell execution, arbitrary Python,
unrestricted file reads/writes, raw SQL mutation, secret retrieval, direct
deployment, or destructive project deletion.

`run_project_script` is not an arbitrary-code loophole. It accepts a
project-scoped, versioned script handle or approved draft, a declared runtime
profile, bounded inputs/outputs, and execution limits. It runs as a visible
CyxWiz task and returns structured output, plots, violations, and artifacts.

### Optional MCP prompts

- `build_tabular_binary_classifier`
- `diagnose_failed_training`
- `compare_experiment_runs`
- `prepare_sealed_test_evaluation`
- `explain_dataset_quality`
- `draft_pycxwiz_analysis_script`
- `diagnose_script_failure`

Prompts are discoverable templates, not trusted execution authority.

## CyxWiz research-plan artifact

Define a versioned project artifact, provisionally `research_plan.cyxplan.json`,
containing:

- problem statement and intended user decision;
- learning task and dataset roles;
- target/label semantics, or an explicit label-free objective;
- baseline and success threshold;
- primary and secondary metrics;
- split and sealed-holdout policy;
- missing-data, leakage, imbalance, privacy, and quality findings;
- proposed preprocessing and fitted-state ownership;
- candidate graph/model families;
- experiment budget, resource limits, and stopping rules;
- expected artifacts;
- selected execution plane and, for scripts, the runtime/environment profile;
- approval state;
- links to the graph, dataset, preprocessing, run, checkpoint, and report
  fingerprints produced from the plan.

The plan is editable by the user. Updating an approved plan creates a new
version and requires renewed approval for affected operations.

## Generic guided workflow

### Stage 1 - Define

Collect the project objective, task type, baseline, metric, success threshold,
constraints, and data policy. Clearly report unsupported task types.

### Stage 2 - Inspect

Use the engine to inspect schemas and bounded samples. Run dataset audit and
profiling. Surface missing values, suspicious metadata/header rows, type
inference, class imbalance, leakage, duplicates, split errors, and resource
estimates.

### Stage 3 - Plan

Produce an editable research plan. Explain assumptions and alternatives.
Require approval before creating derived data or launching work.

### Stage 4 - Draft and validate

Create a CyxGraph draft only from the live capability registry. Validate node
properties, roles, connections, loss/output compatibility, metrics,
preprocessing-state ownership, and dataset mapping through the compiler.

When scripting is the selected plane, create a script draft or patch against
the live scripting and `pycyxwiz` capabilities. Show the exact code and
environment requirements. Static validation does not authorize saving or
execution.

### Stage 5 - Execute bounded experiments

Launch visible tasks with progress, cancellation, budgets, and failure
explanations. Each run links immutable:

- plan version;
- graph fingerprint;
- dataset and split fingerprint;
- fitted preprocessing-state fingerprint;
- script hash, project-environment fingerprint, declared inputs/outputs, and
  execution profile when a script owns the run;
- engine/build and capability version;
- seed and configuration;
- metrics, logs, traces, checkpoint, and artifacts.

### Stage 6 - Compare and decide

Compare every run to the declared baseline and threshold. Report uncertainty,
class-sensitive metrics, resource use, and failed experiments. Do not select a
winner using accuracy alone when the task or imbalance makes it misleading.

### Stage 7 - Sealed evaluation

Keep the final test role outside planning, feature selection, fitting, early
stopping, and experiment ranking. Load the selected checkpoint and fitted
preprocessing state, run one approved final evaluation, and append the result
to the immutable run record.

### Stage 8 - Package and report

With approval, export the checkpoint/model and required preprocessing state.
Produce a report covering data, decisions, experiments, metrics, limitations,
provenance, reproduction steps, and citations.

## APS acceptance scenario

APS may validate the generic workflow without creating APS-specific engine
logic.

The assistant should be able to:

- recognize descriptive source rows before the actual CSV header;
- identify `na` as a missing token when supported/configured;
- establish train and test dataset roles;
- identify `class` as the binary label;
- warn about imbalance and propose suitable metrics and weighted loss;
- propose fitted imputation and scaling state learned only from training data;
- draft cleaning and classifier graphs;
- compile before execution;
- show task progress;
- compare baseline and candidate runs;
- preserve the sealed test data;
- load the selected checkpoint and preprocessing state for final testing;
- generate an evidence-backed report.

Success on APS proves only this scenario. Capability-driven tests must use
additional synthetic and project datasets.

## Security and governance requirements

Treat model output, retrieved documents, user-supplied files, generated plans,
generated code, and external MCP results as untrusted.

Required controls:

- least-privilege, operation-specific permissions;
- user approval for every consequential action unless the user has granted a
  clear, revocable categorical policy;
- visible exact action, target, inputs, and expected effect at approval time;
- no token passthrough or secrets in prompts, logs, artifacts, or tool results;
- bounded and policy-filtered dataset samples, with schema/profile preferred;
- PII and sensitive-column redaction;
- sealed holdout inaccessible to the planner and model context;
- local authenticated broker for engine operations;
- out-of-process sandboxing for any generated code;
- an explicit distinction between trusted user-run embedded scripting and
  isolated execution of AI-generated or externally supplied scripts;
- project-scoped script handles and canonical path checks, with no arbitrary
  path supplied by the model;
- time, memory, data, token, and experiment budgets;
- cancellation and durable task handles for long work;
- prompt-injection marking and output filtering for external MCP content;
- audit of plans, calls, approvals, denials, fingerprints, results, and errors;
- versioned tool schemas and capability negotiation;
- safe failure when the engine, sidecar, plugin, provider, or capability is
  absent.

`SafeExecute`-style exception containment is useful for stability but must not
be represented as a hostile-code security boundary.

## UX requirements

The product must make the agent's state visible:

- current stage: Define, Inspect, Plan, Validate, Execute, Compare, Test, Report;
- what the assistant knows, assumes, proposes, and cannot verify;
- active engine tasks and node/graph progress;
- pending approval and its exact consequences;
- experiment count and remaining budget;
- current baseline, best run, and success threshold;
- data, graph, preprocessing, checkpoint, and artifact provenance;
- citations to CyxWiz documentation/source and external references;
- a stop/cancel control that terminates engine work as well as generation.

Conversation is an interface to the workflow, not the only record of it.
Plans, approvals, tasks, runs, and reports must survive closing the panel.

## Implementation phases

### Phase 0 - Reconcile current assistant truth

- update `tofix42` implementation notes to match the current panel and command
  provider;
- inventory the present context snapshot and RAG limits;
- define which existing plugin permissions apply;
- ensure unsupported actions are stated explicitly.

### Phase 1 - Core read-only Automation API

- define versioned handles, requests, results, errors, provenance, and audit
  events;
- expose read-only project, graph, capability, dataset-audit, task, run,
  checkpoint, trace, and documentation-search operations;
- enforce bounds and policy filters;
- test the API without an assistant or MCP dependency.

### Phase 2 - Read-only MCP sidecar

- implement the local `stdio` sidecar;
- connect through narrow local authenticated IPC;
- expose the phase-one resources and read-only tools;
- add schema, disconnect, cancellation, timeout, and compatibility tests;
- verify that the sidecar cannot bypass engine permissions or bounds.

### Phase 3 - Copilot planning and RAG

- add the versioned research-plan artifact and editor;
- integrate engine capability truth and semantic/project RAG;
- propose cleaning and CyxGraph drafts;
- add approval UX while keeping all proposals non-mutating.

### Phase 4 - Governed experiment loop

- add bounded experiment execution through engine tasks;
- link immutable dataset, graph, preprocessing, run, checkpoint, and metric
  fingerprints;
- integrate run comparison, early stopping, budget, and sealed-holdout policy;
- expose progress in both the graph and assistant UI.

### Phase 5 - Gated lifecycle actions

- add checkpoint loading, final testing, export, and report tools;
- add separately gated script save and isolated script-execution tools after
  the script-worker security boundary is complete;
- add per-tool policies, dry-runs, idempotency, approvals, and audit;
- evaluate external MCP clients and private remote deployments;
- keep arbitrary generated code and deployment out of scope until separately
  threat-modeled.

## First implementation slice

The first shippable slice is deliberately small:

1. define the Core Automation API schemas and capability registry;
2. expose current project, active graph, compile report, node capabilities,
   bounded dataset audit, tasks, traces, runs, script metadata/runtime
   diagnostics, and knowledge search as read-only operations;
3. add a local read-only `stdio` MCP sidecar;
4. allow the existing assistant to call compile, audit, trace explanation, and
   RAG operations;
5. show evidence, correlation IDs, and errors in the assistant panel;
6. expose no mutations and no training-start tool in this slice.

The first slice also exposes no script-save or script-execution tool.

This proves the boundary and security model before autonomous execution is
introduced.

## Acceptance criteria

### Architecture and independence

- the engine builds, starts, trains, tests, and runs Data Studio without the
  assistant plugin or MCP sidecar;
- the same Core Automation API is used by the assistant and MCP adapter;
- no adapter duplicates compiler, task, training, testing, or artifact truth;
- schemas are versioned and unknown versions fail clearly.

### Read-only safety

- dataset inspection is bounded, redacted, and policy-aware;
- no read-only tool can mutate graphs, files, datasets, training, or artifacts;
- disconnected, unavailable, stale-handle, and permission-denied states return
  structured errors;
- external MCP results cannot trigger another tool without a new policy check.

### Scripting correctness

- the assistant reports whether work uses native graph execution, code
  generation/export, embedded trusted scripting, or the isolated script worker;
- generated code is always shown as a draft or patch before save;
- a script cannot run from an arbitrary filesystem path supplied by the model;
- approved script execution is a visible, cancellable task with output, plots,
  timeout/resource status, environment identity, and artifact provenance;
- script execution never silently registers a model in the native
  `TrainingExecutor` or claims a native checkpoint unless an explicit future
  import contract verifies it;
- native graph training remains available and unchanged when scripting or its
  Python environment is unavailable.

### Workflow correctness

- a plan records baseline, metric, threshold, split, data roles, preprocessing
  ownership, candidates, budget, and approvals;
- generated graphs use only capability-reported nodes/properties;
- compiler failure prevents execution and is explainable;
- every experiment is a visible, cancellable task and an immutable run;
- fitted preprocessing state is learned from training data and reused for
  validation/test;
- final test data is sealed until an approved winner is selected;
- reports link all relevant fingerprints and evidence.

### User control and audit

- consequential actions display exact scope and require approval;
- denials and cancellations stop the operation safely;
- retries with the same idempotency key do not duplicate work;
- plans, tool calls, approvals, task state, and results survive UI closure;
- every assistant claim clearly distinguishes retrieved evidence, engine
  result, inference, and unknown.

### Testing

- unit tests cover schema validation, permissions, bounds, redaction,
  idempotency, stale handles, and audit emission;
- integration tests cover engine-to-sidecar disconnect and cancellation;
- adversarial tests cover prompt injection, oversized output, secret-like
  content, invalid paths, untrusted MCP results, and denied tools;
- APS validates one end-to-end scenario, while synthetic datasets prove the
  implementation is generic.

## Non-goals for the first implementation

- replacing the graph compiler or runtime with an LLM;
- embedding a mandatory cloud model provider in the engine;
- unrestricted autonomous training;
- arbitrary shell, Python, filesystem, SQL-mutation, or deployment tools;
- silently executing generated scripts through the current embedded
  interpreter;
- sending complete datasets to a model by default;
- exposing the sealed test set during planning;
- duplicating Data Studio, task, run-comparison, checkpoint, or artifact
  systems inside the plugin;
- fine-tuning a CyxWiz model before the RAG and tool-use evidence set is mature;
- claiming support for engine nodes or algorithms that capability truth marks
  unavailable.

## Dependencies and related tickets

- `tofix42` - local source-aware assistant and RAG foundation;
- `done70` - completed production Data Studio and generic dataset workflow;
- `tofix71` - preprocessing fit/transform state and train/test correctness;
- `tofix73` - backend algorithm capability truth and cross-computation parity.

The copilot must consume the support truth delivered by these systems. It
must not hide or work around their unsupported states.

## Research references

Official public sources reviewed on 2026-07-27:

- [OctOpus product overview](https://www.octoopus.dev/)
- [OctOpus product application tour](https://www.octoopus.dev/app)
- [OctOpus workflow guide](https://www.octoopus.dev/how-to-use)
- [OctOpus enterprise workflow](https://www.octoopus.dev/enterprise)
- [OctOpus security](https://www.octoopus.dev/security)
- [Model Context Protocol server guide](https://modelcontextprotocol.io/docs/develop/build-server)
- [Model Context Protocol server concepts](https://modelcontextprotocol.io/docs/learn/server-concepts)
- [MCP client implementation best practices](https://modelcontextprotocol.io/docs/develop/clients/client-best-practices)
- [MCP Tasks extension](https://tasks.extensions.modelcontextprotocol.io/)
- [MCP security best practices](https://modelcontextprotocol.io/docs/tutorials/security/security_best_practices)

Only `octoopus.dev` material was treated as OctOpus product information.
Similarly named Octopus Deploy material is a different product and is not a
source for this design.
