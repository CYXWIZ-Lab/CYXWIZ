# Weakness To Fix 83 - CyxWiz Engine Production Hardening and Simplification

## Status

Open - product hardening, architecture governance, and release-readiness
umbrella ticket. This document records current weaknesses and coordinates their
existing owners. It does not claim those weaknesses are already fixed, and it
must not become one giant implementation branch.

## Executive assessment

CyxWiz Engine already has a broad and capable product surface: visual graph
authoring, Data Studio, Arrow/Parquet ingestion, model construction, training,
testing, checkpoints, scripting, plugins, monitoring, debugging, serving, and
distributed-compute foundations.

Its primary weakness is not missing feature count. The product surface is
currently broader than its verified production contracts. Several workflows
work individually, but equivalent truth is not always shared by the GUI,
metadata, compiler, executor, backend, task system, checkpoint system, and
logs.

The next product stage must therefore prioritize:

```text
execution truth
  -> stability and responsiveness
  -> persistence and reproducibility
  -> canonical node contracts
  -> modality and objective generalization
  -> additional algorithms, plugins, and AI assistance
```

Adding more visible features before these boundaries are dependable increases
maintenance cost and makes the engine harder for users to trust.

## Decision statement

CyxWiz must enter a consolidation-first production-hardening cycle.

- Existing supported workflows are preserved and proven end to end.
- Unsupported or partial capabilities fail closed with exact reasons.
- One concept has one visible node, one typed contract, and one authoritative
  runtime meaning.
- Long operations are asynchronous, cancellable, and visible in Task View.
- Preferences, compiler claims, runtime execution, monitoring, artifacts, and
  logs report the same observed truth.
- Optional product layers remain plugins or later tickets until the core
  contracts they depend on are stable.

This ticket coordinates and measures that work. Each technical correction
continues in its narrow owning ticket.

## Current strengths that must not regress

- Track70's verified Arrow/Parquet production-data workflow is complete.
- Data Input supports bounded preview, production parsing options, role-aware
  tabular sources, and persistent ingestion caching.
- Train, external Dev/Test, and derived tabular partitions have a typed runtime
  boundary.
- Pipeline execution uses Task View and supports cooperative cancellation for
  verified paths.
- Fill Missing and Standard Scaler can persist training-fitted state and reuse
  it on evaluation data.
- Checkpoints can load a trained model for testing without forcing retraining.
- The Studio Debugger and training traces provide a real observability
  foundation.
- The plugin and scripting systems provide valid extension planes when their
  permissions and artifacts remain governed.

Production hardening must simplify around these working foundations, not
replace them with a second framework.

## Weakness register

| Priority | Weakness | Current risk | Owning ticket |
|---|---|---|---|
| P0 | Computation truth | A run can contain real ArrayFire GPU kernels while loss, metrics, gradients, or optimizer preparation cross back to the host. GUI/compiler placement can therefore overstate complete GPU execution. | `tofix81` |
| P0 | Capability depth and backend parity | The visible algorithm/loss/optimizer surface is broader than proven native execution. XGBoost is not integrated, and Python/native/plugin support can disagree. | `tofix73` |
| P0 | Stability and runtime feedback | A long task that freezes, crashes, reports false progress, or provides no node-level activity destroys user trust even when the underlying operation eventually succeeds. | This ticket's integration gate plus the owning workflow ticket |
| P1 | Exact resume and recovery | Existing checkpoints support testing and warm-start foundations, but complete optimizer/scheduler/RNG/batcher/runtime restoration is not universally available. | `tofix75` |
| P1 | Persisted experiment history | Run Comparison and training records are not yet a complete restart-persistent, reproducible project ledger. | `tofix76` |
| P1 | Duplicate node and parameter contracts | Canonical nodes, legacy aliases, format-specific nodes, duplicated registries, and parameter aliases can diverge between UI and runtime. | `tofix82`, with migration rules shared across the engine |
| P1 | Preprocessing state visibility | Fitted state can be persisted, but file paths do not express state flow as safely as typed graph ports. | `tofix77` |
| P2 | Non-tabular parity | Image, audio, text, sequence, and time-series datasets do not yet have all the role, partition, batching, and schema guarantees proven for Arrow/Parquet tabular data. | `tofix78` |
| P2 | Objective-family generality | Parts of compilation/training remain centered on supervised feature/label assumptions. Target-free estimators and RL require explicit dispatch contracts. | `tofix79` |
| P2 | Non-tabular preview parity | Registered text/image/audio assets lack the shared, bounded, lazy preview experience now available for tabular data. | `tofix80` |
| P3 | Optional intelligence before core proof | An AI engineering copilot could amplify both good and incorrect engine behavior if it is built before commands, permissions, artifacts, and execution truth are authoritative. | `tofix74`, after required core gates |

## Weakness 1 - Computation placement is not yet end-to-end truth

The engine can initialize ArrayFire CUDA and execute device kernels, but that
does not prove the full training step is device-resident. Host-side loss,
metric, gradient, or optimizer work can produce short GPU bursts separated by
transfers and synchronization.

Production correction:

- freeze one backend/device context for the lifetime of a run;
- compile against that exact context;
- reject unsupported operations before training;
- trace deliberate host/device boundaries;
- make Monitor and Task View report observed placement and transfer facts;
- prohibit silent CPU fallback inside a run declared device-resident.

Success is correctness and observability first, utilization second. High Task
Manager percentage alone is not acceptance evidence.

Owner: `tofix81`.

## Weakness 2 - Capability breadth exceeds verified backend depth

CyxWiz contains many node names, algorithms, losses, optimizers, scripting
bridges, and plugin hooks. A visible or serializable node can look implemented
before its complete compiler/runtime/backend route is proven.

Production correction:

- maintain one typed capability record for UI availability, compile support,
  runtime support, device support, checkpoint support, and test evidence;
- hide or label partial capabilities instead of simulating success;
- never substitute a different algorithm, loss, optimizer, or backend;
- add algorithms only through one complete vertical test from graph to result;
- distinguish native gradient boosting from actual XGBoost integration.

Owner: `tofix73`.

## Weakness 3 - Stability and responsiveness are inconsistent product-wide

Several previously investigated operations exposed the same failure pattern:
work begins correctly, but the GUI freezes, progress is misleading, the graph
appears idle, or a failure becomes visible only in the log. Individual fixes
have improved Data Input, conversion, pipelines, training, and testing, but a
shared release gate is still needed.

Every operation expected to exceed one UI frame must:

- snapshot UI-owned state before dispatch;
- execute outside the render thread;
- appear in Task View with monotonic bounded progress or an explicit
  indeterminate state;
- support cooperative cancellation where the backend can stop safely;
- publish completion back on the UI thread;
- leave no partial final artifact after failure or cancellation;
- retain enough trace context to diagnose a crash or stall.

The goal is not animation for decoration. The user must always know what is
running, where it is running, what it is waiting for, and how it ended.

Owner: Tofix83 defines the cross-workflow gate; each affected workflow ticket
owns its implementation.

## Weakness 4 - Training persistence is incomplete

A production experiment system must survive restart and machine interruption.
Loading model weights for testing is useful but not equivalent to exact resume
or reproducible run history.

Required layers:

1. checkpoint v2 with complete, versioned state inventory;
2. transactional restoration of model, optimizer, scheduler, precision,
   counters, RNG, and batcher position where supported;
3. persisted run manifests with graph, dataset, preprocessing, build, device,
   metrics, warnings, and checkpoint identities;
4. GUI workflows for Resume, Test, Compare, and explainable incompatibility.

Owners: `tofix75` and `tofix76`.

## Weakness 5 - Duplicate concepts create architectural drift

Examples include:

- Data Output alongside Export CSV, Export JSON, and Export Parquet;
- canonical nodes alongside visible or executable legacy aliases;
- `file_type`, `format`, and `type` for related format concepts;
- `file_path` and `path` for related path concepts;
- separate metadata, creation, dialog, compiler, truth, runtime, icon, and
  serialization lists.

Compatibility is necessary, but compatibility aliases must not remain new-
graph product concepts indefinitely.

Production correction:

- one visible canonical node per concept;
- one typed parameter resolver with deterministic migration;
- legacy names load safely but normalize before execution;
- genuine alias conflicts fail closed;
- new capabilities extend registries/adapters rather than create sibling node
  families;
- remove duplicate branches only after migration and behavior-parity tests.

The current concrete consolidation is owned by `tofix82`.

## Weakness 6 - Preprocessing state is functional but not graph-explicit

Training-fitted Fill Missing and Standard Scaler state can be written and
reapplied, but path-based coordination remains easy to misconfigure and hard
for the compiler to reason about.

Production correction:

- introduce a typed preprocessing-state artifact contract;
- distinguish Dataset pins from State pins;
- make Fit + Transform versus Transform Only visible in graph semantics;
- prevent fitting behind validation/test roles;
- preserve engine-managed artifact defaults without requiring users to invent
  paths;
- keep file persistence as an artifact implementation, not the graph type.

Owner: `tofix77`.

## Weakness 7 - Production data guarantees are uneven across modalities

Track70 proves the main Arrow/Parquet tabular route. Equivalent role and
partition behavior must still be proven for registered image, audio, text,
sequence, and time-series data.

Production correction:

- use a modality-neutral partition identity;
- preserve external Dev/Test sources in full;
- provide chronological leakage-safe time-series splits;
- keep augmentation, balancing, shuffle, and drop-last phase-aware;
- ensure prefetch wrappers retain their actual sources;
- validate structured, graph-generated, and absent targets without forcing a
  label-column model.

Owner: `tofix78`.

## Weakness 8 - Training objectives are not yet one explicit family system

Supervised tensor training, graph-generated targets, estimator fitting, and
reinforcement learning are different execution contracts. Treating all of
them as variations of X/Y supervised learning produces false label errors and
special cases.

Production correction:

- classify the objective family at compile time;
- validate only the target contract required by that family;
- dispatch to the correct executor without silent reinterpretation;
- keep unsupported families honest and unavailable;
- share artifacts, tasks, logs, and run identity across families without
  forcing one training loop onto all of them.

Owner: `tofix79`.

## Weakness 9 - Preview maturity is tabular-first

Registered tabular assets have bounded paging and shared rendering. Text,
image, and audio still need equivalent registered-identity adapters so Data
Input and Asset Browser do not rediscover or parse source files independently.

Production correction:

- one preview request/result envelope;
- modality adapters over registered datasets;
- bounded lazy samples, thumbnails, records, or waveform metadata;
- cooperative cancellation and stale-request rejection;
- inspection-only behavior that cannot mutate data roles or parsing settings.

Owner: `tofix80`.

## Weakness 10 - Optional features can bloat an unstable core

Plugins, scripting, RAG, MCP, and an AI engineering copilot are valuable, but
they must compose with the engine rather than expand its trusted core.

Production correction:

- core owns typed commands, permissions, immutable snapshots, task identity,
  artifact identity, and audit records;
- scripting and MCP are governed adapters over those commands;
- the AI assistant remains optional and cannot bypass compiler/runtime gates;
- advanced analytics tools prefer plugins when they do not belong to the
  minimal production Data Studio core;
- no assistant may claim success without verified engine evidence.

Owner: `tofix74`, gated by the relevant P0/P1 contracts.

## Consolidation guardrails

Until the P0 release gates pass, new core feature work must answer all of the
following:

1. Does this solve a current verified user need?
2. Can an existing primitive or plugin provide it?
3. Does it introduce a second representation of an existing concept?
4. Is UI, compiler, runtime, device, persistence, and test support complete?
5. Can unsupported states fail before mutation or long execution?
6. Can one engineer trace the feature from graph serialization to final
   artifact without relying on undocumented conventions?

If the answer reveals an unowned dependency, record that dependency instead of
shipping a partial visible feature.

## Release-readiness scorecard

A workflow is production-ready only when all applicable columns are proven:

| Boundary | Required proof |
|---|---|
| Graph | Canonical node and typed pins/parameters serialize and reload. |
| UI | Configuration is truthful, themed, responsive, and explains invalid states. |
| Compile | Capability, schema, objective, and device checks fail before mutation. |
| Runtime | The intended implementation executes without a hidden substitute. |
| Task | Progress, cancellation, terminal status, and error ownership are visible. |
| Device | Requested, resolved, and observed placement agree. |
| Artifact | Output is project-owned, versioned where needed, and atomically published. |
| Persistence | Test/resume/reopen behavior is explicit and reproducible. |
| Logs | Records identify node, run, dataset, device, artifact, and failure reason. |
| Tests | Focused contract tests plus at least one end-to-end production fixture pass. |

No workflow should be marked complete because only its happy-path backend
function exists.

## Recommended implementation order

### Gate A - Trust the running engine

1. `tofix81`: freeze and observe one real computation context.
2. `tofix73`: reconcile visible backend capabilities with executable truth.
3. Apply the Tofix83 async/task/crash gate to Data Input, pipeline execution,
   training, testing, checkpoint operations, and exports.

### Gate B - Trust persisted experiments

4. `tofix75`: complete checkpoint v2 and exact resume for supported stacks.
5. `tofix76`: persist run manifests and Run Comparison history.

### Gate C - Simplify graph semantics

6. `tofix82`: consolidate Data Output and legacy export nodes.
7. `tofix77`: make preprocessing state typed and graph-visible.
8. Run an engine-wide canonical-node/alias audit and remove creation-surface
   duplicates without breaking old graphs.

### Gate D - Generalize proven foundations

9. `tofix78`: extend role/partition contracts beyond tabular data.
10. `tofix79`: introduce objective-family dispatch.
11. `tofix80`: add registered non-tabular preview adapters.

### Gate E - Optional product expansion

12. Proceed with `tofix74` AI-copilot implementation only through the governed
    command, permission, task, artifact, and audit boundaries proven above.
13. Add algorithms or analytics tools through narrow backend modules or
    plugins, with one complete vertical acceptance case each.

The dependency order may be adjusted when two tickets are genuinely
independent, but lower gates must not be declared production-ready while their
required higher-priority contracts are unresolved.

## Acceptance criteria

1. Every visible node has one authoritative capability record and a proven
   runtime route; unsupported nodes are hidden or explicitly unavailable.
2. A selected device is frozen for a run, and observed execution/transfer facts
   match compiler, Monitor, Task View, and logs.
3. No supported long operation blocks the render thread; progress and terminal
   state are truthful.
4. Supported checkpoint-v2 workflows resume exactly or reject with a precise
   missing-state reason.
5. Run history survives restart with validated graph, dataset, preprocessing,
   build, device, metric, and checkpoint identities.
6. New graphs expose canonical nodes only; supported legacy graphs load,
   migrate, execute, and save safely.
7. Preprocessing state can be represented as a typed artifact flow without
   manual path coordination for the normal project workflow.
8. Non-tabular datasets receive explicit role/partition contracts before they
   are advertised as equivalent to tabular production support.
9. Supervised, target-free, graph-target, and RL workflows validate according
   to their actual objective family.
10. Data Input and Asset Browser use registered, bounded preview adapters for
    every advertised modality.
11. Optional AI/plugin/script actions cannot bypass compile, permission,
    device, task, artifact, or audit contracts.
12. At least one tabular classification case, one time-series forecasting case,
    and each newly supported modality complete a documented
    load -> prepare -> train/fit -> test/evaluate -> persist -> reopen cycle.
13. A final simplification audit records duplicate branches removed, legacy
    aliases retained with reasons, unsupported claims removed, and net new
    dependencies introduced.

## Non-goals

- Implementing Tofix73-82 in one change set.
- Reopening the completed Track70 Arrow/Parquet acceptance scope.
- Removing backward compatibility without migration evidence.
- Chasing maximum GPU utilization before correctness and placement truth.
- Adding every known algorithm, file format, model family, or analytics tool to
  the core.
- Treating UI animation as a substitute for real task progress.
- Building the AI copilot as an alternate executor that bypasses the engine.

## Completion rule

Tofix83 is complete only when the production-hardening gates are evidenced by
the scorecard, the owned child tickets are either completed or explicitly
removed from the production claim, and the Engine can describe its supported
capabilities without qualifiers hidden in logs or source code.

The desired result is a smaller and more trustworthy product contract, even if
that means exposing fewer capabilities until their complete vertical paths are
proven.
