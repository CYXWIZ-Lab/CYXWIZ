# TOFIX101: Unified Data Studio and Node Runtime Design

Status: Research and implementation design

Last updated: 2026-08-24

Target branch: `Nodes_Implementation`

## 1. Executive decision

CyxWiz should consolidate Data Explorer and Data Studio into one Data Studio
workspace for tabular data discovery, querying, profiling, cleaning, and
visualization. Reproducible work must be published to the existing Node Editor
and executed by the existing pipeline runtime.

The defining boundary is:

```text
Data Studio = explore + configure + preview + publish
Node system = execute + serialize + reproduce
```

Data Studio must not become a second pipeline runtime. Interactive previews and
published graphs must use the same canonical operation semantics. A node being
visible in the GUI does not prove that it has a backend implementation.
Runtime support is the source of truth.

## 2. Problem statement

The engine currently presents overlapping data workflows:

- Data Explorer provides file browsing, schema inspection, SQL, results,
  statistics, charts, cleaning controls, exports, and handoffs to other panels.
- Data Studio provides dataset selection, DuckDB SQL, analysis, visualization,
  and navigation to the Node Editor.
- The Node Editor contains data input, transformation, analytics,
  visualization, and export nodes, but some palette entries are templates or
  have narrower runtime/storage support than their GUI suggests.

This creates several user and engineering problems:

- Users must choose between two similar workspaces.
- The same operation can have different behavior depending on which panel
  invokes it.
- Analysis can be performed interactively but may need to be rebuilt manually
  as a graph.
- Multiple query and data conversion paths increase memory use and correctness
  risk.
- GUI implementation status can be mistaken for executable backend support.
- Legacy code remains after the visual pipeline was moved to the Node Editor.

The goal is one coherent route from source data to exploration and then to a
reproducible graph without duplicating loaders, operators, exporters, or plots.

## 3. Product direction and references

Databricks is the closest high-level product reference because it brings data
discovery, tabular preview, SQL, profiling, visualization, and downstream work
into a connected workspace. Dataiku contributes an important interaction
pattern: non-destructive operations produce derived datasets/recipes instead
of silently mutating a source.

CyxWiz should adopt those patterns without copying either product wholesale:

| Reference pattern | CyxWiz interpretation |
|---|---|
| Discover and inspect a dataset in context | Select a registered dataset or open a supported file in Data Studio |
| Query and profile without loading everything into a UI table | Use bounded Arrow/Parquet previews and DuckDB-backed analysis |
| Build non-destructive preparation steps | Configure canonical data nodes against an immutable source |
| Turn exploration into an operational workflow | Publish the configured chain to the Node Editor |
| Keep lineage visible | Preserve source, operation parameters, outputs, and graph links |

CyxWiz remains node-first for reproducible execution. It does not need a second
notebook system, a second visual canvas, or a proprietary data catalog to gain
the useful parts of these references.

## 4. Source-verified current state

This section records what was verified in the current branch rather than what
the UI appears to promise.

### 4.1 Data Studio

`DataStudioPanel` is already a small container with three components:

- `QueryEditor`
- `Analyzer`
- `Visualizer`

The panel selects datasets from `DataRegistry` and provides a button to open
the Node Editor. Its former Pipeline tab was removed in favor of the Node
Editor, which is the correct product boundary.

Important limitations found in the current implementation:

- Query, Analyzer, and Visualizer each own a `DuckDBConnector`; connection and
  registration state are therefore repeated.
- These components currently request `ArrowDataset` directly from
  `DataRegistry`, so disk-backed Parquet support is not consistently available.
- Several analysis and visualization actions are still TODOs, including parts
  of correlation, outlier analysis, histograms, real column loading, and real
  plot rendering.
- Query results are useful for interactive exploration, but ad hoc SQL does
  not yet have a canonical runtime-backed SQL node that can be published
  safely.

### 4.2 Data Explorer

`DataExplorerPanel` remains compiled, constructed, registered, and rendered by
`MainWindow`. Its implementation is approximately 2,402 lines plus a 307-line
header and combines:

- file browsing and file dialogs;
- schema loading and worker-thread ownership;
- DuckDB/DataLoader initialization;
- SQL editing and query history;
- string-table result storage and pagination;
- CSV/JSON export;
- descriptive statistics and correlations;
- chart rendering;
- cleaning previews and SQL generation;
- cross-panel handoffs.

This is a module-growth and weak-boundary risk. It also represents a parallel
product path that should be migrated, not expanded.

The query route returns a backend `Tensor` and converts values into a
`vector<vector<string>>` for display. That path is not a suitable canonical
representation for analytical tables because it can lose or flatten schema,
type, null, and categorical semantics.

### 4.3 Existing foundations to retain

The repository already contains useful canonical building blocks:

- `DataRegistry` owns registered datasets and exposes both Arrow and
  Parquet-backed tabular representations.
- `DataPreviewService` provides bounded, cancellable, paginated previews for
  registered Arrow and Parquet datasets. It limits a request to 200 rows,
  preserves schema information, and reads Parquet by row group.
- `PipelineRuntimeCapabilities` records runtime mode, fail behavior,
  implementation owner, required inputs, required parameters, allowed values,
  numeric constraints, aliases, and materializer/storage support.
- `DataStudioExecutionPlan` resolves aliases, rejects unknown or fail-closed
  nodes, validates parameters and arity, detects cycles, and produces a
  topologically ordered plan.
- `PipelineExecutor` and `PipelineOperatorFactory` are the current execution
  owners.
- `NodeMetadataRegistry` remains useful for labels, categories, icons, pins,
  editors, and help text.

These should be strengthened and reused rather than replaced by Data
Studio-specific copies.

### 4.4 Legacy and migration debt

The old Data Studio `PipelineCanvas` and its private node registry remain in
the source tree. `PipelineCanvas` is no longer compiled, but its files still
contain a separate quick-add inventory and about 886 lines of obsolete canvas
behavior. The private data-studio node registry is still compiled even though
the Node Editor is now the intended visual workflow.

These files must not be reused to create another graph model. After verifying
that no active consumer depends on them, remove them as part of the migration.

### 4.5 Runtime and storage gaps

The runtime capability work is a strong start, but operation support is not a
single Boolean. Effective support depends on all of the following:

- canonical node type;
- implementation owner;
- real versus simulated, pass-through, or hard-fail behavior;
- input arity and parameters;
- artifact/pin type;
- current storage backend;
- materialization limits;
- source compatibility.

Examples verified in the current code:

- CSV, JSON, and Parquet export nodes are marked implemented and dispatched by
  `PipelineExecutor`, but their implementations currently require a registered
  Arrow dataset.
- `BarChart` is implemented as an inspection dialog, but currently requires an
  Arrow-backed dataset and reports disk-backed data as unsupported.
- Generic `VisualizeData` and a number of other palette nodes remain templates.
- The SQL Query node is GUI-visible but backend runtime support is blocked; it
  must not be represented as a publishable operation.

These are canonical boundary gaps. Data Studio must report them honestly and
must not hide them by adding private conversion or execution paths.

## 5. Core invariants

The implementation must preserve these rules:

1. **One runtime truth.** Runtime capability, not palette visibility or
   metadata status alone, decides whether an operation can execute or publish.
2. **One reproducible graph.** The Node Editor is the only visual pipeline
   authoring and serialization surface.
3. **One operation implementation.** Preview, one-off execution, and graph
   execution invoke the same canonical operator/executor semantics.
4. **No silent mutation.** Sources remain unchanged; transformations create a
   derived preview or registered output.
5. **Preserve tabular meaning.** Schema, types, nulls, column names, and storage
   identity survive the exploration-to-runtime boundary.
6. **Bound expensive work.** Preview does not imply full materialization.
   Copies, scans, host/device transfers, and disk reads are explicit and
   cancellable where practical.
7. **Fail closed.** Unsupported operations are disabled or rejected with a
   specific reason; they never simulate success.
8. **Cross-platform behavior.** Common behavior uses portable C++; OS-specific
   dialogs and integration stay behind focused adapters.
9. **Remove migrated paths.** Consolidation is incomplete while Data Explorer
   and obsolete Data Studio pipeline implementations remain as permanent
   alternatives.

## 6. Capability model

Data Studio needs a normalized, read-only capability query built from the
existing runtime capability and node metadata registries. It should not add a
third hand-maintained inventory.

An effective capability descriptor should answer:

```text
identity: canonical node type and compatibility aliases
presentation: label, category, help, pins, and parameter editor schema
runtime: owner, support mode, and fail mode
inputs: artifact types, arity, and accepted storage backends
parameters: required values, defaults, enums, and numeric constraints
outputs: artifact type and registration/materialization behavior
actions: previewable, executable once, publishable
reason: precise explanation when an action is unavailable
```

The user-facing states are:

| State | Meaning | UI behavior |
|---|---|---|
| Runtime-backed | A real executor/operator supports the operation for the selected source and parameters | Enable preview, run, and publish as applicable |
| Explore-only | A bounded interactive feature exists but has no canonical node runtime | Allow exploration; label it “Explore only”; disable publish with a reason |
| Unavailable/planned | Runtime is unknown, fail-closed, simulated, pass-through, or incompatible with the selected storage | Hide from normal action lists or disable with the runtime reason |

`NodeImplementationStatus::Implemented` is presentation evidence, not
sufficient runtime evidence. Conversely, a runtime-backed legacy operation may
remain usable while its metadata is migrated, provided the normalized contract
can resolve it unambiguously.

## 7. Target user experience

### 7.1 Source entry

The user enters Data Studio through either route:

1. Select a dataset already registered by a `DataInput` node or another graph
   output.
2. Open a supported file directly.

For a direct file, Data Studio offers two explicit intents:

- **Analyze only:** create a scoped exploration session without modifying the
  graph.
- **Add as DataInput:** create/configure a canonical `DataInput` node and use
  its registered dataset.

The UI must show source path/name, storage backend, row/column counts when
known, and whether the source is temporary or graph-backed.

### 7.2 Runtime flow

```text
File or registered node output
        |
        v
Dataset session (identity + schema + storage capability)
        |
        +--> bounded preview / profile / explore-only SQL
        |
        v
Configure a runtime-backed operation
        |
        v
Validate capability + parameters + source storage
        |
        v
Preview through the canonical runtime on a bounded derived input
        |
        +--> revise parameters
        |
        v
Publish operation chain to Node Editor
        |
        v
Execute, serialize, reproduce, train, visualize, or export
```

### 7.3 Interaction rules

- Selecting a graph-backed dataset should identify its producing node when
  lineage is available.
- Adding an operation creates a non-visual draft step, not a second canvas.
- Preview clearly states whether it uses sampled/bounded rows or the complete
  dataset.
- Publishing creates canonical nodes, parameters, links, and layout in the
  Node Editor, then opens/focuses the created graph.
- Reopening a published result should select its registered output in Data
  Studio without copying the full table.
- “Run once” may execute a transient canonical plan, but it must use the same
  executor as a persisted node.
- “Export once” and “Add export node” differ only in persistence, not export
  implementation.

## 8. Feature map

The feature list should be generated from effective capabilities. The examples
below describe the intended grouping; they do not override runtime truth.

### 8.1 Source and inspection

- Registered dataset selector
- Direct supported file open
- Schema and type view
- Paginated table preview
- Column selection and search
- Row/column counts and storage indicator
- Null/sample summaries
- Source lineage and producing-node navigation

Preview and schema inspection are workspace services and do not require new
nodes. Reproducible source creation maps to `DataInput`.

### 8.2 Query

- Ad hoc DuckDB SQL against the active dataset
- Bounded result preview
- Query validation and elapsed time
- Optional query history scoped to the project/session

Until `SQLQuery` has a real canonical runtime implementation, SQL remains
**Explore-only** and cannot be published. Data Studio must not translate
arbitrary SQL secretly into a different execution path.

### 8.3 Cleaning and transformation

Use existing runtime-backed nodes where their capability contract permits,
including examples such as:

- `FilterRows`
- `SelectColumns`
- `JoinTables`
- `GroupByAggregate`
- `SortRows`
- `FillMissingValues`
- `RemoveDuplicateRows`
- `RenameColumns`
- `SampleRows`
- `OutlierDetector`
- binning, encoding, string manipulation, polynomial features, and formula
  operations that resolve to real runtime implementations

Each editor must come from canonical parameter metadata or a focused editor
owned by that node capability. Data Studio should not reimplement transform
SQL independently.

### 8.4 Analysis

Use existing nodes when runtime-backed, including:

- `DataProfiler`
- `DataValidator`
- `DescribeStats`
- `CorrelationMatrix`
- `ValueCounts`

Lightweight schema/sample summaries may remain interactive. Any analysis that
produces a reusable artifact or is promised as reproducible should execute or
publish through its canonical node.

### 8.5 Visualization

- Interactive exploratory plots may be session-only when clearly labeled.
- Reproducible plots map to implemented visualization nodes such as
  `BarChart`, subject to storage compatibility.
- Generic/template visualization nodes must not be advertised as functional.
- Plot nodes already present in the engine should be integrated rather than
  recreated inside Data Studio.

If `BarChart` cannot consume Parquet-backed data, fix the canonical input
boundary or expose the limitation. Do not materialize the entire dataset
silently in Data Studio.

### 8.6 Export and downstream use

- Reproducible export maps to `ExportCSV`, `ExportJSON`, `ExportParquet`, or the
  canonical `DataOutput` contract.
- Training and model workflows are handed off to existing nodes and the Node
  Editor.
- Plotting uses existing plot/visualization nodes.
- “Export once” may build a transient validated plan, but must call the same
  runtime implementation.

Storage limitations in export nodes should be corrected at the canonical
executor/materializer boundary and covered by tests.

## 9. Proposed module boundaries

Names below describe responsibilities and may be adapted to repository naming
conventions during implementation. The boundaries are more important than the
class names.

### 9.1 Data Studio panel layer

Owns only rendering, user selection, presentation state, and commands. It does
not own DuckDB registration policy, operator algorithms, file parsing, graph
serialization, or background thread lifetime.

Keep Query, Analyze, Visualize, and Prepare views as focused components. Do not
grow `DataStudioPanel` into another all-in-one panel.

### 9.2 Dataset session

A scoped session owns:

- selected dataset identity;
- source/lineage identity;
- current schema and storage backend;
- a shared query context for that dataset;
- preview paging/cancellation state;
- bounded derived results and their lifetime.

This replaces three independent DuckDB connectors and prevents components
from repeatedly registering or copying the same dataset. It must define cache
bounds and invalidation when the source changes.

### 9.3 Runtime capability view

A read-only adapter combines `PipelineRuntimeCapabilities` with
`NodeMetadataRegistry` for Data Studio presentation. It owns no separate list
of nodes. Tests must prove that every advertised action resolves to a real
runtime owner, real fail mode, valid metadata, and compatible input storage.

### 9.4 Preview execution

Retain `DataPreviewService` for source previews. Add bounded operation preview
through the existing execution plan/executor boundary rather than embedding
operator code in the UI.

The preview contract must state:

- sample/full-data policy;
- maximum rows and memory;
- source storage and any materialization;
- cancellation behavior;
- output ownership and cleanup;
- deterministic parameters or seed when sampling is involved.

### 9.5 Graph publication

A focused publisher converts validated draft steps into canonical Node Editor
nodes and links. It should use existing node construction, metadata, parameter,
and serialization APIs. Publication is atomic from the user's perspective:
validation failure must not leave a partial graph.

The publisher does not execute nodes and does not maintain another graph after
publication.

### 9.6 Query service

One scoped DuckDB query service owns connection lifetime, dataset/view
registration, cancellation, and Arrow results. Query, Analyze, and Visualize
consume it. UI components must not each create their own connector.

Ad hoc SQL results remain exploration artifacts until a runtime-backed SQL node
exists.

## 10. Data, ownership, and concurrency model

### 10.1 Canonical analytical representation

Use Arrow tables and Parquet-backed datasets for tabular analysis. Do not
convert an analytical table to a float tensor or strings as an intermediate
canonical form. Convert to tensors only at an explicit ML/training boundary
that defines feature, label, dtype, null, and categorical behavior.

### 10.2 Materialization

- Arrow-backed tables may be shared by ownership-safe references.
- Parquet previews should remain row-group and column selective.
- Full Parquet-to-Arrow materialization must be explicit, bounded, and
  justified by an operation that requires it.
- Temporary derived datasets need an owner, size limit, invalidation rule, and
  cleanup path.
- UI result tables store only the displayed page or bounded result, not an
  unbounded string copy of the dataset.

### 10.3 Tasks and cancellation

Every preview, query, profile, and visualization task needs:

- one owner;
- cancellation on source change and panel/session destruction;
- a completion path that cannot update destroyed UI state;
- immutable or synchronized result publication to the UI thread;
- bounded shutdown behavior;
- no lock held across I/O, DuckDB execution, callbacks, or rendering.

Prefer the engine's structured task mechanism over independent raw
`std::thread` ownership. If the existing task system is insufficient, document
that gap before adding another concurrency abstraction.

## 11. Error and status behavior

Errors should identify the failed boundary and a user action:

- unsupported node runtime;
- unsupported storage backend;
- missing or invalid parameter;
- preview limit or memory guard;
- query parse/execution failure;
- source changed or unavailable;
- task cancelled;
- graph publication rejected;
- export destination failure.

Do not use “implemented” as a user-facing guarantee unless the selected source
and parameters pass effective capability validation. Disabled actions should
explain why they are disabled.

## 12. Implementation route

Each phase is a small vertical change with tests and a removal/migration
decision. Do not combine this work with a broad runtime rewrite.

### Phase 0: Freeze and characterize existing behavior

1. Record the Data Explorer and Data Studio entry points, features, shortcuts,
   file formats, and handoffs.
2. Add characterization tests for source preview, query results, analysis,
   visualization configuration, export, and Node Editor handoff.
3. Inventory every Data Studio action against runtime mode, fail mode,
   implementation owner, artifact type, and storage backend.
4. Capture representative baseline memory, query latency, preview latency, and
   thread count before consolidation.

Exit criteria: current behavior and unsupported cases are reproducible, and no
feature is migrated based only on its GUI label.

### Phase 1: Establish one effective capability contract

1. Add a normalized capability query over existing runtime capabilities and
   metadata; do not add another node inventory.
2. Model Runtime-backed, Explore-only, and Unavailable states explicitly.
3. Make Data Studio menus/actions derive from that query.
4. Add contract tests proving advertised operations have real runtime owners
   and compatible parameters/storage.
5. Display precise disabled reasons.

Exit criteria: no Data Studio action can claim publishability from metadata or
palette visibility alone.

### Phase 2: Unify dataset selection and bounded preview

1. Introduce the scoped dataset-session boundary.
2. Route registered Arrow and Parquet sources through `DataPreviewService`.
3. Add direct-file Analyze-only and Add-as-DataInput intents using existing
   file and node APIs.
4. Show storage, schema, source identity, preview bounds, and lineage.
5. Cancel stale preview work on dataset change.

Exit criteria: one source selector and preview path handles registered Arrow
and Parquet data without full-data UI copies.

### Phase 3: Consolidate DuckDB and exploratory tools

1. Replace the Query, Analyzer, and Visualizer private connectors with one
   scoped query service.
2. Return Arrow results and render bounded pages.
3. Migrate useful Data Explorer query, history, schema, and profile behaviors
   into focused Data Studio components.
4. Mark ad hoc SQL Explore-only.
5. Remove migrated logic from Data Explorer as each slice is validated.

Exit criteria: the same dataset is registered once per session, query results
retain schema/types/nulls, and no migrated feature has two implementations.

### Phase 4: Runtime-backed prepare and analysis previews

1. Present only compatible canonical transform and analysis operations.
2. Reuse canonical parameter definitions/editors.
3. Build a validated non-visual draft operation chain.
4. Execute bounded previews through `DataStudioExecutionPlan` and the canonical
   runtime owner.
5. Surface materialization and unsupported-storage limits before execution.

Exit criteria: preview and graph execution have matching results for the same
bounded input and parameters.

### Phase 5: Publish to the Node Editor

1. Convert a validated source and draft steps to canonical Node Editor nodes,
   parameters, pins, and links.
2. Preserve source identity and result lineage.
3. Validate the complete graph before mutation and avoid partial publication.
4. Focus the created graph and allow its output to be reopened in Data Studio.
5. Add save/reload parity tests for published graphs.

Exit criteria: a user can explore, configure, publish, run, save, reload, and
reproduce a data preparation flow without manually rebuilding it.

### Phase 6: Close canonical storage gaps

1. Define the supported storage matrix for transforms, analytics, plots, and
   exports.
2. Extend canonical executors/materializers where Parquet-backed support is
   required and can remain bounded.
3. Correct `BarChart` and export behavior at their shared runtime boundaries.
4. Keep explicit limitations where safe support is not yet available.
5. Add Arrow/Parquet parity and memory-bound tests.

Exit criteria: Data Studio needs no private full-materialization workaround for
an advertised runtime-backed feature.

### Phase 7: Retire duplicate workspaces and dead canvas code

1. Confirm all retained Data Explorer capabilities have migrated or have an
   explicit removal decision.
2. Remove Data Explorer construction, navigation, registration, and source
   files.
3. Remove obsolete Data Studio `PipelineCanvas` and private node registry files
   after proving there are no active consumers.
4. Remove duplicate DuckDB, export, cleaning, chart, and handoff paths.
5. Update user documentation and screenshots.

Exit criteria: Data Studio is the only data exploration workspace and Node
Editor is the only graph workspace.

### Phase 8: End-to-end, performance, and platform validation

Run representative flows on supported Windows, macOS, and Linux builds:

1. Direct CSV/Parquet file -> preview -> profile -> transform preview ->
   publish -> execute -> export.
2. Existing `DataInput` output -> Data Studio -> plot -> reopen result.
3. Dataset large enough to exercise Parquet row groups and materialization
   guards.
4. Invalid node, invalid parameter, unsupported storage, cancellation, source
   replacement, and failed export.
5. Save/reload and rerun a published graph.
6. E2E training flow using the prepared output as the training input.

Compare with the Phase 0 baseline. Report observed memory, thread count,
latency, copies/materializations, and platform coverage; do not claim an
optimization without evidence.

## 13. Validation plan

### Unit and contract tests

- Effective capability state for implemented, explore-only, fail-closed,
  simulated, pass-through, unknown, and storage-incompatible cases
- Alias normalization and metadata/runtime agreement
- Parameter and input arity validation
- Arrow and Parquet preview paging, projection, null counts, offset bounds,
  cancellation, and row cap
- Dataset-session invalidation and temporary-result cleanup
- Graph publication mapping and atomic failure

### Integration tests

- Same operator result from bounded preview and `PipelineExecutor`
- Data Studio draft -> published graph -> saved graph -> reloaded execution
- Arrow/Parquet behavior for each advertised operation
- One-off and persisted export parity
- Query/Analyze/Visualize sharing one dataset/query session
- Source replacement while work is in flight

### End-to-end tests

- File-originated and node-originated workflows
- Cleaning and analytical workflows
- Visualization and export workflows
- Prepared data passed to an existing training graph
- Unsupported GUI-visible node remains clearly blocked and cannot publish

### Performance and safety checks

- Resident and peak memory on representative small and large datasets
- Number and ownership of idle and active threads
- Preview/query latency and cancellation latency
- Full-table copies and Parquet materialization events
- AddressSanitizer and UndefinedBehaviorSanitizer where supported
- ThreadSanitizer for task/session lifetime tests where supported
- Compiler warnings and existing static analysis

## 14. Acceptance criteria

The consolidation is complete only when all of the following are true:

- Data Studio is the single data exploration and preparation workspace.
- Node Editor is the single visual graph workspace.
- Direct files and registered node outputs use one dataset-session model.
- Arrow and Parquet preview is bounded and type/null preserving.
- Every publishable action resolves to a real canonical runtime owner.
- Explore-only actions are visibly distinct and cannot publish.
- Unsupported GUI-visible nodes fail closed with a precise reason.
- Preview and published execution match for the same input and parameters.
- Existing transform, profile, plot, and export nodes are reused.
- No parallel Data Studio implementation of a node algorithm remains.
- Data Explorer and obsolete Data Studio canvas paths are removed after
  migration.
- E2E training and save/reload workflows pass.
- Memory/thread/latency results are measured against the baseline.
- Supported Windows, macOS, and Linux flows are validated and documented.

## 15. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Data Studio becomes another runtime | Enforce execution through canonical capability and executor boundaries |
| Metadata and runtime drift | Contract tests generated from the normalized capability query |
| Silent full-data materialization | Explicit storage capability, preview bounds, memory guards, and logging/metrics |
| Data type/null loss | Keep Arrow/Parquet canonical for analytical tables |
| UI freezes or lifetime races | Owned cancellable tasks and UI-thread result publication |
| Large-file growth returns | Keep UI, session, query, capability, preview, and publication responsibilities separate |
| Legacy behavior remains forever | Migrate in vertical slices with removal as an exit criterion |
| Cross-platform regressions | Portable core behavior plus Windows/macOS/Linux validation |
| A GUI-visible template appears functional | Fail-closed effective capability state and disabled reason |

## 16. Decisions still required

These choices should be resolved during Phase 0 or at the start of the phase
that needs them:

1. Which direct-file formats are in the first supported scope? The UI must list
   only formats that the selected loader/runtime actually supports.
2. Should exploration-only SQL history be session-scoped or project-persisted?
3. What row, byte, and duration limits define a bounded operation preview?
4. What temporary-dataset quota and eviction policy are acceptable?
5. Should a draft operation chain survive closing Data Studio before it is
   published?
6. Which Parquet-backed operations justify native streaming/row-group support,
   and which should require an explicit materialization step?
7. When should a real runtime-backed `SQLQuery` node enter scope? It is not a
   prerequisite for consolidating exploration.

## 17. Recommended first ticket slice

Start with Phases 0 and 1 only:

> Create a source-verified effective capability inventory for Data Studio,
> expose Runtime-backed, Explore-only, and Unavailable states, and prove by
> tests that no publishable action is derived from GUI metadata alone.

This slice gives later UI work a truthful foundation, introduces no duplicate
runtime, and prevents additional design debt while consolidation proceeds.
