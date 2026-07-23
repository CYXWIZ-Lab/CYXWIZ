# To Fix 71 - Production-Ready Data Studio Core and Modular Analytics Extensions

## Status

Open - architecture and planning ticket. This ticket records future work only;
it does not authorize implementation while the engine and Track70 data
contracts are still being stabilized.

## Decision statement

Make Data Studio a small, reliable core capability for inspecting,
transforming, profiling, and publishing CyxWiz datasets. Keep advanced data
analytics, specialized editors, vendor integrations, and third-party tools
optional through plugins once a stable Arrow-based data extension contract
exists.

```text
Core Data Studio
  registered DatasetAsset
    -> bounded preview and schema
    -> DuckDB query / transformation
    -> analysis and visualization
    -> immutable derived DatasetAsset + provenance
    -> Data Input / Node Editor training handoff

Optional plugin
  DatasetHandle + declared capabilities
    -> specialized source, transform, analysis, visualization, or export
    -> result/report or derived DatasetAsset
```

Data Studio remains an engine feature. Plugins extend it; they do not replace
its essential dataset workflow or own core training and provenance rules.

## Why this exists

CyxWiz already contains the main pieces of a useful Data Studio:

- Arrow-backed registered datasets;
- an in-memory DuckDB connector;
- Query, Analyze, and Visualize tabs;
- a bounded shared `DataPreviewService` for registered tabular data;
- Data Studio execution-plan and runtime-capability validation;
- derived Arrow dataset registration from query results;
- plugin lifecycle, permissions, panels, nodes, data providers, and analytics
  providers.

However, these pieces do not yet form a production-ready workflow. Some paths
are synchronous or unbounded, several UI claims exceed runtime support, old
Table Viewer editing code is disconnected from the current data boundary, and
the plugin data API cannot safely exchange real Arrow datasets.

Adding a large built-in spreadsheet or many power-analytics features now would
increase engine instability and duplicate functionality. The correct next
step is to harden the small core, make its contracts truthful, and create a
narrow extension boundary proven by one reference plugin.

## Current implementation inventory

### Core components to retain and harden

| Capability | Current component | Production direction |
| --- | --- | --- |
| Dataset identity and ownership | `DataRegistry`, `ArrowDataset`, `ParquetBackedDataset` | Keep as the engine source of truth; expose handles, not internals. |
| Bounded inspection | `DataPreviewService` | Make the common preview contract for Data Studio and other dataset surfaces. |
| SQL execution | `DuckDBConnector` | Keep as the core tabular query primitive; add lifecycle and resource controls. |
| Query UI | `DataStudioPanel::QueryEditor` | Keep, but make execution asynchronous, cancellable, bounded, and provenance-aware. |
| Profiling | `DataStudioPanel::Analyzer` | Keep a small baseline profile in core; specialized analyses may be plugins. |
| Visualization | `DataStudioPanel::Visualizer` | Keep essential charts; advanced or vendor-specific visuals may be plugins. |
| Pipeline validation | `DataStudioExecutionPlan`, runtime capability registry | Keep fail-closed validation and truthful support states. |
| Training handoff | Data Input / Node Editor dataset boundary | Use registered dataset identity; never create a second training-data system. |

### Existing paths that require reconciliation

1. `QueryEditor` accepts only registered in-memory Arrow datasets even though
   the Data Studio selector can surface broader registry entries.
2. Query execution and full-result saving are synchronous. `Save as Dataset`
   re-executes the query and can materialize an unrestricted result.
3. Display is capped after query execution, not necessarily at the query or
   scan boundary. A visible row limit is not by itself a memory limit.
4. `DuckDBConnector::RegisterTable` currently builds and appends rows into a
   DuckDB table. Documentation and UI must not claim zero-copy registration
   until an Arrow scanner/view path proves it.
5. Table Viewer retains cell editing, dirty tracking, and direct file-save
   code, but active dataset preview no longer loads through it and its current
   in-memory tabs do not own a reliable source filepath.
6. Excel Reader and Export Excel appear in metadata, while runtime capability
   checks say Excel loading/export is not implemented.
7. The old independent Data Studio pipeline canvas remains in source after the
   visible pipeline workflow moved to the Node Editor. Its ownership and
   serialization compatibility must be resolved before new work uses it.
8. Plugin analytics are registered but are not consumed by the current Data
   Studio Analyze tab.
9. Plugin nodes exchange only scalar, `vector<float>`, and string values; they
   cannot carry a DatasetAsset or Arrow table.
10. `PluginDataset` is an opaque placeholder whose engine bridge is explicitly
    unfinished. Returning C++ `shared_ptr`, `vector`, `map`, or engine objects
    across a public DLL boundary is not a stable third-party data ABI.

## Product boundary

### Data Studio core responsibilities

The engine must provide these capabilities without optional plugins:

1. Select a supported registered tabular DatasetAsset by stable identity.
2. Display truthful backing, schema, row count, label metadata, and bounded
   sample data.
3. Run safe read-only DuckDB queries and deterministic table transformations.
4. Profile missing values, uniqueness, basic distributions, and column types.
5. Provide a small set of dependable table visualizations.
6. Publish query/transformation results as immutable derived DatasetAssets.
7. Record lineage: input dataset identity/fingerprint, operation or query,
   parameters, engine/plugin version, output schema, row count, and timestamp.
8. Hand a derived dataset to Data Input / Node Editor through the existing
   registry boundary.
9. Cancel long work, report progress, enforce configurable resource limits,
   and return typed errors without freezing or crashing the engine.
10. Save and restore user-authored query/transform recipes without embedding
    transient pointers or duplicating dataset contents in project files.

Core Data Studio does not need to reproduce Excel, Power BI, Tableau, pandas,
or a complete database IDE.

### Optional plugin responsibilities

Plugins are suitable for capabilities whose dependencies, release cycles, or
audiences should not burden the core:

- Excel and proprietary file import/export;
- database, warehouse, cloud, and SaaS connectors;
- specialized table editors and data-entry workflows;
- advanced statistical, scientific, geospatial, forecasting, or domain tools;
- Python, Polars, pandas, R, or managed external-process integrations;
- advanced visualizations and dashboards;
- Power BI/Tableau-style connectors or export adapters;
- organization-specific validation and reporting.

Dataset roles, split/leakage rules, registry ownership, provenance, plugin
permissions, and training safety remain core invariants and cannot be
overridden by a plugin.

## Required core workflow

```text
Data Input / registered source
  -> Data Studio selects DatasetAsset
  -> Preview / Analyze / Visualize
  -> DuckDB query or typed transform recipe
  -> bounded result preview
  -> explicit Publish Derived Dataset
  -> atomic registry registration + lineage record
  -> optional Data Input / Node Editor handoff
```

The source remains immutable. Closing a result preview creates no registry
entry. Publishing is explicit and either succeeds atomically or leaves no
partial dataset.

## Editing and dirty-data policy

A spreadsheet-like grid can be useful, but it is not the first production
requirement and must not restore direct in-place source mutation.

The first production cleaning workflow is query/operation based:

- filter or remove rows;
- select, rename, cast, or derive columns;
- fill or flag missing values;
- deduplicate;
- join, aggregate, sort, or sample;
- publish the result as a new dataset version.

If manual cell or row correction is added later, it must use a stable row
identity and a small patch/operation log. The editor applies that log to create
a derived dataset. It must support validation, undo/redo, and provenance, and
must never silently overwrite an external CSV, Parquet file, or registered
Train/Dev/Test source.

The existing Table Viewer grid may be reused as a rendering component only
after its data ownership, paging, editing, and save semantics are separated
from the legacy `DataTable` file path.

## Production execution contract

### Query and transform execution

- Execute outside the UI render thread.
- Support cancellation and bounded progress reporting.
- Apply memory, row, result-size, and elapsed-time limits.
- Preview with explicit `LIMIT`/cursor semantics where possible.
- Do not re-run an unrestricted query merely to implement a UI action without
  showing the cost and requesting explicit publication.
- Preserve logical types and nulls through DuckDB/Arrow conversion.
- Keep one operation owner and one error channel per execution.
- Clean temporary tables, files, and registry reservations after cancellation
  or failure.

### Storage and publication

- Small results may remain in-memory Arrow.
- Large results publish through a disk-backed Arrow/Parquet path.
- Backing selection is recorded and visible.
- Publication uses a unique dataset identity and rejects accidental overwrite.
- A derived dataset fingerprint changes when inputs or transformation semantics
  change.
- Project serialization stores lineage and references, not duplicate tables.

### Truthful capability states

Every source, query feature, transform, visualization, export, and plugin must
resolve to one of:

```text
available
available with limits (explain limits)
provided by installed plugin
unavailable in this build (explain reason)
```

Unsupported Excel, database, or plugin paths must not appear as implemented
nodes merely because metadata or a dialog exists.

## Stable plugin data boundary

Do not publish the current `PluginDataset` placeholder as the production data
API. First define the smallest handle-based contract that can survive engine
and compiler upgrades.

### Required properties

1. Use opaque engine-owned dataset handles with explicit retain/release or
   request-scoped lifetime.
2. Exchange tabular data through the Arrow C Data Interface / C Stream
   Interface or another proven C ABI, not engine C++ class layouts.
3. Expose schema, row count when known, backing capability, dataset identity,
   and bounded scan requests.
4. Keep `DataRegistry` mutation private. A plugin submits a publish request;
   the engine validates and registers the result.
5. Include cancellation, progress, structured error codes, and capability
   negotiation.
6. Version data contracts independently from plugin marketing/version text.
7. Declare filesystem, network, Python, GPU, UI, and registry permissions.
8. Ensure unloading a plugin cannot leave live callbacks, dangling datasets,
   or unusable project state.
9. Prefer out-of-process workers for heavy runtimes, conflicting dependencies,
   untrusted integrations, or tools that need independent crash recovery.

### Narrow extension categories

Evolve the existing provider concepts rather than adding one broad interface:

```text
Source adapter  -> external source to engine-publishable DatasetAsset
Table transform -> DatasetHandle + parameters to derived table/recipe
Analytics       -> DatasetHandle + request to typed report/result
Visualization   -> bounded data view to optional panel/render output
Export/sink     -> DatasetHandle to external artifact with explicit permission
```

A plugin node may reference these operations, but graph pin values must not
copy entire tables into `vector<float>` or expose raw registry pointers.

## Reference plugin requirement

Before declaring the data-plugin API stable, implement exactly one reference
plugin that exercises the complete lifecycle. Excel import/export is a useful
candidate because the current engine advertises related nodes without a real
runtime, but the final choice should favor the smallest dependency footprint.

The reference plugin must prove:

- install, enable, disable, unload, and reload;
- permission disclosure;
- source capability discovery;
- bounded preview;
- Arrow schema/type/null preservation;
- cancellation and typed failure;
- derived dataset publication and lineage;
- project reopen with the plugin present and absent;
- no engine crash or dangling registry state after plugin failure/unload.

Do not add several plugin types until one vertical path is reliable.

## Reliability and security requirements

- Malformed queries, plugin metadata, Arrow streams, or schemas fail closed.
- Query and plugin errors identify the operation and dataset without exposing
  secrets or dumping sensitive rows into logs.
- External processes receive only declared inputs and scoped temporary paths.
- Network access is denied unless the plugin declares and receives permission.
- Dataset handles are read-only unless an explicit derived-output transaction
  is active.
- Plugin panels cannot bypass dataset permissions through global singletons.
- Engine shutdown cancels Data Studio work before destroying DuckDB, Arrow,
  registry, or plugin resources.

## UX requirements

The production surface remains small:

```text
Dataset selector
  Query | Analyze | Visualize
  Result Preview
  Publish Derived Dataset
  Send/Open in Node Editor
  Installed Extensions (only when relevant)
```

- Empty states tell users to load data through Data Input.
- Disk-backed or unsupported datasets show an honest capability reason.
- Long work shows progress and Cancel.
- Result preview and published output are visibly different states.
- The UI shows source identity, output identity, row counts, backing, and the
  transformation recipe before publication.
- Plugin tools are visibly attributed to their provider and show permissions
  and availability.

## Delivery phases

### Phase 0 - Audit and truth cleanup

1. Map every Data Studio action to its actual runtime owner and capability.
2. Add focused tests for current Query, Analyze, Visualize, preview, and Save as
   Dataset behavior.
3. Hide or mark unsupported Excel and other premature nodes truthfully.
4. Decide whether the old PipelineCanvas and Table Viewer editor code is
   migrated, reduced to reusable rendering, or removed.
5. Record current copy/materialization behavior and stop claiming zero-copy
   where it is not proven.

### Phase 1 - Production core

1. Filter dataset selection by supported capability.
2. Move query/analysis execution off the UI thread.
3. Add cancellation, limits, progress, typed failures, and cleanup.
4. Use the shared bounded preview contract for result inspection.
5. Make derived-dataset publication atomic and collision-safe.

### Phase 2 - Lineage and handoff

1. Define the derived dataset lineage record and fingerprint.
2. Persist query/transform recipes and dataset references safely.
3. Add explicit Node Editor handoff through the existing dataset boundary.
4. Prove project reopen and missing-source diagnostics.

### Phase 3 - Plugin data contract

1. Replace/retire the placeholder plugin dataset bridge with a stable handle
   and Arrow C data exchange contract.
2. Integrate plugin analytics and capabilities into Data Studio without giving
   plugins registry internals.
3. Define in-process versus out-of-process execution policy.
4. Build and validate one reference plugin.

### Phase 4 - Optional power tools

Only after Phases 0-3 pass acceptance, consider advanced table editing,
third-party analytics suites, additional connectors, or specialized panels as
independent plugins.

## Acceptance criteria

1. Data Studio can select every dataset it claims to support and rejects other
   modalities with one clear reason.
2. Query, Analyze, and Visualize do not block the UI during bounded work and
   can be cancelled safely.
3. Previewing or closing a result creates no orphan registry state.
4. Publishing creates exactly one immutable derived DatasetAsset with schema,
   row count, backing, fingerprint, and lineage.
5. Re-running the same deterministic recipe on the same input produces
   comparable provenance; changed input or semantics changes the fingerprint.
6. Large results follow a disk-backed path instead of requiring full in-memory
   Arrow materialization.
7. Unsupported Excel and third-party features are not presented as working.
8. No active Data Studio workflow depends on the orphan legacy Table Viewer
   loader/editor ownership model.
9. A derived dataset can be opened in Data Input / Node Editor without copying
   or bypassing the registry contract.
10. Query and plugin failures leave DuckDB, temporary storage, and registry
    state reusable.
11. The production plugin data API does not expose C++ engine object layouts or
    transfer complete tables through scalar/vector node values.
12. One reference plugin passes install, permission, execution, cancellation,
    publication, unload, and missing-plugin project tests.
13. The engine remains fully functional when no analytics plugins are
    installed.
14. Advanced plugins load only when enabled and do not add startup or runtime
    cost to the normal core workflow.

## Test plan

- Small Arrow dataset query, analysis, visualization, preview, and publish.
- Disk-backed Parquet input and large-result publication under memory limits.
- SQL parse error, execution error, cancellation, timeout, and engine shutdown.
- Null, boolean, integer, floating-point, string, timestamp, and categorical
  type preservation.
- Duplicate output name and simulated publication failure with no partial
  registry entry.
- Project save/reopen with valid, changed, missing, and disk-backed inputs.
- Data Studio to Node Editor handoff and training consumption of the published
  dataset.
- Unsupported modality and unsupported Excel capability messaging.
- Plugin absent, disabled, permission denied, failed, cancelled, unloaded, and
  upgraded scenarios.
- Reference plugin Arrow stream/schema corruption and recovery.
- Repeated open/close/query cycles with leak and handle-lifetime checks.

## Non-goals

- Building an Excel, Power BI, Tableau, or database-IDE clone into the engine.
- Replacing Data Input, DataRegistry, Arrow, Parquet, or the training dataset
  boundary.
- Allowing plugins to mutate Train/Dev/Test sources in place.
- Publishing every current C++ plugin interface as a permanent public ABI.
- Adding many connectors or analyses before one reference extension is proven.
- Restoring a second visual pipeline system beside the Node Editor without a
  separate approved architecture decision.
- Treating a row limit in the UI as proof of bounded backend execution.

## Exit condition

This ticket is complete when the plugin-free Data Studio core is dependable for
real dataset inspection, transformation, publication, and training handoff;
its claims match runtime behavior; and one optional reference analytics/data
plugin proves the stable extension boundary without weakening engine stability
or dataset provenance.
