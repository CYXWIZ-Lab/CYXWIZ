# Dashboard To Fix 84 - Graph-Connected Data Dashboard Node

## Status

Open - product design, typed node contract, bounded data-query, deterministic
dataset-audit presentation, and GUI implementation ticket. No Dashboard node
implementation is claimed by this document.

## Decision statement

CyxWiz should provide a **Dashboard** node that connects to a prepared Dataset
and stores a persistent, read-only presentation made from several widgets.

The Dashboard node is an introspection/presentation sink:

```text
Prepared Dataset
      -> Dashboard
           -> KPI cards
           -> bounded table
           -> line/bar charts
           -> scatter plot
           -> histogram
```

It does not clean data, create model features, train a model, or change the
connected dataset. Transformations and durable filtering remain upstream graph
operations.

The node is also distinct from existing runtime panels:

- **Dashboard node:** presents the Dataset connected at one graph location.
- **Training Dashboard:** monitors a supervised/sequence training run,
  including losses, metrics, warnings, checkpoints, and terminal state.
- **RL Training Dashboard:** monitors episodes and policy diagnostics.
- **Studio Debugger:** inspects graph execution, traces, memory, device
  placement, failures, and recommendations.

The initial Dashboard must use existing ImGui, ImPlot, Arrow, DuckDB, and table
rendering foundations. It must not introduce an embedded browser, arbitrary
HTML/JavaScript execution, or a second analytics engine.

## Why this ticket exists

CyxWiz can inspect tables and render individual plots, and it has strong
training/debugging dashboards. It does not currently have a general graph node
that lets a user save several coordinated views of a prepared dataset.

Without a canonical Dashboard node, users must repeatedly open independent
inspection tools or export data to another application to assemble a compact
analysis view. A graph-connected dashboard provides persistent context:

- which dataset stage is being presented;
- which columns and aggregations are used;
- which filters are active for presentation;
- how widgets are arranged; and
- whether the displayed results still match the current dataset identity.

This is useful only if it remains bounded and truthful. A dashboard that scans
an entire large dataset on the render thread, silently samples data, or displays
stale results is worse than an explicit table/plot tool.

## Verified current truth - 2026-08-01

- CyxWiz has a canonical supervised Training Dashboard/plot panel.
- CyxWiz has a separate RL Training Dashboard panel.
- Studio Debugger, Task View, Data Preview, Table Viewer, Arrow, DuckDB, and
  ImPlot provide reusable foundations.
- The Script Editor large-file path establishes a shared responsiveness
  pattern: files above the editable-buffer limit are indexed sequentially on a
  worker, represented by sparse line checkpoints, and viewed through bounded
  read-only pages instead of being copied into a full editor document.
- The repository contains future dashboard concepts, including web/HTML ideas,
  but no implemented general graph-connected Dashboard node was found.
- Existing architectural guidance classifies persistent graph inspection as a
  Category-2 introspection tool: it may have a node form and a full panel while
  sharing one read-only backend.

Relevant areas to reuse rather than duplicate include:

- Data Preview/Table Viewer bounded table rendering;
- Data Registry dataset identity and schema access;
- DuckDB query and aggregation support;
- ImPlot rendering utilities;
- project graph serialization;
- AsyncTaskManager and Task View;
- LargeTextFile sparse indexing and bounded page reads for raw-text viewing;
- project/Asset Browser refresh and stale-data conventions.

## Canonical node contract

### Pins

Version 1 has the smallest useful contract:

```text
Dataset -> Dashboard
```

- one required `Dataset` input pin;
- no output pin;
- no generic `Any` pin;
- no model, loss, metric, or training-control pins;
- no dynamic multi-source pins in the first version.

A dashboard can contain several widgets, but every version-1 widget reads the
same connected dataset. Users can add another Dashboard node for another
dataset. Multi-source dashboards require a later typed-source design and must
not be simulated through string dataset names.

### Execution behavior

- Dashboard is skipped during automated model training and normal data
  transformation execution unless an explicit dashboard refresh is requested.
- Double-click or Configure opens the full Dashboard workspace.
- Opening the view resolves the upstream dataset identity from the executed
  graph/registry.
- If the upstream dataset has not executed, was unloaded, or no longer matches
  the saved schema identity, the dashboard shows an actionable stale/unavailable
  state rather than old charts.
- Dashboard queries are read-only and cannot mutate registry data.
- Presentation filters affect only dashboard queries. A user who wants filtered
  output for downstream nodes must add a Filter/Select/Aggregate node upstream.

### Saved configuration

The graph stores a versioned dashboard specification, not rendered pixels or a
copy of the dataset.

Minimum saved fields:

- `spec_version`;
- dashboard title and optional description;
- ordered widget definitions with stable widget IDs;
- layout position and size in a bounded grid;
- widget type;
- column bindings;
- aggregation and grouping configuration;
- sort and presentation-filter configuration;
- display labels, number formatting, and chart-axis settings;
- last compatible schema fingerprint as a validation hint, never as proof that
  the live dataset still exists.

The specification should serialize as a typed/versioned JSON object within the
graph node contract for portability. Cached query results and thumbnails do
not belong in graph JSON.

## Initial widget set

Version 1 should implement only widgets that reuse proven rendering/query
primitives:

| Widget | Minimum contract |
|---|---|
| KPI card | One numeric aggregation: count, sum, mean, minimum, or maximum. |
| Data table | Bounded, lazy rows and selected columns using the shared table renderer. |
| Line chart | Ordered x column plus one or more numeric y series with bounded points. |
| Bar chart | Category/group plus one numeric aggregation. |
| Scatter plot | Two numeric columns with bounded points. |
| Histogram | One numeric column with validated bin count/range. |
| Filter control | Local equality/range/category selection that constrains dashboard queries only. |

Widgets not backed by a real query/rendering contract remain unavailable. Do
not add maps, arbitrary Vega specifications, custom scripts, pivot editors,
HTML blocks, or plugin-defined widgets in the initial slice.

## Dashboard workspace UX

The node remains compact on the canvas. It should show:

- node name;
- connected/unavailable/stale state;
- widget count;
- dataset summary when resolved; and
- a concise instruction to double-click or Configure.

The full workspace provides:

- dashboard title and dataset/schema status;
- widget palette;
- grid layout with move and resize;
- selected-widget configuration panel;
- refresh and cancel controls;
- active local filters;
- query/task status;
- clear validation messages for incompatible column bindings; and
- one primary page scroll without nested regions hiding critical status.

Creating a widget should follow a predictable flow:

1. choose widget type;
2. choose columns;
3. choose an allowed aggregation/grouping where required;
4. validate against the current schema;
5. preview with a bounded query;
6. place the widget; and
7. save the graph specification.

The UI follows the engine theme and uses the same column naming, null handling,
numeric formatting, and task states as Data Preview/Table Viewer.

## Data-query architecture

Dashboard rendering must not copy or iterate an entire dataset every frame.

```text
Dashboard specification
  -> validate bindings against registered schema
  -> build typed read-only widget query
  -> Arrow compute or DuckDB aggregate/page
  -> bounded widget result
  -> ImGui/ImPlot/table renderer
```

The shared query layer should return a typed result envelope containing:

- dataset identity and schema fingerprint;
- widget ID and specification hash;
- query kind;
- bounded rows/points/categories;
- sampled versus exact status;
- total/result counts when known;
- warnings, truncation reason, and timing;
- terminal status: completed, cancelled, stale, or failed.

Use Arrow compute for small direct projections where it is already sufficient
and DuckDB for grouping/aggregation where that is the established engine path.
Do not add a dashboard-only data engine.

## Large-data and responsiveness contract

- All non-trivial widget queries execute outside the render thread.
- Query work appears in Task View or a shared child-task contract.
- Closing the dashboard, changing a binding, replacing the source, or starting
  a newer refresh invalidates stale results safely.
- Tables page lazily and never materialize all rows for display.
- Scatter and line plots use an explicit bounded point policy.
- Category widgets cap cardinality and expose Other/truncation truth rather
  than producing thousands of unreadable bars.
- Exact aggregate results are labeled exact. Sampled or approximate results
  are labeled with the sampling policy.
- Cancellation cannot replace a previously accepted widget result with a
  partial result.
- Rendering uses immutable result snapshots published back to the UI thread.

### Sequential streaming versus lazy viewing

These are complementary contracts and must not be treated as synonyms:

- **Sequential streaming** reads source bytes/rows once in order to compute a
  complete index or deterministic audit without materializing the source file.
- **Lazy viewing** seeks to bounded pages and retains only the rows/lines needed
  by the visible viewport and a small cache.

The production Asset Browser right-click/open flow for a large text file should
combine both:

1. inspect inexpensive file metadata synchronously;
2. open a read-only large-file tab immediately;
3. fetch and show the first bounded page on a worker;
4. continue sparse indexing sequentially in the background;
5. expose honest indexing progress and cancellation in Task View;
6. load later visible pages on demand as the user scrolls or jumps; and
7. never construct a full editable text buffer, syntax-highlight the entire
   file, or mutate UI document state from a worker.

A full-file audit may share the same sequential pass when explicitly requested,
but opening a file merely to view it must not silently run expensive numeric or
domain analysis. Viewer indexing and dataset profiling remain separate typed
operations even when they reuse low-level streaming primitives.

## Deterministic dataset audit and assisted interpretation

The MetroPT-3 intake demonstrates a production capability that the Dashboard
should eventually present, while not owning its computation:

```text
Raw/registered dataset
  -> deterministic Dataset Audit service
       -> versioned audit result/artifact
            -> Dashboard data-health presentation
            -> optional AI explanation/recommendations
```

The audit service, not the Dashboard and not an AI model, computes source truth.
A bounded sequential scan may produce:

- source identity, byte size, rows scanned, and scan completeness;
- observed schema, column count/types, duplicate names, and conversion errors;
- malformed-row and missing/null counts;
- numeric ranges and bounded cardinality summaries;
- timestamp validity, ordering, duplicate timestamps, observed cadence, and
  gap distribution;
- annotation/event interval coverage when a typed annotation source is
  supplied;
- contradictions between declared metadata and observed data;
- cancellation, truncation, sampling, failure, and timing truth; and
- a versioned audit-method identity so results remain reproducible.

The Dashboard consumes the immutable audit result. It may render data-health
cards, schema/null tables, time-gap timelines, event-coverage summaries, and
links to the affected columns or time ranges. It must not rescan the raw source
from a render function or maintain a dashboard-only profiler.

An optional AI assistant consumes the same typed audit result and project brief
to explain consequences and propose graph steps. For example, it may recommend
chronological splitting or preventing windows from crossing detected gaps. It
must distinguish:

- facts computed by the audit;
- metadata supplied by a README, schema, or annotation file; and
- recommendations inferred from those facts.

The assistant cannot invent row counts, column types, failure intervals,
metrics, or scan completeness. Each important statement should reference the
audit field or supplied project source that supports it.

## Cache and freshness

Dashboard query caching is optional and must be narrow.

A reusable cached result requires all of:

- current dataset content identity;
- schema fingerprint;
- widget specification hash;
- filter hash;
- query-engine/version identity where it affects semantics.

Changing the upstream dataset, schema, widget binding, aggregation, filter, or
query version invalidates the result. The cache stores bounded widget results,
not an ungoverned duplicate of the source dataset.

## Schema and validation truth

- KPI, histogram, scatter, and numeric axes require numeric columns.
- Category/group bindings accept compatible scalar/category columns.
- Line charts require an explicit ordering column; time-like columns receive
  time formatting only when their registered type supports it.
- Missing columns produce a per-widget invalid state without crashing the
  entire workspace.
- Null handling is explicit per widget/query and shown in result metadata.
- Duplicate column names, unsupported nested types, non-finite values, and
  excessive cardinality receive actionable diagnostics.
- Renaming or replacing an upstream column never silently rebinds a widget to a
  different column by position.

## Relationship to existing panels and tools

### Training Dashboard

Training Dashboard remains owned by the active/persisted training run. It does
not require a Dataset pin and must not be converted into the general Dashboard
node. A future explicit Run Metrics source may feed a presentation dashboard,
but that requires a typed run-result contract rather than direct access to live
panel state.

### RL Training Dashboard

RL episode and policy diagnostics remain an execution monitor. The general
Dashboard cannot assume supervised metrics or RL episode semantics.

### Studio Debugger and Task View

Dashboard may link to task/debug records for a failed widget query, but it does
not duplicate timelines, memory traces, crash reports, or task ownership.

### Table Viewer, Data Preview, and visualization tools

Dashboard reuses their bounded query/rendering components. It groups several
saved views into one workspace; it must not fork another table renderer or
implement a competing plotting library.

## Extension boundary

The initial built-in widget set should be a small typed registry containing:

- stable widget ID;
- required binding roles;
- supported column types;
- supported aggregations/options;
- query builder;
- bounded result type; and
- renderer.

This registry prevents a growing switch across dialog, validator, query, and
renderer code. It is initially internal. A public plugin widget API is a later
ticket requiring permission, resource, serialization, compatibility, and
crash-isolation contracts.

## Implementation phases

### Phase 1 - Typed read-only node and saved specification

- Add canonical Dashboard metadata and one Dataset input/no output pins.
- Add string/enum serialization mappings from the same metadata source where
  possible.
- Define versioned dashboard/widget specifications and schema validation.
- Open one full Dashboard workspace keyed by project graph and node ID.
- Prove save, close, reopen, rename, duplicate, and delete behavior.

### Phase 2 - Shared query/result boundary

- Add typed widget queries over a registered Arrow/Parquet dataset identity.
- Implement stale request invalidation, cancellation, and immutable result
  publication.
- Reuse Data Preview paging for table results.
- Add exact/sampled/truncated metadata and Task View visibility.

### Phase 3 - Minimal widgets

- Implement KPI card and bounded table first.
- Add histogram and scatter plot.
- Add line and bar charts with explicit ordering/grouping.
- Add presentation-only filter controls shared across dashboard widgets.

### Phase 4 - Production hardening

- Add bounded layout editing and configuration validation.
- Add query cache identity and invalidation if measurements justify it.
- Test Arrow and disk-backed Parquet sources.
- Test large-data responsiveness, cancellation, schema replacement, and project
  reopen.
- Audit for reused rendering/query components and remove any duplicate path
  created during implementation.

### Phase 5 - Dataset health and assisted interpretation follow-up

- Define a versioned deterministic Dataset Audit result contract independent of
  Dashboard rendering.
- Reuse bounded source streaming and existing registry/schema truth rather than
  adding a dashboard-only scanner.
- Add a read-only data-health view over persisted/current audit results.
- Add explicit provenance for observed facts, supplied metadata, and inferred
  recommendations.
- Permit an AI assistant to explain the typed result only after deterministic
  results are available.
- Validate on MetroPT-3: large-file responsiveness, observed cadence versus
  declared cadence, timestamp gaps, binary cardinality, and event coverage.

This phase is tracked by To Fix 84 but is not a blocker for the recommended
version-1 KPI-card and bounded-table vertical slice.

## Acceptance criteria

1. Node Browser, search, and right-click creation expose one canonical
   Dashboard node.
2. The node has one required Dataset input and no output pin.
3. Double-click and Configure open the same full Dashboard workspace.
4. A graph saves and reloads dashboard title, widget definitions, layout,
   bindings, aggregation, filters, and specification version.
5. Version 1 supports KPI card, bounded table, line chart, bar chart, scatter
   plot, histogram, and local filter controls.
6. Every widget validates column bindings against the current registered
   schema before querying.
7. Arrow and disk-backed Parquet sources produce equivalent widget semantics.
8. Large datasets do not block the render thread or materialize all rows for a
   table/plot.
9. Query progress/cancellation/stale/failure states are visible and truthful.
10. Sampled, truncated, and exact results are labeled correctly.
11. Replacing or unloading the source produces an unavailable/stale state and
    cannot continue displaying an unlabeled old result.
12. Dashboard filters never mutate the source Dataset or affect downstream
    graph execution.
13. Existing Training Dashboard, RL Dashboard, Studio Debugger, Task View, and
    Data Preview behavior remains unchanged.
14. No browser/WebView, arbitrary HTML/JavaScript, new plotting framework, or
    dashboard-only data engine is added.
15. Focused tests and a full Release build pass, followed by manual verification
    on one small dataset and one disk-backed/large dataset.

### Deferred dataset-health acceptance

1. Audit results state whether they are complete, sampled, truncated,
   cancelled, stale, or failed.
2. Reopening a project can reuse an audit only when source identity and audit
   method identity still match.
3. Dashboard data-health widgets read a typed audit result and never rescan the
   source from the render thread.
4. AI explanations separate computed facts, supplied metadata, and inferred
   recommendations with traceable provenance.
5. Opening a large raw text file remains responsive, displays a bounded first
   page, indexes in the background, and never creates a full editable buffer.

## Required verification

- metadata and creation-surface contract tests;
- pin and serialization round trips;
- node duplicate/delete/project-close cleanup tests;
- schema binding tests for numeric, categorical, time-like, missing, renamed,
  duplicate-name, null-heavy, and unsupported nested columns;
- widget query/result tests for every initial widget;
- Arrow/Parquet parity tests;
- pagination, point cap, cardinality cap, sampling/truncation truth tests;
- cancellation and stale-result race tests;
- graph reopen with source available, unavailable, and schema-changed;
- UI-thread ownership tests or deterministic harness coverage for result
  publication;
- Release engine build and manual dashboard construction workflow.

## Non-goals

- Replacing Training Dashboard, RL Dashboard, Studio Debugger, Task View, Data
  Preview, or Table Viewer.
- Transforming, cleaning, joining, or permanently filtering the source data.
- Supporting multiple datasets in one version-1 Dashboard node.
- Arbitrary SQL entry in the first version.
- HTML/JavaScript widgets, embedded web pages, or a web dashboard server.
- Plugin-defined widgets before a separate governed extension contract.
- Real-time collaboration, cloud publishing, presentation sharing, PDF/report
  export, scheduled refresh, or alerting.
- Treating dashboard rendering as part of model training execution.

## Recommended first slice

Implement one Dashboard node with a versioned saved specification, one Dataset
input, and a full workspace containing only a KPI card and bounded table. Reuse
registered dataset identity, Data Preview paging, AsyncTaskManager, Task View,
and existing table rendering. Once that vertical path survives project reopen,
schema replacement, cancellation, and a disk-backed source, add charts through
the same typed widget/query boundary.
