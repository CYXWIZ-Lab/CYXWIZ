# To Fix 82 - Unified Data Output and Export Node Consolidation

## Status

Open - product architecture, compatibility migration, GUI, and runtime
consolidation ticket. No implementation is claimed by this document.

## Decision statement

CyxWiz must expose one canonical **Data Output** node for writing a Dataset to
a file. CSV, JSON, and Parquet are output formats selected in that node; they
must not remain separate first-class nodes in the Node Browser, add-node
search, or right-click menu.

Data Input and Data Output should have a consistent format-driven user
experience, but they remain two distinct node types:

- Data Input is a source: file or registered asset -> Dataset;
- Data Output is a sink: Dataset -> persisted file.

Data Input must not be reused "in reverse." Input discovery, parsing, schema
inference, preview, dataset roles, and loading are different responsibilities
from output format selection, overwrite policy, atomic publication, and write
confirmation. The shared concept is a narrow file-format contract, not a
bidirectional node with mode-dependent pins.

## Essential product outcome

A user should be able to:

1. add one **Data Output** node;
2. connect any supported tabular Dataset;
3. select CSV, JSON, or Parquet in one dialog;
4. choose or accept an engine-managed output path;
5. run the pipeline and see truthful progress, final path, row count, format,
   byte count, and failure diagnostics; and
6. reopen the graph without its output contract changing.

Adding another writable format later should add one writer capability, not a
new node type, new executor branch, new icon case, and new set of property
rules.

## Verified current truth - 2026-08-01

The generic node already exists, so this ticket is a consolidation rather than
a new feature family:

- `NodeType::DataOutput` is registered as "Data Output" with one Dataset input
  and no graph output.
- `DataOutputDialog` provides a file picker and format selection, but currently
  restores and writes only CSV and Parquet.
- `PipelineExecutor::ExecuteDataOutput` supports only CSV and Parquet.
- `ExportCSV`, `ExportJSON`, and `ExportParquet` are separately registered and
  visible, have separate node creation cases, separate truth rendering, and
  separate executor methods.
- `ExportJSON` already has a real Arrow-table writer, but that implementation
  is isolated from the canonical Data Output path.
- Projects already create `<project>/exports`,
  `ProjectManager::GetExportsPath()` owns that location, and both Node Editor
  and Data Studio pipeline execution pass it to
  `PipelineExecutor::SetExportRoot()`.
- `DataOutput` currently uses `file_type`, while runtime also accepts and
  prioritizes the legacy `format` alias. Conflicting aliases are not governed
  by one shared resolver.
- `DataOutputDialog` shows options such as header, compression, and encoding,
  but the canonical executor does not consistently consume those options.
- `SaveDataset` is a different hidden compatibility concept: it can publish an
  in-memory dataset alias and optionally write a file. It must not be silently
  redefined as the canonical sink.
- `DataConvertService` already owns several table writer primitives. Data
  Output must reuse or extract those primitives where semantics match instead
  of growing another broad conversion implementation.

Relevant implementation areas include:

- `cyxwiz-engine/src/core/node_metadata_registry.cpp`
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp`
- `cyxwiz-engine/src/core/pipeline_executor.cpp`
- `cyxwiz-engine/src/core/data_convert_service.*`
- `cyxwiz-engine/src/gui/data_io_dialogs.cpp`
- `cyxwiz-engine/src/gui/node_editor_nodes.cpp`
- `cyxwiz-engine/src/gui/node_editor_io.cpp`
- `cyxwiz-engine/src/gui/properties_truth.cpp`
- `cyxwiz-engine/src/core/data_studio_execution_plan.cpp`

## Canonical node contract

### Pins

```text
Dataset -> Data Output
```

- one required Dataset input pin;
- no output pin;
- no format-specific pins;
- no mode that turns the node into a loader or in-memory dataset publisher.

The runtime may retain an internal dataset reference while executing the sink,
but the canvas contract must not advertise downstream data flow from a node
with no output pin.

### Canonical parameters

| Parameter | Contract |
|---|---|
| `file_path` | Selected file or an empty value requesting an engine-managed path under the project exports directory. |
| `file_type` | Canonical normalized writer ID. Initial required values: `csv`, `json`, `parquet`. |
| `overwrite` | Explicit overwrite policy. Existing files must not be replaced when false. |
| `configured` | UI state only; never a substitute for runtime validation. |

Compatibility aliases:

- accept legacy `path` as an alias for `file_path`;
- accept legacy `format` as an alias for `file_type`;
- normalize compatible aliases during graph migration;
- fail closed with an actionable error when two concrete aliases disagree.

The alias resolver should follow the same principle as
`ResolveDataInputFormatAliases`: one canonical value after migration, no
silent last-writer-wins behavior.

### Format-specific options

The dialog must show an option only when the selected writer implements it.
The initial contract is deliberately small:

| Format | Required semantics | Options shown only when honored |
|---|---|---|
| CSV | One table written as delimited rows. | Header and delimiter if the writer consumes them. |
| JSON | Preserve legacy `ExportJSON` array-of-row-objects behavior during migration. | JSON layout only if more than one layout is implemented and tested. |
| Parquet | One Arrow table written as Parquet. | Compression only when forwarded to the Parquet writer. |

Encoding, compression, header, delimiter, manifest, and similar controls must
not be decorative. An unimplemented option is hidden or disabled with an
explicit reason; it is never serialized as if it affected the output.

## Format and path truth

- New graphs serialize an explicit `file_type`; runtime does not guess a
  different format after the graph is saved.
- The file picker filters and suggested extension follow the selected format.
- A recognized extension that conflicts with `file_type` produces a clear
  validation error or an explicit user-confirmed correction. It must not write
  Parquet bytes to a `.csv` file or silently change the graph contract.
- When `file_path` is empty, the project export root supplies a sanitized
  filename and the correct extension.
- Parent-directory creation, permission failures, existing-file policy, and
  unsupported-format failures are reported before a successful completion is
  published.

Input and output format sets need not be identical. A format appears in Data
Output only when a production writer exists in the current build. Read support
alone is not write support.

### Existing project exports folder is authoritative

This ticket must preserve and complete the export-root behavior that already
exists. It must not introduce another `output`, `exported`, cache, artifact, or
working-directory destination.

- Project creation continues to create `<project>/exports`.
- `ProjectManager::GetExportsPath()` remains the single GUI/project source of
  truth for the default output directory.
- Node Editor and Data Studio continue to pass that value through
  `PipelineExecutor::SetExportRoot()`.
- An empty `file_path` resolves to
  `<project>/exports/<sanitized-dataset-name>.<selected-extension>`.
- An explicit user-selected path remains explicit and is not silently moved
  under the project.
- Headless execution may use the existing temporary
  `<temp>/cyxwiz/exports` fallback only when no project export root was
  supplied. A GUI project run must never fall back to the process working
  directory.
- Successful output should refresh or notify the existing project/Asset
  Browser workflow so the new file becomes discoverable without creating a
  second asset copy.
- The completion result and log must state whether the destination was
  project-managed or explicitly selected.

## Runtime architecture

Introduce or extract one narrow Dataset writer boundary used by canonical
Data Output and by compatible conversion paths:

```text
Data Output node
  -> resolve canonical path + writer ID + verified options
  -> Dataset writer service
       -> CSV writer
       -> JSON writer
       -> Parquet writer
  -> atomic publish
  -> typed result + task/log event
```

The writer capability must own, in one place:

- stable writer ID;
- supported extensions;
- availability in the current build;
- supported option keys and defaults;
- write operation and result;
- user-facing unsupported reason when unavailable.

Do not create a general plugin framework in this ticket. Three verified
built-in writers behind one small typed boundary are sufficient. Future
plugins may contribute writers only after a separate governed writer-extension
contract exists.

Data Output should reuse writer primitives from `DataConvertService` where the
file semantics are identical. It must not call the full conversion workflow if
that would accidentally add conversion-cache, freshness, manifest, or input-
file behavior to a simple Dataset sink. Extract the shared write primitive
instead of coupling the two product operations.

## Safe publication and progress

Large exports are production tasks, not UI-thread work.

- Execute through the existing pipeline task and Task View contract.
- Report at least preparing, writing, finalizing, completed/failed/cancelled.
- Write to a temporary sibling file and rename only after successful close and
  validation where the writer supports atomic replacement.
- A failed or cancelled export must not publish a partial file under the final
  name.
- Preserve the prior final file when overwrite is enabled but the replacement
  fails before atomic publication.
- Log node identity, writer ID, resolved destination, rows, columns, bytes,
  elapsed time, overwrite decision, and terminal status without logging row
  contents.

## Legacy graph migration

The format-specific node types remain readable during a compatibility window,
but they are no longer offered for new graphs.

| Legacy node | Canonical migration |
|---|---|
| `ExportCSV` | `DataOutput(file_type=csv)` |
| `ExportJSON` | `DataOutput(file_type=json)` with legacy array-of-records semantics preserved |
| `ExportParquet` | `DataOutput(file_type=parquet)` |

Migration requirements:

- preserve node ID, position, name, description, input link, and explicit
  output path;
- preserve effective overwrite and format-specific behavior;
- emit a single migration log/diagnostic explaining the canonical replacement;
- serialize the canonical node after an explicit save;
- never fall through to `Dense` or another unrelated node when loading a
  legacy type;
- keep legacy runtime dispatch only until migration tests prove behavioral
  parity, then route aliases through the canonical Data Output executor;
- do not remove enum/string mappings until the supported compatibility window
  is formally closed.

`SaveDataset` stays a documented hidden compatibility alias in this ticket.
Its in-memory publishing behavior requires a separate decision and must not be
lost as collateral cleanup.

## GUI requirements

- Node Browser, search, and right-click add menu show only **Data Output** for
  tabular file export.
- Search keywords include CSV, JSON, Parquet, export, save, write, and output,
  so users still find the canonical node by format name.
- Configure and double-click open the existing Data Output dialog.
- The dialog uses engine theme controls and one format selector.
- Changing format updates only the relevant option section and file filter.
- The node summary shows resolved format and destination without duplicating
  the user description.
- Properties truth and compile diagnostics use the same writer capability
  source as the dialog and runtime.

## Phased implementation

### Phase 1 - One contract and writer truth

- Add a shared Data Output alias resolver and typed writer capability table.
- Make CSV, JSON, and Parquet available through `DataOutput`.
- Move or extract JSON writing into the shared writer boundary.
- Make every visible option reach the selected writer or remove it from the
  visible contract.

### Phase 2 - Legacy normalization

- Normalize `ExportCSV`, `ExportJSON`, and `ExportParquet` to Data Output in
  the execution plan/graph migration boundary.
- Preserve legacy behavior with focused round-trip and output-content tests.
- Remove the three format-specific nodes from creation surfaces while keeping
  old graph loading safe.

### Phase 3 - Production write workflow

- Add atomic final-file publication, truthful overwrite handling, cancellation,
  and progress/result telemetry.
- Verify engine-managed default paths under the project export root.
- Reconcile metadata, dialog, Properties, compiler validation, runtime
  capability, and documentation from one format source.

### Phase 4 - Simplification audit

- Remove obsolete duplicate executor branches only after alias parity tests
  pass.
- Remove duplicate format lists and format-specific GUI truth branches.
- Confirm no new dependency or broad framework was introduced.
- Record any additional writer format as a separate capability addition, not a
  new export node.

## Acceptance criteria

1. New-node surfaces expose one Data Output node and no separate CSV, JSON, or
   Parquet export nodes.
2. One Data Output node successfully writes CSV, JSON, and Parquet with verified
   content and extension.
3. The dialog, metadata, validation, executor, and logs agree on the supported
   formats and options.
4. Conflicting `file_type`/`format` or `file_path`/`path` aliases fail closed or
   migrate deterministically; no silent precedence remains.
5. Existing graphs containing all three legacy export nodes load and execute
   with byte- or semantic-equivalent output as appropriate.
6. Legacy JSON graphs retain array-of-row-objects behavior unless an explicit
   versioned migration says otherwise.
7. `SaveDataset` compatibility behavior is unchanged.
8. Missing paths use exactly `ProjectManager::GetExportsPath()` and therefore
   land in the existing `<project>/exports` folder; explicit paths are honored;
   an extension/format mismatch is actionable.
9. Overwrite false preserves an existing file. Failure or cancellation never
   publishes a partial final file.
10. Large exports run through Task View without freezing the UI and expose a
    truthful terminal state.
11. Adding a test-only fourth writer requires one writer registration and
    generic UI/runtime tests, not a new `NodeType` or executor method.
12. Supported old graphs remain loadable, and saving a migrated graph emits the
    canonical Data Output representation.

## Required verification

- metadata and add-surface tests proving only canonical Data Output is offered;
- dialog/property contract tests for conditional format options;
- pipeline executor tests for CSV, JSON, and Parquet through Data Output;
- legacy graph load/migrate/save round trips for all three old node types;
- JSON semantic-parity fixtures including nulls, strings requiring escaping,
  booleans, integers, and floating-point values;
- path alias, format alias, conflict, extension mismatch, default path, parent
  creation, overwrite, permission failure, cancellation, and atomic-publication
  tests;
- GUI/project tests proving an empty path writes into the existing project
  `exports` directory, never the current working directory or a second export
  folder;
- full Release engine build and manual GUI verification of one Data Output node
  writing each supported format.

## Non-goals

- Reusing Data Input as an output node or adding an input/output mode switch.
- Making every readable format writable without a verified writer.
- Redesigning `SaveDataset`, database export, model export, checkpoint export,
  or report export.
- Adding Excel or SQL output before their writer/runtime contracts exist.
- Building a broad plugin API for writers.
- Keeping duplicate visible nodes merely as shortcuts; search keywords and
  format presets belong on the canonical node.

## Recommended first slice

Make canonical Data Output write CSV, JSON, and Parquet through one shared
writer boundary, with `file_type` as the canonical format parameter. Then hide
the three legacy nodes from creation surfaces while retaining load-time
normalization and parity tests. This delivers the user-visible simplification
before attempting additional formats.
