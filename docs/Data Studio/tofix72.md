# To Fix 72 - Production HDF5 Data Input Adapter and Unified Source Capability Truth

## Status

Open - architecture and planning ticket. This ticket records future work only;
it does not authorize implementation while the current Data Input, preview,
and Track70 dataset-role paths are still being stabilized.

## Decision statement

Add HDF5 to the engine through the existing universal Data Input boundary,
using one production source adapter and one authoritative format-capability
contract.

Do not expose HDF5 in Data Input merely because HighFive is compiled or because
legacy HDF5 utilities exist. The format becomes selectable only when probe,
configuration, bounded preview, loading, registration, graph execution, and
diagnostics form one verified path.

```text
HDF5 source
  -> HDF5 source adapter
       -> probe file and compiled capability
       -> browse/select dataset paths
       -> validate rank, dtype, shape, and row alignment
       -> load a bounded or lazy DatasetAsset
  -> shared Data Preview
  -> Data Input Dataset output
  -> Data Split / Partition Policy
  -> Data Loader
```

The first production slice is numeric tabular HDF5. Arbitrary hierarchical
scientific files, Keras model files, and high-rank tensor/image datasets are
not silently treated as tables.

## Why this exists

The current engine presents conflicting HDF5 truths:

- HighFive and HDF5 are compiled into the current build;
- `DataConvertService` can read and write a limited numeric HDF5 table;
- `DataRegistry` contains a legacy `HDF5Dataset`;
- an HDF5 tree/sample preview renderer exists;
- extension detection recognizes `.h5`, `.hdf5`, and `.hdf`;
- the unified Data Input picker and Format list omit HDF5;
- `TabularLoader` explicitly rejects HDF5;
- `PipelineExecutor` does not execute HDF5 Data Input;
- the runtime-capability registry excludes HDF5 from Data Input;
- the shared registered-dataset preview supports Arrow and Parquet only.

The Format list is therefore correct to omit HDF5 today, but the engine has
enough reusable foundation to complete the support without creating another
loader stack.

## Verified current implementation inventory

### Working components to reuse

| Capability | Current component | Verified status |
| --- | --- | --- |
| HDF5 dependency | HighFive / HDF5 CMake integration | Enabled in the current build; `hdf5.dll` is deployed. |
| Numeric table conversion | `DataConvertService` HDF5 reader/writer | Test passes for 1D/2D numeric `/data` datasets and HDF5/Parquet round trips. |
| Hierarchy inspection | `HDF5Browser` | Can validate a file, browse groups/datasets, and read bounded samples. |
| HDF5 preview shell | `HDF5PreviewRenderer` | Existing tree/details/sample renderer; not connected to shared Data Input preview. |
| Legacy sample dataset | `HDF5Dataset` and `DataRegistry::LoadHDF5` | Supports data/label paths, lazy chunks, cache, and selected numeric tensor layouts. |
| Tabular runtime boundary | `ArrowDataset`, `ParquetBackedDataset`, `DataRegistry` | Existing Dataset identity and training boundary to preserve. |
| Shared bounded preview | `DataPreviewService` | Working for registered Arrow/Parquet datasets. |

### Deliberate blockers and incomplete paths

1. `DataInputDialog::RenderTabularOptions` lists only Auto, CSV, TSV,
   Parquet, Feather, and Arrow/IPC.
2. `TabularLoader::ValidateApplyContext` rejects `hdf5` before an async load.
3. `ApplyContext` has no HDF5 data-path, label-path, layout, or coercion
   contract.
4. Data Input stores a legacy `hdf5_dataset` string but does not provide a
   complete selector, restore it consistently, or pass it to the loader.
5. `PipelineExecutor::ExecuteDataInput` has no HDF5 branch.
6. `pipeline_runtime_capabilities.cpp` excludes HDF5 from the allowed Data
   Input `type` and `file_type` values.
7. `DataPreviewService` recognizes only registered Arrow and Parquet tabular
   backends.
8. Data Input's source preview fallback is a delimited-text reader and must
   never be used on an HDF5 binary.
9. The dedicated legacy `HDF5Dataset` graph node remains marked unsupported
   by PipelineExecutor.
10. Format truth is duplicated across extension detection, file-picker
    filters, dropdown options, loader validation, preview routing, and runtime
    capabilities.

## Scope decision

### First production slice: numeric tabular HDF5

Support a selected HDF5 dataset as a table when:

- the selected dataset is rank 1 or rank 2;
- its element type is a supported numeric primitive;
- rank 1 maps to one column;
- rank 2 maps to rows by columns;
- an optional label dataset is rank 1 and has the same row count;
- conversion/coercion is explicit and reported;
- loading produces the same registered Dataset artifact expected by Data
  Split and Data Loader.

The user must be able to select the data dataset path. Conventional names such
as `/data`, `/features`, `/X`, `/inputs`, or `/values` may be suggested, but
filename or path guessing never silently changes the selected source.

If a separate label dataset is selected, it must be attached as truthful
dataset metadata or an appended label column through one documented rule.
Row-count mismatch is a typed failure.

### Deferred HDF5 modes

The following require a later adapter mode or ticket:

- rank 3/4 image or tensor sample datasets;
- ragged/vlen arrays;
- compound/record types;
- strings and arbitrary object/reference types;
- multiple tables joined from separate groups;
- remote/virtual HDF5 datasets;
- Keras model `.h5` import;
- SWMR/live files;
- transparent distributed or parallel HDF5 I/O.

These files may still receive hierarchy metadata preview with a clear
`loading unsupported for this layout` reason.

## Authoritative source-format capability contract

Introduce one typed descriptor/registry used by every Data Input surface:

```text
SourceFormatCapability
  id
  display_name
  extensions
  build_available
  source_categories
  probe(source) -> supported | unsupported(reason)
  configuration_schema
  preview_capability
  load_capability
  runtime_capability
  adapter_factory
```

The exact type may stay smaller than this sketch, but one source of truth must
answer:

```text
Can this build select the format?
Can it inspect the source?
Can it preview the selected dataset?
Can it load/register a Dataset artifact?
Can a saved graph execute the source?
What limitation or missing dependency prevents each step?
```

The capability registry must drive:

- Data Input file-picker filters;
- Format Options entries;
- Auto extension resolution;
- format-specific configuration panels;
- preflight validation;
- preview routing;
- loader dispatch;
- PipelineExecutor allowed parameter values;
- documentation/support-state text.

Do not create separate UI, loader, preview, and executor lists that can drift.

## Target HDF5 adapter contract

```text
Hdf5SourceAdapter
  probe(path)
    -> HDF5 signature, build capability, dataset hierarchy summary

  list_datasets(path)
    -> path, rank, dimensions, dtype, estimated bytes

  preview(path, selection, PreviewRequest)
    -> bounded schema/sample page or typed unsupported reason

  load(path, selection, LoadPolicy)
    -> DatasetAsset or typed failure
```

For the first tabular slice, the adapter should reuse
`DataConvertService::LoadTable` or extract its tested HDF5-to-Arrow primitive
into a shared implementation. It must not copy that reader into Data Input.

`HDF5Browser` should provide hierarchy discovery and bounded sample inspection.
Its existing renderer may remain a view, but Data Input and Asset Browser must
consume the same adapter result and support state.

## Data Input UX

When HDF5 is available and the adapter is executable:

```text
Format: HDF5

File: D:\data\experiment.h5

Dataset path: /features
  shape: [60000, 170]
  dtype: float32

Optional label path: /labels
  shape: [60000]
  dtype: int32

Import mode: Numeric table
Rows: 60000
Columns: 170 (+ label)
Compatibility: Ready
```

Required behavior:

1. Auto detects `.h5`, `.hdf5`, and `.hdf` only after verifying the HDF5
   signature.
2. The selector shows dataset paths, dimensions, rank, dtype, and estimated
   size.
3. Unsupported groups/datasets remain inspectable but cannot be selected for
   a mode that cannot load them.
4. Preview is bounded and never reads the entire file merely to populate the
   dialog.
5. Apply runs asynchronously and remains cancellable.
6. The loaded backing, selected paths, row count, schema, and coercion policy
   are persisted and shown.
7. Reopening the dialog restores the exact file and dataset selections.
8. When HighFive is absent, HDF5 is unavailable with one dependency reason;
   the UI does not expose a failing selectable option.

## Dataset identity and storage

HDF5 support must produce or register the same Dataset identity consumed by
Track70 role resolution. At minimum, its source fingerprint includes:

- canonical source identity;
- file size and modification/version identity where practical;
- selected data and label dataset paths;
- selected shapes and dtypes;
- adapter/import mode and coercion policy;
- resulting schema and row count.

The first tabular slice may materialize a safe file into Arrow or a Parquet
cache using existing memory policy. Large sources must not be copied into RAM
without a resource estimate and policy decision.

The legacy lazy `HDF5Dataset` may later back high-rank sample mode, but it must
not bypass DatasetAsset identity, Data Split role resolution, schema checks,
or Data Loader ownership.

## Preview integration

HDF5 needs two bounded preview levels:

1. **Hierarchy preview** before load:
   groups, datasets, shape, rank, dtype, and limited sample values through the
   HDF5 adapter/browser.
2. **Registered dataset preview** after load:
   the same shared `DataPreviewService` contract used by other tabular
   Dataset assets.

The existing HDF5 renderer should become a presentation layer over adapter
results. It must not remain an unrelated preview path with different support
claims.

The preview must explicitly distinguish:

```text
Valid HDF5 file; selected dataset is loadable.
Valid HDF5 file; hierarchy preview available, selected layout unsupported.
HDF5 support unavailable in this build.
Not an HDF5 file despite its extension.
```

## Runtime and graph integration

Once the adapter is complete:

1. Add HDF5 fields to the copied Data Input `ApplyContext`.
2. Route HDF5 through the adapter in the async Data Input loader.
3. Register the resulting Dataset under the node's stable dataset identity.
4. Add `hdf5`/`h5` to Data Input runtime capability values.
5. Add an HDF5 branch to `PipelineExecutor::ExecuteDataInput` that invokes the
   same adapter; do not reimplement file reading there.
6. Resolve Auto by extension/signature before dispatch.
7. Preserve selected dataset paths in graph serialization.
8. Make Data Split and Data Loader consume the result without format-specific
   branches.

The dedicated legacy HDF5 node should either delegate to the same adapter or
remain explicitly unsupported until migrated. It must not become a second
behaviorally different HDF5 implementation.

## Typed failures

Failures must identify the exact boundary:

```text
HDF5 support is not compiled into this build.
The selected file does not have a valid HDF5 signature.
Dataset '/features' does not exist.
Dataset '/features' has rank 4; Numeric table mode supports rank 1 or 2.
Dataset '/features' uses unsupported dtype 'compound'.
Label dataset '/labels' has 59999 rows; data dataset has 60000 rows.
The estimated materialization exceeds the configured memory policy.
HDF5 preview was cancelled.
```

Do not fall back to CSV/Arrow parsing or report a generic `failed to read
table` when the source is HDF5.

## Relationship to existing tickets

### ToFix70

HDF5 Data Input produces a generic Dataset artifact. It does not assign or
reinterpret Train/Dev/Test roles. Role assignment and external partition
preservation remain owned by Data Split / Partition Policy.

### ToFix71

The HDF5 adapter is a proof of the production source boundary needed by Data
Studio. Core HDF5 tabular support may live in the engine because it produces a
standard Dataset artifact. Specialized scientific layouts and vendor-specific
analytics may later be plugins after the adapter/capability API is stable.

### Existing DataConvert work

DataConvert remains the transformation/export operator. Its tested HDF5 table
primitive should be reused, but Data Input must not require users to manually
convert every supported HDF5 file to Parquet before loading.

## Non-goals

- Claiming that every HDF5 file is a table.
- Supporting Keras models because they use an `.h5` extension.
- Building another dataset registry or preview service.
- Adding format-specific logic to Data Split or Data Loader.
- Duplicating the existing HighFive reader.
- Loading whole large files to generate a preview.
- Enabling the dropdown before graph execution and persistence are verified.
- Expanding the first slice to every tensor, image, scientific, or compound
  HDF5 layout.

## Acceptance criteria

1. One authoritative capability entry drives HDF5 picker, dropdown, preview,
   validation, loader, executor, and documentation truth.
2. The current build truthfully reports whether HighFive/HDF5 is available.
3. A valid rank-1 or rank-2 numeric HDF5 dataset can be selected by path,
   previewed, loaded through Data Input, registered, saved, reopened, and
   executed.
4. An optional aligned rank-1 label dataset is preserved and exposed through
   the normal Dataset label contract.
5. HDF5 Data Input emits the same Dataset pin contract as CSV/Parquet.
6. Data Split/Data Loader require no HDF5-specific behavior.
7. Preview is bounded and uses the shared adapter/support state.
8. Unsupported rank, dtype, missing path, row mismatch, invalid signature,
   dependency absence, and memory-policy failures are typed and actionable.
9. Auto detection never sends HDF5 binary content to a delimited or generic
   Arrow text parser.
10. Legacy HDF5 utilities either delegate to the shared primitive or remain
    clearly separated without conflicting support claims.
11. HDF5 is not shown as production-supported until Data Input and
    PipelineExecutor tests pass.

## Test plan

### Build and capability

- HighFive present: capability is available and HDF5 appears in supported
  Data Input formats.
- HighFive absent: capability is unavailable with one clear reason.
- Extension-only fake `.h5`: signature probe rejects it.

### Valid table inputs

- Rank-1 numeric `/data` becomes one Arrow column.
- Rank-2 numeric `/features` preserves row and column counts.
- Explicit non-default nested dataset path loads.
- Optional `/labels` with matching rows attaches successfully.
- Auto resolves `.h5` to the HDF5 adapter.

### Invalid or unsupported inputs

- Missing selected path.
- Rank-3/4 dataset in Numeric table mode.
- String, compound, object/reference, and ragged datasets.
- Data/label row-count mismatch.
- Empty dataset.
- File changes between probe and load.
- Materialization exceeds the memory policy.

### Preview and persistence

- Hierarchy preview is bounded and cancellable.
- Registered preview matches loaded schema and sample values.
- Save/reopen restores data path, label path, mode, and support state.
- Asset Browser and Data Input report the same HDF5 capability.

### Runtime

- `DataInput(type=hdf5)` executes through PipelineExecutor.
- Data Input -> Data Split -> Data Loader accepts the HDF5-derived Dataset.
- Train plus external Test HDF5 sources preserve Track70 roles.
- Unsupported HDF5 fails before asynchronous training launch.
- Existing CSV, TSV, Parquet, Feather, and Arrow/IPC paths remain unchanged.

## Delivery phases

1. Consolidate source-format capability truth and add fail-closed HDF5
   capability metadata without exposing the format.
2. Extract/reuse the tested numeric HDF5-to-Arrow primitive and define the
   HDF5 selection contract.
3. Add hierarchy probe, dataset-path selection, bounded preview, and typed
   validation.
4. Wire async Data Input load, Dataset registration, persistence, and reopen.
5. Wire PipelineExecutor to the same adapter and enable the runtime capability.
6. Enable HDF5 in Format Options only after the end-to-end tests pass.
7. Reconcile the legacy HDF5 node/registry/preview paths and remove duplicate
   support declarations.
8. Consider high-rank tensor/sample HDF5 as a separate, explicitly scoped
   adapter mode after the tabular path is stable.

## Immediate rule

Until the acceptance criteria are met:

- keep HDF5 out of the production Data Input Format list;
- preserve the current fail-closed loader/runtime validation;
- describe HDF5 conversion and hierarchy preview as partial capabilities;
- do not label the unified Data Input HDF5 path as implemented.
