# done24 - DataConvert implementation tracker

## Purpose

Track the remaining implementation work for `tofix24` after the initial generic table conversion work.

The goal is not to expand DataConvert into a broad universal converter yet. The immediate goal is to make the existing table conversion contract correct, testable, and truthful in the engine and UI.

## Current engine state

- `DataConvertService` supports CSV, TSV, Parquet, Feather, Arrow, and IPC table formats.
- Generic `Preview()` and `Convert()` APIs exist.
- Legacy CSV wrappers still exist for compatibility.
- Manifest v2 records input/output formats and conversion settings.
- The DataConvert node exposes an optional dataset input pin.
- `DataConvertOptions` already supports `input_table`.
- The service already prefers `input_table` when present.
- Pipeline runtime executes DataConvert from either configured file paths or an upstream dataset input.

## Original main gap

The DataConvert node was exposed as if it could operate from an upstream dataset, but `PipelineExecutor` did not wire that upstream dataset into `DataConvertOptions::input_table`.

That made the node behave mostly like a source/file conversion node, even though the graph contract suggested it could also act as a dataset transform/export node.

Status: fixed in the first implementation slice.

## Implementation checklist

### 1. Wire upstream dataset input

Status: implemented in first slice.

- Detect whether the DataConvert node has an upstream dataset input.
- If present, pass the upstream Arrow table into `DataConvertOptions::input_table`.
- Allow DataConvert to run without `input_path` when `input_table` is present.
- Keep file-path based conversion working unchanged.

### 2. Fix runtime validation

Status: implemented in first slice.

- `output_path` should remain required.
- `input_path` should be required only when no upstream dataset input is available.
- Runtime/compiler errors should clearly explain whether DataConvert expected an input file or an upstream dataset.

### 3. Validate output format vs extension

Status: implemented in first slice.

- If `output_format` is explicit and the output path extension disagrees, fail early with a clear error.
- If `output_format=auto`, infer from the output extension as today.
- Keep the rule table-format specific for now.

### 4. Improve DataConvert dialog behavior

Status: implemented in first slice.

- Auto-update the output file extension when the selected output format changes.
- Show a clear warning when the selected output format and output path extension disagree.
- Keep the dialog focused on supported table formats only.

### 5. Add focused tests

Status: added in first slice.

- Service test: convert from `DataConvertOptions::input_table`.
- Pipeline test: upstream dataset feeds DataConvert through its input pin.
- Validation test: explicit output format and mismatched extension fail clearly.

Validation notes:

- `test_data_convert_service` built and passed after the final cleanup.
- `test_data_convert_pipeline_input` was added, built, and passed after the final cleanup. It verifies `DataInput -> DataConvert` with no `input_path`, using the upstream dataset input pin.
- `test_pipeline_executor_operator_routing` built successfully, but the full executable exceeded the 120 second run timeout and was stopped as stale. No assertion failure was observed before timeout.
- Default Debug build was started and progressed through compilation/linking, but the wrapper process exceeded the command timeout. No compiler error was observed from this slice.

## First implementation slice

- `PipelineExecutor::ExecuteDataConvert` now resolves an upstream dataset connection and passes its Arrow table through `DataConvertOptions::input_table`.
- DataConvert can now run without `input_path` when a dataset input is connected.
- DataConvert runtime required-parameter metadata now requires only `output_path`; the executor/service handle the conditional input requirement.
- `DataConvertService::Convert` accepts in-memory Arrow table input without requiring an inferred file input format.
- `DataConvertService::Convert` rejects explicit output format and output extension mismatches before writing.
- The DataConvert dialog auto-updates output extension when the selected output format changes.
- The DataConvert dialog warns when output path extension and selected format disagree.
- DataConvert node pin text now states that an upstream table replaces the configured input path.
- DataConvert logs now label in-memory table input as `<in-memory Arrow table>` instead of an empty path.
- Focused service and pipeline tests were added for the new behavior.

## Deferred work

These are intentionally not part of this implementation slice:

- Excel adapter.
- Image/audio/video conversion.

Those should be handled as separate typed adapters with clear contracts, dependencies, tests, and UI behavior.

Excel adapter decision record:

- Excel is not implemented in the DataConvert core yet.
- Existing engine paths intentionally fail closed for Excel loading rather than pretending support exists.
- Data registry Excel loading is stubbed and reports that an additional Excel library is required.
- DataTable Excel loading is still a TODO path.
- Existing tests assert this fail-closed behavior.
- Do not expose Excel as a supported DataConvert runtime format until one real implementation path is selected and tested.

Viable Excel implementation options:

1. Use DuckDB `read_xlsx` through a bundled/offline Excel extension, if the extension can be shipped and loaded reliably without network access.
2. Add a native C++ XLSX dependency such as xlsxio/OpenXLSX for reading, with a clear write strategy if Excel export is required.
3. Keep Python/Polars/openpyxl Excel handling inside scripting utilities only; do not make the core engine depend on the Python runtime for DataConvert.

Recommended Excel next step: choose option 1 or 2 explicitly, then implement Excel as a separate adapter with small read/write tests and pipeline reload coverage.

Image/audio/video conversion decision record:

- Image/audio/video conversion should not be folded into the current table-format DataConvert adapter as a hidden special case.
- These are domain conversion contracts, not ordinary Arrow table file conversions.
- Image conversion should define folder/manifest inputs, image metadata columns, optional tensor materialization, and explicit resize/normalize settings.
- Audio conversion should define waveform or spectrogram materialization, sample-rate handling, duration metadata, and label alignment.
- Video conversion should define frame extraction policy, clip/window metadata, sampling rate, and storage strategy for frame tensors or manifests.
- These should become a follow-up Data Studio conversion track after the table adapter layer is closed.

Adapter expansion status:

- JSON/JSONL: implemented as a newline-delimited JSON object adapter. This supports `.jsonl`, `.ndjson`, and `.json` files that contain one JSON object per line. Full nested JSON array flattening is not part of this first adapter.
- TXT: implemented as a strict one-column text adapter. Text input creates a single `text` column with one row per line. Text output requires exactly one column; multi-column tables should use CSV or TSV.
- ARFF: implemented as a dense relational ARFF adapter. Numeric/real/integer attributes become double columns; other attributes become string columns. Missing values use `?`. Sparse ARFF rows and advanced nominal metadata preservation are not part of this first adapter.
- NumPy: implemented for `.npy` numeric 1D/2D arrays. Input maps 1D arrays to one `value` column and 2D arrays to `col_0`, `col_1`, etc. Output writes numeric tables as 2D little-endian float64 `.npy`. `.npz`, non-numeric dtypes, structured dtypes, and Fortran-order arrays are not part of this first adapter.
- HDF5: implemented when HighFive is available. Input reads a 1D/2D numeric dataset from `/data` or common alternatives (`features`, `X`, `x`, `inputs`, `values`). Output writes numeric tables to `/data`. Non-numeric columns, nested groups, ragged arrays, and image tensor dataset semantics are not part of this table adapter.
- DataConvert output reload: PipelineExecutor now reloads converted outputs through `DataConvertService::LoadTable`, so DataConvert-supported formats can be registered for downstream nodes through one service boundary.
- Delimited writer: manual CSV/TSV row writing was replaced with `arrow::csv::WriteCSV`. Arrow quotes headers/string values according to its writer rules.

Adapter validation:

- `test_data_convert_service` covers JSONL -> Parquet, Parquet -> JSONL, TXT -> Parquet, single-column Parquet -> TXT, ARFF -> Parquet, Parquet -> ARFF, NPY -> Parquet, Parquet -> NPY, HDF5 -> Parquet, and Parquet -> HDF5.
- `test_data_convert_pipeline_input` verifies DataConvert output reload through the pipeline path.
- `cyxwiz-engine` Debug target builds with the expanded DataConvert format list and Arrow CSV writer path.

## Completion criteria

- DataConvert works from file input and upstream dataset input.
- DataConvert validation is conditional and truthful.
- Format/extension mismatch is caught before conversion.
- Dialog output extension behavior is predictable.
- Focused service and pipeline tests cover the new behavior.
- Original table conversion behavior remains compatible.

Status: complete for the current table-format contract.
