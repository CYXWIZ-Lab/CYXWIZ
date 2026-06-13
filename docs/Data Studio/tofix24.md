# tofix24 - DataConvert Node Reopened

## Status

`done24.md` was not actually complete. The node existed, but it was still a
Phase 1 `CSV/TSV -> Parquet` implementation in the service, PipelineExecutor,
runtime capabilities, node metadata, dialog text, and properties panel.

This task is reopened as `tofix24.md` until the remaining DataConvert work is
finished and verified.

## Implemented In This Reopen

- Generic `DataConvertService::Preview` and `DataConvertService::Convert`.
- Backward-compatible `PreviewCsv` and `ConvertCsvToParquet` wrappers.
- Input support for supported table file types:
  - `csv`
  - `tsv`
  - `parquet` / `pq`
  - `feather` / `fea`
  - `arrow`
  - `ipc`
- Output support for supported table file types:
  - `csv`
  - `tsv`
  - `parquet` / `pq`
  - `feather` / `fea`
  - `arrow`
  - `ipc`
- Full row-writing for CSV/TSV output inside DataConvert instead of relying on
  the older header-only `ArrowDataset::ExportCSV` helper.
- Parquet output keeps compression and row group size options.
- Feather, Arrow, and IPC output use Arrow IPC file writing.
- Manifest version `2` records resolved input/output formats and includes those
  formats in the freshness hash.
- PipelineExecutor now calls the generic converter and reloads generated output
  through `ArrowDataset::FromFile`.
- Runtime capabilities now allow `auto`, `csv`, `tsv`, `parquet`, `feather`,
  `arrow`, and `ipc` for DataConvert input/output formats.
- Node metadata, node defaults, compact properties, and the DataConvert dialog
  now expose generic data file conversion instead of hard-coded Parquet wording.
- Service test coverage now includes Parquet to CSV and Parquet to Feather
  conversions in addition to the original CSV/TSV to Parquet checks.

## Still Pending

- Manual GUI smoke test through the running app.
- Decide whether DataConvert should consume the optional input dataset pin during
  PipelineExecutor runs. The current behavior still uses `input_path`.
- Decide whether `DataConvert` should remain classified as a source node while
  also exposing an optional input pin.
- Add user-facing validation when explicit output format and file extension
  disagree.
- Add extension auto-update when the user changes output format after choosing
  an output path.
- Consider replacing the manual CSV/TSV writer with `arrow::csv::WriteCSV` if
  the bundled Arrow version exposes a stable writer API.
- Add the remaining data-file extensions that are visible elsewhere in the
  engine but are not supported by DataConvert yet:
  - HDF5: `.h5`, `.hdf5`, `.hdf`
  - ARFF: `.arff`
  - Excel: `.xlsx`, `.xls`
  - JSON tables: `.json`, `.jsonl`
  - Text tables: `.txt`
  - NumPy arrays: `.npy`, `.npz`
- For each remaining extension, first confirm the engine has a real table loader
  and writer path. Metadata/UI visibility alone is not enough to mark the format
  supported.
- HDF5 needs dataset-path selection because one file can contain many datasets.
- ARFF needs a parser decision and type/null handling before it can be converted
  safely.
- Excel needs sheet selection and header/range options.
- JSON/JSONL needs a record-orientation policy.
- NumPy needs shape-to-table rules for 1D/2D arrays and rejection for unsupported
  tensor shapes.
- SQLite/DuckDB query-to-table conversion remains future scope and is not a
  file-extension conversion yet.

## Verification Checklist

- [x] `test_data_convert_service` passes.
- [x] `test_pipeline_executor_operator_routing` passes or the DataConvert
      section is updated for new format behavior.
- [x] Main app build passes.
- [ ] Manual GUI smoke test:
  - CSV to Parquet
  - Parquet to CSV
  - Parquet to Feather
  - TSV preview with tab delimiter
  - Freshness skip with manifest enabled
- [ ] Add focused tests before marking any additional extension supported:
  - HDF5 dataset to Parquet
  - ARFF to Parquet
  - Excel sheet to Parquet
  - JSONL to Parquet
  - NumPy 2D array to Parquet

## Completion Criteria

Move this back to done only after code builds, focused tests pass, and the GUI
can configure at least CSV/TSV, Parquet, Feather, Arrow, and IPC conversions
without presenting stale Phase 1 wording.
