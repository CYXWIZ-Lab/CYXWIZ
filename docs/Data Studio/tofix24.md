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

## Missing Format Coverage

DataConvert is still narrower than the engine's data/file surface. The expected
behavior should be: if a format is a supported CyxWiz data file type, DataConvert
should either convert it to another compatible supported type or reject it with a
specific reason and required option. This includes non-table conversions such as
`.jpg` to `.jpeg` and table extraction conversions such as `.xls` to `.csv`.

Do not implement this as one giant generic converter. Keep the core
`DataConvertService` as an orchestrator and add small typed adapters by data
domain:

- Table adapter: CSV, TSV, Parquet, Feather, Arrow IPC, JSON/JSONL, Excel, HDF5
  dataset slices, ARFF, TXT tables, NumPy 1D/2D arrays.
- Image adapter: JPEG/JPG, PNG, BMP, GIF, TIFF, WebP, TGA where OpenCV/stb can
  read/write the format.
- Text adapter: TXT, JSON/JSONL text records, CSV/TSV text columns, Parquet text
  columns.
- Audio/video adapter: future scope unless the engine has a tested decoder and
  encoder path for the specific format pair.

Known missing supported/visible formats:

- Legacy `DataRegistry` files:
  - `.csv`
  - `.tsv`
  - `.json`
  - `.txt`
  - `.h5`, `.hdf5`, `.hdf`
- Current Arrow/table files:
  - `.csv`
  - `.tsv`
  - `.parquet`, `.pq`
  - `.feather`, `.fea`
  - `.arrow`, `.ipc`
- UI-detected table formats not wired through DataConvert:
  - `.json`, `.jsonl`
  - `.xlsx`, `.xls`
  - `.h5`, `.hdf5`, `.hdf`
  - `.arff`
  - `.txt`
- Image/data source formats not wired through DataConvert:
  - `.jpg`, `.jpeg`
  - `.png`
  - `.bmp`
  - `.gif`
  - `.tiff`
  - `.webp`
  - `.tga` where the loader already accepts it
- Audio/video formats are visible in DataInput detection but should stay pending
  until encoder support is explicit:
  - Audio: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.aiff`, `.aif`
  - Video: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.wmv`
- NumPy arrays remain a desired conversion target/source:
  - `.npy`
  - `.npz`

Required conversion examples:

- `.jpg` <-> `.jpeg`
- `.png` -> `.jpg`
- `.xls` / `.xlsx` -> `.csv`
- `.csv` / `.tsv` / `.jsonl` -> `.parquet`
- `.parquet` -> `.csv` / `.jsonl`
- `.h5` / `.hdf5` dataset path -> `.csv` / `.parquet` / `.txt`
- `.txt` table -> `.csv` / `.parquet`
- `.npy` 2D array -> `.csv` / `.parquet`

Format-specific decisions still required before marking those conversions
supported:

- HDF5 needs dataset-path selection because one file can contain many datasets.
  The UI must expose the dataset path and reject ambiguous files.
- HDF5 to TXT needs an explicit text layout choice: delimited table, one value
  per line, or text dataset export. It should not silently stringify arbitrary
  tensors.
- Excel needs sheet selection, header row, skip rows, range selection, and a
  clear dependency choice.
- JSON/JSONL needs a record-orientation policy: array of objects, line-delimited
  objects, or selected JSON path.
- TXT needs a declared mode: plain text corpus, delimited table, fixed-width
  table, or one-record-per-line text.
- ARFF needs a parser decision plus type/null/category handling.
- NumPy needs shape-to-table rules for 1D/2D arrays and rejection for unsupported
  tensor shapes.
- Image conversion needs image-specific options: quality, alpha handling,
  color-space handling, overwrite policy, and batch folder conversion.
- Audio/video conversion needs explicit codec/container support before appearing
  as enabled UI.
- SQLite/DuckDB query-to-table conversion remains future scope and is not a
  file-extension conversion until query input is modeled as a first-class source.

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
  - HDF5 dataset to TXT with explicit export layout
  - ARFF to Parquet
  - Excel sheet to Parquet
  - Excel sheet to CSV
  - JSONL to Parquet
  - NumPy 2D array to Parquet
  - JPG to JPEG
  - PNG to JPEG with alpha-handling validation

## Completion Criteria

Move this back to done only after code builds, focused tests pass, and the GUI
can configure at least CSV/TSV, Parquet, Feather, Arrow, and IPC conversions
without presenting stale Phase 1 wording.
