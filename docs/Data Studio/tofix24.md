# tofix24 - DataConvert Node

## Purpose

Design and implement a rich `DataConvert` node for CyxWiz Engine.

The node should help users convert datasets between supported formats without
leaving the engine. It should be a Data Studio / preprocessing node, not a model
layer.

## Goal

Give users a visual node that can read one dataset format and write another.

Example:

```text
CSV file
  -> DataConvert
  -> Parquet file
  -> DataInput / DataLoader / training
```

The first practical target is to make large CSV datasets easier and faster to
reuse by converting them to Parquet or Arrow-friendly cached formats.

## Phase 1 Implementation Scope

Implemented first slice:

- Node browser entry: `DataConvert`
- Category: Data Sources / Smart I/O utility
- Rich dialog with Source, Output, Preview, and Run tabs
- Input: CSV or TSV file path
- Output: Parquet file path
- Options: delimiter, header row, skipped rows, Parquet compression, row group size, overwrite, create parent folders, manifest writing
- Preview: Arrow-inferred row count, column count, column names, data types, and nullable flags
- Run: converts CSV/TSV to Parquet and records rows, columns, bytes written, status, and manifest path
- Manifest: `<output>.manifest.json` containing input/output paths, formats, rows, columns, compression, settings hash, timestamp, and file sizes
- Result clarity: conversion returns and displays the generated Parquet output path first; the manifest is labeled as a sidecar metadata file.
- DataInput compatibility: Parquet text-table outputs can be loaded as text datasets without the audit treating the Parquet bytes as plain text.
- Compact Properties panel: shows source path, Parquet output path, sidecar manifest, rows written, and status while keeping detailed editing in the dialog.

Phase 1 is a manual Data Studio utility workflow. The node does not execute as
part of the training `PipelineExecutor` yet. Users run conversion in the dialog,
then point a downstream `DataInput` node at the generated Parquet file.

Important output behavior:

- The converted dataset is the `.parquet` / `.pq` file selected in Output.
- `<output>.manifest.json` is optional sidecar metadata only. It records how the
  file was produced; it is not the dataset to load into `DataInput`.

## Node Name

`DataConvert`

## Node Category

Data / Preprocessing / Utility

## Core Workflow

1. User adds `DataConvert` node.
2. User opens a rich configuration dialog.
3. User selects input file or input table.
4. User selects output format.
5. Node previews schema and sample rows.
6. User adjusts conversion options.
7. User runs conversion.
8. Node writes the output file and records conversion metadata.
9. Downstream nodes can use the converted output.

## Initial Supported Conversions

Phase 1 should focus on table formats:

- `CSV -> Parquet` - implemented

Next table conversions:

- `Parquet -> CSV`
- `CSV -> Arrow IPC / Feather`
- `Parquet -> Arrow IPC / Feather`
- `JSONL -> Parquet`
- `JSONL -> CSV`

Later conversions:

- Excel -> CSV / Parquet
- HDF5 table -> Parquet
- SQLite query -> Parquet
- DuckDB query -> Parquet
- Text folder -> manifest CSV / Parquet
- Image folder -> manifest CSV / Parquet
- Audio folder -> manifest CSV / Parquet

## Rich Dialog Design

The Properties panel should stay compact.

For `DataConvert`, the Properties panel should show:

- General node fields
- Open Dialog button
- Conversion status summary
- Output file path summary

Implemented: the compact Properties panel shows status, source path, Parquet
output path, rows written, and sidecar manifest path. Raw conversion settings
stay in the dialog.

All detailed configuration belongs in the dialog.

### Dialog Tabs

Recommended tabs:

- Source
- Schema
- Output
- Options
- Preview
- Run
- Logs

### Source Tab

Controls:

- Input source type
- Input path picker
- Format auto-detect toggle
- Explicit input format selector
- File encoding
- CSV delimiter
- CSV quote character
- Header row toggle
- Skip rows
- Max rows for preview

Supported source types:

- File
- Directory
- Existing Data Studio table
- SQL query result

### Schema Tab

Purpose:

Let the user inspect and optionally override inferred types before conversion.

Features:

- Column list
- Inferred type
- Target type override
- Nullable toggle
- Include/exclude column
- Rename column
- Missing-value policy
- Date/time format

Common target types:

- string
- int32
- int64
- float32
- float64
- bool
- date
- timestamp
- categorical

### Output Tab

Controls:

- Output path picker
- Output format
- Overwrite existing file toggle
- Create parent folders toggle
- Partitioning options for Parquet
- Compression options
- Row group size for Parquet
- Save conversion manifest toggle

Output formats:

- Parquet
- CSV
- Arrow IPC / Feather
- JSONL

Parquet compression options:

- none
- snappy
- gzip
- zstd
- brotli

### Options Tab

General conversion controls:

- Preserve column order
- Drop empty rows
- Trim string fields
- Normalize line endings
- Convert categorical columns
- Validate row count after write
- Validate schema after write
- Fail on invalid rows
- Save invalid rows to reject file

Large-file controls:

- Chunk size
- Streaming conversion mode
- Memory limit
- Progress update interval

### Preview Tab

Features:

- Input preview
- Output schema preview
- Sample converted rows
- Column statistics
- Null counts
- Type mismatch warnings
- Estimated output size

### Run Tab

Features:

- Convert button
- Cancel button
- Progress bar
- Rows read
- Rows written
- Bytes read
- Bytes written
- Elapsed time
- Throughput
- Output file link/path

Phase 1 note:

- Convert button is implemented.
- Progress/cancel should be added when conversion becomes streaming/async.

### Logs Tab

Shows conversion-specific messages:

- Source detected
- Schema inference result
- Warnings
- Rejected rows
- Conversion errors
- Final summary

## Runtime Behavior

The node should be executable independently from model training.

It should support:

- Manual run from dialog
- Pipeline run as part of Data Studio graph
- Optional automatic skip if output is already fresh

Freshness check should compare:

- Input path
- Input modified time
- Input size
- Conversion settings hash
- Output path
- Output manifest

## Conversion Manifest

When enabled, write a small sidecar manifest next to the output file.

Example:

```json
{
  "node": "DataConvert",
  "version": 1,
  "input_path": "data/sentiment.csv",
  "input_format": "csv",
  "output_path": "data/sentiment.parquet",
  "output_format": "parquet",
  "rows_read": 53043,
  "rows_written": 53043,
  "columns": 12,
  "settings_hash": "...",
  "created_at": "..."
}
```

## Error Handling

Errors should be vivid and actionable.

Examples:

- Input file does not exist.
- Input format could not be detected.
- CSV row has more columns than the header.
- Column type inference failed.
- Output path already exists and overwrite is disabled.
- Parquet compression codec is unavailable.
- Not enough memory for non-streaming conversion.
- Write completed but validation failed.

Each error should tell the user:

- What failed
- Which file/column/row caused it when available
- What setting can fix it

## Compiler / Graph Integration

`DataConvert` should compile as a preprocessing operator.

It should expose:

- Input table/file artifact
- Output table/file artifact
- Schema metadata
- Execution status

Downstream `DataInput` or `DataLoader` nodes should be able to use the converted
output path directly.

## Implementation Notes

Preferred backend:

- DuckDB for reading/writing common table formats where possible
- Arrow for in-memory table interchange
- Native CSV/Arrow helpers already present in the engine where appropriate

Do not manually parse complex formats if DuckDB/Arrow already supports them.

Recommended first implementation:

- CSV input
- Parquet output
- schema preview
- conversion run
- manifest write
- output path passed downstream

## Open Questions

- Should `DataConvert` own the output file, or should it only create a reusable
  artifact path?
- Should conversion happen immediately from the dialog or only when the graph is
  executed?
- Should converted artifacts be stored in the project cache folder by default?
- Should failed rows be written to a reject file by default?
- Should the node support directory batch conversion in phase 1?
- Should the output be registered automatically in `DataRegistry` after
  conversion?

## Recommended Phase 1

Build the smallest useful native version first:

- `CSV -> Parquet`
- Rich dialog
- Schema preview
- Parquet compression selector
- Run/cancel/progress UI
- Conversion manifest
- Clear error messages

After phase 1 is stable, add more formats and Data Studio graph execution.
