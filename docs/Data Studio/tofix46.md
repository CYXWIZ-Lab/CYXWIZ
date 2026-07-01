# tofix46 - DataConvert follow-up adapters

## Purpose

Track the DataConvert work intentionally left outside `done24`.

`done24` closes the current table-format conversion contract: file input,
upstream dataset input, truthful validation, JSONL, TXT, ARFF, NumPy, HDF5,
Arrow-family formats, CSV/TSV writer cleanup, and pipeline reload support.

This follow-up covers the remaining adapter families that need explicit design
and dependency choices before implementation.

## Scope

### 1. Excel adapter

Excel is not currently supported by the DataConvert core.

Current behavior:

- Excel loading fails closed.
- Data registry Excel loading reports that an additional Excel library is required.
- DataTable Excel loading is still a TODO path.
- Existing tests expect this fail-closed behavior.

Required decision before implementation:

1. Use DuckDB `read_xlsx` through a bundled/offline Excel extension.
2. Add a native C++ XLSX dependency such as xlsxio/OpenXLSX.
3. Keep Python/Polars/openpyxl Excel support scripting-only and outside the core engine.

Recommended implementation path:

- Choose option 1 or 2 explicitly.
- Add focused read tests.
- Add focused write tests only if Excel export is required.
- Add pipeline reload coverage.
- Expose Excel in the DataConvert UI/runtime only after the adapter is real.

### 2. Image conversion path

Image conversion should be a domain adapter, not a hidden table-format special
case.

Expected contract:

- Folder or manifest input.
- Image path, label, dimensions, channels, and metadata columns.
- Optional tensor materialization path.
- Explicit resize, normalize, color-mode, and batching settings.
- Clear distinction between metadata-table output and tensor dataset output.

### 3. Audio conversion path

Audio conversion should define a first-class audio dataset contract.

Expected contract:

- File or manifest input.
- Sample rate handling.
- Duration metadata.
- Mono/stereo policy.
- Optional waveform or spectrogram materialization.
- Label alignment for supervised training.

### 4. Video conversion path

Video conversion should define a first-class clip/frame dataset contract.

Expected contract:

- File, folder, or manifest input.
- Frame extraction policy.
- Clip/window sampling policy.
- FPS and duration metadata.
- Optional frame tensor output or manifest-only output.
- Storage strategy for extracted frames/tensors.

## Guardrails

- Do not grow `DataConvertService` into a universal converter without typed
  adapter boundaries.
- Do not expose UI/runtime formats before the backend adapter exists.
- Do not depend on Python runtime behavior for core engine conversion paths.
- Keep each adapter fail-closed with clear errors until implemented.
- Add small service tests first, then pipeline tests, then UI exposure.

## Completion criteria

- Excel has a selected dependency strategy and tested adapter.
- Image/audio/video have explicit conversion contracts.
- Implemented adapters are visible in UI/runtime only when backend support is complete.
- Pipeline reload behavior is tested for every supported output family.
