# To Fix 26 - CyxWiz Error Code System Design

This document defines a stable internal error-code system for CyxWiz.

The goal is not to replace readable error messages. The goal is to give
engineers, QA, support, and internal tooling a short code that identifies
where an error came from, what kind of failure happened, and how to route
investigation.

A good error code should answer:

`Which subsystem failed, what class of failure is it, and where should the team look first?`

---

## Design Goals

- make compiler, runtime, GPU, CPU, data, file, GUI, memory, and export failures easy to classify
- keep codes stable across releases
- make logs searchable and support tickets easier to triage
- avoid duplicate ad-hoc error strings for the same failure class
- keep implementation small enough that engineers will actually use it
- preserve detailed human-readable messages next to each code

---

## Non-Goals

- do not create a large exception framework before the engine needs it
- do not hide real diagnostic details behind opaque codes
- do not force every warning/debug message to have a permanent code
- do not use error codes as user-facing documentation by themselves
- do not duplicate the same failure classification in unrelated modules

---

## Proposed Code Format

Use this format:

`CW-<DOMAIN>-<RANGE>`

Examples:

- `CW-C-0101`
- `CW-R-0204`
- `CW-G-0302`
- `CW-D-0407`

Where:

- `CW` means CyxWiz
- `<DOMAIN>` is a one-letter subsystem family
- `<RANGE>` is a four-digit numeric code inside that subsystem

The letter is intentionally short so codes are easy to scan in logs.
The message should always include the readable explanation.

Example log shape:

`[CW-C-0101] Graph compile failed: no connected loss node was found.`

---

## Domain Prefixes

| Prefix | Domain | Intended owner |
|---|---|---|
| `C` | Compiler / graph validation | `GraphCompiler`, shape validation, compile gates |
| `R` | Runtime / pipeline execution | `PipelineExecutor`, training runtime, execution orchestration |
| `G` | GPU backend | CUDA, ArrayFire GPU, GPU placement, GPU kernel failures |
| `P` | CPU backend | CPU fallback, CPU tensor path, host compute failures |
| `D` | Data contract / schema | dataset shape, label/feature mismatch, Arrow schema, class mismatch |
| `F` | File and IO | file missing, unsupported format, import/export path, permissions |
| `M` | Memory / resource | allocation, VRAM, host RAM, batch too large, resource exhaustion |
| `U` | UI / GUI | editor state, node panel, dialog, user workflow guardrails |
| `X` | External integration | ONNX, GGUF, DuckDB, Python bridge, plugins, network APIs |
| `S` | Serialization / artifact | model save/load, `.cyxmodel`, checkpoints, manifests |
| `T` | Training loop | epochs, optimizer/loss setup, metrics, convergence runtime |
| `A` | Audio / media pipeline | audio loading, spectrograms, media transforms |
| `I` | Image / vision pipeline | image loading, augmentation, vision preprocessing |
| `Q` | Query / scripting | CyxQL, query console, formulas, expressions |
| `N` | Network / remote | HTTP, cloud dataset sources, remote services |

Note:

- `C` is reserved for compiler, not CPU.
- CPU uses `P` for processor/host path to avoid ambiguity.

---

## Numeric Ranges

Each domain owns `0001-9999`.

Recommended range convention inside each domain:

| Range | Meaning |
|---:|---|
| `0001-0099` | generic or unknown failure inside that domain |
| `0100-0199` | validation / precondition failure |
| `0200-0299` | unsupported feature / fail-closed path |
| `0300-0399` | shape, schema, or type mismatch |
| `0400-0499` | configuration / parameter error |
| `0500-0599` | execution failure |
| `0600-0699` | dependency/backend failure |
| `0700-0799` | resource exhaustion |
| `0800-0899` | serialization / persistence failure |
| `0900-0999` | internal invariant violation |

A team can reserve more specific ranges later, but the first implementation
should keep this simple.

---

## First-Pass Code Catalog

### Compiler / graph validation codes

| Code | Meaning | Example |
|---|---|---|
| `CW-C-0001` | generic compiler issue fallback | legacy compiler warning without narrower classification |
| `CW-C-0101` | missing required training path node | graph has model but no loss |
| `CW-C-0102` | unsupported training node | `Conv2D` selected before backend support exists |
| `CW-C-0103` | graph connectivity invalid | loss is not connected to selected model output |
| `CW-C-0301` | tensor shape mismatch | `TimeDistributed` receives non-sequence input |
| `CW-C-0302` | label/output shape mismatch | sequence logits do not match token labels |
| `CW-C-0401` | invalid compiler parameter | split ratios are outside valid range |
| `CW-C-0901` | compiler invariant violation | selected path is empty after validation |

### Runtime / pipeline execution codes

| Code | Meaning | Example |
|---|---|---|
| `CW-R-0101` | pipeline is empty or malformed | no executable node list |
| `CW-R-0201` | node type is unsupported at runtime | fail-closed Data Studio node |
| `CW-R-0202` | operator registration missing | operator-backed node has no factory creator |
| `CW-R-0301` | runtime input dataset missing | node expects input table but none exists |
| `CW-R-0401` | runtime parameter invalid | unsupported aggregation or formula token |
| `CW-R-0501` | operator execution failed | `IPipelineOperator::Apply` returned error |
| `CW-R-0901` | runtime invariant violation | node result map references impossible state |

### GPU backend codes

| Code | Meaning | Example |
|---|---|---|
| `CW-G-0101` | GPU backend unavailable | CUDA device not found |
| `CW-G-0201` | GPU path intentionally disabled | recurrent path CPU-routed by policy |
| `CW-G-0301` | GPU tensor shape unsupported | generated kernel cannot handle layout |
| `CW-G-0501` | GPU kernel execution failed | ArrayFire operation failed |
| `CW-G-0601` | GPU dependency failure | CUDA/driver/runtime mismatch |
| `CW-G-0701` | GPU memory exhausted | VRAM allocation failed |

### CPU / host backend codes

| Code | Meaning | Example |
|---|---|---|
| `CW-P-0101` | CPU backend unavailable | required CPU execution path not compiled |
| `CW-P-0301` | CPU tensor shape unsupported | CPU fallback cannot handle layout |
| `CW-P-0501` | CPU operation failed | BLAS/host tensor operation failed |
| `CW-P-0701` | host memory exhausted | allocation failed during batch creation |

### Data contract codes

| Code | Meaning | Example |
|---|---|---|
| `CW-D-0101` | required column missing | `target_col` not found |
| `CW-D-0102` | required label column missing | training graph has no label target |
| `CW-D-0301` | column type mismatch | text column passed to numeric scaler |
| `CW-D-0302` | feature/label row count mismatch | batcher sees inconsistent dataset size |
| `CW-D-0303` | class-label mismatch | labels outside expected class range |
| `CW-D-0304` | vocabulary coverage warning | selected text sample has high unknown-token ratio |
| `CW-D-0401` | invalid data split | train/val/test ratios invalid |
| `CW-D-0501` | dataset materialization failed | Arrow table registration failed |

### File and IO codes

| Code | Meaning | Example |
|---|---|---|
| `CW-F-0101` | file path missing | import path empty |
| `CW-F-0102` | file not found | dataset path does not exist |
| `CW-F-0201` | file format unsupported | JSON graph source not implemented |
| `CW-F-0401` | file option invalid | unsupported delimiter or encoding |
| `CW-F-0501` | file read failed | CSV/Parquet reader failed |
| `CW-F-0502` | file write failed | export path cannot be written |
| `CW-F-0601` | permission denied | output directory not writable |

### Memory and resource codes

| Code | Meaning | Example |
|---|---|---|
| `CW-M-0101` | resource check failed | device memory query unavailable |
| `CW-M-0701` | host memory exhausted | cannot allocate tensor/table |
| `CW-M-0702` | GPU memory exhausted | VRAM allocation failed |
| `CW-M-0703` | batch too large | requested batch exceeds resource limit |

### UI / GUI codes

| Code | Meaning | Example |
|---|---|---|
| `CW-U-0101` | invalid user workflow state | train clicked with no compiled graph |
| `CW-U-0102` | missing required UI selection | no dataset selected |
| `CW-U-0201` | UI-only node used as runtime node | visualization node executed in pipeline |
| `CW-U-0401` | invalid UI parameter | property editor value rejected |
| `CW-U-0901` | UI state invariant violation | selected node metadata missing |

### Serialization / artifact codes

| Code | Meaning | Example |
|---|---|---|
| `CW-S-0101` | artifact path missing | model save path empty |
| `CW-S-0201` | export format unavailable | GGUF export requested before implementation |
| `CW-S-0202` | export format not compiled | ONNX export missing compile flag |
| `CW-S-0501` | model save failed | `SequentialModel::Save` failed |
| `CW-S-0502` | model load failed | manifest or weights invalid |
| `CW-S-0801` | checkpoint serialization failed | checkpoint metadata invalid |

### External integration codes

| Code | Meaning | Example |
|---|---|---|
| `CW-X-0201` | optional integration unavailable | ONNX export not compiled in |
| `CW-X-0501` | third-party library call failed | DuckDB/Arrow call returned error |
| `CW-X-0601` | plugin dependency failure | plugin load failed |
| `CW-X-0602` | Python bridge failure | embedded Python execution failed |
| `CW-X-0701` | remote service unavailable | cloud dataset API unavailable |

---

## Error Record Shape

Internally, errors should eventually carry structured data:

```cpp
struct ErrorRecord {
    std::string code;          // CW-C-0101
    std::string message;       // readable explanation
    std::string subsystem;     // GraphCompiler, PipelineExecutor, ModelExporter
    std::string node_type;     // optional
    std::string node_name;     // optional
    int node_id = -1;          // optional
    std::string detail;        // optional diagnostic context
    std::string recovery_hint; // optional next action
};
```

This does not require a large exception hierarchy. The first implementation can
start with helper functions that format existing errors consistently.

---

## Message Format

Every emitted error should follow this shape:

`[CODE] Short message. Detail: ... Hint: ...`

Examples:

- `[CW-C-0301] TimeDistributed requires sequence input shape [seq_len, features]. Hint: connect it after an embedding or recurrent layer with return_sequences enabled.`
- `[CW-R-0201] Node type 'ExportSQL' is not supported by PipelineExecutor. Detail: SQL database export is not implemented; fake success is disabled.`
- `[CW-G-0201] GRU CUDA path was CPU-routed by policy. Detail: ArrayFire CUDA recurrent kernel generation can exceed backend limits.`

---

## Severity Levels

Codes identify failure class. Severity identifies impact.

| Severity | Meaning |
|---|---|
| `Info` | useful diagnostic, no failure |
| `Warning` | degraded path or recoverable issue |
| `Error` | operation failed but process can continue |
| `Fatal` | process/session cannot safely continue |

Same code can appear at different severities only if the meaning is still the
same. If the meaning differs, create a different code.

---

## Where Codes Should Appear

First implementation targets:

- `GraphCompiler` compile issues
- `PipelineExecutor` validation and execution failures
- `ModelBuilder` unsupported build cases
- `TrainingExecutor` launch/runtime failures
- `ModelExporter` save/export failures
- data loading/import/export paths
- GPU placement and recurrent CPU-routing warnings

Second implementation targets:

- GUI workflow guardrails
- query console / expression engine
- plugin system
- cloud or remote dataset integrations

---

## App-Requested Engine Log Submission

The app should be able to request that engine logs and structured error records
are packaged and sent to HQ for internal debugging, support, engine
performance analysis, and app performance improvement.

Design intent:

- app-side workflows can ask the engine for recent logs after a failure or performance issue
- submitted logs should include error codes, subsystem, node context, runtime
  details, timings, resource usage, and recovery hints when available
- submissions should be explicit and traceable, not silent background uploads
- sensitive paths, dataset contents, credentials, tokens, and private user data
  must be redacted before upload
- log packages should include enough environment context for debugging:
  - CyxWiz version/build
  - OS and architecture
  - backend selection
  - GPU/CPU summary
  - relevant compile/runtime/export error records
  - recent warning/error log window
  - timing and throughput samples when available
  - memory, CPU, and GPU pressure indicators when available
- HQ ingestion should be treated as an external integration and use `CW-X-*`
  or `CW-N-*` errors for upload failures

This should not be implemented as a separate diagnostic system. It should reuse
the same central error-code catalog and structured error records described in
this document.

---

## Implementation Strategy

## Implementation Progress

### Batch 1/2 first slice - central catalog and formatting

Status: implemented.

- Added a small central error-code catalog in `cyxwiz-backend/include/cyxwiz/error_codes.h`.
- Kept `cyxwiz-engine/src/core/error_codes.h` as a forwarding include for engine code.
- Added `FormatError` and `FormatWarning` helpers.
- Kept the implementation header-only to avoid introducing a larger error framework.
- Added a focused `test_error_codes` target that locks representative stable code strings and formatting behavior.

### Batch 3 compiler preparation

Status: partially implemented.

- `ValidationIssue` now has an optional `error_code` field.
- `GraphCompiler` issue creation accepts an optional stable error code.
- Existing compiler issue messages remain readable and unchanged.
- Graph-level missing training path issues now carry `CW-C-0101`.
- Cycle/connectivity compile issues now carry `CW-C-0103`.
- Invalid compiler parameter issues now carry `CW-C-0401`.
- Invalid split warnings now carry `CW-D-0401`.
- Batch-too-large issues now carry `CW-M-0703`.
- Broad compiler-code assignment remains future rollout work to avoid noisy mass edits.

### Batch 4 runtime first slice

Status: partially implemented.

- `PipelineExecutor` parse and validation failures now use stable `CW-R-*` runtime codes.
- Existing human-readable `last_error_` messages are preserved after the code prefix.
- Unsupported runtime node, missing input dataset, invalid parameter, malformed graph, and invalid runtime state paths now produce searchable codes.

Remaining rollout work:

- Assign more compiler codes to high-value `GraphCompiler` issues.
- Wire `ModelBuilder` and training launch/runtime.
- Surface structured codes in the GUI error panels and Studio debugger where currently only strings are shown.
- Add support-bundle structured error record collection once more subsystems emit codes.

### Batch 5 GPU/backend first slice

Status: partially implemented.

- Linear ArrayFire fallback warnings now use `CW-G-0501`.
- LSTM/GRU ArrayFire fallback warnings now use `CW-G-0501`.
- LSTM/GRU CUDA/JIT policy-routed fallback warnings now use `CW-G-0201`.
- Backend debug hook messages receive the same code-prefixed warning text.

### Batch 5 file/data/export first slice

Status: partially implemented.

- DataConvert public preview/convert/load-table boundary now prefixes common file/input/output failures with `CW-F-*`.
- DataConvert adapter read/write failures are normalized at the public boundary as `CW-F-0501` or `CW-F-0502`.
- Model export unsupported/failed save paths now use `CW-S-*`.
- ONNX-not-compiled export now uses `CW-S-0202`.
- Model import probing now prefixes missing file, unsupported format, read failure, and invalid artifact-header failures.

### Batch 5 training/model-builder first slice

Status: partially implemented.

- `BuiltModel` and `BuiltExecutableModel` now carry an optional coded `error_message`.
- `ModelBuilder` converts model/loss/optimizer setup failures into `CW-T-*` messages.
- Empty/unsupported trainable layer build failures now use `CW-C-0102`.
- Graph-executable construction failures now use `CW-T-0102`.
- `TrainingExecutor::Initialize` logs coded training setup failures instead of dropping builder reasons.
- Sequence training setup guardrails now log coded runtime/data/training errors.
- Training execution exceptions recorded by crash-run tracking now use `CW-T-0501`.

### Batch 6 UI/debugger surfacing first slice

Status: partially implemented.

- `GraphCompiler` error aggregation now includes `[CW-*]` codes for coded issues.
- `GraphCompiler` logs now print issue codes beside node names and readable messages.
- Compile result popup shows the issue code beside the issue level/node name when present.
- Studio Debugger trace issue and session issue panels show the issue code when present.
- Existing `ValidationIssue::message` text is left unchanged for compatibility.

### Batch 7 compiler high-value validation codes

Status: partially implemented.

- Selected training-path template/deferred nodes now use `CW-C-0102`.
- Unsupported sequential model layers and unsupported training-control nodes now use `CW-C-0102`.
- Tensor shape-operation failures now use `CW-C-0301`.
- CrossEntropy/Focal class/output mismatches, BCE output-size mismatches, and manual class-weight count mismatches now use `CW-C-0302`.
- Invalid loss/tensor scalar parameters such as `label_smoothing`, `pos_weight`, `class_weight=manual` without weights, and scalar tensor math parameters now use `CW-C-0401`.
- Required-pin and semantic training-chain wiring validators now use `CW-C-0103`.
- Preprocessing/domain mismatches now use `CW-D-0301`.
- Focused compiler contract tests now assert representative stable codes for unsupported nodes, label/output mismatches, invalid parameters, and preprocessing/domain mismatches.

### Batch 8 debugger/support-bundle structured propagation

Status: implemented.

- `DebugRunStore` now persists `ValidationIssue.error_code` for session-level issues and trace-level issues.
- Old debug run JSON files remain loadable because missing `error_code` fields default to empty.
- `DebugSupportBundleBuilder` now includes structured `error_code` fields for record issues and trace issues.
- Support-bundle redaction still preserves `CW-*` codes while redacting paths, dataset names, previews, and secrets.
- Focused debugger contract tests now assert error-code round-trip through debug-run save/load and support-bundle export.

### Batch 9 preflight/smoke/local-debug issue codes

Status: implemented.

- `PreflightValidator` now emits codes for empty graphs, missing datasets, missing compiled layers, unknown model shapes, missing CrossEntropy class count, missing text vocabulary files, and invalid text preprocessing parameters.
- `SmokeRunExecutor` now emits codes for unsupported sequence smoke runs, materialization failures, missing materialized labels, missing/unregistered datasets, empty training splits, model build failures, non-finite predictions/loss, and missing gradients.
- `DebugExecutor` now emits codes for model build failure, non-finite forward/loss/gradient values, dead-gradient warnings, and dry-run exceptions.
- Existing readable messages are preserved; the code is carried structurally in `ValidationIssue.error_code` for GUI/debugger/support-bundle consumers.

### Batch 10 GUI workflow boundary codes

Status: implemented.

- Main-window Train/Debug popup fallback issues now carry stable codes.
- Missing node-editor workflow state now uses `CW-U-0101`.
- Debug-before-train stale/missing Local Debug warnings now use `CW-U-0101`.
- Training launch blocked popup issues now use `CW-T-0101`.
- Recompile/compile exception popup issues and Studio Debugger compile exception records now use `CW-C-0901`.
- Studio Debugger local-debug exception records now use `CW-T-0501`.

### Batch 11 trace-level debugger issue codes

Status: implemented.

- `DebugNodeTraceContract::AddWarning` and `AddError` now accept optional structured `error_code` values.
- `TextPreprocessingTracer` now emits codes for unregistered text datasets, empty text datasets, invalid sample selections, preprocessing trace exceptions, high unknown-token ratio warnings, and truncation warnings.
- Added `CW-D-0304` for vocabulary coverage warnings so text tracing does not misuse column type or class-label mismatch codes.

### Batch 12 debugger diagnostic-code test coverage

Status: implemented.

- `test_debugger_contracts` now links the small `PreflightValidator` implementation so it can test actual preflight error-code emission.
- Added test coverage for empty-graph preflight codes: missing training path, missing dataset, unknown input shape, and unknown output shape.
- Added structural smoke-result coverage to ensure blocked smoke results preserve `ValidationIssue.error_code`.
- Existing trace-helper and text-preprocessing trace tests now cover structured trace issue codes.
- `test_error_codes` now locks `CW-D-0304` so the vocabulary coverage warning code remains stable.

### Batch 13 final structural coverage hardening

Status: implemented.

- Added `CW-C-0001` as the generic compiler issue fallback for legacy/uncategorized compiler diagnostics.
- `GraphCompiler` now guarantees every issue it emits has a non-empty `error_code`; specific call sites keep their precise codes, otherwise `CW-C-0001` is applied.
- `DebugNodeTraceContract::AddWarning` and `AddError` now default uncoded trace issues to `CW-R-0501`.
- `PreflightValidator` now defaults uncoded helper calls to `CW-T-0101`.
- Compiler contract tests now assert representative compile results have codes on every issue, including warning-only valid compiles.
- `test_error_codes` now locks `CW-C-0001`.

### Batch 1. Define the code catalog

- add a central header/source for stable code constants or enum mappings
- include only the first-pass codes that are actually emitted
- add tests that code strings are stable

### Batch 2. Add formatting helpers

- `FormatError(code, message)`
- `FormatError(code, message, detail, hint)`
- avoid broad exception changes

### Batch 3. Wire compiler errors

- start with existing `GraphCompiler` issue creation
- preserve current human-readable messages
- prepend or attach `CW-C-*` codes

### Batch 4. Wire runtime errors

- start with `PipelineExecutor::last_error_`
- map unsupported node, bad input, missing operator, and parameter failures

### Batch 5. Wire backend/resource/export errors

- GPU and CPU backend warnings/errors
- memory allocation/resource failures
- model save/export/load failures

### Batch 6. Surface codes in UI/logs

- show code in error panels and training launch failures
- make code copyable/searchable
- keep user-facing explanation readable

---

## Lean Guardrails

- one code should describe one stable failure class
- do not create codes for every possible string variant
- do not add a new error framework until formatted records need structured transport
- do not duplicate code definitions across compiler/runtime/UI
- tests should lock the catalog and prevent accidental code renumbering
- old error messages should remain understandable without looking up the code

---

## Open Questions

- Should codes be visible to end users or only in advanced details/logs?
- Should warnings use the same code namespace as errors?
- Should code ownership live in one global catalog or per-domain files under one namespace?
- Should compile issues store codes structurally instead of only embedding them in text?
- Which existing warning should be the first GPU code: recurrent CPU-routing or backend initialization?

---

## Bottom Line

CyxWiz needs stable, searchable error codes, but the design should stay small.

The first useful version is a central catalog plus formatting helpers, then a
slow rollout through compiler, runtime, backend, data, export, and UI paths.
