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
