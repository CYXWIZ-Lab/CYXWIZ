# 45) Debug diagnostics import/export and support-bundle contracts

## 45.1 Scope and design boundary

This section documents diagnostic contracts that extend the base debugger trace pipeline but are implemented as dedicated utilities rather than fully wired production controls in the current engine target.

Covered contracts:

- backend placement trace enrichment (`DebugRuntimeBackendClassifier`)
- memory delta trace enrichment (`DebugMemoryOwnershipTracer`)
- generated-code/export correlation trace enrichment (`DebugExportCorrelationTracer`)
- external crash artifact parsing and stable trace conversion (`DebugWindowsCrashImporter`)
- HQ/diagnostic package production with redaction (`DebugSupportBundleBuilder`)
- overlap and placement of these utilities in production vs test wiring

## 45.2 Where this design sits in the existing chain

Existing runtime debugger flow uses:

- `CrashRunRecorder` for persisted run summary (`current_run.json`)
- `TrainingTraceCollector` for live/last-run telemetry (`current_training_trace.json`)
- `DebugSessionManager::StartSession` for session graph-snapshot trace seeding
- `DebugRecommendationEngine::Build` to produce recommendations from traces + issues + crash + training trace

The contracts in this section add extra trace enrichment and export surfaces that can be attached at trace-production points, but their current call sites are contract-test only.

## 45.3 Data contract modules (leaf interfaces)

### 45.3.1 Runtime backend classification contract

`DebugRuntimeBackendClassifier`

- Input model: `BackendPlacementEntry` with fields:
  - `requested_backend`, `expected_backend`, `fallback_backend`, `status`, `reason_code`, `explanation`, `suggested_action`
- Output classification (`DebugRuntimeBackendClassification`):
  - copies requested/expected/fallback and reason metadata
  - sets `proven` when placement status is in `{Gpu, Cpu, Mixed, Unsupported}`
  - sets `fallback_possible` when fallback exists and status is not hard-unsupported
  - computes `needs_attention` from compiler placement reason policy
- `AttachToTrace(trace, placement)` writes the canonical payload contract:
  - `backend_requested`
  - `backend_expected`
  - `backend_fallback`
  - `backend_status`
  - `backend_reason_code`
  - `backend_explanation`
  - `backend_suggested_action`
  - `backend_proven`
  - `backend_fallback_possible`
  - `backend_needs_attention`
- If `needs_attention` is true, a warning is emitted with reason text.

### 45.3.2 Memory ownership trace contract

`DebugMemoryOwnershipTracer`

- Input model: `DebugMemoryOwnershipInput` with:
  - node identity and graph phase (`node_id`, `node_name`, `node_type`, `phase`, `role`)
  - output shape + dtype + backend
  - `bytes_per_element`
  - `host_budget_bytes`, `device_budget_bytes`
  - before/after memory snapshots (`TrainingTraceEvent`)
- Output trace fields:
  - canonical trace node contract via `DebugNodeTraceContract::Make(...)`
  - `memory_schema = "cyxwiz.debug.memory_ownership.v1"`
  - `memory_observation = "training_trace_delta"`
  - `ownership_proven = false` with explanatory note that ownership is inferred, not allocator-proven
  - `estimated_tensor_bytes`, `bytes_per_element`
  - before/after and delta metrics:
    - CPU allocated/peak/buffers
    - AF allocated/locked bytes and buffer counts
  - budgets + risk flags:
    - `host_oom_risk`
    - `device_oom_risk`
- Warning policy:
  - emit warning for host OOM-risk
  - emit warning for device OOM-risk

### 45.3.3 Export correlation contract

`DebugExportCorrelationTracer`

- Input model: `DebugExportCorrelationInput` fields:
  - `artifact_kind`, `artifact_path`, `exporter_name`, `graph_hash`, `compile_success`,
    `compile_status`, `source_node_ids`, `generated_content`, `message`
- Output trace:
  - `node_name = ExportArtifact` source (or `"GeneratedCodeExport"` fallback)
  - `node_type = "ExportArtifact"`
  - `phase = "ExportCorrelation"`
  - `role = GeneratedCode`
  - `dtype = artifact_kind`
  - `status = ok|failed` from compile flag
  - payload:
    - `schema`
    - artifact path, exporter name, graph hash
    - compile flags + compile status
    - `source_node_ids`
    - deterministic `content_fingerprint`
    - `content_bytes`
    - `message`
- Warning policy:
  - warning if artifact path missing
- Error policy:
  - error if compile failed (status failed + reason)
- `Fingerprint(...)` is a FNV-like rolling hash over generated content bytes.

### 45.3.4 Windows crash import contract

`DebugWindowsCrashImporter`

- Parsing contract:
  - `ParseWerText(content, report_path)` parses line-based `key=value`
  - extracts:
    - process name (`AppName`, `Application Name`, `OriginalFilename`, `Faulting Application Name`, etc.)
    - fault module
    - exception code
    - event time
    - report id
  - sets `report_path`, `raw_excerpt` (up to 512 chars), `available` (any field present)
- Correlation contract:
  - `Correlate(run, report)` returns:
    - matched if unavailable inputs -> false + reason
    - matched by run-id in report excerpt
    - matched by process contains `cyxwiz`
    - matched by report id equality
    - otherwise unmatched with explicit reason
- BuildTrace contract:
  - returns a canonical `DebugTraceRecord` with:
    - `role = Error`
    - `node_name = "WindowsCrashImport"`
    - `node_type = "WER"`
    - `phase = "WindowsCrashImport"`
    - `dtype = "wer"`
    - `status = "captured" | "missing"`
    - schema + stable `error_code = "CW-R-0501"`
    - `matched`, `match_reason`
    - full run/report metadata
  - warning policy:
    - no report available -> warning
    - report available but not matched -> warning

### 45.3.5 Support bundle contract

`DebugSupportBundleBuilder`

- Input model (`DebugSupportBundleInput`):
  - `request_id`
  - `reason`
  - `debug_run` (`DebugRunStoreRecord`)
  - `crash_run` (`CrashRunSummary`)
  - `training_trace` (`TrainingTraceSummary`)
  - `environment` map
  - `recent_logs`
  - `allow_hq_upload`
- Output contract:
  - top-level fields:
    - `schema`
    - `request_id`
    - `reason`
    - `local_first = true`
    - `hq_upload_allowed`
    - `hq_upload_performed = false`
    - `redaction_applied = true`
    - `debug_run`, `crash_run`, `training_trace`, `environment`, `recent_logs`
- Redaction rules:
  - keys containing `path`, `file`, `dataset`, `raw`, `preview`, `token`, `password`, `secret`, `credential` are replaced with `[REDACTED]`
  - strings are redacted with token/password/secret markers replaced by `...[REDACTED]`
- `debug_run` serialization keeps `error_code` values while redacting file/path-like fields.

## 45.4 Wiring and integration status (design truth, not aspiration)

### 45.4.1 Where utilities are currently wired

- These utility `.cpp` files are built only in `test_debugger_contracts` target:
  - `debug_export_correlation_tracer.cpp`
  - `debug_memory_ownership_tracer.cpp`
  - `debug_runtime_backend_classifier.cpp`
  - `debug_support_bundle_builder.cpp`
  - `debug_windows_crash_importer.cpp`
- They are not in the main `ENGINE_SOURCES` list used by executable `cyxwiz-engine`.

### 45.4.2 Existing production crash handling overlap

- `CrashRunRecorder` already:
  - reads local WER report metadata when suspicious crash is detected
  - populates `CrashRunSummary` fields (`windows_fault_module`, `windows_exception_code`, `windows_report_id`, `windows_report_path`, etc.)
  - appends a generic warning suggesting matching APPCRASH data
- Therefore `DebugWindowsCrashImporter` is currently an additional, currently-unwired importer/normalizer path, not the one feeding standard recovery snapshots.

### 45.4.3 Recommendation engine consumption implications

- `DebugRecommendationEngine::Build` currently consumes:
  - `traces`
  - `issues`
  - `smoke_result`
  - `CrashRunSummary`
  - `TrainingTraceSummary`
- It does not yet derive recommendations from:
  - `backend_*` enrichment payload
  - `memory_*` deltas
  - `export` or `WindowsCrashImport` generated traces
- Those fields are present in contract tests but are not guaranteed in production path yet.

## 45.5 Contract flow diagrams (ASCII)

```text
Backend placement path (policy -> trace enrichment)

BackendPlacementEntry
  -> Classify()
      -> classification{proven, fallback_possible, needs_attention, ...}
  -> AttachToTrace(session_trace)
      -> payload backend_* fields
      -> optional warning when needs_attention
```

```text
Memory trace path (training step observation)

TrainingTraceEvent before/after + budgets
  -> BuildTrace()
      -> canonical DebugTraceRecord
      -> bytes before/after, deltas, flags, risks
      -> warnings for host/device OOM-risk
```

```text
Export and WER support path

generated artifacts -> ExportCorrelationInput -> BuildTrace()
  -> trace[phase=ExportCorrelation, role=GeneratedCode]

WER report text -> ParseWerText() -> Parse structure
  + CrashRunSummary -> Correlate()
  -> BuildTrace() -> trace[phase=WindowsCrashImport]

debug_run + crash_run + training_trace + env + logs
  -> Build()
    -> redacted support bundle payload
    -> local-first export contract, no upload attempt
```

```text
Production today vs target wiring

Engine runtime build:
  BuildSession -> GraphSnapshot trace
  -> CrashRunRecorder + TrainingTraceCollector persistence
  -> recommendation engine consumes last_run/training_trace

Contract utility layer (test target):
  BackendClassifier / MemoryOwnership / ExportCorrelation / CrashImporter / SupportBundle
  -> validated by test_debugger_contracts
  -> not yet mandatory in default production trace pipeline
```

## 45.6 Evidence anchors

| Claim | Source |
|---|---|
| Backend classifier inputs and attach payload keys | `cyxwiz-engine/src/core/debug_runtime_backend_classifier.h:24-29`, `cyxwiz-engine/src/core/debug_runtime_backend_classifier.cpp:21-36`, `cyxwiz-engine/src/core/debug_runtime_backend_classifier.cpp:37-53` |
| Memory ownership input model, schema, deltas, risk flags, warnings | `cyxwiz-engine/src/core/debug_memory_ownership_tracer.h:13-39`, `cyxwiz-engine/src/core/debug_memory_ownership_tracer.cpp:31-110`, `cyxwiz-engine/src/core/debug_memory_ownership_tracer.cpp:112-136` |
| Export correlation role/phase/payload/fingerprint | `cyxwiz-engine/src/core/debug_export_correlation_tracer.h:11-33`, `cyxwiz-engine/src/core/debug_export_correlation_tracer.cpp:5-46`, `cyxwiz-engine/src/core/debug_export_correlation_tracer.cpp:48-56` |
| WER parser/correlation/build-trace contracts and `CW-R-0501` | `cyxwiz-engine/src/core/debug_windows_crash_importer.h:27-43`, `cyxwiz-engine/src/core/debug_windows_crash_importer.cpp:78-191` |
| Support bundle schema, redaction model, nested serialization | `cyxwiz-engine/src/core/debug_support_bundle_builder.h:15-40`, `cyxwiz-engine/src/core/debug_support_bundle_builder.cpp:91-117`, `cyxwiz-engine/src/core/debug_support_bundle_builder.cpp:119-167`, `cyxwiz-engine/src/core/debug_support_bundle_builder.cpp:169-286` |
| Production WER attachment is done by CrashRunRecorder (not this importer) | `cyxwiz-engine/src/core/crash_run_recorder.h:29-57`, `cyxwiz-engine/src/core/crash_run_recorder.cpp:136-151`, `cyxwiz-engine/src/core/crash_run_recorder.cpp:371-381` |
| Utilities are included in debugger contract test target; not in `ENGINE_SOURCES` | `cyxwiz-engine/CMakeLists.txt:86-150`, `cyxwiz-engine/CMakeLists.txt:3264-3275` |
| Test contract coverage for each module | `cyxwiz-engine/tests/test_debugger_contracts.cpp:330-380`, `cyxwiz-engine/tests/test_debugger_contracts.cpp:476-545`, `cyxwiz-engine/tests/test_debugger_contracts.cpp:547-611`, `cyxwiz-engine/tests/test_debugger_contracts.cpp:613-687`, `cyxwiz-engine/tests/test_debugger_contracts.cpp:689-843` |
