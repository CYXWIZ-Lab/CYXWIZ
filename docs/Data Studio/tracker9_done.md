# Tracker 9 - Studio Debugger Implementation Roadmap

This tracker belongs to `tofix9.md`.

`tofix9.md` defines the Studio Debugger architecture. This tracker keeps the
work incremental, testable, and truthful while the debugger grows into one of
CyxWiz's core engineering advantages.

Core engineering rule:

Do not build a second execution framework or a broad telemetry product. Extend
the existing Studio Debugger through narrow trace producers, typed/debuggable
records, deterministic tests, and truthful UI/runtime contracts.

---

## Current Code Truth

Implemented baseline:

- frozen debugger session snapshots through `DebugSessionManager`
- reusable preflight checks through `PreflightValidator`
- text sample preprocessing traces through `TextPreprocessingTracer`
- text smoke runs through `SmokeRunExecutor`
- synthetic model sanity checks through `DebugExecutor`
- persisted debug runs through `DebugRunStore`
- training/crash heartbeat traces through `TrainingTraceCollector` and
  `CrashRunRecorder`
- runtime, memory, trace, Studio-event, and recommendation lenses in
  `StudioDebuggerPanel`
- rule-based next-step guidance through `DebugRecommendationEngine`

Not yet real enough:

- general node-by-node graph execution tracing
- deterministic test coverage for most debugger contracts
- canonical operator-backed Arrow preprocessing tracing
- per-operation CPU/GPU fallback classification
- per-node tensor allocation ownership
- generated-code/export correlation
- robust WER/Event Viewer crash import
- opt-in support bundle for HQ diagnostics
- polished debugger workflow over a stable trace model

---

## Batch Rule

Every implementation batch must include:

- one narrow debugger capability
- a clear trace contract or payload contract
- deterministic tests when the code is testable without the GUI
- UI changes only after the underlying trace data exists
- tracker update with validation result

Do not combine unrelated debugger areas in one batch.

---

## Roadmap Phases

| Phase | Status | Commit | Tests | Notes |
|---|---|---|---|---|
| Audit current debugger code | Complete | | Docs-only; no tests needed | `tofix9.md` now has a code-truth audit comparing the implemented debugger pieces against the architecture goal. |
| Tracker and guardrails | Complete | | Docs-only; no tests needed | Created this tracker to keep implementation batches narrow and test-backed. |
| Debug contract tests | Complete | | Build passed; `test_debugger_contracts.exe` passed | Added deterministic debugger contract executable covering frozen session snapshots, graph snapshot trace payloads, recommendation rules, smoke-run result value contracts, debug-run store save/load/list round-trip behavior, and text preprocessing trace payloads for tokenizer/vocab/padding records. Full successful `SmokeRunExecutor` integration remains deferred to the smoke sample/integration phase because it pulls the model builder, batcher, materializer, and registry dependency chain. |
| Node trace contract | Complete | | Build passed; `test_debugger_contracts.exe` passed | Added `DebugNodeTraceContract` schema helper for canonical node trace records with node id/type, phase, role, input/output shape summaries, dtype, backend, status, warning/error counts, and deterministic issue helpers. |
| Graph trace execution slice | Complete | | Build passed; `test_debugger_contracts.exe` passed | Added `DebugGraphTraceExecutor`, a narrow trace-step runner that emits canonical `DebugNodeTraceContract` records for explicit node execution steps without replacing `PipelineExecutor`; first deterministic contract covers DataInput -> StandardScaler -> DataOutput trace shape. |
| Node inspector data | Complete | | Build passed; `test_debugger_contracts.exe` passed | Added core `DebugNodeInspector` summary builder to derive selected-node inspector data from stable trace records: shape, dtype, backend, status, duration, issues, and related recommendations. |
| Operator-backed preprocessing traces | Complete | | Build passed; `test_debugger_contracts.exe` passed | Added `DebugOperatorTraceAdapter` to convert Arrow operator-backed input/output tables into canonical node trace steps; first deterministic contract traces `TextTokenizerOperator` table shape/schema changes. |
| Smoke sample selection | Complete | | Build passed; test_debugger_contracts.exe passed | Added `DebugSmokeSampleSelector` for deterministic first-N smoke sample indexes and optional label-stratified round-robin indexes; integration into full `SmokeRunExecutor` remains a later heavier batch. |
| Runtime backend classification | Complete | | Build passed; `test_debugger_contracts.exe` passed | Added `DebugRuntimeBackendClassifier` to attach compiler-owned backend placement status, reason code, fallback path, proof flag, and attention warning metadata to canonical debugger traces without inventing a second backend taxonomy. |
| Memory ownership tracing | Complete | | Build passed; `test_debugger_contracts.exe` passed | Added `DebugMemoryOwnershipTracer` for per-node memory trace payloads derived from before/after training memory snapshots, including estimated tensor bytes, CPU/ArrayFire deltas, peak markers, host/device OOM-risk flags, and explicit `ownership_proven=false` metadata so the debugger stays truthful until allocator-level ownership exists. |
| Generated-code/export correlation | Complete | | Build passed; `test_debugger_contracts.exe` passed | Added `DebugExportCorrelationTracer` for stable `GeneratedCode` trace payloads that link export artifacts to run id, graph hash, compile status, source node ids, artifact path/type, exporter name, content size, and deterministic content fingerprint without coupling the debugger to one exporter implementation. |
| Windows crash import | Complete | | Build passed; `test_debugger_contracts.exe` passed | Added `DebugWindowsCrashImporter` for deterministic WER text parsing, CyxWiz/run correlation, stable crash trace payloads, and a first `tofix26` overlap via `CW-R-0501` runtime execution failure code metadata. |
| HQ support bundle | Complete | | Build passed after correcting test severity enum; `test_debugger_contracts.exe` passed | Added `DebugSupportBundleBuilder` for explicit local-first HQ diagnostic bundles that package debug run records, traces, crash heartbeat, training trace, environment summary, and recent logs with redaction applied by default; the builder records upload permission but never performs network upload. |
| UI workflow polish | Complete | | `cyxwiz-engine` target passed; `test_debugger_contracts.exe` passed | Added trace search and attention-only filters to the Studio Debugger timeline so the active lens can be narrowed to failures, warnings, error-role traces, issue-bearing traces, and traces with warning/error counts without changing trace contracts. Also cleared unrelated GUI build blockers in `node_metadata_registry.cpp` and `pipeline_executor.cpp` so the polished panel validates in the full engine target. |

---

## First Implementation Recommendation

Start with `Debug contract tests`.

Reason:

- the debugger already has many useful pieces, but most are not protected by
  deterministic tests
- tests will lock down the existing contracts before node-by-node tracing adds
  more moving parts
- this avoids UI-first work that looks useful but rests on unstable payloads

Suggested first batch:

1. Add tests for `DebugSessionManager::StartSession` and graph snapshot trace.
2. Add tests for `DebugRunStore` save/load/list behavior using an isolated temp
   directory if the current store API permits it, or document the store test seam
   if it does not.
3. Add tests for `DebugRecommendationEngine` using synthetic traces for high
   unknown-token ratio, truncation, invalid loss, and missing gradients.
4. Update this tracker with exact build/test results.

---

## Done Criteria

`tofix9.md` can only be marked done when:

- node-by-node tracing exists for representative real graph paths
- preprocessing, model, runtime, memory, crash, and Studio-event traces use one
  coherent run/session model
- the debugger can identify where computation changed shape, failed, fell back,
  produced invalid values, or used unexpected data
- tests cover the non-GUI trace contracts and recommendation logic
- UI lenses display stable trace data rather than inventing state locally
- unsupported debugger surfaces are hidden, labeled partial, or fail closed
