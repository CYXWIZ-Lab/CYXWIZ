# track37_done - Backend Placement Implementation Tracking

## Status

Done. `tofix37` acceptance was audited in Slice 16, focused tests passed, and
larger follow-up work was split into `tofix60.md`.

## Scope Guardrail

Implement `tofix37` in small slices. Do not fold persistent cache, deep
preflight, native CUDA recurrent kernels, or pinned host-memory transfer into
the first runtime-writer slice.

## Slice 1 - Runtime Writers and Shared Signatures

Goal: add more non-recurrent runtime fallback writers while keeping the existing
compiler observation contract.

Included:

- Embedding runtime fallback observation writer.
- ReLU runtime fallback observation writer.
- Sigmoid runtime fallback observation writer.
- Tanh runtime fallback observation writer.
- Shared Embedding and Activation shape-signature helpers.
- Compiler-side consumption of the same helpers for cached tensor-layer
  placement observations.
- Focused tests for helper stability and cached compiler routing.

Deferred:

- Loss utility fallback writers.
- Linear/matmul fallback writers beyond existing Dense coverage.
- Graph-runtime tensor-op fallback writers.
- Persistent per-device cache.
- General JIT preflight outcome model.
- GRU probe body.
- Deep preflight mode.
- Native/fused recurrent CUDA.
- Studio debugger timeline work with `tofix32`.
- Real pinned host-memory transfer backend.

## Status

- 2026-07-08: Started Slice 1 implementation.
- 2026-07-08: Slice 1 implemented and verified with
  `test_recurrent_backend_placement.exe`.
- 2026-07-08: Slice 2 implemented and verified with
  `test_recurrent_backend_placement.exe`.
- 2026-07-08: Slice 3 implemented and verified with
  `test_recurrent_backend_placement.exe`.
- 2026-07-08: Slice 4 implemented and verified with
  `test_recurrent_backend_placement.exe`.
- 2026-07-08: Slice 5 implemented and verified with
  `test_recurrent_backend_placement.exe`.
- 2026-07-08: Slice 6 implemented and verified with
  `test_recurrent_backend_placement.exe`.
- 2026-07-08: Slice 7 implemented and verified with
  `test_recurrent_backend_placement.exe`.
- 2026-07-08: Slice 8 implemented and verified with
  `test_recurrent_backend_placement.exe`.
- 2026-07-08: Slice 9 implemented and verified with
  `test_recurrent_backend_placement.exe`.
- 2026-07-08: Slice 10 implemented and verified with
  `test_recurrent_backend_placement.exe`.
- 2026-07-08: Slice 11 implemented and verified with
  `test_recurrent_backend_placement.exe` and
  `test_debugger_contracts.exe`.
- 2026-07-08: Slice 12 implemented and verified with
  `test_recurrent_backend_placement.exe` and
  `test_debugger_contracts.exe`.
- 2026-07-08: Slice 13 implemented and verified with
  `test_recurrent_backend_placement.exe`.

## Slice 2 - Linear Runtime Writer

Goal: cover the existing `LinearLayer` ArrayFire fallback paths without adding a
compiler-facing pseudo node.

Included:

- Shared Linear shape-signature helper with lhs, rhs, output, dtype, and bias.
- Linear initialization fallback observation writer.
- Linear forward fallback observation writer.
- Linear backward fallback observation writer.
- Focused signature and active-device cache retrieval test.

Deferred:

- Compiler routing for Linear remains deferred because the graph compiler does
  not currently expose a distinct `Linear` model node separate from `Dense`.

## Slice 3 - Loss Utility Runtime Writer

Goal: record ArrayFire loss fallback observations with helper-built signatures.

Included:

- Shared Loss shape-signature helper with prediction shape, target shape,
  reduction, and dtype.
- Central loss fallback observation writer in `loss_utils`.
- Supervised regression, probability, and classification losses pass target
  shape and reduction into the writer.
- Focused signature and active-device cache retrieval test.

Deferred:

- Metric-learning loss call sites still use the legacy single-tensor helper
  shape because their second tensor is not always a target label in the same
  semantic sense as supervised losses.

## Slice 4 - Graph-Runtime Tensor Primitive Runtime Writers

Goal: record observations at the existing `Tensor` ArrayFire fallback
boundaries used by graph-runtime tensor ops.

Included:

- Shared generic tensor-op shape-signature helper with input shapes, output
  shape, dtype, and small operation attributes.
- `Tensor::Cat` fallback observation writer for graph Concatenate.
- `Tensor::Dot` fallback observation writer for graph TensorDot.
- Tensor comparison fallback observation writers for TensorCompare scalar and
  tensor forms.
- Tensor logical fallback observation writers for TensorLogicalMask binary and
  unary forms.
- Focused signature and active-device cache retrieval test.

Deferred:

- Compiler-side graph-runtime cached routing remains deferred because
  `CompiledGraphNode` does not currently carry input/output shape metadata.
- Graph executable wrapper-level observation remains deferred because the real
  ArrayFire fallback boundary is inside `Tensor`; wrapper exceptions are not
  necessarily backend fallbacks.

## Slice 5 - Persistent Placement Observation Cache

Goal: persist the existing in-memory placement observation cache without adding
hidden startup file I/O or changing compiler hot-path behavior.

Included:

- Public opt-in cache save/load helpers on the existing placement observation
  API.
- Cache JSON includes schema version, CyxWiz backend version, ArrayFire backend
  name, saved timestamp, and per-observation op/backend/device/dtype/shape,
  reason, source, detail, and timestamp fields.
- Existing exact-match lookup semantics remain unchanged after load.
- Focused save/clear/load active-device round-trip test.

Deferred:

- Automatic cache file location and lifecycle policy.
- Compiler auto-loading of persistent failures.
- Deep preflight writing failures into the persistent cache.

## Slice 6 - Explicit Compiler Cache Consumption

Goal: let callers opt into persistent placement cache consumption without
adding hidden global file I/O to normal compile.

Included:

- `GraphCompiler::Compile` accepts an optional placement observation cache path.
- When provided, the compiler loads the persistent cache before backend
  placement reports are built.
- Cache load failures become compile warnings instead of blocking graph
  validation.
- Focused regression proving a persisted Dense fallback routes the exact same
  compiled shape to CPU through the existing placement report path.

Deferred:

- Studio/user preference for the cache file path.
- Automatic runtime save/load lifecycle.
- Deep preflight cache writes.

## Slice 7 - Structured Recurrent Preflight Outcomes

Goal: make the existing recurrent CUDA preflight probe return a structured
outcome without changing compiler placement behavior.

Included:

- Shared probe outcome enum: safe, unsafe, timeout, unsupported, inconclusive.
- Shared probe result struct with reason code, detail, optional observation,
  and stable outcome names.
- Existing `TryRunRecurrentCudaPreflightProbe` now wraps the structured result
  and preserves the old boolean failure-observation behavior.
- LSTM probe success/failure now reports structured safe/unsafe outcomes.
- Unsupported GRU/bidirectional/out-of-budget probe requests report
  unsupported instead of being indistinguishable from no-op.
- Focused test for stable outcome naming and unsupported GRU probe behavior.

Deferred:

- Timeout enforcement.
- Dedicated GRU executable probe body.
- Compiler/debugger surfacing of safe/inconclusive probe details.
- Deep preflight probe budgets.

## Slice 8 - GRU Recurrent Preflight Probe Body

Goal: add a bounded GRU executable probe to the explicit preflight API without
changing normal compiler placement policy.

Included:

- Single-step ArrayFire CUDA GRU preflight body using synthetic float32 tensors.
- Structured preflight dispatch now supports bounded single-direction LSTM and
  GRU requests.
- Bidirectional recurrent requests remain explicitly unsupported.
- Normal graph compile still routes GRU to CPU with the existing
  `GruArrayFireCudaProbeRequired` reason.
- Focused test now checks unsupported bidirectional GRU while preserving the
  existing GRU CPU placement regression.

Deferred:

- Opting normal compile into GRU CUDA placement.
- Full-sequence/deep GRU preflight.
- Timeout enforcement for executable probes.

## Slice 9 - Recurrent Preflight Timeout Budget

Goal: make the structured recurrent preflight `timeout` outcome actionable
without adding unsafe thread cancellation around CUDA work.

Included:

- `RecurrentCudaPlacementRequest` now carries a `preflight_timeout_ms` budget
  with a conservative default.
- Zero-budget requests return `timeout` without launching a CUDA probe.
- Completed probes that exceed their wall-clock budget return `timeout` and
  record a preflight observation with `backend_compile_timeout`.
- The legacy boolean preflight wrapper treats timeout observations as failed
  probes so existing compiler callers remain conservative.
- Focused test covers stable timeout naming and zero-budget timeout observation
  propagation.

Deferred:

- Hard cancellation of in-flight CUDA work.
- Per-op/deep-preflight budget scheduling.
- Studio controls for probe timeout tuning.

## Slice 10 - Explicit Deep Recurrent Preflight Mode

Goal: add an opt-in deep recurrent probe that exercises the full bounded
sequence length while keeping normal compile and default preflight cheap.

Included:

- `RecurrentCudaPlacementRequest` now carries a `deep_preflight` flag.
- Default preflight remains single-step.
- Deep preflight runs the existing synthetic LSTM/GRU step for `seq_len`
  iterations under the same bounded-shape gate.
- Probe attempt de-duplication now includes probe mode, so single-step and deep
  probes can both be run for the same shape.
- Probe result details include mode and step count.
- Focused test covers the deep flag through the deterministic zero-budget
  timeout path.

Deferred:

- Deep preflight UI controls.
- Per-step timeout checks or hard cancellation while CUDA work is in flight.
- Using successful deep probes to opt GRU into normal CUDA placement.

## Slice 11 - Placement Observation Surfacing

Goal: expose existing placement observation evidence through the compiler,
Studio placement table, and debugger trace payload without adding a separate
debugger subsystem.

Included:

- `BackendPlacementEntry` now carries optional observation source, device,
  dtype, shape signature, detail, and timestamp fields.
- Cached tensor-layer fallback placement copies observation metadata into the
  placement entry.
- Recurrent cached/preflight failure observations now CPU-route conservatively
  for any recorded failure reason, not only CUDA formal-parameter overflow.
- Recurrent placement entries copy observation metadata for debugger/UI use.
- Debug runtime backend classifier attaches observation metadata to trace
  payloads when available.
- Existing Studio backend placement table appends observation evidence under
  the reason/action text.

Deferred:

- A dedicated debugger timeline UI for runtime fallback events.
- Runtime event streaming for fallback observations as they happen.
- Full `tofix32` debugger integration.

## Slice 12 - Structured Probe Scope and Outcome Metadata

Goal: expose probe outcome and scope as structured placement observation
fields instead of relying only on detail-string parsing.

Included:

- `BackendPlacementObservation` now carries optional `probe_outcome` and
  `probe_scope` fields.
- Persistent observation cache JSON saves and loads those fields.
- Recurrent preflight failure observations set `probe_outcome` to `timeout` or
  `unsafe` and set `probe_scope` to `normal_compile` or `deep_preflight`.
- Compiler placement entries carry probe outcome/scope metadata.
- Debugger trace payloads include backend observation probe outcome and scope.
- Studio backend placement table appends probe outcome/scope beside source and
  dtype when observation evidence exists.

Deferred:

- Recording successful probe observations as cache facts.
- Dedicated timeline UI for safe/unsafe/timeout/inconclusive probe events.
- Runtime streaming of probe events while a deep preflight is running.

## Slice 13 - Placement Observation Snapshot API

Goal: provide a small read-only primitive that debugger/timeline code can use
to consume recorded placement observations without introducing a background
event stream yet.

Included:

- Public `SnapshotBackendPlacementObservations()` API.
- Snapshot returns copies of current in-memory observations and does not expose
  the backend cache lock to callers.
- Snapshot output is sorted by timestamp and stable observation identity fields
  for deterministic debugger/support-bundle use.
- Focused test verifies snapshot visibility and metadata preservation for a
  runtime fallback observation.

Deferred:

- Live runtime event streaming.
- Dedicated Studio timeline UI.
- Support-bundle export formatting for placement observations.

## Slice 14 - Support Bundle Placement Observation Export

Goal: make placement fallback/preflight observations available in local support
bundles without adding a live debugger event stream.

Included:

- `DebugSupportBundleInput` now accepts placement observation snapshots as
  explicit input data.
- Support bundle JSON exports placement observation op/backend/device/dtype,
  shape signature, reason, source, timestamp, probe outcome, and probe scope.
- Free-form placement observation detail is redacted before export.
- Focused debugger contract coverage verifies placement observation export,
  stable reason/source fields, probe metadata, and detail redaction.

Deferred:

- Wiring every support-bundle caller to pass
  `SnapshotBackendPlacementObservations()`.
- Live runtime event streaming for placement observations.
- Dedicated Studio timeline UI.

## Slice 15 - Pinned Memory Unsupported Boundary Audit

Goal: satisfy the `pin_memory=true` acceptance path conservatively: either add
a real pinned host-memory transfer backend, or keep the setting visibly
unsupported.

Included:

- Confirmed the current DataLoader truth path preserves `pin_memory=true` while
  marking it `Unsupported` and `RequiresDialog`.
- Confirmed the compiler path surfaces unsupported `pin_memory=true` as a
  warning instead of silently accepting a no-op performance setting.
- Kept real pinned allocation/transfer work deferred because the current
  batchers do not own explicit pinned staging buffers or profiled host-to-device
  transfer points.

Deferred:

- Backend/runtime pinned host allocator and free path.
- Batcher-owned pinned staging buffers.
- Profiled CPU-to-GPU transfer path and benchmark coverage.

## Slice 16 - Acceptance Audit

Goal: close the ticket against its written acceptance criteria and separate
implemented requirements from larger follow-up work.

Acceptance status:

- At least three more non-recurrent runtime paths record placement
  observations: satisfied. Embedding, ReLU, Sigmoid, Tanh, Linear, loss
  utilities, and tensor primitive paths now write runtime fallback
  observations through shared helpers.
- Compiler consumes cached observations through the generic tensor-layer path:
  satisfied. Compiler placement reads in-memory and persistent observations
  and routes matching tensor-layer fallback evidence conservatively.
- Shape signatures are helper-built and shared between compiler/runtime:
  satisfied. Dense, embedding, activation, linear, loss, tensor-op, and
  recurrent helpers are exported through the backend placement observation
  contract and reused by runtime/compiler paths.
- JIT preflight has a general framework with operator-specific probe bodies:
  satisfied for the recurrent scope. The framework exposes structured outcomes,
  reason codes, timeout handling, and operator-specific LSTM/GRU probe bodies.
- GRU is represented explicitly as the next recurrent probe target: satisfied.
  GRU has its own bounded probe body while normal compile remains conservative.
- Deep preflight is opt-in and bounded by timeout/budget: satisfied. Deep mode
  is explicit on `RecurrentCudaPlacementRequest`; timeout is structured and
  recorded as placement evidence.
- Persistent cache format is documented and versioned: satisfied through the
  persistent placement observation cache schema/version and trace slices 5-6.
- Debugger follow-up is linked to `tofix32` rather than duplicated: satisfied.
  Current work exposes observations through placement reports, Studio table,
  debugger trace payloads, snapshot API, and support bundles while leaving live
  timeline/event streaming for the debugger follow-up.
- `pin_memory=true` changes a real backend or remains visibly unsupported:
  satisfied as visibly unsupported. Properties truth and compiler warnings both
  report that current batchers ignore `pin_memory=true`.

Verification already run during this ticket:

- `test_recurrent_backend_placement`
- `test_debugger_contracts`
- `test_properties_truth`
- `test_graph_compiler_deferred_nodes`

Remaining follow-up work:

- Wire `SnapshotBackendPlacementObservations()` into any future real support
  bundle collection call site. Current tree only has the builder/test boundary.
- Add live debugger timeline/event streaming under `tofix32`.
- Implement native/fused recurrent CUDA only after a separate kernel/dependency
  design and correctness plan.
- Implement real pinned host-memory transfer only when batchers own pinned
  staging buffers and transfer profiling.
