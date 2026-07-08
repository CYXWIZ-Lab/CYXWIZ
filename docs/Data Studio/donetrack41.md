# donetrack41 - Pinned Host Memory Implementation Tracking

## Scope Guardrail

Implement `tofix41` in small slices. Do not add a fake pinned allocator or UI
claim until the runtime actually changes host-to-device transfer behavior.

## Slice 1 - Compiler Transfer Status Contract

Goal: replace scattered `pin_memory=true` warning text with a structured
compiler/runtime contract that later training, dashboard, and debugger code can
consume.

Included:

- Added shared `PinMemoryTransferMode` names:
  - `regular_host_memory`
  - `pinned_host_memory`
  - `pinned_requested_but_unsupported`
  - `pinned_requested_but_not_applicable`
- Added shared `PinMemoryTransferReason` names for not-requested,
  backend-unavailable, CPU-not-applicable, and active pinned transfer cases.
- Added `PinMemoryTransferStatus` to `TrainingConfiguration`.
- Existing DataLoader compile path now populates the status when
  `pin_memory=true` is requested.
- Existing compiler warning behavior is preserved.
- Focused compiler test now asserts request, node id, batch size, effective
  mode, reason code, and warning classification.

Deferred:

- Runtime/training-dashboard/debugger event surfacing.
- CPU-backend not-applicable detection.
- Real pinned host allocator and batcher-owned staging buffers.

## Slice 2 - Training Runtime Transfer Warning

Goal: surface the compiled pin-memory transfer status when training starts,
without pretending pinned memory is implemented.

Included:

- `TrainingExecutor` now formats the compiled `PinMemoryTransferStatus`.
- When `pin_memory=true` is unsupported, training logs a warning with requested
  state, effective mode, reason code, backend, batch size, and DataLoader node.
- The same warning is recorded through `TrainingTraceCollector` after the run
  trace is active, so dashboard/debugger surfaces can consume it.
- Explicitly requested non-warning statuses use a runtime event instead of a
  warning; `pin_memory=false` remains quiet.

Deferred:

- Dashboard-specific rendering.
- Studio Debugger-specific rendering.
- CPU-backend not-applicable detection.
- Real pinned allocator and transfer path.

## Slice 3 - Structured Trace and Support Bundle Export

Goal: make the runtime transfer status machine-readable for debugger and
support-bundle consumers instead of requiring string parsing.

Included:

- Added pin-memory transfer fields to `TrainingTraceEvent`:
  `pin_memory_requested`, `transfer_mode`, `transfer_reason`,
  `transfer_backend`, and `transfer_batch_size`.
- Added `RecordPinMemoryTransferStatus` so the training runtime emits one typed
  DataLoader transfer event after the trace run is active.
- Trace persistence now round-trips the transfer fields.
- Support bundles now export node id/name and the transfer fields in recent
  training events.
- Debugger contract test now asserts support-bundle export for the structured
  transfer status.

Deferred:

- Dashboard-specific rendering beyond existing warning/event visibility.
- CPU-backend not-applicable detection.
- Real pinned allocator and transfer path.

## Slice 4 - CPU-Only Applicability Truth

Goal: distinguish unsupported GPU-transfer requests from CPU-only graphs where
pinned host transfer is not applicable.

Included:

- DataLoader parsing now records the `pin_memory` request without emitting the
  final warning before backend placement is known.
- A post-placement compiler finalizer classifies explicit requests as:
  - `pinned_requested_but_not_applicable` with
    `pin_memory_cpu_backend_not_applicable` when compiled placement is CPU-only.
  - `pinned_requested_but_unsupported` with
    `pinned_host_memory_backend_unavailable` for GPU/mixed/unknown placement
    until a real pinned transfer backend exists.
- Compiler warnings now use the finalized structured status message.
- Focused compiler test covers the CPU-only not-applicable path.

Deferred:

- Dashboard-specific rendering beyond existing warning/event visibility.
- Real pinned allocator and transfer path.

## Slice 5 - DataLoader Property Truth Wording

Goal: make the DataLoader side-panel truth text match the runtime contract.

Included:

- Updated `pin_memory` property truth copy to describe it as a GPU
  host-to-device transfer optimization request.
- Clarified that `pin_memory` is not a data materialization accelerator.
- Pointed users to compile/training effective transfer mode rather than
  implying the property itself changes runtime behavior.
- Focused properties truth test now asserts the scoped wording.

Deferred:

- Dedicated dashboard widget for effective transfer mode.
- Real pinned allocator and transfer path.

## Slice 6 - Dashboard and Debugger Transfer Visibility

Goal: show the effective transfer status from structured trace fields, not only
from warning prose.

Included:

- Training Dashboard warning area now shows the latest pin-memory transfer event
  when present, including mode, reason, backend, and batch size.
- Studio Debugger training truth summary now shows the latest pin-memory
  transfer status.
- Studio Debugger runtime timeline tooltips now show transfer fields for the
  `DataLoader.PinMemoryTransfer` event.
- The display is driven by `TrainingTraceEvent` fields, so future
  `pinned_host_memory` success events will show without adding another UI path.

Deferred:

- Real pinned allocator and transfer path.

## Slice 7 - Documentation Truth Alignment

Goal: remove stale documentation claims that implied pinned host memory is an
active default.

Included:

- `docs/dataset.md` now defaults example DataLoader config to
  `pin_memory=false` and explains that `pin_memory=true` is currently a
  compatibility/runtime capability request.
- `docs/video_audio.md` now defaults parallel loader examples to
  `pin_memory=false` and labels the feature as unsupported today.
- `docs/HIGH_LEVEL_PIPELINE_WORKFLOW.md` now treats pinned CPU memory as a
  future backend mitigation, not a current bottleneck solution.
- `web_doc/backend/api/device.md` now labels pinned allocation as a future API
  shape and warns that current training batchers fall back to regular host
  memory.

Lean boundary:

- No fake allocator was added.
- The current implementation satisfies the ticket through the explicit
  structured fallback path.
- A real pinned backend remains a separate backend/runtime ticket because it
  needs owned pinned buffers, explicit allocator/free semantics, backend
  capability checks, and profiling.
