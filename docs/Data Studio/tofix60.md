# tofix60 - Backend Placement Follow-up: Debugger Timeline, Native Recurrent CUDA, and Pinned Transfer

## Purpose

Continue the work closed in `tofix37` without mixing large GPU/runtime projects
into the completed placement-observation ticket.

`tofix37` established:

- runtime fallback observations across multiple backend paths,
- stable shared shape signatures,
- persistent placement observation cache,
- structured recurrent JIT preflight outcomes,
- bounded LSTM/GRU preflight probes,
- opt-in deep preflight,
- debugger/support-bundle surfacing of placement evidence,
- truthful `pin_memory=true` unsupported reporting.

This ticket tracks the larger follow-up work that still needs design,
profiling, and dedicated validation.

## Follow-up Work

### 1. Live placement observation debugger timeline

This overlaps with `tofix32`. Do not duplicate debugger architecture. Extend the
existing debugger/event surfaces only where needed.

Expose placement observations as a runtime timeline:

- event time,
- op type and node identity when available,
- backend and device signature,
- dtype and shape signature,
- source: runtime fallback vs preflight probe,
- reason code,
- probe outcome and probe scope,
- concise user-facing explanation,
- "not VRAM" explanation for CUDA formal-parameter overflow.

Keep support-bundle export as the durable/offline artifact. The timeline should
be a live debugging surface, not a second cache.

### 2. Support-bundle snapshot wiring

`tofix37` added `SnapshotBackendPlacementObservations()` and support-bundle JSON
export, but the current tree only has the builder/test boundary.

When a real support-bundle collection call site exists:

- collect `SnapshotBackendPlacementObservations()`,
- pass the snapshot into `DebugSupportBundleInput`,
- preserve redaction of free-form detail strings,
- keep the builder deterministic and free of direct global runtime reads.

### 3. Native or fused recurrent CUDA path

ArrayFire JIT recurrent loops remain a weak point. The long-term fix should be
controlled kernel boundaries, not more string classification.

Investigate:

- fused/native CUDA recurrent kernels,
- cuDNN-style GRU/LSTM integration if acceptable for the dependency model,
- CPU-vs-GPU deterministic correctness tests,
- gradient correctness tests,
- timeout protection,
- benchmark and profiler coverage,
- clear fallback when kernels are unavailable.

GRU, BiGRU, and BiLSTM should stay conservatively CPU-routed until this path is
proven safe or a bounded probe proves the exact target shape/device safe enough
for the selected placement policy.

### 4. Real pinned host-memory transfer backend

`pin_memory=true` is currently preserved but visibly unsupported. Add support
only when it changes actual data movement.

Required work:

- backend/runtime pinned host allocator and free path,
- batcher-owned pinned staging buffers,
- fallback to regular host memory when pinned allocation is unavailable,
- explicit host-to-device transfer points that can be profiled,
- CUDA/ArrayFire backend checks before using pinned memory,
- cleanup/shutdown ownership so pinned pages are not leaked,
- benchmark comparing regular host memory vs pinned memory on at least one real
  GPU workload.

This is input-transfer work. It will not fix CUDA generated-kernel
formal-parameter overflow in recurrent JIT paths.

## Acceptance Criteria

- Live debugger timeline shows placement observation events without requiring
  users to read raw ArrayFire/NVRTC logs.
- Support-bundle collection includes the current placement observation snapshot
  once a real collection call site exists.
- Native/fused recurrent CUDA has correctness, fallback, and benchmark evidence
  before enabling GPU placement for recurrent paths beyond the current
  conservative policy.
- `pin_memory=true` either changes a real pinned transfer path or remains
  visibly unsupported in compiler/UI truth surfaces.
- New runtime paths reuse the existing placement observation schema, reason
  codes, shape-signature helpers, and persistent cache contracts.

## Non-goals

- Reopening `tofix37`.
- Treating successful synthetic probes as proof that a full training step is
  safe.
- Adding broad GPU dependencies without a dependency and deployment decision.
- Building pinned-memory UI controls before the backend transfer path exists.
