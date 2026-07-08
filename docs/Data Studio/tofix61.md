# tofix61 - Real Pinned Host-Memory Transfer Backend

## Status

Open.

## Background

`done41` closed the `pin_memory=true` truth gap through structured fallback:
CyxWiz now reports whether pinned transfer is unsupported, not applicable, or
available for a future backend. It deliberately did not add a fake pinned
allocator.

This ticket is the real backend/runtime implementation follow-up. It overlaps
with the pinned-transfer item in `tofix60`, but is narrower and should own the
actual pinned host-memory work.

## Problem

Current batchers allocate regular host memory and upload through the existing
tensor/ArrayFire paths. `pin_memory=true` is preserved as a capability request,
but current training does not allocate page-locked host buffers or prove a
different host-to-device transfer path.

Adding pinned memory without changing the actual data movement would recreate
the UI/runtime truth problem fixed in `done41`.

## Required Design

- Add an owned pinned host-memory allocation/free abstraction.
- Keep regular host allocation as the default path.
- Support CUDA pinned allocation only when the active backend/runtime can use
  it.
- Avoid forcing a hard CUDA runtime dependency on non-CUDA deployments unless
  a dependency/deployment decision explicitly accepts that cost.
- Make DataLoader/batcher ownership explicit: which component owns pinned
  staging buffers, how long they live, and when they are released.
- Keep cleanup deterministic so page-locked host memory cannot leak across
  failed runs, cancellation, or shutdown.

## Required Runtime Behavior

- `pin_memory=false` stays unchanged and uses regular host memory.
- `pin_memory=true` with an available pinned backend uses a real pinned staging
  buffer and reports `pinned_host_memory`.
- `pin_memory=true` without pinned support falls back to regular host memory
  and keeps the existing structured warning path from `done41`.
- CPU-only training reports `pinned_requested_but_not_applicable`.
- GPU/mixed training reports the actual effective transfer mode in compiler
  issues, training trace, support bundles, Training Dashboard, and Studio
  Debugger.

## Validation

- Unit test allocator ownership and fallback behavior.
- Add batcher-level tests proving pinned requests do not change batch contents
  or labels.
- Add runtime trace tests for the success path:
  - `requested=true`
  - `transfer_mode=pinned_host_memory`
  - `transfer_reason=pinned_host_memory_active`
  - backend/device metadata present where available.
- Add fallback tests for allocator failure/unavailable backend.
- Add at least one benchmark comparing regular host memory vs pinned host
  memory on a real GPU workload.

## Non-goals

- Do not move TF-IDF/materialization compute to GPU.
- Do not rewrite the full data pipeline.
- Do not make pinned memory mandatory for CUDA training.
- Do not claim a speedup without benchmark evidence.
- Do not add new UI controls before the backend path changes real data
  movement.

## Acceptance Criteria

- `pin_memory=true` changes an actual host-to-device transfer path when the
  pinned backend is available.
- Unsupported or unavailable pinned transfer keeps the structured fallback from
  `done41`.
- Pinned buffers have clear ownership and deterministic cleanup.
- CPU-only, unsupported GPU, and successful pinned GPU paths are all covered by
  tests.
- Dashboard/debugger/support-bundle surfaces show the effective transfer mode
  without requiring log inspection.
- Benchmark evidence is recorded before presenting pinned memory as a
  performance feature.
