# done41 - Pinned Host Memory and GPU Transfer Backend Truth

## Status

Implemented through the structured fallback path.

The runtime still does not provide a real pinned host-memory allocator. This
ticket is satisfied by making `pin_memory=true` non-silent: compiler, training
trace, support bundles, Training Dashboard, Studio Debugger, and properties
truth now report whether the request is unsupported, not applicable, or
available for a future backend.

## Problem

CyxWiz exposes a `pin_memory` option on DataLoader nodes, but the engine does
not yet provide a real pinned host-memory transfer backend.

Today this means:

- `pin_memory = false` uses normal CPU/RAM batch memory.
- `pin_memory = true` may be accepted by graph configuration.
- The option does not currently enable real CUDA pinned host-memory transfers.
- Runtime may warn that no pinned host-memory transfer backend exists yet.

This creates a truth gap between the graph property and engine behavior. Users
can enable the option and reasonably expect faster CPU-to-GPU batch transfer,
but the backend does not currently deliver that behavior.

## Expected Engine Behavior

When `pin_memory = true` and training uses a GPU backend, CyxWiz should either:

- Use a real pinned host-memory allocation and transfer path, or
- Emit a structured, visible warning that the option is unsupported on the
  current runtime/backend and training is falling back to regular host memory.

The user should not have to infer this from raw logs.

## Scope

### Backend Runtime

- Add a real pinned host-memory abstraction for GPU-bound batch transfers.
- Support CUDA pinned host allocation where the CUDA backend is active.
- Keep CPU-only execution unchanged.
- Ensure batch tensors created by DataLoader can use the pinned path when
  eligible.
- Make the transfer path explicit in runtime diagnostics:
  - `regular_host_memory`
  - `pinned_host_memory`
  - `pinned_requested_but_unsupported`
  - `pinned_requested_but_not_applicable`

### DataLoader Contract

- Treat `pin_memory` as a runtime capability request, not just a stored graph
  property.
- Report whether the request was honored.
- Report why it was not honored when unsupported or not applicable.
- Avoid silent fallback.

### Diagnostics and Debugger

- Add a structured warning/event when pinned memory is requested but unavailable.
- Include:
  - node id/name
  - backend name
  - batch size
  - feature shape when available
  - requested setting
  - effective transfer mode
  - reason for fallback
- Surface this in:
  - task logs
  - training dashboard warnings
  - Studio Debugger training/runtime trace

### Compiler and Preflight

- Add preflight capability checks for `pin_memory`.
- If a graph requests `pin_memory = true`, the compiler/preflight should show:
  - supported and will be used
  - unsupported and will fall back
  - not applicable because backend is CPU
- This should be a warning, not a hard compile failure, unless a future strict
  mode is enabled.

### UI / Properties Truth

- DataLoader properties should clearly describe `pin_memory` as:
  - GPU transfer optimization request
  - requires supported backend/runtime
  - not a materialization accelerator
- The UI should show the effective runtime status after training starts.

## Non-goals

- Do not move TF-IDF/materialization compute to GPU in this ticket.
- Do not change Arrow table materialization semantics.
- Do not implement a full GPU data pipeline rewrite.
- Do not make `pin_memory` a required setting for CUDA training.

## Acceptance Criteria

- Enabling `pin_memory` on a DataLoader no longer creates silent behavior.
- CUDA training can report whether pinned host-memory transfer is actually used.
- Unsupported pinned-memory requests produce structured warnings.
- Training Dashboard and Studio Debugger show the effective transfer mode.
- CPU-only training clearly reports that pinned memory is not applicable.
- Existing training graphs continue to run when `pin_memory = false`.

## Notes

This is related to GPU compute performance, but it is separate from TF-IDF
materialization. TF-IDF materialization currently runs on CPU/RAM. Pinned memory
only helps later when batches are transferred from host memory to GPU memory for
model computation.
