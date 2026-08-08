# To Fix 10 - Memory Leak and Ownership Cleanup Roadmap

Status: Completed.

This note tracks the memory-leak and ownership cleanup completed for the
backend Tensor path. The remaining row-major Tensor/ArrayFire layout work
has moved to `tofix15.md`.

Short version:
- One Tensor operator leak was fixed on 2026-05-18.
- Tensor CPU storage now has a small RAII owner backed by
  `MemoryManager`.
- The old raw ArrayFire pointer concern has been partially resolved.
- The remaining layout-aware residency work is tracked in `tofix15.md`.

## Current State

The targeted ownership cleanup for this note is complete.

Already improved:
- Elementwise Tensor GPU operators no longer heap-allocate temporary
  `af::array` wrappers.
- `MemoryManager` now tracks live bytes instead of cumulative bytes.
- `Tensor` now stores the ArrayFire cache as `std::unique_ptr<af::array>`
  with explicit host/device freshness flags instead of a raw pointer.
- `Tensor` CPU storage now lives behind `TensorHostBuffer`, a private
  RAII owner that allocates and releases through `MemoryManager`.
- Tensor copy and move ownership now has explicit regression coverage.
- Tensor-local host/device freshness transitions now have regression
  coverage for lazy host materialization and host mutation after device
  caching.
- Standalone ReLU, Sigmoid, and Tanh GPU paths now preserve Tensor
  device residency instead of forcing immediate host copies.
- SGD and Adam GPU optimizer steps now have regression coverage proving
  parameter updates remain device-resident until host data is requested.

Moved out:
- Higher-level layer paths still need a row-major-aware device residency
  design before 2D/3D ArrayFire conversion copies can be removed safely.
- Device residency is still not a broad first-class contract outside
  Tensor itself.

## Why This Matters

The remaining ownership model is the kind that creates long-tail bugs:

- leak diagnosis is still harder than it should be
- ownership must stay centralized in the Tensor host/device wrappers
- future device-residency work will be easier if the host/device model
  is cleaned up first

This is also important for reliability:

- long-running training sessions
- repeated graph edits and reloads
- repeated debug runs
- GPU execution paths that may fail and fall back

## Leak Surfaces To Target Later

### 1. Tensor CPU storage

Status: done for the current backend pass.

`Tensor` no longer owns CPU storage through a public raw ownership pair.
The CPU buffer is hidden behind `TensorHostBuffer`, which allocates and
deallocates through `MemoryManager`. Allocation failure now raises
`std::bad_alloc` instead of allowing a later null dereference.

Coverage:
- constructor/destructor memory accounting
- copy construction
- copy assignment
- move construction
- move assignment

### 2. Tensor device cache and residency model

The header no longer exposes `af_array_` as a raw pointer. It is now a
`std::unique_ptr<af::array>` with `host_current_` and `device_current_`
freshness flags.

That is an improvement, but the broader residency model is still not a
first-class contract. Later cleanup should define the authoritative
host/device state transitions clearly and reduce unnecessary forced host
materialization.

The Tensor-local freshness transitions are now covered by tests for:
- ArrayFire-backed Tensor construction without immediate host allocation
- host data materialization on read
- host mutation invalidating a cached device array

### 3. Host-device churn

Several backend paths still do this pattern:
- create an ArrayFire array from host memory
- run GPU work
- copy the result back to host immediately

This is not a leak in the narrow sense, but it is still ownership
churn, and it creates pressure for more temporary allocations.

The first narrow cleanup removed this pattern from standalone activation
GPU branches:
- `ReLU::Forward` / `ReLU::Backward`
- `Sigmoid::Forward` / `Sigmoid::Backward`
- `Tanh::Forward` / `Tanh::Backward`

Those branches now use `Tensor::GetArray()` and return `Tensor(af::array)`,
so host memory is allocated only when callers request host data.

The optimizer audit found the GPU update branches already use
`Tensor::GetArray()` and `Tensor::SetFromArray()` rather than explicit host
copies. Regression coverage now verifies SGD and Adam keep GPU-updated
parameters device-resident until host data is requested.

The sequential/layer audit found that many remaining `.host(...)` copies
are not simple churn. They sit at row-major conversion boundaries, such as:
- `AfToTensor` for semantic 2D row-major tensors
- `AfToTensor3DRowMajor` for LSTM-style 3D tensors
- Multi-head attention caches and output tensors

Those cannot be safely replaced with plain `Tensor(af::array)` because
CyxWiz host tensors are row-major while ArrayFire arrays are column-major.
Removing those copies needs a row-major-aware device cache contract, not a
local mechanical edit.

### 4. Old documentation drift

Some docs still describe the older Tensor/device story.
Those should be updated after the ownership cleanup lands so the docs
match the actual contract.

## Completion State

1. Done: replace raw Tensor host storage with a safer owned buffer type.
2. Done: keep the ArrayFire cache as `std::unique_ptr<af::array>` and audit
   the Tensor-local host/device freshness transitions.
3. Done: audit Tensor constructors, assignment operators, and factories for
   any remaining manual ownership edge cases.
4. Done: audit layer and sequential code for repeated temporary
   host-device materialization.
5. Moved to `tofix15.md`: design a row-major-aware 2D/3D device
   residency contract before removing layout conversion host copies.
6. Done: run backend unit and focused residency verification for this
   ownership pass.

## Progress Log

### 2026-06-02

- Replaced Tensor host-buffer ownership with private `TensorHostBuffer`.
- Kept allocation accounting routed through `MemoryManager`.
- Removed Tensor-level `void* data_` and `owns_data_` ownership state.
- Added copy/move ownership regression coverage in `tests/unit/test_tensor.cpp`.
- Added ArrayFire freshness regression coverage for lazy host
  materialization and host mutation after device caching.
- Removed forced host copies from standalone ReLU, Sigmoid, and Tanh GPU
  branches.
- Added standalone activation tests, including ArrayFire lazy host
  materialization coverage.
- Fixed activation backward GPU paths so they no longer preallocate host
  gradient tensors before returning device-backed results.
- Audited optimizer GPU update branches; they already use Tensor device
  cache APIs instead of explicit host copies.
- Added SGD and Adam GPU optimizer residency regression coverage.
- Audited sequential/layer ArrayFire paths. Remaining host copies are
  mostly row-major layout conversion boundaries and should not be removed
  without a Tensor device-layout contract.
- Verified with:
  - `cmake --build build --config Debug --target cyxwiz-tests -- /m:1 /v:minimal`
  - `build\bin\Debug\cyxwiz-tests.exe`
  - `build\bin\Debug\cyxwiz-tests.exe "[activation]" --success`
  - `build\bin\Debug\cyxwiz-tests.exe "*GPU step*" --success`
  - `ctest --test-dir build -C Debug -R "Tensor" --output-on-failure`
  - `ctest --test-dir build -C Debug -R "(activation|Tensor)" --output-on-failure`
  - `ctest --test-dir build -C Debug -R "(optimizer|Tensor)" --output-on-failure`
  - `rg "[^\x00-\x7F]" "docs/Data Studio/tofix10.md" cyxwiz-backend\include\cyxwiz\tensor.h cyxwiz-backend\src\core\tensor.cpp cyxwiz-backend\src\algorithms\activations tests\unit\test_tensor.cpp tests\unit\test_activations.cpp -n`

## What This Note Is Not

This is not the debugger architecture note in `tofix9.md`.
It is also not the general backlog in `tofix.md`.

This file exists to isolate memory cleanup work so it can be tackled as
a focused pass later, with a clear boundary around ownership and leak
behavior.
