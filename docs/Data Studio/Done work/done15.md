# To Fix 15 - Tensor ArrayFire Layout And Residency Contract

Status: Completed.

Completed implementation:

- Added private Tensor device-layout state for `None`, `ArrayFireNative`,
  `RowMajor2D`, and `RowMajor3D`.
- `Tensor::GetArray()` now reuses only ArrayFire-native cache data and
  rejects shapes beyond ArrayFire's four-dimensional limit explicitly.
- `Tensor::SetFromArray()` marks ArrayFire-native device layout.
- `Tensor::GetArrayRowMajor2D()` and `Tensor::FromArrayRowMajor2D()` now
  participate in the layout cache contract instead of being one-off
  conversion helpers.
- Added explicit `GetArrayRowMajor3D()`, `SetFromArrayRowMajor3D()`, and
  `FromArrayRowMajor3D()` APIs for semantic `[dim0, dim1, dim2]` ArrayFire
  views over row-major Tensor data.
- Tensor copy, move, and assignment preserve device-layout state.
- Host materialization now converts according to the cached device layout.
- Layer-local 3D row-major helpers now delegate to Tensor's 3D row-major
  contract instead of duplicating the conversion logic.
- Added Tensor tests for 3D row-major round trip, lazy row-major
  materialization, and copy/move preservation of row-major device layout.

Validation:

- `cmake --build build --config Debug --target cyxwiz-tests -- /m:1 /v:minimal`
- `build\bin\Debug\cyxwiz-tests.exe`
- Result: all 99 test cases and 596 assertions passed.

This note defines the next backend design step after the `tofix10.md`
ownership cleanup.

The goal is to make Tensor and ArrayFire layout semantics explicit before
removing the remaining host copies in higher-level layers. This must be a
careful contract pass, not a mechanical replacement of `.host(...)` with
`Tensor(af::array)`.

## Current State

Tensor host storage is row-major.

ArrayFire storage is column-major and supports up to four dimensions.

The backend now has a Tensor-local device cache:

- `Tensor::GetArray()`
- `Tensor::SetFromArray(const af::array&)`
- `Tensor::GetArrayRowMajor2D()`
- `Tensor::FromArrayRowMajor2D(const af::array&)`
- host/device freshness flags
- `std::unique_ptr<af::array>` cache ownership

This is enough for simple elementwise and optimizer paths, but it is not
enough to safely remove all host transfers in layout-sensitive layers.

## Problem

Several remaining `.host(...)` calls are not just performance churn. They
are layout conversion boundaries.

Examples:

- 2D row-major tensors used as `[batch, features]`
- 3D recurrent tensors used as `[batch, seq, features]`
- LSTM/GRU cache tensors
- Multi-head attention projected Q/K/V caches
- attention weights and context caches

Replacing those with plain `Tensor(af::array)` can preserve device
residency but silently change semantic indexing, because the Tensor shape
would look correct while the underlying ArrayFire layout contract would be
different.

## Required Contract

Tensor needs to know not just whether device data is current, but what
layout that device data represents.

A minimal contract should answer:

- Is the Tensor host layout row-major?
- Is the cached ArrayFire array a plain ArrayFire semantic array?
- Is the cached ArrayFire array a row-major semantic 2D view?
- Is the cached ArrayFire array a row-major semantic 3D view?
- Can the cached device layout be used directly by this caller?
- If not, who performs the conversion, and does that conversion allocate
  host memory or only device memory?

## Proposed Minimal Design

Add a private Tensor device-layout state.

Possible enum:

```cpp
enum class TensorDeviceLayout {
    None,
    ArrayFireNative,
    RowMajor2D,
    RowMajor3D
};
```

Expected meaning:

- `None`: no device cache is present.
- `ArrayFireNative`: `af_array_` follows normal ArrayFire dimension
  semantics. This is suitable for generic elementwise operations and
  tensors created directly from an `af::array`.
- `RowMajor2D`: `af_array_` semantically represents a row-major Tensor
  shape `[rows, cols]`.
- `RowMajor3D`: `af_array_` semantically represents a row-major Tensor
  shape `[dim0, dim1, dim2]`.

This enum should stay private until the behavior is proven. Public APIs
should remain narrow.

## Candidate Tensor APIs

Keep existing APIs:

- `GetArray()`
- `SetFromArray(const af::array&)`
- `GetArrayRowMajor2D()`
- `FromArrayRowMajor2D(const af::array&)`

Add only if needed:

- `GetArrayRowMajor3D()`
- `FromArrayRowMajor3D(const af::array&)`
- `SetFromArrayRowMajor2D(const af::array&)`
- `SetFromArrayRowMajor3D(const af::array&)`

Prefer `SetFrom...` APIs over constructors when the layout is not
ArrayFire-native. Constructors should not hide layout assumptions.

## Rules

1. `Data()` always returns row-major host memory.
2. Const `Data()` may materialize host memory from the current device
   cache.
3. Mutable `Data()` marks host memory as authoritative and invalidates
   the device cache.
4. `GetArray()` returns ArrayFire-native layout. It must not pretend that
   2D row-major host memory is directly ArrayFire-native if callers need
   semantic `[rows, cols]` behavior.
5. `GetArrayRowMajor2D()` returns an ArrayFire array whose semantic axes
   match Tensor `[rows, cols]`.
6. `GetArrayRowMajor3D()` should return an ArrayFire array whose semantic
   axes match Tensor `[dim0, dim1, dim2]`.
7. Higher-rank Tensor device conversion must be explicit. Do not silently
   truncate shapes beyond ArrayFire's four-dimensional limit.

## Implementation Order

1. Add failing tests for 2D and 3D layout contracts before changing
   internals.
2. Add private `TensorDeviceLayout` state.
3. Teach Tensor copy/move/assignment to preserve layout state.
4. Make `SetFromArray` explicitly mark `ArrayFireNative`.
5. Make row-major 2D setters/getters explicitly mark or request
   `RowMajor2D`.
6. Add row-major 3D APIs and tests.
7. Replace local layer helpers only after Tensor owns the layout
   conversion contract.
8. Remove redundant `.host(...)` calls one layer family at a time.

## Tests Needed

Tensor tests:

- 2D host data -> row-major 2D ArrayFire -> Tensor round trip
- 3D host data -> row-major 3D ArrayFire -> Tensor round trip
- copy preserves device layout
- move preserves device layout
- host mutation invalidates layout-specific device cache
- default `GetArray()` rejects or handles unsupported higher-rank shapes
  explicitly

Layer tests:

- Linear 2D GPU forward preserves row-major output
- LSTM 3D GPU forward/backward preserves `[batch, seq, hidden]`
- GRU 3D GPU forward/backward preserves `[batch, seq, hidden]`
- MultiHeadAttention forward preserves `[batch, seq, embed]`
- attention weight cache preserves `[batch, heads, q, kv]`

Regression tests:

- no host allocation before host `Data()` read for safe device-resident
  paths
- no row-major scrambling after a device-resident round trip

## Non-Goals

Do not:

- expose the layout enum publicly until there is a proven need
- convert all layers in one pass
- change Tensor host layout away from row-major
- make Tensor depend on CUDA-specific concepts
- hide ArrayFire's four-dimensional limit

## Risks

The main risk is silent correctness failure, not a crash.

The dangerous bug shape is:

```text
shape looks correct, values are stored in the wrong semantic order
```

That is why this work must be driven by shape plus value tests, especially
for 2D and 3D tensors with non-square dimensions.

## Relationship To To Fix 10

`tofix10.md` closed the safe ownership and narrow residency work:

- Tensor host storage is RAII-owned.
- Tensor ArrayFire cache is RAII-owned.
- simple Tensor/activation/optimizer paths can stay device-resident.

This note owns the remaining layout-aware residency work.
