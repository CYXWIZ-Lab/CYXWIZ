# pycyxwiz Tensor-First Architecture (Future Design)

**Version:** 0.1 (proposal)  
**Status:** Draft for discussion  
**Last Updated:** 2026-03-06

## 1. Executive Summary

Current Python performance issues come from mixed data paths:

- Python list -> `std::vector<std::vector<double>>` -> backend compute
- NumPy interop added recently, but core C++ APIs are still vector-first

Proposed future architecture is **Tensor-first**:

- Primary compute object in Python: `cx.Tensor`
- Primary backend object in C++: `Tensor` backed by device-native storage
- NumPy path becomes adapter/convenience, not the core execution path
- List path remains compatibility-only and is deprecated over time

This keeps CyxWiz valuable beyond NumPy by preserving device/backend orchestration and engine-integrated workflows.

## 2. Problem Statement (Current Gaps)

- Python boundary overhead from list/vector marshalling is high.
- Multiple representations (`list`, `ndarray`, `Tensor`) create unclear "best path".
- `cx.linalg` internals are not uniformly Tensor-native yet.
- Optional ArrayFire backend is abstracted, but high-performance path is not consistently exposed in Python.

## 3. Design Goals

1. Single fast path for heavy compute: `Tensor -> Tensor`.
2. NumPy interoperability with minimal copies at boundaries.
3. Backend abstraction preserved (ArrayFire today, future backends tomorrow).
4. Explicit and predictable device/dtype/shape behavior.
5. Gradual migration with backward compatibility.

## 4. Non-Goals

- Rewriting all algorithms in one release.
- Removing list APIs immediately.
- Hard-locking Python ABI to raw `af::array` types.

## 5. Target Architecture

## 5.1 Layered Model

1. Python API Layer
- User-facing `Tensor`, `linalg`, `signal`, `stats`, `timeseries`.
- Overloads accept `Tensor` first, `ndarray` second, legacy list last.

2. Interop Adapter Layer
- `ndarray <-> Tensor` conversions.
- DLPack / array-interface support where feasible.

3. Core Compute Layer (C++)
- Operation interfaces consume/return `Tensor`.
- No vector-of-vector in performance-critical interfaces.

4. Backend Kernel Layer
- ArrayFire backend implementation (current primary acceleration path).
- CPU fallback kernels.
- Future backend providers behind same interfaces.

## 5.2 Canonical Data Structures

- Python canonical: `cx.Tensor`
- C++ canonical: `cyxwiz::Tensor`
- Backend storage (internal): device-native storage handle (ArrayFire now)

### Internal Tensor Metadata

- Shape
- Strides
- DType
- Device
- Layout tag (row-major/column-major view semantics)
- Ownership/lifetime flags

## 5.3 Backend Boundary Contract

Public Python and public C++ module APIs should not expose raw `af::array` directly.

Reasons:

- Preserves backend optionality and version independence.
- Avoids hard ABI coupling to ArrayFire packaging.
- Keeps migration path open for additional providers.

Use raw backend handles only in internal kernel implementations and optional expert-level interop hooks.

## 6. API Direction

## 6.1 Python API Priority Order

For compute-heavy APIs:

1. `Tensor` overload (primary, fastest)
2. `ndarray` overload (convenience)
3. list overload (legacy compatibility)

Example target style:

```python
import pycyxwiz as cx
import numpy as np

A = cx.asarray(np.random.randn(1024, 1024), dtype=cx.float32, device="cuda")
B = cx.asarray(np.random.randn(1024, 1024), dtype=cx.float32, device="cuda")
C = cx.linalg.matmul(A, B)         # Tensor -> Tensor fast path

C_np = C.to_numpy()                # boundary conversion
```

## 6.2 `cx.asarray` and Conversion Policy

`cx.asarray(x, dtype=None, device=None, copy=False)`:

- If `x` is `Tensor`, returns view/reference when safe.
- If `x` is `ndarray`, uses direct adapter path.
- If `x` is list, converts via NumPy then Tensor (single normalized route).

## 6.3 `cx.linalg` Return Policy

- Tensor input -> Tensor output.
- ndarray input -> ndarray output (convenience overload).
- list input -> list output only for legacy overloads.

## 7. Interop Strategy

## 7.1 NumPy

- Keep explicit `Tensor.from_numpy(...)` and `Tensor.to_numpy()`.
- Add/keep ndarray overloads for common APIs.
- Prefer one boundary conversion, then stay in Tensor path.

## 7.2 DLPack / CUDA Array Interface (Planned)

Add optional zero/low-copy bridges with CuPy/PyTorch ecosystems:

- `Tensor.from_dlpack(...)`
- `Tensor.to_dlpack()`

This enables direct GPU memory handoff in mixed-stack workloads.

## 8. Performance Model

## 8.1 Fast Path

- Data enters Tensor once.
- Multiple ops execute on same backend/device without host round-trips.
- Sync only on explicit materialization or cross-device ops.

## 8.2 Slow Path (Accepted but discouraged)

- Repeated `ndarray <-> Tensor` conversions inside tight loops.
- Any list/vector conversion path for large matrix workloads.

## 9. Migration and Compatibility Plan

## Phase 0 (Done)

- NumPy ndarray overloads for major `cx.linalg` functions.

## Phase 1

- Add Tensor overloads for all major `cx.linalg` functions.
- Ensure Tensor path bypasses vector-of-vector internals.

## Phase 2

- Expand ndarray/Tensor-native pattern to `signal`, `stats`, `timeseries`.
- Add explicit conversion helpers and clear docs decision tree.

## Phase 3

- Add DLPack interop.
- Add deprecation warnings for list-path linalg usage in performance-sensitive APIs.

## Phase 4

- Optional: set Tensor-first APIs as primary docs examples and move list APIs to compatibility appendix.

## 10. Validation Plan

## 10.1 Correctness

- Cross-check Tensor and ndarray overload results against NumPy/SciPy baselines.
- Shape/dtype/device behavior tests.
- Error behavior tests for invalid shape/dtype/device combinations.

## 10.2 Performance

Benchmark matrix:

- list path vs ndarray path vs Tensor path
- sizes: small/medium/large
- CPU and GPU backends
- ops: `matmul`, `solve`, `svd`, `norm`, `eig`

Report:

- end-to-end latency
- conversion cost breakdown
- achieved backend utilization

## 11. Design Decision Record (Current)

- Decision: Keep backend abstraction, do not expose raw `af::array` as main public API.
- Decision: `Tensor` is canonical compute structure.
- Decision: ndarray is first-class interop, not core internal representation.
- Decision: list APIs remain compatibility-only, with gradual de-emphasis.

## 12. Open Questions

1. Should `eye/zeros/ones` return Tensor by default in a future major version?
2. Should ndarray overloads always return ndarray, or follow global mode settings?
3. What deprecation timeline is acceptable for list-path linalg in your ecosystem?
4. Do we want explicit stream control in Python for advanced async workflows?
