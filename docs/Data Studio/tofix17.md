# To Fix 17 - GPU Execution And CPU Fallback

**Created:** 2026-06-04
**Source:** Follow-up work split out from `done1.md` while completing
Priority 6 graph-runtime tensor work. The immediate trigger was
`TensorDot`, but the issue is broader: many tensor/model paths are
correct and testable first, then should become ArrayFire/GPU-backed only
after the runtime contract is stable.

## Boundary

This file tracks GPU execution support and CPU fallback policy. It is
not part of the active Priority 6 correctness flow unless a specific
GPU behavior is required to make a node correct.

Do not mark a node "GPU accelerated" based on metadata, ArrayFire being
linked, or global backend selection alone. A node is GPU-backed only when
its hot-path implementation operates on ArrayFire/device arrays without
unnecessary host round trips, has a documented CPU fallback, and has a
focused test or benchmark proving the selected path.

## Current Observation

`TensorDot` is now graph-runtime computable, but its current primitive
implementation in `Tensor::Dot` uses host loops over `Tensor::Data<T>()`.
That makes the operation correct and trainable, but not GPU-first.

This is likely true for other tensor groups completed under
`done1.md` Priority 6 as well. Some older backend modules already use
ArrayFire directly, while newer tensor primitive work may be CPU-backed
for clarity and contract safety.

## Priority 1 - Inventory Actual Execution Paths

**Goal:** audit which tensor/model/data operations are truly GPU-backed,
CPU-backed, or mixed.

**Scope candidates from `done1.md`:**
- tensor shape/index: reshape, view, squeeze, unsqueeze, permute,
  broadcast, expand, index select,
- tensor unary/scalar math: abs, exp, log, sqrt, pow, clip, sign,
- tensor reductions: sum, mean, max, min, prod, var, std,
- graph fan-in: add, multiply, average, concatenate,
- mask/compare: scalar compare, unary logical mask, two-input compare,
  binary logical mask,
- linalg: TensorDot now; TensorBatchMatMul remains deferred,
- sequential model modules touched by graph executable handoff.

**Completion criteria:**
- a small table or doc note listing CPU/GPU/mixed status per group,
- no assumptions from node metadata alone,
- one owner file/path named for each execution path,
- gaps moved into this file instead of mixed into active correctness
  work.

## Priority 2 - Define Backend Selection Policy

**Goal:** make CPU fallback explicit and predictable.

**Policy to design:**
- prefer ArrayFire implementation when the active backend and dtype are
  supported,
- fall back to CPU when ArrayFire is unavailable, unsupported, or would
  force unsafe shape/layout conversion,
- log or surface the fallback only when it affects a user-visible
  performance promise,
- never silently change numerical semantics between CPU and GPU paths.

**Completion criteria:**
- one backend policy helper or narrow utility, not scattered ad hoc
  checks,
- clear behavior for CPU-only builds,
- clear behavior for CUDA/OpenCL backend failures,
- tests for GPU-unavailable fallback that do not require a real GPU.

## Priority 3 - Add ArrayFire Tensor Primitive Paths

**Goal:** upgrade proven tensor primitives one group at a time.

**Start with low-risk groups:**
- elementwise unary/scalar math,
- same-shape Add/Multiply/Average,
- row-wise reductions where ArrayFire semantics map directly.

**Then handle:**
- `TensorDot`: 1D dot via `sum(a * b)`, 2D row-wise dot via
  per-row reduction to `[batch, 1]`,
- concatenate/split only after layout and dim semantics are verified,
- index/broadcast operations only after backward accumulation matches
  CPU behavior.

**Completion criteria:**
- CPU and ArrayFire outputs match for focused shape/dtype cases,
- backward gradients match existing CPU contracts where training uses
  the op,
- no host/device round-trip inside the hot loop except at API
  boundaries,
- unsupported dtypes have explicit fallback or error behavior.

## Priority 4 - TensorDot GPU Slice

**Goal:** make the newly exposed `TensorDot` contract GPU-first without
changing its user-facing semantics.

**Current CPU contract:**
- 1D + 1D same length -> scalar tensor `{1}`,
- 2D + 2D same shape `[batch, features] -> [batch, 1]`,
- graph backward supports Float32/Float64 training gradients.

**GPU work:**
- implement ArrayFire forward for Float32/Float64 first,
- verify row-wise shape orientation with ArrayFire dimensions,
- keep CPU fallback for integer tensors and unsupported backends,
- add tests that compare CPU and ArrayFire paths where ArrayFire is
  available, plus fallback tests that run everywhere.

**Completion criteria:**
- direct tensor tests for 1D and 2D dot,
- graph executable forward/backward tests still pass,
- no change to Studio metadata or compiler behavior unless performance
  wording is added.

## Priority 5 - Graph Runtime Device Residency

**Goal:** avoid graph execution bouncing tensors between CPU and GPU at
each node.

**Questions to answer:**
- does `GraphExecutableModel` cache host tensors, device tensors, or a
  wrapper that can hold either,
- where should ArrayFire conversion happen for DataInput batches,
- how do shared-input gradients accumulate without forcing host copies,
- should graph-op nodes use backend `Tensor` primitives or lower
  directly to ArrayFire modules.

**Completion criteria:**
- one narrow tensor residency contract,
- no broad rewrite of `SequentialModel`,
- sequential training behavior remains unchanged unless explicitly
  tested,
- graph fan-in tests cover both shared-input accumulation and normal
  independent inputs.

## Priority 6 - Benchmarks And Diagnostics

**Goal:** prove GPU work improves real workloads instead of only adding
code paths.

**Candidate checks:**
- focused microbenchmarks for TensorDot, elementwise ops, reductions,
  and merge ops,
- one small GUI/runtime training smoke graph with GPU backend active,
- logs that identify backend/fallback in debug builds or diagnostics.

**Completion criteria:**
- before/after numbers for at least one realistic tensor size,
- no performance promise in UI unless measured,
- fallback diagnostics are useful without being noisy.

## Priority 7 - Frontend And Documentation Wording

**Goal:** keep Studio wording honest.

**Rules:**
- implemented means executable, not necessarily GPU-accelerated,
- GPU-backed should appear only for verified paths,
- CPU fallback should be described as normal behavior, not a failure,
- node dialogs/tooltips should not expose backend internals unless they
  affect expected runtime behavior.

**Completion criteria:**
- node metadata and help text do not overclaim GPU support,
- build help explains CPU fallback and ArrayFire backend selection,
- any GPU diagnostics have a stable user-facing meaning.

## Deferred

- `TensorBatchMatMul` remains deferred until its graph-runtime shape and
  gradient contract are implemented. GPU support should come after that,
  not before it.
- Advanced linalg, broadcasting gradients, and mixed precision should be
  handled only after the simpler TensorDot/elementwise/reduction paths
  are proven.
