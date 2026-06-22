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

**Status 2026-06-07:** This file is still active, but it no longer means
"CPU fallback has not been worked on." `done2.md`, `done10.md`, and
`done15.md` closed substantial backend slices:

- Tensor host storage/residency is safer and no longer forces immediate
  CPU materialization for every ArrayFire result.
- Optimizer GPU update paths were audited and have focused residency
  regression coverage.
- CPU fallback coverage was added across factory activations, core
  losses, metric-learning losses, legacy Dense/Dropout/CNN/normalization
  layers, attention dropout, and important linalg paths such as SVD and
  eigendecomposition.
- The remaining work here is policy, inventory, diagnostics, performance
  proof, and GPU-first primitive coverage for graph/runtime paths that
  are still CPU-backed or mixed.

`TensorDot` is now graph-runtime computable. Its primitive
implementation in `Tensor::Dot` has a narrow ArrayFire-backed forward
path for Float32/Float64 1D vector dot and 2D row-wise dot, with explicit
CPU fallback for integer tensors, zero-sized tensors, CPU-only builds,
and ArrayFire failures. Graph backward still uses the existing
graph-executable gradient code rather than an ArrayFire-specific gradient
kernel.

Other tensor groups completed under `done1.md` Priority 6 still need the
same treatment one group at a time. Some older backend modules already
use ArrayFire directly, while newer tensor primitive work may be
CPU-backed for clarity and contract safety.

## Priority 1 - Inventory Actual Execution Paths

**Goal:** audit which tensor/model/data operations are truly GPU-backed,
CPU-backed, or mixed.

**Current truth:** operation-level CPU fallback is much broader than when
this file was created. The inventory should now focus on classifying
`GPU-backed`, `CPU-backed by design`, `mixed`, and `GPU-required`
behavior instead of simply searching for missing CPU fallback.

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

**Status 2026-06-22:** first graph-runtime placement inventory slice
landed in code. `GraphCompiler` now adds backend placement entries for
enabled `graph_op_node_ids` instead of reporting only sequential/model
layers.

Current graph-runtime placement truth:
- `Add`, `Multiply`, `Average`: `mixed`; they execute through Tensor
  elementwise primitives that can use ArrayFire for supported
  dtype/shape paths and CPU fallback otherwise.
- `TensorDot`: `mixed`; Float32/Float64 forward uses ArrayFire for 1D
  and 2D row-wise dot where available, while integer/unsupported paths
  and graph backward remain CPU/fallback-driven.
- `Concatenate`: `mixed`; Float32/Float64 2D concat can use ArrayFire
  through the row-major tensor bridge, while integer, higher-rank,
  CPU-only, and backend failure paths fall back to CPU.
- `TensorCompare`: `mixed`; Float32/Float64 tensor and scalar
  comparisons can use ArrayFire where available, while mixed dtype,
  unsupported dtype, CPU-only, and backend failure paths fall back to
  CPU.
- `TensorLogicalMask`: `mixed`; Float32/Float64 matching-dtype tensor
  logical operations and unary logical not can use ArrayFire, while
  mixed dtype, integer, CPU-only, and backend failure paths fall back to
  CPU.

This is surfaced through the existing Compile popup/backend placement
report rather than a new duplicate diagnostic system.

## Priority 2 - Define Backend Selection Policy

**Goal:** make CPU fallback explicit and predictable.

**Current truth:** many operations already have CPU fallback behavior,
but the policy is still scattered. This priority is about making fallback
selection and diagnostics consistent, not redoing the fallback work
already closed in `done2.md`.

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

**Status 2026-06-22:** the first narrow policy hook is the existing
backend placement capability registry. It now distinguishes graph-runtime
`mixed` and `cpu` placement with stable reason codes instead of relying
on node metadata alone:
- `graph_runtime_arrayfire_mixed`
- `graph_runtime_cpu_backed`

Remaining policy work:
- move more operation groups into this registry as they are audited,
- avoid adding warnings for normal mixed/CPU fallback unless the UI has
  made a performance promise,
- add tests for CPU-only fallback behavior at the primitive level where
  practical.

## Priority 3 - Add ArrayFire Tensor Primitive Paths

**Goal:** upgrade proven tensor primitives one group at a time.

**Current truth:** several backend/layer paths already use ArrayFire or
preserve device residency. This priority is for graph-executable tensor
primitive hot paths that remain CPU-backed after correctness landed.

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

**Status 2026-06-22:** first comparison primitive slice complete.
`cyxwiz-backend/src/core/tensor_comparison.cpp` now routes Float32 and
Float64 tensor/scalar comparisons through ArrayFire when available.
Unsupported dtypes, mixed tensor dtypes, CPU-only builds, and ArrayFire
exceptions continue through the existing CPU comparison loops. Logical
masks remain CPU-backed until dtype truthiness semantics are audited.

**Status 2026-06-22:** first scalar reduction primitive slice complete.
`cyxwiz-backend/src/core/tensor_reductions.cpp` now routes Float32 and
Float64 scalar `Sum`, `Mean`, `Max`, `Min`, and `Prod` through ArrayFire
when available. Integer reductions, empty-tensor fallback behavior,
dimension reductions, `Var`, and `Std` originally continued through the
existing CPU paths until shape/layout and numerical parity were audited.

**Status 2026-06-22:** first 2D dimension reduction primitive slice
complete. Float32 and Float64 2D `Sum`, `Mean`, `Max`, `Min`, and `Prod`
now use the row-major ArrayFire bridge for non-empty reduced axes and
preserve device residency for both keepdim and non-keepdim outputs.
Integer reductions, empty-axis identity fallback, higher-rank dimension
reductions, `Var`, and `Std` originally remained CPU-backed.

**Status 2026-06-22:** variance reduction primitive slice complete.
Float32 and Float64 scalar `Var`/`Std` now use ArrayFire where
available, and Float32/Float64 2D dimension `Var` plus `Std` through
`Var(...).Sqrt()` preserve device residency for supported row-major 2D
paths. The implementation computes population variance explicitly as
`mean((x - mean(x))^2)` so it matches the existing CPU contract instead
of relying on backend-specific variance defaults. Integer variance,
empty reductions, higher-rank dimension variance, CPU-only builds, and
ArrayFire failures continue through the existing CPU fallback.

**Status 2026-06-22:** first shape residency slice complete.
`Tensor::Reshape` now preserves native ArrayFire device data without
materializing host storage when the source tensor is already device
current, and it updates the native ArrayFire dimensions with `moddims`.
Shapes with rank above ArrayFire's four-dimensional limit still fall
back through the existing host path until reshape semantics for those
layout bridges are audited.

**Status 2026-06-22:** row-major 2D reshape residency slice complete.
`Tensor::Reshape` now preserves device residency for already
device-current row-major 2D tensors reshaped into another 2D shape. The
path reshapes the row-major linear view on device so CPU and ArrayFire
outputs keep the same row-major flatten order. Empty tensors,
rank-changing reshapes, CPU-only builds, and ArrayFire failures continue
through the existing CPU fallback.

**Status 2026-06-22:** row-major 3D reshape residency slice complete.
`Tensor::Reshape` now also preserves device residency for already
device-current row-major 3D tensors reshaped into another 3D shape. It
uses the same row-major linear view strategy as the 2D path, with
ArrayFire dimension order reversed before restoring the semantic
row-major 3D view. Empty tensors, rank-changing reshapes, CPU-only
builds, and ArrayFire failures continue through the existing CPU
fallback.

**Status 2026-06-22:** first 2D transpose residency slice complete.
`Tensor::Transpose()` now routes Float32 and Float64 2D tensors through
the row-major ArrayFire bridge and preserves device-resident output.
Integer dtypes, ArrayFire failures, and arbitrary-rank
`Transpose(dim0, dim1)` continue through the existing CPU path.

**Status 2026-06-22:** 2D dim-transpose residency slice complete.
`Tensor::Transpose(dim0, dim1)` now also routes 2D Float32/Float64 axis
swaps through the row-major ArrayFire bridge. Integer dtypes,
higher-rank transposes, CPU-only builds, and ArrayFire failures continue
through the existing CPU path.

**Status 2026-06-22:** first 2D permute residency slice complete.
`Tensor::Permute({1, 0})` now routes 2D Float32/Float64 axis swaps
through the row-major ArrayFire bridge. Identity, integer, higher-rank,
CPU-only, and ArrayFire failure paths continue through the existing CPU
implementation.

**Status 2026-06-22:** first unary/scalar elementwise primitive slice
complete. Float32 and Float64 scalar arithmetic, scalar `Pow`, matching
dtype tensor `Pow`, `Sqrt`, `Exp`, `Log`, `Abs`, `Sign`, `Clip`, and
unary negate now use ArrayFire where available and preserve
device-resident outputs. Integer paths, mixed-dtype tensor `Pow`, and
ArrayFire failures continue through the existing CPU implementation.

**Status 2026-06-22:** first concat primitive slice complete.
`Tensor::Cat` now routes Float32 and Float64 2D concatenation through
ArrayFire using the row-major tensor bridge. Integer dtypes, higher-rank
concat, CPU-only builds, and ArrayFire failures continue through the
existing CPU row-major copy path. Graph-runtime Concatenate placement is
now reported as `mixed` instead of CPU-backed.

**Status 2026-06-22:** first stack primitive slice complete.
`Tensor::Stack` now routes matching Float32/Float64 1D tensor lists
through ArrayFire for axis `0` and `-1`/`1`, producing a row-major 2D
device-resident result without going through the existing
`Unsqueeze`/`Cat` fallback path. Integer dtypes, mismatched input
shapes/dtypes, higher-rank inputs, CPU-only builds, and ArrayFire
failures continue through the existing fallback implementation.

**Status 2026-06-22:** 2D stack primitive slice complete.
`Tensor::Stack` now also routes matching Float32/Float64 2D tensor lists
through ArrayFire for axis `0`, `1`, and `-1`/`2`, producing a
row-major 3D device-resident result. The implementation expands each
input through a row-major linear device view before joining, so the
output keeps the CPU stack order. Integer dtypes, mismatched inputs,
higher-rank inputs, CPU-only builds, and ArrayFire failures continue
through the existing fallback implementation.

**Status 2026-06-22:** first logical mask primitive slice complete.
`TensorLogicalMask` now routes Float32 and Float64 matching-dtype
logical `and`/`or` plus unary logical-not through ArrayFire using the
same `value != 0` truthiness contract as the CPU implementation. Mixed
dtype, integer, broadcast materialization, CPU-only builds, and ArrayFire
failures continue through the existing CPU path. Graph-runtime
TensorLogicalMask placement is now reported as `mixed`.

**Status 2026-06-22:** first broadcast/expand primitive slice complete.
`Tensor::Expand` and `Tensor::BroadcastTo` now route same-rank 2D
Float32/Float64 expansions through ArrayFire using the row-major tensor
bridge. Rank-changing left-padding broadcasts, integer dtypes,
higher-rank tensors, empty tensors, CPU-only builds, and ArrayFire
failures continue through the existing CPU row-major materialization
path.

**Status 2026-06-22:** first index-select primitive slice complete.
`Tensor::IndexSelect` now routes non-empty 2D Float32/Float64 gathers
through ArrayFire using the row-major tensor bridge. Integer dtypes,
empty index lists, higher-rank tensors, CPU-only builds, and ArrayFire
failures continue through the existing CPU row-major gather path.

**Status 2026-06-22:** first slice primitive slice complete.
`Tensor::Slice` now routes 2D Float32/Float64 range slicing through
ArrayFire using the row-major tensor bridge, including stepped slices.
Integer dtypes, higher-rank tensors, CPU-only builds, and ArrayFire
failures continue through the existing CPU row-major slice path.

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

**Status 2026-06-22:** first forward slice complete.
`cyxwiz-backend/src/core/tensor_linalg.cpp` now routes Float32/Float64
1D and 2D `Tensor::Dot` through ArrayFire when available. The 2D path
uses the row-major Tensor/ArrayFire layout bridge and reduces along the
feature axis so the result stays `[batch, 1]`. Unsupported dtypes,
zero-sized tensors, CPU-only builds, and ArrayFire exceptions continue
through the existing CPU implementation.

Focused coverage was added in `tests/unit/test_tensor.cpp`:
- existing CPU tests still cover integer 1D dot and Float32 2D row-wise
  dot,
- new ArrayFire tests prove Float32 vector dot and Float64 row-wise dot
  keep output device-resident until host data is explicitly requested.

Remaining TensorDot work:
- add a benchmark with a realistic row-wise dot size,
- add ArrayFire-specific graph/runtime smoke coverage when the test
  environment has a stable device backend,
- decide whether graph backward should gain an ArrayFire-specific
  gradient path or stay on the current graph-executable gradient loops.

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

**Current truth:** docs should say "ArrayFire-backed where available" or
"CPU fallback available" only for verified operation groups. Avoid
reintroducing old blanket claims that the backend is uniformly GPU
accelerated or uniformly CPU-primary.

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
