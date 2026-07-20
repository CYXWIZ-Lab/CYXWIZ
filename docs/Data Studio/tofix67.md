# To Fix 67 - Cross-Operation CUDA JIT and GPU Resource Resilience

## Status

Open - architecture, evidence, and implementation follow-up.

## Purpose

The LSTM/GRU ArrayFire failure exposed a general CyxWiz engineering problem:
any ArrayFire-first operation can encounter a CUDA compile, launch, allocation,
dtype, shape, or device failure that is not predictable from free VRAM alone.

This ticket defines one engine-wide resilience architecture. It is not an LSTM
ticket and must not create LSTM-only workarounds.

## Research finding

The observed recurrent error was:

```text
NVRTC_ERROR_COMPILATION
Formal parameter space overflowed
... bytes required, max 4096 bytes allowed
```

This is **not**:

- VRAM exhaustion;
- a Windows pagefile shortage;
- a tensor-buffer overrun;
- a CyxWiz CPU memory allocator error.

It is a CUDA generated-kernel parameter-space failure. ArrayFire builds a lazy
JIT expression tree and may fuse many operations into one generated CUDA
kernel. The generated kernel needs a parameter/argument block for buffers,
strides, offsets, dimensions, scalars, and expression metadata. For the target
device/ABI in the observed run, the generated block exceeded the 4096-byte
limit before the kernel could execute.

CUDA parameter capacity is target/device/compute-capability dependent. It is
not an indicator of free VRAM. NVIDIA PTX describes `.param` space as the
mechanism for passing kernel/function inputs; ArrayFire/NVRTC report the error
when their generated code exceeds the applicable limit.

Research references to re-check when implementation begins:

- ArrayFire JIT/fusion behavior and explicit evaluation boundaries:
  <https://arrayfire.com/blog/performance-of-arrayfire-jit-code-generation/>
- ArrayFire JIT merges expressions into a small number of kernels:
  <https://arrayfire.com/?p=8831>
- NVIDIA PTX parameter state space:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- NVIDIA CUDA device-side parameter buffer documentation (4 KB example):
  <https://docs.nvidia.com/cuda/archive/11.4.0/cuda-c-programming-guide/index.html>
- NVIDIA cuDNN recurrent-layer performance guidance:
  <https://docs.nvidia.com/deeplearning/performance/dl-performance-recurrent/index.html>

## Existing foundation - do not duplicate it

`done25.md`, closed `tofix37.md`, and `tofix60.md` already establish important
pieces:

- compiler backend placement reports;
- structured fallback reasons including `cuda_jit_param_overflow` and
  `gpu_out_of_memory`;
- device/dtype/op/shape placement observations;
- recurrent and generic fallback writers;
- compiler consumption of known unsafe observations;
- bounded operator-specific probes and deep-preflight direction;
- debugger/support-bundle placement evidence.

This ticket must extend that spine. Do not add a parallel cache, a second error
taxonomy, or node-specific log formats.

## Problem model

```text
Native CyxWiz operation
  -> ArrayFire CUDA expression / library call
  -> one of:
       JIT compilation fails
       parameter-space limit exceeded
       kernel launch or execution fails
       GPU allocation fails
       dtype/shape is unsupported
       probe exceeds its budget
       device/runtime is unavailable
  -> structured engine decision
  -> continue on a supported path or fail closed
```

The correct decision depends on operation, shape, dtype, backend, exact device,
and failure class. A generic "GPU failed, use CPU" message is not enough for
users, placement planning, or future optimization.

## Architecture decision

### 1. One GPU-resilience service

Create a small shared service/interface around the existing placement spine.
It owns only common contracts:

```text
GpuExecutionKey
  backend + device signature + engine/backend version
  operation family + operation variant + dtype + stable shape signature

GpuExecutionEvidence
  source: static_policy | preflight_probe | runtime_success | runtime_fallback
  outcome: supported | unsafe | unsupported | timeout | inconclusive
  reason code + sanitized detail + timestamp + optional benchmark summary

GpuExecutionDecision
  attempt_gpu | staged_arrayfire | native_provider | cpu | fail_closed
  user-facing explanation + evidence reference
```

It must be a lightweight typed layer over the existing capability/observation
contracts, not a giant new framework or a wrapper around every ArrayFire call.

### 2. Operator-owned execution plans

Each operation family owns its actual execution strategy and stable shape
signature. The shared service never tries to synthesize a generic GPU kernel.

Initial families:

- recurrent: LSTM, GRU, bidirectional variants, forward/backward;
- dense/matmul and related bias/activation chains;
- embedding/indexing;
- attention and normalization;
- convolution/pooling;
- activation, loss, optimizer, reduction, transform, and evaluation paths.

Every ArrayFire-first family must declare one of:

| Mode | Meaning |
| --- | --- |
| `direct_arrayfire` | Existing backend call is proven for the key. |
| `staged_arrayfire` | The operation uses intentional materialization boundaries. |
| `native_provider` | A CyxWiz native CUDA/cuDNN-style implementation owns execution. |
| `cpu` | Correct CPU implementation is selected by policy/evidence. |
| `unsupported` | No correct implementation; fail before execution. |

### 3. Deliberate staging, not blanket `eval()`

The immediate engineering answer to unsafe JIT fusion is multiple, meaningful
GPU stages. ArrayFire `eval()` is currently the mechanism for placing a JIT
materialization boundary, but it is not an instruction to materialize every
intermediate operation.

Example recurrent staging:

```text
input projection / matmul              -> materialize
recurrent projection / matmul          -> materialize
gate combination + activation          -> materialize when JIT tree is risky
cell/hidden state update               -> materialize only when required
```

This trades a few kernel launches and intermediate buffers for a compilable GPU
path. Boundaries must be selected from exact-shape benchmarks and correctness
tests. A blanket `eval()` after each expression is prohibited because it causes
unnecessary launches, synchronization pressure, and temporary allocations.

### 4. Native provider threshold

When staged ArrayFire remains unsafe or slower than a defined threshold, the
operation should graduate to a native provider:

- a CyxWiz CUDA implementation with compact parameter descriptors/pointers;
- or a tightly scoped cuDNN integration for supported recurrent contracts.

This is the long-term solution for recurrent training. Do not force the native
engine to implement every possible framework feature. Start with the exact
CyxWiz LSTM/GRU contracts that have CPU correctness and realistic workloads.

Native-provider requirements:

- compact launch interface; never pass a large generated expression graph as
  kernel parameters;
- explicit workspace ownership and allocation limits;
- device/dtype/shape capability query;
- deterministic CPU-vs-GPU forward and backward parity tests;
- benchmark gate against CPU and staged ArrayFire;
- clean CPU fallback or fail-closed unsupported result.

### 5. Evidence-driven placement loop

```text
static capability policy
  -> normal compile preflight
  -> matching unsafe evidence? route to CPU/staged/native directly
  -> optional deep preflight for selected operations
  -> runtime execution
  -> runtime success/failure evidence recorded
  -> next compile sees exact device/shape evidence
```

No successful synthetic probe proves a full training graph safe. Runtime
evidence remains authoritative for the exact operation path.

## Failure taxonomy and response policy

| Reason | Meaning | Default response |
| --- | --- | --- |
| `cuda_jit_param_overflow` | Generated parameter block exceeds target limit | Try approved staged/native path; otherwise CPU; explain not VRAM |
| `arrayfire_jit_compile_failure` | Other generated-code compile failure | Record key; bounded retry only if an alternative plan exists; otherwise CPU/fail closed |
| `gpu_out_of_memory` | Actual device allocation failure | Release temporary work, report VRAM condition, optionally reduce batch only with explicit user policy; otherwise CPU/fail closed |
| `unsupported_dtype` | Backend cannot execute dtype | Use validated dtype conversion only if contract permits; otherwise CPU/fail closed |
| `unsupported_shape` | Backend cannot execute shape/layout | Select an approved alternate plan or CPU/fail closed |
| `backend_compile_timeout` | Probe/JIT compilation exceeded budget | Cache timeout; avoid repeat; CPU/fail closed |
| `gpu_backend_exception` | Device loss, launch failure, driver/runtime issue | Mark backend unavailable for the run; do not keep retrying; CPU/fail closed |
| `backend_internal_error` | Unknown backend fault | Preserve safe diagnostics, fail conservatively, and create evidence for investigation |

Only actual allocation failures may be described as GPU-memory pressure. Never
label parameter-space overflow as "out of GPU memory" or pagefile exhaustion.

## User and debugger experience

```text
Backend placement: Staged ArrayFire CUDA
Reason: known JIT parameter-space risk for this device/shape
Evidence: runtime fallback on NVIDIA GPU / CUDA / Float32 / shape signature
Fallback: CPU if staged plan fails
```

The debugger/support bundle must show:

- planned and actual execution mode;
- backend and device signature;
- operation/dtype/shape signature;
- reason code and evidence source;
- whether the issue is VRAM, JIT compilation, shape/dtype, timeout, or device;
- staged plan name or native provider version where applicable;
- timing and memory telemetry when safely available.

## Implementation order

1. Audit existing `done25`, `tofix37`, and `tofix60` contracts; consolidate
   names only where evidence shows drift. Do not rewrite working cache code.
2. Add the typed `GpuExecutionDecision` layer over existing compiler/runtime
   placement structures.
3. Audit all ArrayFire-first operation families and assign an explicit initial
   mode: direct, staged, CPU, or unsupported.
4. Implement targeted staged plans for the highest-value expression chains;
   start with recurrent, dense/matmul, embedding, and attention.
5. Add common runtime success/failure observation recording where missing.
6. Add bounded, operator-specific deep preflight only for high-cost paths.
7. Benchmark exact workloads and select the native-provider threshold.
8. Design and implement one native recurrent provider only after the staged
   approach and benchmarks prove the need.

## Validation

- Existing native CPU results remain unchanged.
- Each staged plan has CPU-vs-GPU forward parity and, where training is
  supported, backward/gradient parity.
- Every fallback class has a deterministic mocked/unit test plus at least one
  real backend smoke where hardware is available.
- An unsafe exact key is not retried every batch or every subsequent compile.
- A true `gpu_out_of_memory` error and `cuda_jit_param_overflow` receive
  distinct reason codes, UI text, and support-bundle records.
- Normal compile does not run expensive probes; deep preflight is opt-in,
  budgeted, and cancellable.
- Native ArrayFire-free graphs and ArrayFire-disabled builds remain valid.
- No operation silently changes precision, layout, batch size, or backend to
  avoid a failure.

## Non-goals

- Do not remove ArrayFire globally.
- Do not insert `eval()` after every tensor operation.
- Do not rely on free VRAM or Windows pagefile settings to predict JIT parameter
  overflow.
- Do not make PyTorch/JAX an implicit replacement for native CyxWiz execution.
- Do not add cuDNN or custom CUDA dependencies until a narrow provider contract
  and support matrix are approved.
- Do not promise all GPU/device/framework combinations without test evidence.

## Relationship to existing work

- `done22.md` first documents the ArrayFire CUDA JIT fusion issue.
- `done25.md` establishes placement/fallback/cache foundations.
- `tofix37.md` closes additional runtime writers, probes, and debugger
  surfacing; it remains historical evidence.
- `tofix60.md` carries current native/fused recurrent and GPU pipeline follow-up.
- `tofix32.md` owns broader performance/debugger visualization work.

