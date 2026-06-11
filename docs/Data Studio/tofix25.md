# tofix25 - Compiler-Owned Backend Placement and Preflight

## Purpose

Design the next stage of CyxWiz compiler validation so users can understand
which parts of a graph will run on GPU, which parts will run on CPU, and why.

The immediate trigger was the sentiment-analysis GRU/LSTM warning:

```text
ArrayFire GRULayer::Forward failed ... NVRTC_ERROR_COMPILATION ...
Formal parameter space overflowed (... bytes required, max 4096 bytes allowed)
... falling back to CPU
```

That warning is not only a GRU or LSTM problem. It is evidence that the compiler
needs a broader backend-placement responsibility for the whole graph.

## Goal

The compiler should become the user-facing authority for graph execution
placement before training starts.

For every executable node or layer, the compiler should be able to report:

- expected backend: GPU, CPU, mixed, or unknown,
- reason for the placement decision,
- whether the placement is required or only a fallback risk,
- whether training can continue,
- what the user can change to improve placement,
- what runtime logs should confirm.

The runtime backend still owns the actual execution. The compiler should own the
preflight explanation and should warn the user before slow CPU fallback happens.

## Why This Matters

Today, backend warnings can appear late during training. The user may see:

- a full ArrayFire/NVRTC compiler dump,
- repeated warnings every batch,
- vague CPU fallback behavior,
- slower training without a clear graph-level explanation.

The compiler already knows most of the graph shape and node properties:

- batch size,
- sequence length,
- input size,
- hidden size,
- bidirectionality,
- layer count,
- return-sequence mode,
- node type,
- data domain,
- materializer path,
- supported runtime axis.

That makes the compiler the right place to turn backend limitations into clear,
actionable graph warnings.

## Current Interim State

CyxWiz now has an initial shared recurrent placement helper used by both:

- `GraphCompiler` preflight warnings,
- backend GRU/LSTM runtime guards.

This removes scattered GRU/LSTM threshold checks and gives compiler/runtime one
policy surface.

This is still an estimator. It is not yet a true CUDA JIT probe.

## Required Future Work

### 1. Backend Placement Report

Add a structured placement report to the compiler output.

Each entry should include:

- node id,
- node name,
- node type,
- requested backend,
- expected backend,
- fallback backend,
- status: `gpu`, `cpu`, `mixed`, `risk`, `unsupported`, `unknown`,
- reason code,
- human-readable explanation,
- suggested user action.

Example reason codes:

- `cuda_jit_param_overflow_risk`,
- `arrayfire_jit_compile_failure_risk`,
- `backend_not_available`,
- `operator_cpu_only`,
- `gpu_memory_risk`,
- `unsupported_dtype_on_gpu`,
- `unsupported_shape_on_gpu`,
- `materializer_cpu_only`,
- `plugin_backend_unknown`.

### 2. Real CUDA JIT Probe

Replace or supplement estimators with a real preflight probe where practical.

The probe should:

- run a tiny representative operation for the exact node shape,
- use the active backend and device,
- catch ArrayFire/NVRTC compile failures,
- cache results by device, backend, node type, dtype, and shape signature,
- avoid changing model weights or training state,
- run only during compile/preflight, not every batch.

The probe result should be converted into compiler issues before training
starts.

### 3. Whole-Graph Backend Summary

The compiler should produce a clear summary such as:

```text
Graph backend plan:
- GPU: Embedding, Dense, Loss
- CPU: GRU recurrent step
- Mixed: Training loop, Arrow batching
- Risk: none
```

The UI should surface this in compile/start-training output, not only in logs.

### 4. Cross-Node Coverage

Do not limit this system to GRU/LSTM.

Audit and classify:

- Dense,
- Conv2D / pooling,
- Embedding,
- LSTM / GRU / RNN,
- attention nodes,
- loss nodes,
- optimizers,
- preprocessing operators,
- materializer operators,
- plugin nodes.

Every node should eventually declare or derive its backend placement capability.

### 5. Runtime Confirmation

Runtime should still confirm actual placement.

Rules:

- compiler warns before training,
- runtime emits one concise confirmation/fallback event per node/shape/backend,
- repeated per-batch NVRTC dumps should be suppressed,
- runtime fallback should include the compiler reason code when available.

### 6. UI Behavior

The user should see vivid but calm messages.

Example:

```text
GRU node "GRU 48" is valid, but its recurrent step is expected to run on CPU
instead of ArrayFire CUDA on this GPU. The generated CUDA kernel is likely to
exceed the 4096-byte formal parameter limit. Training can continue, but this
layer may be slower. Try hidden_size=32, shorter sequence length, fewer layers,
or a future fused CUDA recurrent kernel.
```

The UI should distinguish:

- compile error: graph cannot run,
- compile warning: graph can run but has fallback or performance risk,
- runtime fallback: actual backend changed during execution.

## Compiler Ownership Model

The compiler should not only translate graph nodes into training config. It
should also validate the execution contract between frontend graph, materializer,
backend, and device.

Compiler responsibilities:

- graph validity,
- schema validity,
- node runtime support,
- materializer support,
- backend placement prediction,
- backend fallback explanation,
- user-facing warnings before execution.

Backend responsibilities:

- execute operations,
- protect correctness with CPU fallback,
- expose structured failure/fallback reasons,
- provide probe APIs where exact backend behavior cannot be predicted statically.

UI responsibilities:

- show compiler issues clearly,
- expose backend placement summary,
- avoid hiding important performance fallbacks in raw logs only.

## Open Questions

- Should backend placement be part of `TrainingConfiguration`, a separate
  `ExecutionPlan`, or both?
- Should CUDA JIT probes run automatically or only when the user enables deep
  preflight?
- How long can preflight probing take before it becomes intrusive?
- Should probe cache live per project, per user, or per device?
- How should plugin nodes declare GPU/CPU capabilities?
- Should CPU fallback be allowed silently for some nodes and blocked for others?

## Recommended First Slice

1. Add a typed backend placement report to compiler output.
2. Route GRU/LSTM recurrent placement warnings through that report.
3. Suppress repeated runtime NVRTC dumps for the same node/shape/backend.
4. Show placement summary in the start-training compile panel.
5. Add tests proving compiler warnings and runtime fallback reason codes match.

## BiGRU GPU Enablement Note

Do not enable GPU placement for bidirectional GRU yet.

The compiled `GRUModule` has a promising architecture for BiGRU: split the
bidirectional layer into one forward single-direction GRU and one reverse
single-direction GRU, then concatenate their outputs. That avoids needing one
large monolithic bidirectional ArrayFire kernel.

However, an initial deterministic `BiGRUModule` reference test exposed a hang
when the split path entered the single-direction ArrayFire GRU forward path.
Until that lower-level path is proven correct and bounded, BiGRU should remain
CPU-routed with an explicit compiler/runtime warning.

Required validation before enabling BiGRU GPU:

- CPU reference tests for raw `GRULayer` bidirectional output.
- CPU reference tests for `GRUModule` split bidirectional output.
- Single-direction GRU ArrayFire forward correctness tests versus CPU.
- Split BiGRU ArrayFire correctness tests versus CPU/reference.
- Runtime timeout/fallback protection for ArrayFire recurrent probes.
- Shape/device cache so a failed GRU CUDA probe does not repeat every batch.
- Training smoke test proving no accuracy regression versus CPU for a small
  deterministic graph.

Only after these pass should compiler placement treat split BiGRU as GPU-capable.

## BiLSTM GPU Enablement Note

Do not enable GPU placement for bidirectional LSTM by default yet.

The current ArrayFire CUDA LSTM path contains bidirectional code, but sentiment
testing showed NVRTC formal-parameter overflow even for a small configuration:

```text
LSTM(in=64, hidden=8, layers=1, bidir=true, return_seq=false)
ArrayFire LSTMLayer::Forward failed ...
Formal parameter space overflowed (5160 bytes required, max 4096 bytes allowed)
```

Later batches produced even larger generated-kernel parameter footprints. That
means the static estimator was too optimistic for bidirectional LSTM. Until a
real CUDA JIT probe or a fused/native recurrent kernel exists, bidirectional
LSTM should be CPU-routed with an explicit compiler/runtime warning.

Required validation before enabling BiLSTM GPU:

- CPU reference tests for bidirectional LSTM output and gradients.
- ArrayFire single-direction LSTM correctness tests versus CPU.
- ArrayFire bidirectional LSTM correctness tests versus CPU.
- Probe or timeout protection for generated-kernel compile failures.
- One-time fallback logging for each node/shape/backend.
- Training smoke test proving no accuracy regression versus CPU.

## Why GPU Overflow Can Still Occur

The current placement policy is a conservative static estimator, not a real
CUDA JIT probe. It can still be wrong.

Observed example from sentiment training:

```text
LSTM(in=64, hidden=24, layers=1, bidir=false, return_seq=false)
batch_size=128, seq_len=128
ArrayFire LSTMLayer::Forward failed ...
Formal parameter space overflowed (4244 bytes required, max 4096 bytes allowed)
```

The compiler initially emitted no warning for that single-direction LSTM shape,
but runtime still hit ArrayFire/NVRTC overflow. This proves the estimator cannot
be the final solution.

The missing pieces are:

- a real CUDA/ArrayFire JIT probe for the exact node shape,
- a per-device/per-shape probe cache,
- a way for runtime fallback results to feed back into future compiler
  placement decisions,
- one-time fallback logging per node/shape/backend instead of repeated
  per-batch compiler dumps,
- a backend placement report that marks each layer as GPU, CPU, mixed, risk, or
  unknown,
- correctness tests proving CPU and GPU recurrent outputs match before enabling
  a shape on GPU,
- timeout protection so a probe or GPU path cannot hang training.

Until those pieces exist, compiler placement should bias conservative for
ArrayFire recurrent layers. If a recurrent shape has shown overflow in real
training, route it to CPU and explain that this is a performance fallback, not a
training failure.

## Kernel Overflow Is Not VRAM Overflow

The ArrayFire/NVRTC message:

```text
Formal parameter space overflowed (... bytes required, max 4096 bytes allowed)
```

does not mean the GPU ran out of memory.

Example from the GTX 1050 Ti sentiment run:

```text
GPU memory used: ~684 MB / 4096 MB
GPU utilization: ~24%
ArrayFire LSTMLayer::Forward failed ...
Formal parameter space overflowed (4244 bytes required, max 4096 bytes allowed)
```

The GPU still had plenty of VRAM. The failure happened before the generated
kernel could run because CUDA has a small fixed limit for kernel launch
parameters. ArrayFire JIT fusion can generate a recurrent kernel whose argument
footprint exceeds that limit even when tensor memory is available.

User-facing explanation:

- VRAM is storage for tensors and model data.
- CUDA formal parameter space is the small argument block passed to one kernel
  launch.
- A graph can have enough VRAM and still fail kernel compilation if the generated
  kernel signature is too large.

This is why the warning should not say "GPU memory is full." It should say the
generated recurrent CUDA kernel is too complex for the backend launch-parameter
limit, and training is falling back to CPU for that layer.

Mitigations:

- reduce recurrent shape complexity (`hidden_size`, `seq_len`, `layers`,
  `bidirectional`),
- add `eval()` barriers to break large ArrayFire fused expressions,
- use a fused/native CUDA recurrent kernel,
- run a real JIT probe during compiler preflight and cache the result,
- suppress repeated per-batch NVRTC dumps after the first fallback for the same
  node/shape/backend.

### How `eval()` Helps

ArrayFire normally uses lazy JIT fusion. It records operations and may combine a
long expression chain into one generated CUDA kernel.

That is usually good for performance, but recurrent layers can create very large
expression trees:

```cpp
gates = x_t + h_proj;
i = sigmoid(gates_i);
f = sigmoid(gates_f);
g = tanh(gates_g);
o = sigmoid(gates_o);
c = f * c + i * g;
h = o * tanh(c);
```

Without barriers, ArrayFire may try to fuse too much of that work into one CUDA
kernel. The generated kernel can then exceed CUDA's 4096-byte formal-parameter
limit.

`eval()` tells ArrayFire:

```text
Run this part now and materialize the result before building the next expression.
```

So instead of accumulating one huge fused kernel, the recurrent path becomes
several smaller kernels. Each smaller kernel has a smaller launch-parameter
footprint and is less likely to hit the formal-parameter overflow.

Example:

```cpp
af::array gates = x_t + h_proj;
gates.eval();

af::array i_gate = af::sigmoid(gates(af::span, af::seq(0, hidden_size - 1)));
i_gate.eval();

af::array c_next = f_gate * c_prev + i_gate * g_gate;
c_next.eval();
```

Tradeoff:

- Too little `eval()` can create oversized fused kernels that fail to compile.
- Too much `eval()` can reduce performance by preventing useful fusion.

Use `eval()` at high-risk boundaries first:

- after matmul,
- after bias add,
- after gate slicing,
- after sigmoid/tanh gate activation,
- after recurrent state updates,
- after joins/concats/slice assignments.

## Current Conservative Runtime Policy

For now, recurrent GPU placement should be treated as follows:

- `GRU + bidirectional=true`: CPU. Current ArrayFire GRU forward path supports
  single-direction only.
- `LSTM + bidirectional=true`: CPU. Current ArrayFire bidirectional LSTM path can
  overflow generated-kernel formal parameters even at small hidden sizes.
- `GRU + bidirectional=false`: CPU for ArrayFire CUDA today. Real sentiment
  training showed generated-kernel formal-parameter overflow even at
  `batch_size=64`, `seq_len=64`, `input_size=64`, `hidden_size=32`, so the
  compiler/runtime now conservatively route GRU recurrent steps to CPU until an
  exact backend probe or fused/native CUDA recurrent kernel exists.
- `LSTM + bidirectional=false`: GPU only for very small shapes until a real probe
  replaces the estimator.

The compiler and backend runtime should use the same placement policy so the UI
warning and the actual execution path do not disagree.

Current implementation slice:

- `TrainingConfiguration` now carries a typed `backend_placements` report.
- Recurrent compiler warnings include a stable placement `reason_code`.
- The compiler now emits placement entries for every compiled model/tensor layer,
  not only GRU/LSTM. Generic layers such as Embedding, Dense, Flatten, tensor
  shape ops, reductions, pooling, convolution, dropout, and activations are
  marked GPU-capable under the active ArrayFire backend with reason code
  `arrayfire_tensor_op_capable`. Compiled layer types without a precise rule are
  reported as `backend_capability_unclassified` instead of being incorrectly
  promised as GPU-capable.
- Generic and unclassified layer placement rules now live in
  `backend_placement_capabilities.h`, which is the first small step toward a
  reusable backend capability registry.
- Layer capability classification now returns an explicit kind:
  `ArrayFireTensor`, `Recurrent`, or `Unclassified`, so future backend/plugin
  placement rules can attach metadata without more ad hoc boolean checks.
- Backend placement status and generic reason strings now have named constants
  in `graph_compiler.h`, reducing string drift between compiler, runtime logs,
  UI, and tests.
- Recurrent CUDA placement reason strings now have named constants in
  `recurrent_cuda_placement.h`, keeping compiler preflight, backend fallback,
  and tests on the same reason-code contract.
- `TrainingConfiguration::SummarizeBackendPlacements()` now provides a shared
  whole-graph placement count for compiler logs, training startup logs, and the
  compile-result popup.
- Backend placement entries now expose `NeedsUserAttention()`, and training
  startup logs warn for `cpu`, `mixed`, `risk`, `unsupported`, and `unknown`
  placements. Unknown backend capability should be visible to users; it should
  not look the same as a confirmed GPU placement.
- Focused placement tests now cover both direct unknown-summary behavior and a
  real compiled graph whose layer is reported as
  `backend_capability_unclassified`.
- Graph compile logs include a backend placement plan.
- Compile findings now log at their actual severity, so warning console filters
  see placement warnings instead of missing an info-level line containing
  `[WARN]`.
- The compile result popup includes a `Backend placement` summary block.
- `TrainingExecutor` logs the backend placement plan at training start and emits
  CPU/risk/unsupported placement entries at warning severity.
- Runtime recurrent CUDA fallback classifies formal-parameter overflow and
  disables the affected ArrayFire recurrent path for the rest of the process, so
  later batches go directly to CPU instead of repeatedly dumping NVRTC output.

## 2026-06-11 GRU GPU Fallback Finding

Sentiment testing with the regularized GRU graph confirmed the broader issue:

```text
TextTokenizer max_length=64
DataLoader batch_size=64
GRU(in=64, hidden=32, layers=1, bidir=false, return_seq=false)
ArrayFire GRULayer::Forward failed ...
Formal parameter space overflowed (5508-17724 bytes required, max 4096 bytes allowed)
falling back to CPU
```

The graph compiled with `0 errors / 0 warnings`, and the GPU still had plenty of
VRAM available. GPU memory was roughly 595 MB out of 4096 MB while the recurrent
step still failed. That confirms this is not a normal GPU memory threshold
problem.

This is a backend code generation/resource problem:

- VRAM capacity decides whether tensors, model weights, and temporary buffers
  can fit in GPU memory.
- CUDA generated-kernel formal parameter space is a separate fixed-size launch
  argument block. In these runs the limit is 4096 bytes.
- ArrayFire CUDA JIT can generate recurrent kernels whose launch argument
  footprint exceeds 4096 bytes even when VRAM is mostly idle.
- A generic "GPU memory percent" threshold cannot reliably predict this class
  of failure.

An attempted local mitigation added additional `eval()` barriers inside the GRU
ArrayFire forward path. The focused GRU/LSTM tests still passed, but real
sentiment training continued to overflow. This means `eval()` barriers are a
useful mitigation but not a sufficient architectural fix for GRU GPU execution.

### Architectural Fix Direction

Do not keep patching individual recurrent failures as isolated warnings. The
engine needs a central backend placement architecture:

1. Compiler backend planner
   - Assign each node/layer a planned backend: CUDA, OpenCL, CPU, mixed, risk,
     or unknown.
   - Use node type, tensor shape, dtype, backend, device, and known capability
     rules.
   - Emit user-facing warnings before training starts.

2. Runtime placement enforcement
   - Runtime should follow the compiler placement plan instead of repeatedly
     trying CUDA and falling back every batch.
   - If runtime discovers a new backend failure, it should cache that result by
     node id, op type, shape, dtype, backend, and device.

3. Backend capability registry
   - Each op declares whether it is CUDA-capable, CPU-only, shape-limited, or
     probe-required.
   - Recurrent examples:
     - Dense: CUDA-capable when ArrayFire backend supports the dtype/shape.
     - Embedding: CUDA-capable when the index path is verified.
     - GRU ArrayFire recurrent loop: probe-required / conservative CPU for
       sentiment-scale shapes until a safer GPU implementation exists.
     - BiGRU: CPU until split forward/reverse GPU correctness and timeout
       coverage exist.
     - LSTM: restricted, with conservative CPU placement for risky shapes.

4. Generic backend failure classifier
   - Parse backend errors into structured reasons:
     - `cuda_jit_param_overflow`
     - `gpu_out_of_memory`
     - `unsupported_dtype`
     - `unsupported_shape`
     - `backend_compile_timeout`
     - `backend_internal_error`
   - Log once per node/shape/backend, not once per batch.
   - Feed the result into the placement cache so later compiles warn correctly.

5. Real GPU GRU path
   - ArrayFire JIT recurrent loops may be the wrong long-term abstraction for
     GRU/LSTM training.
   - A reliable GPU GRU likely needs a fused/native CUDA or cuDNN-style
     implementation with controlled kernel boundaries, plus CPU-vs-GPU
     correctness tests.

User-facing wording should avoid saying "GPU memory is full" for this failure.
Use language like:

```text
This GRU shape is expected to run on CPU because the current ArrayFire CUDA
recurrent implementation can exceed CUDA's generated-kernel launch-parameter
limit. This is separate from VRAM capacity. Training can continue, but the GRU
step may be slower until a fused/native CUDA recurrent kernel or backend probe
system is available.
```
