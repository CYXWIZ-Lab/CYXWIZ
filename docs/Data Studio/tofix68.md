# To Fix 68 - Optional NVIDIA Neural-Network Fast Path (CUDA/cuDNN)

## Status

Open - long-term architecture and feasibility ticket. No implementation is
authorized by this document.

## Decision statement

Keep ArrayFire as CyxWiz's portable native-compute foundation. Add an
**optional, narrowly scoped NVIDIA provider** for selected neural-network
operations where ArrayFire JIT staging is either not reliable enough or does
not meet the measured performance target.

The provider may use CUDA and cuDNN. It is an accelerator for NVIDIA systems,
not a replacement for ArrayFire, the native CPU path, or the future external
Python runtime layer.

## Why this exists

CyxWiz chose ArrayFire so one engine code path can reach CPU, NVIDIA CUDA,
OpenCL, and oneAPI-capable environments. That remains valuable.

However, ArrayFire-first recurrent execution can construct large lazy JIT
expressions. The observed NVRTC formal-parameter-space overflow is an example:
it is a generated-kernel limitation, not VRAM exhaustion. Intentional staging
can resolve many cases, but cuDNN provides purpose-built NVIDIA DNN kernels
and recurrent implementations that avoid treating an LSTM/GRU as one arbitrary
fused element-wise expression.

Relevant external references to revalidate at implementation time:

- ArrayFire backend selection: <https://arrayfire.org/docs/unifiedbackend.htm>
- ArrayFire CUDA interoperation: <https://arrayfire.org/docs/interop_cuda.htm>
- NVIDIA cuDNN overview: <https://docs.nvidia.com/deeplearning/cudnn/latest/developer/overview.html>
- NVIDIA cuDNN recurrent performance guidance: <https://docs.nvidia.com/deeplearning/performance/dl-performance-recurrent/index.html>

## Product boundary

```text
                         CyxWiz operation plan
                                   |
                  +----------------+----------------+
                  |                                 |
          portable/default path              NVIDIA capability present
                  |                                 |
        ArrayFire CPU/CUDA/OpenCL/oneAPI      native CUDA/cuDNN provider
                  |                                 |
                  +---------- shared result/fallback contract ----------+
                                   |
                          staged ArrayFire or CPU
```

The NVIDIA provider is selected only when all of these are true:

1. the user selected an NVIDIA CUDA device;
2. the installed driver, CUDA runtime, and cuDNN version satisfy a tested
   compatibility entry;
3. the operation, dtype, layout, shape, direction, and training/inference mode
   are explicitly supported;
4. the provider passes its capability query and workspace budget check.

Otherwise CyxWiz follows the existing ArrayFire/staged/CPU execution decision.
There is no hidden precision change, batch-size reduction, or framework switch.

## Initial scope

Start with only high-value neural-network primitives that have a clear native
CyxWiz contract and a CPU correctness reference:

| Priority | Operation | First supported contract |
| --- | --- | --- |
| P1 | LSTM forward/inference | Float32, dense contiguous input, one direction, fixed documented layout |
| P2 | LSTM training | P1 plus forward/backward parity and explicit reserve/workspace ownership |
| P3 | GRU | Same restricted contract after LSTM is stable |
| P4 | convolution | Only if benchmark evidence shows a material benefit over the current path |

Bidirectional recurrent layers, variable-length packing, projections, exotic
dtypes, arbitrary layouts, arbitrary custom activations, and a generic CUDA
operator framework are out of scope for the first provider.

## Lean architecture rules

- One provider interface, not a new parallel graph compiler.
- One native library boundary, owned by the backend; no cuDNN headers exposed
  through public engine or plugin APIs.
- One capability table per released provider version, not scattered runtime
  heuristics.
- Reuse `GpuExecutionKey`, `GpuExecutionEvidence`, `GpuExecutionDecision`,
  reason codes, support bundles, and fallback policy from `tofix67.md`.
- Compile and load the provider only when its build option is enabled; load it
  only when a supported NVIDIA device is selected.
- No mandatory CUDA/cuDNN installation for CPU, OpenCL, oneAPI, or ArrayFire
  CUDA users who do not need the provider.

## Proposed internal contract

```text
INeuralNetworkProvider
  provider_id() -> "nvidia.cudnn"
  version() -> provider / CUDA / cuDNN build identity
  query_capability(NeuralOpRequest) -> Supported | Unsupported(reason)
  estimate_resources(NeuralOpRequest) -> workspace + reserve limits
  execute(NeuralOpRequest, NeuralOpBuffers) -> result or typed failure
```

`NeuralOpRequest` must be a typed, compact descriptor: operation variant,
tensor dimensions and strides, dtype, layout, training mode, algorithm policy,
and deterministic-mode request. It must not contain ArrayFire expression trees,
raw Python objects, or user-controlled CUDA pointers.

The operation owner creates the request; the provider does not inspect the
CyxWiz graph or make placement policy decisions.

## ArrayFire interoperability policy

Initial delivery may use explicit host/device boundary copies where needed.
They are easier to own, diagnose, and test than allocator sharing.

Zero-copy ArrayFire/CUDA interoperation is a later optimization only after:

- both sides use the same selected device and compatible CUDA context;
- stream ordering, buffer lifetime, and error propagation are specified;
- ArrayFire array evaluation and lock/unlock ownership are validated;
- benchmark evidence proves copying is the actual bottleneck.

Do not introduce DLPack, shared allocators, or general tensor-pointer export in
the first provider. A performance feature must not weaken memory safety or make
fallback behavior nondeterministic.

## Build, distribution, and LTS policy

### Build options

```text
CYXWIZ_ENABLE_NVIDIA_DNN_PROVIDER=OFF   # default for portable builds
CYXWIZ_REQUIRE_CUDNN=OFF                # configure fails only when explicitly requested
```

When enabled, CMake must locate supported CUDA Toolkit and cuDNN components,
record their versions in build metadata, and fail clearly on mismatches. The
core engine must continue to build without them.

### Packaging choices to decide before coding

1. **Provider add-on package (preferred):** ship the portable engine normally;
   distribute a version-matched NVIDIA provider package that declares its CUDA,
   cuDNN, driver, and OS support matrix.
2. **Separate NVIDIA engine distribution:** acceptable only if add-on loading
   proves operationally unreliable; it increases release/test burden.
3. **Static or bundled cuDNN:** do not assume this is permitted. Legal and
   release engineering must approve NVIDIA redistribution terms for each
   targeted cuDNN release before any package is published.

The provider package needs its own semantic version, compatibility matrix,
upgrade/migration rule, and rollback path. Treat CUDA, driver, and cuDNN as
external platform dependencies, not details hidden from support.

## Execution and fallback policy

```text
compile operation
  -> query NVIDIA provider
  -> supported + resource budget accepted: select native_provider
  -> unsupported/unavailable: select existing staged ArrayFire or CPU plan
  -> native runtime failure: record typed evidence; do not retry every batch
  -> use approved fallback when correctness permits, otherwise fail closed
```

Required reason codes use the shared taxonomy, for example:

- `nvidia_provider_unavailable`
- `nvidia_provider_unsupported_contract`
- `nvidia_provider_workspace_exhausted`
- `nvidia_provider_execution_failed`
- existing `cuda_jit_param_overflow` for ArrayFire JIT failures

Provider failure must never be reported as generic GPU-memory exhaustion unless
the allocation/workspace failure proves that condition.

## User experience

Users choose a normal device policy such as `Auto`, `CPU`, or an available GPU.
They do not choose cuDNN algorithms or manage CUDA streams in the normal UI.

The debugger and run details must truthfully show:

```text
Backend placement: NVIDIA neural-network provider (cuDNN)
Operation: LSTM forward, Float32, training disabled
Device: NVIDIA GPU / driver identity
Fallback: staged ArrayFire CUDA, then CPU
Provider: cyxwiz.nvidia-dnn 1.x / CUDA x / cuDNN y
```

If unavailable, display the selected alternate path and a concise reason. A
missing add-on must not make a saved graph unloadable; it becomes a placement
decision, not a broken project.

## Validation and release gates

- CPU vs provider forward parity for every supported LSTM shape/layout.
- Training gradients and state-update parity before training is supported.
- Determinism behavior explicitly documented and tested where supported by the
  selected cuDNN configuration.
- Exact-shape performance benchmarks against direct/staged ArrayFire and CPU;
  retain the provider only where it meets a documented value threshold.
- Capability, missing-runtime, version-mismatch, workspace failure, provider
  crash, cancellation, and fallback tests.
- Repeated-run leak checks for device buffers, cuDNN handles, workspace, and
  reserve-space lifetimes.
- Release-matrix smoke tests across each supported driver/CUDA/cuDNN/OS tuple.
- Full portable build and ArrayFire-only test suite must remain green without
  CUDA or cuDNN installed.

## Delivery phases

1. Approve operation contract, support matrix, packaging/licensing decision,
   benchmark threshold, and ownership model.
2. Add CMake detection and an unloadable provider skeleton with capability
   reporting only; no graph execution.
3. Implement inference-only LSTM P1 with CPU parity and diagnostics.
4. Add resource telemetry, failure evidence, and approved fallback integration.
5. Benchmark representative workloads and decide whether P2 training is worth
   its workspace, maintenance, and test cost.
6. Implement training only after the P1 release proves adoption and support
   readiness; consider GRU only after LSTM remains maintainable.

## Non-goals

- Replacing ArrayFire or making NVIDIA the required CyxWiz backend.
- Making cuDNN a public scripting API.
- Supporting arbitrary PyTorch, JAX, Flax, or Hugging Face models through this
  provider.
- Building a generic custom CUDA kernel framework.
- Automatically downloading CUDA or cuDNN components.
- Treating a cuDNN failure as an automatic reason to silently change a model's
  numerical semantics.

## Relationship to existing work

- `tofix67.md` owns cross-operation GPU resilience, error taxonomy, placement
  evidence, staged ArrayFire, and the native-provider threshold.
- `done25.md`, closed `tofix37.md`, and `tofix60.md` provide existing
  placement/fallback and recurrent-runtime foundations.
- `BACKEND_EXTERN/` owns isolated Python framework runtimes. This ticket is a
  native C++/CUDA provider and must remain separate from those workers.

