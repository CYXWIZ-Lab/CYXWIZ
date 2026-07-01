# tofix37 - Backend Placement Follow-up: Runtime Writers, Persistent Probes, and Native Recurrent CUDA

## Purpose

Continue the GPU placement work closed in `done25` without bloating that ticket
past its core compiler/runtime contract.

`done25` established the placement spine:

- compiler-owned backend placement reports,
- recurrent CPU/GPU placement policy,
- per-device placement observation cache,
- first bounded LSTM CUDA preflight probe,
- structured backend fallback reason codes,
- generic active-device observation helpers,
- Dense runtime fallback observation writing,
- generic compiler consumption of cached ArrayFire tensor-layer fallbacks.

This ticket tracks the next GPU/JIT work that should build on that spine.

## Follow-up Work

### 1. Add more runtime fallback writers

Dense is the first non-recurrent ArrayFire runtime writer. Extend the same
observation contract to the next high-value backend paths:

- Embedding,
- activation kernels: ReLU, Sigmoid, Tanh,
- loss utility ArrayFire fallbacks,
- linear/matmul paths,
- graph-runtime tensor ops where stable shape signatures already exist.

Each writer should record:

- op type,
- backend,
- active device signature,
- dtype,
- stable shape signature,
- structured fallback reason,
- source=`runtime_fallback`,
- concise detail string.

### 2. Stabilize shape signatures per op family

Avoid ad hoc strings. Add helper builders for each family before recording
observations:

- Embedding: vocab size, embedding dim, input rank/index dtype.
- Activation: input shape and dtype.
- Loss: logits/prediction shape, target shape, reduction, dtype.
- Matmul/Linear: lhs/rhs/output dimensions and dtype.
- Graph tensor ops: input/output ranks and op-specific axes.

Compiler and runtime must use the same helper for the same op family.

### 3. Persistent per-device probe cache

The current observation cache is process-local. Add a persistent cache layer
after the in-memory contract has more runtime writers.

Cache key should include:

- CyxWiz engine/backend version,
- ArrayFire backend name,
- device signature,
- dtype,
- op type,
- shape signature,
- reason code,
- source,
- timestamp.

The compiler should trust persistent failures conservatively but should not
treat a successful synthetic probe as full safety proof.

### 4. General JIT preflight safety framework

The current bounded LSTM CUDA probe is the first proof point, not the final
architecture. Preflight should become a general safety framework for GPU/JIT
placement, while each executable probe remains operator-specific.

Add a shared JIT safety layer that owns:

- cache keys by device, backend, op type, dtype, and stable shape signature,
- structured probe outcomes: safe, unsafe, timeout, unsupported, inconclusive,
- structured failure reasons shared with runtime fallback observations,
- timeout and budget enforcement,
- conservative CPU routing when a matching unsafe observation exists,
- debugger-visible probe decisions and details.

Probe bodies must stay operator-specific because ArrayFire/CUDA generated-kernel
failure depends on the actual op graph, not just tensor allocation size.

Initial operator probe priorities:

- GRU recurrent probe, because sentiment training currently CPU-routes GRU after
  observed CUDA generated-kernel formal-parameter overflow,
- LSTM probe hardening and coverage for more sequence shapes,
- linear/matmul probe for dense tensor workloads,
- embedding probe for text models,
- convolution probe for vision models.

Do not treat a successful synthetic probe as a guarantee that the full training
step is safe. Runtime fallback observations must still write back into the same
cache so real failures teach future compiles.

### 5. Deep preflight mode

Keep normal compile fast. Add an explicit deep preflight mode for users who want
stronger GPU/JIT checks before a long training run.

Deep preflight may:

- run more representative synthetic probes,
- probe selected tensor ops beyond the initial LSTM probe,
- report per-op probe duration,
- stop probing when a timeout budget is exceeded,
- write failures to the persistent cache.

### 6. Native/fused recurrent CUDA path

ArrayFire JIT recurrent loops remain the weak point. The long-term fix is not
more string classification; it is controlled kernel boundaries.

Investigate:

- fused/native CUDA recurrent kernels,
- cuDNN-style GRU/LSTM integration if acceptable for the dependency model,
- CPU-vs-GPU deterministic correctness tests,
- gradient correctness tests,
- timeout protection,
- benchmark and profiler coverage.

GRU, BiGRU, and BiLSTM should stay conservatively CPU-routed until either a
bounded GRU-specific preflight probe proves safe for the target shape/device, or
this native/fused path is proven.

### 7. Debugger/UI surfacing

This overlaps with `tofix32`.

Expose placement observations in the Studio debugger:

- compiler placement plan,
- runtime fallback events,
- reason-code timeline,
- device signature,
- shape signature,
- source: runtime fallback vs preflight probe,
- probe scope: normal compile vs deep preflight,
- probe outcome: safe, unsafe, timeout, unsupported, inconclusive,
- "not VRAM" explanation for CUDA formal-parameter overflow.

The debugger should make backend fallback understandable without requiring users
to read raw ArrayFire/NVRTC logs.

### 8. Pinned host-memory transfer backend

`DataLoader.pin_memory` is currently serialized for compatibility and surfaced
truthfully as unsupported. That boundary was intentional: current batchers do
not allocate pinned/page-locked host memory, and the runtime does not have an
explicit pinned CPU-to-GPU transfer path.

Add the real backend only when it changes actual data movement:

- backend/runtime pinned host allocator and free path,
- batcher-owned pinned staging buffers,
- fallback to regular host memory when pinned allocation is unavailable,
- explicit host-to-device transfer points that can be profiled,
- CUDA/ArrayFire backend checks before using pinned memory,
- cleanup/shutdown ownership so pinned pages are not leaked,
- benchmark comparing regular host memory vs pinned memory on at least one
  real GPU workload.

This is GPU-pipeline work, not recurrent-kernel work. It should improve input
transfer throughput once the engine has explicit CPU-to-GPU staging, but it will
not fix CUDA generated-kernel formal-parameter overflow in GRU/LSTM recurrent
JIT paths.

## Acceptance Criteria

- At least three more non-recurrent runtime paths record placement observations.
- Compiler consumes cached observations through the generic tensor-layer path.
- Shape signatures are helper-built and shared between compiler/runtime.
- JIT preflight has a general framework with operator-specific probe bodies.
- GRU is represented explicitly as the next recurrent probe target rather than
  hidden behind the LSTM-only probe.
- Deep preflight is opt-in and bounded by timeout/budget.
- Persistent cache format is documented and versioned.
- Debugger follow-up is linked to `tofix32` rather than duplicated.
- pin_memory=true changes a real pinned host-memory transfer path, or remains
  visibly unsupported.


