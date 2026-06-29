# tofix37 - Backend Placement Follow-up: Runtime Writers, Persistent Probes, and Native Recurrent CUDA

## Purpose

Continue the GPU placement work closed in `done25` without bloating that ticket
past its core compiler/runtime contract.

`done25` established the placement spine:

- compiler-owned backend placement reports,
- recurrent CPU/GPU placement policy,
- per-device placement observation cache,
- bounded LSTM CUDA preflight probe,
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

### 4. Deep preflight mode

Keep normal compile fast. Add an explicit deep preflight mode for users who want
stronger GPU/JIT checks before a long training run.

Deep preflight may:

- run more representative synthetic probes,
- probe selected tensor ops beyond LSTM,
- report per-op probe duration,
- stop probing when a timeout budget is exceeded,
- write failures to the persistent cache.

### 5. Native/fused recurrent CUDA path

ArrayFire JIT recurrent loops remain the weak point. The long-term fix is not
more string classification; it is controlled kernel boundaries.

Investigate:

- fused/native CUDA recurrent kernels,
- cuDNN-style GRU/LSTM integration if acceptable for the dependency model,
- CPU-vs-GPU deterministic correctness tests,
- gradient correctness tests,
- timeout protection,
- benchmark and profiler coverage.

GRU, BiGRU, and BiLSTM should stay conservatively CPU-routed until this path is
proven.

### 6. Debugger/UI surfacing

This overlaps with `tofix32`.

Expose placement observations in the Studio debugger:

- compiler placement plan,
- runtime fallback events,
- reason-code timeline,
- device signature,
- shape signature,
- source: runtime fallback vs preflight probe,
- "not VRAM" explanation for CUDA formal-parameter overflow.

The debugger should make backend fallback understandable without requiring users
to read raw ArrayFire/NVRTC logs.

## Acceptance Criteria

- At least three more non-recurrent runtime paths record placement observations.
- Compiler consumes cached observations through the generic tensor-layer path.
- Shape signatures are helper-built and shared between compiler/runtime.
- Deep preflight is opt-in and bounded by timeout/budget.
- Persistent cache format is documented and versioned.
- Debugger follow-up is linked to `tofix32` rather than duplicated.

