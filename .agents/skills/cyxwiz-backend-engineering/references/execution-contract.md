# CyxWiz Backend Execution Contract

## Contents

1. Runtime architecture
2. Device and run authority
3. Native ownership and CPU fallback
4. Placement and runtime evidence
5. Tensor ownership and residency
6. ArrayFire JIT and materialization
7. Reporting and GUI behavior
8. Performance and portability
9. Prohibited patterns

## 1. Runtime Architecture

The GUI Engine requires ArrayFire to configure, link, and start. ArrayFire is
the current compute platform for supported Tensor math. A backend-only build
may omit ArrayFire only when the Engine is disabled; describe that output as a
reduced native development/test build.

Supported ArrayFire execution selections are:

- CPU: `AF_BACKEND_CPU`;
- CUDA;
- OpenCL;
- oneAPI, when the installed ArrayFire build provides it.

Metal and Vulkan may remain compatibility enum values, but they are not
ArrayFire backends. Do not advertise or activate them as ArrayFire devices.

## 2. Device and Run Authority

Apply a pending GUI device selection before model construction. Capture one
immutable `ExecutionDeviceContext` for the run with:

- execution platform;
- requested backend/device;
- effective backend/device and stable identity;
- capability generation;
- native CPU fallback policy.

Do not mutate the active process-wide ArrayFire device during a run. Disable
or queue GUI selection changes until the next run. Query current backend truth
when an operation executes; do not cache a first-call `use_gpu` decision that
survives later device switches.

Keep requested and effective fields separate. Startup activation fallback may
change the effective device, but it must not erase what the user requested.

## 3. Native Ownership and CPU Fallback

ArrayFire CPU executes the same ArrayFire operation graph on CPU. It is not a
native C++ compatibility fallback.

Use these distinct routes:

```text
Tensor/array compute
  -> selected ArrayFire backend
  -> explicit observed native C++ fallback only for a declared gap

host scalar/control ingress
  -> declared native-owned bounded operation
  -> host/CPU/UI consumer at the same cadence
```

A native-owned operation is not an ArrayFire fallback. Use that classification
only when all of these facts are true before execution:

1. the input is already host-owned and bounded to a scalar or similarly tiny
   control value;
2. the immediate consumer is host, native CPU, or UI code at the same bounded
   cadence;
3. no vector, batch, Tensor, model, or device-resident downstream consumer
   needs the result;
4. routing through ArrayFire would introduce a per-update upload, evaluation,
   synchronization, and readback rather than preserve useful residency;
5. capability and placement metadata declare the native route, operation,
   reason, bound, cadence, and consumer.

Report the effective native route in runtime evidence, but do not increment
fallback counters or describe it as recovery from ArrayFire. Strict native
fallback policy does not reject a correctly declared native-owned operation;
a separate placement policy may reject mixed execution when the run contract
requires it.

For example, evaluating one host scalar with `std::sin` at 60 Hz for a native
MuJoCo control input or UI signal scope can be native-owned. Generating a
waveform vector or batch, applying sine to a Tensor, or feeding the result to a
device-resident stage is ArrayFire-first and should use `af::sin`. Likewise,
host-originated Arrow data does not make dense linear or polynomial fitting a
native-control operation: model matrix math belongs on ArrayFire. An existing
native fitting algorithm may remain only as a declared production exception
or compatibility fallback until numerical and supported-domain parity exists.

Do not establish an arbitrary element-count threshold as the ownership rule.
Decide from residency, shape and batching, operation fusion, consumers, and
transfer/synchronization cadence. Benchmark representative fixed workloads
before claiming either route is faster.

Keep an existing native path when it covers unsupported shape/dtype/algorithm
semantics, a known reliable production exception, or focused diagnostics.
Examples such as recurrent kernel overflow require explicit policy and tests;
they do not justify bypassing ArrayFire for unrelated operations.

Every native fallback attempt must:

1. identify operation, reason, selected backend/device, shape, target, and
   policy;
2. increment run-scoped fallback evidence;
3. appear in the canonical trace and user-visible verdict;
4. fail before native compute when strict residency forbids fallback.

Do not catch an ArrayFire exception and quietly run CPU code. Do not infer that
fallback occurred from a placement prediction; only runtime observation may
claim it.

## 4. Placement and Runtime Evidence

Compile a shared `ExecutionPlacementPlan` from the same capability source used
by preflight and GUI reporting. Distinguish:

- fatal executability blockers;
- strict-residency blockers;
- executable ArrayFire stages;
- declared native-owned bounded control stages;
- declared native compatibility gaps;
- bounded output and ingress boundaries.

Treat placement as expectation and runtime trace as fact. Record stage
backend/device, context identity, fallback events, host synchronization,
transfer bytes, terminal reason, and a residency verdict.

Store run evidence under the deterministic application debug-run root. Tests
must inject isolated roots; neither production nor tests may derive canonical
truth from process current working directory.

## 5. Tensor Ownership and Residency

Classify the supported training flow as:

```text
host ingress -> device tensor -> forward -> loss -> metrics
             -> backward -> accumulation -> optimizer -> bounded output
```

Keep large intermediates and parameters on the selected ArrayFire backend.
Avoid switching between ArrayFire-native and semantic row-major layouts
through host memory. Convert device-to-device when conversion is required.

Use one rank-aware semantic Tensor accessor/setter contract across layers,
losses, activations, gradients, and optimizers. Reuse existing adapters only
when they add shape inference or real compatibility behavior.

Host access semantics:

- Use `ReadData<T>()` for a deliberate read-only host boundary. Preserve a
  valid device representation.
- Use `MutableData<T>()` for host mutation. Invalidate device state explicitly.
- Do not use compatibility `Data<T>()` in guarded successful ArrayFire paths.
- Attribute direct `af::array::host()` reads under the same output/fallback
  evidence model.

Validate shape, rank, dtype, target form, reduction, and broadcasting before
device operations. Preserve logical row-major API semantics regardless of
ArrayFire storage order.

Use RAII for ArrayFire handles and host resources. Check shape products,
strides, byte counts, indices, and narrowing conversions before allocation or
kernel launch. Avoid per-batch allocation churn when reusable state has clear
ownership, but do not introduce mutable global caches.

Keep stochastic operations reproducible under the configured seed. Preserve
random-stream and update-count semantics when changing batching, accumulation,
or evaluation cadence.

## 6. ArrayFire JIT and Materialization

ArrayFire lazily builds expressions. A device `.eval()` realizes an expression
on the selected backend; it does not imply a device-to-host transfer.

Evaluate deliberately when needed to:

- cap expression depth or kernel argument pressure;
- avoid known JIT/kernel overflow;
- establish a lifetime boundary for temporary arrays;
- make profiling stages meaningful;
- prevent repeated evaluation of a reused result.

Avoid evaluating after every primitive. Avoid using host reads as a JIT flush.
Track allocations and synchronize only when evidence or an external output
requires it.

Cache small immutable device tensors, such as class weights, when reuse avoids
per-batch upload. Invalidate or rebuild cache state when shape, values, dtype,
device context, or ownership can change.

## 7. Reporting and GUI Behavior

Keep reductions on device and read bounded scalars at one configured cadence.
Preserve exact terminal, validation, early-stopping, and checkpoint decisions.
Do not recompute or reread the same metric for logs, callbacks, and plots.

Keep progress callbacks independent of fresh metric reads. Display selected,
requested, and effective device truth accurately. Display native fallback,
native-owned control execution, host synchronization, transfer categories,
reporting cadence, and terminal residency verdict without conflating them.

Run large dataset preparation, cache export, and model materialization outside
the render thread. Name those tasks separately from Tensor host
materialization so users can distinguish preprocessing from fallback.

Bound observability overhead. Aggregate, sample, or cadence-control hot-loop
logs and trace detail; never emit per-element or per-column production logs.
Instrumentation must reveal execution truth without becoming the bottleneck.

## 8. Performance and Portability

Optimize only after correctness and trace truth exist. Use a fixed Release
fixture, seed, batch size, warmup, and measured window. Prefer multiple samples
and report median throughput. Always report exact backend/device and fallback
policy with performance numbers.

Measure at least:

- batches or samples per second;
- host-sync events and bytes per batch;
- fallback count;
- major stage timings;
- device utilization when a reliable source is available.

Do not rank CPU, CUDA, OpenCL, or oneAPI hardware from unmatched fixtures or a
single short run.

Keep ArrayFire-specific implementation behind current Tensor/device/layer
boundaries. Favor narrow contracts that can support a future provider change,
but do not build a broad multi-backend framework before a replacement exists.

Preserve public C++/Python API, graph/config serialization, checkpoint format,
and plugin contracts. When a contract must change, version or migrate it
explicitly and test old persisted inputs.

## 9. Prohibited Patterns

- Silent native CPU execution after an ArrayFire exception.
- Describing undeclared scalar CPU work as native-owned after it has executed.
- Extending the native-control route to batch, Tensor, regression, or
  device-consumed computation because the current input happens to be small.
- Labeling ArrayFire CPU as native fallback.
- Hard-coded backend/device labels in GUI or trace output.
- First-call or process-lifetime backend capability caches.
- Full Tensor host reads for scalar metrics, shape repair, layout conversion,
  gradient accumulation, or optimizer math.
- Per-layer semantic conversion helpers that duplicate Tensor behavior.
- Mutable host access used only to inspect data.
- UI callback frequency forcing compute synchronization frequency.
- High-cardinality logging, tracing, allocation, or formatting in hot loops.
- Performance claims without fixed workload and runtime execution evidence.
- Removing compatibility fallback before supported-domain parity is proven.
