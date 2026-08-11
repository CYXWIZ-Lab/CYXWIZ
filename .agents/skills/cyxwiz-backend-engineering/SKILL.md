---
name: cyxwiz-backend-engineering
description: Apply CyxWiz backend computation guardrails when designing, implementing, optimizing, debugging, or reviewing Tensor operations, ArrayFire code, layers, losses, optimizers, metrics, training execution, device selection, placement, fallback, tracing, materialization, or performance changes in cyxwiz-backend and cyxwiz-engine. Enforce ArrayFire-first execution, explicit native C++ CPU fallback, device residency, numerical correctness, lean structure, measured optimization, and required tests before commit.
---

# CyxWiz Backend Engineering

Preserve correctness, execution truth, residency, and performance, in that
order. Make the selected ArrayFire backend do supported computation; never
recover speed by hiding native CPU work.

Read [execution-contract.md](references/execution-contract.md) before changing
device selection, Tensor ownership/layout, host access, fallback, placement,
tracing, or reporting boundaries. Read
[validation-matrix.md](references/validation-matrix.md) before planning tests
or preparing a commit.

## Non-Negotiable Invariants

- Treat the GUI Engine as ArrayFire-dependent. Keep backend-only native builds
  clearly reduced development/test configurations.
- Treat ArrayFire CPU as an ArrayFire backend, not native C++ CPU fallback.
- Try the selected ArrayFire backend first for every supported operation.
- Keep native C++ CPU paths for declared compatibility gaps, known production
  exceptions, and focused debugging. Never enter them silently.
- Bind one immutable `ExecutionDeviceContext` before model construction and
  keep requested and effective backend/device truth distinct.
- Keep large tensors device-resident from ingress through forward, loss,
  metrics, backward, accumulation, and optimizer execution.
- Permit host access only at named ingress, bounded scalar reporting,
  checkpoint/export, debug, or declared fallback boundaries.
- Use one existing semantic Tensor contract. Do not add per-layer conversion
  helpers or alternate ownership abstractions.
- Optimize from measured evidence. Record backend/device, fallback, sync count
  and bytes, workload, and throughput before and after.
- Preserve numerical, shape, dtype, reduction, and gradient semantics before
  accepting a performance result.
- Use RAII, checked shape/byte arithmetic, deterministic seeds, and bounded
  hot-loop allocation and instrumentation.
- Preserve public API, serialized graph/config, checkpoint, and plugin
  contracts unless an intentional migration is part of the ticket.
- Run focused tests and required integration/build gates before commit. Never
  claim validation that was not run.

## Engineering Workflow

### 1. Establish Current Truth

Read the implementation, callers, tests, capability/placement metadata, and
trace path before editing. Identify requested/effective device, supported
shapes and dtypes, Tensor layout transitions, host reads, device evaluations,
exception handlers, and CPU branches.

Classify each boundary as device compute, host ingress, bounded output, debug,
or native fallback. Reproduce defects with a focused fixture. Capture a
Release baseline before performance changes.

Check allocation ownership, shape-product overflow, thread/global-device
assumptions, stochastic seeds, serialization compatibility, and logging volume
when the change touches those concerns.

### 2. Define the Contract

State the successful ArrayFire path and unsupported boundary. Decide what
strict mode must do before native compute can begin. Name legitimate host
outputs and their cadence. Specify numerical and trace assertions, not only an
expected log line.

Do not remove a native path merely because ArrayFire is required at startup.
First prove all callers, shapes, algorithms, and known failure cases have an
equivalent reliable ArrayFire implementation.

### 3. Design the Smallest Coherent Change

Extend established Tensor, device, placement, fallback, and trace primitives.
Add an abstraction only when it removes real duplication or enforces a typed
boundary. Keep provider-specific code behind existing backend boundaries so a
future backend can replace ArrayFire without changing GUI or graph semantics.

Reject a second Tensor representation, process-wide cached backend decisions,
hidden host staging, duplicated CPU/accelerator algorithms when one ArrayFire
expression suffices, and speculative multi-backend frameworks.

### 4. Implement ArrayFire First

Use rank-aware semantic device accessors. Preserve layout and reuse immutable
device constants where practical. Use numerically stable formulations and
validate dimensions before launching a kernel.

Distinguish these operations:

- `GetSemanticArray()` / `SetFromSemanticArray()`: device computation;
- `ReadData<T>()`: intentional read-only host output preserving device state;
- `MutableData<T>()`: host mutation that invalidates device state;
- ArrayFire `.eval()`: device-side JIT realization, not host materialization.

Use `.eval()` deliberately to bound JIT expression depth, lifetime, memory, or
known kernel-overflow risk. Do not scatter eager evaluation without a profile
or correctness reason.

Route unavoidable native work through shared fallback policy and recording.
Include operation, reason, selected device, shape, target `native_cpu`, and
policy. Make strict mode fail before native compute.

### 5. Keep Reporting Off the Hot Path

Accumulate reductions on device. Read bounded scalars only at configured
reporting or required decision boundaries. Keep progress responsive without
demanding fresh metrics every callback. Run dataset preparation and large
materialization as background tasks, not on the GUI render thread.

Keep instrumentation bounded. Do not emit per-element, per-column, or other
high-cardinality debug logs inside production hot loops; aggregate or sample
them at an explicit cadence.

Classify checkpoint, export, debug, and scalar reads explicitly. An observed
host transfer is not automatically a defect, but an unknown or repeated large
training-time transfer is.

### 6. Validate in Layers

Run the smallest numerical and residency tests while iterating. Then run the
integration, backend matrix, Release build, live trace, and benchmark gates in
[validation-matrix.md](references/validation-matrix.md).

When fallback exists, prove compatibility mode records and completes it, while
strict mode records and rejects it before native compute. Exercise ArrayFire
CPU and every relevant installed accelerator. Skip unavailable optional
backends truthfully rather than passing under another device.

### 7. Review Before Commit

Inspect the complete diff for duplicate logic, accidental host copies,
unattributed exceptions, stale device labels, unrelated changes, and missing
tests. Run `git diff --check`, required tests, and affected Release builds.
Record benchmark and trace evidence for performance changes. Commit only the
intended files after every required gate passes.

## Completion Report

Report changed behavior and ownership boundaries, backend/device and fallback
policy tested, every validation gate run, measured sync/fallback/throughput
changes, and explicit unsupported paths or residual risks.

Do not call a path GPU-resident based only on GUI labels, placement
predictions, Task Manager, or a low fallback count. Require runtime trace and
host-transfer evidence from the actual run.
