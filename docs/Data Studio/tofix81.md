# To Fix 81 - ArrayFire Execution Truth and Single-Device Compute Residency

## Status

Open - backend architecture, correctness, observability, and performance
ticket. No implementation is claimed by this document.

## Decision statement

CyxWiz tensor training must execute on one explicit, immutable ArrayFire
backend and device for the lifetime of a run. A selected CUDA, OpenCL, or CPU
device is a runtime contract, not a GUI preference or compiler annotation.

Once a host batch crosses the declared input boundary, all supported model
forward, loss, metric, backward, gradient-accumulation, and optimizer compute
must remain on that selected ArrayFire platform until a deliberately declared
output boundary. If any required operation cannot honor the selected device,
compile/preflight must reject the run with the exact unsupported operation.
The engine must not silently mix CPU and GPU computation or silently choose a
different device.

Host-side file parsing, Arrow batching, checkpoint I/O, and bounded scalar
reporting remain valid explicit boundaries. They must be distinguished from
an undeclared compute fallback.

## Why this ticket exists

The APS classifier was used as a generic production probe. It showed that
ArrayFire CUDA is initialized and executes some kernels, but the complete
training step is not device-resident:

```text
Arrow batch on host
  -> upload to selected CUDA device
  -> Dense/ReLU/Dropout forward on CUDA
  -> weighted BCEWithLogits on CPU
  -> predictions copied to host for accuracy
  -> loss gradient produced on CPU
  -> gradient uploaded for model backward
  -> gradients copied to host for accumulation/averaging
  -> gradients uploaded for the ArrayFire Adam update
```

This serial CPU/device crossing leaves both processors underutilized. Short
GPU bursts are separated by transfers and synchronization, while host work
uses only a small part of total CPU capacity. A low Task Manager graph is
therefore consistent with real CUDA kernels and a structurally mixed training
pipeline.

The issue is generic. APS and `pos_weight=59` exposed it; no APS-specific
schema, class weight, graph, or hardware rule belongs in the fix.

## Verified current truth - 2026-08-01

### Backend selection is real but not authoritative end to end

- Engine initialization tries ArrayFire CUDA, then OpenCL, then CPU.
- Device selection calls ArrayFire backend and device selection APIs.
- Preferences state, compiler placement text, runtime execution, and monitor
  reporting are not one authoritative contract.
- Compiler placement can report ArrayFire-capable layers as `gpu` even while
  the active runtime backend is CPU. It describes nominal support, not
  observed execution.

### Backend decisions can become stale

Several backend algorithms cache a first-call GPU decision in process-global
static flags such as `s_gpu_checked` and `s_use_gpu`. Linear, activation,
optimizer, transform, linear-algebra, and dimensionality-reduction paths
contain variants of this pattern. A later Preferences change can therefore
leave execution using a decision made for an earlier backend.

An execution task needs a frozen device context. Individual operations must
not discover and cache global backend state independently.

### Supported-looking training stages force host computation

- `BCEWithLogitsLoss` routes forward and backward through a host path when
  `pos_weight != 1`.
- Training accuracy reads `predictions.Data<float>()`, synchronizing and
  materializing predictions on the host.
- Gradient shape conversion and gradient accumulation/averaging use host
  `Data<float>()` loops.
- Optimizer code may then upload those gradients again for ArrayFire compute.
- Repeated eager `.eval()` calls create additional synchronization/kernel
  boundaries and can prevent useful ArrayFire JIT fusion.

Relevant implementation areas include:

- `cyxwiz-backend/src/core/engine.cpp`
- `cyxwiz-backend/src/core/device.cpp`
- `cyxwiz-backend/src/core/tensor.cpp`
- `cyxwiz-backend/src/algorithms/layers/linear.cpp`
- `cyxwiz-backend/src/algorithms/optimizers/optimizer_utils.cpp`
- `cyxwiz-backend/src/algorithms/optimizers/adam_family.cpp`
- `cyxwiz-backend/src/algorithms/losses/probability_losses.cpp`
- `cyxwiz-engine/src/core/training_executor.cpp`
- `cyxwiz-engine/src/core/backend_placement_capabilities.h`

### Existing monitoring is not compute truth

Tools > Monitor currently derives its GPU percentage from used VRAM divided
by total VRAM. That is a memory-allocation ratio, not GPU engine utilization.
It cannot prove saturation, identify the active compute device, or show
host/device transfer stalls.

### Controlled observation

On the local GTX 1050 Ti, deterministic CPU and CUDA trials produced matching
model results but CUDA was slower:

| Model and batch | CPU epoch | CUDA epoch | Observation |
| --- | ---: | ---: | --- |
| `170 -> 128 -> 64 -> 1`, batch 254 | 72.6 s | 76.7 s | CUDA about 5.6% slower |
| `170 -> 512 -> 512 -> 1`, batch 2048 | 14.1 s | 15.2 s | CUDA about 7.8% slower |

CUDA activity was visible but bursty and unsaturated. No ArrayFire fallback
message appeared. This is evidence of a mixed pipeline and excessive
synchronization, not evidence that ArrayFire CUDA is entirely inactive.

## Essential runtime contract

### 1. One immutable execution device context

Create one typed context when compile/preflight binds a run:

```text
ExecutionDeviceContext
  platform: ArrayFire
  backend: CPU | CUDA | OpenCL
  logical_device_id
  stable_device_identity
  capability_generation
  fallback_policy: ForbidUndeclaredFallback
```

The context is installed before model/tensor construction and remains stable
until the task reaches a terminal state. Preferences select the next run's
context; they do not mutate an active training task.

Remove per-operation one-time backend caches. Capability checks must consume
the bound context or a cache keyed by its generation and device identity.

### 2. One executable placement plan

The compiler must produce a placement plan from the selected device and the
actual forward/backward/optimizer capabilities of every required operation.
The runtime must consume that same plan. The GUI must not infer placement from
layer names or from generic ArrayFire availability.

The plan covers:

- dataset-to-tensor ingress;
- every model operation in forward and backward;
- selected loss and all of its configured parameters;
- training and validation metrics;
- gradient accumulation, clipping, scaling, and regularization;
- the selected optimizer and its state updates; and
- declared host output/checkpoint/reporting boundaries.

Unsupported placement is a preflight error. A runtime backend failure is a
failed task with a structured event; it is not permission to continue on CPU.

### 3. Device-resident tensor lifecycle

Tensor ownership must make residency and synchronization explicit. Internal
training code must not call host `Data<T>()` merely for convenience. Provide
ArrayFire/device implementations for supported operations and restrict host
materialization to named boundaries.

For a selected GPU run, the normal steady-state batch path is:

```text
host batch -> one ingress transfer -> device-resident training step
           -> bounded scalar/event readback when required
```

Metrics should reduce on the selected device and return only the bounded
aggregate needed by the dashboard. Gradient accumulation and optimizer state
must stay resident across batches and steps.

### 4. Runtime computation truth

Every run must record observed facts, not predictions:

- requested and effective ArrayFire backend/device;
- device name, stable identity, driver/runtime, and ArrayFire version;
- placement-plan fingerprint;
- operation/stage backend and device;
- host-to-device and device-to-host transfer count and bytes;
- explicit synchronization count and reason;
- undeclared fallback attempts and terminal result;
- per-stage duration and bounded utilization samples where supported; and
- whether the run remained single-device-resident.

Compiler placement, Tasks, Training Dashboard, Debugger, support bundle, and
persisted run history must read this shared runtime record. Do not maintain
separate truth models in each UI.

### 5. Full ArrayFire platform support

The supported tensor-training contract must be verified on:

- ArrayFire CPU;
- ArrayFire CUDA when available; and
- ArrayFire OpenCL when available.

Every enumerated compatible device must have a stable selectable identity and
must be usable for a complete supported run, including changing devices
between runs in the same engine process. Platform-specific unsupported
operations must be declared before execution.

“Full device support” in this ticket means correct selection and complete
single-device execution on one chosen device. Multi-GPU/distributed training
is a separate capability and is not implied.

## Implementation phases

### Phase 0 - Inventory and invariant tests

- Inventory all backend discovery, `Data<T>()`, ArrayFire array conversion,
  `.eval()`, synchronization, and CPU fallback sites in the supported training
  stack.
- Add a deterministic runtime probe that reproduces CPU, CUDA, and OpenCL
  selection where those platforms are installed.
- Turn the APS weighted-BCE path into a small synthetic regression fixture.

### Phase 1 - Authoritative device context

- Add the immutable execution context and stable device identity.
- Make Preferences configure the next context and show requested versus
  effective selection.
- Remove or replace process-global first-call backend caches.
- Prove CPU -> CUDA -> OpenCL -> CPU switching between runs without restart.

### Phase 2 - Placement capability and strict preflight

- Extend capability truth to forward, backward, configured loss, metrics,
  gradient transforms, and optimizer state.
- Build one placement plan and make compiler/runtime/UI consume it.
- Forbid silent CPU/GPU substitution and fail with the precise unsupported
  stage, operation, parameter combination, backend, and device.

### Phase 3 - Complete one device-resident training vertical

Make the common dense supervised stack fully resident first:

- Dense/Linear, ReLU, Sigmoid, and Dropout;
- BCEWithLogits including non-unit `pos_weight`;
- binary accuracy and loss aggregation;
- backward and shape handling;
- gradient accumulation and averaging; and
- Adam/AdamW state and parameter updates.

This phase is complete only when transfer tracing proves no undeclared
device-to-host-to-device cycle inside a training step.

### Phase 4 - Expand the supported matrix

- Extend residency to every algorithm that the capability registry marks as
  graph-training supported.
- Add CPU/CUDA/OpenCL forward, backward, optimizer, and multi-batch parity.
- Keep operations unavailable on a platform fail-closed until their complete
  path passes.
- Coordinate numerical algorithm parity with `tofix39` and `tofix73` rather
  than duplicating their reference matrices.

### Phase 5 - Observability and product truth

- Replace the misleading monitor percentage with separately named VRAM usage
  and real compute/copy utilization where the platform exposes it.
- Show the active backend/device, residency verdict, transfers, syncs, and
  fallback failures in Tasks, Dashboard, and Debugger.
- Persist the same execution record with run/checkpoint provenance.

### Phase 6 - Performance refinement

- After correctness and residency tests pass, remove unnecessary eager
  evaluations and synchronization points.
- Measure ArrayFire JIT fusion, batch sizing, transfer overlap, and kernel
  timing on representative small and large workloads.
- Define performance baselines per device class. Do not promise GPU speedup
  for workloads too small to amortize launch and transfer cost.

## Acceptance criteria

### Selection and truth

- A run records one requested and one effective ArrayFire backend/device with
  stable identity.
- Changing the selected device between runs in one process changes actual
  execution; no stale static cache preserves an earlier choice.
- Compiler placement equals the bound runtime plan and cannot report `gpu`
  while an ArrayFire CPU run is active.
- Preferences, Tasks, Dashboard, Debugger, monitor, support bundle, and run
  history show consistent facts from one execution record.

### No mixed compute

- A supported CUDA or OpenCL training step has no undeclared host compute or
  device-to-host-to-device cycle after ingress.
- Weighted BCEWithLogits, metrics, backward, gradient accumulation, and Adam
  remain on the selected device.
- CPU is a first-class explicit selection, not a silent fallback target.
- An unsupported operation or parameter combination fails before training;
  an unexpected runtime fallback attempt terminates the task visibly.

### Platform and numerical verification

- The common dense vertical passes deterministic forward, loss, metric,
  backward, accumulation, optimizer, and multi-batch parity on ArrayFire CPU,
  CUDA, and OpenCL where installed.
- Device-switch tests pass without restarting the engine.
- Transfer/synchronization counters and stage timing are testable without the
  GUI.
- CPU-only builds and machines remain supported through the explicit
  ArrayFire CPU plan and truthful capability reporting.

### Performance evidence

- Benchmarks report batch size, model shape, device identity, backend,
  transfers, syncs, stage timings, throughput, and utilization source.
- GPU performance claims are made only from complete device-resident runs.
- The monitor never labels VRAM allocation percentage as GPU compute usage.

## Relationship to existing tickets

- `tofix39` owns broad numerical computation parity and training lifecycle
  truth. This ticket consumes those references for device parity and owns
  runtime placement/residency truth.
- `done41` made unsupported pinned-memory requests visible. A real pinned
  transfer implementation may improve ingress later, but it cannot substitute
  for device-resident compute.
- `tofix73` owns algorithm availability, loss/optimizer parity inventory, and
  optional XGBoost integration. This ticket determines whether an otherwise
  supported algorithm can execute completely on the selected ArrayFire
  device.
- checkpoint v2 and persisted run history remain owned by `tofix75` and
  `tofix76`; they should persist this ticket's execution identity when those
  schemas are implemented.

## Non-goals

- hard-coding APS, its class ratio, or `pos_weight=59`;
- claiming that high utilization is required for numerical correctness;
- moving Arrow parsing, filesystem I/O, GUI rendering, or checkpoint writing
  onto a GPU;
- hiding an unsupported operation behind an automatic CPU fallback;
- adding another device abstraction beside ArrayFire;
- implementing multi-GPU or distributed training; and
- expanding the algorithm catalog before existing supported paths are
  truthful and resident.

## Recommended first implementation slice

1. Add the immutable execution context and replace cached first-call backend
   decisions in Linear, ReLU/Sigmoid, and Adam.
2. Make compiler placement consume the selected context and fail if the dense
   vertical is incomplete.
3. Implement ArrayFire weighted BCEWithLogits forward/backward, device-side
   binary metrics, and device-side gradient accumulation.
4. Add observed transfer/synchronization counters and a single-device
   residency assertion to the headless training trace.
5. Prove the synthetic weighted-binary fixture on ArrayFire CPU, CUDA, and
   OpenCL where available before changing utilization or performance code.

This slice closes the exact mixed path observed in the APS investigation while
establishing a generic invariant for every later model and device.
