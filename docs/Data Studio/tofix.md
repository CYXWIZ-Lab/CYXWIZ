# To Fix - Known Issues & Deferred Work

Tracked issues and design inconsistencies found during Phase 0-1
development. Each entry has a severity, root cause, and suggested fix.

## Backend Issues

### Backend compute library: memory tracking, leak risk, and GPU performance debt

**Status:** Completed 2026-06-02. The original structural problem is now
closed: Tensor host storage is tracked through `MemoryManager`, Tensor
has explicit host/device freshness, ArrayFire cache ownership, layout-aware
2D/3D row-major conversion APIs, optimizer update paths keep parameters
and state device-resident, activation paths no longer force host copies,
and the layer/loss conversion helpers delegate row-major layout handling
to Tensor instead of duplicating conversion code.

**Severity:** High - this affects the credibility of the ML backend,
runtime speed, and our ability to diagnose leaks correctly.

**Context:** Static review of the backend compute library in
`cyxwiz-backend` shows that the main problems are structural, not
isolated. The current Tensor abstraction is CPU-owned, many GPU paths
copy to and from host on every operation, and memory tracking does not
reflect real live allocations.

**Status update 2026-06-02:** First tensor-residency pass landed.
`Tensor` now keeps a cached ArrayFire array with host/device validity
flags, `GetArray()` reuses current device state instead of rebuilding
from host every call, `SetFromArray()` and GPU factories/operators keep
results device-resident until `Data()` is requested, and tensor host
buffers are routed through `MemoryManager`. This reduces repeated
host-device churn in generic tensor operations and makes core tensor
host allocations visible to memory diagnostics. Remaining work is still
substantial: layer-local `AfToTensor` helpers, optimizer state, and
module caches still need to stop materializing back to host after every
step.

**Status update 2026-06-02, follow-up:** Activation Tensor/ArrayFire
helpers now use `Tensor::GetArray()` and `Tensor(arr)`, so activation
forward/backward paths can reuse cached device state and return
device-resident results. The layer/loss `AfToTensor` helpers also keep
non-2D ArrayFire outputs resident. The layout-sensitive 2D transpose
paths still materialize to host until CyxWiz has an explicit row-major
device-view contract.

**Status update 2026-06-02, 2D contract:** `Tensor` now exposes
central ArrayFire conversion APIs for row-major 2D tensors:
`GetArrayRowMajor2D()` and `FromArrayRowMajor2D()`. The duplicated
2D conversion code in `layer.cpp` and `loss.cpp` now routes through
these methods, and a tensor unit test verifies that a row-major
`[rows, cols]` tensor round-trips through ArrayFire without layout
scrambling. `FromArrayRowMajor2D()` still materializes to host because
`Tensor` does not yet carry a device-layout enum that would allow
device dims `[cols, rows]` to represent logical shape `[rows, cols]`.

**Status update 2026-06-02, optimizer residency:** Optimizer GPU
paths now use `Tensor::GetArray()` and `SetFromArray()` instead of
constructing ArrayFire arrays from `Data()` and copying updated
parameters/state back through `host(...)` every step. SGD, Adam,
AdamW, RMSprop, AdaGrad, NAdam, Adadelta, and LAMB now keep updated
parameters and optimizer buffers device-resident until host data is
explicitly requested. Optimizer state initialization now uses
`Tensor::Zeros()` so state tensors can follow the same residency
contract. Added focused SGD/Adam update tests alongside the existing
tensor residency tests.

**Status update 2026-06-02, final:** Tensor now owns the layout-aware
ArrayFire residency contract. A private device-layout state distinguishes
no cache, ArrayFire-native cache, row-major 2D cache, and row-major 3D
cache. `GetArrayRowMajor3D()`, `SetFromArrayRowMajor2D()`,
`SetFromArrayRowMajor3D()`, and `FromArrayRowMajor3D()` were added, and
copy/move/assignment preserve the device-layout state. Host materialization
now converts according to the cached layout. Focused tests cover row-major
2D/3D round trips, non-constant ArrayFire-native construction and
`SetFromArray()`, lazy host materialization, host mutation invalidation,
activation residency, and optimizer residency. Validation passed with
`build\bin\Debug\cyxwiz-tests.exe`: 101 test cases and 625 assertions.

#### 1. MemoryManager is misleading and not suitable for leak diagnosis

**Status:** Resolved 2026-05-18. `MemoryManager` now records
allocation sizes per pointer and decrements the live counter on
deallocation, so `GetAllocatedBytes()` reports current live bytes
instead of cumulative bytes.

**Files:**
- `cyxwiz-backend/src/core/memory_manager.cpp`
- `cyxwiz-backend/include/cyxwiz/memory_manager.h`

**Issue:** `MemoryManager::Allocate()` increments
`g_allocated_bytes`, but `MemoryManager::Deallocate()` does not
decrement it because allocation sizes are not tracked. As a result,
`GetAllocatedBytes()` behaves more like "bytes ever allocated through
this API" than "current live bytes." Worse, most of the hot tensor
paths do not use `MemoryManager` at all - they use direct
`malloc/free`, so the tracker misses the actual core allocations.

**Impact:**
- leak counters are not trustworthy
- peak memory reporting is noisy
- backend diagnostics can falsely suggest leaks or hide real ones

**Suggested fix:**
1. Either remove the current memory tracker entirely until it is
   honest, or store size metadata per allocation and decrement on
   deallocation.
2. Route Tensor core allocations through one tracked allocator.
3. Separate host-memory tracking from device-memory tracking.

#### 2. Tensor is CPU-owned, so GPU ops repeatedly bounce through host memory

**Status:** Resolved 2026-06-02. Tensor is no longer purely CPU-owned.
It keeps host and ArrayFire device state with explicit freshness flags,
uses lazy host materialization, and routes ArrayFire result tensors through
device-resident constructors/setters until `Data()` is requested.

**Files:**
- `cyxwiz-backend/include/cyxwiz/tensor.h`
- `cyxwiz-backend/src/core/tensor.cpp`

**Issue:** `Tensor` stores CPU memory in `data_`, while
`GetArray()` creates a fresh ArrayFire array and copies host data into
it every time. `SetFromArray()` then copies the result back to CPU.
This means the backend does not keep tensors resident on device across
operations.

**Symptoms in code:**
- `GetArray()` always constructs a new AF array
- `SetFromArray()` always copies device data back to CPU
- factory methods such as `Zeros`, `Ones`, and `Random` do GPU work
  and immediately copy the result to host

**Impact:**
- unnecessary host->device and device->host copies
- poor scaling across chained linalg/training operations
- "GPU acceleration" often behaves like temporary offload, not true
  device-native execution

**Suggested fix:**
1. Redesign `Tensor` with persistent device residency.
2. Add host/device dirty flags and explicit synchronization points.
3. Make `GetArray()` return cached device state when valid instead of
   recreating arrays from host memory every time.

#### 3. Tensor GPU operator implementations leak heap objects on exceptions

**Status:** Resolved 2026-05-18. The elementwise GPU operators now
use stack `af::array` values instead of heap-allocated wrappers, so
ArrayFire exceptions no longer leak operator-side heap objects.

**File:** `cyxwiz-backend/src/core/tensor.cpp`

**Issue:** The Tensor elementwise operators (`operator+`, `operator-`,
`operator*`, `operator/`) allocate `af::array*` with `new` and only
delete them on the success path. If an ArrayFire exception is thrown
before cleanup, the catch block falls through to CPU without releasing
those heap allocations.

**Impact:**
- exception-path leaks
- long-running sessions can accumulate leaked `af::array` wrappers
- hard-to-reproduce growth when GPU paths fail intermittently

**Suggested fix:**
1. Replace raw `af::array*` heap allocation with stack `af::array`
   values or `std::unique_ptr`.
2. Remove manual cleanup patterns in favor of RAII.
3. Audit all ArrayFire exception paths for identical issues.

#### 4. Tensor API advertises device transfer, but implementation is missing

**Status:** Resolved 2026-05-18. The dead `Tensor::ToDevice` /
`Tensor::ToCPU` declarations were removed from the public Tensor
header, and the backend README example was updated so it no longer
advertises an unsupported transfer API.

**File:** `cyxwiz-backend/include/cyxwiz/tensor.h`

**Issue:** `Tensor` declares `ToDevice(Device*)` and `ToCPU()`, but
there is no implementation in the backend source for these methods.

**Impact:**
- API suggests true device management exists when it does not
- encourages design drift and confusion about ownership semantics
- makes optimization harder because the abstraction boundary is not
  honest

**Suggested fix:**
1. Either implement `ToDevice` / `ToCPU` for real device residency
   management, or remove them until the design is complete.
2. Make device transition semantics explicit in the Tensor contract.

#### 5. Cached `af_array_` state is mostly unused and adds complexity

**Status:** Resolved 2026-06-02. The cached ArrayFire state is now part
of the Tensor residency contract. Cache validity is tracked through
host/device freshness flags and a private device-layout tag, and callers
use explicit row-major 2D/3D APIs when semantic Tensor axes matter.

**Files:**
- `cyxwiz-backend/include/cyxwiz/tensor.h`
- `cyxwiz-backend/src/core/tensor.cpp`

**Issue:** `Tensor` owns an `af_array_` pointer, but `GetArray()`
ignores it and constructs a new ArrayFire array from CPU data each
time. The code is carrying both host ownership and partial device
ownership without using either model cleanly.

**Impact:**
- extra complexity without performance benefit
- higher risk of stale-state bugs during future optimization work

**Suggested fix:**
1. Decide whether Tensor is host-primary or device-primary.
2. Remove dead cached state if not used.
3. If caching remains, make it authoritative and synchronized.

#### 6. Optimizer GPU paths still copy parameter state back to CPU every step

**Status:** Resolved 2026-06-02. The optimizer GPU update
paths now update parameters and optimizer buffers via Tensor's
ArrayFire residency APIs and no longer call `host(...)` in the GPU
step path. Future end-to-end profiling can still improve placement
decisions, but the original per-step forced host copy issue is closed.

**File:** `cyxwiz-backend/src/algorithms/optimizer.cpp`

**Issue:** SGD, Adam, and related optimizers create ArrayFire arrays
from host parameter buffers, update them on GPU, then copy the updated
parameters and optimizer state back to CPU every step.

**Impact:**
- heavy per-step host-device traffic
- GPU advantage is reduced on small and medium training workloads
- optimizer state never stays resident on device

**Suggested fix:**
1. Keep parameters and optimizer buffers on device across steps.
2. Only sync to host at explicit boundaries (serialization, logging,
   Python conversion, UI inspection).

#### 7. Layer forward/backward code repeats the same host-device churn

**Status:** Resolved for the shared Tensor/layer conversion contract
2026-06-02. Activation, loss, Linear, recurrent 3D helper, and generic
layer conversion paths now use Tensor residency and row-major conversion
APIs instead of duplicating host-copy conversion code. A few specialized
layer families may still choose host materialization for algorithm-specific
fallbacks or caches, but the original shared conversion debt is closed.

**Files:**
- `cyxwiz-backend/src/algorithms/layers/linear.cpp`
- `cyxwiz-backend/src/algorithms/sequential.cpp`
- plus similar patterns in activation/loss/layer code

**Issue:** Many layer implementations do GPU math and immediately copy
results back to CPU tensors. Backward paths do the same for
gradients, cached activations, and running statistics.

**Examples:**
- Linear layer forward copies AF output back to CPU result tensor
- Linear backward copies gradients and grad_input back to CPU
- BatchNorm rebuilds running stats from GPU results into new CPU
  tensors

**Impact:**
- chained modules pay repeated synchronization costs
- training throughput drops as graphs get deeper
- cache tensors do not stay where the computation is happening

**Suggested fix:**
1. Move module caches and intermediate training state to device.
2. Defer host materialization until needed for serialization or UI.
3. Audit all hot-path modules for avoidable `host(...)` calls.

#### 8. Core conclusion

This was not a "one leak in one file" situation. The backend compute
library previously behaved as:

- CPU-owned Tensor model
- temporary GPU offload for many operations
- frequent host-device synchronization
- inaccurate memory accounting

That combination has been addressed at the Tensor residency and shared
ArrayFire conversion boundary. Remaining performance work should now be
tracked as layer-specific optimization, not as this foundational backend
debt.

#### Recommended fix order

1. Done - rework `Tensor` residency and synchronization model.
2. Done - replace exception-prone raw ArrayFire heap allocations with RAII.
3. Done - make memory tracking truthful and route core allocations through it.
4. Done - keep optimizer state and shared layer intermediates device-resident
   where the Tensor contract can represent the layout safely.
5. Follow-up - targeted profiling / ASan / leak-check runs can now be used
   for measurement rather than structural cleanup.

### ~~Forward pass crash for image training~~ RESOLVED (22902ef9)

**Severity:** ~~Critical~~ Fixed - image training now works end-to-end.

**Status:** `FlattenLayer::Forward` is now a pure CPU reshape that
preserves the incoming dtype and row-major `[batch, features]`
layout. `DenseLayer::Forward` already expects that layout and applies
`x @ W^T`, so the Flatten->Dense handoff is consistent again. The old
ArrayFire reshape round-trip that could scramble layout was removed.

**Files:** `cyxwiz-backend/src/algorithms/layer.cpp` (FlattenLayer
+ DenseLayer Forward methods).

---

## Node Pin Design Inconsistencies

### Status update 2026-04-17 - pin pass + compile-gate enforcement landed (8 commits)

Two-layer "stop fooling the user" sweep. **Descriptions** make the
canvas self-documenting; **compile-gate enforcement** makes it
authoritative at compile time even though the runtime still ignores
topology (see the TrainingExecutor entry below).

Pin tooltip + name consistency pass (4 commits):

1. **`9ded26e9` - pin tooltip framework + data chain.** Added
   `NodePin.description` field; the pin-hover popup at
   `node_editor.cpp:362` renders it below the generic tooltip.
   Populated for DataInput, DataSplit, DataLoader, MSELoss,
   CrossEntropyLoss. Renamed DataInput's first output pin
   "Features" -> "Data" so the chain reads `Data + Labels`
   end-to-end. Backward compatible (links restore by ordinal
   index, not name).
2. **`27da03b7` - model + optimizer descriptions.** Dense, Conv1/2/3D,
   MaxPool/AvgPool, Flatten, Dropout, BatchNorm/LayerNorm/GroupNorm/
   InstanceNorm, Output, optimizers (SGD/Adam/AdamW/RMSprop/Adagrad/
   NAdam), recurrent (RNN/LSTM/GRU), Embedding.
3. **`2cc0672f` - losses + attention + transformer.** BCE /
   BCEWithLogits / L1 / SmoothL1 / Huber / NLL losses;
   MultiHeadAttention / SelfAttention / CrossAttention with
   Q/K/V/Mask explained; TransformerEncoder/Decoder with Memory pin.
4. **`8d65a0c2` - image transforms + time series + data ops.**
   Augmentation, Normalize, OneHotEncode (notes the Labels->Tensor
   type change), Bidirectional/TimeDistributed wrappers,
   PositionalEncoding, TimeSeriesWindow / TimeSeriesSplit /
   TimeSeriesFeatures, TensorReshape, and all 9 image transforms
   (Resize / CenterCrop / RandomCrop / Horizontal/VerticalFlip /
   ImageRotate / ColorJitter / GaussianBlur / Grayscale).

Pure activation nodes (ReLU/Sigmoid/Tanh/Softmax/etc.) intentionally
left without descriptions - the node name says everything the user
needs.

Compile-gate pin-connectivity enforcement (4 commits). Five checks
now fire as Error-level issues in `GraphCompiler::Compile`, blocking
training on structurally-wrong graphs:

5. **`8c468316` - required inputs + Loss.Targets reachability.**
   `ValidateRequiredInputsConnected` flags any input pin with
   `is_required=true` that has no incoming link.
   `ValidateLossTargetsReachLabels` BFS'es upstream from each Loss
   node's Targets pin and requires an ancestor output pin of
   `PinType::Labels`.
6. **`2e34d027` - Loss.Predictions reachability.**
   `ValidateLossPredictionsReachModel` mirrors the Targets check
   but requires an ancestor node of `IsModelLayer()` type or
   `NodeType::Output`. Catches wiring Predictions to
   DataInput.Data (no model in between) or to the Labels stream
   (swapped pins).
7. **`071c1dd5` - Optimizer.Loss reachability.**
   `ValidateOptimizerReachesLoss` closes the `compile -> loss ->
   optimizer` chain. The pin-type system's "Tensor is universal"
   rule would otherwise let a Dense.Output be wired straight into
   Optimizer.Loss.
8. **`4beddfe4` - required outputs + optional markers.**
   `ValidateRequiredOutputsConnected` flags dangling producers.
   Legitimate optional outputs explicitly marked
   `is_required=false`: LSTM/GRU/RNN.Hidden, all attention
   AttnWeights, DataSplit.{Val*,Test*}, TimeSeriesSplit.
   {Validation,Test}, Output.Predictions, Optimizer.State.
   Pragmatically, **DataSplit.Train Labels** and
   **DataLoader.Labels** are also marked optional as a
   concession to the current registry-driven runtime - every
   example graph (`mnist_mlp`, the sentiment classifiers, etc.)
   routes labels direct from DataInput to Loss, bypassing the
   split/loader. The optional markers are "flip to required
   when the runtime walks pins" markers; comments in
   `node_editor_nodes.cpp` flag the handoff.

Four cyxgraph test fixtures under
`examples/cyxgraph/test_pin_connectivity/` exercise each check
(01_targets_disconnected, 02_targets_wrong_source,
03_predictions_wrong_source, 04_optimizer_loss_wrong_source).

What this fix does NOT address:
- The architectural "TrainingExecutor walks pins, not registry"
  issue (below). The canvas now LOOKS honest AND compile-time
  refuses to launch broken graphs, but the runtime still ignores
  topology. Training a compile-passing graph is safe; training
  with topology-dependent behavior (e.g., shuffled labels via
  DataLoader.Labels) won't actually honor the wiring until
  the runtime fix lands.
- The image-dialog Normalize foot-gun (Image-only) is unchanged -
  see the Normalize entry below.

### DataInput node does too much - RESOLVED except LabelSelect node

**Original four concerns** (dual pins, label column in dialog,
normalize params, domain-unaware label warning) are all either
addressed by the 2026-04-17 pin pass or predate it. Only remaining
concern: **a dedicated `LabelSelect` / `ColumnSelect` node** would
pull label-column choice off the DataInput dialog onto the canvas.
Useful but not urgent - the dialog approach works fine for the
current single-source-per-DataInput model.

### DataLoader pin layout - RESOLVED

Dual Data/Labels in/out is the intentional design as of 2026-04-17.
Matches DataInput.Data/Labels and DataSplit.{Train,Val,Test}
{Data,Labels} so the canvas reads consistently. Hover descriptions
on every pin. The "switch to single Dataset stream" proposal was
dropped - it would have hidden the `(X, y)` split ML practitioners
expect.

### Normalize node in DataInput vs graph

**Severity:** Low (current state) / Medium (latent) - current code is
not actually exposing the foot-gun, but the structural issue remains.

**State as of 2026-04-17 audit:**
- Tabular dialog (`RenderTabularOptions`): NO Normalize controls.
  Already clean. The earlier "needs removing" claim was incorrect.
- Image dialog (`RenderImageOptions:1818-1825`): STILL has
  `Normalize to [0, 1]` + `Convert to RGB` checkboxes, and `Apply`
  writes `parameters["normalize"] / parameters["rgb"]` on the
  DataInput node. The earlier "Phase 0 removed these" claim was
  also incorrect.
- The double-normalization risk is therefore Image-only, not the
  cross-category foot-gun previously described.

**Why the Image dialog still has it:** image pixel scaling currently
happens at load time inside the image loader, not via a graph
Normalize node. There's no image-specific graph node that owns
[0,1] scaling + RGB conversion as of this audit. Removing the
dialog toggles without first introducing an `ImageNormalize` graph
node would break the image training path silently.

**Fix sequence (revised):**
1. Add an `ImageNormalize` (or extend the existing Normalize node
   to handle image-shaped tensors) graph node that owns pixel-
   scaling + channel-mode conversion.
2. Remove the Image dialog's Normalize / RGB checkboxes and the
   matching `parameters["normalize"] / parameters["rgb"]` writes
   in `Apply`.
3. Auto-insert the new node when migrating old image graphs (or
   document the migration).

Not started - deferred until the canvas-honest pin pass lands first.

---

## Memory Tab UX

### ~~Image dataset shows 0B in Memory tab~~ LANDED 2026-04-17

**Status:** Resolved across three commits today:

- `2941e511` - Memory tab estimate now honors actual dialog
  `target_width_` x `target_height_` x (rgb ? 3 : 1) instead of
  a hardcoded 224x224x3. Both the fresh Apply and the project-
  reopen restore paths read the same values, so a 28x28 grayscale
  MNIST dataset reports ~20 MB instead of the nonsensical ~14 GB
  it used to show.
- `d4d4fe02` - Fixed a dispatch bug surfaced while testing this:
  the tabular-CSV Apply branch had no `file_category_` check, so
  an Image-category Apply with a stale `file_path_` value from a
  previous session would route to `LoadTabularCSV` and fail with
  "cannot stat 'dataset.csv'" instead of reaching the image folder
  branch. Gated the branch on
  `file_category_ == Tabular || TimeSeries`.
- (commit pending) - Image folder load is now async via
  `AsyncTaskManager`, mirroring the text path. Previously
  synchronous, which froze the UI during folder scans on large
  datasets (ImageNet-scale). `AsyncLoadState.backend == 3` feeds
  back into `PollAsyncLoadResult` which emits the
  `"N images, M classes"` description and an "Loaded images from
  ..." status. Closes the drift between CLAUDE.md's claim that
  "Image / Audio follow the same async contract" and the pre-fix
  reality where only text was actually async.

Audio folder load is also async now via `AudioLoader::LaunchAsyncLoad`
with `AsyncTaskManager` backend tag `4`, so the remaining follow-up
there is just any future UX polish.

---

## Compile Gate Improvements

### Memory estimation is approximate

**Severity:** Low - the 4x multiplier is a rough heuristic.

**Issue:** GPU memory estimation uses `params x 4 x sizeof(float)`
plus a 4x multiplier for activations/optimizer/CUDA overhead. This
is conservative for small models and optimistic for large ones.
The threshold of 3 GB for blocking is tuned for a 4 GB GPU (GTX
1050 Ti) and may be wrong for other hardware.

**Suggested fix:** Query actual GPU memory via ArrayFire's device
info API or CUDA runtime. Use per-layer activation size calculation
instead of a flat multiplier. Allow the user to set their GPU
memory limit in engine settings.

### ~~Label column check is not domain-aware~~ FIXED

**Status:** The label column warning now checks `file_category` on the
DataInput node. Image datasets (file_category="image") skip the
warning - their labels come from folder structure or CSV mapping.
See `graph_compiler.cpp` Check 3.

### ~~Local Debug - pre-train validation with synthetic data~~ LANDED 2026-04-18

**Status:** Shipped end-to-end across 4 worktree-session commits on
`feat/local-debug` (now merged + branch retired). Plan doc at
`docs/plans/local_debug_mode.md` was the execution guide.

Commits (merged into `Nodes_Implementation` via `--no-ff` merges):
- `aac9308c` - extract `BuildSequentialFromConfig` into `model_builder`
  (shared between `TrainingExecutor` and `DebugExecutor`; pure move).
- `bbae175b` - `DebugResult` struct scaffolding (`LayerTrace`,
  `GradNormEntry`, `DebugStage`, shared severity via `ValidationIssue`).
- `d5388ad3` - `SyntheticBatch` helper. Domain-dispatched shape:
  Tabular `[1, input_size]` float; Text `[1, seq_len]` int64 token
  IDs clamped to the Embedding's `num_embeddings`. Image / TimeSeries
  / Audio fall back to Tabular in v1.
- `3a805e28` - unit test `test_debug_executor` exercising the builder
  + synthetic batch on minimal Dense->ReLU->Dense.
- `081976d9` - `DebugExecutor::Run`: builds model, one forward
  (per-layer shape capture + NaN/Inf), loss, one backward, one
  optimizer step, L2 grad norms per learnable param, flags dead
  subgraphs (`params_missing_grad > 0` -> Warning) and NaN grads ->
  Error.
- `4388c30c` - UI wiring: F6 shortcut, toolbar button (yellow-green,
  next to Compile), `MainWindow::LocalDebugGraph()` mirrors
  `StartTrainingFromGraph`, extends the compile popup with
  `compile_result_mode_ in {Compile, Debug, BlockedTrain}`.
- `8f6f7d3f` - optional strict mode + staleness tracking.
  `engine_config.require_debug_before_train` flag plus a
  `last_debug_graph_hash_` cache. Graph changed since last successful
  Debug -> Warning in compile popup ("Consider F6 before F5"). Strict
  mode upgrades the warning to Error and blocks Train.

**What you can do now:** press F6 on any compile-passing graph. You
get per-layer shape trace, loss value + finiteness flag, grad norm
table with NaN/zero detection, ~200ms. No dataset I/O, no network.

**Registry isolation verified:** the synthetic batch is constructed
on stack tensors; `DebugExecutor::Run` never touches
`cyxwiz::DataRegistry`. Unit test asserts registry state unchanged
after Run.

**Still pending:** async execution (move `Run()` off the UI thread) -
only needed for models where the one-shot forward+backward exceeds
~500ms, which the v1 test suite doesn't hit.

---

## Generic Parameter Editor

### Properties panel parameter editing is basic

**Severity:** Low - works but not polished.

**Issue:** The generic parameter editor (added in Phase 1.4) renders
ALL node.parameters as plain text InputText fields. This works but
is not ideal:
- Numeric params (width, height, probability) should use
  ImGui::InputInt or ImGui::SliderFloat for better UX
- Enum params (mode: exact/fit/fill/center) should use a dropdown
- Boolean params (shuffle, drop_last) should use a checkbox
- Some params are internal state (dataset_name, data_loaded) that
  shouldn't be user-editable

**Suggested fix:** Add a parameter type hint system. Each node type
registers its parameters with types (int, float, enum, bool, hidden)
and the Properties panel renders the appropriate widget. Or use the
existing NodeMetadata system if it supports parameter type hints.

---

## Async / Threading

### ProcessCompletedCallbacks wired into main render loop

**Status:** Resolved 2026-05-18. `MainWindow::Render()` now drains
`AsyncTaskManager::ProcessCompletedCallbacks()` once per frame before
rendering the rest of the UI, so queued completion callbacks can fire
on the main thread.

**Files:**
- `cyxwiz-engine/src/gui/main_window.cpp`
- `cyxwiz-engine/src/core/async_task_manager.cpp`

**Note:** `DataInputDialog` still keeps its direct polling path for
load-result handoff, which is fine. The callback pump is now available
for other async users that register completion callbacks.

---

## Registry / Lifecycle

### Image datasets cleaned up on node delete

**Status:** Resolved. `NodeEditor::DeleteNode` and `ClearGraph` now
call `UnregisterNodeDatasetIfOwned`, which unregisters tabular,
image, audio, and text datasets by stored `dataset_name`.

**Files:**
- `cyxwiz-engine/src/gui/node_editor_nodes.cpp`
- `cyxwiz-engine/src/core/data_registry_utils.cpp`

**Note:** This matches the async image loader's re-apply cleanup, so
deleting a DataInput node or clearing the graph no longer leaves
stale image entries behind.

---

## UX Enhancements

### ~~Pin color validation feedback~~ IMPLEMENTED

**Status:** Pins follow a state machine on the node:
- Default (graph never compiled / just edited): red hollow
- CompileFailed: red solid
- CompilePassed: green hollow
- Trained (training run finished): green solid

After Compile, validated nodes' pins turn green, error nodes' pins
turn red solid. After a training run finishes, all nodes flip to
solid green. State auto-clears on any graph modification (add/delete
node or link). Standard data-flow links inherit the source node's
state color. See `NodeEditor::NodePinState` in `node_editor.h` and
the pin rendering switch blocks in `node_editor.cpp`.

### TrainingExecutor should walk pin connections, not registry lookups

**Severity:** High - the canvas is currently a "lie." Users see a
visual graph of `DataInput -> DataSplit -> DataLoader -> ... -> Loss`
with separate label and tensor streams, but the executor doesn't
actually traverse pin connections. It reads `dataset_name` from the
DataInput node parameters, fetches the dataset from `DataRegistry`
by name, and runs training. The pin wires are decorative.

**Symptoms users hit:**
- Removing a wire between DataLoader and Loss has no effect - training
  still works because labels come from the registry, not the pin.
- DataSplit's Train/Val/Test outputs are visual fiction. The actual
  split happens inside the batcher based on the DataSplit node's
  ratio params, regardless of what the user wired downstream.
- A graph that is structurally wrong (e.g. labels never reach loss)
  trains anyway because the executor doesn't care about topology.

**Why this is hard (~3-5 days):**
1. `TrainingExecutor` currently takes a single `dataset` + `label_column`
   and runs a hardcoded data -> preprocess -> model -> loss -> optimizer
   loop. It needs to become topology-driven.
2. Each pin-to-pin edge needs runtime semantics: a `Tensor` edge
   carries a batch tensor, a `Labels` edge carries a label batch.
3. Need a runtime "stream" abstraction: each pin holds the most-recent
   value during a forward pass; nodes consume their input pins and
   produce their output pins.
4. DataSplit becomes a real node that *takes* the dataset stream and
   *emits* train/val/test streams. The compiler/executor needs to know
   which downstream subgraph belongs to which split.
5. DataLoader becomes a real batching node sitting between DataSplit
   and the model - currently DataLoader's batch_size is just a number
   the executor reads, not something the graph actually does.
6. Loss nodes need to consume both the model's prediction stream AND
   the upstream label stream - currently the executor hands labels in
   directly because there's no concept of "which pin gives me labels."
7. Backwards-compat: the layered model (Dense -> ReLU -> Dense -> ...) is
   still a single chain; only the data/label plumbing around it changes.

**Suggested approach (when we tackle it):**
- Phase A: introduce a `RuntimeContext` with `std::map<int, Tensor>`
  keyed by pin_id. Compile builds a topological execution plan over
  the graph; each node's `Execute(RuntimeContext&)` reads input pins
  and writes output pins. Layer chain stays inside the model - only
  the data path becomes pin-driven.
- Phase B: DataSplit, DataLoader, and Loss nodes implement the new
  runtime interface. Existing layer nodes keep their current path
  (compiled into a `Model` object, still called via `model.Forward`).
- Phase C: remove `dataset_name` lookup from `StartTrainingFromGraph`
  - the dataset stream comes from the `DataInput` node executing into
  its output pins.

**Files involved:**
- `cyxwiz-engine/src/core/training_executor.h/.cpp`
- `cyxwiz-engine/src/core/training_manager.cpp`
- `cyxwiz-engine/src/gui/main_window.cpp` (StartTrainingFromGraph)
- `cyxwiz-engine/src/core/graph_compiler.cpp` (build execution plan)
- New file: `cyxwiz-engine/src/core/runtime_context.h/.cpp`

**Workaround state as of 2026-04-17 (8 pin commits):** the canvas is
now self-documenting (hover descriptions on every meaningful pin)
AND the compile gate enforces five structural invariants:
required inputs connected, required outputs consumed (minus the
explicitly-optional set), Loss.Targets BFS reaches a
`PinType::Labels` source, Loss.Predictions BFS reaches a model
layer / Output node, and Optimizer.Loss BFS reaches an actual
Loss node. See the pin-section status block above for details and
commit hashes.

What the compile gate CAN'T enforce yet (runtime still needed):
- DataSplit.Train Labels and DataLoader.Labels are marked optional
  outputs so the existing example graphs (which bypass them
  entirely) still compile. Once `TrainingExecutor` walks pins,
  flip those is_required flags and the "labels must route through
  the loader for shuffling alignment" rule becomes compile-time
  enforced.
- A graph with shuffled data but unshuffled labels silently trains
  against misaligned pairs - the structure is valid, only the
  runtime behavior is wrong.

Net effect: the canvas is no longer "the lesser of two lies" - it's
a contract at compile time, just not yet at runtime. Training a
compile-passing graph is safe; the gap is topology-dependent
behavior (like the shuffle-alignment concern above) that only the
runtime fix can close.

**DataLoader params that are currently UI-only (need executor wiring):**
The Properties panel exposes a full set of training-loop knobs on
the DataLoader node, but only `epochs`, `batch_size`, `shuffle`, and
`drop_last` are actually honored at runtime today. The rest are
stored on the node, saved to the project file, and read by the
graph compiler into TrainingConfiguration, but the executor never
acts on them. When option (b) lands, wire these through:

- `grad_accum_steps` (default 1) - accumulate gradients over N
  forward passes before stepping the optimizer. Effective batch
  size = `batch_size x grad_accum_steps`. Lets users train with
  large effective batches on a small GPU. Touches the training
  inner loop in `training_executor.cpp` and the optimizer step.
- `seed` (default 42) - RNG seed for shuffle order. Currently the
  batcher seeds itself; should accept this from config so two runs
  with the same seed produce the same epoch order.
- `num_workers` (default 4) - now implemented;
  implemented across training batchers for parallel sample loading /
  feature-column materialization. Remaining improvement is a shared
  prefetch layer.
- `prefetch_factor` (default 2) - batches to prefetch per worker.
  Pairs with num_workers; meaningless without it.
- `pin_memory` (default false) - allocate batches in pinned host
  memory for faster H2D copy. CUDA-only optimization, ignored on
  OpenCL/CPU. Touches the batcher's tensor allocation path.
- `log_interval` (default 10) - currently the executor logs every
  batch. Should respect this and only log every N batches.
- `validation_freq` (default 1) - currently validation runs every
  epoch. Should run only every N epochs.

None of these are blocking - the existing pipeline trains correctly
without them. They're quality-of-life knobs that match what users
expect from PyTorch / Keras DataLoader. Adding them to the executor
is straightforward once option (b) reorganizes the data flow.

---

## Data Quality / Sanity Audits

### Dataset sanity audit at Apply time (cross-domain)

**Status:** DONE - 2026-06-03. Apply-time compact audit is implemented
for tabular, Parquet-backed tabular metadata, image, audio, and text.
Deferred follow-ups are intentionally outside this compact Apply-time
scope: Parquet data-page sampling needs a bounded worker/timeout design,
and text vocabulary coverage belongs in the richer TextTokenizer /
TextVocabulary node dialog.

Landed a small core audit boundary:

- Added `cyxwiz-engine/src/core/dataset_audit.h/.cpp`.
- Added `test_dataset_audit`.
- Loaders now run metadata/table-level audits after successful async
  registration/probe and return compact audit counts through
  `AsyncLoadState`.
- `DataInputDialog::PollAsyncLoadResult` appends the audit summary to
  the Apply status and stores `audit_errors` / `audit_warnings` on the
  node parameters.
- Added a first read-only Audit tab in `DataInputDialog` showing loaded
  dataset status, audit error/warning counts, and current-session issue
  messages.

This first slice is intentionally warn/report oriented. Image and audio
sample checks now run, but they report aggregate counts and class-level
ratios only. Current-session issue messages include a bounded set of
example paths for suspicious image/audio files. Apply is refused when
the audit emits threshold errors.
For class-subdirectory image/audio datasets, the audit also reports
classes where more than 20% of sampled files are suspicious.

**Severity:** High - silently corrupt datasets produce fake training
metrics and waste GPU hours before the user notices.

**Origin:** Caught during Phase 2 audio testing on the Kaggle
"Binary Drone Audio" dataset. 1089 / 11704 files (9.3%) were genuinely
all-zero silent WAVs with valid headers - `wave.open()` reports 16347
frames at 16kHz but every sample is literally 0. The mel spectrogram of
silence is constant `log(1e-10) = -23.0259`, which looks identical to a
backend bug. Paired with a separate data-leakage bug in the val pipeline,
training reported a fake 100% val accuracy and nobody would have noticed
the silent files without end-to-end testing on a real Kaggle dataset.

The audio domain now has a fix (`AudioDataset::ExtractFeatures` detects
all-zero samples and returns invalid, with rate-limited warnings in
`GetItem`), but the **broader lesson is that every domain needs a
quality audit that runs at Apply time**. Unit tests cannot catch this
class of issue - only integration tests on real datasets can.

**Concrete sanity checks to add:**

- **Audio** (partially done)
  - [done] Silent file detection (all samples == 0)
  - [done] Apply-time metadata audit for empty/single-class datasets
    and flat-layout label CSV presence
  - [done] RMS energy threshold sample audit (files below -60 dBFS RMS
    are reported as near-silent)
  - [done] Report aggregate warning: "N/M files are suspicious"
    including zero-byte, decode-failed, all-zero, and low-RMS counts
  - [done] Report per-class suspicious audio ratios for class-subdir layouts
  - [done] Include bounded example paths for suspicious audio files
  - [done] If suspicious ratio > 20% in any class, refuse Apply with an error
- **Image**
  - [done] Apply-time metadata audit for empty/single-class datasets
    and class-name count mismatch
  - [done] Decode a deterministic sample of N files (N = min(100, total))
  - [done] Count: 1x1 corrupt thumbnails, 0-byte placeholders, fully-black
    images (pixel sum == 0), fully-white images (pixel sum == 255*H*W*C)
  - [done] Report aggregate warning: "N/M images appear blank or corrupt"
  - [done] Report per-class suspicious image ratios for class-subdir layouts
  - [done] Include bounded example paths for suspicious image files
  - [done] Refuse if > 20% bad in any class
- **Tabular**
  - [done] Check label column: unique value count and severe class balance
  - [done] Warn if label has only 1 unique value (no signal)
  - [done] Check Arrow feature columns: 100% null, 100% constant,
    NaN, infinite values
  - [done] Extend Parquet-backed audit beyond schema/label presence with
    metadata-only checks: file size, schema availability, duplicate column
    names, schema/metadata column-count mismatch, row-group presence, and
    empty row groups
  - TODO: Parquet data-page/sample audit for null/constant/NaN/label
    distribution; first row-group test path hung and should not run on the
    UI Apply path until it has a bounded worker/timeout design
  - [done] Warn for each degenerate Arrow column, refuse if > 50% of
    feature columns are all-null or constant
- **Text** (Phase 3)
  - [done] Apply-time metadata audit for empty/single-class datasets,
    missing label column, and empty vocabulary
  - [done] Bounded source sample audit for empty strings,
    single-character strings, UTF-8 replacement markers, and binary-like
    NUL bytes across plain text, CSV/TSV, JSON/JSONL, and folder corpora
  - DEFERRED: Vocabulary coverage check belongs with the richer
    TextTokenizer/TextVocabulary node dialog, not the compact
    Apply-time audit. Apply should keep only summary coverage metrics
    once that dialog exists.

**Where it goes:** A new `DatasetAudit` helper in
`cyxwiz-engine/src/core/dataset_audit.h/.cpp` with domain-specific
static methods. The first slice is called by the async loaders after
registration/probe and before the UI thread marks the node loaded.
Results are currently shown as a compact Apply status suffix and a
read-only Audit tab with counts plus the current-session issue list.
Suspicious image/audio issues include bounded example paths. Audit
errors refuse Apply while keeping the Audit tab visible. Richer
drill-down details remain TODO.

**Estimated scope:** ~1-2 days per domain. Audio is partially done
(silent detection). Image is the next priority (same class of issue
affects cats/dogs and similar).

### Rate-limited sample loading warnings in AudioDataset

**Status:** DONE - `AudioDataset::GetItem` uses an atomic counter and
prints the first 10 bad files at full detail, then every 100th. Full
warning-per-sample spam would flood the log on datasets with many
silent files (1089 in the drone set we tested).

---

## Training UX

### Sub-epoch validation (fractional validation_freq)

**Severity:** Enhancement - smooths the val curve for long-epoch domains.

**Issue:** Validation currently runs once per epoch - standard ML practice
and correct for fast-epoch domains (tabular MNIST finishes an epoch in
seconds). But for audio/image pipelines where an epoch can take 5-10
minutes, you stare at an empty val line for the full duration of epoch 1
and assume it's broken. Users who stop training before the first epoch
completes never see any val feedback at all.

**Current workaround:** Shrink max_duration to match the actual clip
length (drone clips are ~1s; setting max_duration=1 instead of 5 cuts
epoch time by ~5x because feature cols shrink from ~313 to ~63 and the
Linear(input->512) first layer shrinks proportionally).

**Suggested fix:** Reinterpret `validation_freq` on the DataLoader node
as follows:
- integer N >= 1 -> run validation every N epochs (current behavior)
- 0 < N < 1 -> run validation every N x total_batches batches, on a
  *random subsample* of the val split sized for ~2-3 seconds of
  validation time
- Default stays 1 (every epoch)

This gives users a continuous val curve during long epochs. A value
like `0.25` means "validate 4 times per epoch" - at batches 25%, 50%,
75%, 100%. The subsample size should be `min(val_set_size, 256)` so
validation cost stays bounded regardless of how big the val split is.

**Files:** `cyxwiz-engine/src/core/graph_compiler.cpp` (parse float,
store on TrainingConfiguration), `cyxwiz-engine/src/core/training_executor.cpp`
(RunTrainingEpochArrow - insert validation calls at the configured
fractional boundaries), `cyxwiz-engine/src/gui/properties.cpp`
(DataLoader properties section - change validation_freq widget to
float InputText with tooltip explaining integer vs fractional).

**Scope:** ~0.5 day. Useful pairing with the option-(b) executor
refactor since both touch the inner training loop.

---

## Future Architecture

### Variable-shape Sample type (v3 consideration)

The current pipeline pre-flattens images in the batcher, which means
the Flatten graph node is a conceptual marker, not a real transform.
A proper solution would introduce a `Sample` type with dynamic shape
that flows through the graph, with each node transforming the shape.
`DataLoader` would enforce uniform shape at batch time and error if
shapes don't match.

This is Option B from the design doc - deferred to v3 because it
requires a ground-up rewrite of the tensor plumbing.

### Explicit Decode node (v3 consideration)

Option C from the design doc: `DataInput` produces file references,
a `Decode` node reads the bytes and produces the raw tensor. Maximum
composability but adds a required node everyone will forget. Also v3.

---

## Studio <-> Engine Integration (2026-04-16)

**Severity:** Feature request - Studio and Engine's node editor
currently live in separate panes with no cross-navigation. Users
must hunt for the Node Editor in the Engine toolbar after doing
exploratory work in Studio.

### ~~Open Node Editor from CyxWiz Studio~~ LANDED 2026-04-17

**Status:** Resolved. `DataStudioPanel::RenderToolbar` now renders
an "Open Node Editor" button that invokes a callback wired by
`MainWindow` to `node_editor_->Show()`. Replaced the old static
`TextDisabled` hint. Falls back to the hint text if the callback
isn't wired (defensive).

**Files touched:**
- `cyxwiz-engine/src/gui/data_studio/data_studio_panel.h/.cpp` -
  added `SetOpenNodeEditorCallback` + button rendering.
- `cyxwiz-engine/src/gui/main_window.cpp:347` - wires the callback
  alongside the existing NAS / training callbacks.

Same callback pattern is ready to reuse for other cross-pane
buttons (Training Dashboard, Dataset Manager) if/when needed.

### Query Console in CyxWiz Studio

**Idea:** A modal / docked console in Studio where the user can type
SQL-like queries to inspect and update the current graph. Much
faster than opening Properties and clicking through nodes one by
one when bulk-editing.

**Use cases:**
```sql
-- Inspect:
SELECT id, type, name, parameters.learning_rate FROM nodes
  WHERE type = 'Adam';

SELECT id, name FROM nodes WHERE connected_to = 'DataInput';

-- Bulk update:
UPDATE nodes SET parameters.learning_rate = 0.01 WHERE type = 'Adam';
UPDATE nodes SET parameters.batch_size = 64
  WHERE type = 'DataLoader';

-- Connection audit:
SELECT start_pin, end_pin, start_node, end_node FROM links
  WHERE start_node.type = 'Dense';
```

**Wiring sketch:**
- New Studio panel (`QueryConsolePanel`) with ImGui text input +
  result table.
- Query grammar: subset of SQL - SELECT / UPDATE over the `nodes`
  and `links` virtual tables. No joins in v1.
- Backend: iterate the engine's `NodeEditor::GetNodes()` /
  `GetLinks()`, apply the WHERE clause as an in-memory filter, then
  project or mutate. Mutations dirty-flag the graph for re-save.
- DuckDB could host the query engine if we want full SQL - there's
  precedent in the Data Studio Phase 4 roadmap which integrated
  DuckDB for the Query Editor. Reuse that dependency.
- Output: ImGui table with selectable rows; double-click a row to
  focus that node in the Node Editor (if open).

### Why these two together

The common thread is **Studio should be a dashboard, not a dead-end
tab**. Today it shows data but can't reach across to the graph
that processes the data. The two buttons (Node Editor open, Query
Console) plus the existing Training Dashboard button would make
Studio the natural hub for exploratory -> pipeline-editing ->
training workflows.

**Severity note:** Neither is a bug; both are UX / developer-
productivity wins. Scope-wise, Open Node Editor is ~1 hour
(button + existing panel-raise path). Query Console is a
multi-session lift - grammar / parser / engine-NodeEditor bridge
/ mutation safety / undo integration.

---

## Phase 3 Text - Deferred Items (2026-04-14)

Phase 3 text training is **functionally complete and live-verified**
on sentiment_mental_health.csv (52681 samples, 7 classes). Six commits
landed this session - async Apply, text registry, JSON loader
TextTokenizer/TextVocabulary/TextPadding registration, compile gate
registry-first probe, v2 regularized example graph, Embedding AF
backward fix. What's still deferred:

### ~~Phase 3 engine-side wiring bundle not yet committed~~ LANDED

**Status update 2026-04-16:** All Phase 3 engine-side wiring is now
committed to the `Nodes_Implementation` branch. Relevant commits:
- `bce6b023` - text dataset registry
- `f9ee4ec4` - Phase 3 text graph compiler extractors +
  preprocessing domain
- `7378b6d0` - Phase 3 text dataset batcher + CSV loader
- `70360ef6` - Phase 3 text training manager + engine dispatch
- `80789995` - Register Text* node types in cyxgraph JSON loaders
- `7b7bd34b` - async Apply + preview panel + header null bytes fix
- `52808745` - `node_executors/` framework with Cat-1 IPipelineOperator
  base (the "unfinished scaffolding" is now the foundation for 29+
  live Cat-1 operators across the whole Tool-to-Node migration)

Kept in this tofix strictly as a landed-commit pointer so the
history of how Phase 3 reached git is traceable.

### num_workers support is partial

**Status:** Mostly resolved. `GraphCompiler` now forwards
`num_workers` to `TrainingConfiguration`. `DatasetBatcher`,
`TextDatasetBatcher`, `ImageDatasetBatcher`, and `AudioDatasetBatcher`
honor it by loading samples across worker threads. `ArrowDatasetBatcher`
and `ParquetArrowBatcher` honor it by parallelizing feature-column
materialization for wide tabular batches.

**Remaining issue:** current implementations parallelize within one
batch rather than prefetching the next batch. A shared prefetching
layer would still be cleaner than per-batcher worker splits.

**Suggested fix:** Add a shared prefetching layer to `IBatcher` so
all batchers can overlap sample loading with model execution.
Non-trivial because batcher state (shuffle, epoch progress, phase)
is currently single-threaded.

**Files:**
- `cyxwiz-engine/src/core/text_dataset_batcher.cpp`
- `cyxwiz-engine/src/core/image_dataset_batcher.cpp`
- `cyxwiz-engine/src/core/audio_dataset_batcher.cpp`
- `cyxwiz-engine/src/core/dataset_batcher.cpp`
- `cyxwiz-engine/src/core/parquet_arrow_batcher.cpp`
- `cyxwiz-engine/src/core/graph_compiler.cpp`
- `cyxwiz-engine/src/gui/data_input_dialog.cpp`
- `cyxwiz-engine/src/gui/main_window.cpp`

### Training logs silent mid-epoch - no per-batch feedback

**Severity:** Medium - UX gotcha, caused me to misdiagnose "hung"
training in the 2026-04-14 session.

**Issue:** During training, the log emits rich output up through the
first batch of epoch 1 (model summary, `DEBUG Arrow`, `DEBUG
CrossEntropy` sample dump), then goes completely silent for
60-130 seconds until the epoch-end summary line. The per-batch debug
prints are gated by `if (epoch == 1 && batch_num == 1)` in
`training_executor.cpp:1073`.

**Status: RESOLVED 2026-04-15.** Periodic progress log added to both
`RunTrainingEpochArrow` and `RunTrainingEpoch` (non-Arrow path).
Fires on batch 1 of every epoch (so the user sees immediate feedback
the loop entered) and every 50 batches thereafter. Format:
```
Epoch 1 [100/659] loss=1.2345 acc=52.30% (5.2s, 19.2 batches/s)
```
Elapsed time and throughput computed from `std::chrono::steady_clock`
captured at the top of the batch loop. Rate warms up naturally as the
batcher and GPU pools stabilize, so the "batches/s" reading is
informative across the whole epoch. No per-invocation time throttle
- 50 batches is long enough that a fast epoch won't spam.

### LSTM verified with text; GRU / Transformer still unverified

**Status update 2026-04-16:** LSTM text path is now verified
end-to-end on the mental_health sentiment corpus. Smoke-test graph
shipped in commit `4cdaf7ad`; full CPU BPTT + AF Forward + AF
Backward all operational (`38d0a250`, `f4ac9b57`, `dc727ef6`).
Weights update correctly on both CPU and GPU backends.

**Still unverified:** GRU and Transformer text paths. The backend
has both (`GRUModule`, `TransformerEncoderLayer`), but no
end-to-end smoke test over text exists. Same risk profile as LSTM
pre-fix - a shape-handling bug in the recurrent / attention layers
won't surface until someone runs Embedding -> GRU -> Dense or
Embedding -> TransformerEncoder -> Dense through to convergence.

**Fix:** Port `mental_health_sentiment_lstm.cyxgraph` to GRU and
Transformer variants, run each to convergence, fix whatever breaks.
Medium-size task - LSTM-style investigation + fixes could reoccur.

### ~~Text preview doesn't show class distribution~~ RESOLVED 2026-05-18

**Severity:** Low - nice-to-have.

**Status:** `RenderTextPreview` now shows a compact class distribution
summary above the preview table when the mapped label column exists.
`LoadColumnList()` computes per-label counts from the preview rows,
and the renderer refreshes the distribution if the user edits the
label-column mapping after loading the preview.

**Issue:** `RenderTextPreview` (added in 7b7bd34b) shows a CSV head
table with mapped text/label columns highlighted green or red. It
does NOT show a class-distribution bar chart or sample-per-class
count - users can't see class imbalance until training starts.

**Fix:** Parse the label column during `LoadColumnList` and
compute a per-value count, then render a small horizontal bar
chart above the CSV head table.

**Files:** `cyxwiz-engine/src/gui/data_input_dialog.cpp` -
`LoadColumnList()` + `RenderTextPreview()`.

### `node_config_dialog.h` git-binary state pre-HEAD~3

**Severity:** Low - historical, mostly a diff-tool annoyance.

**Issue:** The header had 2 literal 0x00 bytes inside a Win32
`OPENFILENAME` filter default arg (`"All Files\0*.*\0"`) that MSVC
compiled fine but flipped the file to binary mode in git. Fixed in
7b7bd34b for this branch forward, but the corruption is still in
master / origin/master / HEAD~3 and earlier. Any cross-boundary diff
(e.g. `git log -p` across the fix) still shows as binary.

**Fix (low priority):** A history-rewriting filter-branch pass could
fix past commits. Not worth the pain unless someone needs a clean
historical diff.

### v1 / v2 example graphs live in `examples/cyxgraph/` alongside pre-v2 content

**Severity:** Low - organizational.

**Issue:** v2 regularized variant was added as a sibling file rather
than in a `v2/` subdirectory. As more variants land (LSTM, Transformer,
etc.) the flat layout will get noisy.

**Fix:** Move to `examples/cyxgraph/text/` subdirectory (which already
exists as untracked directory from the prior session). Already
planned for the Phase 3 engine-side bundle commit.

---

## ~~LSTM Layer - Broken AF Forward + Missing CPU Backward~~ RESOLVED (2026-04-16)

**Status update 2026-04-16 (final):** LSTM is now fully operational
end-to-end on both backends. Three commits landed today:
- `38d0a250` - CPU Forward populates row-major caches, CPU BPTT
  reads from them, weights update correctly.
- `f4ac9b57` - AF Forward operational (4 stacked bugs fixed via
  row-major 3D helpers, weight init guard, h_n/c_n null-data check).
- `dc727ef6` - AF Backward operational, full GPU LSTM end-to-end.

CPU path remains as the correctness oracle; AF path validated
against it numerically. Kept in tofix as a landed-commit pointer
plus the perf-optimization subsection below (still open).

**AF Forward - DONE (2026-04-16):**
- ~~3D column-major scrambling at TensorToAf boundary~~ Fixed via
  `TensorToAf3DRowMajor` / `AfToTensor3DRowMajor` helpers
  (dim-reversal + `af::reorder(2, 1, 0)`).
- ~~Slice assignment shape mismatch~~ Fixed by wrapping each
  `h_states(t, span, span) = h` RHS in `af::moddims(..., dim4(1,
  batch, hidden))` so the rank matches the rank-3 proxy.
- ~~Hoisted weight init guard~~ Re-initializes W_ih_/W_hh_/biases
  via `Tensor::Random/Zeros` if the AF backend silently failed to
  populate them (mirror of the CPU fallback's existing check).
- ~~h_n_/c_n_ init check~~ Now AND'd with `Data<float>() == nullptr`.
  Default-constructed `Tensor()` has `shape={}` so `NumElements()`
  returns 1 (product of empty range) but `Data()` is null -
  the previous `NumElements() == 0` check passed through and
  `TensorToAf` then tripped on null data. Caught via diagnostic
  log of tensor shape on null-data throw.
- AF Forward numerical output validated against CPU Forward -
  loss numbers match within fp32 noise (~10ppm) on the mini
  sentiment LSTM smoke test. Same monotonic loss-down /
  acc-up curve.

**AF Backward - DONE (2026-04-16):**
- Reads row-major caches (populated by AF Forward via
  `AfToTensor3DRowMajor`) back to AF column-major via
  `TensorToAf3DRowMajor`. Uses the same slice-shape moddims
  pattern as Forward for the `d_layer_input(t, span, span) = dx_t`
  write. BPTT math identical to the legacy AF code; just rewired
  for the 3D helpers.
- `kAfBackwardEnabled = true` constant. On AF exception falls
  through to CPU BPTT (existing path). Bidirectional Backward
  not implemented - bidirectional graphs always take CPU.
- Validated against AF Forward + CPU BPTT baseline - loss
  numerically identical at the precision logged (4 decimal
  places). Zero AF exceptions, zero CPU fallback warnings.
- Legacy `#if 0` block deleted (149 lines removed from
  `layer.cpp`). The new code is the single source of truth.

Net win: LSTM training is fully on GPU end-to-end (Forward AND
Backward via AF). CPU BPTT remains as the dependable fallback.
Throughput on the mini smoke graph dropped to ~5 batches/s
(vs ~19 with CPU BPTT) because tiny hidden=32 means GPU kernel
launch overhead exceeds compute. The hidden=128
sentiment_lstm graph is where AF should dominate - that's
the next benchmark.

### LSTM AF perf optimizations (deferred follow-up)

Identified 2026-04-16 after AF Backward landed. Current AF LSTM is
correct end-to-end but has known per-batch overhead that dominates
on small models (hidden < 64). For the hidden=128 sentiment_lstm
benchmark these may not be needed, but for users with smaller LSTMs
or tighter latency budgets they're worth landing.

**1. Pre-compute activations + tanh(c) in bulk in AF Backward.**
   Currently each timestep recomputes `sigmoid(gates(span, seq(...)))`
   for i / f / o gates and `tanh(c_t)`. Hoist these to a single
   pre-loop bulk computation - `af::sigmoid(cached_gates(span, span,
   seq(0, H-1)))` etc. - then slice in the loop. Saves ~4 sigmoid/tanh
   kernel launches per timestep. For seq=32, that's ~128 fewer
   kernel launches per Backward call. Biggest win on small graphs
   where launch overhead dominates.

**2. Cache AF weight arrays as `af::array` member variables.**
   Currently `af::array W_ih = TensorToAf(W_ih_[layer])` runs once
   per Forward call (and again for AF Backward) - host->GPU copy of
   the weight matrices. Add `std::vector<af::array> af_W_ih_,
   af_W_hh_, af_b_ih_, af_b_hh_;` populated lazily, invalidated
   in `SetParameters()` (called by Adam after each step). Saves
   one copy per layer per Forward+Backward.

**3. Skip the Tensor cache round-trip when AF runs end-to-end.**
   Current architecture: AF Forward writes Tensor caches via
   `AfToTensor3DRowMajor`, AF Backward reads them back via
   `TensorToAf3DRowMajor`. That's 8 bulk Tensor<->AF transfers per
   batch (4 per direction x Forward + Backward). Add parallel
   `std::vector<af::array> af_cached_inputs_, af_cached_gates_,
   af_cached_h_, af_cached_c_;` that AF Forward writes directly
   and AF Backward reads directly. Tensor caches still populated
   for CPU BPTT fallback. Saves 8 transfers per batch.

**4. Benchmark the full `test_02_sentiment_lstm.cyxgraph`** before
   deciding which of (1)-(3) actually matter. Mini graph hidden=32
   exposes only kernel-launch overhead; real-world hidden=128 with
   embed=64 and vocab=10k is ~16x more compute per matmul where AF
   should already dominate. Measure first, optimize specific
   bottlenecks second.

**5. Bidirectional AF Backward.** Forward AND Backward currently
   fall through to CPU for bidirectional graphs. Mirror the
   forward-direction code with the reverse-direction weights and
   reversed timestep loop. ~150 lines.

**Severity:** ~~High~~ Medium (after CPU BPTT landed) - LSTM weights
update correctly on CPU. AF path is a perf-only follow-up. Original
severity kept below for historical context.

**Original severity (before 2026-04-16 fix):** High - LSTMLayer was
a frozen random projection during training. Weights never updated.
Every model using LSTM trained only the *downstream* layers.

**Discovered:** 2026-04-15 LSTM smoke test
(`examples/cyxgraph/text/test_02_sentiment_lstm.cyxgraph`) - the goal
was to verify the Phase 3 text pipeline works with an LSTM head in
place of the flat Dense head. Plumbing worked end-to-end
(`LSTMModule`, `training_executor` LSTM case, `Embedding -> LSTM`
shape-tracking lookahead, `IsModelLayer` recurrent entries) but
training produced per-batch warning spam from two distinct bugs.

**Three stacked problems in `cyxwiz-backend/src/algorithms/layer.cpp`:**

### 1. `LSTMLayer::Forward` ArrayFire path throws on `[batch, seq, features]` input

**Location:** `layer.cpp:2038-2255` (AF try/catch block inside the
`#ifdef CYXWIZ_HAS_ARRAYFIRE`).

**Symptom on first batch:**
```
ArrayFire LSTMLayer::Forward failed: ArrayFire Exception (Invalid
input argument:202)
```

**Symptom on subsequent batches:**
```
ArrayFire LSTMLayer::Forward failed: ArrayFire Exception (Invalid
input size:203)
```

**Root cause (same bug family as the pre-fix EmbeddingLayer backward,
commit 84ef7211):** the `af::reorder(x, 1, 0, 2)` and `af::moddims`
calls assume row-major dim ordering, but ArrayFire is column-major.
When `TensorToAf` converts a row-major `[batch=64, seq=128, feat=64]`
tensor, it lands in AF with dims interpreted column-major, so:
- `x.dims(0) = 64` is read as `batch_size` but is actually `feat`
- `x.dims(1) = 128` is read as `seq_len` - correct by coincidence
- `x.dims(2) = 64` is read as `input_dim` but is actually `batch`

The values happen to be numerically right for this specific shape
(batch == feat == 64), but the SEMANTICS are scrambled. The
subsequent `af::reorder(1, 0, 2)` swap and `moddims` reshapes
operate on the wrong axes and either the argument shapes don't
match expected kernel args (ERR_ARG 202) or the internal state
gets wedged and later calls trip ERR_SIZE 203.

**Gated off** as of 2026-04-15: `LSTMLayer::Forward:2050` now has
`constexpr bool kAfPathEnabled = false;` in front of the `try` so
the AF block never runs. CPU fallback handles 3D input correctly.

### 2. CPU `LSTMLayer::Forward` fallback doesn't populate the four AF-format caches

**Location:** `layer.cpp:2257-2379` (CPU fallback path).

**Issue:** The AF Forward path, when it worked, populated four caches
with `AfToTensor` wrappers:
- `cached_inputs_`
- `cached_gates_`
- `cached_hidden_states_`
- `cached_cell_states_`

The CPU fallback computes `h`, `c`, gates, and intermediates as
`std::vector<float>` locals and never writes them into those caches.
On exit, only the final `output` tensor is returned - the per-layer
per-timestep state needed for BPTT is thrown away.

**Fix:** add an `std::vector<float>` scratch buffer during the CPU
forward pass that collects `gates` / `h_states` / `c_states` per
layer, then wrap them as Tensors at the end and push into the cache
vectors. Use row-major `[seq_len, batch, hidden]` layout to match
what the AF backward code reads - OR rewrite the backward in CPU
and drop AF format entirely (see next item).

### 3. `LSTMLayer::Backward` has no CPU implementation

**Location:** `layer.cpp:2381-2527` (single AF-only backward).

**Issue:** `Backward` starts with a stub:

```cpp
if (cached_inputs_.empty() || cached_gates_.empty() ||
    cached_hidden_states_.empty() || cached_cell_states_.empty()) {
    // ... warn once, return Zeros(cached_input_.Shape())
}
```

Fix #2 above will make this stub stop triggering, but then we still
need a CPU code path for the case where AF backward itself fails
(mirror of the Embedding CPU fallback we added yesterday).

Currently the warning is one-shot via `std::atomic<bool> warned_once`
at `layer.cpp:2394-2402` so at least the log isn't spammed - but the
underlying reality is "LSTM weights don't update, ever".

**The missing CPU backward is the real work:** ~80-100 lines of
backpropagation-through-time math over the cached timesteps. For
each timestep from seq_len-1 down to 0:
- Compute gradients w.r.t. output gates (i, f, g, o)
- Propagate through the tanh / sigmoid activations
- Accumulate into dW_ih, dW_hh, db_ih, db_hh
- Compute gradient w.r.t. previous hidden state (`dh_next`)
- Compute gradient w.r.t. previous cell state (`dc_next`)
- Compute gradient w.r.t. current input step (accumulate into `dx`)

Same algorithm as the existing AF backward at `layer.cpp:2391-2527`,
just expressed as explicit CPU loops instead of `af::matmul` /
`af::tile` / `af::sigmoid` calls. A known-good reference
implementation: karpathy's min-char-rnn.py, PyTorch's LSTM C++
source, or the CPU path any other deep learning framework that
supports backprop.

### Recommended fix sequence (separate session)

1. **Add CPU backward (biggest chunk).** Without this, any LSTM
   gradient stays zero and the layer can't learn. Write it to read
   from whatever caches Forward is willing to populate - row-major
   CPU layout, not AF-format.
2. **Populate caches in CPU Forward.** Gates, hidden states, cell
   states per layer per timestep.
3. **Delete the one-shot warn stub** - once Backward has real CPU
   path, the "empty caches" state means something is wrong, and we
   want a real error, not a silent zero return.
4. **(Optional) Debug the AF path** for performance. The CPU path
   will work correctly but be ~10x slower than a fixed GPU path
   would be. Diminishing returns relative to the above.

### Test after fix

- Rerun `test_02_sentiment_lstm.cyxgraph` for 8 epochs.
- Expected: val_acc climbs to at least match v2's 60.7% (frozen
  embeddings + MLP head). A working LSTM over learnable embeddings
  should plausibly beat it, not tie.
- Regression check: v2 (`mental_health_sentiment_classifier_v2`)
  should still train with identical numbers - we haven't touched
  the Dense-head path.

### Files touched during the 2026-04-15 smoke test that need revisiting

- `cyxwiz-backend/include/cyxwiz/sequential.h` - `LSTMModule`
  declaration (keep)
- `cyxwiz-backend/src/algorithms/sequential.cpp` - `LSTMModule::Forward`
  last-step slice + `Backward` re-expand (keep; last-step mode works
  correctly once the underlying `LSTMLayer::Backward` produces real
  gradients)
- `cyxwiz-backend/src/algorithms/layer.cpp` - `LSTMLayer::Forward`
  AF gate + `Backward` stub (remove/rewrite during the fix)
- `cyxwiz-engine/src/core/training_executor.cpp` - Embedding `->`
  recurrent lookahead + LSTM case (keep)
- `cyxwiz-engine/src/core/graph_compiler.cpp` - `IsModelLayer`
  recurrent entries (keep)
- `examples/cyxgraph/text/test_02_sentiment_lstm.cyxgraph` - the
  smoke test graph that surfaced all this (keep as a regression
  fixture)

---

## GRULayer - CPU Forward + CPU BPTT landed 2026-04-18 (smoke test pending) | AF path still TODO

**Status (2026-04-18):** CPU side rewritten end-to-end in
`cyxwiz-backend/src/algorithms/layer.cpp`. Three changes:
1. **CPU Forward populates caches.** New row-major layout:
   - `cached_inputs_[L]`         `[seq, batch, layer_input_size]`
   - `cached_gates_[L]`          `[seq, batch, 4 * H]` -
     `[r_post | z_post | n_post | hn_pre]` per (t,b). The 4th slot
     stores the unmodulated h-side projection feeding `n` so BPTT
     can correctly split `d_n` into x-side (`d_x_proj_n = dn_pre`)
     and h-side (`d_hn_pre = dn_pre * r`) parts.
   - `cached_hidden_states_[L]`  `[seq + 1, batch, H]` (idx 0 = h_0)
2. **CPU Backward (BPTT) implemented** following the LSTM CPU BPTT
   recipe but adapted for GRU's asymmetric n-gate. Crucial: maintains
   *separate* `dgates_x` and `dgates_h` vectors because the n-slot
   values differ (the legacy AF backward used a single `dgates` for
   both, AND zeroed the r-slot - two bugs that would have left
   training broken even if AF Forward worked).
3. **AF Forward gated `#if 0`.** Mirror of LSTM's pre-revival state.
   AF path needs the same `TensorToAf3DRowMajor` + slice-write
   `af::moddims(rhs, dim4(1,batch,hidden))` treatment LSTM got;
   tracked as a perf follow-up below.

The empty-cache `Backward` guard now returns zeros sized by
`cached_input_.Shape()` (mirrors LSTM) and warns once instead of
per-batch.

**Pending verification:** `scripts/rebuild.sh` blocked on missing
`vcpkg/scripts/buildsystems/vcpkg.cmake` - need either the junction
to `D:/vcpkg/` or a fresh cmake configure. Once builds, run
`test_04_sentiment_gru_mini.cyxgraph` for 3 epochs; expect monotonic
loss + acc climbing above the 7-class random baseline (~14%) into
the 30-50% range like the LSTM mini.

**TODO once verified:** delete `GRUModule`'s one-shot warning in
`cyxwiz-backend/src/algorithms/sequential.cpp:297-304` and update
the `gru_wired_pending_layer_fix.md` memory.

---

### Original triage (2026-04-17) - kept for context

**Severity:** Medium - GRU node is now wired end-to-end (graph
compiler -> `GRUModule` -> training_executor case), but the underlying
`GRULayer` has the SAME three-bug pattern LSTM had pre-2026-04-16.
Smoke test runs without crashing but loss stays flat - gradients are
zero. `GRUModule` constructor logs a one-shot warning so users know
upfront not to expect convergence.

**Surface symptoms (from `test_04_sentiment_gru_mini.cyxgraph` smoke
test, 2026-04-17):**

```
[warn] ArrayFire GRULayer::Forward failed: ArrayFire Exception
       (Invalid input size:203): Size mismatch between input and output
       In function af::array::array_proxy::operator =
       In file src\api\cpp\array.cpp:578
       falling back to CPU
[warn] GRULayer::Backward called without cached data from Forward,
       returning zero gradients
```

Repeats per batch. CPU Forward succeeds (output shape correct), CPU
Backward returns `Tensor::Zeros(grad_output.Shape())`, weights never
update, loss is flat across epochs.

**Three stacked problems in `cyxwiz-backend/src/algorithms/layer.cpp`:**

### 1. `GRULayer::Forward` ArrayFire path throws size-mismatch on assign

**Location:** `layer.cpp:3098-3209` (AF try/catch block inside
`#ifdef CYXWIZ_HAS_ARRAYFIRE`).

**Symptom:** AF exception ERR_SIZE 203 inside `af::array_proxy::
operator=`. Same family as the LSTM AF Forward bug - almost certainly
a row-major / column-major mismatch at one of the 3D slice writes:
- `h_states(0, af::span, af::span) = h;` (line 3156)
- `h_states(t + 1, af::span, af::span) = h;` (line 3177)
- `all_gates(t, af::span, af::span) = gates;` (line 3179)
- `h_full(layer, af::span, af::span) = h;` (line 3192)

The RHS shape (`[batch, hidden]` 2D) doesn't match the rank-3 proxy
on the LHS.

**Fix (mirror LSTM):** wrap each RHS in `af::moddims(rhs,
af::dim4(1, batch, hidden))` so the rank matches. Likely also need
the row-major 3D helpers (`TensorToAf3DRowMajor` /
`AfToTensor3DRowMajor`) at the boundary if input dim interpretation
is also scrambled. See `LSTMLayer::Forward` post-`38d0a250`/`f4ac9b57`
for the reference fix.

### 2. CPU `GRULayer::Forward` fallback doesn't populate the AF caches

**Location:** `layer.cpp:3215-3344` (CPU fallback path).

**Issue:** AF Forward (when it worked) populated:
- `cached_inputs_`
- `cached_gates_`
- `cached_hidden_states_`

CPU fallback computes everything as `std::vector<float>` locals and
discards them on exit. Backward then sees empty caches.

**Fix:** During the CPU pass, gather per-layer per-timestep gates
(reset / update / new) and hidden states into row-major scratch
buffers, then wrap as Tensors and push into the cache vectors before
return. Same shape contract as the AF path: `[seq_len, batch,
3*hidden]` for gates, `[seq_len + 1, batch, hidden]` for hidden
states.

### 3. `GRULayer::Backward` returns zero gradients on cache miss

**Location:** `layer.cpp:3347-...` (start of Backward).

**Issue:** Mirror of the LSTM stub:

```cpp
if (cached_inputs_.empty() || cached_gates_.empty() ||
    cached_hidden_states_.empty()) {
    spdlog::warn("GRULayer::Backward called without cached data from
                  Forward, returning zero gradients");
    return Tensor::Zeros(grad_output.Shape());
}
```

Once #2 lands the warning stops, but the AF Backward path may also
fail and need a CPU equivalent. ~60-80 lines of BPTT over the GRU
gate equations (reset / update / new), per timestep:
- d_h_t = grad_output[t] + dh_next
- d_z = d_h_t * (h_prev - n_t) * z_t * (1 - z_t)
- d_n = d_h_t * (1 - z_t) * (1 - n_t^2)
- d_r = d_n * (h_proj_n) * r_t * (1 - r_t)
- Accumulate into dW_ih, dW_hh, db_ih, db_hh
- dh_next = d_h_t * z_t + (matmul terms through gates)

Reference: PyTorch's GRU C++ source, or the equivalent of LSTM CPU
BPTT we landed in `38d0a250`.

### Recommended fix sequence (separate session)

1. **Add CPU Backward (biggest chunk).** Without this, GRU can never
   learn on CPU. Read from whatever caches Forward populates.
2. **Populate caches in CPU Forward** so #1 has something to read.
3. **Delete the zero-grad fallback in Backward** - once Backward has
   a real CPU path, empty caches mean something's wrong; want a real
   error, not silent zero.
4. **(Optional) Fix AF Forward / Backward** for performance. CPU
   path will work but be slower than a fixed GPU path.

The AF Forward fix is where LSTM dragged on - the slice-shape
moddims pattern needs to be applied at every 3D RHS write. Worth
landing the CPU path first so the smoke test goes green, then
revisit AF as a perf pass.

### Test after fix

- Rerun `test_04_sentiment_gru_mini.cyxgraph` for 3 epochs (same
  config as the LSTM mini).
- Expected: monotonic loss decrease, val_acc climbs above the 7-class
  random baseline (~14%) into the 30-50% range like the LSTM mini.
- Side-by-side: `test_03_sentiment_lstm_mini` and
  `test_04_sentiment_gru_mini` should produce comparable convergence
  curves (GRU slightly faster per-batch, similar final accuracy).

### Files touched during the 2026-04-17 wiring pass that need revisiting

- `cyxwiz-backend/include/cyxwiz/sequential.h` - `GRUModule`
  declaration (keep)
- `cyxwiz-backend/src/algorithms/sequential.cpp` - `GRUModule`
  implementation + one-shot constructor warning (keep; remove the
  warning once GRULayer is fixed)
- `cyxwiz-engine/src/core/training_executor.cpp` - `GRU` case
  branch (keep)
- `examples/cyxgraph/text/test_04_sentiment_gru_mini.cyxgraph` -
  the smoke test graph that surfaced all this (keep as a regression
  fixture)
- `cyxwiz-backend/src/algorithms/layer.cpp` - `GRULayer::Forward` AF
  block + CPU fallback + `Backward` zero-grad stub (rewrite during
  the fix)

---

## ~~DataInputDialog::Apply is a 700-line category switch~~ LANDED 2026-04-18

**Status:** Fully refactored into a polymorphic `DataLoader` hierarchy
across 8 plan commits on `feat/dataloader-refactor` (plus 4 merge
commits into `Nodes_Implementation`). Plan doc at
`docs/plans/dataloader_refactor.md` was the execution guide, expanded
mid-plan per user feedback to cover every per-category concern (not
just async).

Plan commits (all merged, branch + worktree retired):
- `f0ae575c` - DataLoader interface + ApplyContext + AsyncLoadState
  hoist + TabularLoader (commit 1/8)
- `7000fb85` - TextLoader (commit 2/8)
- `8ac3d1e7` - ImageLoader (commit 3/8)
- `22574497` - AudioLoader (commit 4/8)
- `798e48e0` - Loader-driven restore + poll description (commit 5/8).
  Dialog-reopen restore and `PollAsyncLoadResult`'s backend switch
  both route through loader methods.
- `e4b91437` - Training dispatch via loader + `StartTrainingCommon`
  (commit 6/8). `MainWindow::StartTrainingFromGraph` 140-line switch
  -> 15 lines. `TrainingManager::StartTrainingCommon` extracts shared
  plumbing across the 5 `StartTraining*` methods.
- `7f003b24` - Compile gate via loader (commit 7/8). 5-way
  `IsXDataset` OR -> `GetByRegisteredDataset != nullptr`.
  `labels_from_structure` and `Domain()` both route through the
  loader.
- `83788a36` - Node-param schema + stale-param pruning + MakeSynthetic
  stub (commit 8/8). `Apply` now prunes stale per-category params on
  category switch; `MakeSynthetic` is a forward-looking surface for
  Local Debug integration.

**Net effect:** The DataLoader interface owns every concern that
used to switch on `file_category_` / `loaded_backend_` / `IsXDataset`:
Apply launch, Poll finalization, dialog-reopen restore, memory tab
render, training dispatch, compile-gate domain + labels-from-
structure, node-param schema, MakeSynthetic stub. Adding a new data
type is now one new `DataLoader` subclass + one `RegisterLoader` call;
every UI and runtime path sees it automatically.

**DataRegistry:** Option A (routing sidecar) shipped - `name_to_category_`
map populated by each `Register*` method so `GetByRegisteredDataset`
is a single lookup instead of 5 parallel `IsXDataset` calls. The 5
underlying maps (Arrow / Parquet / Image / Audio / Text) stay intact;
the 16 callsites across Data Studio / properties / visualizer keep
working unchanged.

**Follow-up work explicitly scoped out of v1** (for future sessions):
- `AsyncLoadState` variant consolidation (per-backend extras like
  `vocab_size` / `feature_rows,cols` live as flat fields today; could
  move to a `std::variant<TabularExtra, TextExtra, ...>`). Cosmetic.
- `IBatcher` + `TrainingExecutor` constructor consolidation. Today
  `TrainingExecutor` has 4 constructors (Legacy, Arrow, Parquet,
  IBatcher). `IBatcher` is already the shared base for image / audio
  / text batchers; Arrow + Parquet could join it with moderate effort.
- `MakeSynthetic` stubs need real implementations when Local Debug's
  per-category dispatch lands.
- Plugin data loaders (`PluginDataLoaderRegistry`) stay on their own
  runtime registration path; integrating them with
  `GetByRegisteredDataset` is deferred.

---

## ~~Node registration is split across three hand-maintained lists~~ LANDED 2026-04-17

**Status:** Resolved across commits `c3ad2731` (search palette) +
`897c41fd` (context menu). `NodeMetadataRegistry` is now the single
source of truth for browser + search palette + context menu. Adding
a new node type requires one `RegisterNode(...)` call in the
appropriate `Initialize*Nodes()` function; all three UIs pick it up
automatically.

Side benefit: the hand-coded context menu had silently omitted most
of the `NodeCategory` enum (Pooling, Normalization, Recurrent,
Training, Attention, ShapeOps, MergeOps, Upsampling, Regularization,
TextProcessing, TimeSeries, Audio, RL, DNN, Utility, Signal,
DataPipeline) - users couldn't add those via right-click at all.
Registry-driven menu auto-surfaces every registered category.

**Known remaining dual-maintained lists** (same unification pattern
applies - separate follow-ups, not blockers):
- `StringToNodeType` dual maps in `node_editor_io.cpp:20` +
  `pattern_library.cpp:331`. Requires reverse `name -> type` lookup
  on the registry plus alias handling (e.g. `"BatchNorm2D"` vs
  `BatchNorm`).
- `ShouldShowOpenDialogButton` whitelist (`node_config_dialog.cpp:
  970`). Could derive from `NodeConfigDialogFactory::HasDialog(type)`
  directly.

---

## Tool-to-Node Migration - ~40 NodeTypes with standalone panels but no graph integration

**Severity:** HIGH (architectural) - this is the biggest consistency gap
in the codebase. ~80 floating analytical panels live under
`cyxwiz-engine/src/gui/panels/*.h` and most have a corresponding
NodeType in `node_editor.h`, but almost none of them are wired to
the execution pipeline. The v1 "everything is a node" philosophy is
only half-honored.

**Status update 2026-04-16 - Phase 4 time-series block CLOSED.** The
`TimeSeriesWindow`, `TimeSeriesFeatures`, and `TimeSeriesSplit`
NodeTypes listed below as "dead" are now live Cat-1 operators with
working executors (commits `aacbadcd`, `272dca2e`). The Phase 4
time-series row in the inventory table can be struck through. Two
NEW NodeTypes were added as part of this work (`LogTransform`,
`Differencing`) - both also live operators. See
`docs/phase4_time_series_plan.md` "What actually shipped" for the
full breakdown.

**Status update 2026-04-16 - Tool-to-Node migration ROUND 2 closed.**
In this session, 24 additional Cat-1 operators landed across seven
blocks, bringing the total from 5 -> **29 live Cat-1 operators**:
- **Text analytics** (4): `TextTokenizer`, `TFIDFVectorizer`,
  `CountVectorizer`, `SentimentAnalyzer` (lexicon-based)
- **Linear algebra** (1): `PCANode` (SVD-based)
- **Clustering** (4): `KMeansCluster`, `DBSCANCluster`,
  `HierarchicalCluster`, `GMMCluster` (all GPU-accelerated via AF)
- **Signal processing** (3): `FFTNode`, `Convolution1D`,
  `FilterDesigner` (combines design+apply)
- **Classical regression** (2): `LinearRegressionNode` (multi-
  predictor OLS), `PolynomialRegressionNode` (univariate)
- **Data preprocessing** (7): `StandardScaler`, `MinMaxScaler`,
  `RobustScaler`, `LabelEncoder`, `OrdinalEncoder`, `TargetEncoder`,
  `OutlierDetector` - closes the Phase 4 preprocessing block
- **Phase 5 time-series analysis** (3): `TimeSeriesDecomposition`,
  `ARIMAForecaster`, `ExponentialSmoothing` - in-sample fit only
  (horizon=0 preserves row count; future-rows forecasting deferred)

**Of the remaining ~18 dead NodeTypes, every single one is now
deferred with explicit reasoning:**
- **Cat-2 introspection (don't force into Cat-1)** - ACF/PACF,
  StationarityTest, SeasonalityDetector, HypothesisTest,
  DistributionFitter, ConfusionMatrix, ROC/PR curves,
  LearningCurves, FeatureImportance, CrossValidation,
  RegressionMetrics, GradCAM, SaliencyMap, WordFrequency,
  GradientDescentViz, ConvexityAnalyzer, SVD/QR/Cholesky/Eigen.
- **No backend implementation** - DecisionTree, RandomForest, GBM,
  SVM, KNN, NaiveBayes, LogisticRegression.
- **Cat-3 dev utility (stays panel)** - MatrixCalculator, LPSolver,
  QPSolver, NumericalDiff, NumericalInt.
- **Awkward Arrow schema** - IFFT (complex-pair input),
  WaveletTransform (variable-length coeffs/level), ARIMA/ES
  (forecast rows change row count), TSNE/UMAP (iterative +
  interactive), WordEmbeddings (needs GloVe/Word2Vec file loader),
  NamedEntityRecognizer (needs real ML model).
- **Plumbing only** - TimeSeriesDecomposition (backend exists,
  NodeType exists in enum, no CreateNode case yet; when wired it's
  honestly Cat-1).

The framework (`node_executors/`, `PipelineMaterializer`,
`IPipelineOperator`, `feature_matrix_utils.h`,
`text_column_utils.h`, `ts_column_utils.h`) is proven across six
domains. Adding new Cat-1 operators is now ~1 hour of boilerplate
per operator.

**Discovered:** 2026-04-15, when the user asked why the graph has both
an `Embedding` node and a separate standalone `Word Embeddings`
floating panel. The answer - that the `WordEmbeddings` NodeType
exists in the catalogue but has zero backing - generalizes to most
of the Phase 4 and Phase 5 Tool-to-Node groups in `node_editor.h`.
The enum even self-documents the plan:

```cpp
// ===== Machine Learning Algorithms (Phase 4 - Tool-to-Node Migration) =====
// ===== Linear Algebra Nodes (Phase 5 - Tool-to-Node Migration) =====
// ===== Time Series Analysis Nodes (Phase 5) =====
// ===== Additional Text Processing (Phase 5) =====
```

Phase 4 and Phase 5 were explicitly declared as "turn the tool panels
into graph nodes" work. The NodeType declarations landed; the wiring
did not.

### Framework status (as of Phase 4 close, 2026-04-16)

`cyxwiz-engine/src/core/node_executors/` is now committed and proven
via five working Cat-1 operators (LogTransform, Differencing,
TimeSeriesFeatures, TimeSeriesWindow multivariate, TimeSeriesSplit).
The pattern for adding a new Cat-1 pipeline op is:

1. Subclass `IPipelineOperator` (see `pipeline_operator.h`)
2. Implement `Configure(params, error)` reading string params
3. Implement `Apply(input_table)` returning a transformed Arrow table
4. Register via `PipelineOperatorFactory::RegisterCreator`
5. Add `.cpp`/`.h` to CMakeLists, NodeType to both loader maps,
   CreateNode defaults, GraphCompiler preprocessing specs

Shared utilities in `ts_column_utils.h` (`ReadColumnAsFloat`,
`ReplaceColumnWithFloat`) cover the column-read and column-replace
boilerplate. `PipelineMaterializer` walks the graph and dispatches
all registered ops automatically - no per-op integration needed in
the dispatcher.

### What's already partially started

`cyxwiz-engine/src/core/node_executors/` (originally untracked in git,
now committed as of `52808745`)
contains scaffolding for a per-node execution framework separate from
`training_executor`:

- `node_executor.h` - `ExecutorState` (Idle/Configuring/Executing/
  Completed/Error/Cancelled) + `CodeFramework` (Sklearn, PyCyxWiz,
  PyTorch) + base executor interface.
- `node_executor_factory.h` - factory to instantiate executors by
  NodeType.
- `kmeans_executor.{cpp,h}` - first concrete executor (KMeans
  clustering), mirroring `KMeansPanel`'s functionality but as a
  node-graph operation.

**This is exactly the migration framework that was needed.** It was
started and abandoned. The pattern is the right one - most tool
nodes are NOT neural-network layers; they're classical ML / stats /
signal processing operations that execute once and produce a result.
They shouldn't be in `training_executor.cpp`'s layer-building
switch. They belong in a per-node executor framework.

### The full "dead NodeType + live panel" inventory

Each row: `NodeType` exists in `node_editor.h` but has **no training_
executor case, no executor**, while the matching floating panel
already implements the actual computation. Dropping the node in a
graph today is a visual no-op.

**Text analytics (Phase 5 block):**
| NodeType | Panel file | Status |
|---|---|---|
| `WordEmbeddings` | `embeddings_panel.h` | ~~dead~~ **DEFERRED** - see entry below |
| `TFIDFVectorizer` | `tfidf_panel.h` | ~~dead~~ **LIVE** (Cat-1 operator, 2026-04-16) |
| `CountVectorizer` | *(none found)* | ~~dead~~ **LIVE** (Cat-1 operator, 2026-04-16) |
| `SentimentAnalyzer` | `sentiment_panel.h` | ~~dead~~ **LIVE** (Cat-1 lexicon-based, 2026-04-16) |
| `NamedEntityRecognizer` | *(none found)* | **DEFERRED** - see entry below |
| `WordFrequencyNode` | `word_frequency_panel.h` | **RECLASSIFIED** to Cat-2 panel, not Cat-1 node - see entry below |
| (tokenization - see note) | `tokenization_panel.h` | covered by `TextTokenizer` Phase 3 node (live Cat-1 since 2026-04-16) |

**Machine learning algorithms (Phase 4 block):**
| NodeType | Panel file | Status |
|---|---|---|
| `KMeansCluster` | `kmeans_panel.h` | ~~dead~~ **LIVE** (Cat-1 GPU KMeans, 2026-04-16) |
| `DBSCANCluster` | `dbscan_panel.h` | ~~dead~~ **LIVE** (Cat-1 GPU DBSCAN, 2026-04-16) |
| `HierarchicalCluster` | `hierarchical_panel.h` | ~~dead~~ **LIVE** (Cat-1 agglomerative, 2026-04-16) |
| `GMMCluster` | `gmm_panel.h` | ~~dead~~ **LIVE** (Cat-1 GMM w/ hard labels, 2026-04-16) |
| `PCANode` | `dim_reduction_panel.h` | ~~dead~~ **LIVE** (Cat-1 SVD-based PCA, 2026-04-16) |
| `TSNENode` / `UMAPNode` | `dim_reduction_panel.h` | **DEFERRED** - iterative, Cat-2 introspection better fit |
| `LinearRegressionNode` | `regression_panel.h` | ~~dead~~ **LIVE** (Cat-1 multi-predictor OLS, 2026-04-16) |
| `PolynomialRegressionNode` | `regression_panel.h` | ~~dead~~ **LIVE** (Cat-1 univariate, 2026-04-16) |
| `LogisticRegressionNode` | `regression_panel.h` | **DEFERRED** - no backend impl yet |
| `DecisionTreeClassifier` | *(none found)* | **DEFERRED** - no backend impl |
| `RandomForestClassifier` | *(none found)* | **DEFERRED** - no backend impl |
| `GradientBoostingClassifier` | *(none found)* | **DEFERRED** - no backend impl |
| `SVMClassifier` / `SVMRegressor` | *(none found)* | **DEFERRED** - no backend impl |
| `KNNClassifier` | *(none found)* | **DEFERRED** - no backend impl |
| `NaiveBayesClassifier` | *(none found)* | **DEFERRED** - no backend impl |

**Model evaluation (Phase 4 block) - ALL DEFERRED as Cat-2 introspection:**
| NodeType | Panel file | Status |
|---|---|---|
| `ConfusionMatrixNode` | `confusion_matrix_panel.h` | **DEFERRED** - Cat-2 introspection |
| `ROCCurveNode` | `roc_auc_panel.h` | **DEFERRED** - Cat-2 introspection |
| `PRCurveNode` | `pr_curve_panel.h` | **DEFERRED** - Cat-2 introspection |
| `LearningCurvesNode` | `learning_curves_panel.h` | **DEFERRED** - Cat-2 introspection |
| `FeatureImportanceNode` | `feature_importance_panel.h` | **DEFERRED** - Cat-2 introspection |
| `CrossValidationNode` | `cross_validation_panel.h` | **DEFERRED** - Cat-2 orchestration (not a transform) |
| `RegressionMetricsNode` | `regression_panel.h` | **DEFERRED** - Cat-2 introspection |

*Rationale:* model evaluation produces scalar metrics and interactive
plots, not transformed data rows. They belong as Cat-2 panels that
hook into any trained-model point and render on demand. See CLAUDE.md
"Node vs Panel: Four Tool Categories".

**Linear algebra (Phase 5 block):**
| NodeType | Panel file | Status |
|---|---|---|
| `PCANode` | `dim_reduction_panel.h` | ~~dead~~ **LIVE** (Cat-1 SVD-based PCA, 2026-04-16) |
| `SVDNode` | `svd_panel.h` | **DEFERRED** - Cat-2 introspection (multi-output decomposition) |
| `QRDecomposition` | `qr_panel.h` | **DEFERRED** - Cat-2 introspection (multi-output decomposition) |
| `CholeskyDecomposition` | `cholesky_panel.h` | **DEFERRED** - Cat-2 introspection (multi-output decomposition) |
| `EigenDecomposition` | `eigen_decomp_panel.h` | **DEFERRED** - Cat-2 introspection (multi-output decomposition) |
| `MatrixCalculator` | `matrix_calculator_panel.h` | **DEFERRED** - Cat-3 dev utility (stays panel) |

**Time series analysis (Phase 5 block):**
| NodeType | Panel file | Status |
|---|---|---|
| `TimeSeriesDecomposition` | `decomposition_panel.h` | ~~dead~~ **LIVE** (Cat-1, classical+STL, 2026-04-16) |
| `ARIMAForecaster` | `forecasting_panel.h` | ~~dead~~ **LIVE** (Cat-1, in-sample fit, 2026-04-16) |
| `ExponentialSmoothing` | `forecasting_panel.h` | ~~dead~~ **LIVE** (Cat-1, simple/holt/holt_winters in-sample, 2026-04-16) |
| `ACFNode` / `PACFNode` | `acf_pacf_panel.h` | **DEFERRED** - Cat-2 introspection (correlation arrays don't align with rows) |
| `StationarityTest` | `stationarity_panel.h` | **DEFERRED** - Cat-2 introspection (scalar test results) |
| `SeasonalityDetector` | `seasonality_panel.h` | **DEFERRED** - Cat-2 introspection (periodogram + detected periods) |

*Forecasting note:* ARIMAForecaster and ExponentialSmoothing are
wired with `horizon=0` so only in-sample fitted values (+ residuals)
are written to the table - row count preserved, no alignment break.
True out-of-sample forecasting (horizon > 0) would append future
rows to the table, which needs a dedicated "forecast-rows" operator
or a Cat-2 visualization panel. Tracked separately as a future
enhancement; the Phase 5 NodeTypes ship today in the in-sample
semantic that matches residual-diagnostic workflows.

**Signal processing (Phase 4 block):**
| NodeType | Panel file | Status |
|---|---|---|
| `FFTNode` | `fft_panel.h` | ~~dead~~ **LIVE** (Cat-1 1D FFT, 2026-04-16) |
| `IFFTNode` | `fft_panel.h` | **DEFERRED** - see entry below (needs complex-pair input schema) |
| `FilterDesigner` | `filter_designer_panel.h` | ~~dead~~ **LIVE** (Cat-1 design+apply, 2026-04-16) |
| `Convolution1D` | `convolution_panel.h` | ~~dead~~ **LIVE** (Cat-1, kernel as param, 2026-04-16) |
| `WaveletTransform` | `wavelet_panel.h` | **DEFERRED** - see entry below (variable-length coeffs per level) |

**Statistics (Phase 5 block) - ALL DEFERRED as Cat-2 introspection:**
| NodeType | Panel file | Status |
|---|---|---|
| `HypothesisTest` | `hypothesis_test_panel.h` | **DEFERRED** - Cat-2 introspection (scalar test result) |
| `DistributionFitter` | `distribution_fitter_panel.h` | **DEFERRED** - Cat-2 introspection (fit parameters + goodness-of-fit) |

*Rationale:* hypothesis tests produce a scalar test statistic +
p-value + interpretation; distribution fitters produce fit parameters
and Kolmogorov-Smirnov statistics. Neither transforms row-level data.
The `DataAnalyzer::OneSampleTTest` / `TwoSampleTTest` / `PairedTTest` /
`ChiSquareTest` + `FitNormal` / `FitExponential` backend methods
exist; they should drive Cat-2 panels, NOT be forced into Cat-1
schema.

**Deep learning interpretation (Phase 5 block) - ALL DEFERRED as Cat-2:**
| NodeType | Panel file | Status |
|---|---|---|
| `GradCAMNode` | `gradcam_panel.h` | **DEFERRED** - Cat-2 introspection (saliency heatmap on a trained model) |
| `SaliencyMapNode` | `visualization_panel.h` | **DEFERRED** - Cat-2 introspection |

**Optimization (Phase 5 block) - ALL DEFERRED:**
| NodeType | Panel file | Status |
|---|---|---|
| `GradientDescentViz` | `gradient_descent_panel.h` | **DEFERRED** - Cat-2 visualization |
| `ConvexityAnalyzer` | `convexity_panel.h` | **DEFERRED** - Cat-2 scalar analysis |
| `LPSolver` | `lp_panel.h` | **DEFERRED** - Cat-3 dev utility (stand-alone solver, not a pipeline transform) |
| `QPSolver` | `qp_panel.h` | **DEFERRED** - Cat-3 dev utility |
| `NumericalDifferentiation` | `differentiation_panel.h` | **DEFERRED** - Cat-3 dev utility |
| `NumericalIntegration` | `integration_panel.h` | **DEFERRED** - Cat-3 dev utility |

**Utility (Phase 4 block):**
| NodeType | Panel file | Status |
|---|---|---|
| `CalculatorNode` | `calculator_panel.h` | dead |
| `UnitConverter` | `unit_converter_panel.h` | dead |
| `RegexTester` | `regex_tester_panel.h` | dead |
| `JSONPathExtractor` | `json_viewer_panel.h` | dead |
| `DataProfiler` | `data_profiler_panel.h` | dead |

**Data preprocessing (Phase 4 block) - ALL LIVE (2026-04-16):**
| NodeType | Panel file | Status |
|---|---|---|
| `StandardScaler` | `standardization_panel.h` | ~~dead~~ **LIVE** (Cat-1, z-score, 2026-04-16) |
| `MinMaxScaler` | `feature_scaling_panel.h` | ~~dead~~ **LIVE** (Cat-1, custom range, 2026-04-16) |
| `RobustScaler` | `feature_scaling_panel.h` | ~~dead~~ **LIVE** (Cat-1, median/IQR, 2026-04-16) |
| `OutlierDetector` | `outlier_detection_panel.h` | ~~dead~~ **LIVE** (Cat-1, IQR/Z-score flag, 2026-04-16) |
| `LabelEncoder` | *(none found)* | ~~dead~~ **LIVE** (Cat-1, single col, 2026-04-16) |
| `OrdinalEncoder` | *(none found)* | ~~dead~~ **LIVE** (Cat-1, multi col, auto-order, 2026-04-16) |
| `TargetEncoder` | *(none found)* | ~~dead~~ **LIVE** (Cat-1, smoothed mean, 2026-04-16) |

*Deferred sub-features:*
- OutlierDetector `action=remove` (row deletion breaks downstream
  alignment) and `action=clip` (reasonable but more complex) -
  v1 only wires `action=flag`.
- OutlierDetector `method=isolation_forest` and `method=lof` - no
  backend impl yet; only IQR + Z-score supported in v1.
- OrdinalEncoder custom per-column `categories` ordering - v1 only
  supports `categories=auto` (alphabetical); custom ordering
  needs a nested param schema.

### Why this matters

- **User confusion.** Drop a `WordEmbeddings` or `KMeansCluster` node
  on the canvas, nothing happens. Right-click doesn't open the
  relevant panel. The panel is hiding in the toolbar menu. Two
  unrelated surfaces for the same concept.
- **Lost v1 promise.** The original CyxWiz Engine v1 pitch was
  "everything is a node." The current hybrid state violates that -
  some nodes are real, some are dead catalogue entries, and the
  actual tools live outside the graph.
- **Feature discovery.** 80+ analytical panels are buried in a
  toolbar menu. If they were all connectable nodes with rich
  configure dialogs, the node browser would surface them naturally.
- **Graph reproducibility.** A panel session produces results that
  vanish when the panel closes. A node-graph session produces a
  persistent pipeline that can be saved, loaded, re-run, and
  shared.

### Recommended fix sequence (multi-session)

Not a single commit - this is phase-scale work. The right order:

**Step 1 - ~~Finish the `node_executors` framework.~~ DONE (2026-04-16).**
Framework shipped in commit `52808745` with `IPipelineOperator` base
class, `PipelineBand` enum, `PipelineOperatorFactory`, and the
`PipelineMaterializer` dispatcher wired into training launch
(`8b6055b0`). Five working Cat-1 operators ride on it
(LogTransform, Differencing, TimeSeriesFeatures, TimeSeriesWindow
multivariate, TimeSeriesSplit). KMeans executor (Cat-2 introspection
variant) also committed as part of the framework landing. See
`docs/phase4_time_series_plan.md` "What actually shipped" for details.

**Step 2 - Establish the "rich dialog on double-click" convention
for tool nodes.** The existing `DataInputDialog` and `TokenizerDialog`
are the model: a node-config-dialog dispatch in `node_config_dialog`
that looks up the node's type and opens a custom dialog. Refactor
`EmbeddingsPanel` / `KMeansPanel` / `SVDPanel` etc. to live inside
a node-config dialog instead of as floating windows.

**Step 3 - Batch-migrate by category.** Start with one category end
to end before moving to the next, to avoid leaving everything half-
done. Suggested order:
1. Text analytics (user wants this, fewest executors, shares
   infrastructure with the Phase 3 text path)
2. Linear algebra (lowest complexity, pure math, easy validators)
3. Clustering ML (KMeans already scaffolded)
4. Signal processing (FFT / wavelet)
5. Statistics
6. Remaining ML algorithms
7. ~~Time series~~ DONE - Phase 4 shipped Window / Features / Split
   as real operators (2026-04-16).
8. Model evaluation + interpretation (runs AFTER training, different
   execution context)

**Step 4 - Delete the standalone panels.** Once a tool has a rich
node dialog, remove its toolbar-menu button and delete the `.h/.cpp`
for the standalone panel. This is the signal that consolidation
actually shipped.

**Step 5 - Document the node-first convention in CLAUDE.md.** Add
a "Nodes vs panels" section that says: (a) analytical tools MUST be
graph nodes with rich dialogs, never standalone panels; (b)
standalone panels are reserved for debug/monitoring (console,
profiler, memory monitor, job status, etc.) that aren't part of the
user's pipeline; (c) Properties panel is for simple key-value
layer params (Dense units, Dropout rate, etc.), rich dialog is for
anything with preview, validation, or multi-tab config.

### Scope estimate

- Per category: 1-3 sessions depending on node count and complexity
- Full migration: ~6-10 sessions
- Alternative: leave the dead NodeTypes deleted from the catalogue
  entirely until they're ready, so users aren't misled. Cheap fix,
  addresses the user-confusion problem without doing the real
  consolidation. Would cut the enum by ~40 entries and the Add menu
  by ~40 entries.

### Deferred text-analytics items (2026-04-16)

The text-analytics block landed `TextTokenizer`, `TFIDFVectorizer`,
`CountVectorizer`, and `SentimentAnalyzer` as live Cat-1 operators.
These four were skipped deliberately and are parked here:

**`WordEmbeddings` - needs pretrained file loader.** A Cat-1 operator
that emits random embeddings is worse than not shipping one: Band 4
already has a trainable `Embedding` layer that learns task-specific
representations end-to-end, so random mean-pooled embeddings in
Band 1 add noise without adding signal. The honest version needs a
file picker dialog + GloVe / Word2Vec / FastText parser, token-to-
vector lookup with OOV fallback, and vocabulary alignment against
the input corpus. Estimated ~1 day. Until then the `WordEmbeddings`
NodeType stays dead and users get directed to the Band 4
`Embedding` layer.

**`NamedEntityRecognizer` - needs a real ML model.** No honest
rule-based shortcut. Real NER either wraps spaCy (Python dep,
licensing, startup cost, crosses process boundary), embeds a small
ONNX CRF model (need to pick one and ship weights), or runs a
BiLSTM-CRF trained on CoNLL-2003 (training pipeline + checkpoint
distribution). Each option is multi-day. Deferred indefinitely
until a product need forces a choice.

**`WordFrequencyNode` - reclassified as Cat-2 introspection panel.**
Corpus-level word counts are a debugging/exploration view, not a
transformation that feeds downstream nodes. It belongs in the
"Introspection tools" category (panel that hooks any pipeline point
read-only, rendered on-demand when the user opens it). Keeping the
existing `word_frequency_panel.h` as-is and NOT wiring it as a
Cat-1 NodeType is the correct call. If the NodeType still shows in
the Add menu, it should be removed or routed to open the panel.

**Pretrained Sentiment model variants (BERT/DistilBERT/etc.).** The
shipped `SentimentAnalyzer` wraps lexicon-based sentiment (simple /
VADER / AFINN) which is zero-dependency and deterministic. Neural
sentiment classifiers would need the same pretrained-weights
distribution story as NER - deferred on identical grounds.

### Deferred signal-processing items (2026-04-16)

**`IFFTNode` - complex-pair column schema.** Inverse FFT needs
complex-valued input (pairs of real + imaginary columns, or a
single complex-typed column which Arrow doesn't natively support).
The honest version either (a) reads paired `{prefix}_real` +
`{prefix}_imag` columns and reconstructs complex input, or (b)
chains against a produced-by-FFT table using a matching-schema
contract. (a) is simpler but awkward; (b) requires FFT to emit
the raw complex output alongside magnitude/phase. Neither is
urgently needed - most ML pipelines consume frequency-domain
features, not round-trip to time-domain. Deferred until a concrete
use case forces the schema decision.

**`WaveletTransform` - variable-length coefficients per level.** DWT
produces a final approximation vector plus one detail vector per
level, each ~half the length of the previous. Mapping to Arrow
requires either (a) a long-format table with `level, index,
coefficient_type, value` rows, (b) separate output tables per
level, or (c) nested list-type columns. All three are awkward for
downstream training/viz. Better served as a Cat-2 introspection
panel that renders the multi-resolution decomposition
interactively, which the existing `wavelet_panel.h` already
supplies. Deferred indefinitely.

---

## TextTokenizer is a config extractor, not a pipeline operation

**Status update 2026-04-16:** Step 1 of Fix B landed - a real
`TextTokenizerOperator` (Cat-1, Band 1, single op combining
tokenize + vocab + padding) is in place at `cyxwiz-engine/src/
core/node_executors/text_tokenizer_operator.{h,cpp}`. It's
registered in `PipelineOperatorFactory` under
`gui::NodeType::TextTokenizer` so dropping the node in an Arrow
graph fires it through `PipelineMaterializer`. Output schema
matches what `ArrowDatasetBatcher` already consumes (wide
`tok_0..tok_{max-1}` float columns + `y` int column), so no
batcher changes needed. CreateNode defaults gained `text_col` /
`label_col` / `min_word_freq` / `max_vocab_size` while keeping
the legacy `tokenizer_type` / `max_length` / `lowercase` /
`min_freq` / `padding` / `truncation` aliases so the existing
GraphCompiler config-extractor path keeps working unchanged for
RegisterTextDataset graphs.

**Status update 2026-06-03:** Frontend exposure slice landed. The
`TokenizerDialog` now writes the canonical text-node params
(`text_col`, `label_col`, `tokenizer_type`, `max_length`,
`min_word_freq`, `max_vocab_size`) and keeps the compiler/operator
aliases (`min_freq`, `text_column`, `padding`, `truncation`) in sync.
It is registered for `TextTokenizer`, `TextVocabulary`, and
`TextPadding`, so all three nodes expose an Open Dialog path. The
stale unsupported BPE/pretrained/special-token UI was removed from
this dialog. This does **not** complete full Fix B; the remaining work
is still the backend dataflow rewrite below.

**Status update 2026-06-03:** Backend activation slice landed for
single-file text CSV/TSV. `TextLoader` now registers the raw text file
as an Arrow table under the dataset name, while still registering the
legacy `TextDatasetEntry` metadata for compile/train/test
compatibility. Loader ownership now prefers explicit Text/Image/Audio
entries over generic Arrow ownership for original dataset names, so
legacy text graphs still route through `TextLoader`; materialized
`__materialized` Arrow outputs still route through `TabularLoader`.
Added `test_text_tokenizer_operator` to validate the Arrow input
contract for `TextTokenizerOperator`.

**Still to do for full Fix B:**
- DataInput text branch: CSV/TSV now dual-registers raw Arrow +
  TextDatasetEntry. Finish the rewire by making Arrow the canonical
  path, deciding the JSON/TXT/folder-corpus raw table story, and then
  removing the legacy `TextDatasetEntry` dependency.
- MainWindow dispatch: route `IsArrowDataset` after materializer
  for text graphs (already happens - `__materialized` flows
  through Arrow path).
- GraphCompiler: delete the `ExtractTextTokenizer` /
  `ExtractTextVocabulary` / `ExtractTextPadding` extractors and
  the `text_preprocessing` config struct now that the operator
  carries config inline.
- TextDatasetBatcher: delete (no consumers after the dispatch
  rewire). `formats/text_dataset.{h,cpp}` may also become dead
  code - check usage first.
- Optional: split TextTokenizer back into TextTokenizer +
  TextVocabulary + TextPadding for graph addressability. v1 is
  one combined node since TimeSeriesWindow set the precedent
  for "combine tightly-coupled steps in one operator."
- Update existing text smoke graphs (test_01, v1, v2,
  test_02_lstm) to the new pipeline shape, and re-run as the
  Fix B regression check.

The new operator path is currently INERT for legacy text
graphs (PipelineMaterializer skips non-Arrow datasets), so this
landed without breaking existing tests. Step 2+ (DataInput
rewire) is the next session's scope.

**Severity:** HIGH (architectural) - blocks the "preprocess once,
train many" workflow and makes the `TextTokenizer` / `TextVocabulary`
/ `TextPadding` nodes visually misleading (they have input/output
pins but no data flows through them).

**Discovered:** 2026-04-16 while discussing node architecture. The
user asked whether a user could build a graph like
`ReadCorpus -> TextTokenizer -> WriteFile(tokenized.jsonl)` to
pre-tokenize a corpus once and reuse the tokenized file across many
training runs. Answer: no, not today.

### Current state (the shortcut we took in Phase 3)

The `TextTokenizer` / `TextVocabulary` / `TextPadding` nodes are
implemented as **configuration extractors** in
`cyxwiz-engine/src/core/graph_compiler.cpp:1065-1121`. When Compile
runs:

1. The compiler scans the graph for these node types via the
   `kPreprocessingExtractors` table.
2. For each one found, it reads the node's parameters into
   `config.text_preprocessing` (tokenizer type, max_length,
   min_word_freq, max_vocab_size, pad_value, etc.).
3. At train time, `TextDatasetBatcher` consults
   `config.text_preprocessing` and does the tokenization **inside
   the batcher, on-the-fly, during training**.

**No data ever flows through the TextTokenizer node.** Its pins are
visual decoration. If you connected it to a hypothetical `WriteFile`
node, nothing would happen - the compiler would pull its config and
the "data stream" would be fiction. The same applies to
`TextVocabulary` and `TextPadding`.

This was the fastest path to shipping Phase 3 text training - it
avoided rewriting `TextDatasetBatcher` to consume pre-tokenized input
- but it violates the single-responsibility node principle and it
blocks any workflow that wants to treat tokenization as a
first-class step in a data pipeline.

### The target state (Fix B, confirmed 2026-04-16)

Make the text preprocessing nodes **real operations** that actually
transform their inputs:

```
DataInput(Text) -> TextTokenizer -> TextPadding -> DataSplit -> DataLoader -> Embedding -> ...
```

Where each node produces an Arrow-compatible output with the
post-transform representation. The schema would be something like:

- DataInput(Text) output: `{text: str, label: int}` per row
- TextTokenizer output: `{text: str, ids: list<int>, label: int}` per row
- TextPadding output: `{ids: int[max_length], label: int}` per row
- DataSplit output: same schema, partitioned into train/val/test
- DataLoader yields: `{ids: int[batch, max_length], labels: int[batch]}`

Critically, this also unlocks the user's use case:

```
DataInput(Text corpus) -> TextTokenizer -> WriteFile(tokenized.parquet)
```

And in a separate training session:

```
DataInput(tokenized.parquet) -> DataSplit -> DataLoader -> Embedding -> ...
```

Tokenize once, reuse the cached file across every model architecture
experiment you run on that corpus. Matches the standard ML
engineering workflow (sklearn `Pipeline.fit_transform()` + cache,
Spark `DataFrame.cache()`, HuggingFace `datasets.Dataset.map()` +
memory-mapped arrow).

### What has to change

**Backend / core:**
- `TextDatasetBatcher` rewrites from "reads raw text + tokenizes
  on-the-fly" to "reads pre-tokenized Arrow table + batches IDs".
- New row format for tokenized data that can round-trip to/from
  disk (likely Arrow `list<int32>` column for token IDs).
- `TextTokenizer` node gets a real `Execute(ArrowTable in) -> ArrowTable out`
  implementation, registered via the `node_executors/` framework
  that someone already scaffolded (see the other tofix entry
  "Tool-to-Node Migration").
- Same for `TextVocabulary` (fit-and-apply) and `TextPadding`
  (stateless pad/truncate).

**Graph compiler:**
- Delete the `ExtractTextTokenizer` / `ExtractTextVocabulary` /
  `ExtractTextPadding` extractors from
  `graph_compiler.cpp:1065-1121`.
- Delete the `text_preprocessing` config shortcut from
  `TrainingConfiguration` (it's no longer needed - the tokenized
  IDs are already in the Arrow table flowing into DataLoader).
- Text training graphs become indistinguishable from tabular
  training graphs: the model sees `int[batch, seq_len]` token ID
  tensors regardless of whether they were tokenized in the same
  session or pre-tokenized and loaded from disk.

**New nodes needed:**
- `WriteFile` / `ExportParquet` (or extend existing `ExportParquet`)
  to be usable after a tokenizer in a data-processing graph. The
  `ExportParquet` NodeType already exists at `node_editor.h:342` -
  may already do what we need, needs verification.
- `DataInput` needs to accept a pre-tokenized file as a new
  `FileCategory` (maybe `Text (pre-tokenized)` or just auto-detect
  based on schema).

**Migration & regression:**
- v1 / v2 / LSTM example graphs stay valid because they use the
  "tokenize inline" pattern. Under the new architecture, their
  TextTokenizer/Vocabulary/Padding nodes do real work at graph
  execute time, before the DataLoader - the graph is unchanged,
  the execution semantics change. Both v1 and v2 should still
  train to identical metrics as a regression check.

### Scope estimate

- 1-2 sessions for the backend rewrite (TextDatasetBatcher, node
  executor framework, Arrow row format)
- 1 session for graph compiler cleanup (delete extractors,
  TextTokenizer becomes a real node)
- 1 session for the new "read pre-tokenized file" DataInput
  variant
- Regression: v1 and v2 re-run, numbers match

Total: ~3-4 sessions. Depends on `node_executors/` framework
maturation (shares scope with the tool-to-node migration - both
want the same plumbing).

### Why this is the right fix even though it's a rewrite

- Matches what every other ML framework does (HuggingFace
  `datasets`, sklearn `Pipeline`, Spark ML, Beam, TFX)
- Makes the node graph honest: input pins / output pins carry
  actual data, not config metadata
- Unlocks the "preprocess once, train many" workflow for all
  text data, not just tokenization (normalization, feature
  engineering, augmentation offline, etc.)
- Unblocks Phase 4 Time-series: the same pattern applies to
  `TimeSeriesWindow` (sliding window as a real operation) and
  `TimeSeriesFeatures` (lag/rolling features as real transforms)
- Removes the "extractor" shortcut that's structurally identical
  to the dead Tool-to-Node NodeTypes - the extractor pattern was
  a workaround for not having a `node_executors` framework; now
  that scaffolding exists, we can do this right.
