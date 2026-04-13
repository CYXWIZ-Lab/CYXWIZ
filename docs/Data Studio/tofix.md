# To Fix — Known Issues & Deferred Work

Tracked issues and design inconsistencies found during Phase 0–1
development. Each entry has a severity, root cause, and suggested fix.

## Backend Issues

### ~~Forward pass crash for image training~~ RESOLVED (22902ef9)

**Severity:** ~~Critical~~ Fixed — image training now works end-to-end.

**Status:** FlattenLayer batch-dimension bug was fixed (d6e203fa) —
batch is now correctly read from `x.dims(0)` since `TensorToAf` maps
our row-major `[batch, ...]` shape directly to AF dims. Workarounds
(pre-flatten in batcher, Flatten skip in executor) were reverted.

**Remaining issue:** Training still crashes at the Forward pass even
after the Flatten fix. Tested with both 224x224 (OOM candidate) and
34x34 (14 MB estimated, should fit). The crash happens inside the
`Forward(batch.data)` call after the Flatten produces its output.

**Root cause (needs investigation):** Likely a tensor layout mismatch
between `FlattenLayer::Forward` output and `LinearLayer::Forward`
input. Questions to answer:
1. What layout does `LinearLayer::Forward` expect? `[batch, features]`
   or `[features, batch]`? Check the matmul in LinearLayer.
2. What does `FlattenLayer::Forward` actually produce? After the fix,
   output is `af::moddims(x, af::dim4(batch, flat_features))` =
   AF dims `[batch, flat_features]`.
3. Does the matmul convention match? If LinearLayer does
   `af::matmul(weights_.T, x)` it expects `[features, batch]`. If
   `af::matmul(x, weights_)` it expects `[batch, features]`.
4. The tabular MNIST path works with 2D `[batch, 784]` tensors that
   never go through FlattenLayer. So the issue is specific to the
   Flatten→Linear transition with 4D→2D reshaping.

**Debug plan:** Add spdlog::info after FlattenLayer::Forward logging
the output AF dims. Add spdlog::info at LinearLayer::Forward entry
logging input AF dims + weight dims. Compare with the working MNIST
tabular path to find the layout mismatch.

**Files:** `cyxwiz-backend/src/algorithms/layer.cpp` (FlattenLayer
+ LinearLayer Forward methods).

---

## Node Pin Design Inconsistencies

### DataInput node does too much

**Severity:** Medium — causes confusion and compile warnings.

**Issue:** The DataInput node currently:
1. Has two output pins (Data + Label) — implies it separates features
   from labels internally
2. Carries label column selection in its dialog — label awareness
   should be a separate concern
3. Still has Normalize params in its parameters (from the legacy
   dialog) — normalization should be the Normalize node's job
4. The "No label column selected" compile warning fires because the
   compile gate reads `label_column` from the DataInput node, but
   for image datasets labels come from the folder structure, not a
   column

**Suggested fix (pin standardization pass):**
- DataInput outputs a single "Dataset" pin (stream of sample+label
  pairs). No separate Data/Label pins.
- A dedicated "LabelSelect" or "ColumnSelect" node handles label
  column choice for tabular data
- Remove all Normalize params from DataInput's parameter list
- Compile gate's label check should be domain-aware: tabular needs
  an explicit label column, image gets labels from folder structure
  or CSV automatically

**Scope:** This is a "Phase 1.5: Pin layout standardization" task.
Touches DataInput node creation, dialog, compile gate, Properties
panel, and potentially StartTrainingFromGraph dispatch.

### DataLoader pin layout

**Severity:** Low — functional but inconsistent.

**Issue:** DataLoader has one input and two outputs (Data + Labels).
But if DataInput already separates them, and DataLoader also
separates them, the semantics are duplicated. If we switch to a
single Dataset stream, DataLoader takes one stream in and outputs
one batched stream out.

### Normalize node in DataInput vs graph

**Severity:** Medium — user confusion.

**Issue:** DataInput's dialog has Normalize params (mean/std) AND the
user can add a separate Normalize node on the canvas. If both are
set, double-normalization occurs silently. Per the single-
responsibility principle, normalization belongs exclusively on the
Normalize graph node, never on the DataInput dialog.

**Fix:** Remove Normalize params from DataInput's Apply and dialog UI.
Already partially done for Image category (Phase 0 removed
target_width/height/normalize/rgb from image dialog for Phase 1).
Need to also remove for Tabular category's dialog.

---

## Memory Tab UX

### Image dataset shows 0B in Memory tab

**Severity:** Low — cosmetic.

**Issue:** After loading an image dataset, the Memory tab's "Current
Status" shows "In Memory — 0 B" because `loaded_memory_bytes_ = 0`
(images are lazy-loaded via LRU cache, no upfront RAM).

**Suggested fix:** Show estimated full-load size instead:
`num_images × target_w × target_h × channels × sizeof(float)`
with a "(estimated, lazy-loaded)" suffix so users have context.

**File:** `cyxwiz-engine/src/gui/data_input_dialog.cpp` Apply folder
branch, around `loaded_memory_bytes_ = 0`.

---

## Compile Gate Improvements

### Memory estimation is approximate

**Severity:** Low — the 4x multiplier is a rough heuristic.

**Issue:** GPU memory estimation uses `params × 4 × sizeof(float)`
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
warning — their labels come from folder structure or CSV mapping.
See `graph_compiler.cpp` Check 3.

---

## Generic Parameter Editor

### Properties panel parameter editing is basic

**Severity:** Low — works but not polished.

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

### ProcessCompletedCallbacks never called

**Severity:** Low — currently unused, no impact.

**Issue:** `AsyncTaskManager::ProcessCompletedCallbacks()` is declared
and implemented but never wired into the main render loop. Completion
callbacks queued by RunAsync never fire. The DataInput dialog works
around this by polling task state directly via `PollAsyncLoadResult`.

**Suggested fix:** Add `AsyncTaskManager::Instance().ProcessCompletedCallbacks()`
to `MainWindow::Render()` or `Application::Update()`. One line, but
needs testing to ensure no callback races with ImGui state.

**File:** `cyxwiz-engine/src/core/async_task_manager.h:148`.

---

## Registry / Lifecycle

### Image datasets not cleaned up on node delete

**Severity:** Medium — orphan entries leak.

**Issue:** `NodeEditor::DeleteNode` and `ClearGraph` call
`UnregisterNodeDatasetIfOwned` which checks Arrow and Parquet maps
but NOT the new `image_dataset_entries_` map. Deleting a DataInput
node that loaded images leaves an orphan entry.

**Suggested fix:** Extend `UnregisterNodeDatasetIfOwned` in
`node_editor_nodes.cpp` to also call
`registry.UnregisterImageDataset(name)`.

**File:** `cyxwiz-engine/src/gui/node_editor_nodes.cpp`, the
`UnregisterNodeDatasetIfOwned` helper.

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

**Severity:** High — the canvas is currently a "lie." Users see a
visual graph of `DataInput → DataSplit → DataLoader → ... → Loss`
with separate label and tensor streams, but the executor doesn't
actually traverse pin connections. It reads `dataset_name` from the
DataInput node parameters, fetches the dataset from `DataRegistry`
by name, and runs training. The pin wires are decorative.

**Symptoms users hit:**
- Removing a wire between DataLoader and Loss has no effect — training
  still works because labels come from the registry, not the pin.
- DataSplit's Train/Val/Test outputs are visual fiction. The actual
  split happens inside the batcher based on the DataSplit node's
  ratio params, regardless of what the user wired downstream.
- A graph that is structurally wrong (e.g. labels never reach loss)
  trains anyway because the executor doesn't care about topology.

**Why this is hard (~3-5 days):**
1. `TrainingExecutor` currently takes a single `dataset` + `label_column`
   and runs a hardcoded data → preprocess → model → loss → optimizer
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
   and the model — currently DataLoader's batch_size is just a number
   the executor reads, not something the graph actually does.
6. Loss nodes need to consume both the model's prediction stream AND
   the upstream label stream — currently the executor hands labels in
   directly because there's no concept of "which pin gives me labels."
7. Backwards-compat: the layered model (Dense → ReLU → Dense → ...) is
   still a single chain; only the data/label plumbing around it changes.

**Suggested approach (when we tackle it):**
- Phase A: introduce a `RuntimeContext` with `std::map<int, Tensor>`
  keyed by pin_id. Compile builds a topological execution plan over
  the graph; each node's `Execute(RuntimeContext&)` reads input pins
  and writes output pins. Layer chain stays inside the model — only
  the data path becomes pin-driven.
- Phase B: DataSplit, DataLoader, and Loss nodes implement the new
  runtime interface. Existing layer nodes keep their current path
  (compiled into a `Model` object, still called via `model.Forward`).
- Phase C: remove `dataset_name` lookup from `StartTrainingFromGraph`
  — the dataset stream comes from the `DataInput` node executing into
  its output pins.

**Files involved:**
- `cyxwiz-engine/src/core/training_executor.h/.cpp`
- `cyxwiz-engine/src/core/training_manager.cpp`
- `cyxwiz-engine/src/gui/main_window.cpp` (StartTrainingFromGraph)
- `cyxwiz-engine/src/core/graph_compiler.cpp` (build execution plan)
- New file: `cyxwiz-engine/src/core/runtime_context.h/.cpp`

**Workaround until then:** the pin layout fix (DataLoader 2-in/2-out,
Targets pin → Labels type) makes the canvas at least *look* honest
even though the executor still bypasses it. Users can wire the graph
correctly and see green compile pins; the runtime just ignores the
wiring details. This is the lesser of two lies.

**DataLoader params that are currently UI-only (need executor wiring):**
The Properties panel exposes a full set of training-loop knobs on
the DataLoader node, but only `epochs`, `batch_size`, `shuffle`, and
`drop_last` are actually honored at runtime today. The rest are
stored on the node, saved to the project file, and read by the
graph compiler into TrainingConfiguration, but the executor never
acts on them. When option (b) lands, wire these through:

- `grad_accum_steps` (default 1) — accumulate gradients over N
  forward passes before stepping the optimizer. Effective batch
  size = `batch_size × grad_accum_steps`. Lets users train with
  large effective batches on a small GPU. Touches the training
  inner loop in `training_executor.cpp` and the optimizer step.
- `seed` (default 42) — RNG seed for shuffle order. Currently the
  batcher seeds itself; should accept this from config so two runs
  with the same seed produce the same epoch order.
- `num_workers` (default 4) — already has a "not yet implemented"
  warning. Needs a worker pool in the batcher for parallel sample
  loading. Critical for image/audio pipelines where decode is the
  bottleneck.
- `prefetch_factor` (default 2) — batches to prefetch per worker.
  Pairs with num_workers; meaningless without it.
- `pin_memory` (default false) — allocate batches in pinned host
  memory for faster H2D copy. CUDA-only optimization, ignored on
  OpenCL/CPU. Touches the batcher's tensor allocation path.
- `log_interval` (default 10) — currently the executor logs every
  batch. Should respect this and only log every N batches.
- `validation_freq` (default 1) — currently validation runs every
  epoch. Should run only every N epochs.

None of these are blocking — the existing pipeline trains correctly
without them. They're quality-of-life knobs that match what users
expect from PyTorch / Keras DataLoader. Adding them to the executor
is straightforward once option (b) reorganizes the data flow.

---

## Future Architecture

### Variable-shape Sample type (v3 consideration)

The current pipeline pre-flattens images in the batcher, which means
the Flatten graph node is a conceptual marker, not a real transform.
A proper solution would introduce a `Sample` type with dynamic shape
that flows through the graph, with each node transforming the shape.
`DataLoader` would enforce uniform shape at batch time and error if
shapes don't match.

This is Option B from the design doc — deferred to v3 because it
requires a ground-up rewrite of the tensor plumbing.

### Explicit Decode node (v3 consideration)

Option C from the design doc: `DataInput` produces file references,
a `Decode` node reads the bytes and produces the raw tensor. Maximum
composability but adds a required node everyone will forget. Also v3.
