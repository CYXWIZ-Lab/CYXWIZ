# To Fix — Known Issues & Deferred Work

Tracked issues and design inconsistencies found during Phase 0–1
development. Each entry has a severity, root cause, and suggested fix.

## Backend Issues

### FlattenModule batch-dimension bug (BLOCKS image training)

**Severity:** Critical — image training crashes at Forward pass.

**Root cause:** `FlattenLayer::Forward` in
`cyxwiz-backend/src/algorithms/layer.cpp:966-986` uses
`(x.numdims() > 3) ? x.dims(3) : 1` to detect batch size. This only
works for 4D input where batch is the 4th ArrayFire dimension. For 2D
input `[features, batch]` (AF column-major), `numdims()` returns 2,
so `batch_size = 1`, and the entire tensor collapses into a single
vector `[features*batch, 1]`.

**Current workaround:** `ImageDatasetBatcher` outputs pre-flattened
`[batch, H*W*C]` tensors and `TrainingExecutor` skips FlattenModule
in Image mode. Training reaches the Forward pass but crashes inside
the matmul (confirmed via log checkpoint X3 → no X4).

**Suggested fix:** Make FlattenLayer detect batch from the last AF
dimension regardless of ndims:
```cpp
// Replace:
dim_t batch_size = (x.numdims() > 3) ? x.dims(3) : 1;
// With:
dim_t batch_size = x.dims(x.numdims() - 1);
dim_t flat_size = x.elements() / batch_size;
```
Or make it a no-op for 2D input (already flat). Then remove the Image
mode skip in TrainingExecutor and the pre-flatten in the batcher.

**Files:** `cyxwiz-backend/src/algorithms/layer.cpp:966-986`,
`cyxwiz-engine/src/core/training_executor.cpp` (Flatten skip),
`cyxwiz-engine/src/core/image_dataset_batcher.cpp` (pre-flatten).

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

### Label column check is not domain-aware

**Severity:** Medium — false warning for image datasets.

**Issue:** The compile gate warns "No label column selected" for ALL
datasets, including image datasets where labels come from folder
structure (ClassSubdirs) or CSV mapping (FlatWithCSV), not from a
column in the data table. The warning is misleading for image users.

**Suggested fix:** Make the label check domain-aware:
- `PreprocessingDomain::Tabular` → warn if no label column
- `PreprocessingDomain::Image` → skip the check (labels are
  implicit in the dataset layout)

**File:** `cyxwiz-engine/src/core/graph_compiler.cpp`, around the
label_column check in the Compile function.

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
