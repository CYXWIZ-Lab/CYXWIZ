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

## Data Quality / Sanity Audits

### Dataset sanity audit at Apply time (cross-domain)

**Severity:** High — silently corrupt datasets produce fake training
metrics and waste GPU hours before the user notices.

**Origin:** Caught during Phase 2 audio testing on the Kaggle
"Binary Drone Audio" dataset. 1089 / 11704 files (9.3%) were genuinely
all-zero silent WAVs with valid headers — `wave.open()` reports 16347
frames at 16kHz but every sample is literally 0. The mel spectrogram of
silence is constant `log(1e-10) = -23.0259`, which looks identical to a
backend bug. Paired with a separate data-leakage bug in the val pipeline,
training reported a fake 100% val accuracy and nobody would have noticed
the silent files without end-to-end testing on a real Kaggle dataset.

The audio domain now has a fix (`AudioDataset::ExtractFeatures` detects
all-zero samples and returns invalid, with rate-limited warnings in
`GetItem`), but the **broader lesson is that every domain needs a
quality audit that runs at Apply time**. Unit tests cannot catch this
class of issue — only integration tests on real datasets can.

**Concrete sanity checks to add:**

- **Audio** (partially done)
  - ✓ Silent file detection (all samples == 0)
  - TODO: RMS energy threshold (files below -60 dBFS peak are likely corrupt)
  - TODO: Report: "N/M files have silence, M/N ratio per class"
  - TODO: If silent ratio > 20% in any class, refuse to Apply with an error
- **Image**
  - Decode a random sample of N files (N = min(100, total))
  - Count: 1×1 corrupt thumbnails, 0-byte placeholders, fully-black
    images (pixel sum == 0), fully-white images (pixel sum == 255*H*W*C)
  - Report: "N/M images appear blank or corrupt"
  - Refuse if > 20% bad in any class
- **Tabular**
  - Check label column: unique value count, class balance
  - Warn if label has only 1 unique value (no signal)
  - Check feature columns: 100% NaN, 100% constant, infinite values
  - Warn for each degenerate column, refuse if > 50% of columns degenerate
- **Text** (Phase 3)
  - Count empty strings, single-character strings, non-UTF-8 files
  - Vocabulary coverage check

**Where it goes:** A new `DatasetAudit` helper in
`cyxwiz-engine/src/core/dataset_audit.h/.cpp` with domain-specific
static methods. Called from `DataInputDialog::Apply` after registration
and before setting `data_loaded=true`. Results shown in a new "Audit"
tab on the dialog with pass/warn/fail indicators and a list of
problematic files.

**Estimated scope:** ~1-2 days per domain. Audio is partially done
(silent detection). Image is the next priority (same class of issue
affects cats/dogs and similar).

### Rate-limited sample loading warnings in AudioDataset

**Status:** DONE — `AudioDataset::GetItem` uses an atomic counter and
prints the first 10 bad files at full detail, then every 100th. Full
warning-per-sample spam would flood the log on datasets with many
silent files (1089 in the drone set we tested).

---

## Training UX

### Sub-epoch validation (fractional validation_freq)

**Severity:** Enhancement — smooths the val curve for long-epoch domains.

**Issue:** Validation currently runs once per epoch — standard ML practice
and correct for fast-epoch domains (tabular MNIST finishes an epoch in
seconds). But for audio/image pipelines where an epoch can take 5-10
minutes, you stare at an empty val line for the full duration of epoch 1
and assume it's broken. Users who stop training before the first epoch
completes never see any val feedback at all.

**Current workaround:** Shrink max_duration to match the actual clip
length (drone clips are ~1s; setting max_duration=1 instead of 5 cuts
epoch time by ~5x because feature cols shrink from ~313 to ~63 and the
Linear(input→512) first layer shrinks proportionally).

**Suggested fix:** Reinterpret `validation_freq` on the DataLoader node
as follows:
- integer N ≥ 1 → run validation every N epochs (current behavior)
- 0 < N < 1 → run validation every N × total_batches batches, on a
  *random subsample* of the val split sized for ~2-3 seconds of
  validation time
- Default stays 1 (every epoch)

This gives users a continuous val curve during long epochs. A value
like `0.25` means "validate 4 times per epoch" — at batches 25%, 50%,
75%, 100%. The subsample size should be `min(val_set_size, 256)` so
validation cost stays bounded regardless of how big the val split is.

**Files:** `cyxwiz-engine/src/core/graph_compiler.cpp` (parse float,
store on TrainingConfiguration), `cyxwiz-engine/src/core/training_executor.cpp`
(RunTrainingEpochArrow — insert validation calls at the configured
fractional boundaries), `cyxwiz-engine/src/gui/properties.cpp`
(DataLoader properties section — change validation_freq widget to
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

This is Option B from the design doc — deferred to v3 because it
requires a ground-up rewrite of the tensor plumbing.

### Explicit Decode node (v3 consideration)

Option C from the design doc: `DataInput` produces file references,
a `Decode` node reads the bytes and produces the raw tensor. Maximum
composability but adds a required node everyone will forget. Also v3.

---

## Phase 3 Text — Deferred Items (2026-04-14)

Phase 3 text training is **functionally complete and live-verified**
on sentiment_mental_health.csv (52681 samples, 7 classes). Six commits
landed this session — async Apply, text registry, JSON loader
TextTokenizer/TextVocabulary/TextPadding registration, compile gate
registry-first probe, v2 regularized example graph, Embedding AF
backward fix. What's still deferred:

### Phase 3 engine-side wiring bundle not yet committed

**Severity:** High — the entire Phase 3 text pipeline lives in the
working tree but isn't in git. A stray `git stash` or `git checkout`
loses ~14 files of working code.

**Files still uncommitted at end of 2026-04-14 session:**
- `cyxwiz-engine/src/core/training_executor.cpp` — text dispatch +
  the `if (epoch == 1 && batch_num == 1)` DEBUG Arrow / DEBUG
  CrossEntropy dumps (left over from embedding debugging, should
  be gated by a config flag)
- `cyxwiz-engine/src/core/training_manager.{cpp,h}` — `StartTrainingText`
- `cyxwiz-engine/src/core/graph_compiler.{cpp,h}` — TextPreprocessingConfig
  + 3 extractors (ExtractTextTokenizer / ExtractTextVocabulary /
  ExtractTextPadding) + PreprocessingDomain::Text. The compile-gate
  registry-first fix IS committed (603eb8e4) but the extractors are
  not — they live in the same file and were sheared off via surgical
  commit.
- `cyxwiz-engine/src/core/formats/text_dataset.{cpp,h}`
- `cyxwiz-engine/src/core/text_dataset_batcher.{cpp,h}` (untracked —
  new files)
- `cyxwiz-engine/src/core/node_executors/` (untracked — unfinished
  KMeans executor scaffolding, unrelated to Phase 3)
- `cyxwiz-engine/src/gui/main_window.cpp` — IsTextDataset dispatch
- `cyxwiz-engine/src/gui/node_editor_nodes.cpp`
- `cyxwiz-engine/src/gui/data_input_dialog.h`
- `cyxwiz-engine/src/gui/panels/toolbar_profile_menu.cpp` (untracked)
- `cyxwiz-engine/src/gui/IconsTabler.h` (untracked)
- `cyxwiz-engine/resources/fonts/tabler-icons.ttf` (untracked)
- `cyxwiz-engine/resources/node_icons/` (untracked)
- `cyxwiz-engine/CMakeLists.txt` — text_dataset_batcher.{cpp,h} sources

**Plan:** Split into 3-4 focused commits (next session):
1. `feat: Phase 3 text graph compiler extractors + training dispatch`
2. `feat: Phase 3 text dataset batcher + format loader`
3. `feat: Toolbar profile menu + Tabler icons` (unrelated UI polish)
4. `feat: KMeans node executor scaffolding` (separate pattern, unfinished)

### num_workers=4 not implemented — DataLoader runs single-threaded

**Severity:** Medium — slows training on multi-core hosts.

**Issue:** Every compile logs
`GraphCompiler: num_workers=4 requested but not yet implemented -
batching runs single-threaded`. The node param is plumbed through
to `TrainingConfiguration` but no batcher honors it.

**Fix:** Implement a pool in the `IBatcher` base class or in
`TextDatasetBatcher` / `ArrowDatasetBatcher` to prefetch N batches
in parallel. Non-trivial because batcher state (shuffle, epoch
progress) is currently single-threaded.

**Files:** `cyxwiz-engine/src/core/text_dataset_batcher.cpp` (when
committed), `cyxwiz-engine/src/core/graph_compiler.cpp:238` (warning
emit site).

### Training logs silent mid-epoch — no per-batch feedback

**Severity:** Medium — UX gotcha, caused me to misdiagnose "hung"
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
— 50 batches is long enough that a fast epoch won't spam.

### Only MLP head tested with text — LSTM / GRU / Transformer paths unverified

**Severity:** Medium — Phase 3 is "done" for MLP over flattened
embeddings only. The backend has LSTM/GRU/Transformer/Attention
layers but they haven't been run on a text training graph end-to-end.

**Risk:** The Embedding → Flatten → Dense head only exercises a
small corner of the NLP-capable codepaths. LSTM over `[batch,
seq_len, embed_dim]` without Flatten is the obvious next test;
that'll expose any shape-handling bugs in the recurrent layers
(same category as the Embedding AF backward bug we just fixed).

**Fix:** Create a Phase 3.x example graph
(`examples/cyxgraph/mental_health_sentiment_lstm.cyxgraph`) that
uses Embedding → LSTM(hidden=128) → Dense(7), train it, fix whatever
breaks.

### Text preview doesn't show class distribution

**Severity:** Low — nice-to-have.

**Issue:** `RenderTextPreview` (added in 7b7bd34b) shows a CSV head
table with mapped text/label columns highlighted green or red. It
does NOT show a class-distribution bar chart or sample-per-class
count — users can't see class imbalance until training starts.

**Fix:** Parse the label column during `LoadColumnList` and
compute a per-value count, then render a small horizontal bar
chart above the CSV head table.

**Files:** `cyxwiz-engine/src/gui/data_input_dialog.cpp` —
`LoadColumnList()` + `RenderTextPreview()`.

### `node_config_dialog.h` git-binary state pre-HEAD~3

**Severity:** Low — historical, mostly a diff-tool annoyance.

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

**Severity:** Low — organizational.

**Issue:** v2 regularized variant was added as a sibling file rather
than in a `v2/` subdirectory. As more variants land (LSTM, Transformer,
etc.) the flat layout will get noisy.

**Fix:** Move to `examples/cyxgraph/text/` subdirectory (which already
exists as untracked directory from the prior session). Already
planned for the Phase 3 engine-side bundle commit.

---

## LSTM Layer — Broken AF Forward + Missing CPU Backward (2026-04-15)

**Status update 2026-04-16:** CPU Forward now populates row-major CPU
caches and CPU Backward implements BPTT reading from those caches.
LSTM weights now update during training on the CPU path. ArrayFire
Forward + Backward remain gated off (`kAfPathEnabled = false`) —
the column-major dim-ordering bug that prevents AF Forward from
handling `[batch, seq, features]` input is still open and is the
remaining work on this entry. CPU path is the correctness oracle for
any future AF fix. See commit with CPU BPTT for implementation
details.

**AF Forward — DONE (2026-04-16):**
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
  returns 1 (product of empty range) but `Data()` is null —
  the previous `NumElements() == 0` check passed through and
  `TensorToAf` then tripped on null data. Caught via diagnostic
  log of tensor shape on null-data throw.
- AF Forward numerical output validated against CPU Forward —
  loss numbers match within fp32 noise (~10ppm) on the mini
  sentiment LSTM smoke test. Same monotonic loss-down /
  acc-up curve.

**AF Backward — still pending (perf only):**
- ~130-line AF backward under `#if 0` in `layer.cpp` is the next
  perf upgrade. Currently unused — Backward goes through CPU
  BPTT, which works correctly but pays a Tensor↔CPU round-trip
  that AF Backward would skip. AF Forward now serves as the
  oracle for AF Backward validation.
- Needs the same 3D-helper / slice-moddims treatment on its cache
  reads + dx output. Cache consistency is already solved: AF
  Forward writes row-major caches via `AfToTensor3DRowMajor`,
  AF Backward will need to read them back via
  `TensorToAf3DRowMajor`.
- A `last_forward_was_af_` flag on LSTMLayer would let Backward
  dispatch between CPU BPTT and AF backward. Not strictly needed
  if AF Backward also reads row-major caches.

Net win: LSTM training is correct AND uses GPU on the Forward
pass. CPU Backward remains the gradient computation. End-to-end
throughput should beat pure CPU on hidden_size >= 64 graphs
where the Forward matmul dominates (the sentiment_lstm
hidden=128 graph is the obvious follow-up benchmark).

**Severity:** ~~High~~ Medium (after CPU BPTT landed) — LSTM weights
update correctly on CPU. AF path is a perf-only follow-up. Original
severity kept below for historical context.

**Original severity (before 2026-04-16 fix):** High — LSTMLayer was
a frozen random projection during training. Weights never updated.
Every model using LSTM trained only the *downstream* layers.

**Discovered:** 2026-04-15 LSTM smoke test
(`examples/cyxgraph/text/test_02_sentiment_lstm.cyxgraph`) — the goal
was to verify the Phase 3 text pipeline works with an LSTM head in
place of the flat Dense head. Plumbing worked end-to-end
(`LSTMModule`, `training_executor` LSTM case, `Embedding → LSTM`
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
- `x.dims(1) = 128` is read as `seq_len` — correct by coincidence
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
On exit, only the final `output` tensor is returned — the per-layer
per-timestep state needed for BPTT is thrown away.

**Fix:** add an `std::vector<float>` scratch buffer during the CPU
forward pass that collects `gates` / `h_states` / `c_states` per
layer, then wrap them as Tensors at the end and push into the cache
vectors. Use row-major `[seq_len, batch, hidden]` layout to match
what the AF backward code reads — OR rewrite the backward in CPU
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
at `layer.cpp:2394-2402` so at least the log isn't spammed — but the
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
   from whatever caches Forward is willing to populate — row-major
   CPU layout, not AF-format.
2. **Populate caches in CPU Forward.** Gates, hidden states, cell
   states per layer per timestep.
3. **Delete the one-shot warn stub** — once Backward has real CPU
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
  should still train with identical numbers — we haven't touched
  the Dense-head path.

### Files touched during the 2026-04-15 smoke test that need revisiting

- `cyxwiz-backend/include/cyxwiz/sequential.h` — `LSTMModule`
  declaration (keep)
- `cyxwiz-backend/src/algorithms/sequential.cpp` — `LSTMModule::Forward`
  last-step slice + `Backward` re-expand (keep; last-step mode works
  correctly once the underlying `LSTMLayer::Backward` produces real
  gradients)
- `cyxwiz-backend/src/algorithms/layer.cpp` — `LSTMLayer::Forward`
  AF gate + `Backward` stub (remove/rewrite during the fix)
- `cyxwiz-engine/src/core/training_executor.cpp` — Embedding `→`
  recurrent lookahead + LSTM case (keep)
- `cyxwiz-engine/src/core/graph_compiler.cpp` — `IsModelLayer`
  recurrent entries (keep)
- `examples/cyxgraph/text/test_02_sentiment_lstm.cyxgraph` — the
  smoke test graph that surfaced all this (keep as a regression
  fixture)

---

## Tool-to-Node Migration — ~40 NodeTypes with standalone panels but no graph integration

**Severity:** HIGH (architectural) — this is the biggest consistency gap
in the codebase. ~80 floating analytical panels live under
`cyxwiz-engine/src/gui/panels/*.h` and most have a corresponding
NodeType in `node_editor.h`, but almost none of them are wired to
the execution pipeline. The v1 "everything is a node" philosophy is
only half-honored.

**Status update 2026-04-16 — Phase 4 time-series block CLOSED.** The
`TimeSeriesWindow`, `TimeSeriesFeatures`, and `TimeSeriesSplit`
NodeTypes listed below as "dead" are now live Cat-1 operators with
working executors (commits `aacbadcd`, `272dca2e`). The Phase 4
time-series row in the inventory table can be struck through. Two
NEW NodeTypes were added as part of this work (`LogTransform`,
`Differencing`) — both also live operators. See
`docs/phase4_time_series_plan.md` "What actually shipped" for the
full breakdown. ~37 dead NodeTypes remain; migration framework
(`node_executors/`, `PipelineMaterializer`, `IPipelineOperator`)
is now proven and ready for text analytics / linear algebra / etc.

**Discovered:** 2026-04-15, when the user asked why the graph has both
an `Embedding` node and a separate standalone `Word Embeddings`
floating panel. The answer — that the `WordEmbeddings` NodeType
exists in the catalogue but has zero backing — generalizes to most
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
all registered ops automatically — no per-op integration needed in
the dispatcher.

### What's already partially started

`cyxwiz-engine/src/core/node_executors/` (originally untracked in git,
now committed as of `52808745`)
contains scaffolding for a per-node execution framework separate from
`training_executor`:

- `node_executor.h` — `ExecutorState` (Idle/Configuring/Executing/
  Completed/Error/Cancelled) + `CodeFramework` (Sklearn, PyCyxWiz,
  PyTorch) + base executor interface.
- `node_executor_factory.h` — factory to instantiate executors by
  NodeType.
- `kmeans_executor.{cpp,h}` — first concrete executor (KMeans
  clustering), mirroring `KMeansPanel`'s functionality but as a
  node-graph operation.

**This is exactly the migration framework that was needed.** It was
started and abandoned. The pattern is the right one — most tool
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
| `WordEmbeddings` | `embeddings_panel.h` | dead — see separate entry in this doc |
| `TFIDFVectorizer` | `tfidf_panel.h` | dead |
| `CountVectorizer` | *(none found)* | dead |
| `SentimentAnalyzer` | `sentiment_panel.h` | dead |
| `NamedEntityRecognizer` | *(none found)* | dead |
| `WordFrequencyNode` | `word_frequency_panel.h` | dead |
| (tokenization — see note) | `tokenization_panel.h` | partially covered by `TextTokenizer` Phase 3 node |

**Machine learning algorithms (Phase 4 block):**
| NodeType | Panel file | Status |
|---|---|---|
| `KMeansCluster` | `kmeans_panel.h` | **scaffolded** — `kmeans_executor.{cpp,h}` exists untracked |
| `DBSCANCluster` | `dbscan_panel.h` | dead |
| `HierarchicalCluster` | `hierarchical_panel.h` | dead |
| `GMMCluster` | `gmm_panel.h` | dead |
| `PCANode` / `TSNENode` / `UMAPNode` | `dim_reduction_panel.h` | dead |
| `DecisionTreeClassifier` | *(shares `regression_panel.h`?)* | dead |
| `RandomForestClassifier` | *(none found)* | dead |
| `GradientBoostingClassifier` | *(none found)* | dead |
| `SVMClassifier` / `SVMRegressor` | *(none found)* | dead |
| `KNNClassifier` | *(none found)* | dead |
| `NaiveBayesClassifier` | *(none found)* | dead |
| `LogisticRegressionNode` / `LinearRegressionNode` / `PolynomialRegressionNode` | `regression_panel.h` | dead |

**Model evaluation (Phase 4 block):**
| NodeType | Panel file | Status |
|---|---|---|
| `ConfusionMatrixNode` | `confusion_matrix_panel.h` | dead |
| `ROCCurveNode` | `roc_auc_panel.h` | dead |
| `PRCurveNode` | `pr_curve_panel.h` | dead |
| `LearningCurvesNode` | `learning_curves_panel.h` | dead |
| `FeatureImportanceNode` | `feature_importance_panel.h` | dead |
| `CrossValidationNode` | `cross_validation_panel.h` | dead |
| `RegressionMetricsNode` | *(shares `regression_panel.h`?)* | dead |

**Linear algebra (Phase 5 block):**
| NodeType | Panel file | Status |
|---|---|---|
| `SVDNode` | `svd_panel.h` | dead |
| `QRDecomposition` | `qr_panel.h` | dead |
| `CholeskyDecomposition` | `cholesky_panel.h` | dead |
| `EigenDecomposition` | `eigen_decomp_panel.h` | dead |
| `MatrixCalculator` | `matrix_calculator_panel.h` | dead |

**Time series analysis (Phase 5 block):**
| NodeType | Panel file | Status |
|---|---|---|
| `TimeSeriesDecomposition` | `decomposition_panel.h` | dead |
| `ACFNode` / `PACFNode` | `acf_pacf_panel.h` | dead |
| `StationarityTest` | `stationarity_panel.h` | dead |
| `SeasonalityDetector` | `seasonality_panel.h` | dead |
| `ARIMAForecaster` / `ExponentialSmoothing` | `forecasting_panel.h` | dead |

**Signal processing (Phase 4 block):**
| NodeType | Panel file | Status |
|---|---|---|
| `FFTNode` / `IFFTNode` | `fft_panel.h` | dead |
| `FilterDesigner` | `filter_designer_panel.h` | dead |
| `Convolution1D` | `convolution_panel.h` | dead |
| `WaveletTransform` | `wavelet_panel.h` | dead |

**Statistics (Phase 5 block):**
| NodeType | Panel file | Status |
|---|---|---|
| `HypothesisTest` | `hypothesis_test_panel.h` | dead |
| `DistributionFitter` | `distribution_fitter_panel.h` | dead |

**Deep learning interpretation (Phase 5 block):**
| NodeType | Panel file | Status |
|---|---|---|
| `GradCAMNode` | `gradcam_panel.h` | dead |
| `SaliencyMapNode` | *(in `visualization_panel.h`?)* | dead |

**Optimization (Phase 5 block):**
| NodeType | Panel file | Status |
|---|---|---|
| `GradientDescentViz` | `gradient_descent_panel.h` | dead |
| `ConvexityAnalyzer` | `convexity_panel.h` | dead |
| `LPSolver` | `lp_panel.h` | dead |
| `QPSolver` | `qp_panel.h` | dead |
| `NumericalDifferentiation` | `differentiation_panel.h` | dead |
| `NumericalIntegration` | `integration_panel.h` | dead |

**Utility (Phase 4 block):**
| NodeType | Panel file | Status |
|---|---|---|
| `CalculatorNode` | `calculator_panel.h` | dead |
| `UnitConverter` | `unit_converter_panel.h` | dead |
| `RegexTester` | `regex_tester_panel.h` | dead |
| `JSONPathExtractor` | `json_viewer_panel.h` | dead |
| `DataProfiler` | `data_profiler_panel.h` | dead |

**Data preprocessing (Phase 4 block):**
| NodeType | Panel file | Status |
|---|---|---|
| `StandardScaler` | `standardization_panel.h` | dead |
| `MinMaxScaler` / `RobustScaler` | `feature_scaling_panel.h` | dead |
| `OutlierDetector` | `outlier_detection_panel.h` | dead |
| `LabelEncoder` / `OrdinalEncoder` / `TargetEncoder` | *(none found)* | dead |

### Why this matters

- **User confusion.** Drop a `WordEmbeddings` or `KMeansCluster` node
  on the canvas, nothing happens. Right-click doesn't open the
  relevant panel. The panel is hiding in the toolbar menu. Two
  unrelated surfaces for the same concept.
- **Lost v1 promise.** The original CyxWiz Engine v1 pitch was
  "everything is a node." The current hybrid state violates that —
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

Not a single commit — this is phase-scale work. The right order:

**Step 1 — ~~Finish the `node_executors` framework.~~ DONE (2026-04-16).**
Framework shipped in commit `52808745` with `IPipelineOperator` base
class, `PipelineBand` enum, `PipelineOperatorFactory`, and the
`PipelineMaterializer` dispatcher wired into training launch
(`8b6055b0`). Five working Cat-1 operators ride on it
(LogTransform, Differencing, TimeSeriesFeatures, TimeSeriesWindow
multivariate, TimeSeriesSplit). KMeans executor (Cat-2 introspection
variant) also committed as part of the framework landing. See
`docs/phase4_time_series_plan.md` "What actually shipped" for details.

**Step 2 — Establish the "rich dialog on double-click" convention
for tool nodes.** The existing `DataInputDialog` and `TokenizerDialog`
are the model: a node-config-dialog dispatch in `node_config_dialog`
that looks up the node's type and opens a custom dialog. Refactor
`EmbeddingsPanel` / `KMeansPanel` / `SVDPanel` etc. to live inside
a node-config dialog instead of as floating windows.

**Step 3 — Batch-migrate by category.** Start with one category end
to end before moving to the next, to avoid leaving everything half-
done. Suggested order:
1. Text analytics (user wants this, fewest executors, shares
   infrastructure with the Phase 3 text path)
2. Linear algebra (lowest complexity, pure math, easy validators)
3. Clustering ML (KMeans already scaffolded)
4. Signal processing (FFT / wavelet)
5. Statistics
6. Remaining ML algorithms
7. ~~Time series~~ DONE — Phase 4 shipped Window / Features / Split
   as real operators (2026-04-16).
8. Model evaluation + interpretation (runs AFTER training, different
   execution context)

**Step 4 — Delete the standalone panels.** Once a tool has a rich
node dialog, remove its toolbar-menu button and delete the `.h/.cpp`
for the standalone panel. This is the signal that consolidation
actually shipped.

**Step 5 — Document the node-first convention in CLAUDE.md.** Add
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

---

## TextTokenizer is a config extractor, not a pipeline operation

**Severity:** HIGH (architectural) — blocks the "preprocess once,
train many" workflow and makes the `TextTokenizer` / `TextVocabulary`
/ `TextPadding` nodes visually misleading (they have input/output
pins but no data flows through them).

**Discovered:** 2026-04-16 while discussing node architecture. The
user asked whether a user could build a graph like
`ReadCorpus → TextTokenizer → WriteFile(tokenized.jsonl)` to
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
node, nothing would happen — the compiler would pull its config and
the "data stream" would be fiction. The same applies to
`TextVocabulary` and `TextPadding`.

This was the fastest path to shipping Phase 3 text training — it
avoided rewriting `TextDatasetBatcher` to consume pre-tokenized input
— but it violates the single-responsibility node principle and it
blocks any workflow that wants to treat tokenization as a
first-class step in a data pipeline.

### The target state (Fix B, confirmed 2026-04-16)

Make the text preprocessing nodes **real operations** that actually
transform their inputs:

```
DataInput(Text) → TextTokenizer → TextPadding → DataSplit → DataLoader → Embedding → ...
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
DataInput(Text corpus) → TextTokenizer → WriteFile(tokenized.parquet)
```

And in a separate training session:

```
DataInput(tokenized.parquet) → DataSplit → DataLoader → Embedding → ...
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
  `TrainingConfiguration` (it's no longer needed — the tokenized
  IDs are already in the Arrow table flowing into DataLoader).
- Text training graphs become indistinguishable from tabular
  training graphs: the model sees `int[batch, seq_len]` token ID
  tensors regardless of whether they were tokenized in the same
  session or pre-tokenized and loaded from disk.

**New nodes needed:**
- `WriteFile` / `ExportParquet` (or extend existing `ExportParquet`)
  to be usable after a tokenizer in a data-processing graph. The
  `ExportParquet` NodeType already exists at `node_editor.h:342` —
  may already do what we need, needs verification.
- `DataInput` needs to accept a pre-tokenized file as a new
  `FileCategory` (maybe `Text (pre-tokenized)` or just auto-detect
  based on schema).

**Migration & regression:**
- v1 / v2 / LSTM example graphs stay valid because they use the
  "tokenize inline" pattern. Under the new architecture, their
  TextTokenizer/Vocabulary/Padding nodes do real work at graph
  execute time, before the DataLoader — the graph is unchanged,
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
maturation (shares scope with the tool-to-node migration — both
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
  to the dead Tool-to-Node NodeTypes — the extractor pattern was
  a workaround for not having a `node_executors` framework; now
  that scaffolding exists, we can do this right.
