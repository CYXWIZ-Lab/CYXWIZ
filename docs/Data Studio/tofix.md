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

**Severity:** High — LSTMLayer is currently a frozen random projection
during training. Weights never update. Every model using LSTM will
train the *downstream* layers only (Dense heads can still learn from
the frozen features, so training doesn't crash — val_acc will climb
but to a much lower ceiling than a working LSTM should reach).

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
