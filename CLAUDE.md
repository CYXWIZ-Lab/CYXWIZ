# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

CyxWiz is a **decentralized ML compute platform**:

1. **CyxWiz Engine** - Desktop IDE with visual node editor (C++/ImGui)
2. **CyxWiz Server Node** - Distributed compute worker (C++/ArrayFire)
3. **CyxWiz Central Server** - Network orchestrator (Rust/gRPC)

**Tech Stack**: C++20, Rust, Python | ImGui, ImNodes, ImPlot | ArrayFire (CUDA/OpenCL) | gRPC | Solana

## Architecture

```
Engine (Desktop)                    Server Node (Compute)
  ├─ cyxwiz-backend.dll               ├─ cyxwiz-backend.dll
  ├─ cyxwiz-protocol (gRPC)           ├─ cyxwiz-protocol (gRPC)
  └─ P2P → Server Node                └─ Executes training jobs

Central Server (Rust) — Node registry, job scheduling, Solana payments
```

**Key Principles**: Backend is shared DLL | Protocol-first (.proto) | Cross-platform | Reservation-based payment (pay for TIME)

## Data Path Direction (v0.2.0 in-flight)

**Arrow is THE tabular data path. The legacy `DatasetHandle / DatasetBatcher /
StreamingDataset / LoadStreamingDataset / LRUCache` stack is being torn out —
do not propose reconnecting it.** When you see dead UI (Memory Policy radio,
LRU cache chunks, prefetch, Write-to-disc, etc.) those are remnants from the
pre-Arrow lazy path. Remove them when you find them, don't try to revive.

### Two backends, one entry point

`DataInputDialog::Apply` calls `DataRegistry::LoadTabularCSV` which auto-
detects which backend to use:

- **In-memory Arrow (default fast path)** — file_size × 0.75 < available RAM.
  `LoadCSVToArrow` reads the whole CSV as one chunk into an `ArrowDataset`,
  then `CompactIntegerColumns` auto-downcasts int64 → uint8/16 where values
  fit (typical 8x memory savings on pixel data). Used for the vast majority
  of training datasets.
- **Disk-backed Parquet (larger-than-RAM)** — file too big for the 75% RAM
  threshold, OR user ticked Force disk-backed in the Memory tab Advanced
  section. `ConvertCsvToParquet` streams the CSV through to a Snappy
  Parquet cache in `<temp>/cyxwiz/cache/`, then `ParquetBackedDataset::Open`
  memory-maps the file. Training reads pages lazily via `ParquetArrowBatcher`.
  RAM use is bounded by the OS page cache, not the file size.

Both backends register under the same name in `DataRegistry`. The dispatcher
in `MainWindow::StartTrainingFromGraph` checks `IsArrowDataset` then
`IsParquetBackedDataset` and routes to the appropriate `TrainingExecutor`
constructor (`StartTrainingArrow` or `StartTrainingParquet`). Same model,
same loss, same optimizer — only the batcher differs.

### Async load + UX

- CSV loads (sync `LoadCSVToArrow` AND the slow `ConvertCsvToParquet` step)
  run on an `AsyncTaskManager` worker thread launched from
  `DataInputDialog::Apply`. The dialog stays responsive: scroll preview,
  switch tabs, even close it. OK and Apply buttons grey out via
  `NodeConfigDialog::IsBusy()` while loading; Cancel stays enabled.
- **Text loads follow the same pattern.** `TextDataset` construction
  (tokenize + build vocab) takes 1.5-2s on a 52k-row CSV which is too
  much for the UI thread, so the Apply path snapshots dialog state,
  hands the probe + `RegisterTextDataset` to a worker, and drains the
  result via `PollAsyncLoadResult` on the next UI frame. Backend code
  is 5 in `AsyncLoadState`. Image / Audio follow the same contract.
- `AsyncLoadState` is a `shared_ptr` captured by the worker, so the dialog
  being destroyed mid-load is safe (worker writes to memory it owns).
  `PollAsyncLoadResult` runs at the top of `RenderContent` every frame and
  drains the result via an `atomic<bool> done` publish barrier.
- Constructor restores load state by **probing the registry directly**, not
  by trusting `node->parameters["data_loaded"]`. The hint goes stale when
  async Apply finishes after the dialog closes (PollAsyncLoadResult only
  runs while the dialog is visible). Trusting the registry sidesteps that
  race; the constructor also re-syncs the param hint with reality. The
  **compile gate uses the same registry-first principle** — see the
  compile gate section below.

### Registry orphan cleanup

`DataRegistry::UnregisterTabularDataset(name)` and
`ClearAllTabularDatasets()` are wired into every lifecycle event so
datasets don't leak across sessions:

- **Project close / new / open**: `MainWindow::OnProjectClosed` calls
  `ClearAllTabularDatasets`. Since CreateProject and OpenProject both
  invoke CloseProject first, this single hook covers all three.
- **DataInput node delete**: `NodeEditor::DeleteNode` calls
  `UnregisterNodeDatasetIfOwned` before erasing the node.
- **Graph clear (Clear All button)**: `NodeEditor::ClearGraph` walks all
  nodes and unregisters before `nodes_.clear()`.
- **Re-Apply with a different file**: `DataInputDialog::Apply` captures
  the OLD `dataset_name` and unregisters it before launching the new load.
- **Re-Apply with the same file**: `LoadTabularCSV` calls
  `UnregisterTabularDataset(name)` at entry to clear any stale entry in
  the OTHER map (e.g. toggling Force disk-backed).

### Parquet cache hygiene

`ParquetBackedDataset::PruneCache(max_total_bytes=10GB, max_age_days=30)`:
- Two-pass: mtime expiry first, then LRU-by-mtime if over the size cap.
- Runs once at engine startup (`MainWindow` constructor) and after every
  successful `ConvertCsvToParquet`.
- `try_remove` swallows errors so Windows mmap-locked files (cache files
  currently being trained on) are silently skipped — next prune retries.

### Compile gate (validation tiers)

`GraphCompiler::Compile` populates `TrainingConfiguration::issues` (a
`vector<ValidationIssue>` with Error/Warning/Info levels) instead of
stopping at the first error. `is_valid` is computed as "no Error-level
issues"; warnings allow training but show in the popup.

`MainWindow::StartTrainingFromGraph` runs `BuildCompileResult` as a
pre-flight gate. If the result has any Error-level issues, it triggers
the same popup as the Compile button (with a "Cannot Start Training"
header) and refuses to launch training. No more silent log-and-return.

Key checks added on top of the original structural validation:
- DataInput's `data_loaded` parameter must be "true"
- Dataset name must resolve in `DataRegistry` (Arrow / Parquet / Image /
  Audio / Text) — the gate probes the registry directly, NOT the
  `node->parameters["data_loaded"]` hint. That hint is cached dialog
  state and can go stale when an async Apply completes after the dialog
  has already closed (PollAsyncLoadResult only runs while the dialog is
  visible, so the provisional `data_loaded="false"` set at Apply launch
  time gets stuck). Registry is the source of truth; when the gate
  overrides a stale hint it logs an info line so it's traceable.
- Label column should be set (warning if not)
- DataSplit ratios sum to 1.0 ± 0.05
- batch_size > 0 and ≤ train rows (error if too big, warning if > half)

### General principles

- Tabular data: `DataInputDialog` → `LoadTabularCSV` → in-memory Arrow OR
  disk-backed Parquet → `ArrowDatasetBatcher` / `ParquetArrowBatcher` →
  training. One entry point. Branch happens inside the registry, hidden
  from the user.
- Text data (Phase 3): `DataInputDialog` → async `TextDataset` probe +
  `RegisterTextDataset` on an `AsyncTaskManager` worker →
  `TextDatasetBatcher` → training. Same async contract as Arrow — OK /
  Apply buttons grey via `IsBusy()` while loading, Cancel stays enabled,
  result drains through `PollAsyncLoadResult`. Tokenizer / vocab /
  padding are overridable by `TextTokenizer` / `TextVocabulary` /
  `TextPadding` graph nodes; fallback is dialog defaults.
- Image / Audio: similar async + registry pattern as above, dispatched
  by `FileCategory` in `DataInputDialog::Apply`.
- Speed and memory come from Arrow optimizations, not user knobs.
  Auto-detect good defaults, bake them in, hide the controls.
- Node editor follows **single-responsibility**: one concern per node.
  Separate nodes for Normalize, Preprocess, DataSplit, DataLoader, etc.
  Do not couple concerns into one node.
- Users care about speed and memory, not configuration options. When
  adding features, prefer smart defaults over knobs. The Force disk-backed
  checkbox is the lone exception — an escape hatch for benchmarking the
  disk-backed code path on small files.
- **Adding a new NodeType — single point of truth as of 2026-04-17.**
  Registering a node in `NodeMetadataRegistry` (via an
  `Initialize*Nodes()` function in
  `cyxwiz-engine/src/core/node_metadata_registry.cpp`) makes it appear
  in all three UIs automatically: the **Node Browser** panel, the
  **search palette** (Ctrl+Space / type-to-find), and the
  **right-click context menu**. The two consumers iterate the registry
  at first use (`node_editor_add_search.cpp` +
  `node_editor_context_menu.cpp`). Plus one creation case in
  `node_editor_nodes.cpp` (`CreateNode` switch) to give it pins and
  params on the canvas. Two things remain dual-maintained:
  - `StringToNodeType` maps in `node_editor_io.cpp:20` AND
    `patterns/pattern_library.cpp:331` — both are string-to-enum
    lookup tables for cyxgraph JSON loading; forgetting either falls
    through to `NodeType::Dense` with a misleading warning and
    crashes at compile time with `invalid stoi argument`. (Lesson
    from the 2026-04-14 v2 text-graph incident.)
  - `ShouldShowOpenDialogButton` whitelist in
    `node_config_dialog.cpp:970` — add the NodeType here if the node
    should expose an "Open Dialog..." button in the Properties panel.
  Both of the above are known remaining drift points, tracked for
  future unification into the registry.

## Pipeline Architecture: Four Bands (2026-04-16 architectural pass)

Every training graph flows through four conceptual bands, in order.
The band boundaries determine what's cacheable, what's phase-aware,
and where bugs hide. Keep them clean.

**Band 1 — Data preparation (stateless, cacheable)**
  DataInput (all categories), TextTokenizer, TextVocabulary,
  TextPadding, LogTransform, Differencing, TimeSeriesWindow, static
  Resize, static format conversion, TFIDFVectorizer.
  - Deterministic given inputs. Identical behavior train/val/test.
  - **Can be pre-computed once and written to disk** — the
    "preprocess once, train many" workflow lives here.
  - Must not read any training state (no phase flag, no
    train/val/test awareness).

**Band 2 — Partitioning (deterministic, cacheable)**
  DataSplit (random, stratified), TimeSeriesSplit (chronological).
  - Assigns each row to a partition (train/val/test).
  - Deterministic given seed.
  - Still cacheable — partition assignments can be saved.
  - No phase flag yet; rows know *which group* they belong to but
    nothing has said "currently training vs validating".

**Band 3 — Iteration + phase-aware preprocessing**
  DataLoader (batching, shuffling, epoch control, `SetPhase`),
  Augmentation, BatchNorm (train updates running stats, val uses
  them), Input Dropout, MixUp, CutMix.
  - Phase flag is live here. Nodes change behavior based on
    train/val/test.
  - **NOT cacheable** — must re-run every epoch with current phase.
  - This is where "training mode" semantically begins.

**Band 4 — Model + optimization**
  Embedding, Dense, Conv2D, Flatten, LSTM, ReLU, Dropout, ...,
  CrossEntropyLoss, MSELoss, Adam, AdamW, Output.
  - Forward → loss → backward → optimizer update.
  - This is where weights change.

**Why this matters:**
- The "preprocess once, train many" cache boundary is **between
  Band 2 and Band 3**, not between DataSplit and DataLoader in
  some arbitrary way.
- Phase-aware nodes (Band 3) can NEVER be moved to Band 1 — they
  require the train/val phase flag which only exists after
  DataLoader.
- Data leakage happens when Band 3 state bleeds into Band 1 (e.g.,
  fitting a scaler on all data before splitting — scaler is
  Band 1, but it read train+val+test, which is a Band 2 violation).

### Nodes should be REAL operations, not config extractors

**Important architectural rule (post-2026-04-16):** new nodes
should transform data flowing through them, not act as
configuration extractors that the graph compiler scans for.

The Phase 3 text preprocessing nodes (`TextTokenizer` /
`TextVocabulary` / `TextPadding`) were implemented as config
extractors in `graph_compiler.cpp:1065-1121` as a shipping
shortcut — they have input/output pins but no data flows through
them. The compiler scans for their presence and pulls their params
into `config.text_preprocessing`; actual tokenization runs inside
`TextDatasetBatcher` on-the-fly during training. This blocks the
"tokenize once, write to disk, reuse" workflow and makes the pins
visually misleading.

**Fix B** (tracked in tofix.md "TextTokenizer is a config extractor,
not a pipeline operation") will rewrite these nodes as real
operations that transform Arrow tables. **Phase 4 and later work
MUST NOT repeat this shortcut.** New preprocessing nodes should
execute via the `cyxwiz-engine/src/core/node_executors/` framework
(scaffolded but unfinished) and transform their inputs in the
data-processing pipeline, not just carry config metadata.

## Node vs Panel: Four Tool Categories

Three orthogonal axes answer "should this be a node or a panel?":
- **Automated vs manual** — runs during training, or only when
  user opens it?
- **Pipeline data vs dev input** — operates on user's dataset, or
  on arbitrary dev input?
- **Transform vs inspect** — changes data, or just shows it?

The resulting four categories:

| Category | When | Data | Effect | Form |
|---|---|---|---|---|
| **1. Pipeline operations** | Automated | Pipeline | Transform | **Node** (rich dialog or Properties) |
| **2. Introspection tools** | Manual | Pipeline | Inspect only | **Panel AND/OR node** (shared backend) |
| **3. Dev utilities** | Manual | Dev input | Inspect or compute | **Panel** |
| **4. Debug / monitoring** | Automated | Engine state | Inspect only | **Panel** |

**Concrete examples per category:**
- **Cat 1 (always nodes):** TextTokenizer, Normalize, DataSplit,
  KMeans, Dense, Embedding, LSTM, FFT on a column, HypothesisTest
  on a column, TFIDFVectorizer.
- **Cat 2 (panels now, optional node form later):** Data Explorer,
  Data Profiler, Variable Explorer, Table Viewer, Visualization
  Panel, EmbeddingInspector (future). These hook into any pipeline
  point and read data without transforming.
- **Cat 3 (always panels):** Calculator, UUID generator, Hash
  generator, Unit converter, Regex tester, JSON viewer
  (paste-and-view), Random generator.
- **Cat 4 (always panels):** Console, Memory monitor, Job status,
  Profiling, Plugin manager, Python settings, Task progress,
  Wallet, Theme editor, Training dashboard.

**The refined v1 "everything is a node" rule:**
- Every **pipeline concept** (Cat 1) is a node. Non-negotiable.
- **Introspection** (Cat 2) is preferably a node (persistent in
  saved graphs) but also fine as a panel (one-shot debugging).
  Both forms should share backend rendering code.
- **Dev utilities** (Cat 3) are panels. Calculator has no data
  stream; it's a dev ergonomics tool.
- **Debug / monitoring** (Cat 4) are panels. They read engine
  internal state, not user pipeline data.

### Introspection is orthogonal to the pipeline bands

Inspection tools don't belong to any single band. They read from
any band without participating in the flow:

```
Band 1 → Band 2 → Band 3 → Band 4
   │       │       │       │
   ▼       ▼       ▼       ▼
 [Inspection tools can hook any point, read-only,
  skipped during automated training, invoked by the
  user when they need to understand "what does the
  data look like HERE?"]
```

Introspection nodes (future) connect to a pipeline point via an
input pin but have no transformed output pin — they sink data for
display only. They are skipped during the automated training path
and executed only when the user opens the node's dialog.

### Dialog vs Properties panel

The trigger for "rich dialog on double-click" is NOT "the node has
more than 2 params" — it's **"the node needs preview, validation,
or interactive inspection"**:

- **Properties panel** (flat key-value): Dense(units), Dropout(rate),
  ReLU, activations, BatchNorm, Conv2D kernel/stride — few
  parameters, no preview needed.
- **Rich dialog**: DataInput (async load, file picker, live preview,
  multi-tab config), TokenizerDialog (method selector, vocab
  preview), WordEmbeddings (file picker, dim validation, vocab
  alignment report), Category 2 inspection tools (interactive
  plots).

Existing precedents: `DataInputDialog`, `TokenizerDialog`.

## Completed Features

| Feature | Summary | Key Files |
|---------|---------|-----------|
| **Plugin System** | DLL loading, permissions, crash isolation, 5 registries | `src/plugin/` |
| **MuJoCo Plugin** | Physics sim, 3D viewport, 7 envs, RL nodes | `plugins/simulation/mujoco/` |
| **CyxWiz Studio** | KNIME-style node editor, 3 execution modes, v2.0 format | `src/gui/node_editor.*` |
| **Dataset Manager** | 3-pane layout, analytics, memory management | `src/gui/panels/dataset_panel.*` |
| **Annotation System** | Segmentation annotations, COCO/YOLO/VOC export | `src/core/annotation_manager.*` |
| **Backend Features** | Tokenizer, upsampling, time-series, audio, RL | `cyxwiz-backend/src/algorithms/` |
| **Transformer/RNN** | Embedding, LSTM, GRU, MultiHeadAttention, Transformer | `cyxwiz-backend/include/cyxwiz/layer.h` |

## P2P Training (Critical)

**Flow**: Engine → Central Server (reserve node) → P2P stream to Server Node → Unlimited jobs within reservation

**HOTEL ROOM Model**:
- **Disconnect**: Closes stream, keeps reservation (can reconnect)
- **Release**: Ends reservation, triggers payment

**Critical**: Use `stream_->WritesDone()` NOT `TryCancel()` (breaks reconnect)

**Key Files**: `src/network/p2p_client.cpp`, `src/job_execution_service.cpp`, `proto/execution.proto`

## Code Organization

| Directory | Purpose |
|-----------|---------|
| `cyxwiz-protocol/proto/` | gRPC definitions (.proto files) |
| `cyxwiz-backend/include/cyxwiz/` | Public API (Tensor, Layer, Model, Optimizer) |
| `cyxwiz-backend/python/` | pybind11 bindings → `pycyxwiz` module |
| `cyxwiz-engine/src/core/` | ProjectManager, DataRegistry, TrainingExecutor |
| `cyxwiz-engine/src/gui/` | ImGui panels, node editor, themes |
| `cyxwiz-engine/src/plugin/` | Plugin loader, manager, registries, security |
| `cyxwiz-engine/src/network/` | P2P client, gRPC client |
| `plugins/` | MuJoCo, MLflow logger, image nodes |

## Build Commands

**Preferred for incremental edit-compile-test (safe):**

```bash
# Git Bash / WSL
./scripts/rebuild.sh

# Native Windows (cmd)
scripts\rebuild.cmd
```

The wrappers capture the build log to `build/last-build.log`, check
cmake's real exit code, verify the binary actually moved forward, and
surface a unique warning summary. Use these in any scripted /
AI-driven workflow.

**Raw cmake (for manual one-offs only):**

```bash
cmake --preset windows-release && cmake --build build/windows-release --config Release
./build/windows-release/bin/cyxwiz-engine

# Central Server (Rust)
cd cyxwiz-central-server && cargo run --release
```

**DO NOT** pipe `cmake --build` through `tail`/`head` and trust the
exit code: bash pipelines return the LAST command's status, not
cmake's. We lost ~1h on 2026-04-17 running a zombie binary because
`cmake --build ... | tail -3` reported exit 0 while compile was
actually failing; the old binary stayed on disk and manual testing
showed the pre-refactor behavior. The wrappers above avoid the
pipe entirely.

## Development Tasks

**Start any new feature or non-trivial task with `/engineer`** — it's
the project's design-first skill (`.claude/skills/engineer/SKILL.md`).
Runs a short Understand → Principles → Plan pass before you write code,
so we don't repeat the DataInputDialog::Apply 700-line-switch disaster.
Mandatory for new features, multi-file changes, or any code that'll
grow past 200 lines. Skip only for trivial single-file edits.

The skill's anti-pattern checklist catches the specific mistakes this
codebase has repeatedly made (hand-maintained parallel lists, switches
on type that should be polymorphism, duplicated async blocks across
categories, whitelists that disagree with factories). Read it once
even if you're not invoking it, so you recognize the shapes.

| Task | Steps |
|------|-------|
| Add ML Algorithm | Header in `include/cyxwiz/` → Impl in `src/algorithms/` → Python bindings |
| Add gRPC Service | Define in `proto/*.proto` → Rebuild → Impl server/client |
| Add GUI Panel | Create in `src/gui/panels/` → Add to CMakeLists → Integrate in MainWindow |
| Any of the above, or anything else | **Invoke `/engineer` first** — produces the design note you then execute against |

## Cross-Platform Notes

```cpp
// File paths - always use forward slashes or std::filesystem::path
std::filesystem::path p = "data/models/model.h5";  // Good

// Platform detection
#if defined(_WIN32)    // Windows
#elif defined(__APPLE__)  // macOS
#elif defined(__linux__)  // Linux
#endif
```

## Dependencies

- **ArrayFire** (manual): Set `ArrayFire_DIR` env var
- **vcpkg**: imgui, glfw3, grpc, spdlog, pybind11, catch2

## Parallel sessions (multi-agent / multi-session workflow)

The tofix list has grown beyond what fits in one session, and several
items (DataLoader refactor, Local Debug, Tool-to-Node migration, etc.)
are architecturally independent. Running them in **parallel sessions**
— separate Claude Code instances or subagents — is supported, with
two isolation levers:

1. **Git worktrees** (recommended for large multi-commit work).
   Each session gets its own checkout pointing at a separate branch
   and its own `build/` directory, so compiles don't collide.

   ```bash
   # From repo root
   git worktree add ../CyxWiz_dataloader -b feat/dataloader
   git worktree add ../CyxWiz_localdebug  -b feat/local-debug
   ```

   Open Claude Code in each worktree directory. Each session sees
   its own branch state; commits land on separate branches. Merge /
   PR back to master when done.

2. **Subagent worktree mode** (for one-shot delegated work).
   When launching an Agent via the Agent tool, pass
   `isolation: "worktree"` — the runtime creates a temporary
   worktree for the agent's edits and returns the path + branch at
   the end. Cleanup is automatic if no changes land. Use for
   "please try this approach in isolation, report back with a diff"
   patterns.

**Coordination rules to avoid stepping on each other:**

- Every parallel session picks up ONE plan from `docs/plans/`. Don't
  let two sessions work the same plan — they'll produce conflicting
  refactors of the same files.
- The `tofix.md` and CLAUDE.md are merge-conflict hotspots. If you
  touch them in a session, rebase frequently or coordinate writes to
  a small section per session.
- The four per-category data loaders
  (`data_input_dialog.cpp::Apply`) are ONE file; the DataLoader
  refactor must run in its own session and merge before anything else
  touches that file.
- Build dirs must be per-worktree. A shared `build/` dir across
  sessions = mysterious "cmake reports success but wrong code ran"
  bugs (same class as the 2026-04-17 pipeline-masking incident).
- Protocol: the user's current session owns `master`. Worktree
  sessions branch off master and PR back when their plan is
  complete.

## TODOs

- Phase 8: KNIME Parity (data preview, database connectors, loop nodes)
- Blockchain Integration (Solana escrow/payment)
- Model Marketplace (NFT-based)
- Import/Export (ONNX, PyTorch, TensorFlow)

## Quick Reference

**Entry points**: `cyxwiz-*/src/main.cpp` or `main.rs`
**Engine panels**: `cyxwiz-engine/src/gui/panels/*.cpp`
**Plugin docs**: `docs/plugin_developer_guide.md`
**Build output**: `build/<preset>/bin/`
