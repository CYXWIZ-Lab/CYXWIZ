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
- **Adding a new NodeType: update TWO `StringToNodeType` maps** —
  `cyxwiz-engine/src/gui/node_editor_io.cpp:20` (used by File → Open)
  AND `cyxwiz-engine/src/gui/patterns/pattern_library.cpp:331` (used by
  the pattern library). Both are string-to-enum lookup tables for
  cyxgraph JSON loading; forgetting either one falls through to
  `NodeType::Dense` with a misleading warning and then crashes at
  compile time with `invalid stoi argument` because the node params
  don't match the Dense layout. (Lesson from the 2026-04-14 v2 text
  graph incident — the Phase 3 nodes existed in the enum and the add
  menu but were missing from both loader maps.)

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

```bash
# Quick build
cmake --preset windows-release && cmake --build build/windows-release --config Release

# Run
./build/windows-release/bin/cyxwiz-engine

# Central Server (Rust)
cd cyxwiz-central-server && cargo run --release
```

## Development Tasks

| Task | Steps |
|------|-------|
| Add ML Algorithm | Header in `include/cyxwiz/` → Impl in `src/algorithms/` → Python bindings |
| Add gRPC Service | Define in `proto/*.proto` → Rebuild → Impl server/client |
| Add GUI Panel | Create in `src/gui/panels/` → Add to CMakeLists → Integrate in MainWindow |

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
