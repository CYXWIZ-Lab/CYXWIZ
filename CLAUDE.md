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

- Tabular data: `DataInputDialog` → `LoadCSVToArrow` → `ArrowDataset` →
  `ArrowDatasetBatcher` → training. One path. No branching on memory_policy.
- Speed and memory come from Arrow optimizations, not user knobs. Auto-detect
  good defaults, bake them in, hide the controls.
- `CompactIntegerColumns` (in `data_registry.cpp`) auto-downcasts int64 → uint8/16
  when values fit, saving memory and batching time.
- `block_size` for Arrow CSV reader is set to 256 MB by default in
  `LoadCSVToArrow` so the table loads as one chunk and the batcher hits the
  fast `raw_values()` path. Don't expose as a UI control unless explicitly asked.
- Node editor follows **single-responsibility**: one concern per node. Separate
  nodes for Normalize, Preprocess, DataSplit, DataLoader, etc. Do not couple
  concerns into one node.
- Users care about speed and memory, not configuration options. When adding
  features, prefer smart defaults over knobs.

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
