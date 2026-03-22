# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current Work Status (Updated: 2026-03-21)

### CyxWiz Engine - Plugin System ✅ (UPDATED)

**Complete plugin architecture with dynamic DLL loading, permission security, and crash isolation:**

- ✅ **Phase 1**: Plugin types, interfaces (`IPlugin`, `ITrainingHook`, `INodeProvider`, `IPanelProvider`, `IDataProvider`, `IAnalyticsProvider`), manifest parsing
- ✅ **Phase 2**: `PluginContext` (sandboxed API surface), permission system (dangerous vs safe), crash isolation (SEH/signals)
- ✅ **Phase 3**: `PluginLoader` (DLL loading via LoadLibrary/dlopen, Ed25519 signature verification)
- ✅ **Phase 4**: `PluginRegistry` integration — 5 registries connecting plugins to engine subsystems
- ✅ **Phase 5**: Plugin Manager UI panel (ImGui) — install, enable/disable, unload, permission badges
- ✅ **Phase 6**: Example plugins (MLflow Logger, Image Nodes) + Developer Guide
- ✅ **Phase 7**: Crash isolation for QueryInterface registration, recursive plugin discovery for nested plugins

**Architecture:**
```
Plugin DLL (.dll/.so/.dylib)
  ├── Links: cyxwiz-plugin-sdk (static lib)
  └── Exports: CreatePlugin() / DestroyPlugin()

Engine
  ├── PluginManager      — discovery, loading, lifecycle
  ├── PluginLoader       — DLL loading, manifest parsing, signature verification
  ├── PluginContext       — sandboxed API (logging + 5 registration methods)
  ├── 5 Registries       — NodeRegistry, PanelRegistry, DataLoaderRegistry,
  │                        TrainingHookManager, AnalyticsRegistry
  ├── PermissionStore    — persisted user approval decisions
  ├── PermissionDialog   — ImGui approval UI for dangerous permissions
  └── SafeExecute        — crash isolation (SEH on Windows, signals on Unix)
```

**Key Files:**
| File | Purpose |
|------|---------|
| `src/plugin/plugin_types.h` | All types, enums, manifest, interfaces, entry macro |
| `src/plugin/plugin_context.h/cpp` | Sandboxed API surface for plugins |
| `src/plugin/plugin_loader.h/cpp` | DLL loading, manifest parsing, signature verification |
| `src/plugin/plugin_manager.h/cpp` | Discovery, lifecycle orchestration |
| `src/plugin/interfaces/i_*.h` | 5 optional plugin interfaces |
| `src/plugin/registries/*.h/cpp` | 5 engine-side registries |
| `src/plugin/security/*.h/cpp` | Permission store/dialog, crash isolation, Ed25519 |
| `src/gui/panels/plugin_manager_panel.h/cpp` | Plugin Manager UI panel |
| `plugins/examples/mlflow_logger/` | Example: training hooks + early stopping |
| `plugins/examples/image_nodes/` | Example: custom nodes + settings panel |
| `docs/plugin_developer_guide.md` | Full developer documentation |

**Plugin Search Paths (recursive discovery):**
- `<cwd>/plugins/` (project-specific) — supports nested dirs like `plugins/simulation/mujoco/`
- `%APPDATA%/cyxwiz/plugins/` (Windows user) or `~/.cyxwiz/plugins/` (Linux/macOS)

**Recent Fixes (2026-02-04):**
- Wrapped QueryInterface registration (panels, nodes, hooks) in SafeExecute for crash isolation
- Fixed std::string ABI issues across DLL boundary in plugin callbacks
- Changed plugin discovery to recursive for nested plugin directories

**Building Plugins:**
```cmake
add_library(my_plugin SHARED my_plugin.cpp)
target_link_libraries(my_plugin PRIVATE cyxwiz-plugin-sdk)
```

**Permission Model:**
- Safe (auto-granted): GPU, DataRegistry, Training, UIModify
- Dangerous (user approval required): FileSystem, Network, SystemCommands, Python

---

### CyxWiz Engine - MuJoCo Simulation Plugin ✅ (Phase 1 Complete)

**Production-grade MuJoCo physics integration as a plugin (DLL):**

- ✅ **Phase 1**: Core plugin — physics wrapper, 3D renderer, viewport panel, environment library (7 envs)
- ✅ **Phase 2**: Node editor integration — 5 RL node types with code generation
- ✅ **Phase 3**: Cross-DLL safety (NodeTypeCallback API), plugin nodes in context menu, Menagerie library + URL import
- 🔲 **Phase 4 (Planned)**: Simulink-style simulation executor, node graph driving physics

**Current Architecture:**
```
plugins/simulation/mujoco/
  ├── src/mujoco_plugin.h/cpp       — Plugin lifecycle, node types, code gen
  ├── src/mj_env_manager.h/cpp      — MuJoCo physics wrapper (step/reset/observe)
  ├── src/mj_renderer.h/cpp         — 3D OpenGL rendering (CPU readback → ImGui texture)
  ├── src/mj_viewport_panel.h/cpp   — Live 3D viewport with Play/Pause/Step
  ├── src/mj_env_library.h/cpp      — Environment catalog (7 built-in + 30+ Menagerie)
  ├── src/mj_env_browser_panel.h/cpp — Browse/load/import environments UI
  ├── src/mj_menagerie_downloader.h/cpp — GitHub Menagerie model downloader + URL import
  ├── assets/*.xml                   — 7 MJCF model files
  └── CMakeLists.txt                 — Links mujoco, glad, glfw, plugin-sdk
```

**Built-in Environments:**
- Classic Control: InvertedPendulum, CartPole, Reacher
- Locomotion: Hopper, Walker2D, HalfCheetah
- Manipulation: Pusher
- Menagerie (30+ downloadable): Franka Panda, UR5e, Unitree Go2, ANYmal, Spot, Shadow Hand, etc.

**Node Editor Nodes (in Reinforcement Learning context menu):**
| Node | Purpose |
|------|---------|
| MuJoCoEnv | Load MJCF model as Gymnasium env |
| RewardFunction | Reward shaping (alive bonus, control cost, velocity) |
| ObservationFilter | Filter/normalize observations (qpos, qvel, sensors) |
| RLAgent | PPO/SAC agent with Stable-Baselines3 code gen |
| MuJoCo Plant | Simulink-style node with dynamic actuator/sensor pins |

**Cross-DLL Safety:**
- `NodeTypeCallback` API avoids passing `std::vector`/`std::string` across DLL boundary
- Plugin nodes dynamically injected into node editor context menu
- `RegisterDirect()` for engine-side deep copy of node type info

**URL Import:**
- Paste any GitHub URL (tree/blob) or direct MJCF URL in Environment Library
- Downloads to `~/.cyxwiz/imported/<model_name>/`
- Auto-adds to library as "Imported" category
- Signal source nodes (Constant, Slider, Sine, Scope)
- Simulation executor (real-time physics stepping from node graph)
- RL training executor (episode loop with live 3D visualization)
- MuJoCo Menagerie library (70+ robot models, bundled + downloadable)

---

### CyxWiz Engine - Unified Canvas Architecture ✅ (COMPLETE)

**KNIME-inspired unified canvas merging Node Editor and Data Studio Pipeline:**

- ✅ **Phase 5**: Data Studio Simplification — Removed duplicate Pipeline Canvas, moved to Node Editor
- ✅ **Phase 6**: UI/UX Improvements — Execution state visualization, node/pin tooltips
- ✅ **Phase 7**: Save/Load Integration — Extended .cyxgraph v2.0 format with execution_mode

**Architecture:**
```
Unified Node Editor (node_editor.cpp/h)
  ├── ExecutionMode: CodeGeneration | DuckDBPipeline | LocalTraining
  ├── NodeCategory: Input | Transform | ML | RL | Visualization | Output
  ├── PipelineExecutor: DuckDB/Arrow backend for data pipelines
  ├── Execution Visualization: Pulsing animation, state colors
  └── Enhanced Tooltips: Pin types, connection status, node state
```

**Execution States:**
| State | Visual | Tooltip |
|-------|--------|---------|
| Idle | Default color | "Ready" |
| Pending | Queue indicator | "Waiting to execute" |
| Executing | Pulsing blue border | "Executing..." |
| Completed | Green checkmark | "Completed successfully" |
| Error | Red X | Error message displayed |

**File Format v2.0:**
```json
{
  "version": "2.0",
  "framework": 0,
  "execution_mode": 1,
  "nodes": [
    {
      "id": 1,
      "type": "DataInput",
      "category": 0,
      "pos_x": 100.0,
      "pos_y": 100.0,
      "properties": {}
    }
  ],
  "links": []
}
```

**Key Files:**
| File | Purpose |
|------|---------|
| `src/gui/node_editor.h` | NodeExecutionState enum, execution visualization state |
| `src/gui/node_editor.cpp` | Execution state rendering, enhanced tooltips |
| `src/gui/node_editor_io.cpp` | Save/load v2.0 format with backward compatibility |
| `src/gui/data_studio/data_studio_panel.cpp` | Simplified (removed pipeline_canvas_) |
| `src/gui/data_studio/data_studio_panel.h` | Simplified (removed pipeline_canvas_) |
| `docs/Data Studio/knime_comparison.md` | Comprehensive KNIME feature comparison |

**KNIME Comparison:**
- ✅ **Similarities**: Visual node editor, drag-drop, execution engine, professional UI
- ✅ **Differentiators**: GPU training, multi-framework code gen, RL support, P2P compute
- 🔲 **Gaps**: Interactive data preview, database connectors, loop nodes, annotations
- 📋 **Roadmap**: Phase 8 planned for KNIME parity features

**Phase 5 Changes:**
- Removed `PipelineCanvas` class from Data Studio
- Moved visual pipeline editing to Node Editor with ExecutionMode switch
- Updated Data Studio to focus on Query/Analyze/Visualize tabs only
- Added tooltip in Data Studio toolbar directing users to Node Editor for pipelines

**Phase 6 Changes:**
- Added `NodeExecutionState` enum (Idle, Pending, Executing, Completed, Error)
- Implemented pulsing animation for executing nodes (blue glow effect)
- Added state-based node coloring (green for completed, red for error)
- Added pin tooltips showing type (Tensor, Labels, Parameters, Loss, Optimizer, Dataset)
- Added pin tooltips showing connection status (connected/not connected)
- Enhanced node tooltips with execution state and error messages

**Phase 7 Changes:**
- Extended .cyxgraph format to v2.0 with `execution_mode` field
- Added `category` field to each node for better organization
- Implemented backward compatibility for v1.0 files (auto-detects version)
- v1.0 files load with default `CodeGeneration` mode

**Branch Status:**
- Branch: `unified-canvas`
- Status: Complete, tested, pushed to origin
- Ready to merge to `master`

---

### CyxWiz Engine - Dataset Manager Redesign ✅

**Professional DBGate-style 3-pane layout:**

- ✅ **3-Pane Layout** - Sidebar (dataset tree), Main content (tabs), Status bar
- ✅ **Dataset Tree View** - Collapsible nodes with train/val/test splits
- ✅ **Draggable Splitter** - Adjustable sidebar width (150-350px)
- ✅ **4 Content Tabs** - Preview, Pipeline, Training, Details (simplified from 6)
- ✅ **Compact Styling** - Professional look with reduced margins
- ✅ **Analytics System** - Parallelized computation (12 threads), progress bar, notifications

**Key Files:**
| File | Purpose |
|------|---------|
| `src/gui/panels/dataset_panel.cpp` | Main panel with 3-pane layout |
| `src/gui/panels/dataset_panel.h` | Layout state, ContentTab enum |
| `src/core/async_task_manager.cpp` | Fixed GetTask() for fast-completing tasks |
| `src/utils/dataset_analyzer.cpp` | Parallelized analytics computation |

**Layout Structure:**
```
+------------------------------------------------------------------+
| TOOLBAR: [Refresh] [Memory Bar] [Search]               [Settings]|
+----------+-------------------------------------------------------+
| SIDEBAR  | TABS: [Preview] [Pipeline] [Training] [Details]       |
+----------+-------------------------------------------------------+
| Dataset  |                                                       |
| Tree     |  [Tab Content Area]                                   |
| - mnist  |                                                       |
|   train  |  Clean, professional content with proper spacing      |
|   val    |                                                       |
|   test   |                                                       |
+----------+-------------------------------------------------------+
| STATUS: Ready | mnist (60,000 samples) | Memory: 2.1 GB          |
+------------------------------------------------------------------+
```

**Analytics Features:**
- Class distribution with bar chart
- Color/grayscale histograms
- Brightness/contrast statistics
- Outlier detection (IQR method)
- Quality analysis (blur, noise, exposure)
- Duplicate detection (pHash, aHash, dHash)

---

### CyxWiz Engine - Annotation System ✅

**Production-ready annotation system for semantic segmentation:**

- ✅ **AnnotationManager** - Central annotation storage (`src/core/annotation_manager.h/cpp`)
- ✅ **Batch Navigation UI** - Prev/Next/Go-to for dataset images
- ✅ **Annotation List UI** - View, select, delete annotations per image
- ✅ **Class Management** - Add/select class labels
- ✅ **Save from Tools** - Convert mask to annotation with one click
- ✅ **Export Formats** - COCO JSON, YOLO txt, Pascal VOC XML
- ✅ **Training Integration** - `GetAnnotatedBatch()` for segmentation training

**Key Files:**
| File | Purpose |
|------|---------|
| `src/core/annotation_manager.h` | Data structures (Annotation, AnnotationSet) |
| `src/core/annotation_manager.cpp` | Storage, export, mask generation |
| `src/core/dataset_batcher.h` | `AnnotatedBatch` struct, `GetAnnotatedBatch()` |
| `src/gui/panels/dataset_panel_interactive.cpp` | UI for annotation workflow |

**Usage - Training with Annotations:**
```cpp
DatasetBatcher batcher(dataset, 32, DatasetSplit::Train);
while (batcher.HasNext()) {
    AnnotatedBatch batch = batcher.GetNextAnnotatedBatch("my_dataset");
    // batch.images: [B, H, W, C] - input images
    // batch.masks:  [B, H, W]    - segmentation masks (class IDs per pixel)
}
```

**Usage - Export Annotations:**
```cpp
auto& ann_mgr = DataRegistry::Instance().GetAnnotationManager();
ann_mgr.ExportCOCO("dataset", "coco.json");   // For Detectron2, MMDet
ann_mgr.ExportYOLO("dataset", "yolo/");       // For YOLOv5/v8
ann_mgr.ExportVOC("dataset", "voc/");         // For Pascal VOC tools
```

---

### CyxWiz Backend - Feature Gap Implementation ✅ (5 Phases Complete)

**Phase 1: Text Tokenization** — Vocabulary, Tokenizer (Word/Whitespace/Character), TextDataset
**Phase 2: Upsampling Layers** — ConvTranspose2D, Upsample2D, PixelShuffle
**Phase 3: Time-Series Windowing** — Sliding windows, lag/rolling/diff features, chronological split
**Phase 4: Audio Processing** — libsndfile I/O, Spectrogram, Mel, MFCC, augmentation (via FFTW3)
**Phase 5: RL Interface** — ReplayBuffer, EpsilonSchedule, GymConnector (Python bridge)

**Key Files:**
| Component | Header | Implementation |
|-----------|--------|---------------|
| Tokenizer | `cyxwiz-backend/include/cyxwiz/tokenizer.h` | `src/algorithms/tokenizer.cpp` |
| Upsampling | `cyxwiz-backend/include/cyxwiz/layer.h` | `src/algorithms/layer.cpp` |
| Time-Series | `cyxwiz-backend/include/cyxwiz/time_series.h` | `src/algorithms/time_series.cpp` |
| Audio | `cyxwiz-backend/include/cyxwiz/audio_processing.h` | `src/algorithms/audio_processing.cpp` |
| RL | `cyxwiz-backend/include/cyxwiz/rl_interface.h` | `src/algorithms/rl_interface.cpp` |
| GymConnector | `cyxwiz-engine/src/core/gym_connector.h` | `src/core/gym_connector.cpp` |
| TextDataset | `cyxwiz-engine/src/core/formats/text_dataset.h` | `src/core/formats/text_dataset.cpp` |
| TimeSeriesDataset | `cyxwiz-engine/src/core/formats/timeseries_dataset.h` | `src/core/formats/timeseries_dataset.cpp` |
| AudioDataset | `cyxwiz-engine/src/core/formats/audio_dataset.h` | `src/core/formats/audio_dataset.cpp` |

**Node Editor nodes added:** TextTokenizer, TextVocabulary, TextPadding, ConvTranspose2D, Upsample, PixelShuffle, TimeSeriesWindow, TimeSeriesFeatures, TimeSeriesSplit, AudioInput, Spectrogram, MelSpectrogram, MFCC, AudioAugmentation, GymEnvironment, ReplayBuffer, PolicyNetwork, ValueNetwork, RLTraining

**Python bindings:** All 5 phases exposed via pybind11 in `cyxwiz-backend/python/bindings.cpp`

**Documentation:** `docs/feature_usage_examples.md` — comprehensive reference with 16 sections and 5 end-to-end examples

---

### CyxWiz Backend - Transformer & RNN Layers ✅

- ✅ EmbeddingLayer (pretrained weights, padding_idx, max_norm)
- ✅ LSTMLayer (multi-layer, bidirectional, dropout)
- ✅ GRULayer (multi-layer, bidirectional, dropout)
- ✅ MultiHeadAttentionLayer (self/cross attention)
- ✅ TransformerEncoderLayer (Pre-LN and Post-LN)
- ✅ TransformerDecoderLayer (cross-attention, causal masking)
- ✅ FocalLoss, TripletLoss, ContrastiveLoss
- ✅ All exposed via pybind11

---

### CyxCloud - DOCKER TESTING COMPLETE ✅

**Repository:** https://github.com/CYXWIZ-Lab/cyxcloud

**Completed:**
- ✅ All Rust code compiles and tests pass
- ✅ Blockchain programs deployed to Solana Devnet
- ✅ GitHub Actions CI/CD for releases
- ✅ v0.1.0-alpha released with binaries for Linux, macOS, Windows
- ✅ Documentation updated (README, USAGE, USECASE)
- ✅ Docker Compose configured for full stack
- ✅ Arch Linux setup instructions added
- ✅ **Docker testing complete (2025-12-31)**

**Docker Test Results:**
| Component | Status | Details |
|-----------|--------|---------|
| PostgreSQL | ✅ Healthy | Metadata storage working |
| Redis | ✅ Healthy | Cache layer working |
| Gateway | ✅ Healthy | HTTP :8080, gRPC :50052 |
| Node 1 | ✅ Online | 100 GB storage registered |
| Node 2 | ✅ Online | 100 GB storage registered |
| Node 3 | ✅ Online | 100 GB storage registered |
| CyxWiz API | ✅ Running | Authentication service |

**Verified Features:**
- ✅ S3-compatible API (PUT/GET/DELETE objects)
- ✅ Erasure coding (10 data + 4 parity shards)
- ✅ 3x replication across all nodes
- ✅ CLI authentication via CyxWiz API
- ✅ CLI upload/download/list/delete operations
- ✅ Node registration and heartbeat
- ✅ Chunk distribution verified in database

**CLI Commands Tested:**
```bash
# Authentication (registration via website only for security)
cyxcloud login -e user@example.com  # Prompts for password

# Storage operations
cyxcloud status                           # Check gateway status
cyxcloud upload -b bucket file.txt        # Upload file
cyxcloud list bucket                      # List objects
cyxcloud download bucket -k file -o out   # Download file
cyxcloud delete bucket key -f             # Delete object
cyxcloud whoami                           # Show user profile
```

**Docker Services:**
| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL | 5432 | Metadata storage |
| Redis | 6379 | Caching layer |
| Gateway | 8080, 50052 | HTTP/S3 API + gRPC |
| CyxWiz API | 3002 | Authentication |
| Node 1 | 50061, 4001, 9091 | Storage node |
| Node 2 | 50062, 4002, 9092 | Storage node |
| Node 3 | 50063, 4003, 9093 | Storage node |

**Quick Start:**
```bash
cd D:/Dev/CyxWiz_Claude/cyx_cloud
docker compose up -d --build
curl http://localhost:8080/health  # Should return "OK"
```

---

## Project Overview

CyxWiz is a **decentralized ML compute platform** consisting of three interconnected projects:

1. **CyxWiz Engine** - Desktop client with visual node editor (C++ with ImGui/Dear ImGui)
2. **CyxWiz Server Node** - Distributed compute worker (C++ with ArrayFire)
3. **CyxWiz Central Server** - Network orchestrator (Rust with gRPC)

**Core Technology Stack:**
- **Languages**: C++20, Rust, Python (scripting)
- **GUI**: Dear ImGui, ImNodes, ImPlot, GLFW, OpenGL
- **Compute**: ArrayFire (CUDA/OpenCL/CPU backends)
- **Networking**: gRPC with Protocol Buffers
- **Blockchain**: Solana (primary), Polygon (secondary)
- **Build System**: CMake 3.20+, Cargo (Rust)
- **Dependencies**: vcpkg for C++

## Platform Support

- **Desktop**: Windows, macOS, Linux (all three components)
- **Android**: Backend library only (cyxwiz-backend as shared library)
- **Build Modes**: Debug (logging, memory tracking), Release (optimizations)

## Architecture

### Component Relationships

```
CyxWiz Engine (Desktop Client)
    ├─ Links: cyxwiz-backend.dll/.so (compute)
    ├─ Links: cyxwiz-protocol (gRPC stubs)
    └─ Connects: Central Server via gRPC

CyxWiz Server Node (Compute Worker)
    ├─ Links: cyxwiz-backend.dll/.so (compute)
    ├─ Links: cyxwiz-protocol (gRPC stubs)
    ├─ Connects: Central Server for job assignment
    └─ Executes: ML training jobs using ArrayFire

CyxWiz Central Server (Orchestrator - Rust)
    ├─ Implements: gRPC server (Job/Node services)
    ├─ Manages: Node registry, job scheduling
    ├─ Integrates: PostgreSQL, Redis, Solana blockchain
    └─ Coordinates: P2P connections between Engine and Nodes
```

### Key Design Principles

1. **Backend is Shared**: `cyxwiz-backend` is a DLL/SO used by both Engine and Server Node
2. **Protocol First**: All network communication defined in `.proto` files
3. **Cross-Platform**: All C++ code must work on Windows/macOS/Linux
4. **Python Scripting**: Embedded Python interpreter in Engine (pybind11)
5. **Debug vs Release**: Debug builds have extensive logging and memory tracking
6. **Reservation-Based Payment**: Users pay for TIME, not per-job (see P2P Training Flow below)

## P2P Training Flow (Reservation-Based)

**Key Principle**: User pays for TIME (reservation), can submit UNLIMITED jobs within that time slot.

**Flow**: Engine → Central Server (ListFreeNodes, ReserveNode) → P2P Connect to Server Node → Stream training with Pause/Resume/Stop/NewJob controls.

**Key Files**:
- Engine: `src/network/p2p_client.cpp`, `src/gui/panels/p2p_training_panel.cpp`
- Server Node: `src/job_execution_service.cpp`, `src/remote_data_loader.cpp`
- Proto: `proto/execution.proto`, `proto/reservation.proto`

**HOTEL ROOM Model** (Disconnect vs Release):
- **Disconnect**: Closes stream, keeps reservation (can reconnect)
- **Release**: Ends reservation, triggers payment (cannot reconnect)

**Critical**: Use `stream_->WritesDone()` for graceful close, NOT `TryCancel()` (breaks reconnect).

## Build System

### CMake Structure

- **Root `CMakeLists.txt`**: Orchestrates all subprojects
- **CMakePresets.json**: Platform-specific configurations
  - `windows-debug`, `windows-release`
  - `linux-debug`, `linux-release`
  - `macos-debug`, `macos-release`
  - `android-release` (backend only)

### Build Configuration Options

```cmake
CYXWIZ_BUILD_ENGINE=ON/OFF          # Build desktop client
CYXWIZ_BUILD_SERVER_NODE=ON/OFF     # Build compute node
CYXWIZ_BUILD_CENTRAL_SERVER=ON/OFF  # Build orchestrator
CYXWIZ_BUILD_TESTS=ON/OFF           # Build unit tests
CYXWIZ_ENABLE_CUDA=ON/OFF           # Enable CUDA backend
CYXWIZ_ENABLE_OPENCL=ON/OFF         # Enable OpenCL backend
CYXWIZ_ANDROID_BUILD=ON/OFF         # Android build mode
```

### Platform-Specific Flags

**Debug Build**:
- Defines: `CYXWIZ_DEBUG`, `CYXWIZ_ENABLE_LOGGING`, `CYXWIZ_ENABLE_PROFILING`
- Compiler flags: `-g -O0` (GCC/Clang), `/Zi /Od` (MSVC)

**Release Build**:
- Defines: `CYXWIZ_RELEASE`, `NDEBUG`
- Compiler flags: `-O3` (GCC/Clang), `/O2` (MSVC)

## Development Workflow

### Building the Project

**Quick Build (All Platforms)**:
```bash
# Windows
scripts\build.bat

# Linux/macOS
./scripts/build.sh
```

**Manual Build**:
```bash
# Configure
cmake --preset <platform>-<config>
# Example: cmake --preset windows-release

# Build
cmake --build build/<preset-name> --config Release

# Run tests
cd build/<preset-name>
ctest --output-on-failure
```

**Building Individual Components**:
```bash
# Engine only
cmake --preset windows-release -DCYXWIZ_BUILD_SERVER_NODE=OFF -DCYXWIZ_BUILD_CENTRAL_SERVER=OFF
cmake --build build/windows-release

# Server Node only
cmake --preset linux-release -DCYXWIZ_BUILD_ENGINE=OFF -DCYXWIZ_BUILD_CENTRAL_SERVER=OFF
cmake --build build/linux-release

# Central Server (Rust)
cd cyxwiz-central-server
cargo build --release
```

### Running Components

```bash
# Engine (Desktop Client)
./build/windows-release/bin/cyxwiz-engine

# Server Node
./build/windows-release/bin/cyxwiz-server-node

# Central Server
cd cyxwiz-central-server
cargo run --release
```

### Testing

```bash
# Run all tests
cd build/<preset-name>
ctest --output-on-failure

# Run specific test
./bin/cyxwiz-tests "[tensor]"

# Rust tests
cd cyxwiz-central-server
cargo test
```

## Code Organization

| Component | Purpose | Key Directories |
|-----------|---------|-----------------|
| `cyxwiz-protocol/` | gRPC definitions | `proto/*.proto` (common, job, node, compute) |
| `cyxwiz-backend/` | ML compute library | `include/cyxwiz/` (API), `src/algorithms/`, `python/` (pybind11) |
| `cyxwiz-engine/` | Desktop IDE | `src/core/`, `src/gui/`, `src/scripting/`, `src/network/` |
| `cyxwiz-server-node/` | Compute worker | `src/` (job_executor, metrics_collector) |
| `cyxwiz-central-server/` | Rust orchestrator | `src/api/`, `src/scheduler/`, `src/blockchain/` |

**Backend Key Classes**: `Tensor`, `Device`, `Optimizer`, `Layer`, `Model`
**Engine Key Systems**: ProjectManager, AsyncTaskManager, DataRegistry, NodeEditor, TrainingExecutor
**Python Module**: `pycyxwiz` (built with pybind11, location: `build/<preset>/python/`)

## Common Development Tasks

| Task | Steps |
|------|-------|
| **Add ML Algorithm** | 1. Header in `include/cyxwiz/` 2. Impl in `src/algorithms/` 3. Tests 4. Python bindings |
| **Add gRPC Service** | 1. Define in `proto/*.proto` 2. Rebuild 3. Impl server (Rust/C++) 4. Impl client (C++) |
| **Add GUI Panel** | 1. Create `.h/.cpp` in `src/gui/` 2. Add to CMakeLists 3. Integrate in MainWindow |

**Debug Macros**: `CYXWIZ_DEBUG`, `CYXWIZ_ENABLE_LOGGING`, `CYXWIZ_ENABLE_PROFILING`
**Logging**: `spdlog::debug/info/warn/error("message: {}", value);`

## External Dependencies

- **ArrayFire** (manual): GPU acceleration - set `ArrayFire_DIR` env var
- **vcpkg** (managed): imgui, glfw3, glad, grpc, protobuf, spdlog, pybind11, catch2, boost
- **Optional**: ImNodes, ImPlot, btop (see source repos)

## Blockchain Integration

**Solana**: CYXWIZ SPL token, escrow-based payments (90% node, 10% platform)
**Programs**: JobEscrow, PaymentStreaming, NodeStaking (in `cyxwiz-blockchain/`)

## Important Notes

### Cross-Platform Considerations

**File Paths**: Always use forward slashes `/` or `std::filesystem::path`
```cpp
// Good
std::filesystem::path p = "data/models/model.h5";

// Bad (Windows-specific)
std::string p = "data\\models\\model.h5";
```

**DLL Export/Import**:
```cpp
#ifdef _WIN32
    #ifdef CYXWIZ_BACKEND_EXPORTS
        #define CYXWIZ_API __declspec(dllexport)
    #else
        #define CYXWIZ_API __declspec(dllimport)
    #endif
#else
    #define CYXWIZ_API __attribute__((visibility("default")))
#endif
```

**Platform Detection**:
```cpp
#if defined(_WIN32)
    // Windows
#elif defined(__APPLE__)
    // macOS
#elif defined(__linux__)
    // Linux
#elif defined(__ANDROID__)
    // Android
#endif
```

### Security Considerations

1. **Never commit**: Private keys, wallet files, API keys
2. **Sandboxing**: Use Docker for untrusted workloads on Server Nodes
3. **Validation**: Verify all gRPC inputs on server side
4. **Authentication**: Implement JWT tokens for gRPC (TODO)
5. **Encryption**: Use TLS for all gRPC connections (TODO)

### Performance

**ArrayFire Best Practices**:
- Batch operations instead of loops
- Keep data on GPU (avoid CPU↔GPU transfers)
- Use `af::sync()` only when necessary
- Profile with `af::timer`

**ImGui Performance**:
- Minimize `ImGui::Text()` calls in hot loops
- Use `ImGuiListClipper` for long lists
- Cache computed values instead of recalculating

## Troubleshooting

### Build Issues

**"ArrayFire not found"**:
- Install ArrayFire from https://arrayfire.com/download
- Set `CMAKE_PREFIX_PATH` to ArrayFire installation directory
- Or set `ArrayFire_DIR` environment variable

**"vcpkg dependencies missing"**:
```bash
cd vcpkg
./vcpkg install
```

**"gRPC generation failed"**:
- Ensure protobuf and gRPC are installed via vcpkg
- Check that `.proto` files have no syntax errors
- Rebuild from clean: `rm -rf build && cmake --preset <preset>`

### Runtime Issues

**"Failed to initialize Python"**:
- Ensure Python 3.8+ is installed
- Check that pybind11 was found during CMake configuration
- On Windows, ensure Python DLL is in PATH

**"ArrayFire error: driver not found"**:
- Install CUDA Toolkit (for CUDA backend)
- Install OpenCL drivers (for OpenCL backend)
- Fall back to CPU: `af::setBackend(AF_BACKEND_CPU)`

**"gRPC connection refused"**:
- Ensure Central Server is running
- Check firewall settings
- Verify server address and port (default: `localhost:50051`)

## Future Work (TODOs in Code)

High-priority tasks marked with `// TODO:` throughout codebase:

1. ~~**ImNodes Integration**~~ ✅ - Visual node editor (DONE - full pipeline builder with code generation)
2. ~~**ImPlot Integration**~~ ✅ - Real-time training plots (DONE - PlotWindow implemented)
3. ~~**Training Controls**~~ ✅ - Start/Pause/Resume/Stop training (DONE - full P2P implementation)
4. ~~**P2P Training**~~ ✅ - Direct Engine↔Node communication (DONE - bidirectional streaming)
5. ~~**Multi-Job Training**~~ ✅ - Multiple jobs per reservation (DONE - unlimited jobs within reserved time)
6. ~~**Job Execution**~~ ✅ - Complete job executor in Server Node (DONE - real training with RemoteDataLoader)
7. ~~**Authentication**~~ ✅ - JWT tokens for gRPC (DONE - P2P auth tokens implemented)
8. ~~**Unified Canvas**~~ ✅ - KNIME-inspired architecture (DONE - Phases 5-7 complete)
9. **Phase 8: KNIME Parity** - Interactive data preview, database connectors, loop nodes, workflow annotations, subgraphs
10. **btop Integration** - Server Node monitoring TUI
11. **Blockchain Integration** - Solana payment processor (escrow/payment release)
12. **Docker Support** - Containerized job execution
13. **Model Marketplace** - NFT-based model sharing
14. **Federated Learning** - Privacy-preserving training
15. **Import/Export** - ONNX, PyTorch, TensorFlow model formats

## Quick Reference

### File Locations

- **Main entry points**: `cyxwiz-*/src/main.cpp` or `main.rs`
- **Public API**: `cyxwiz-backend/include/cyxwiz/*.h`
- **gRPC definitions**: `cyxwiz-protocol/proto/*.proto`
- **Engine GUI panels**: `cyxwiz-engine/src/gui/panels/*.cpp`
- **Engine core**: `cyxwiz-engine/src/core/*.cpp` (ProjectManager, etc.)
- **Engine scripting**: `cyxwiz-engine/src/scripting/*.cpp`
- **Plugin system**: `cyxwiz-engine/src/plugin/` (loader, manager, context, registries, security)
- **Plugin examples**: `plugins/examples/` (mlflow_logger, image_nodes)
- **Plugin docs**: `docs/plugin_developer_guide.md`
- **Data Studio docs**: `docs/Data Studio/` (knime_comparison.md, architecture)
- **Build output**: `build/<preset>/bin/` and `build/<preset>/lib/`
- **Tests**: `tests/unit/*.cpp`
- **Resources**: `cyxwiz-engine/resources/` (fonts, icons, etc.)

### Key Commands

```bash
# Build
cmake --preset windows-release && cmake --build build/windows-release

# Test
cd build/windows-release && ctest

# Run Engine
./build/windows-release/bin/cyxwiz-engine

# Run Server Node
./build/windows-release/bin/cyxwiz-server-node

# Run Central Server
cd cyxwiz-central-server && cargo run --release

# Clean build
rm -rf build

# Install vcpkg dependencies
./vcpkg/vcpkg install
```

### Contact

For questions about the codebase architecture or design decisions, refer to:
- Architecture diagrams in `docs/architecture.md`
- Blockchain specification in `docs/blockchain.md`
- README.md for general project overview
