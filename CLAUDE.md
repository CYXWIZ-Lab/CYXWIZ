# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

### cyxwiz-protocol/ (gRPC Definitions)

**Purpose**: Shared protocol definitions for all components

**Files**:
- `proto/common.proto` - Common types (StatusCode, DeviceType, TensorInfo)
- `proto/job.proto` - Job submission, status, results
- `proto/node.proto` - Node registration, heartbeat, metrics
- `proto/compute.proto` - Direct compute operations

**Generated Code**: CMake automatically generates C++ code from `.proto` files into `build/<preset>/`

**Adding New Messages**:
1. Edit `.proto` file in `cyxwiz-protocol/proto/`
2. CMake will regenerate code on next build
3. Include generated header: `#include "job.pb.h"` or `#include "node.grpc.pb.h"`

### cyxwiz-backend/ (Compute Library)

**Purpose**: Core ML algorithms and ArrayFire integration

**Structure**:
```
include/cyxwiz/    # Public API headers
    cyxwiz.h       # Main header (include this)
    tensor.h       # Tensor operations
    device.h       # Device management
    optimizer.h    # Optimizers (SGD, Adam, AdamW)
    loss.h         # Loss functions
    activation.h   # Activation functions
    layer.h        # Neural network layers
    model.h        # Model training/inference

src/core/          # Core implementation
src/algorithms/    # ML algorithms
python/            # Python bindings (pybind11)
```

**Key Classes**:
- `Tensor`: Multi-dimensional array (wraps ArrayFire array)
- `Device`: GPU/CPU device abstraction
- `Optimizer`: Base class for optimizers (SGD, Adam, AdamW, RMSprop)
- `Layer`: Base class for NN layers
- `Model`: High-level training interface

**ArrayFire Integration**:
- Backend selection: CPU, CUDA, OpenCL, Metal
- Conditional compilation: `#ifdef CYXWIZ_HAS_ARRAYFIRE`
- Device management: `af::setDevice()`, `af::info()`
- Tensor wrapping: Internal `af::array*` pointer

**Python Bindings**:
- Module name: `pycyxwiz`
- Built with pybind11
- Install location: `build/<preset>/python/`
- Usage: `import pycyxwiz; pycyxwiz.initialize()`

### cyxwiz-engine/ (Desktop Client)

**Purpose**: Visual IDE for building and training ML models

**Structure**:
```
src/
    main.cpp              # Entry point
    application.cpp       # Main application loop, GLFW window management
    core/
        project_manager.cpp/h  # Singleton for project state management
        async_task_manager.cpp/h  # Background task execution with progress callbacks
        data_registry.cpp/h   # Dataset registry with LRU memory management
    gui/
        main_window.cpp   # Dockable main window with all panels
        node_editor.h     # Visual node editor (TODO: integrate ImNodes)
        console.h         # Command console with Python REPL
        viewport.h        # Training visualization
        properties.h      # Property panel
        theme.cpp/h       # Theme system with multiple presets
        dock_style.cpp/h  # Custom dock styling
        icons.h           # FontAwesome 6 icon definitions
        IconsFontAwesome6.h  # FontAwesome 6 codepoints
        panels/
            toolbar.cpp/h     # Main menu bar (File, Edit, View, etc.)
            asset_browser.cpp/h  # Project file browser with filters
            plot_window.cpp/h    # ImPlot-based visualization windows
            plot_test_control.cpp/h  # Plot testing interface
            table_viewer.cpp/h   # Multi-tab data table viewer with async loading
            dataset_panel.cpp/h  # Dataset manager with memory management
            script_editor.cpp/h  # Code editor with async file loading
    scripting/
        scripting_engine.cpp/h  # Embedded Python interpreter with pybind11
    network/
        grpc_client.cpp   # gRPC client for Central Server
        job_manager.cpp   # Job submission and monitoring
```

**ImGui Integration**:
- Docking enabled: `ImGuiConfigFlags_DockingEnable`
- Viewports enabled: `ImGuiConfigFlags_ViewportsEnable`
- Backend: `imgui_impl_glfw` + `imgui_impl_opengl3`
- Themes: Dark, Light, Classic, Nord, Dracula (configurable via View menu)
- Icons: FontAwesome 6 Free Solid

**Implemented Features**:
- **ProjectManager**: Singleton managing project lifecycle (create, open, save, close)
- **File Menu**: New/Open/Close Project, New/Open Script, Save/Save As, Auto Save, Recent Projects, Exit with confirmation
- **Asset Browser**: File tree with filters (Scripts, Models, Datasets, etc.), context menus, "View in Table" for data files
- **Script Editor**: Python/CyxWiz script editing with syntax highlighting, save confirmation dialogs, async file loading with progress indicators
- **Python Console**: Interactive REPL with async execution and cancellation support
- **Theme System**: Multiple color presets, custom fonts (Inter, JetBrains Mono)
- **Account Settings**: Login/logout UI (placeholder for auth API)
- **Table Viewer**: Multi-tab data viewer for CSV/Excel/HDF5 files with async loading, pagination, filtering, and export
- **Dataset Manager**: Dataset configuration with memory management (LRU eviction via TrimMemory), schema configuration for custom data formats
- **Async Task System**: Background task execution with progress reporting via AsyncTaskManager

**TODO Features** (marked in code):
- ImNodes integration for visual node editor
- Import/Export model formats (ONNX, PyTorch, TensorFlow)
- Training controls (Start, Pause, Stop)
- Server connection and job submission
- Preferences/Settings dialog

### cyxwiz-server-node/ (Compute Worker)

**Purpose**: Execute ML training jobs on local hardware

**Structure**:
```
src/
    main.cpp              # Entry point
    node_server.cpp       # gRPC server (TODO)
    job_executor.cpp      # Job execution engine (TODO)
    metrics_collector.cpp # Resource monitoring (TODO)
```

**TODO Implementations**:
- gRPC server for receiving jobs from Central Server
- Job execution using `cyxwiz-backend`
- Docker/container support for sandboxing
- btop library integration for TUI monitoring
- Metrics collection (CPU, GPU, memory, network)
- Heartbeat mechanism to Central Server

### cyxwiz-central-server/ (Orchestrator - Rust)

**Purpose**: Coordinate the decentralized network

**Structure**:
```rust
src/
    main.rs           # Entry point
    // TODO: Add modules
    api/              # gRPC server implementation
    scheduler/        # Job scheduling logic
    database/         # PostgreSQL/SQLite access
    cache/            # Redis integration
    blockchain/       # Solana connector
```

**Dependencies** (Cargo.toml):
- `tonic` - gRPC framework
- `sqlx` - Database access
- `redis` - Caching
- `solana-sdk` - Blockchain integration
- `tokio` - Async runtime

**TODO Implementations**:
- gRPC service implementations (JobService, NodeService)
- Node registry and discovery
- Job scheduler (match jobs to nodes)
- Payment processor (Solana integration)
- Metrics and monitoring
- RESTful API for web dashboard

## Common Development Tasks

### Adding a New ML Algorithm

1. **Define Interface** in `cyxwiz-backend/include/cyxwiz/<name>.h`
2. **Implement** in `cyxwiz-backend/src/algorithms/<name>.cpp`
3. **Add to CMakeLists.txt** in `cyxwiz-backend/CMakeLists.txt`
4. **Write Tests** in `tests/unit/test_<name>.cpp`
5. **Expose to Python** in `cyxwiz-backend/python/bindings.cpp`

**Example - Adding a new optimizer**:
```cpp
// In optimizer.h
class MyOptimizer : public Optimizer {
public:
    MyOptimizer(double lr);
    void Step(...) override;
    void ZeroGrad() override;
};

// In optimizer.cpp
MyOptimizer::MyOptimizer(double lr) {
    learning_rate_ = lr;
}

// In CreateOptimizer factory
case OptimizerType::MyOptimizer:
    return std::make_unique<MyOptimizer>(learning_rate);
```

### Adding a New gRPC Service

1. **Define Message** in `cyxwiz-protocol/proto/<name>.proto`
2. **Define Service** in same file with `service` keyword
3. **Rebuild** - CMake regenerates code automatically
4. **Implement Server** (Rust in Central Server or C++ in Server Node)
5. **Implement Client** (C++ in Engine)

**Example**:
```protobuf
// In proto/example.proto
service ExampleService {
    rpc DoSomething(Request) returns (Response);
}

message Request {
    string data = 1;
}

message Response {
    string result = 1;
}
```

### Adding a New GUI Panel

1. **Create header** `cyxwiz-engine/src/gui/<name>.h`
2. **Create implementation** `cyxwiz-engine/src/gui/<name>.cpp`
3. **Add to CMakeLists.txt**
4. **Integrate in MainWindow**:
```cpp
// In main_window.h
std::unique_ptr<MyPanel> my_panel_;

// In main_window.cpp constructor
my_panel_ = std::make_unique<MyPanel>();

// In Render()
if (my_panel_) my_panel_->Render();
```

### Debugging

**Debug Build Macros**:
- `CYXWIZ_DEBUG` - Defined in debug builds
- `CYXWIZ_ENABLE_LOGGING` - Enables verbose logging
- `CYXWIZ_ENABLE_PROFILING` - Enables performance profiling

**Logging** (spdlog):
```cpp
#include <spdlog/spdlog.h>

spdlog::debug("Debug message: {}", value);
spdlog::info("Info message");
spdlog::warn("Warning");
spdlog::error("Error: {}", error_msg);
```

**Memory Tracking** (Debug Mode):
```cpp
#ifdef CYXWIZ_DEBUG
    size_t allocated = cyxwiz::MemoryManager::GetAllocatedBytes();
    size_t peak = cyxwiz::MemoryManager::GetPeakBytes();
#endif
```

## External Dependencies

### Required (Must Install Manually)

**ArrayFire** - GPU acceleration library
- Download: https://arrayfire.com/download
- Backends: CUDA (NVIDIA), OpenCL (AMD/Intel), CPU
- Installation: Set `ArrayFire_DIR` environment variable or install to standard path
- CMake will warn if not found and build without GPU support

### Managed by vcpkg

All other C++ dependencies are installed via vcpkg:
- imgui (with docking, GLFW, OpenGL3)
- glfw3, glad
- grpc, protobuf
- spdlog, fmt, nlohmann-json
- sqlite3, openssl
- pybind11, catch2, boost

**Setup vcpkg**:
```bash
git clone https://github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.sh  # or bootstrap-vcpkg.bat on Windows
```

**Install dependencies**:
```bash
./vcpkg install  # Reads vcpkg.json
```

### Optional Libraries (TODO in code)

**ImNodes** - Visual node editor
- Source: https://github.com/Nelarius/imnodes
- Integration: Manual (not in vcpkg)
- Used in: `cyxwiz-engine/src/gui/node_editor.cpp`

**ImPlot** - Real-time plotting
- Source: https://github.com/epezent/implot
- Integration: May be in vcpkg, check with `vcpkg search implot`
- Used in: `cyxwiz-engine/src/gui/viewport.cpp`

**btop** - Terminal UI for resource monitoring
- Source: https://github.com/aristocratos/btop
- Integration: Library extraction needed
- Used in: `cyxwiz-server-node/src/metrics_collector.cpp`

## Blockchain Integration

### Solana Setup (Central Server)

**Dependencies** (in Cargo.toml):
```toml
solana-sdk = "1.17"
solana-client = "1.17"
```

**Smart Contracts** (Rust/Anchor):
- Location: `cyxwiz-blockchain/` (TODO: create this directory)
- Programs: JobEscrow, PaymentStreaming, NodeStaking
- Deploy: `solana program deploy <program.so>`

**Token**: CYXWIZ (SPL Token)
- Standard: SPL Token (Solana)
- Bridge: Wormhole for Polygon interoperability

### Payment Flow

1. User submits job → Engine calls Central Server gRPC
2. Central Server creates escrow on Solana
3. Job assigned to Server Node
4. Node executes job, reports progress
5. On completion, payment released from escrow
6. 90% to Node, 10% to platform

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

1. **ImNodes Integration** - Visual node editor in Engine (basic structure ready)
2. ~~**ImPlot Integration**~~ - Real-time training plots (DONE - PlotWindow implemented)
3. **btop Integration** - Server Node monitoring TUI
4. **gRPC Services** - Full implementation of all services
5. **Job Execution** - Complete job executor in Server Node
6. **Blockchain Integration** - Solana payment processor
7. **Authentication** - JWT tokens for gRPC (UI ready, API integration TODO)
8. **Docker Support** - Containerized job execution
9. **Model Marketplace** - NFT-based model sharing
10. **Federated Learning** - Privacy-preserving training
11. **Import/Export** - ONNX, PyTorch, TensorFlow model formats
12. **Training Controls** - Start, Pause, Stop training with real-time feedback

## Quick Reference

### File Locations

- **Main entry points**: `cyxwiz-*/src/main.cpp` or `main.rs`
- **Public API**: `cyxwiz-backend/include/cyxwiz/*.h`
- **gRPC definitions**: `cyxwiz-protocol/proto/*.proto`
- **Engine GUI panels**: `cyxwiz-engine/src/gui/panels/*.cpp`
- **Engine core**: `cyxwiz-engine/src/core/*.cpp` (ProjectManager, etc.)
- **Engine scripting**: `cyxwiz-engine/src/scripting/*.cpp`
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
