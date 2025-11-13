# CyxWiz - Decentralized ML Compute Platform

CyxWiz is a revolutionary platform that combines the visual design capabilities of Unreal Engine with the computational power of MATLAB, creating a decentralized network for machine learning model training and deployment.

## 🎯 Overview

The platform consists of three main components:

1. **CyxWiz Engine** - Desktop client with rich GUI for building and designing ML models
2. **CyxWiz Server Node** - Also known as miners. Distributed compute nodes that train models
3. **[CyxWiz Central Server](cyxwiz-central-server/README.md)** - Orchestrator managing the decentralized network

## ✨ Features

- **Visual Node Editor**: Drag-and-drop interface for building ML models
- **Python Scripting**: Full Python support for advanced workflows
- **Decentralized Computing**: Leverage distributed GPU/CPU resources
- **Blockchain Integration**: Solana-based token economy (CYXWIZ coin)
- **Real-time Monitoring**: Track training progress with live visualizations
- **Cross-platform**: Windows, macOS, Linux support
- **Android Backend**: Run compute backend on Android devices

## 🏗️ Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌────────────────┐
│  CyxWiz Engine  │◄───────►│ CyxWiz Central   │◄───────►│ Server Node 1  │
│  (Desktop GUI)  │  gRPC   │    Server        │  gRPC   │  (Compute)     │
└─────────────────┘         │  (Orchestrator)  │         └────────────────┘
                            └──────────────────┘                  │
                                     │                            │
                                     │         ┌──────────────────┘
                                     ▼         ▼
                            ┌──────────────────────┐
                            │  Solana Blockchain   │
                            │  (CYXWIZ Token)      │
                            └──────────────────────┘
```

## 🚀 Quick Start

### Automated Build Scripts (Recommended for Beginners)

We provide automated setup and build scripts for easy first-time setup.
These scripts are available in both the project root and the `scripts/` folder:

**Windows:**
```bash
# 1. First-time setup (installs prerequisites, bootstraps vcpkg)
setup.bat

# 2. Build all components
build.bat

# Or build specific components:
build.bat --server-node      # Build only Server Node
build.bat --engine           # Build only Engine
build.bat --central-server   # Build only Central Server
build.bat --debug            # Build in Debug mode
build.bat --clean            # Clean build directory first
build.bat -j 16              # Use 16 parallel jobs
```

**Linux/macOS:**
```bash
# 1. First-time setup
./setup.sh

# 2. Build all components
./build.sh

# Or build specific components:
./build.sh --server-node     # Build only Server Node
./build.sh --engine          # Build only Engine
./build.sh --central-server  # Build only Central Server
./build.sh --debug           # Build in Debug mode
./build.sh --clean           # Clean build directory first
./build.sh -j 16             # Use 16 parallel jobs
```

**What these scripts do:**
- `setup.bat`/`setup.sh`: Check for required tools (Visual Studio, CMake, Rust), clone and bootstrap vcpkg, explain dependencies
- `build.bat`/`build.sh`: Configure CMake, build C++ components, build Rust Central Server, show build summary

**Skip to [Running the Applications](#running-the-applications)** after building.

---

### Manual Build (For Advanced Users)

If you prefer manual control or want to understand the build process in detail:

#### Prerequisites

##### Required for All Platforms
- **C++ Compiler**:
  - Windows: Visual Studio 2022 (Community Edition or higher) with C++ Desktop Development workload
  - Linux: GCC 9+ or Clang 12+
  - macOS: Xcode Command Line Tools (Clang 12+)
- **CMake**: 3.20 or higher
- **vcpkg**: For C++ dependency management (automatically bootstrapped by scripts, or see [vcpkg setup](#understanding-vcpkg-dependency-management) below)
- **Python**: 3.8+ (for Engine scripting support)
- **Rust**: 1.70+ with Cargo (for Central Server)

##### Optional
- **ArrayFire**: GPU acceleration library (download from https://arrayfire.com/download)
  - If not installed, builds will use CPU-only mode
  - OpenCL backend supported by default

#### Understanding vcpkg Dependency Management

CyxWiz uses **vcpkg manifest mode** for automatic C++ dependency management. This means:

1. **Dependencies are declared** in `vcpkg.json` at the repository root
2. **CMake automatically downloads and builds** these dependencies during configuration
3. **Dependencies are cached** in `vcpkg/` directory for subsequent builds

**Current dependencies** (see `vcpkg.json`):
```json
{
  "dependencies": [
    {"name": "imgui", "features": ["docking-experimental", "glfw-binding", "opengl3-binding"]},
    "implot",     // Real-time plotting library
    "glfw3",      // Window/input handling
    "glad",       // OpenGL loader
    "grpc",       // Network communication
    "protobuf",   // Message serialization
    "spdlog",     // Logging library
    "nlohmann-json", // JSON parsing
    "fmt",        // String formatting
    "sqlite3",    // Database
    "openssl",    // Encryption
    "pybind11",   // Python bindings
    "catch2"      // Testing framework
  ]
}
```

**First build timing:**
- vcpkg will download and compile **34 packages** (including transitive dependencies)
- This takes **3-5 minutes** on first run
- Subsequent builds are **instant** as packages are cached

**How it works:**
1. When you run CMake with `-DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake`
2. vcpkg reads `vcpkg.json` and installs all listed dependencies
3. CMake's `find_package()` automatically finds these installed packages
4. No manual installation of libraries needed!

### Building on Windows

#### Method 1: Complete Build (All Components)

```bash
# 1. Clone the repository
git clone https://github.com/CYXWIZ-Lab/CYXWIZ.git
cd CyxWiz
# 1.1. Install vcpkg (if not already installed)
git clone https://github.com/microsoft/vcpkg
cd vcpkg
.\bootstrap-vcpkg.bat
cd ..

# 2. Configure CMake with Visual Studio generator
# This step installs 34 vcpkg dependencies (takes ~3-5 minutes)
cmake -B build/windows-release -S . ^
  -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DCYXWIZ_BUILD_ENGINE=ON ^
  -DCYXWIZ_BUILD_SERVER_NODE=ON ^
  -DCYXWIZ_BUILD_CENTRAL_SERVER=ON ^
  -DCYXWIZ_BUILD_TESTS=ON

# 3. Build all components (parallel build with 8 cores)
cmake --build build/windows-release --config Release -j 8

# 4. Build Central Server (Rust component)
cd cyxwiz-central-server

# Set PROTOC environment variable (required for protobuf compilation)
# Windows (Git Bash/MSYS):
export PROTOC="../vcpkg/packages/protobuf_x64-windows/tools/protobuf/protoc.exe"
# Windows (CMD):
# set PROTOC=%CD%\..\vcpkg\packages\protobuf_x64-windows\tools\protobuf\protoc.exe
# Windows (PowerShell):
# $env:PROTOC = "../vcpkg/packages/protobuf_x64-windows/tools/protobuf/protoc.exe"

cargo build --release
cd ..

# 5. Executables are located in:
# - build/windows-release/bin/Release/cyxwiz-engine.exe
# - build/windows-release/bin/Release/cyxwiz-server-node.exe
# - cyxwiz-central-server/target/release/cyxwiz-central-server.exe
```

**First Build Timing:**
- vcpkg dependency installation: ~3-5 minutes (34 packages)
- CMake configuration: ~3-4 minutes (protobuf generation, compiler detection)
- C++ compilation: ~2-3 minutes (protocol library + components)
- Rust compilation: ~30-60 seconds (Central Server)
- **Total: ~10-15 minutes**

Subsequent builds are much faster (1-2 minutes) as dependencies are cached.

#### Method 2: Build Individual Components

**Server Node only:**
```bash
# Configure (if not already done)
cmake -B build/windows-release -S . -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake

# Build just the Server Node
cmake --build build/windows-release --config Release --target cyxwiz-server-node -j 8
```

**Engine only:**
```bash
cmake --build build/windows-release --config Release --target cyxwiz-engine -j 8
```

**Central Server only:**
```bash
cd cyxwiz-central-server

# Set PROTOC environment variable (required)
export PROTOC="../vcpkg/packages/protobuf_x64-windows/tools/protobuf/protoc.exe"  # Git Bash/MSYS

cargo build --release
```

#### Troubleshooting Windows Build Issues

**Issue: "Ninja not found" error**
```bash
# Use Visual Studio generator instead (recommended for Windows)
cmake -G "Visual Studio 17 2022" -A x64 ...
```

**Issue: "CMAKE_CXX_COMPILER not set"**
- Ensure Visual Studio 2022 is installed with C++ Desktop Development workload
- Open "Developer Command Prompt for VS 2022" instead of regular cmd
- Or clean build directory: `rmdir /s /q build\windows-release` and reconfigure

**Issue: Port conflicts (50051, 50052, 50053 already in use)**
```bash
# Find and kill processes using these ports
netstat -ano | findstr :50051
taskkill /F /PID <process_id>
```

**Optional - ArrayFire GPU Acceleration:**
```bash
# Download ArrayFire from https://arrayfire.com/download
# Install to default location (C:\Program Files\ArrayFire)
# Set environment variable:
setx ArrayFire_DIR "C:\Program Files\ArrayFire\v3\cmake"
# Then reconfigure and rebuild
cmake -B build/windows-release -S . -G "Visual Studio 17 2022" -A x64 -DCYXWIZ_ENABLE_CUDA=ON
```

### Building on Linux/macOS

#### Linux (Ubuntu/Debian)

```bash
# 1. Install system dependencies
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git ninja-build \
  pkg-config libssl-dev \
  python3 python3-dev \
  curl

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 2. Clone the repository
git clone https://github.com/CYXWIZ-Lab/CYXWIZ.git
cd CyxWiz

# 3. Configure CMake with Ninja generator
cmake -B build/linux-release -S . -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DCYXWIZ_BUILD_ENGINE=ON \
  -DCYXWIZ_BUILD_SERVER_NODE=ON \
  -DCYXWIZ_BUILD_CENTRAL_SERVER=ON \
  -DCYXWIZ_BUILD_TESTS=ON

# 4. Build C++ components
cmake --build build/linux-release -j $(nproc)

# 5. Build Central Server (Rust)
cd cyxwiz-central-server

# Set PROTOC environment variable
export PROTOC="../vcpkg/packages/protobuf_x64-linux/tools/protobuf/protoc"  # Linux
# export PROTOC="../vcpkg/packages/protobuf_x64-osx/tools/protobuf/protoc"  # macOS

cargo build --release
cd ..

# 6. Run applications
./build/linux-release/bin/cyxwiz-engine
./build/linux-release/bin/cyxwiz-server-node
./cyxwiz-central-server/target/release/cyxwiz-central-server
```

#### macOS

```bash
# 1. Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake ninja pkg-config openssl python@3.11

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 2. Clone repository
git clone https://github.com/CYXWIZ-Lab/CYXWIZ.git
cd CyxWiz

# 3. Configure CMake
cmake -B build/macos-release -S . -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DCYXWIZ_BUILD_ENGINE=ON \
  -DCYXWIZ_BUILD_SERVER_NODE=ON \
  -DCYXWIZ_BUILD_CENTRAL_SERVER=ON

# 4. Build
cmake --build build/macos-release -j $(sysctl -n hw.ncpu)

# 5. Build Central Server
cd cyxwiz-central-server

# Set PROTOC environment variable
export PROTOC="../vcpkg/packages/protobuf_x64-osx/tools/protobuf/protoc"  # macOS

cargo build --release
cd ..

# 6. Run
./build/macos-release/bin/cyxwiz-engine
```

### Running the Applications

#### Central Server (Network Orchestrator)

**Prerequisites**:
- **PROTOC** environment variable must be set (see build instructions above)
- **Database**: SQLite (auto-created) OR PostgreSQL
- **Redis** (optional): Server runs in mock mode if unavailable
- **Solana** (optional): Payment processing disabled if keypair not found

```bash
cd cyxwiz-central-server

# Set PROTOC if not already set
export PROTOC="../vcpkg/packages/protobuf_x64-windows/tools/protobuf/protoc.exe"  # Windows
# export PROTOC="../vcpkg/packages/protobuf_x64-linux/tools/protobuf/protoc"    # Linux
# export PROTOC="../vcpkg/packages/protobuf_x64-osx/tools/protobuf/protoc"      # macOS

# Run in gRPC/REST server mode (default - for production use)
cargo run --release

# Or run in TUI-only mode (for monitoring)
cargo run --release -- --tui
```

**Default Mode (gRPC/REST Server)**:
- gRPC server on `0.0.0.0:50051` - Accepts job submissions, node registrations
- REST API on `0.0.0.0:8080` - Web dashboard and health checks
- Server Nodes can connect and register automatically

**TUI Mode** (`--tui` flag):
- Terminal dashboard shows:
  - Network statistics (nodes, jobs)
  - System health (database, Redis, Solana)
  - Job throughput graphs
  - Top nodes by reputation
- No gRPC/REST services (monitoring only)

#### Server Node (Compute Worker)
```bash
# Windows
.\build\windows-release\bin\Release\cyxwiz-server-node.exe

# Linux/macOS
./build/linux-release/bin/cyxwiz-server-node  # or macos-release

# The Server Node will:
# 1. Detect hardware (CPU/GPU) and report capabilities
# 2. Attempt to register with Central Server (localhost:50051)
# 3. Send heartbeat every 10 seconds if connected
# 4. Start local Deployment service on port 50052
# 5. Start local Terminal service on port 50053
```

**Server Node Modes:**
- **Network Mode**: Automatically activated when Central Server is reachable at `localhost:50051`
  - Node registers with unique ID and session token
  - Sends periodic heartbeat (10-second interval)
  - Receives job assignments from Central Server
  - Reports hardware capabilities (devices, memory, compute units)
- **Standalone Mode**: Falls back to this mode if Central Server is unreachable
  - Runs local deployment and terminal services only (ports 50052-50053)
  - No job assignments from network
  - Useful for local development and testing

**Success Indicators**:
- Network Mode: Look for `✓ Successfully registered with Central Server` and `Heartbeat sent successfully`
- Standalone Mode: Look for `Connection refused` warnings and `Running in standalone mode`

#### Engine (Desktop Client)
```bash
# Windows
.\build\windows-release\bin\Release\cyxwiz-engine.exe

# Linux/macOS
./build/linux-release/bin/cyxwiz-engine  # or macos-release
```

### Troubleshooting

#### General Issues

**"ArrayFire not found" warning:**
- This is **expected** and **normal**
- GPU acceleration is optional
- Project builds successfully with CPU-only mode
- To enable GPU: Install ArrayFire from https://arrayfire.com/download and set `ArrayFire_DIR`

**"vcpkg dependencies failed to install":**
```bash
# Clean vcpkg cache and retry
cd vcpkg
./vcpkg remove --outdated
./vcpkg install

# Or use binary cache
./vcpkg install --x-use-aria2
```

**CMake configuration errors:**
```bash
# Clean build directory and reconfigure
rm -rf build/windows-release  # or linux-release/macos-release
cmake -B build/windows-release -S . [... your options ...]
```

#### Platform-Specific Issues

**Linux: "Could not find OpenGL"**
```bash
sudo apt-get install libgl1-mesa-dev libglu1-mesa-dev
```

**Linux: "GLFW not found"**
```bash
sudo apt-get install libglfw3-dev libglfw3
```

**macOS: "Python.h not found"**
```bash
brew install python@3.11
# Then reconfigure with:
cmake -DPYTHON_EXECUTABLE=/usr/local/bin/python3.11 ...
```

**Rust compilation errors:**
```bash
# Update Rust to latest stable
rustup update stable

# Clear Cargo cache
cd cyxwiz-central-server
cargo clean
cargo build --release
```

#### Runtime Issues

**"Connection refused" errors:**
- Normal if Central Server isn't running
- Server Node falls back to standalone mode automatically
- Engine can still design models locally

**Port conflicts:**
```bash
# Linux/macOS: Find processes using ports
lsof -i :50051
lsof -i :50052
lsof -i :50053

# Kill process
kill -9 <PID>

# Windows: See Windows troubleshooting section above
```

## 📦 Project Structure

```
CyxWiz/
├── cyxwiz-backend/         # Shared compute library (ArrayFire-based)
├── cyxwiz-engine/          # Desktop client (ImGui + OpenGL)
│   ├── src/
│   │   ├── plotting/       # Plotting system (ImPlot + matplotlib)
│   │   └── gui/            # GUI panels and windows
│   └── python/             # Python bindings for plotting
├── cyxwiz-server-node/     # Compute worker node
├── cyxwiz-central-server/  # Orchestrator (Rust) - See README
├── cyxwiz-protocol/        # gRPC protocol definitions
├── scripts/                # Build and setup scripts (setup.bat/sh, build.bat/sh)
├── docs/                   # Documentation
├── tests/                  # Unit and integration tests
├── vcpkg/                  # Dependency management (auto-installed)
├── setup.bat / setup.sh    # Setup scripts (also in scripts/)
└── build.bat / build.sh    # Build scripts (also in scripts/)
```

## 🔧 Development

### Development Workflow

#### Quick Iteration Cycle

```bash
# 1. Make code changes to C++ components
# 2. Rebuild only changed components (fast incremental build)
cmake --build build/windows-release --config Release --target cyxwiz-server-node -j 8

# 3. For Rust changes (Central Server)
cd cyxwiz-central-server
cargo build --release  # Incremental compilation is automatic
```

#### Full Rebuild (After Major Changes)

```bash
# Clean and rebuild C++ components
rm -rf build/windows-release  # or rmdir /s /q on Windows
cmake -B build/windows-release -S . -G "Visual Studio 17 2022" -A x64 \
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build/windows-release --config Release -j 8

# Clean and rebuild Rust
cd cyxwiz-central-server
cargo clean
cargo build --release
```

#### Watching for Changes (Rust Development)

```bash
# Install cargo-watch
cargo install cargo-watch

# Auto-rebuild on file changes
cd cyxwiz-central-server
cargo watch -x 'build --release'
```

### Running Tests

#### C++ Tests (Catch2)

```bash
# Windows
cd build\windows-release
ctest --output-on-failure --config Release

# Linux/macOS
cd build/linux-release  # or macos-release
ctest --output-on-failure

# Run specific test
./bin/cyxwiz-tests "[tensor]"
```

#### Rust Tests

```bash
cd cyxwiz-central-server
cargo test
cargo test -- --nocapture  # Show stdout
cargo test --release  # Test release build
```

### Building for Android (Backend Only)

```bash
# Install Android NDK
# Set ANDROID_NDK_HOME environment variable

cmake --preset android-release
cmake --build build/android-release

# Output: cyxwiz-backend as shared library (.so)
# Location: build/android-release/lib/libcyxwiz-backend.so
```

### Central Server Development

The Central Server is a Rust application with a Terminal User Interface (TUI).

**See the [Central Server README](cyxwiz-central-server/README.md) for:**
- Detailed architecture documentation
- TUI keyboard shortcuts and navigation
- Database schema and migration guide
- gRPC service implementation guide
- Blockchain integration guide

#### Running in Development Mode

```bash
cd cyxwiz-central-server

# Run with debug logging
RUST_LOG=debug cargo run

# Run with specific log level
RUST_LOG=cyxwiz_central_server=trace cargo run

# Run tests
cargo test

# Check for lint issues
cargo clippy

# Format code
cargo fmt
```

#### Database Operations

```bash
# SQLite is used by default (no setup needed)
# Database file: cyxwiz-central-server/cyxwiz.db

# Migrations run automatically on startup
# See: src/database/migrations.rs

# To use PostgreSQL instead:
# 1. Install PostgreSQL
# 2. Create database: createdb cyxwiz
# 3. Update config.toml:
#    database.url = "postgresql://user:pass@localhost/cyxwiz"
```

#### Redis Setup (Optional)

```bash
# Install Redis (for caching)
# Windows: Download from https://github.com/microsoftarchive/redis/releases
# Linux: sudo apt-get install redis-server
# macOS: brew install redis

# Start Redis
redis-server

# Central Server will auto-connect
# Falls back to mock mode if Redis unavailable
```

### Protocol Development (gRPC)

#### Adding New Messages

```bash
# 1. Edit .proto files in cyxwiz-protocol/proto/
vim cyxwiz-protocol/proto/node.proto

# 2. Rebuild (CMake auto-regenerates code)
cmake --build build/windows-release --config Release

# 3. Generated files appear in:
# - C++: build/windows-release/cyxwiz-protocol/*.pb.{h,cc}
# - Headers are automatically included in components
```

#### Testing gRPC Services

```bash
# Use grpcurl to test endpoints
# Install: go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest

# List services
grpcurl -plaintext localhost:50051 list

# Call method
grpcurl -plaintext -d '{"node_id": "test"}' \
  localhost:50051 cyxwiz.protocol.NodeService/Heartbeat
```

### Adding C++ Dependencies with vcpkg

This guide shows how to add new C++ libraries to the project using vcpkg manifest mode.

#### Example: How ImGui Was Added

Let's walk through how ImGui (with docking support) was integrated into CyxWiz. You can follow the same process for any vcpkg package.

##### Step 1: Search for the Package

```bash
# Search vcpkg for available packages
cd vcpkg
./vcpkg search imgui

# Output:
# imgui                 1.90.1           Bloat-free Immediate Mode Graphical User interface for C++ with minimal dependencies
# imgui[docking-experimental]            Experimental docking branch
# imgui[freetype]                        FreeType Font Renderer for Dear ImGui
# imgui[glfw-binding]                    Make available GLFW binding
# imgui[opengl3-binding]                 Make available OpenGL3/ES/ES2 (modern) binding
# ...

# Check package details
./vcpkg info imgui
```

##### Step 2: Add to vcpkg.json

Edit `vcpkg.json` in the repository root:

```json
{
  "name": "cyxwiz",
  "version": "0.1.0",
  "dependencies": [
    {
      "name": "imgui",
      "features": [
        "docking-experimental",  // Enable docking windows
        "glfw-binding",          // GLFW integration
        "opengl3-binding"        // OpenGL3 rendering backend
      ]
    },
    // ... other dependencies
  ]
}
```

**Key points:**
- Use `{"name": "...", "features": [...]}` syntax to specify optional features
- Use simple string `"package-name"` for packages without features
- Features are optional components provided by the package
- Check available features with: `./vcpkg search imgui`

##### Step 3: Add find_package() in CMakeLists.txt

Edit the CMakeLists.txt of the component that needs ImGui (e.g., `cyxwiz-engine/CMakeLists.txt`):

```cmake
# Find the package (vcpkg makes it available)
find_package(imgui CONFIG REQUIRED)

# Add your executable or library
add_executable(cyxwiz-engine
    src/main.cpp
    src/gui/main_window.cpp
    # ... other sources
)

# Link against the package
target_link_libraries(cyxwiz-engine PRIVATE
    imgui::imgui  # Main ImGui library
)

# If you need specific bindings:
# target_link_libraries(cyxwiz-engine PRIVATE
#     imgui::imgui
#     glfw
#     glad
# )
```

**Package naming conventions:**
- Most packages use `PackageName::PackageName` (e.g., `imgui::imgui`, `spdlog::spdlog`)
- Some have multiple targets (e.g., `protobuf::libprotobuf`, `protobuf::libprotoc`)
- Check CMake integration docs: `./vcpkg info imgui` or look in `vcpkg/ports/imgui/`

##### Step 4: Include Headers in Your Code

```cpp
// In your C++ source file (e.g., cyxwiz-engine/src/main.cpp)
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

int main() {
    // ImGui context setup
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;  // Enable docking

    // ... rest of your code
}
```

##### Step 5: Rebuild the Project

```bash
# Clean build to download new dependency
rm -rf build/windows-release  # Windows: rmdir /s /q build\windows-release

# Reconfigure (vcpkg installs imgui automatically)
cmake -B build/windows-release -S . -G "Visual Studio 17 2022" -A x64 \
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake

# Build
cmake --build build/windows-release --config Release -j 8
```

**What happens:**
1. CMake reads `vcpkg.json` and tells vcpkg to install `imgui` with specified features
2. vcpkg downloads, compiles, and caches ImGui (takes ~30 seconds)
3. `find_package(imgui CONFIG REQUIRED)` locates the installed package
4. `target_link_libraries()` adds include paths and links the library
5. Your code can now `#include <imgui.h>` and use ImGui functions

#### Adding Other Packages - Quick Reference

**Example 1: Adding Boost (header-only)**
```json
// vcpkg.json
{
  "dependencies": ["boost-asio", "boost-filesystem"]
}
```
```cmake
# CMakeLists.txt
find_package(Boost REQUIRED COMPONENTS system filesystem)
target_link_libraries(my-app PRIVATE Boost::system Boost::filesystem)
```

**Example 2: Adding OpenCV**
```json
// vcpkg.json
{
  "dependencies": [
    {"name": "opencv4", "features": ["jpeg", "png", "tiff"]}
  ]
}
```
```cmake
# CMakeLists.txt
find_package(OpenCV CONFIG REQUIRED)
target_link_libraries(my-app PRIVATE opencv_core opencv_imgproc opencv_highgui)
```

**Example 3: Adding cURL**
```json
// vcpkg.json
{
  "dependencies": ["curl"]
}
```
```cmake
# CMakeLists.txt
find_package(CURL REQUIRED)
target_link_libraries(my-app PRIVATE CURL::libcurl)
```

#### Troubleshooting Package Addition

**Error: "Could not find a package configuration file provided by X"**
```bash
# Check if package is in vcpkg.json
cat vcpkg.json

# Clean and reconfigure
rm -rf build/windows-release
cmake -B build/windows-release -S . -G "Visual Studio 17 2022" -A x64 \
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake
```

**Error: "No such file or directory" when including headers**
```cmake
# Make sure you linked the library
target_link_libraries(your-target PRIVATE package::package)

# Some packages need explicit include directories
target_include_directories(your-target PRIVATE ${PACKAGE_INCLUDE_DIRS})
```

**Package not found in vcpkg**
```bash
# Update vcpkg to latest version
cd vcpkg
git pull
./bootstrap-vcpkg.sh  # or .bat on Windows

# Search again
./vcpkg search <package-name>

# If still not found, check community ports:
# https://github.com/microsoft/vcpkg/tree/master/ports
```

#### Best Practices for Dependency Management

1. **Always specify features explicitly** - Makes build reproducible
2. **Pin vcpkg baseline** - Use `builtin-baseline` in vcpkg.json to lock versions
3. **Test on clean machine** - Run `setup.bat`/`setup.sh` scripts to verify
4. **Document why dependencies are needed** - Add comments in vcpkg.json
5. **Minimize dependencies** - Only add what you actually use
6. **Check license compatibility** - Use `./vcpkg info <package>` to see license

#### Updating Dependencies

```bash
# Update vcpkg itself
cd vcpkg
git pull
./bootstrap-vcpkg.sh  # or .bat

# Update all packages to latest baseline
cd ..
rm -rf build/
cmake -B build/windows-release -S . -G "Visual Studio 17 2022" -A x64 \
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake
```

#### Useful vcpkg Commands

```bash
cd vcpkg

# List installed packages
./vcpkg list

# Search for packages
./vcpkg search <keyword>

# Get package details
./vcpkg info <package-name>

# Remove package cache (force rebuild)
./vcpkg remove --outdated

# Export installed packages (for CI/CD)
./vcpkg export --zip
```

## 📚 Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api.md)
- [Blockchain Integration](docs/blockchain.md)
- [Contributing Guide](docs/CONTRIBUTING.md)

## 🌐 Blockchain

CyxWiz uses Solana for its decentralized token economy:

- **Token**: CYXWIZ (SPL Token)
- **Use Cases**: Compute payments, node staking, governance
- **Smart Contracts**: Job escrow, payment streaming, rewards

See [Blockchain Architecture](docs/blockchain.md) for details.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Links

- Website: https://cyxwiz.io
- Documentation: https://docs.cyxwiz.io
- Discord: https://discord.gg/cyxwiz
- Twitter: https://twitter.com/cyxwiz

## 🙏 Acknowledgments

- ArrayFire for GPU acceleration
- Dear ImGui for the GUI framework
- gRPC for network communication
- Solana for blockchain infrastructure

---

## 📋 Quick Reference

### Automated Build Scripts (Easiest)

```bash
# Windows - First time setup and build
setup.bat      # Install prerequisites, bootstrap vcpkg
build.bat      # Build all components
build.bat --server-node   # Build only Server Node
build.bat --clean         # Clean build

# Linux/macOS - First time setup and build
./setup.sh     # Install prerequisites, bootstrap vcpkg
./build.sh     # Build all components
./build.sh --server-node  # Build only Server Node
./build.sh --clean        # Clean build
```

### Manual Build Commands

```bash
# Full build (first time)
cmake -B build/windows-release -S . -G "Visual Studio 17 2022" -A x64 \
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build/windows-release --config Release -j 8
cd cyxwiz-central-server && cargo build --release && cd ..

# Incremental build (after changes)
cmake --build build/windows-release --config Release --target cyxwiz-server-node -j 8

# Clean build
rm -rf build/windows-release  # Windows: rmdir /s /q build\windows-release
```

### Running Components

```bash
# Central Server (Terminal UI)
cd cyxwiz-central-server && cargo run --release

# Server Node (Compute Worker)
# Windows: .\build\windows-release\bin\Release\cyxwiz-server-node.exe
# Linux:   ./build/linux-release/bin/cyxwiz-server-node

# Engine (Desktop Client)
# Windows: .\build\windows-release\bin\Release\cyxwiz-engine.exe
# Linux:   ./build/linux-release/bin/cyxwiz-engine
```

### Useful Development Commands

```bash
# Run tests
cd build/windows-release && ctest --output-on-failure
cd cyxwiz-central-server && cargo test

# Format code
cd cyxwiz-central-server && cargo fmt

# Check for issues
cd cyxwiz-central-server && cargo clippy

# View logs with colors
cd cyxwiz-central-server && RUST_LOG=debug cargo run 2>&1 | less -R

# Kill background processes on ports
# Windows: taskkill /F /PID <pid>
# Linux:   kill -9 <pid>
```

### Build Timings (Reference)

| Stage | First Build | Incremental |
|-------|-------------|-------------|
| vcpkg dependencies | 3-5 min | 0 sec (cached) |
| CMake configuration | 3-4 min | 5-10 sec |
| C++ compilation | 2-3 min | 30-60 sec |
| Rust compilation | 30-60 sec | 5-10 sec |
| **Total** | **10-15 min** | **1-2 min** |

### Port Reference

| Service | Port | Protocol |
|---------|------|----------|
| Central Server gRPC | 50051 | gRPC/HTTP2 |
| Central Server REST | 8080 | HTTP |
| Server Node Deployment | 50052 | gRPC/HTTP2 |
| Server Node Terminal | 50053 | gRPC/HTTP2 |

### Component Sizes (Approximate)

| Component | Lines of Code | Dependencies |
|-----------|---------------|--------------|
| cyxwiz-backend | ~5,000 LOC | ArrayFire, Python |
| cyxwiz-engine | ~3,000 LOC | ImGui, OpenGL, GLFW |
| cyxwiz-server-node | ~2,500 LOC | gRPC, protobuf |
| cyxwiz-central-server | ~4,000 LOC | Tokio, SQLx, Redis |
| cyxwiz-protocol | ~1,000 LOC | protobuf |
| **Total** | **~15,500 LOC** | 34 vcpkg packages |
