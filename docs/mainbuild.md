# CyxWiz Complete Build Guide

This guide provides detailed instructions for building the entire CyxWiz platform from source. For a quick overview, see [README.md](../README.md).

## Build Time Expectations

| Method | Estimated Time | Notes |
|--------|---------------|-------|
| **Pre-built Binaries** | 15-25 minutes | Recommended for most users |
| **Complete Source Build** | 2-4 hours | First-time build, includes all dependencies |
| **Incremental Build** | 1-5 minutes | After initial build |

## Prerequisites

### All Platforms
- **CMake** 3.20 or higher
- **Git** for cloning repositories
- **Python** 3.8+ (for scripting support)
- **Internet connection** (for downloading dependencies)

### Windows
- **Visual Studio 2022** (Community or higher) with "Desktop development with C++" workload
- **Windows SDK** 10.0.19041.0 or higher

### Linux
- **GCC** 11+ or **Clang** 14+
- **Development packages**: `build-essential`, `pkg-config`, `libgl1-mesa-dev`, `libxrandr-dev`, `libxinerama-dev`, `libxcursor-dev`, `libxi-dev`

### macOS
- **Xcode** 14+ or Command Line Tools
- **Homebrew** (recommended for dependencies)

---

## Option 1: Fast Build with Pre-built Binaries (Recommended)

This method uses vcpkg's binary caching to download pre-compiled dependencies instead of building them from source. This is **significantly faster** (15-25 minutes vs 2-4 hours).

### Step 1: Clone the Repository

```bash
git clone https://github.com/cyxwiz/cyxwiz.git
cd cyxwiz
```

### Step 2: Set Up vcpkg with Binary Caching

#### Windows (PowerShell)
```powershell
# Clone vcpkg
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Enable binary caching (uses GitHub packages or local cache)
$env:VCPKG_BINARY_SOURCES = "clear;default,readwrite"

# Or use a local cache directory
$env:VCPKG_DEFAULT_BINARY_CACHE = "C:\vcpkg-cache"
mkdir C:\vcpkg-cache -Force

cd ..
```

#### Linux/macOS
```bash
# Clone vcpkg
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh

# Enable binary caching
export VCPKG_BINARY_SOURCES="clear;default,readwrite"

# Or use a local cache directory
export VCPKG_DEFAULT_BINARY_CACHE="$HOME/.vcpkg-cache"
mkdir -p "$HOME/.vcpkg-cache"

cd ..
```

### Step 3: Configure and Build

#### Windows
```powershell
# Configure with vcpkg toolchain
cmake -B build -S . `
    -DCMAKE_TOOLCHAIN_FILE="./vcpkg/scripts/buildsystems/vcpkg.cmake" `
    -DCMAKE_BUILD_TYPE=Release

# Build (use all CPU cores)
cmake --build build --config Release --parallel
```

#### Linux/macOS
```bash
# Configure with vcpkg toolchain
cmake -B build -S . \
    -DCMAKE_TOOLCHAIN_FILE="./vcpkg/scripts/buildsystems/vcpkg.cmake" \
    -DCMAKE_BUILD_TYPE=Release

# Build (use all CPU cores)
cmake --build build --config Release --parallel
```

### Step 4: Run

```bash
# Engine (Desktop Client)
./build/bin/Release/cyxwiz-engine

# Server Node
./build/bin/Release/cyxwiz-server-node
```

---

## Option 2: Complete Source Build

> **Warning**: Building all dependencies from source can take **2-4 hours** on a typical machine. This includes compiling gRPC, protobuf, OpenSSL, and other large libraries.

Use this method if:
- You need to modify dependencies
- Pre-built binaries aren't available for your platform
- You want reproducible builds

### Step 1: Clone and Set Up vcpkg

```bash
git clone https://github.com/cyxwiz/cyxwiz.git
cd cyxwiz
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh  # or bootstrap-vcpkg.bat on Windows
cd ..
```

### Step 2: Install Dependencies (This Takes Time)

```bash
# Install all dependencies from source
./vcpkg/vcpkg install --triplet x64-windows  # Windows
./vcpkg/vcpkg install --triplet x64-linux    # Linux
./vcpkg/vcpkg install --triplet x64-osx      # macOS
```

**Expected build times for major dependencies:**
- gRPC + protobuf: ~45 minutes
- OpenSSL: ~15 minutes
- Boost: ~20 minutes
- ImGui + backends: ~5 minutes
- Other dependencies: ~30 minutes

### Step 3: Configure and Build

```bash
cmake -B build -S . \
    -DCMAKE_TOOLCHAIN_FILE="./vcpkg/scripts/buildsystems/vcpkg.cmake" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release --parallel
```

---

## Option 3: Using Ninja for Faster Builds

Ninja is a build system focused on speed. It can be **30-50% faster** than MSBuild (Windows) or Make (Linux/macOS) for incremental builds.

### Install Ninja

#### Windows
```powershell
# Using Chocolatey
choco install ninja

# Or using Scoop
scoop install ninja

# Or download from https://ninja-build.org/
```

#### Linux
```bash
# Ubuntu/Debian
sudo apt install ninja-build

# Fedora
sudo dnf install ninja-build

# Arch
sudo pacman -S ninja
```

#### macOS
```bash
brew install ninja
```

### Build with Ninja

```bash
# Configure with Ninja generator
cmake -B build -S . -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="./vcpkg/scripts/buildsystems/vcpkg.cmake" \
    -DCMAKE_BUILD_TYPE=Release

# Build (Ninja automatically uses all cores)
cmake --build build
```

### Ninja + Pre-built Binaries (Fastest Option)

Combine Ninja with vcpkg binary caching for the fastest possible build:

```bash
# Set up binary caching
export VCPKG_BINARY_SOURCES="clear;default,readwrite"

# Configure with Ninja
cmake -B build -S . -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="./vcpkg/scripts/buildsystems/vcpkg.cmake" \
    -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build
```

---

## Build Configuration Options

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `CYXWIZ_BUILD_ENGINE` | ON | Build the desktop client |
| `CYXWIZ_BUILD_SERVER_NODE` | ON | Build the compute node |
| `CYXWIZ_BUILD_CENTRAL_SERVER` | ON | Build the orchestrator |
| `CYXWIZ_BUILD_TESTS` | ON | Build unit tests |
| `CYXWIZ_ENABLE_CUDA` | OFF | Enable CUDA backend |
| `CYXWIZ_ENABLE_OPENCL` | OFF | Enable OpenCL backend |

### Examples

**Build only the Engine:**
```bash
cmake -B build -S . \
    -DCYXWIZ_BUILD_SERVER_NODE=OFF \
    -DCYXWIZ_BUILD_CENTRAL_SERVER=OFF \
    -DCMAKE_BUILD_TYPE=Release
```

**Build with CUDA support:**
```bash
cmake -B build -S . \
    -DCYXWIZ_ENABLE_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release
```

**Debug build:**
```bash
cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Debug
```

---

## Build Optimization Tips

### 1. Use a Fast SSD
Building on an SSD vs HDD can reduce build times by 50% or more due to the many small file operations.

### 2. Increase Parallel Jobs
```bash
# Use all available cores
cmake --build build --parallel

# Or specify exact number
cmake --build build --parallel 8
```

### 3. Use ccache (Linux/macOS)

ccache caches compilation results to speed up rebuilds:

```bash
# Install
sudo apt install ccache  # Linux
brew install ccache      # macOS

# Configure CMake to use ccache
cmake -B build -S . \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
```

### 4. Use sccache (Windows)

```powershell
# Install
cargo install sccache

# Configure
$env:CMAKE_C_COMPILER_LAUNCHER = "sccache"
$env:CMAKE_CXX_COMPILER_LAUNCHER = "sccache"
```

### 5. Disable Unused Components
If you only need the Engine, disable Server Node and Central Server builds to save time.

---

## Platform-Specific Instructions

### Windows with Visual Studio

```powershell
# Use CMake presets
cmake --preset windows-release
cmake --build --preset windows-release

# Or open in Visual Studio
cmake -B build -G "Visual Studio 17 2022" -A x64
start build\CyxWiz.sln
```

### Linux

```bash
# Install dependencies
sudo apt update
sudo apt install build-essential cmake ninja-build pkg-config \
    libgl1-mesa-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev

# Build
cmake --preset linux-release
cmake --build --preset linux-release
```

### macOS

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install dependencies via Homebrew
brew install cmake ninja

# Build
cmake --preset macos-release
cmake --build --preset macos-release
```

---

## Troubleshooting

### "vcpkg dependencies failed to install"

```bash
# Clear vcpkg cache and retry
rm -rf vcpkg/installed vcpkg/buildtrees
./vcpkg/vcpkg install
```

### "CMake can't find package X"

Ensure vcpkg toolchain is set:
```bash
cmake -B build -DCMAKE_TOOLCHAIN_FILE="./vcpkg/scripts/buildsystems/vcpkg.cmake" ...
```

### "Build runs out of memory"

Limit parallel jobs:
```bash
cmake --build build --parallel 2
```

### "Ninja not found"

Ensure Ninja is in your PATH, or specify the full path:
```bash
cmake -B build -G Ninja -DCMAKE_MAKE_PROGRAM=/path/to/ninja
```

### "gRPC build fails on Windows"

Ensure you have Windows SDK 10.0.19041.0 or higher installed via Visual Studio Installer.

### "ArrayFire not found"

Install ArrayFire manually from https://arrayfire.com/download and set:
```bash
cmake -B build -DArrayFire_DIR=/path/to/arrayfire/share/ArrayFire/cmake
```

---

## Central Server (Rust)

The Central Server is built separately using Cargo:

```bash
cd cyxwiz-central-server

# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Run
cargo run --release
```

---

## Verifying the Build

After building, verify everything works:

```bash
# Check executables exist
ls build/bin/Release/

# Run tests
cd build
ctest --output-on-failure

# Start the Engine
./build/bin/Release/cyxwiz-engine
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Configure (Release) | `cmake -B build -DCMAKE_BUILD_TYPE=Release` |
| Build | `cmake --build build --config Release --parallel` |
| Build with Ninja | `cmake -B build -G Ninja && cmake --build build` |
| Run tests | `cd build && ctest` |
| Clean build | `rm -rf build && cmake -B build ...` |
| Rebuild single target | `cmake --build build --target cyxwiz-engine` |

---

For more information about the project architecture and development workflow, see [CLAUDE.md](../CLAUDE.md).
