# Building CyxWiz

Complete build instructions for all CyxWiz components on Windows, Linux, and macOS.

## Prerequisites

### Required Tools

| Tool | Version | Windows | Linux | macOS |
|------|---------|---------|-------|-------|
| **CMake** | 3.20+ | [cmake.org](https://cmake.org/download/) | `apt install cmake` | `brew install cmake` |
| **Git** | 2.0+ | [git-scm.com](https://git-scm.com/) | `apt install git` | `brew install git` |
| **Rust** | 1.70+ | [rustup.rs](https://rustup.rs/) | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` | Same as Linux |
| **Python** | 3.8+ | [python.org](https://www.python.org/) | `apt install python3` | `brew install python` |

### C++ Compiler

| Platform | Compiler | Installation |
|----------|----------|--------------|
| **Windows** | MSVC 2022+ | Visual Studio 2022 (C++ workload) |
| **Linux** | GCC 10+ or Clang 12+ | `apt install build-essential` |
| **macOS** | Clang 12+ | `xcode-select --install` |

### Optional Dependencies

| Tool | Purpose | Installation |
|------|---------|--------------|
| **CUDA Toolkit** | NVIDIA GPU support | [developer.nvidia.com](https://developer.nvidia.com/cuda-toolkit) |
| **ArrayFire** | GPU acceleration | [arrayfire.com/download](https://arrayfire.com/download) |
| **Docker** | Containerized builds | [docker.com](https://www.docker.com/) |

## Quick Start

### Clone Repository

```bash
git clone https://github.com/CYXWIZ-Lab/CYXWIZ.git
cd CyxWiz
```

### Windows

```batch
REM From Developer Command Prompt for VS 2022

REM First-time setup
setup.bat

REM Build all components
build.bat

REM Or build release only
build.bat release
```

### Linux/macOS

```bash
# First-time setup
chmod +x scripts/*.sh
./scripts/setup.sh

# Build all components
./scripts/build.sh

# Or build release only
./scripts/build.sh release
```

## Manual Build Process

### Step 1: Install vcpkg Dependencies

```bash
# Clone vcpkg (if not present)
git clone https://github.com/microsoft/vcpkg.git

# Bootstrap vcpkg
cd vcpkg
./bootstrap-vcpkg.sh  # Linux/macOS
# or
.\bootstrap-vcpkg.bat  # Windows

# Install dependencies (reads vcpkg.json)
./vcpkg install
cd ..
```

### Step 2: Configure with CMake

```bash
# Windows Debug
cmake --preset windows-debug

# Windows Release
cmake --preset windows-release

# Linux Debug
cmake --preset linux-debug

# Linux Release
cmake --preset linux-release

# macOS Debug
cmake --preset macos-debug

# macOS Release
cmake --preset macos-release
```

### Step 3: Build

```bash
# Build configured preset
cmake --build build/<preset-name> --config Release

# Build with parallel jobs
cmake --build build/windows-release --config Release -j 8

# Build specific target
cmake --build build/windows-release --target cyxwiz-engine
```

### Step 4: Run Tests

```bash
cd build/<preset-name>
ctest --output-on-failure

# Run specific test
./bin/cyxwiz-tests "[tensor]"
```

## CMake Presets

Available presets in `CMakePresets.json`:

| Preset | Platform | Build Type | Description |
|--------|----------|------------|-------------|
| `windows-debug` | Windows | Debug | Development with symbols |
| `windows-release` | Windows | Release | Optimized production |
| `linux-debug` | Linux | Debug | Development with symbols |
| `linux-release` | Linux | Release | Optimized production |
| `macos-debug` | macOS | Debug | Development with symbols |
| `macos-release` | macOS | Release | Optimized production |
| `android-release` | Android | Release | Mobile backend only |

## Build Options

### CMake Configuration Options

```bash
# Enable/disable components
cmake --preset windows-release \
  -DCYXWIZ_BUILD_ENGINE=ON \
  -DCYXWIZ_BUILD_SERVER_NODE=ON \
  -DCYXWIZ_BUILD_TESTS=ON

# GPU backends
cmake --preset linux-release \
  -DCYXWIZ_ENABLE_CUDA=ON \
  -DCYXWIZ_ENABLE_OPENCL=ON

# Python bindings
cmake --preset linux-release \
  -DCYXWIZ_BUILD_PYTHON=ON \
  -DPython3_EXECUTABLE=/usr/bin/python3
```

### Option Reference

| Option | Default | Description |
|--------|---------|-------------|
| `CYXWIZ_BUILD_ENGINE` | ON | Build desktop client |
| `CYXWIZ_BUILD_SERVER_NODE` | ON | Build compute node |
| `CYXWIZ_BUILD_TESTS` | ON | Build test suite |
| `CYXWIZ_BUILD_PYTHON` | ON | Build Python bindings |
| `CYXWIZ_ENABLE_CUDA` | OFF | Enable CUDA backend |
| `CYXWIZ_ENABLE_OPENCL` | ON | Enable OpenCL backend |
| `CYXWIZ_ENABLE_PROFILING` | OFF | Enable profiling |

## Building Individual Components

### CyxWiz Engine Only

```bash
cmake --preset windows-release \
  -DCYXWIZ_BUILD_SERVER_NODE=OFF

cmake --build build/windows-release --target cyxwiz-engine
```

### CyxWiz Server Node Only

```bash
cmake --preset linux-release \
  -DCYXWIZ_BUILD_ENGINE=OFF

cmake --build build/linux-release --target cyxwiz-server-node
```

### CyxWiz Backend Library Only

```bash
cmake --build build/windows-release --target cyxwiz-backend
```

### Central Server (Rust)

```bash
cd cyxwiz-central-server

# Set protoc path (Windows)
set PROTOC=..\vcpkg\packages\protobuf_x64-windows\tools\protobuf\protoc.exe

# Build
cargo build --release

# Run
cargo run --release
```

## Build Artifacts

### Windows

```
build/windows-release/
├── bin/
│   ├── cyxwiz-engine.exe
│   ├── cyxwiz-server-node.exe
│   └── cyxwiz-tests.exe
├── lib/
│   ├── cyxwiz-backend.dll
│   └── cyxwiz-backend.lib
└── python/
    └── pycyxwiz.pyd
```

### Linux

```
build/linux-release/
├── bin/
│   ├── cyxwiz-engine
│   ├── cyxwiz-server-node
│   └── cyxwiz-tests
├── lib/
│   └── libcyxwiz-backend.so
└── python/
    └── pycyxwiz.so
```

## Building with ArrayFire

### Install ArrayFire

Download from [arrayfire.com/download](https://arrayfire.com/download) or:

```bash
# Ubuntu
sudo apt install arrayfire-unified3

# macOS
brew install arrayfire
```

### Set Environment

```bash
# Windows
set ArrayFire_DIR=C:\Program Files\ArrayFire\v3\cmake

# Linux
export ArrayFire_DIR=/opt/arrayfire/share/ArrayFire/cmake

# macOS
export ArrayFire_DIR=/usr/local/share/ArrayFire/cmake
```

### Configure with ArrayFire

```bash
cmake --preset linux-release \
  -DArrayFire_DIR=/opt/arrayfire/share/ArrayFire/cmake \
  -DCYXWIZ_ENABLE_CUDA=ON
```

## Building with CUDA

### Prerequisites

1. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
2. Install NVIDIA driver
3. Verify: `nvidia-smi`

### Configure

```bash
cmake --preset linux-release \
  -DCYXWIZ_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86"
```

### CUDA Architecture Reference

| GPU Generation | Architecture | CUDA Arch |
|----------------|--------------|-----------|
| RTX 20xx | Turing | 75 |
| RTX 30xx | Ampere | 86 |
| RTX 40xx | Ada Lovelace | 89 |

## Cross-Compilation

### Android (Backend Only)

```bash
# Set NDK path
export ANDROID_NDK_HOME=/path/to/android-ndk

# Configure
cmake --preset android-release \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-26

# Build
cmake --build build/android-release
```

## Troubleshooting

### Common Issues

**"ArrayFire not found"**
```bash
# Set ArrayFire_DIR or CMAKE_PREFIX_PATH
cmake --preset linux-release -DArrayFire_DIR=/opt/arrayfire/share/ArrayFire/cmake
```

**"vcpkg dependencies missing"**
```bash
cd vcpkg
./vcpkg install
```

**"gRPC generation failed"**
```bash
# Ensure protobuf is installed
./vcpkg install protobuf grpc

# Check .proto syntax
protoc --cpp_out=. cyxwiz-protocol/proto/*.proto
```

**"Python not found"**
```bash
cmake --preset linux-release -DPython3_EXECUTABLE=/usr/bin/python3.10
```

**"CUDA not found"**
```bash
# Verify CUDA installation
nvcc --version

# Set CUDA path
cmake --preset linux-release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### Clean Build

```bash
# Remove build directory
rm -rf build/

# Or clean specific preset
cmake --build build/windows-release --target clean
```

### Verbose Build

```bash
# CMake verbose
cmake --build build/windows-release --verbose

# Ninja verbose
cmake --build build/windows-release -- -v
```

## CI/CD Build

### GitHub Actions Example

```yaml
name: Build

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y build-essential cmake

      - name: Setup vcpkg
        run: |
          git clone https://github.com/microsoft/vcpkg.git
          ./vcpkg/bootstrap-vcpkg.sh
          ./vcpkg/vcpkg install

      - name: Configure
        run: cmake --preset linux-release

      - name: Build
        run: cmake --build build/linux-release -j $(nproc)

      - name: Test
        run: |
          cd build/linux-release
          ctest --output-on-failure
```

---

**Next**: [Installation](installation.md) | [Testing](testing.md)
