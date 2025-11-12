# Building CyxWiz

## Current Build Status

✅ **Setup Complete:**
- vcpkg dependency manager installed
- Project structure created
- C API (`extern "C"`) wrapper added
- Build configuration files ready

🔄 **In Progress:**
- Installing C++ dependencies (15-30 minutes)

## Prerequisites

### Required
- **CMake** 3.20+ - https://cmake.org/download
- **C++ Compiler**:
  - Windows: Visual Studio 2019+ (MSVC) or MinGW-w64
  - Linux: GCC 9+ or Clang 12+
  - macOS: Xcode Command Line Tools (Clang 12+)
- **Python** 3.8+ - For scripting support
- **Git** - For cloning dependencies

### Optional
- **ArrayFire** - GPU acceleration (CUDA/OpenCL/Metal)
  - Download: https://arrayfire.com/download
  - Without this, backend builds in CPU-only mode
- **Ninja** - Faster builds (CMake can use it)
- **Rust** 1.70+ - For Central Server component

## Build Steps

### 1. Install vcpkg (Done ✅)

```bash
git clone https://github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.sh  # or bootstrap-vcpkg.bat on Windows
```

### 2. Install Dependencies (In Progress 🔄)

```bash
cd vcpkg
./vcpkg install  # Reads vcpkg.json
```

**Note:** This takes 15-30 minutes as it compiles everything from source.

**Installing:**
- ImGui (with docking, GLFW, OpenGL3 backends)
- GLFW3 (window management)
- GLAD (OpenGL loader)
- gRPC + Protobuf (networking)
- spdlog (logging)
- fmt (formatting)
- nlohmann-json (JSON)
- SQLite3 (database)
- OpenSSL (crypto)
- pybind11 (Python bindings)
- Catch2 (testing)

### 3. Configure CMake (Next)

**Windows:**
```bash
cmake --preset windows-release
```

**Linux:**
```bash
cmake --preset linux-release
```

**macOS:**
```bash
cmake --preset macos-release
```

**Custom:**
```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake
```

### 4. Build (Next)

```bash
# Build all components
cmake --build build/windows-release --config Release

# Or build specific component
cmake --build build/windows-release --target cyxwiz-backend --config Release
cmake --build build/windows-release --target cyxwiz-engine --config Release
```

### 5. Run (After Build)

```bash
# Desktop client
./build/windows-release/bin/cyxwiz-engine.exe

# Server node
./build/windows-release/bin/cyxwiz-server-node.exe

# Central server (Rust)
cd cyxwiz-central-server
cargo run --release
```

### 6. Test (After Build)

```bash
cd build/windows-release
ctest --output-on-failure
```

## Build Options

Configure build with CMake options:

```bash
cmake --preset windows-release \
  -DCYXWIZ_BUILD_ENGINE=ON \
  -DCYXWIZ_BUILD_SERVER_NODE=ON \
  -DCYXWIZ_BUILD_TESTS=ON \
  -DCYXWIZ_ENABLE_CUDA=OFF \
  -DCYXWIZ_ENABLE_OPENCL=ON
```

Options:
- `CYXWIZ_BUILD_ENGINE` - Build desktop client (default: ON)
- `CYXWIZ_BUILD_SERVER_NODE` - Build compute worker (default: ON)
- `CYXWIZ_BUILD_CENTRAL_SERVER` - Note for Rust build (default: ON)
- `CYXWIZ_BUILD_TESTS` - Build unit tests (default: ON)
- `CYXWIZ_ENABLE_CUDA` - Enable CUDA backend (default: OFF)
- `CYXWIZ_ENABLE_OPENCL` - Enable OpenCL backend (default: ON)
- `CYXWIZ_ANDROID_BUILD` - Build for Android (backend only)

## Debug vs Release

**Debug Build:**
- Symbols included (`-g`)
- No optimization (`-O0`)
- Logging enabled
- Memory tracking enabled
- Slower but easier to debug

**Release Build:**
- Full optimization (`-O3` / `-O2`)
- No symbols
- Minimal logging
- Faster execution

Switch between:
```bash
# Debug
cmake --preset windows-debug
cmake --build build/windows-debug

# Release
cmake --preset windows-release
cmake --build build/windows-release
```

## Troubleshooting

### "ArrayFire not found"
ArrayFire is optional for GPU acceleration. Backend will build in CPU-only mode without it.

To add it:
1. Download from https://arrayfire.com/download
2. Install to default location or set `ArrayFire_DIR` environment variable
3. Reconfigure CMake

### "vcpkg dependencies missing"
```bash
cd vcpkg
./vcpkg install
```

### "gRPC/Protobuf errors"
Clear build and reconfigure:
```bash
rm -rf build
cmake --preset windows-release
```

### Python not found
Ensure Python 3.8+ is installed and in PATH:
```bash
python --version
```

Install pybind11:
```bash
pip install pybind11
```

## Clean Build

```bash
rm -rf build
rm -rf vcpkg_installed
cd vcpkg && ./vcpkg install
cmake --preset windows-release
cmake --build build/windows-release
```

## Build Time Estimates

First build (with vcpkg dependencies):
- **Dependencies**: 15-30 minutes
- **CyxWiz**: 5-10 minutes
- **Total**: 20-40 minutes

Subsequent builds:
- **Incremental**: 30 seconds - 2 minutes
- **Clean rebuild**: 5-10 minutes

## Platform-Specific Notes

### Windows
- Requires Visual Studio 2019+ or MinGW-w64
- May need to run in Developer Command Prompt
- CMake automatically finds MSVC

### Linux
- Install build essentials: `sudo apt install build-essential cmake ninja-build`
- May need: `sudo apt install libgl1-mesa-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev`

### macOS
- Install Xcode Command Line Tools: `xcode-select --install`
- Recommended: Install Homebrew and use `brew install cmake ninja`

### Android (Backend Only)
```bash
cmake --preset android-release \
  -DANDROID_NDK=/path/to/ndk
cmake --build build/android-release
```

## Next Steps After Build

1. **Run the Engine**: `./build/bin/cyxwiz-engine`
2. **Try C API Example**: Build `examples/c_api_example.c`
3. **Implement Algorithms**: Start with `cyxwiz-backend/src/algorithms/`
4. **Add Features**: See CLAUDE.md for development guide

## Getting Help

- See **CLAUDE.md** for comprehensive developer guide
- Check **README.md** for project overview
- Review **PROJECT_STRUCTURE.md** for file organization
