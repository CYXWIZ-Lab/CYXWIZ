# Building CyxWiz on macOS - Complete Guide

This document provides a comprehensive guide for building the CyxWiz Engine on macOS, including all encountered issues, solutions, and optimization tips based on a successful build.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Prerequisites](#prerequisites)
3. [Initial Setup](#initial-setup)
4. [Build Process](#build-process)
5. [Errors Encountered & Solutions](#errors-encountered--solutions)
6. [Build Times](#build-times)
7. [Running the Engine](#running-the-engine)
8. [Tips for Smooth Builds](#tips-for-smooth-builds)
9. [Troubleshooting](#troubleshooting)

---

## System Requirements

**Tested Configuration:**
- **OS**: macOS 15.3 (Sequoia)
- **Chip**: Apple Silicon / Intel x86_64
- **RAM**: 8GB minimum (16GB+ recommended for vcpkg compilation)
- **Disk Space**: ~15GB for full build (vcpkg cache, dependencies, binaries)
- **Xcode Command Line Tools**: Latest version

---

## Prerequisites

### Required Tools

1. **CMake 3.20+**
   ```bash
   brew install cmake
   ```
   Verified version: 3.31.5

2. **C++ Compiler**
   - AppleClang 17.0+ (via Xcode Command Line Tools)
   ```bash
   xcode-select --install
   ```

3. **Python 3.8+**
   ```bash
   brew install python@3.14
   ```

4. **Rust & Cargo 1.92+** (for Central Server component)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

5. **Build Tools** (Critical for macOS)
   ```bash
   brew install autoconf autoconf-archive automake libtool ninja
   ```

   **Note**: These tools are REQUIRED before CMake configuration. Missing these will cause build failures.

6. **Git with Submodule Support**
   ```bash
   git --version  # Should be 2.x+
   ```

---

## Initial Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/cyxwiz.git
cd cyxwiz
```

### 2. Initialize Git Submodules

**⚠️ CRITICAL STEP**: The project has external dependencies that must be initialized:

```bash
# Initialize ImGuiColorTextEdit submodule
git submodule update --init cyxwiz-engine/external/ImGuiColorTextEdit

# Update to latest ImGui-compatible version
cd cyxwiz-engine/external/ImGuiColorTextEdit
git fetch origin
git checkout ca2f9f1  # Latest version with ImGui 1.91+ compatibility
cd ../../..
```

### 3. Clone imnodes External Dependency

```bash
cd cyxwiz-engine/external
git clone https://github.com/Nelarius/imnodes.git
cd ../..
```

### 4. Run Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

**Expected Duration**: 45-60 minutes (first run)
- vcpkg bootstrap: ~2-3 minutes
- Installing 37 packages: 40-55 minutes
  - If interrupted, cached packages (~33/37) will be reused on retry

**Known Issue**: The setup may timeout on `abseil` (package 10/37). This is expected. The vcpkg cache will save progress, so simply retry CMake configuration later.

---

## Build Process

### Quick Build (Recommended)

**For Engine + Server Node (Fully Supported on macOS)**:

```bash
# Configure CMake
cmake -B build/macos-release -S . \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DCYXWIZ_BUILD_ENGINE=ON \
  -DCYXWIZ_BUILD_SERVER_NODE=ON \
  -DCYXWIZ_BUILD_TESTS=OFF

# Build
ninja -C build/macos-release
```

**For Engine Only** (if you don't need Server Node):

```bash
# Configure CMake
cmake -B build/macos-release -S . \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DCYXWIZ_BUILD_ENGINE=ON \
  -DCYXWIZ_BUILD_SERVER_NODE=OFF \
  -DCYXWIZ_BUILD_TESTS=OFF

# Build
ninja -C build/macos-release
```

### Step-by-Step Build

#### 1. CMake Configuration

```bash
cmake -B build/macos-release -S . \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DCYXWIZ_BUILD_ENGINE=ON \
  -DCYXWIZ_BUILD_SERVER_NODE=OFF \
  -DCYXWIZ_BUILD_TESTS=OFF
```

**Expected Output:**
```
-- Building CyxWiz for macOS
-- Release build - Optimizations enabled
-- Platform: macOS
-- Build Type: Release
-- C++ Standard: 20
-- Components:
--   Engine (Desktop Client): ON
--   Server Node: OFF
--   Tests: OFF
-- Configuring done (16.8s)
-- Generating done (0.3s)
```

#### 2. Build with Ninja

```bash
ninja -C build/macos-release
```

**Expected Output:**
```
ninja: Entering directory `build/macos-release'
[1/119] Generating protobuf/gRPC code for proto/common.proto
...
[119/119] Linking CXX executable bin/cyxwiz-engine
```

#### 3. Verify Build

```bash
ls -lh build/macos-release/bin/cyxwiz-engine
file build/macos-release/bin/cyxwiz-engine
```

**Expected:**
```
-rwxr-xr-x  1 user  staff    29M Dec 13 16:26 build/macos-release/bin/cyxwiz-engine
build/macos-release/bin/cyxwiz-engine: Mach-O 64-bit executable x86_64
```

---

## Errors Encountered & Solutions

### Error 1: Missing macOS Build Tools

**Error:**
```
CMake Error: python3 currently requires the following programs:
    autoconf autoconf-archive automake libtoolize
CMake Error: CMake was unable to find a build program corresponding to "Ninja"
```

**Cause**: macOS system lacks required build automation tools.

**Solution:**
```bash
brew install autoconf autoconf-archive automake libtool ninja
```

**Prevention**: Install these tools BEFORE running setup.sh or CMake.

---

### Error 2: Missing Git Submodule (ImGuiColorTextEdit)

**Error:**
```
CMake Error: Cannot find source file:
    external/ImGuiColorTextEdit/TextEditor.cpp
```

**Cause**: Git submodule not initialized.

**Solution:**
```bash
git submodule update --init cyxwiz-engine/external/ImGuiColorTextEdit
```

---

### Error 3: ImGui Compatibility - GetKeyIndex() Not Found

**Error:**
```
/cyxwiz-engine/external/ImGuiColorTextEdit/TextEditor.cpp:714:77:
error: no member named 'GetKeyIndex' in namespace 'ImGui'
```

**Cause**: ImGuiColorTextEdit submodule uses outdated ImGui API. The `GetKeyIndex()` function was removed in ImGui 1.87+, but vcpkg installs ImGui 1.91.9.

**Solution**: Update ImGuiColorTextEdit to latest commit:
```bash
cd cyxwiz-engine/external/ImGuiColorTextEdit
git fetch origin
git checkout ca2f9f1  # Contains fix for ImGui 1.87+
cd ../../..
```

**Commit Details**: `ca2f9f1` - "Merge pull request #166: Update to work on latest Dear ImGui: removed obsolete calls to GetKeyIndex()"

---

### Error 4: Missing imnodes External Library

**Error:**
```
CMake Error: Cannot find source file:
    external/imnodes/imnodes.cpp
```

**Cause**: imnodes library not cloned.

**Solution:**
```bash
cd cyxwiz-engine/external
git clone https://github.com/Nelarius/imnodes.git
cd ../..
```

---

### Error 5: Server Node macOS Compatibility Issues (RESOLVED)

**Status**: ✅ **FIXED** - Server Node now builds successfully on macOS!

The Server Node component initially had Linux-specific code that prevented macOS builds. These issues have been resolved with platform-specific fixes.

#### Issue 5a: Missing signal.h for kill() Function

**Error:**
```
terminal_handler.cpp:165:9: error: use of undeclared identifier 'kill'
```

**Cause**: macOS requires explicit `signal.h` include for `kill()` function.

**Solution**: Add signal.h to macOS section in `cyxwiz-server-node/src/terminal_handler.cpp`:
```cpp
#elif defined(__APPLE__)
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <signal.h>      // REQUIRED for kill()
#include <util.h>        // macOS uses util.h for openpty()
```

**File**: `cyxwiz-server-node/src/terminal_handler.cpp:14`

#### Issue 5b: Missing gethostname() Declaration

**Error:**
```
login_panel.cpp:267:25: error: use of undeclared identifier 'gethostname'
```

**Cause**: `unistd.h` not included for Unix/macOS platforms.

**Solution**: Add unistd.h include in `cyxwiz-server-node/src/gui/panels/login_panel.cpp`:
```cpp
#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>  // for gethostname on Unix/macOS
#endif
```

**File**: `cyxwiz-server-node/src/gui/panels/login_panel.cpp`

#### Issue 5c: Missing mkdir() Declaration

**Error:**
```
allocation_panel.cpp:581:5: error: use of undeclared identifier 'mkdir'
```

**Cause**: `sys/stat.h` not included for Unix/macOS platforms.

**Solution**: Add sys/stat.h include in `cyxwiz-server-node/src/gui/panels/allocation_panel.cpp`:
```cpp
#include <imgui.h>
#include <imgui_internal.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <nlohmann/json.hpp>

#ifndef _WIN32
#include <sys/stat.h>  // for mkdir on Unix/macOS
#endif
```

**File**: `cyxwiz-server-node/src/gui/panels/allocation_panel.cpp`

#### Issue 5d: Missing CFNetwork Framework (Critical!)

**Error:**
```
Undefined symbols for architecture x86_64:
  "_CFHostCancelInfoResolution", referenced from:
      httplib::detail::getaddrinfo_with_timeout(...) in openai_api_server.cpp.o
  "_CFHostCreateWithName", referenced from:
      httplib::detail::getaddrinfo_with_timeout(...) in openai_api_server.cpp.o
  [... more CFHost* symbols]
ld: symbol(s) not found for architecture x86_64
```

**Cause**: The cpp-httplib library uses CFHost APIs for DNS resolution with timeout on macOS. These APIs are part of the **CFNetwork framework**, not CoreFoundation.

**Solution**: Add CFNetwork framework to macOS linker flags in `cyxwiz-server-node/CMakeLists.txt`:
```cmake
# Platform-specific libraries for daemon
if(WIN32)
    target_link_libraries(cyxwiz-server-daemon PRIVATE ws2_32 pdh iphlpapi)
    # ... Windows compile definitions ...
elseif(APPLE)
    find_library(COREFOUNDATION_LIBRARY CoreFoundation REQUIRED)
    find_library(CFNETWORK_LIBRARY CFNetwork REQUIRED)
    target_link_libraries(cyxwiz-server-daemon PRIVATE
        util pthread
        ${COREFOUNDATION_LIBRARY}
        ${CFNETWORK_LIBRARY}
    )
else()
    target_link_libraries(cyxwiz-server-daemon PRIVATE util pthread)
endif()
```

**File**: `cyxwiz-server-node/CMakeLists.txt:191-194`

**Why CFNetwork?**: While CoreFoundation provides basic framework functionality, the CFHost* APIs (CFHostCreateWithName, CFHostStartInfoResolution, etc.) are specifically part of CFNetwork framework on macOS.

---

### Building Server Node on macOS (Updated)

**Server Node is now fully supported on macOS!** Use the following configuration:

```bash
# Configure for Server Node + Engine
cmake -B build/macos-release -S . -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DCYXWIZ_BUILD_ENGINE=ON \
  -DCYXWIZ_BUILD_SERVER_NODE=ON \
  -DCYXWIZ_BUILD_TESTS=OFF

# Build
ninja -C build/macos-release
```

**Expected Binaries:**
```bash
build/macos-release/bin/
├── cyxwiz-engine           # Desktop Engine (29 MB)
├── cyxwiz-server-daemon    # Server Node daemon (27 MB)
└── cyxwiz-server-gui       # Server Node GUI/TUI (28 MB)
```

**Verify Server Node:**
```bash
./build/macos-release/bin/cyxwiz-server-daemon --help
./build/macos-release/bin/cyxwiz-server-gui --help
```

---

### Error 6: Missing startup_scripts.txt

**Error:**
```
Error copying file (if different) from "startup_scripts.txt" to
"/build/macos-release/bin/startup_scripts.txt".
```

**Cause**: CMake expects startup_scripts.txt in project root.

**Solution:**
```bash
touch startup_scripts.txt
ninja -C build/macos-release  # Retry build
```

---

## Build Times

**Total Build Time**: ~50-65 minutes (first build)

### Breakdown:

| Phase | Duration | Notes |
|-------|----------|-------|
| **vcpkg Bootstrap** | 2-3 min | One-time setup |
| **vcpkg Package Install** | 40-55 min | 37 packages (first run) |
| - abseil | ~15-20 min | Largest package, may timeout |
| - python3 | ~23 min | Full Python interpreter build |
| - pybind11 | ~16 sec | Header-only, fast |
| - spdlog | ~27 sec | Fast compile |
| - stb | ~3 sec | Header-only |
| **CMake Configuration** | 16-18 sec | Quick with cached packages |
| **Ninja Build (Engine)** | 2-4 min | 119 targets, parallel jobs |
| - Protobuf code generation | ~5 sec | 6 proto files |
| - C++ compilation | ~3 min | Parallel compilation |
| - Linking | ~10 sec | Final executable |

### Subsequent Builds:
- **Incremental (1 file change)**: 5-15 seconds
- **Clean rebuild (Engine only)**: 2-4 minutes
- **Full clean + CMake**: 3-5 minutes

**Optimization**: Use `-j N` flag with ninja for custom parallel jobs:
```bash
ninja -C build/macos-release -j 16  # 16 parallel jobs
```

---

## Running the Engine

### Launch the Executable

```bash
./build/macos-release/bin/cyxwiz-engine
```

### Verify Libraries

Check dynamic library dependencies:
```bash
otool -L build/macos-release/bin/cyxwiz-engine
```

### Application Structure

The executable bundles with:
- **Resources**: `build/macos-release/bin/resources/` (copied from `resources/`)
- **Scripts**: `build/macos-release/bin/scripts/` (copied from `scripts/`)
- **Config**: `build/macos-release/bin/startup_scripts.txt`
- **Test Data**: `build/macos-release/bin/test_data.csv`

---

## Tips for Smooth Builds

### 1. Install All Prerequisites First

```bash
# One-command setup
brew install cmake python@3.14 autoconf autoconf-archive automake libtool ninja

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 2. Initialize Submodules Before Setup

```bash
git submodule update --init cyxwiz-engine/external/ImGuiColorTextEdit
cd cyxwiz-engine/external/ImGuiColorTextEdit && git checkout ca2f9f1 && cd ../../..
cd cyxwiz-engine/external && git clone https://github.com/Nelarius/imnodes.git && cd ../..
```

### 3. Use Ninja Instead of Make

Ninja is 2-3x faster than Unix Makefiles:
```bash
cmake -G Ninja ...  # Always specify Ninja
```

### 4. Enable Parallel Compilation

```bash
# Auto-detect CPU cores
ninja -C build/macos-release

# Manual cores (useful for limiting CPU usage)
ninja -C build/macos-release -j 8
```

### 5. Monitor vcpkg Progress

If vcpkg appears stuck:
```bash
# Check vcpkg cache (packages already built)
ls -la ~/.cache/vcpkg/archives

# Count cached packages
ls ~/.cache/vcpkg/archives | wc -l
```

### 6. Use ccache for Faster Rebuilds (Optional)

```bash
brew install ccache
cmake -B build/macos-release -S . \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  ...
```

### 7. Build Only What You Need

```bash
# Engine only (recommended for macOS)
-DCYXWIZ_BUILD_ENGINE=ON
-DCYXWIZ_BUILD_SERVER_NODE=OFF
-DCYXWIZ_BUILD_TESTS=OFF

# Include tests
-DCYXWIZ_BUILD_TESTS=ON
```

---

## Troubleshooting

### Problem: CMake Can't Find vcpkg Packages

**Symptoms:**
```
CMake Error: Could not find vcpkg package fmt
```

**Solution:**
```bash
# Verify vcpkg exists
ls vcpkg/vcpkg

# Re-bootstrap vcpkg
cd vcpkg && ./bootstrap-vcpkg.sh && cd ..

# Clean and reconfigure
rm -rf build/macos-release
cmake -B build/macos-release ...
```

---

### Problem: Ninja Build Fails with "cannot execute binary file"

**Symptoms:**
```
ninja: error: ninja: cannot execute binary file
```

**Solution:**
```bash
# Reinstall ninja
brew uninstall ninja
brew install ninja

# Verify
which ninja
ninja --version
```

---

### Problem: Python Headers Not Found

**Symptoms:**
```
fatal error: 'Python.h' file not found
```

**Solution:**
```bash
# Install Python development headers
brew install python@3.14

# Set Python path
export PYTHON_INCLUDE_DIR=/usr/local/opt/python@3.14/Frameworks/Python.framework/Versions/3.14/include/python3.14

# Reconfigure
cmake -B build/macos-release ...
```

---

### Problem: Linker Warnings about Duplicate Libraries

**Symptoms:**
```
ld: warning: ignoring duplicate libraries: 'libfmt.a', 'libspdlog.a'
```

**Impact**: Harmless warning, does not affect executable.

**Explanation**: Some dependencies declare overlapping libraries in their CMake targets. The linker ignores duplicates automatically.

---

### Problem: vcpkg Timeout on abseil

**Symptoms:**
```
Building abseil... (stuck for >30 minutes)
```

**Solution:**
- **Option 1**: Wait it out (abseil has 200+ components, takes 15-20 min on some systems)
- **Option 2**: Interrupt and let vcpkg cache partial progress, then retry
- **Option 3**: Use pre-built vcpkg binary cache (advanced)

---

## Build Artifacts

### Successful Build Outputs

```
build/macos-release/
├── bin/
│   ├── cyxwiz-engine          # Desktop Engine executable (29 MB)
│   ├── cyxwiz-server-daemon   # Server Node daemon (27 MB)
│   ├── cyxwiz-server-gui      # Server Node GUI/TUI (28 MB)
│   ├── resources/             # Assets, fonts, icons
│   ├── scripts/               # Python scripts
│   ├── startup_scripts.txt    # Startup configuration
│   └── test_data.csv          # Sample data
├── lib/
│   ├── libcyxwiz-backend.dylib    # Backend shared library
│   └── libcyxwiz-protocol.a       # Protocol static library
└── vcpkg_installed/
    └── x64-osx/               # 37+ vcpkg packages
```

### Binary Size

- **cyxwiz-engine**: 29 MB (Release build, stripped)
- **cyxwiz-server-daemon**: 27 MB (Release build)
- **cyxwiz-server-gui**: 28 MB (Release build)
- **libcyxwiz-backend.dylib**: ~2.5 MB

---

## Platform-Specific Notes

### macOS Sequoia (15.3+)
- Xcode Command Line Tools 17.0+ required
- No known compatibility issues

### macOS Ventura/Sonoma (13.x/14.x)
- Fully compatible
- Ensure latest Xcode Command Line Tools

### Apple Silicon (M1/M2/M3)
- Builds natively for ARM64
- vcpkg uses `arm64-osx` triplet
- Python may default to Intel build - verify with `python3 --version`

### Intel Macs
- Builds natively for x86_64
- vcpkg uses `x64-osx` triplet
- Slower vcpkg compilation compared to Apple Silicon

---

## Summary

**Working Build Configuration (Verified):**
```bash
# Prerequisites
brew install cmake python@3.14 autoconf autoconf-archive automake libtool ninja
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Submodules
git submodule update --init cyxwiz-engine/external/ImGuiColorTextEdit
cd cyxwiz-engine/external/ImGuiColorTextEdit && git checkout ca2f9f1 && cd ../../..
cd cyxwiz-engine/external && git clone https://github.com/Nelarius/imnodes.git && cd ../..

# Create missing file
touch startup_scripts.txt

# Configure (Engine + Server Node)
cmake -B build/macos-release -S . -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DCYXWIZ_BUILD_ENGINE=ON \
  -DCYXWIZ_BUILD_SERVER_NODE=ON \
  -DCYXWIZ_BUILD_TESTS=OFF

# Build
ninja -C build/macos-release

# Run
./build/macos-release/bin/cyxwiz-engine
```

**Total Time**: ~50-60 minutes (first build), ~2-4 minutes (subsequent builds)

---

## Additional Resources

- **CMake Documentation**: https://cmake.org/documentation/
- **vcpkg Documentation**: https://vcpkg.io/
- **Ninja Build System**: https://ninja-build.org/
- **ImGui**: https://github.com/ocornut/imgui
- **gRPC C++**: https://grpc.io/docs/languages/cpp/

---

## Build Verification Checklist

- [ ] All prerequisites installed
- [ ] Git submodules initialized
- [ ] imnodes cloned
- [ ] ImGuiColorTextEdit updated to ca2f9f1
- [ ] startup_scripts.txt created
- [ ] CMake configuration successful
- [ ] Ninja build completed without errors
- [ ] Executable created (29 MB, Mach-O x86_64)
- [ ] Resources copied to bin/
- [ ] Application launches successfully

---

**Last Updated**: December 14, 2024
**Tested On**: macOS 15.3 (Sequoia), AppleClang 17.0, CMake 3.31.5
**Build Result**: ✅ Successful (Engine + Server Node components)
**Server Node**: ✅ Fully supported on macOS (requires CFNetwork framework)
