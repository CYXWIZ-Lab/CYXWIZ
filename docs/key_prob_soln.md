# CyxWiz Cross-Platform Build: Key Problems & Solutions

This document captures the key challenges encountered during cross-platform CI/CD setup and their solutions. Use this as a reference when troubleshooting build issues.

---

## Table of Contents

1. [vcpkg Dependency Management](#1-vcpkg-dependency-management)
2. [Disk Space on CI Runners](#2-disk-space-on-ci-runners)
3. [External Dependencies (ArrayFire)](#3-external-dependencies-arrayfire)
4. [Path Differences Across Platforms](#4-path-differences-across-platforms)
5. [Platform-Specific Quirks](#5-platform-specific-quirks)
6. [Build Timeouts](#6-build-timeouts)
7. [CMake Configuration Issues](#7-cmake-configuration-issues)

---

## 1. vcpkg Dependency Management

### Problem
vcpkg builds all dependencies from source by default. CyxWiz has heavy dependencies:
- **gRPC + protobuf**: ~45 minutes to build
- **OpenCV**: ~30 minutes
- **llama-cpp**: ~20 minutes
- **ONNX Runtime**: ~15 minutes

**Total first-time build: 2-4 hours**

### Solution: Binary Caching

Enable vcpkg binary caching to download pre-built binaries instead of building from source.

**For GitHub Actions:**
```yaml
env:
  VCPKG_BINARY_SOURCES: "clear;x-gha,readwrite"

steps:
  - name: Export GitHub Actions cache variables
    uses: actions/github-script@v7
    with:
      script: |
        core.exportVariable('ACTIONS_CACHE_URL', process.env.ACTIONS_CACHE_URL || '');
        core.exportVariable('ACTIONS_RUNTIME_TOKEN', process.env.ACTIONS_RUNTIME_TOKEN || '');
```

**For Local Development:**
```bash
# Linux/macOS
export VCPKG_BINARY_SOURCES="clear;default,readwrite"
export VCPKG_DEFAULT_BINARY_CACHE="$HOME/.vcpkg-cache"

# Windows PowerShell
$env:VCPKG_BINARY_SOURCES = "clear;default,readwrite"
$env:VCPKG_DEFAULT_BINARY_CACHE = "C:\vcpkg-cache"
```

### Key Insight
- First build populates the cache (slow)
- Subsequent builds download from cache (fast, ~15 min)
- Don't use `doNotCache: true` in CI unless you have a specific reason

---

## 2. Disk Space on CI Runners

### Problem
GitHub Actions runners have ~14GB free space. Our dependencies + build artifacts exceeded this:
- vcpkg dependencies: ~8GB
- ArrayFire: ~2GB
- Build artifacts: ~4GB

**Error:**
```
System.IO.IOException: No space left on device
```

### Solution: Pre-build Cleanup

Add a cleanup step before installing dependencies:

```yaml
- name: Free Disk Space
  run: |
    sudo rm -rf /usr/share/dotnet          # ~6GB
    sudo rm -rf /usr/local/lib/android     # ~10GB
    sudo rm -rf /opt/ghc                   # ~2GB
    sudo rm -rf /opt/hostedtoolcache/CodeQL
    sudo rm -rf /usr/local/share/boost
    sudo apt-get clean
    df -h  # Verify ~30GB free
```

### Alternative Solutions
- Use GitHub's larger runners (150GB, requires paid plan)
- Reduce dependencies for CI builds
- Build components separately

---

## 3. External Dependencies (ArrayFire)

### Problem
ArrayFire download URLs changed from S3 to a new CDN. Old URLs returned 404/connection errors.

**Old (broken):**
```
https://arrayfire.s3.amazonaws.com/3.9.0/ArrayFire-v3.9.0-Linux-x86_64.sh
```

**New (working):**
```
https://arrayfire.gateway.scarf.sh/linux/3.10.0/ArrayFire.sh
```

### Solution: Updated Installation Scripts

**Windows:**
```powershell
Invoke-WebRequest -Uri https://arrayfire.gateway.scarf.sh/windows/3.10.0/ArrayFire.exe -OutFile af.exe
Start-Process -FilePath af.exe -ArgumentList /S -Wait
```

**Linux:**
```bash
wget https://arrayfire.gateway.scarf.sh/linux/3.10.0/ArrayFire.sh
chmod +x ArrayFire.sh
yes | sudo ./ArrayFire.sh --prefix=/opt/arrayfire
```

**macOS:**
```bash
brew install arrayfire  # Simplest option
```

### Key Insight
- External download URLs break over time
- Use package managers when possible (brew, apt)
- Consider hosting your own mirrors for critical dependencies
- Pin specific versions in your CI scripts

---

## 4. Path Differences Across Platforms

### Problem
Hardcoded paths don't work across platforms:
```cmake
# This breaks in CI because vcpkg is in a different location
CMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake
```

### Solution: Use Environment Variables

**lukka/run-vcpkg** sets `VCPKG_ROOT` automatically. Use it:

**Windows (PowerShell):**
```powershell
cmake -B build -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
```

**Linux/macOS (Bash):**
```bash
cmake -B build -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
```

### CMakePresets.json

Use variables that resolve at configure time:
```json
{
  "cacheVariables": {
    "CMAKE_TOOLCHAIN_FILE": {
      "value": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake",
      "type": "FILEPATH"
    }
  }
}
```

---

## 5. Platform-Specific Quirks

### Windows

| Issue | Solution |
|-------|----------|
| PowerShell vs CMD syntax | Use `shell: pwsh` explicitly |
| Path separators | Use forward slashes `/` in CMake |
| DLL not found at runtime | Copy DLLs to exe directory or add to PATH |
| Long path issues | Enable long paths in registry or use shorter paths |

### Linux

| Issue | Solution |
|-------|----------|
| Interactive installer prompts | Pipe `yes \|` to installers |
| Missing dev packages | Install with apt: `libgl1-mesa-dev`, `libxrandr-dev`, etc. |
| Library not found at runtime | Set `LD_LIBRARY_PATH` or use `rpath` |
| Permissions | Use `sudo` for system-wide installs |

### macOS

| Issue | Solution |
|-------|----------|
| Missing Python for vcpkg | `brew install python@3.11` |
| Gatekeeper blocks binaries | `xattr -cr <binary>` |
| Apple Silicon vs Intel | Use universal binaries or separate builds |
| Code signing | Sign with `codesign` or distribute unsigned with instructions |

---

## 6. Build Timeouts

### Problem
GitHub Actions has default timeout of 6 hours, but individual jobs may need more time for first-time builds.

### Solution: Explicit Timeouts

```yaml
jobs:
  build:
    timeout-minutes: 180  # 3 hours for first build
```

### Optimization: Parallel Jobs

Don't build all platforms sequentially. Run them in parallel:
```yaml
jobs:
  build-windows:
    runs-on: windows-latest
  build-linux:
    runs-on: ubuntu-22.04
  build-macos:
    runs-on: macos-14
```

---

## 7. CMake Configuration Issues

### Problem: Invalid Generator

```json
"generator": "Visual Studio 18 2026"  // Doesn't exist!
```

### Solution: Use Ninja for Cross-Platform

```json
{
  "generator": "Ninja"
}
```

Ninja is:
- Cross-platform (Windows, Linux, macOS)
- Faster than MSBuild/Make for incremental builds
- Simpler configuration

### Problem: ArrayFire Not Found

CMake can't find ArrayFire even though it's installed.

### Solution: Specify Path Explicitly

```bash
# Find where ArrayFire cmake config is
find /opt/arrayfire -name "ArrayFireConfig.cmake"

# Use it
cmake -B build -DArrayFire_DIR=/opt/arrayfire/share/ArrayFire/cmake
```

Or for auto-detection in CI:
```bash
AF_DIR=$(find /opt/arrayfire -name "ArrayFireConfig.cmake" -exec dirname {} \; | head -1)
cmake -B build ${AF_DIR:+-DArrayFire_DIR="$AF_DIR"}
```

---

## Quick Reference: CI Workflow Template

```yaml
name: Build

on: [push, pull_request]

env:
  VCPKG_BINARY_SOURCES: "clear;x-gha,readwrite"

jobs:
  build:
    strategy:
      matrix:
        os: [windows-latest, ubuntu-22.04, macos-14]
    runs-on: ${{ matrix.os }}
    timeout-minutes: 180

    steps:
      - uses: actions/checkout@v4

      # Free disk space (Linux only)
      - name: Free Disk Space
        if: runner.os == 'Linux'
        run: |
          sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc
          sudo apt-get clean

      # Export cache variables
      - uses: actions/github-script@v7
        with:
          script: |
            core.exportVariable('ACTIONS_CACHE_URL', process.env.ACTIONS_CACHE_URL || '');
            core.exportVariable('ACTIONS_RUNTIME_TOKEN', process.env.ACTIONS_RUNTIME_TOKEN || '');

      # Setup vcpkg
      - uses: lukka/run-vcpkg@v11
        with:
          vcpkgGitCommitId: 594ad8871e1e8e45f8e626c015fd611163430207

      # Install ArrayFire
      - name: Install ArrayFire (Windows)
        if: runner.os == 'Windows'
        shell: pwsh
        run: |
          Invoke-WebRequest -Uri https://arrayfire.gateway.scarf.sh/windows/3.10.0/ArrayFire.exe -OutFile af.exe
          Start-Process -FilePath af.exe -ArgumentList /S -Wait

      - name: Install ArrayFire (Linux)
        if: runner.os == 'Linux'
        run: |
          wget -q https://arrayfire.gateway.scarf.sh/linux/3.10.0/ArrayFire.sh
          yes | sudo ./ArrayFire.sh --prefix=/opt/arrayfire

      - name: Install ArrayFire (macOS)
        if: runner.os == 'macOS'
        run: brew install arrayfire

      # Configure & Build
      - name: Configure
        run: cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=${{ env.VCPKG_ROOT }}/scripts/buildsystems/vcpkg.cmake

      - name: Build
        run: cmake --build build --parallel
```

---

## Lessons Learned

1. **Cache Everything** - First build is slow, subsequent builds should be fast
2. **Test Locally First** - Use `act` to run GitHub Actions locally before pushing
3. **Pin Versions** - External URLs and package versions change; pin them
4. **Use Package Managers** - brew, apt, vcpkg > manual downloads
5. **Fail Fast** - Add good error messages and early validation
6. **Document Quirks** - Platform differences are subtle; document them

---

## Related Documentation

- [INSTALL.md](../INSTALL.md) - User installation guide
- [mainbuild.md](mainbuild.md) - Developer build guide
- [CLAUDE.md](../CLAUDE.md) - AI assistant context
