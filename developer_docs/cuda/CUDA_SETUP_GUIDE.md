# CUDA Toolkit Setup Guide for CyxWiz

## Overview
This guide walks you through enabling CUDA support in CyxWiz to get accurate GPU memory reporting using NVIDIA's CUDA runtime API.

## Prerequisites
- **Windows 10/11** (64-bit)
- **NVIDIA GPU** (GTX 1050 Ti or newer)
- **Visual Studio 2019/2022** (already installed)
- **CMake 3.20+** (already installed)
- **ArrayFire with CUDA backend** (should already be installed)

## Step 1: Download CUDA Toolkit

### Option A: Latest Version (Recommended)
**CUDA Toolkit 12.x** (latest stable)
- URL: https://developer.nvidia.com/cuda-downloads
- Select: **Windows → x86_64 → 10/11 → exe (network)**
- Size: ~350 MB installer (downloads ~3-4 GB during installation)

### Option B: Legacy Version
**CUDA Toolkit 11.8** (if you have compatibility issues)
- URL: https://developer.nvidia.com/cuda-11-8-0-download-archive
- Select: **Windows → x86_64 → 10/11 → exe (network)**

## Step 2: Install CUDA Toolkit

### Installation Steps:
1. **Run the installer** as Administrator
2. **Installation Type**: Choose "Express" (recommended) or "Custom"
3. **Components to Install** (Express installs all automatically):
   - ✅ CUDA Toolkit (required)
   - ✅ CUDA Runtime (required)
   - ✅ CUDA Visual Studio Integration (required)
   - ✅ CUDA Samples (optional, useful for testing)
   - ✅ CUDA Documentation (optional)
   - ⚠️ Driver Update (skip if you have recent drivers)

4. **Installation Path**: Default is fine (usually `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`)

5. **Wait for installation** (~10-20 minutes)

### Verify Installation:
Open a **new** PowerShell/CMD window (important - to refresh environment variables):

```powershell
# Check CUDA version
nvcc --version

# Expected output:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2024 NVIDIA Corporation
# Built on ...
# Cuda compilation tools, release 12.x, ...

# Check environment variables
echo $env:CUDA_PATH
# Expected: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x

# Check if CUDA is in PATH
where nvcc
# Expected: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin\nvcc.exe
```

## Step 3: Reconfigure CMake with CUDA Enabled

### Clean Previous Build (Important!)
```powershell
cd D:\Dev\CyxWiz_Claude

# Remove old build directory to force CMake reconfiguration
Remove-Item -Recurse -Force build\windows-release -ErrorAction SilentlyContinue
```

### Configure with CUDA Enabled
```powershell
# Configure with CUDA enabled
cmake -B build/windows-release -S . `
  -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DCYXWIZ_ENABLE_CUDA=ON `
  -DCMAKE_BUILD_TYPE=Release
```

### Expected Output:
Look for these messages in the CMake output:
```
-- CUDA Toolkit found: 12.x
-- ArrayFire found: ...
-- Building cyxwiz-backend with CUDA support
```

### If CMake Can't Find CUDA:
Manually set the CUDA path:
```powershell
cmake -B build/windows-release -S . `
  -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DCYXWIZ_ENABLE_CUDA=ON `
  -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x" `
  -DCMAKE_BUILD_TYPE=Release
```

## Step 4: Build the Project

```powershell
# Build all components
cmake --build build/windows-release --config Release -j 8

# Or build just the backend and server node
cmake --build build/windows-release --config Release --target cyxwiz-backend -j 8
cmake --build build/windows-release --config Release --target cyxwiz-server-node -j 8
```

### Expected Build Output:
```
Building CXX object cyxwiz-backend/CMakeFiles/cyxwiz-backend.dir/src/core/device.cpp.obj
  device.cpp
  CYXWIZ_ENABLE_CUDA is defined
Linking CXX shared library cyxwiz-backend.dll
```

### Check for CUDA Linking:
```powershell
# Verify CUDA runtime is linked
dumpbin /DEPENDENTS build\windows-release\bin\cyxwiz-backend.dll

# Expected to see:
#   cudart64_12.dll (or cudart64_11.dll)
```

## Step 5: Test CUDA Memory Detection

### Run Server Node with CUDA:
```powershell
cd build\windows-release\bin

# Run server node
.\cyxwiz-server-node.exe
```

### Expected Output:
Look for these log messages:
```
[info] Initializing CyxWiz Backend...
[info] ArrayFire v3.x (CUDA, 64-bit, ...)
[info] Platform: NVIDIA CUDA
[info] Device 0: NVIDIA GeForce GTX 1050 Ti
[debug] CUDA device 0: 4.00 GB total, 3.85 GB free
[info] Successfully registered with Central Server
[info] Device: NVIDIA GeForce GTX 1050 Ti (CUDA)
[info] Memory: 4096 MB total, 3932 MB available
```

**Key difference from before:**
- ✅ Now: `CUDA device 0: 4.00 GB total, 3.85 GB free` (accurate CUDA API query)
- ❌ Before: `Using estimated memory for GTX 1050 Ti: 4 GB` (fallback estimation)

## Troubleshooting

### Issue 1: "CUDA Toolkit not found" during CMake configuration
**Cause:** CMake can't find CUDA installation
**Solution:**
```powershell
# Check CUDA_PATH environment variable
echo $env:CUDA_PATH

# If empty, set it manually:
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"

# Then reconfigure CMake
cmake -B build/windows-release -S . -DCYXWIZ_ENABLE_CUDA=ON ...
```

### Issue 2: "cudart64_XX.dll not found" when running executable
**Cause:** CUDA runtime DLL not in PATH
**Solution:**
```powershell
# Add CUDA bin directory to PATH
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin"

# Or copy the DLL to the build directory
Copy-Item "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin\cudart64_12.dll" `
  -Destination "build\windows-release\bin\"
```

### Issue 3: "cudaMemGetInfo returned error: no CUDA-capable device"
**Cause:** ArrayFire is not using CUDA backend
**Solution:**
```cpp
// In your code, explicitly set CUDA backend
af::setBackend(AF_BACKEND_CUDA);
```

Or check ArrayFire installation:
```powershell
# Verify ArrayFire CUDA backend exists
Test-Path "C:\Program Files\ArrayFire\v3\lib\afcuda.lib"
```

### Issue 4: Build error "cuda_runtime.h: No such file or directory"
**Cause:** CUDA Toolkit not properly installed or CMake can't find it
**Solution:**
```powershell
# Verify CUDA headers exist
Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\include\cuda_runtime.h"

# If file exists but CMake can't find it, set CUDAToolkit_ROOT:
cmake -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x" ...
```

### Issue 5: "LINK : fatal error LNK1104: cannot open file 'cudart.lib'"
**Cause:** CUDA libraries not found by linker
**Solution:**
```powershell
# Verify CUDA libraries exist
Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\lib\x64\cudart.lib"

# If exists, add to CMake:
cmake -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x" ...
```

## Verification Checklist

After installation, verify:

- [ ] `nvcc --version` shows CUDA version
- [ ] `$env:CUDA_PATH` is set to CUDA installation directory
- [ ] CMake configuration shows "CUDA Toolkit found: 12.x"
- [ ] Build completes without CUDA-related errors
- [ ] `dumpbin /DEPENDENTS cyxwiz-backend.dll` shows `cudart64_XX.dll`
- [ ] Server Node logs show "CUDA device X: Y.Y GB total, Z.Z GB free"
- [ ] No "Using estimated memory" messages in logs

## Performance Impact

**Before CUDA Toolkit (OpenCL + estimation):**
- ✅ Works fine for computation
- ⚠️ Memory reporting is estimated or may be inaccurate
- ⚠️ May report 0 MB if OpenCL query fails

**After CUDA Toolkit:**
- ✅ Accurate real-time memory reporting via `cudaMemGetInfo()`
- ✅ Can detect actual available VRAM (accounts for other GPU processes)
- ✅ Better job scheduling (Central Server knows exact node capacity)
- ✅ No change to computation performance

## When to Skip CUDA Toolkit Installation

You can skip CUDA Toolkit if:
- ✅ You only have AMD or Intel GPUs (use OpenCL backend)
- ✅ You're fine with memory estimation (works for most use cases)
- ✅ You're using CPU backend only
- ✅ You're just testing and don't need accurate memory reporting

## Summary of Changes Made

### Files Modified:
1. **cyxwiz-backend/CMakeLists.txt** (lines 83-97)
   - Added `find_package(CUDAToolkit QUIET)`
   - Added `CUDA::cudart` to link libraries
   - Added helpful warning messages if CUDA not found

2. **cyxwiz-backend/src/core/device.cpp** (already prepared)
   - Lines 7-9: CUDA runtime header included
   - Lines 45-66: CUDA memory query using `cudaMemGetInfo()`

### No Code Changes Needed:
- All CUDA integration code already exists
- Just need to enable it via CMake flag and install CUDA Toolkit

## Next Steps

1. ✅ **Install CUDA Toolkit** (this guide)
2. ✅ **Reconfigure and rebuild** with CUDA enabled
3. ✅ **Test memory detection** with Server Node
4. 🔜 **Continue to Phase 5** (Job Execution & Scheduling)

## Additional Resources

- NVIDIA CUDA Documentation: https://docs.nvidia.com/cuda/
- CUDA Toolkit Archive: https://developer.nvidia.com/cuda-toolkit-archive
- CMake FindCUDAToolkit: https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
- ArrayFire CUDA Backend: https://arrayfire.org/docs/using_on_windows.htm

---

**Questions?** Check the troubleshooting section or ask for help!
