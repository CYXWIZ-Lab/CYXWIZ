# CyxWiz Platform-Specific Build Guide

This document outlines the modular build strategy for Windows to reduce binary size by shipping only the required compute backend libraries.

---

## Overview

ArrayFire supports multiple compute backends (CUDA, OpenCL, CPU). Rather than shipping all backends in a single bloated package, we create separate distributions for each hardware configuration.

**Benefits:**
- Reduced download/install size for end users
- Users only get libraries relevant to their hardware
- CPU fallback is always available via ArrayFire

---

## Directory Structure

```
setup/
├── nvidia/                 # For NVIDIA GPU users (CUDA)
│   ├── bin/
│   │   ├── cyxwiz-engine.exe
│   │   └── cyxwiz-server-node.exe
│   └── lib/
│       ├── arrayfire/
│       ├── cuda/
│       └── common/
│
├── amd/                    # For AMD GPU users (OpenCL)
│   ├── bin/
│   │   ├── cyxwiz-engine.exe
│   │   └── cyxwiz-server-node.exe
│   └── lib/
│       ├── arrayfire/
│       ├── opencl/
│       └── common/
│
├── intel/                  # For Intel GPU users (OpenCL)
│   ├── bin/
│   │   ├── cyxwiz-engine.exe
│   │   └── cyxwiz-server-node.exe
│   └── lib/
│       ├── arrayfire/
│       ├── opencl/
│       └── common/
│
└── cpu/                    # CPU-only (smallest, universal fallback)
    ├── bin/
    │   ├── cyxwiz-engine.exe
    │   └── cyxwiz-server-node.exe
    └── lib/
        ├── arrayfire/
        └── common/
```

---

## Required Libraries by Variant

### NVIDIA (CUDA) Variant

**ArrayFire Libraries:**
| Library | Description |
|---------|-------------|
| `af.dll` | ArrayFire core library |
| `afcuda.dll` | CUDA backend |
| `forge.dll` | ArrayFire visualization (optional) |

**CUDA Runtime Libraries:**
| Library | Description |
|---------|-------------|
| `cudart64_*.dll` | CUDA Runtime |
| `cublas64_*.dll` | cuBLAS (matrix operations) |
| `cublasLt64_*.dll` | cuBLAS Lt (lightweight) |
| `cufft64_*.dll` | cuFFT (FFT operations) |
| `cusolver64_*.dll` | cuSOLVER (linear algebra) |
| `cusparse64_*.dll` | cuSPARSE (sparse matrices) |
| `nvrtc64_*.dll` | NVRTC (runtime compilation) |
| `nvJitLink_*.dll` | JIT linker (CUDA 12+) |

**Note:** CUDA version should match ArrayFire build. Typically CUDA 11.x or 12.x.

---

### AMD (OpenCL) Variant

**ArrayFire Libraries:**
| Library | Description |
|---------|-------------|
| `af.dll` | ArrayFire core library |
| `afopencl.dll` | OpenCL backend |
| `forge.dll` | ArrayFire visualization (optional) |

**OpenCL Libraries:**
| Library | Description |
|---------|-------------|
| `OpenCL.dll` | OpenCL ICD loader (ships with drivers) |
| `clFFT.dll` | clFFT (FFT for OpenCL) |
| `clBLAS.dll` | clBLAS (BLAS for OpenCL) |

**Note:** AMD users must have AMD drivers installed which provide OpenCL runtime.

---

### Intel (OpenCL) Variant

> **Important:** ArrayFire uses **OpenCL** for Intel GPU acceleration, **not oneAPI/SYCL**.
> Intel provides OpenCL support through their GPU drivers. The full oneAPI toolkit is NOT required.

**ArrayFire Libraries:**
| Library | Description |
|---------|-------------|
| `af.dll` | ArrayFire core library |
| `afopencl.dll` | OpenCL backend |
| `forge.dll` | ArrayFire visualization (optional) |

**OpenCL Libraries:**
| Library | Description |
|---------|-------------|
| `OpenCL.dll` | OpenCL ICD loader |
| `clFFT.dll` | clFFT (FFT for OpenCL) |
| `clBLAS.dll` | clBLAS (BLAS for OpenCL) |

**Intel GPU Driver Requirements:**
- **Integrated GPUs (UHD/Iris):** Intel Graphics Driver includes OpenCL runtime
- **Discrete GPUs (Arc):** Intel Arc Graphics Driver includes OpenCL runtime
- Download from: https://www.intel.com/content/www/us/en/download-center/home.html

**Note:** Users do NOT need to install the full Intel oneAPI toolkit. The standard Intel GPU driver provides OpenCL support. oneAPI is only needed if you want Intel MKL for CPU optimizations.

---

### CPU-Only Variant

**ArrayFire Libraries:**
| Library | Description |
|---------|-------------|
| `af.dll` | ArrayFire core library |
| `afcpu.dll` | CPU backend |
| `forge.dll` | ArrayFire visualization (optional) |

**CPU Math Libraries:**
| Library | Description |
|---------|-------------|
| `fftw3.dll` | FFTW3 (FFT operations) |
| `fftw3f.dll` | FFTW3 single precision |
| `libopenblas.dll` | OpenBLAS (BLAS/LAPACK) |

**Or Intel MKL (alternative to OpenBLAS/FFTW):**
| Library | Description |
|---------|-------------|
| `mkl_core.dll` | MKL core |
| `mkl_intel_thread.dll` | MKL threading |
| `mkl_rt.dll` | MKL runtime |

---

## Common Libraries (All Variants)

These libraries are required by all variants:

### GUI Libraries (Engine only)
| Library | Description |
|---------|-------------|
| `glfw3.dll` | Window/input management |
| `opengl32.dll` | OpenGL (system provided) |

### Networking Libraries
| Library | Description |
|---------|-------------|
| `grpc.dll` | gRPC core |
| `grpc++.dll` | gRPC C++ bindings |
| `libprotobuf.dll` | Protocol Buffers |
| `abseil_*.dll` | Abseil libraries (gRPC dependency) |
| `cares.dll` | c-ares (async DNS) |
| `re2.dll` | RE2 regex |
| `zlib1.dll` | Compression |

### Security Libraries
| Library | Description |
|---------|-------------|
| `libssl-*.dll` | OpenSSL SSL |
| `libcrypto-*.dll` | OpenSSL Crypto |

### Python Integration (Engine only)
| Library | Description |
|---------|-------------|
| `python3*.dll` | Python runtime |
| `python3.lib` | Python import library |

### C++ Runtime
| Library | Description |
|---------|-------------|
| `msvcp140.dll` | MSVC C++ runtime |
| `vcruntime140.dll` | MSVC runtime |
| `vcruntime140_1.dll` | MSVC runtime (additional) |

**Note:** Users may need to install Visual C++ Redistributable if not present.

---

## Build Configuration

### CMake Presets for Each Variant

Add these presets to `CMakePresets.json`:

```json
{
  "name": "windows-nvidia",
  "displayName": "Windows NVIDIA (CUDA)",
  "inherits": "windows-release",
  "cacheVariables": {
    "CYXWIZ_BACKEND": "CUDA",
    "ArrayFire_CUDA_BACKEND": "ON",
    "ArrayFire_OpenCL_BACKEND": "OFF",
    "ArrayFire_CPU_BACKEND": "ON"
  }
},
{
  "name": "windows-amd",
  "displayName": "Windows AMD (OpenCL)",
  "inherits": "windows-release",
  "cacheVariables": {
    "CYXWIZ_BACKEND": "OpenCL",
    "ArrayFire_CUDA_BACKEND": "OFF",
    "ArrayFire_OpenCL_BACKEND": "ON",
    "ArrayFire_CPU_BACKEND": "ON"
  }
},
{
  "name": "windows-cpu",
  "displayName": "Windows CPU Only",
  "inherits": "windows-release",
  "cacheVariables": {
    "CYXWIZ_BACKEND": "CPU",
    "ArrayFire_CUDA_BACKEND": "OFF",
    "ArrayFire_OpenCL_BACKEND": "OFF",
    "ArrayFire_CPU_BACKEND": "ON"
  }
}
```

### Build Scripts

Create `scripts/build-setup.bat`:

```batch
@echo off
REM Build all platform variants and create setup folders

echo Building NVIDIA variant...
cmake --preset windows-nvidia
cmake --build build/windows-nvidia --config Release

echo Building AMD variant...
cmake --preset windows-amd
cmake --build build/windows-amd --config Release

echo Building CPU variant...
cmake --preset windows-cpu
cmake --build build/windows-cpu --config Release

echo Creating setup folders...
call scripts\create-setup-folders.bat
```

---

## Packaging Script

Create `scripts/create-setup-folders.bat`:

```batch
@echo off
REM Create setup folders with appropriate libraries

set SETUP_DIR=setup
set AF_DIR=C:\Program Files\ArrayFire\v3

REM Clean previous setup
if exist %SETUP_DIR% rmdir /s /q %SETUP_DIR%

REM === NVIDIA Setup ===
echo Creating NVIDIA setup...
mkdir %SETUP_DIR%\nvidia\bin
mkdir %SETUP_DIR%\nvidia\lib

copy build\windows-nvidia\bin\Release\cyxwiz-engine.exe %SETUP_DIR%\nvidia\bin\
copy build\windows-nvidia\bin\Release\cyxwiz-server-node.exe %SETUP_DIR%\nvidia\bin\

REM ArrayFire CUDA libs
copy "%AF_DIR%\lib\af.dll" %SETUP_DIR%\nvidia\lib\
copy "%AF_DIR%\lib\afcuda.dll" %SETUP_DIR%\nvidia\lib\

REM CUDA runtime (from CUDA toolkit)
copy "%CUDA_PATH%\bin\cudart64_*.dll" %SETUP_DIR%\nvidia\lib\
copy "%CUDA_PATH%\bin\cublas64_*.dll" %SETUP_DIR%\nvidia\lib\
copy "%CUDA_PATH%\bin\cufft64_*.dll" %SETUP_DIR%\nvidia\lib\
copy "%CUDA_PATH%\bin\cusolver64_*.dll" %SETUP_DIR%\nvidia\lib\
copy "%CUDA_PATH%\bin\cusparse64_*.dll" %SETUP_DIR%\nvidia\lib\
copy "%CUDA_PATH%\bin\nvrtc64_*.dll" %SETUP_DIR%\nvidia\lib\

REM === AMD Setup ===
echo Creating AMD setup...
mkdir %SETUP_DIR%\amd\bin
mkdir %SETUP_DIR%\amd\lib

copy build\windows-amd\bin\Release\cyxwiz-engine.exe %SETUP_DIR%\amd\bin\
copy build\windows-amd\bin\Release\cyxwiz-server-node.exe %SETUP_DIR%\amd\bin\

REM ArrayFire OpenCL libs
copy "%AF_DIR%\lib\af.dll" %SETUP_DIR%\amd\lib\
copy "%AF_DIR%\lib\afopencl.dll" %SETUP_DIR%\amd\lib\

REM OpenCL libs
copy "%AF_DIR%\lib\clFFT.dll" %SETUP_DIR%\amd\lib\
copy "%AF_DIR%\lib\clBLAS.dll" %SETUP_DIR%\amd\lib\

REM === CPU Setup ===
echo Creating CPU setup...
mkdir %SETUP_DIR%\cpu\bin
mkdir %SETUP_DIR%\cpu\lib

copy build\windows-cpu\bin\Release\cyxwiz-engine.exe %SETUP_DIR%\cpu\bin\
copy build\windows-cpu\bin\Release\cyxwiz-server-node.exe %SETUP_DIR%\cpu\bin\

REM ArrayFire CPU libs
copy "%AF_DIR%\lib\af.dll" %SETUP_DIR%\cpu\lib\
copy "%AF_DIR%\lib\afcpu.dll" %SETUP_DIR%\cpu\lib\

REM CPU math libs
copy "%AF_DIR%\lib\fftw3.dll" %SETUP_DIR%\cpu\lib\
copy "%AF_DIR%\lib\libopenblas.dll" %SETUP_DIR%\cpu\lib\

REM === Common libs for all ===
for %%V in (nvidia amd cpu) do (
    echo Copying common libs to %%V...
    copy build\windows-%%V\bin\Release\*.dll %SETUP_DIR%\%%V\lib\ 2>nul
)

echo Setup folders created successfully!
dir /s %SETUP_DIR%
```

---

## Runtime Backend Selection

ArrayFire automatically selects the best available backend at runtime. The selection order is:
1. CUDA (if `afcuda.dll` present and NVIDIA GPU available)
2. OpenCL (if `afopencl.dll` present and OpenCL device available)
3. CPU (always available as fallback)

To force a specific backend in code:
```cpp
#include <arrayfire.h>

// Set backend explicitly
af::setBackend(AF_BACKEND_CUDA);    // Force CUDA
af::setBackend(AF_BACKEND_OPENCL);  // Force OpenCL
af::setBackend(AF_BACKEND_CPU);     // Force CPU

// Query available backends
int backends = af::getAvailableBackends();
bool hasCUDA = backends & AF_BACKEND_CUDA;
bool hasOpenCL = backends & AF_BACKEND_OPENCL;
bool hasCPU = backends & AF_BACKEND_CPU;
```

---

## Installer Options

### Option A: Separate Downloads
- User downloads only the variant they need
- Smallest download size per user
- Simple distribution

### Option B: Single Installer with Selection
- One installer that asks user about their GPU
- Copies only the required files
- Better UX for non-technical users

### Option C: Auto-Detection Installer
- Installer detects GPU hardware automatically
- Installs appropriate variant
- Most user-friendly

**Recommended:** Start with Option A for simplicity, move to Option C for production release.

---

## Hybrid Smart Installer (Recommended for Production)

The best user experience combines a small bootstrap installer with automatic hardware detection.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         User Download Flow                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. User visits cyxwiz.com/download                                      │
│     │                                                                    │
│     ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Website detects GPU via WebGL                                    │   │
│  │  "We detected: NVIDIA GeForce RTX 3080"                          │   │
│  │  [Download for NVIDIA] [Other Options ▼]                         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│     │                                                                    │
│     ▼                                                                    │
│  2. User downloads: CyxWiz-Setup.exe (~5 MB)                            │
│     │                                                                    │
│     ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Bootstrap Installer                                              │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │  Detecting hardware...                                      │  │   │
│  │  │  ✓ Found: NVIDIA GeForce RTX 3080 (CUDA 12.0)              │  │   │
│  │  │  ✓ Driver version: 545.92                                   │  │   │
│  │  │                                                             │  │   │
│  │  │  Recommended: CUDA variant (~500 MB)                        │  │   │
│  │  │  [Install Recommended]  [Choose Different ▼]                │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│     │                                                                    │
│     ▼                                                                    │
│  3. Installer downloads variant from CDN                                 │
│     │                                                                    │
│     ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Downloading CyxWiz for NVIDIA...                                 │   │
│  │  ████████████████████████░░░░░░░░  75%  (375 MB / 500 MB)        │   │
│  │  Speed: 25 MB/s  |  ETA: 5 seconds                               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│     │                                                                    │
│     ▼                                                                    │
│  4. Install complete!                                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Website GPU Detection (JavaScript)

Add this to the download page to auto-detect user's GPU:

```javascript
// detect-gpu.js - GPU detection for download page
function detectGPU() {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');

    if (!gl) {
        return { vendor: 'unknown', renderer: 'unknown', recommended: 'cpu' };
    }

    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    const vendor = debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : 'unknown';
    const renderer = debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : 'unknown';

    // Determine recommended variant
    let recommended = 'cpu';
    const rendererLower = renderer.toLowerCase();

    if (rendererLower.includes('nvidia') || rendererLower.includes('geforce') ||
        rendererLower.includes('rtx') || rendererLower.includes('gtx') ||
        rendererLower.includes('quadro') || rendererLower.includes('tesla')) {
        recommended = 'nvidia';
    } else if (rendererLower.includes('amd') || rendererLower.includes('radeon') ||
               rendererLower.includes('rx ')) {
        recommended = 'amd';
    } else if (rendererLower.includes('intel') || rendererLower.includes('iris') ||
               rendererLower.includes('uhd') || rendererLower.includes('hd graphics')) {
        recommended = 'intel';
    }

    return { vendor, renderer, recommended };
}

// Usage on download page
const gpu = detectGPU();
console.log(`Detected: ${gpu.renderer}`);
console.log(`Recommended variant: ${gpu.recommended}`);

// Update download button
document.getElementById('download-btn').href =
    `https://cdn.cyxwiz.com/releases/latest/CyxWiz-${gpu.recommended}.exe`;
document.getElementById('gpu-info').textContent =
    `Detected: ${gpu.renderer} - Downloading ${gpu.recommended.toUpperCase()} variant`;
```

### Bootstrap Installer (C++)

The small bootstrap installer detects hardware and downloads the appropriate variant:

```cpp
// bootstrap_installer.cpp
#include <windows.h>
#include <dxgi.h>
#include <string>
#include <vector>

#pragma comment(lib, "dxgi.lib")

struct GPUInfo {
    std::wstring name;
    std::wstring vendor;
    size_t dedicated_memory;
    enum class Type { NVIDIA, AMD, INTEL, OTHER } type;
};

std::vector<GPUInfo> DetectGPUs() {
    std::vector<GPUInfo> gpus;

    IDXGIFactory* factory = nullptr;
    if (FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&factory))) {
        return gpus;
    }

    IDXGIAdapter* adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC desc;
        adapter->GetDesc(&desc);

        GPUInfo info;
        info.name = desc.Description;
        info.dedicated_memory = desc.DedicatedVideoMemory;

        // Detect vendor by PCI vendor ID
        switch (desc.VendorId) {
            case 0x10DE: // NVIDIA
                info.type = GPUInfo::Type::NVIDIA;
                info.vendor = L"NVIDIA";
                break;
            case 0x1002: // AMD
                info.type = GPUInfo::Type::AMD;
                info.vendor = L"AMD";
                break;
            case 0x8086: // Intel
                info.type = GPUInfo::Type::INTEL;
                info.vendor = L"Intel";
                break;
            default:
                info.type = GPUInfo::Type::OTHER;
                info.vendor = L"Unknown";
        }

        gpus.push_back(info);
        adapter->Release();
    }

    factory->Release();
    return gpus;
}

std::string GetRecommendedVariant(const std::vector<GPUInfo>& gpus) {
    // Prioritize discrete GPUs
    for (const auto& gpu : gpus) {
        // Skip integrated Intel if discrete GPU available
        if (gpu.dedicated_memory > 1024 * 1024 * 1024) { // > 1GB VRAM
            switch (gpu.type) {
                case GPUInfo::Type::NVIDIA: return "nvidia";
                case GPUInfo::Type::AMD: return "amd";
                case GPUInfo::Type::INTEL: return "intel";
                default: break;
            }
        }
    }

    // Fallback to any GPU
    for (const auto& gpu : gpus) {
        switch (gpu.type) {
            case GPUInfo::Type::NVIDIA: return "nvidia";
            case GPUInfo::Type::AMD: return "amd";
            case GPUInfo::Type::INTEL: return "intel";
            default: break;
        }
    }

    return "cpu";
}

// Download URLs
const std::map<std::string, std::string> DOWNLOAD_URLS = {
    {"nvidia", "https://cdn.cyxwiz.com/releases/latest/cyxwiz-nvidia.zip"},
    {"amd", "https://cdn.cyxwiz.com/releases/latest/cyxwiz-amd.zip"},
    {"intel", "https://cdn.cyxwiz.com/releases/latest/cyxwiz-intel.zip"},
    {"cpu", "https://cdn.cyxwiz.com/releases/latest/cyxwiz-cpu.zip"}
};
```

### Startup Mismatch Detection

Add this check to Engine/Server Node startup to warn if wrong variant installed:

```cpp
// hardware_check.cpp - Add to application startup
#include <arrayfire.h>
#include <spdlog/spdlog.h>

void CheckHardwareMismatch() {
    int available = af::getAvailableBackends();

    bool has_cuda = available & AF_BACKEND_CUDA;
    bool has_opencl = available & AF_BACKEND_OPENCL;
    bool has_cpu = available & AF_BACKEND_CPU;

    // Detect actual hardware
    auto gpus = DetectGPUs(); // From bootstrap code above
    bool has_nvidia_hw = false;
    bool has_amd_hw = false;

    for (const auto& gpu : gpus) {
        if (gpu.type == GPUInfo::Type::NVIDIA) has_nvidia_hw = true;
        if (gpu.type == GPUInfo::Type::AMD) has_amd_hw = true;
    }

    // Check for mismatches
    if (has_nvidia_hw && !has_cuda) {
        spdlog::warn("NVIDIA GPU detected but CUDA backend not available!");
        spdlog::warn("You may have installed the wrong variant.");
        spdlog::warn("Download the NVIDIA variant for better performance.");

        // Show dialog to user
        ShowMismatchDialog(
            "Hardware Mismatch Detected",
            "You have an NVIDIA GPU but installed a non-CUDA version.\n"
            "For best performance, download the NVIDIA variant from cyxwiz.com"
        );
    }

    if (has_amd_hw && !has_opencl && has_cuda) {
        spdlog::warn("AMD GPU detected but OpenCL backend not available!");
        spdlog::warn("You may have installed the NVIDIA variant on an AMD system.");
    }

    // Log what's available
    spdlog::info("Available backends: CUDA={}, OpenCL={}, CPU={}",
                 has_cuda, has_opencl, has_cpu);
}
```

### CDN/Hosting Structure

```
cdn.cyxwiz.com/
├── releases/
│   ├── latest/                      # Symlinks to latest version
│   │   ├── cyxwiz-nvidia.zip
│   │   ├── cyxwiz-amd.zip
│   │   ├── cyxwiz-intel.zip
│   │   ├── cyxwiz-cpu.zip
│   │   ├── CyxWiz-Setup.exe         # Bootstrap installer
│   │   └── checksums.sha256
│   │
│   ├── v1.0.0/
│   │   ├── cyxwiz-nvidia.zip
│   │   ├── cyxwiz-amd.zip
│   │   ├── cyxwiz-intel.zip
│   │   ├── cyxwiz-cpu.zip
│   │   └── checksums.sha256
│   │
│   └── v1.1.0/
│       └── ...
│
└── metadata/
    └── releases.json                 # Version info, sizes, checksums
```

### releases.json Format

```json
{
  "latest": "1.1.0",
  "releases": {
    "1.1.0": {
      "date": "2025-12-19",
      "variants": {
        "nvidia": {
          "url": "https://cdn.cyxwiz.com/releases/v1.1.0/cyxwiz-nvidia.zip",
          "size": 524288000,
          "size_human": "500 MB",
          "sha256": "abc123...",
          "cuda_version": "12.0",
          "min_driver": "525.60"
        },
        "amd": {
          "url": "https://cdn.cyxwiz.com/releases/v1.1.0/cyxwiz-amd.zip",
          "size": 209715200,
          "size_human": "200 MB",
          "sha256": "def456...",
          "opencl_version": "3.0",
          "min_driver": "23.5.1"
        },
        "intel": {
          "url": "https://cdn.cyxwiz.com/releases/v1.1.0/cyxwiz-intel.zip",
          "size": 209715200,
          "size_human": "200 MB",
          "sha256": "ghi789..."
        },
        "cpu": {
          "url": "https://cdn.cyxwiz.com/releases/v1.1.0/cyxwiz-cpu.zip",
          "size": 125829120,
          "size_human": "120 MB",
          "sha256": "jkl012..."
        }
      },
      "changelog": "https://cyxwiz.com/changelog/1.1.0"
    }
  }
}
```

### Implementation Phases

| Phase | Description | Priority |
|-------|-------------|----------|
| **Phase 1** | Separate ZIP downloads per variant | MVP |
| **Phase 2** | Website GPU detection + smart download buttons | High |
| **Phase 3** | Bootstrap installer with auto-detection | Medium |
| **Phase 4** | In-app mismatch warning | Medium |
| **Phase 5** | Auto-update with variant awareness | Low |

---

## Size Estimates

| Variant | Estimated Size | Notes |
|---------|---------------|-------|
| NVIDIA (CUDA) | ~500-800 MB | CUDA runtime is large |
| AMD (OpenCL) | ~150-250 MB | OpenCL libs are smaller |
| Intel (OpenCL) | ~150-250 MB | Same as AMD |
| CPU Only | ~100-150 MB | Smallest variant |

**Current single build:** ~1-1.5 GB (all backends)

---

## TODO

- [ ] Create CMake presets for each variant
- [ ] Implement `scripts/build-setup.bat`
- [ ] Implement `scripts/create-setup-folders.bat`
- [ ] Test each variant on appropriate hardware
- [ ] Create NSIS/WiX installer scripts
- [ ] Add auto-detection for GPU hardware
- [ ] Document minimum driver versions for each platform
- [ ] Add checksum verification for downloads

---

## Related Documentation

- [Main Build Documentation](mainbuild.md)
- [ArrayFire Installation](https://arrayfire.org/docs/installing.htm)
- [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads)
- [AMD ROCm/OpenCL](https://rocm.docs.amd.com/)
- [Intel Graphics Drivers](https://www.intel.com/content/www/us/en/download-center/home.html) (includes OpenCL runtime)
- [Intel Arc Drivers](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/software/drivers.html)

### ArrayFire Backend Notes

| GPU Vendor | ArrayFire Backend | Runtime Required |
|------------|------------------|------------------|
| NVIDIA | CUDA (`afcuda.dll`) | CUDA Toolkit |
| AMD | OpenCL (`afopencl.dll`) | AMD GPU Driver (includes OpenCL) |
| Intel | OpenCL (`afopencl.dll`) | Intel GPU Driver (includes OpenCL) |
| CPU | CPU (`afcpu.dll`) | OpenBLAS/FFTW or Intel MKL |

> **Note:** ArrayFire does NOT currently have a oneAPI/SYCL backend. Intel GPU support is via OpenCL only.
> Intel has been developing SYCL as a future direction, but ArrayFire's stable Intel GPU path remains OpenCL.

---

*Last updated: 2025-12-19*
