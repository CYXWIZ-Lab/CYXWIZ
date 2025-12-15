# ONNX Runtime GPU Support

This document covers ONNX Runtime GPU configuration for CyxWiz across Windows, Linux, and macOS.

## Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| ONNX Runtime | 1.19.2+ | Via vcpkg (onnxruntime-gpu) |
| CUDA Toolkit | 12.6+ | Required for ONNX Runtime 1.19+ |
| cuDNN | 9.17 (12.9 binaries) | From `C:\Program Files\NVIDIA\CUDNN\v9.17\bin\12.9\` |
| GPU | Compute 7.0+ recommended | See compatibility notes below |

**Current CyxWiz Configuration** (tested):
- CUDA Toolkit: 12.6
- cuDNN: 9.17 with 12.9 binaries
- ONNX Runtime: 1.19.2 (vcpkg)

## Installation

### Windows

1. **Install CUDA Toolkit 12.x**
   ```
   Download from: https://developer.nvidia.com/cuda-downloads
   Default path: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\
   ```

2. **Install cuDNN 9.x**
   ```
   Download from: https://developer.nvidia.com/cudnn
   Default path: C:\Program Files\NVIDIA\CUDNN\v9.x\
   ```

3. **Required DLLs** (copy to build output or add to PATH):
   ```
   From CUDA (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin\):
   - cublas64_12.dll
   - cublasLt64_12.dll
   - cudart64_12.dll
   - cufft64_11.dll
   - curand64_10.dll

   From cuDNN (C:\Program Files\NVIDIA\CUDNN\v9.x\bin\12.x\):
   - cudnn64_9.dll
   - cudnn_adv64_9.dll
   - cudnn_cnn64_9.dll
   - cudnn_engines_precompiled64_9.dll
   - cudnn_engines_runtime_compiled64_9.dll
   - cudnn_graph64_9.dll
   - cudnn_heuristic64_9.dll
   - cudnn_ops64_9.dll

   From vcpkg (build/vcpkg_installed/x64-windows/bin/):
   - onnxruntime.dll
   - onnxruntime_providers_cuda.dll
   - onnxruntime_providers_shared.dll
   - onnxruntime_providers_tensorrt.dll (optional)
   ```

### Linux

1. **Install CUDA Toolkit 12.x**
   ```bash
   # Ubuntu/Debian
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt update
   sudo apt install cuda-toolkit-12-6

   # Add to PATH
   export PATH=/usr/local/cuda-12.6/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
   ```

2. **Install cuDNN 9.x**
   ```bash
   # Download from NVIDIA and extract
   tar -xvf cudnn-linux-x86_64-9.x.x.x_cuda12-archive.tar.xz
   sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
   sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
   sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
   ```

3. **Required libraries**:
   ```
   libonnxruntime.so
   libonnxruntime_providers_cuda.so
   libonnxruntime_providers_shared.so
   libcudnn.so.9
   libcublas.so.12
   libcublasLt.so.12
   ```

### macOS

macOS does not support NVIDIA CUDA. ONNX Runtime uses CPU or CoreML execution providers.

```bash
# Install via vcpkg
vcpkg install onnxruntime

# Or via Homebrew
brew install onnxruntime
```

## GPU Architecture Compatibility

### Supported Architectures

| Architecture | Compute Capability | GPUs | Status |
|--------------|-------------------|------|--------|
| Volta | 7.0, 7.2 | Tesla V100, Titan V | Full support |
| Turing | 7.5 | RTX 20xx, GTX 16xx | Full support |
| Ampere | 8.0, 8.6 | RTX 30xx, A100 | Full support |
| Ada Lovelace | 8.9 | RTX 40xx | Full support |
| Hopper | 9.0 | H100 | Full support |

### Limited Support (Pascal Architecture)

| Architecture | Compute Capability | GPUs | Status |
|--------------|-------------------|------|--------|
| Pascal | 6.0, 6.1, 6.2 | GTX 10xx, Titan X/Xp | **Partial** |

**Pascal Limitations**:
- MLP models (Gemm operations): **Work on CUDA**
- CNN models (Conv2D operations): **Fail with cuDNN error**

**Root Cause**: The pre-built ONNX Runtime binaries from vcpkg are compiled by Microsoft targeting newer GPU architectures (Volta 7.0+). Pascal architecture (compute 6.1) CUDA kernels are not included in these builds.

**Solutions**:
1. **Use CPU for CNN models** - Fast and reliable (0.05ms/sample for MNIST)
2. **Build ONNX Runtime from source** with `-DCMAKE_CUDA_ARCHITECTURES=61`
3. **Upgrade GPU** to Turing (GTX 16xx/RTX 20xx) or newer

### Error Message (Pascal + CNN)

```
CUDNN failure 5003: CUDNN_STATUS_EXECUTION_FAILED_CUDART
file=...onnxruntime\contrib_ops\cuda\fused_conv.cc
expr=cudnnConvolutionForward(...)
```

## Usage in CyxWiz

### ONNXLoader API

```cpp
#include "model_loader.h"

// Default: tries CUDA first, falls back to CPU
ONNXLoader loader;
loader.Load("model.onnx");

// Force CPU execution (for CNN models on Pascal GPUs)
ONNXLoader loader;
loader.SetForceCPU(true);  // Call before Load()
loader.Load("cnn_model.onnx");

// Check execution provider
if (loader.IsUsingCUDA()) {
    std::cout << "Running on GPU\n";
} else {
    std::cout << "Running on CPU\n";
}
```

### test_mnist_onnx Options

```bash
# Default (tries CUDA)
./test_mnist_onnx --model mnist.onnx --samples 50

# Force CPU execution
./test_mnist_onnx --model mnist.onnx --cpu

# Disable graph optimizations (avoids FusedConv)
./test_mnist_onnx --model mnist.onnx --no-opt
```

## Troubleshooting

### DLL Not Found (Windows)

**Error**: `onnxruntime_providers_cuda.dll not found`

**Solution**: Copy DLLs from vcpkg to build output:
```powershell
Copy-Item "build\vcpkg_installed\x64-windows\bin\onnxruntime*.dll" "build\bin\Release\"
```

### cuDNN Version Mismatch

**Error**: `CUDA execution provider not available`

**Solution**: Ensure cuDNN version matches CUDA version:
- CUDA 12.x requires cuDNN 9.x from the `12.x` subfolder
- Check: `C:\Program Files\NVIDIA\CUDNN\v9.x\bin\12.x\` (not just `bin\`)

### Wrong cuDNN in PATH

**Error**: cuDNN from another application (e.g., ArrayFire) takes precedence

**Solution**: Copy full NVIDIA cuDNN to build directory:
```powershell
Copy-Item "C:\Program Files\NVIDIA\CUDNN\v9.17\bin\12.9\cudnn*.dll" "build\bin\Release\"
```

### Pascal GPU + CNN Model Failure

**Error**: `CUDNN_STATUS_EXECUTION_FAILED_CUDART (5003)`

**Solution**: Use CPU for CNN models:
```cpp
ONNXLoader loader;
loader.SetForceCPU(true);
loader.Load("cnn_model.onnx");
```

Or via command line:
```bash
./test_mnist_onnx --cpu
```

## Performance Comparison

Tested on GTX 1050 Ti (Pascal, 4GB VRAM):

### MNIST CNN Model (28x28 input, 10 classes)
| Execution Provider | Avg Latency | Status |
|-------------------|-------------|--------|
| CUDA | N/A | Fails (Pascal not supported) |
| CPU | 0.046 ms | Works perfectly |

### MLP Models (Gemm operations)
| Execution Provider | Avg Latency | Status |
|-------------------|-------------|--------|
| CUDA | ~0.5 ms | Works |
| CPU | ~0.04 ms | Works |

**Note**: For small models, CPU is often faster due to GPU memory transfer overhead. CUDA benefits larger models with more compute.

## CMake Configuration

The root `CMakeLists.txt` handles ONNX Runtime detection:

```cmake
# Enable ONNX support
option(CYXWIZ_ENABLE_ONNX "Enable ONNX Runtime support" ON)

# Automatic detection
if(CYXWIZ_ENABLE_ONNX)
    find_package(onnxruntime CONFIG QUIET)
    if(NOT onnxruntime_FOUND)
        find_package(ONNXRuntime QUIET)  # Custom FindONNXRuntime.cmake
    endif()
endif()

# Result variables
# CYXWIZ_HAS_ONNX - Runtime available
# CYXWIZ_HAS_ONNX_EXPORT - Export (protobuf) available
```

## References

- [ONNX Runtime CUDA EP Documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus)
- [cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/installation/overview.html)
- [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases)
