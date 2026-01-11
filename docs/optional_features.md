# CyxWiz Optional Features

The pre-built release binaries include core functionality. For advanced ML inference features, you can build from source with additional dependencies.

---

## Feature Comparison

| Feature | Pre-built Binary | Full Build |
|---------|-----------------|------------|
| GUI (ImGui, ImPlot) | ✅ | ✅ |
| Networking (gRPC) | ✅ | ✅ |
| GPU Training (ArrayFire) | ✅ | ✅ |
| ONNX Model Inference | ❌ | ✅ |
| GGUF/LLM Inference | ❌ | ✅ |
| PyTorch Model Loading | ❌ | ✅ |
| Python Scripting | ❌ | ✅ |

---

## Enabling Optional Features

### Option 1: Use Full vcpkg.json (Recommended)

The repository includes two vcpkg manifests:
- `vcpkg-ci.json` - Minimal dependencies (used for CI builds)
- `vcpkg.json` - Full dependencies (use this for local builds)

Simply build with the default `vcpkg.json`:

```bash
# Clone repository
git clone https://github.com/CYXWIZ-Lab/CYXWIZ.git
cd CYXWIZ

# Setup vcpkg
git clone https://github.com/microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh  # or .bat on Windows

# Build with all features
cmake -B build \
  -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release --parallel
```

### Option 2: Enable Specific Features

You can enable individual features via CMake options:

```bash
cmake -B build \
  -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DCYXWIZ_ENABLE_ONNX=ON \
  -DCYXWIZ_ENABLE_GGUF=ON \
  -DCYXWIZ_ENABLE_PYTORCH=ON \
  -DCYXWIZ_ENABLE_PYTHON=ON
```

---

## Installing Optional Dependencies

### ONNX Runtime (Model Inference)

Enables loading and running ONNX models for inference.

**Windows:**
```powershell
# Via vcpkg (CPU)
./vcpkg/vcpkg install onnxruntime

# Or download GPU version manually
# https://github.com/microsoft/onnxruntime/releases
# Extract to: external/onnxruntime-gpu/
```

**Linux:**
```bash
./vcpkg/vcpkg install onnxruntime
```

**macOS:**
```bash
# Download from GitHub releases (vcpkg has compatibility issues)
# https://github.com/microsoft/onnxruntime/releases
# Extract to: third_party/onnxruntime/
```

**CMake:**
```bash
cmake -B build -DCYXWIZ_ENABLE_ONNX=ON ...
```

---

### llama.cpp / GGUF (LLM Inference)

Enables loading and running GGUF models (LLaMA, Mistral, etc.).

**Windows:**
```powershell
# CPU only
./vcpkg/vcpkg install llama-cpp

# With CUDA support
./vcpkg/vcpkg install llama-cpp[cuda]
```

**Linux:**
```bash
# CPU only
./vcpkg/vcpkg install llama-cpp

# With CUDA support
./vcpkg/vcpkg install llama-cpp[cuda]
```

**macOS:**
```bash
# With Metal support (Apple Silicon)
./vcpkg/vcpkg install llama-cpp[metal]
```

**CMake:**
```bash
cmake -B build -DCYXWIZ_ENABLE_GGUF=ON ...
```

---

### LibTorch / PyTorch (Model Loading)

Enables loading PyTorch models and TorchScript.

**All Platforms:**

1. Download LibTorch from https://pytorch.org/get-started/locally/
   - Select: LibTorch → C++/Java → Your OS → CUDA version (or CPU)

2. Extract to a known location:
   ```bash
   # Example
   /opt/libtorch/        # Linux
   C:\libtorch\          # Windows
   ~/libtorch/           # macOS
   ```

3. Set environment variable:
   ```bash
   export TORCH_DIR=/opt/libtorch  # Linux/macOS
   $env:TORCH_DIR = "C:\libtorch"  # Windows PowerShell
   ```

**CMake:**
```bash
cmake -B build -DCYXWIZ_ENABLE_PYTORCH=ON -DTORCH_DIR=/opt/libtorch ...
```

---

### Python Scripting

Enables the embedded Python interpreter for scripting.

**Windows:**
```powershell
# Install Python 3.8+
winget install Python.Python.3.11

# vcpkg will find pybind11
./vcpkg/vcpkg install pybind11
```

**Linux:**
```bash
sudo apt install python3-dev python3-pip
./vcpkg/vcpkg install pybind11
```

**macOS:**
```bash
brew install python@3.11
./vcpkg/vcpkg install pybind11
```

**CMake:**
```bash
cmake -B build -DCYXWIZ_ENABLE_PYTHON=ON ...
```

---

## Full Build Example

Build with all optional features enabled:

```bash
# Setup
git clone https://github.com/CYXWIZ-Lab/CYXWIZ.git
cd CYXWIZ
git clone https://github.com/microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh

# Install ArrayFire (required)
# See INSTALL.md for platform-specific instructions

# Configure with all features
cmake -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCYXWIZ_ENABLE_ONNX=ON \
  -DCYXWIZ_ENABLE_GGUF=ON \
  -DCYXWIZ_ENABLE_PYTORCH=ON \
  -DCYXWIZ_ENABLE_PYTHON=ON

# Build (this will take 1-2 hours first time)
cmake --build build --parallel

# Run
./build/bin/cyxwiz-engine
```

---

## Verifying Features

After building, you can verify which features are enabled:

```bash
# Check CMake configuration output
cmake -B build ... 2>&1 | grep -E "ONNX|GGUF|PyTorch|Python"
```

Expected output:
```
-- ONNX Runtime found (CONFIG) - ONNX support enabled
-- llama.cpp found (CONFIG) - GGUF support enabled
-- LibTorch found (CONFIG) - PyTorch support enabled
```

---

## Troubleshooting

### "ONNX Runtime not found"
- Ensure onnxruntime is installed via vcpkg or manually
- Check `CMAKE_PREFIX_PATH` includes the install location

### "llama.cpp not found"
- Run `./vcpkg/vcpkg install llama-cpp`
- For GPU: `./vcpkg/vcpkg install llama-cpp[cuda]`

### "LibTorch not found"
- Download from pytorch.org
- Set `TORCH_DIR` environment variable
- Or pass `-DTORCH_DIR=/path/to/libtorch` to cmake

### "Python not found"
- Install Python 3.8+ development headers
- Linux: `sudo apt install python3-dev`
- macOS: `brew install python@3.11`

---

## Related Documentation

- [INSTALL.md](../INSTALL.md) - Installation guide
- [mainbuild.md](mainbuild.md) - Build from source
- [key_prob_soln.md](key_prob_soln.md) - Troubleshooting
