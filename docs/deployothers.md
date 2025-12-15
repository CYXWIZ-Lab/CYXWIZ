# Deploying Other Model Formats

This document outlines the support for additional model formats on the CyxWiz Server Node beyond the native `.cyxmodel` format.

## Currently Supported

| Format | Status | Description |
|--------|--------|-------------|
| `.cyxmodel` | **Working** | CyxWiz native format (binary + directory) |
| `.onnx` | **Working** | ONNX Runtime with CUDA/CPU execution providers |
| `.safetensors` | **Working** | HuggingFace safe tensor format (export only) |
| `.gguf` | Stub | GGML format for LLMs (future) |
| `.pt/.pth` | Stub | PyTorch models via LibTorch (future) |

---

# ONNX Support Implementation (Completed)

## Overview
ONNX support has been implemented in CyxWiz, enabling:
- **Import**: Load `.onnx` files for inference on Server Node with CUDA/CPU execution providers
- **Export**: Convert trained CyxWiz Sequential models to ONNX format

**Build Requirements:**
- vcpkg packages: `onnxruntime-gpu` (Windows x64) or `onnxruntime` (other platforms), `onnx`
- CMake option: `CYXWIZ_ENABLE_ONNX=ON` (enabled by default)

---

## Phase 1: Dependencies & Build System

### 1.1 Update vcpkg.json
```json
{
  "dependencies": [
    ...existing...,
    {
      "name": "onnxruntime-gpu",
      "platform": "windows & x64"
    },
    {
      "name": "onnxruntime",
      "platform": "!(windows & x64)"
    },
    "onnx"
  ]
}
```

### 1.2 Update CMakeLists.txt (root)
- Add `option(CYXWIZ_ENABLE_ONNX "Enable ONNX Runtime support" ON)`
- Add `find_package(onnxruntime CONFIG QUIET)` with fallback handling
- Set `CYXWIZ_HAS_ONNX` compile definition

### 1.3 Update cyxwiz-server-node/CMakeLists.txt
- Link `onnxruntime::onnxruntime` to daemon and GUI targets
- Add `CYXWIZ_HAS_ONNX` compile definition

### 1.4 Update cyxwiz-engine/CMakeLists.txt
- Link `onnx` and `onnx_proto` for export capability
- Add `CYXWIZ_HAS_ONNX` compile definition

---

## Phase 2: Server Node - ONNX Import

### Files to Modify:
- `cyxwiz-server-node/src/model_loader.cpp` (lines 50-148)
- `cyxwiz-server-node/src/model_loader.h`

### Implementation:

**ONNXLoader::Impl class additions:**
```cpp
#ifdef CYXWIZ_HAS_ONNX
std::unique_ptr<Ort::Env> env;
std::unique_ptr<Ort::Session> session;
std::unique_ptr<Ort::SessionOptions> session_options;
std::vector<std::string> input_names_str, output_names_str;
std::vector<const char*> input_names, output_names;
bool using_cuda = false;
#endif
```

**Load() implementation:**
1. Create ONNX Runtime environment
2. Configure session with CUDA EP (try/catch fallback to CPU)
3. Create session from model file
4. Extract input/output specs from model metadata
5. Cache I/O names for inference

**Infer() implementation:**
1. Convert CyxWiz::Tensor to Ort::Value
2. Run session with cached I/O names
3. Convert outputs back to CyxWiz::Tensor

---

## Phase 3: Engine - ONNX Export

### Files to Modify:
- `cyxwiz-engine/src/core/model_exporter.cpp`
- `cyxwiz-engine/src/core/model_exporter.h`

### Implementation:

**ExportONNX() using ONNX protobuf:**
1. Create ModelProto with IR version, producer info
2. Set opset version from options
3. Iterate model modules, map to ONNX ops
4. Add weights as initializers
5. Serialize to file

**Layer Mapping:**
| CyxWiz | ONNX Op | Notes |
|--------|---------|-------|
| Linear | Gemm | transB=1 for weight transpose |
| ReLU | Relu | Direct mapping |
| Sigmoid | Sigmoid | Direct mapping |
| Tanh | Tanh | Direct mapping |
| Softmax | Softmax | axis=-1 |
| Dropout | Identity/Skip | Skip in inference mode |
| LeakyReLU | LeakyRelu | alpha attribute |
| Flatten | Flatten | axis=1 |

---

## Phase 4: Testing

### Test Files to Create:
- `cyxwiz-server-node/tests/test_onnx_loader.cpp`
- `cyxwiz-engine/tests/test_onnx_export.cpp`

### Test Cases:
1. Load ONNX Model Zoo models (ResNet, MNIST)
2. Verify input/output specs extraction
3. Run inference with test data
4. Export CyxWiz model -> load in ONNXLoader -> compare outputs

---

## File Summary

| File | Action | Description |
|------|--------|-------------|
| `vcpkg.json` | Modify | Add onnxruntime-gpu, onnx |
| `CMakeLists.txt` | Modify | Add ONNX option and find_package |
| `cyxwiz-server-node/CMakeLists.txt` | Modify | Link onnxruntime |
| `cyxwiz-engine/CMakeLists.txt` | Modify | Link onnx proto |
| `cyxwiz-server-node/src/model_loader.cpp` | Modify | Implement ONNXLoader |
| `cyxwiz-server-node/src/model_loader.h` | Modify | Add Impl members |
| `cyxwiz-engine/src/core/model_exporter.cpp` | Modify | Implement ExportONNX |
| `cyxwiz-server-node/tests/test_onnx_loader.cpp` | Create | Unit tests |
| `cyxwiz-engine/tests/test_onnx_export.cpp` | Create | Unit tests |

---

## Implementation Order

1. **vcpkg.json** - Add dependencies
2. **Root CMakeLists.txt** - Add ONNX option
3. **cyxwiz-server-node/CMakeLists.txt** - Link library
4. **model_loader.cpp** - Implement ONNXLoader::Load()
5. **model_loader.cpp** - Implement ONNXLoader::Infer()
6. **Test import** - Verify with ONNX Model Zoo
7. **cyxwiz-engine/CMakeLists.txt** - Link onnx proto
8. **model_exporter.cpp** - Implement ExportONNX()
9. **Test export** - Roundtrip verification

---

## Implementation Status

| Step | Status | Notes |
|------|--------|-------|
| 1. vcpkg.json dependencies | **Done** | onnxruntime-gpu, onnxruntime, onnx |
| 2. Root CMakeLists.txt | **Done** | CYXWIZ_HAS_ONNX, CYXWIZ_HAS_ONNX_EXPORT |
| 3. Server Node CMakeLists.txt | **Done** | Linked for daemon and GUI targets |
| 4. ONNXLoader::Load() | **Done** | CUDA EP + CPU fallback |
| 5. ONNXLoader::Infer() | **Done** | Tensor conversion implemented |
| 6. Engine CMakeLists.txt | **Done** | ONNX proto linked |
| 7. ExportONNX() | **Done** | Full ONNX graph generation |
| 8. Unit tests | Pending | Create test files |

---

# Future: GGUF Support (LLMs)

For supporting large language models via llama.cpp:

**Dependencies:**
- llama.cpp library

**Use Cases:**
- Deploy quantized LLMs (Llama, Mistral, etc.)
- Text generation inference
- Embeddings extraction

---

# Future: PyTorch Support

For native PyTorch model loading via LibTorch:

**Dependencies:**
- LibTorch (PyTorch C++ frontend)

**Use Cases:**
- Load `.pt` / `.pth` files directly
- TorchScript models
- Research model deployment
