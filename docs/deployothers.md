# Deploying Other Model Formats

This document outlines the support for additional model formats on the CyxWiz Server Node beyond the native `.cyxmodel` format.

## Currently Supported

| Format | Status | Description |
|--------|--------|-------------|
| `.cyxmodel` | **Working** | CyxWiz native format (binary + directory) |
| `.onnx` | **Working** | ONNX Runtime with CUDA/CPU execution providers |
| `.safetensors` | **Working** | HuggingFace safe tensor format (export only) |
| `.gguf` | **Working** | llama.cpp for LLMs with CUDA/Metal/CPU backends |
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

# GGUF Support Implementation (Completed)

## Overview
GGUF support has been implemented in CyxWiz using llama.cpp, enabling:
- **Text Generation**: Load GGUF models for chat/completion inference
- **Embeddings**: Extract vector embeddings for semantic search/RAG
- **GPU Acceleration**: CUDA (NVIDIA), Metal (macOS), Vulkan support

**Build Requirements:**
- vcpkg packages: `llama-cpp` (GPU support auto-detected at build time)
- CMake option: `CYXWIZ_ENABLE_GGUF=ON` (enabled by default)

---

## Key Features

### Configuration (call before Load)
- `SetContextSize(int n_ctx)` - Context window (512 to 128K)
- `SetGPULayers(int n_gpu_layers)` - GPU offloading (0 = CPU only)
- `SetThreads(int n_threads)` - CPU parallelism

### Sampling Parameters
- `SetTemperature(float)` - Randomness (0 = deterministic, 2 = creative)
- `SetMaxTokens(int)` - Max generation length
- `SetTopP(float)` - Nucleus sampling
- `SetTopK(int)` - Top-K sampling
- `SetRepeatPenalty(float)` - Repetition control

### Inference Modes
| Input Key | Mode | Description |
|-----------|------|-------------|
| `prompt` or `text` | Text Generation | Generate completion text |
| `text` + embedding model | Embeddings | Extract embedding vector |
| `input_ids` | Token Generation | Raw token ID input/output |

---

## Files Modified

| File | Changes |
|------|---------|
| `vcpkg.json` | Added llama-cpp dependency (platform-specific) |
| `cmake/FindLlamaCpp.cmake` | **NEW** - CMake find module for llama.cpp |
| `CMakeLists.txt` | Added CYXWIZ_ENABLE_GGUF option |
| `cyxwiz-server-node/CMakeLists.txt` | Link llama target |
| `cyxwiz-server-node/src/model_loader.h` | GGUFLoader config methods |
| `cyxwiz-server-node/src/model_loader.cpp` | Full GGUFLoader implementation |
| `cyxwiz-server-node/src/http/openai_api_server.cpp` | Added `/v1/completions` endpoint |
| `cyxwiz-engine/src/gui/panels/deployment_dialog.h` | GGUF config members |
| `cyxwiz-engine/src/gui/panels/deployment_dialog.cpp` | RenderGGUFConfig() UI |

---

## Deployment Dialog UI

When a `.gguf` file is selected, the deployment dialog shows:
- **GPU Layers slider** (0-100)
- **Context Size dropdown** (512 to 128K presets)
- **Temperature slider** (0.0-2.0)
- **Max Tokens slider** (16-4096)
- **Advanced Sampling** (collapsible):
  - Top-P, Top-K, Repeat Penalty
- **Embedding Mode checkbox**

---

## HTTP REST API Endpoints

The Server Node exposes OpenAI-compatible REST API endpoints for inference:

### GET /health
Health check endpoint.
```bash
curl http://localhost:8080/health
```

### GET /v1/deployments
List all active deployments.
```bash
curl http://localhost:8080/v1/deployments
```

### POST /v1/completions
Text completion endpoint for LLM models (GGUF).

**Request:**
```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "deployment_id": "dep_xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "prompt": "Tell me a joke:",
    "max_tokens": 100,
    "temperature": 0.8
  }'
```

**Response:**
```json
{
  "id": "cmpl-xxxxxxxx",
  "object": "text_completion",
  "created": 1765808157,
  "model": "dep_xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "choices": [{
    "text": "Why don't skeletons fight each other? They don't have the guts.",
    "index": 0,
    "logprobs": null,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": -1,
    "completion_tokens": -1,
    "total_tokens": -1
  }
}
```

### POST /v1/predict
Generic tensor-based prediction endpoint (for ONNX, CyxModel).

**Request:**
```bash
curl -X POST http://localhost:8080/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "deployment_id": "dep_xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "input": [[1.0, 2.0, 3.0, 4.0]]
  }'
```


### POST /v1/chat/completions
OpenAI-compatible chat completion endpoint for conversational LLMs (GGUF).

**Request:**
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "deployment_id": "dep_xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-xxxxxxxx",
  "object": "chat.completion",
  "created": 1765808157,
  "model": "dep_xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you today?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": -1,
    "completion_tokens": -1,
    "total_tokens": -1
  }
}
```

**Prompt Format:** Uses ChatML internally (`<|im_start|>role\ncontent<|im_end|>`)

### POST /v1/embeddings
Generate text embeddings for semantic search, RAG, or similarity tasks.

**Request:**
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "deployment_id": "dep_xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "input": "Hello world"
  }'
```

**Response:**
```json
{
  "object": "list",
  "data": [{
    "object": "embedding",
    "index": 0,
    "embedding": [0.123, -0.456, 0.789, ...]
  }],
  "model": "dep_xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "usage": {
    "prompt_tokens": -1,
    "total_tokens": -1
  }
}
```

**Multiple inputs:** Pass an array to `input` for batch embedding.

---

## Testing GGUF Deployment

### Step 1: Start the Server Node Daemon
```bash
./build/bin/Release/cyxwiz-server-daemon.exe
```

The daemon starts with:
- HTTP REST API on port 8080
- gRPC Deployment endpoint on port 50056
- gRPC Inference endpoint on port 50057

### Step 2: Deploy a GGUF Model

**Option A: Via CyxWiz Engine GUI**
1. Launch the Engine: `./build/bin/Release/cyxwiz-engine.exe`
2. Open the Deployment Dialog (View → Deploy Model or toolbar)
3. Connect to Server Node at `localhost:50056`
4. Browse and select a `.gguf` file
5. Configure GPU layers, context size, temperature
6. Click "Deploy"

**Option B: Via gRPC (programmatic)**
Use the `DeploymentService.AssignDeployment` RPC with model path and config.

### Step 3: Test Inference via HTTP

```bash
# List deployments to get the deployment_id
curl http://localhost:8080/v1/deployments

# Run text completion
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "deployment_id": "dep_e0db9ec9-3bf9-c2ea-c076-b5550c8f847a",
    "prompt": "What is machine learning?",
    "max_tokens": 200
  }'
```

### Example: Testing with GPT-OSS 20B

```bash
# Download model (example from HuggingFace)
# Model: lmstudio-community/gpt-oss-20b-GGUF

# Deploy via Engine GUI, then test:
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "deployment_id": "YOUR_DEPLOYMENT_ID",
    "prompt": "Tell me a short funny joke:",
    "max_tokens": 100,
    "temperature": 0.8
  }'

# Expected output includes generated text in choices[0].text
```

---

## Implementation Status

| Step | Status | Notes |
|------|--------|-------|
| 1. vcpkg.json dependencies | **Done** | llama-cpp package |
| 2. Root CMakeLists.txt | **Done** | CYXWIZ_ENABLE_GGUF option |
| 3. Server Node CMakeLists.txt | **Done** | Linked llama target |
| 4. GGUFLoader::Load() | **Done** | CPU + GPU layer offloading |
| 5. GGUFLoader::Infer() | **Done** | Text generation + embeddings |
| 6. Deployment Dialog UI | **Done** | Full config UI for GGUF |
| 7. /v1/completions endpoint | **Done** | OpenAI-compatible text API |
| 8. Unit tests | Pending | Create test files |

---

## Technical Notes

### llama.cpp API (as of Dec 2024)
The implementation uses the updated llama.cpp API with vocab-based functions:
- `llama_model_get_vocab()` - Get vocab from model
- `llama_n_vocab(vocab)` - Get vocabulary size
- `llama_model_n_embd(model)` - Get embedding dimension
- `llama_tokenize(vocab, ...)` - Tokenize text
- `llama_token_to_piece(vocab, ...)` - Detokenize token
- `llama_token_is_eog(vocab, ...)` - Check end-of-generation
- `llama_model_load_from_file()` - Load model
- `llama_init_from_model()` - Create context
- `llama_memory_clear(llama_get_memory(ctx), true)` - Clear KV cache

### Sampler Chain Setup
```cpp
llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
    64,              // penalty_last_n
    repeat_penalty,  // penalty_repeat
    0.0f,            // penalty_freq
    0.0f             // penalty_present
));
```

---

# Quick Reference

## Server Node Ports

| Port | Protocol | Service |
|------|----------|---------|
| 8080 | HTTP | REST API (`/health`, `/v1/completions`, `/v1/predict`, `/v1/deployments`) |
| 50052 | gRPC | P2P Job Execution Service |
| 50053 | gRPC | Terminal Handler |
| 50054 | gRPC | IPC Service (GUI communication) |
| 50055 | gRPC | Node Service |
| 50056 | gRPC | Deployment Handler |
| 50057 | gRPC | Inference Service |

## Supported Model Formats Summary

| Format | Loader | Use Case | GPU Support |
|--------|--------|----------|-------------|
| `.cyxmodel` | CyxModelLoader | Native CyxWiz models | ArrayFire (CUDA/OpenCL) |
| `.onnx` | ONNXLoader | Cross-framework models | CUDA EP, CPU fallback |
| `.gguf` | GGUFLoader | LLM text generation | CUDA, Metal, Vulkan, CPU |
| `.safetensors` | - | Export only | - |

---

# Troubleshooting

## GGUF Model Loading Issues

**"Failed to load model"**
- Verify the `.gguf` file path is correct
- Check file permissions
- Ensure sufficient RAM (model size + context buffer)

**Slow inference on CPU**
- Increase GPU layers in deployment config
- Reduce context size for faster processing
- Use smaller quantization (Q4 vs Q8)

**Out of memory**
- Reduce `n_gpu_layers` to offload fewer layers to GPU
- Reduce `n_ctx` context size
- Use more aggressive quantization

## ONNX Model Loading Issues

**"CUDA EP not available"**
- CUDA Toolkit must be installed
- Falls back to CPU automatically
- Check `onnxruntime-gpu` was linked

**"Input shape mismatch"**
- Verify input tensor dimensions match model expectations
- Check batch size (usually first dimension)

## Build Issues

**"llama not found"**
- Run `vcpkg install llama-cpp`
- Ensure `CYXWIZ_ENABLE_GGUF=ON` in CMake

**"onnxruntime not found"**
- Run `vcpkg install onnxruntime-gpu` (Windows x64) or `vcpkg install onnxruntime`
- Ensure `CYXWIZ_ENABLE_ONNX=ON` in CMake

---

# Future: PyTorch Support

For native PyTorch model loading via LibTorch:

**Dependencies:**
- LibTorch (PyTorch C++ frontend)

**Use Cases:**
- Load `.pt` / `.pth` files directly
- TorchScript models
- Research model deployment

---

# Changelog

## December 2024
- Added GGUF/llama.cpp support with full text generation
- Added `/v1/completions` HTTP endpoint for LLM inference
- Updated llama.cpp API to use vocab-based functions
- Added deployment dialog UI for GGUF configuration
- Tested with GPT-OSS 20B (MXFP4 quantization)
- Added /v1/chat/completions endpoint (OpenAI-compatible chat API)
- Added /v1/embeddings endpoint (text embeddings for RAG/search)
