# Model Deployment Guide

A comprehensive guide to deploying trained models from CyxWiz Engine to Server Nodes for production inference.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Supported Model Formats](#supported-model-formats)
4. [Prerequisites](#prerequisites)
5. [Workflow Summary](#workflow-summary)
6. [Case Study: MNIST MLP Deployment](#case-study-mnist-mlp-deployment)
   - [Step 1: Build and Train in Engine](#step-1-build-and-train-in-engine)
   - [Step 2: Export Trained Model](#step-2-export-trained-model)
   - [Step 3: Start Server Node Daemon](#step-3-start-server-node-daemon)
   - [Step 4: Deploy Model to Server Node](#step-4-deploy-model-to-server-node)
   - [Step 5: Run Inference](#step-5-run-inference)
7. [Testing Deployed Models](#testing-deployed-models)
   - [Testing Embedded Deployment](#testing-embedded-deployment-in-engine-server)
   - [Testing with Python Scripts](#testing-with-python-scripts)
   - [Testing with PowerShell](#testing-with-powershell-windows)
   - [Common Test Scenarios](#common-test-scenarios)
   - [Debugging Failed Deployments](#debugging-failed-deployments)
8. [API Reference](#api-reference)
   - [HTTP REST API](#http-rest-api)
   - [gRPC InferenceService](#grpc-inferenceservice)
9. [Model Format Specification](#model-format-specification)
10. [Troubleshooting](#troubleshooting)

---

## Overview

CyxWiz provides a complete pipeline from model development to production deployment:

1. **Design** - Build neural network architectures using the visual node editor
2. **Train** - Train locally with real-time visualization
3. **Export** - Save trained model in `.cyxmodel` format
4. **Deploy** - Send model to Server Node for inference serving
5. **Serve** - Access predictions via HTTP REST API or gRPC

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CyxWiz Engine                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ Node Editor │  │  Training   │  │   Export    │  │   Deploy    │    │
│  │  (Design)   │→ │  (Local)    │→ │ (.cyxmodel) │→ │   (gRPC)    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────┬──────┘    │
└────────────────────────────────────────────────────────────┼───────────┘
                                                              │
                                                         gRPC │ Deploy
                                                              │
┌─────────────────────────────────────────────────────────────▼───────────┐
│                        Server Node Daemon                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────┐  │
│  │ Deployment  │  │   Model     │  │       Inference Endpoints       │  │
│  │  Manager    │→ │   Loader    │→ │  HTTP :8080  │  gRPC :50057    │  │
│  └─────────────┘  └─────────────┘  └───────┬──────┴────────┬────────┘  │
└────────────────────────────────────────────┼───────────────┼────────────┘
                                              │               │
                                    ┌─────────▼───────────────▼─────────┐
                                    │         Client Applications        │
                                    │  curl, Python, Node.js, grpcurl   │
                                    └───────────────────────────────────┘
```

### Component Roles

| Component | Role | Port |
|-----------|------|------|
| **CyxWiz Engine** | Model design, training, export | N/A (desktop app) |
| **Server Node Daemon** | Model hosting, inference serving | Multiple (see below) |
| **DeploymentManager** | Model lifecycle management | Internal |
| **HTTP REST API** | RESTful inference endpoint | 8080 |
| **gRPC InferenceService** | High-performance RPC inference | 50057 |

### Server Node Ports

| Port | Service | Protocol |
|------|---------|----------|
| 50052 | P2P JobExecutionService | gRPC |
| 50053 | Terminal Handler | gRPC |
| 50054 | IPC Daemon Service (GUI) | gRPC |
| 50055 | Node Service | gRPC |
| 50056 | Deployment Handler | gRPC |
| 50057 | **InferenceService** | gRPC |
| 8080 | **HTTP REST API** | HTTP |

---

## Supported Model Formats

The Server Node can load and serve models in various formats. Here's the current implementation status:

### Format Support Matrix

| Format | Extension | Status | Description |
|--------|-----------|--------|-------------|
| **CyxModel** | `.cyxmodel` | **Fully Working** | Native format from CyxWiz Engine |
| **ONNX** | `.onnx` | **Fully Working** | Open Neural Network Exchange (CUDA + CPU fallback) |
| **GGUF** | `.gguf` | **Fully Working** | GGML format for LLMs (llama.cpp integration) |
| **PyTorch** | `.pt`, `.pth` | Planned | TorchScript models (requires LibTorch) |
| **SafeTensors** | `.safetensors` | Planned | HuggingFace safe serialization |

### CyxModel Format (Fully Implemented)

The native `.cyxmodel` format supports:

- **Architecture reconstruction** from saved metadata
- **Weight loading** for all supported layer types
- **Inference execution** via SequentialModel

**Supported Layers:**
| Layer Type | Inference Support |
|------------|-------------------|
| Dense/Linear | Yes |
| ReLU | Yes |
| Sigmoid | Yes |
| Tanh | Yes |
| Softmax | Yes |
| LeakyReLU | Yes |
| ELU | Yes |
| GELU | Yes |
| Swish/SiLU | Yes |
| Mish | Yes |
| Dropout | Yes (disabled at inference) |
| Flatten | Yes |
| BatchNorm | Partial |
| Conv2D | Not yet |
| LSTM/GRU | Not yet |

### ONNX Format (Fully Implemented)

The ONNX (Open Neural Network Exchange) format enables importing models from PyTorch, TensorFlow, and other frameworks.

**Features:**
- CUDA execution provider with automatic CPU fallback
- Automatic input name mapping (handles common names like "Input3", "input", etc.)
- Tensor shape and type specification from model metadata

**Deploy ONNX models via Engine:**
1. **Deploy > Deploy Model** (or Ctrl+Shift+D)
2. Select **Server Node** deployment type
3. Browse to your `.onnx` file
4. Click **Deploy**

**Test ONNX inference:**
```bash
# Get model input/output specs
curl http://localhost:8080/v1/deployments/<deployment_id>

# Run prediction (adjust shape to match model)
curl -X POST http://localhost:8080/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"deployment_id": "<deployment_id>", "input": [[...]]}'
```

### GGUF Format (Fully Implemented)

The GGUF format (llama.cpp) enables deploying Large Language Models for text generation, chat, and embeddings.

**Features:**
- Configurable GPU layers offloading
- Adjustable context size
- Text generation with sampling parameters (temperature, top_p, top_k)
- Text embedding generation for semantic search
- OpenAI-compatible API endpoints

**Deploy GGUF models via Engine:**
1. **Deploy > Deploy Model** (or Ctrl+Shift+D)
2. Select **Server Node** deployment type
3. Browse to your `.gguf` file
4. Configure GGUF settings:
   - **GPU Layers**: Number of layers to offload to GPU (0 = CPU only)
   - **Context Size**: Maximum context window (default: 2048)
5. Click **Deploy**

**Test GGUF text generation:**
```bash
# Text completion
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "deployment_id": "<deployment_id>",
    "prompt": "Hello, how are you?",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

### Other Formats (Planned)

The following formats are planned for future releases:

#### PyTorch/TorchScript (Planned)
```
Dependency: LibTorch (PyTorch C++ API)
Use Case: Models exported via torch.jit.script or torch.jit.trace
Status: Class exists, needs LibTorch integration
```

#### SafeTensors (Planned)
```
Dependency: Custom parser or safetensors library
Use Case: HuggingFace model weights (safer than pickle)
Status: Not yet started
```

### Format Detection

The Server Node automatically detects model format by:
1. File extension (`.cyxmodel`, `.onnx`, `.gguf`, `.pt`)
2. Explicit `format` parameter in deployment request
3. File magic bytes (fallback)

```cpp
// Factory creates appropriate loader based on format
auto loader = ModelLoaderFactory::Create("cyxmodel");  // or "onnx", "gguf", etc.
```

### Recommended Workflow by Use Case

| Use Case | Recommended Format | Notes |
|----------|-------------------|-------|
| Models trained in CyxWiz Engine | `.cyxmodel` | Native, fully supported |
| Pre-trained vision models | `.onnx` | Fully supported with CUDA |
| Large Language Models | `.gguf` | Fully supported with llama.cpp |
| PyTorch research models | `.pt` | When LibTorch support is added |
| HuggingFace transformers | `.safetensors` | When SafeTensors support is added |

### Contributing New Format Support

To add support for a new format:

1. **Create loader class** in `cyxwiz-server-node/src/model_loader.cpp`
2. **Implement the interface:**
   ```cpp
   class NewFormatLoader : public ModelLoader {
       bool Load(const std::string& path) override;
       bool Infer(const TensorMap& inputs, TensorMap& outputs) override;
       std::vector<TensorSpec> GetInputSpecs() const override;
       std::vector<TensorSpec> GetOutputSpecs() const override;
       // ...
   };
   ```
3. **Register in factory:**
   ```cpp
   if (fmt == "newformat") {
       return std::make_unique<NewFormatLoader>();
   }
   ```
4. **Add dependencies** to `CMakeLists.txt`

---

## Prerequisites

### Software

```bash
# Build the Engine and Server Node
cmake --preset windows-release
cmake --build build --config Release -j 8

# Verify executables exist
ls build/bin/Release/cyxwiz-engine.exe
ls build/bin/Release/cyxwiz-server-daemon.exe
```

### Required Tools for Testing

```bash
# HTTP testing
curl --version

# gRPC testing (optional)
# Install grpcurl: https://github.com/fullstorydev/grpcurl
grpcurl --version
```

---

## Workflow Summary

```
┌──────────────────┐
│  1. Design Model │  (Engine: Node Editor)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  2. Train Model  │  (Engine: Tools > Start Training)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  3. Export Model │  (Engine: File > Export Model > .cyxmodel)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 4. Start Daemon  │  (Terminal: cyxwiz-server-daemon.exe)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  5. Deploy Model │  (gRPC: DeploymentService.CreateDeployment)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  6. Run Inference│  (HTTP: POST /v1/predict or gRPC: Infer)
└──────────────────┘
```

---

## Case Study: MNIST MLP Deployment

This walkthrough deploys a trained MNIST digit classifier from the Engine to a Server Node.

### Step 1: Build and Train in Engine

#### 1.1 Launch Engine

```bash
./build/bin/Release/cyxwiz-engine.exe
```

#### 1.2 Load MNIST Architecture

**Option A: Use Pre-built Pattern**
1. **File > New from Pattern**
2. Select **"MNIST MLP Classifier"** (or "MLP Basic")
3. Click **Apply**

**Option B: Build Manually**

Create this architecture in the Node Editor:

```
Input [784]
    │
    ▼
Dense (512) → ReLU → Dropout (0.2)
    │
    ▼
Dense (256) → ReLU → Dropout (0.2)
    │
    ▼
Dense (10) → Softmax
    │
    ▼
Output [10]
```

1. Right-click > **Input/Output > Input** (shape: `[784]`)
2. Right-click > **Layers > Dense** (units: `512`)
3. Right-click > **Activations > ReLU**
4. Right-click > **Layers > Dropout** (rate: `0.2`)
5. Right-click > **Layers > Dense** (units: `256`)
6. Right-click > **Activations > ReLU**
7. Right-click > **Layers > Dropout** (rate: `0.2`)
8. Right-click > **Layers > Dense** (units: `10`)
9. Right-click > **Activations > Softmax**
10. Right-click > **Input/Output > Output**
11. Connect all nodes in sequence

#### 1.3 Configure Training

| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |

#### 1.4 Train the Model

1. **Tools > Validate Graph** (ensure green checkmark)
2. **Tools > Compile Graph** (Ctrl+B)
3. **Tools > Start Training** (F5)
4. Wait for training to complete (~5-10 minutes)

**Expected Results:**
```
[Epoch 10/10] Loss: 0.0456, Acc: 98.67%
Training complete!
Test Accuracy: ~97-98%
```

### Step 2: Export Trained Model

#### 2.1 Save Graph (Optional)

**File > Save As** → `mnist_mlp.cyxgraph`

#### 2.2 Export Model with Weights

1. **File > Export Model**
2. Choose format: **CyxModel (.cyxmodel)**
3. Save to: `models/mnist_mlp.cyxmodel`
4. Options:
   - [x] Include trained weights
   - [x] Include graph definition
   - [x] Include training history
   - [ ] Include optimizer state (optional)

#### 2.3 Verify Export

The `.cyxmodel` directory structure:

```
mnist_mlp.cyxmodel/
├── manifest.json          # Model metadata
├── graph.cyxgraph         # Node graph definition
├── config.json            # Training configuration
├── history.json           # Training history (loss/accuracy curves)
└── weights/
    ├── manifest.json      # Tensor metadata
    ├── layer0_weight.bin  # Dense(784→512) weights
    ├── layer0_bias.bin    # Dense(784→512) bias
    ├── layer1_weight.bin  # Dense(512→256) weights
    ├── layer1_bias.bin    # Dense(512→256) bias
    ├── layer2_weight.bin  # Dense(256→10) weights
    └── layer2_bias.bin    # Dense(256→10) bias
```

**Sample `manifest.json`:**
```json
{
  "version": "1.0",
  "format": "cyxmodel",
  "created": "2025-12-13T15:30:00Z",
  "cyxwiz_version": "0.2.0",
  "model": {
    "name": "MNIST MLP Classifier",
    "type": "Sequential",
    "num_parameters": 533770,
    "num_layers": 8
  },
  "training": {
    "epochs_trained": 10,
    "final_accuracy": 0.9867,
    "final_loss": 0.0456
  },
  "content": {
    "has_optimizer_state": false,
    "has_training_history": true,
    "has_graph": true
  }
}
```

### Step 3: Start Server Node Daemon

#### 3.1 Launch Daemon

```bash
# Basic launch
./build/bin/Release/cyxwiz-server-daemon.exe

# With custom ports
./build/bin/Release/cyxwiz-server-daemon.exe \
  --http-port=8080 \
  --inference-addr=0.0.0.0:50057

# With TLS (production)
./build/bin/Release/cyxwiz-server-daemon.exe \
  --tls-auto \
  --http-port=8080
```

#### 3.2 Verify Startup

Expected console output:
```
[INFO] CyxWiz Server Daemon v0.2.0
[INFO] ========================================
[INFO] Node ID: node_1702489200
[INFO] IPC service: localhost:50054
[INFO] Central server: localhost:50051 (not connected)
[INFO] ========================================
[INFO] Server Daemon is ready!
[INFO]   IPC service (GUI):    localhost:50054
[INFO]   P2P service (Engine): 0.0.0.0:50052
[INFO]   Node service:         0.0.0.0:50055
[INFO]   Deployment endpoint:  0.0.0.0:50056
[INFO]   Terminal endpoint:    0.0.0.0:50053
[INFO]   HTTP REST API:        http://0.0.0.0:8080
[INFO]   Inference gRPC:       0.0.0.0:50057
[INFO]   TLS encryption:       DISABLED
[INFO] ========================================
[INFO] Press Ctrl+C to shutdown
```

#### 3.3 Test Health Endpoint

```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "server_type": "cyxwiz-server-node",
  "active_deployments": 0
}
```

### Step 4: Deploy Model to Server Node

#### 4.1 Copy Model to Server

```bash
# Create models directory if needed
mkdir -p ./models

# Copy exported model
cp -r /path/to/mnist_mlp.cyxmodel ./models/
```

#### 4.2 Deploy via gRPC

**Using grpcurl:**
```bash
grpcurl -plaintext -d '{
  "config": {
    "deployment_id": "mnist-v1",
    "type": 1,
    "model": {
      "model_id": "mnist_mlp",
      "name": "MNIST MLP",
      "format": 0,
      "local_path": "./models/mnist_mlp.cyxmodel"
    }
  }
}' localhost:50056 cyxwiz.protocol.DeploymentService/CreateDeployment
```

**Programmatic (C++ Client):**
```cpp
#include <grpcpp/grpcpp.h>
#include "deployment.grpc.pb.h"

auto channel = grpc::CreateChannel("localhost:50056",
    grpc::InsecureChannelCredentials());
auto stub = cyxwiz::protocol::DeploymentService::NewStub(channel);

cyxwiz::protocol::CreateDeploymentRequest request;
auto* config = request.mutable_config();
config->set_deployment_id("mnist-v1");
config->set_type(cyxwiz::protocol::DEPLOYMENT_TYPE_LOCAL);

auto* model = config->mutable_model();
model->set_model_id("mnist_mlp");
model->set_name("MNIST MLP");
model->set_local_path("./models/mnist_mlp.cyxmodel");

cyxwiz::protocol::CreateDeploymentResponse response;
grpc::ClientContext context;
auto status = stub->CreateDeployment(&context, request, &response);

if (status.ok() && response.status() == cyxwiz::protocol::STATUS_SUCCESS) {
    std::cout << "Deployment ID: " << response.deployment().deployment_id() << std::endl;
}
```

#### 4.3 Verify Deployment

```bash
# Check deployment status
curl http://localhost:8080/v1/deployments/mnist-v1

# List all deployments
curl http://localhost:8080/v1/models
```

Expected deployment response:
```json
{
  "id": "mnist-v1",
  "status": 5,
  "input_specs": [
    {"name": "input", "shape": [1, 784], "dtype": "float32"}
  ],
  "output_specs": [
    {"name": "output", "shape": [1, 10], "dtype": "float32"}
  ],
  "metrics": {
    "request_count": 0,
    "avg_latency_ms": 0.0
  }
}
```

### Step 5: Run Inference

#### 5.1 Prepare Input Data

MNIST images are 28x28 grayscale pixels, flattened to 784 floats (normalized 0-1):

```python
# Python example: Load MNIST test image
import numpy as np
from PIL import Image

# Load a test image (28x28 grayscale)
img = Image.open("digit_5.png").convert("L")
img = np.array(img).flatten() / 255.0  # Normalize to [0, 1]

# Convert to list for JSON
input_data = img.tolist()  # 784 float values
```

#### 5.2 HTTP REST API Inference

**Request:**
```bash
curl -X POST http://localhost:8080/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "deployment_id": "mnist-v1",
    "input": [[0.0, 0.0, ..., 0.5, 0.9, ..., 0.0]]
  }'
```

**Response:**
```json
{
  "deployment_id": "mnist-v1",
  "output": [0.01, 0.02, 0.01, 0.03, 0.02, 0.85, 0.02, 0.01, 0.02, 0.01],
  "shape": [1, 10],
  "latency_ms": 2.34
}
```

**Interpretation:** Highest probability (0.85) at index 5 → Predicted digit: **5**

#### 5.3 Python Client Example

```python
import requests
import numpy as np

# Prepare input (example: digit "5")
input_data = np.random.rand(784).tolist()  # Replace with actual image

# Send inference request
response = requests.post(
    "http://localhost:8080/v1/predict",
    json={
        "deployment_id": "mnist-v1",
        "input": [input_data]
    }
)

result = response.json()
output = np.array(result["output"])
predicted_digit = np.argmax(output)
confidence = output[predicted_digit]

print(f"Predicted digit: {predicted_digit}")
print(f"Confidence: {confidence:.2%}")
print(f"Latency: {result['latency_ms']:.2f}ms")
```

#### 5.4 gRPC Inference

**Using grpcurl:**
```bash
# Note: Binary tensor data needs base64 encoding
grpcurl -plaintext -d '{
  "deployment_id": "mnist-v1",
  "inputs": [{
    "name": "input",
    "shape": [1, 784],
    "dtype": 1,
    "data": "<base64-encoded-float32-array>"
  }]
}' localhost:50057 cyxwiz.protocol.InferenceService/Infer
```

**C++ Client:**
```cpp
#include <grpcpp/grpcpp.h>
#include "inference.grpc.pb.h"

// Connect
auto channel = grpc::CreateChannel("localhost:50057",
    grpc::InsecureChannelCredentials());
auto stub = cyxwiz::protocol::InferenceService::NewStub(channel);

// Prepare request
cyxwiz::protocol::InferRequest request;
request.set_deployment_id("mnist-v1");

auto* input = request.add_inputs();
input->set_name("input");
input->add_shape(1);
input->add_shape(784);
input->set_dtype(cyxwiz::protocol::DATA_TYPE_FLOAT32);

// Set input data (784 floats)
std::vector<float> input_data(784, 0.0f);
// ... fill with actual image data ...
input->set_data(input_data.data(), input_data.size() * sizeof(float));

// Run inference
cyxwiz::protocol::InferResponse response;
grpc::ClientContext context;
auto status = stub->Infer(&context, request, &response);

if (status.ok() && response.status() == cyxwiz::protocol::STATUS_SUCCESS) {
    // Process output
    const auto& output = response.outputs(0);
    const float* probs = reinterpret_cast<const float*>(output.data().data());

    int predicted = std::max_element(probs, probs + 10) - probs;
    std::cout << "Predicted digit: " << predicted << std::endl;
    std::cout << "Latency: " << response.latency_ms() << "ms" << std::endl;
}
```

#### 5.5 Batch Inference

For higher throughput, send multiple images in one request:

```bash
curl -X POST http://localhost:8080/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "deployment_id": "mnist-v1",
    "input": [
      [0.0, 0.0, ..., 0.5, 0.9, ..., 0.0],
      [0.0, 0.1, ..., 0.8, 0.2, ..., 0.0],
      [0.0, 0.0, ..., 0.3, 0.7, ..., 0.0]
    ]
  }'
```

Response will have shape `[3, 10]` for 3 predictions.

---

## Testing Deployed Models

This section covers how to test deployed models using various tools and approaches.

### Testing Embedded Deployment (In-Engine Server)

The embedded deployment runs an HTTP server directly within the CyxWiz Engine, useful for quick testing without starting the Server Node daemon.

#### Start Embedded Deployment

1. **Deploy > Deploy Model** (or Ctrl+Shift+D)
2. Select **Embedded** deployment type
3. Browse and select your `.cyxmodel` folder
4. Set port (default: 8084)
5. Click **Deploy**

#### Test Endpoints with curl

**Health Check:**
```bash
curl http://localhost:8084/health
```

Expected response:
```json
{
  "status": "healthy",
  "server_type": "cyxwiz-engine-embedded",
  "model_loaded": true,
  "model_name": "my.cyxmodel",
  "request_count": 0
}
```

**Model Info:**
```bash
curl http://localhost:8084/v1/model
```

Expected response:
```json
{
  "model_name": "my.cyxmodel",
  "model_path": "C:\\path\\to\\my.cyxmodel",
  "num_layers": 7,
  "layers": [
    {"index": 0, "name": "Linear(784 -> 512)", "has_parameters": true},
    {"index": 1, "name": "ReLU", "has_parameters": false},
    {"index": 2, "name": "Dropout(p=0.200000)", "has_parameters": false},
    {"index": 3, "name": "Linear(512 -> 256)", "has_parameters": true},
    {"index": 4, "name": "ReLU", "has_parameters": false},
    {"index": 5, "name": "Dropout(p=0.200000)", "has_parameters": false},
    {"index": 6, "name": "Linear(256 -> 10)", "has_parameters": true}
  ]
}
```

**Run Prediction:**
```bash
# Test with zeros (784 input features for MNIST)
curl -X POST http://localhost:8084/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [0.0, 0.0, ..., 0.0]}'  # 784 values
```

Expected response:
```json
{
  "output": [-0.43, -1.16, -0.67, -0.39, -0.65, 0.86, 0.01, -0.96, 0.84, -0.37],
  "shape": [1, 10],
  "latency_ms": 2.34
}
```

### Testing with Python Scripts

#### Quick Test Script

```python
#!/usr/bin/env python3
"""test_deployment.py - Test a deployed CyxWiz model"""

import requests
import numpy as np
import argparse

def test_health(base_url):
    """Test health endpoint"""
    r = requests.get(f"{base_url}/health")
    print(f"Health: {r.json()}")
    return r.json()["status"] == "healthy"

def test_model_info(base_url):
    """Test model info endpoint"""
    r = requests.get(f"{base_url}/v1/model")
    info = r.json()
    print(f"Model: {info['model_name']}")
    print(f"Layers: {info['num_layers']}")
    for layer in info.get("layers", []):
        print(f"  [{layer['index']}] {layer['name']}")
    return info["num_layers"] > 0

def test_predict(base_url, input_size=784):
    """Test prediction with random input"""
    # Generate random normalized input
    input_data = np.random.rand(input_size).tolist()

    r = requests.post(
        f"{base_url}/v1/predict",
        json={"input": input_data}
    )
    result = r.json()

    output = np.array(result["output"])
    predicted = np.argmax(output)

    print(f"Prediction: class {predicted}")
    print(f"Confidence: {output[predicted]:.4f}")
    print(f"Latency: {result['latency_ms']:.2f}ms")
    return True

def test_mnist_image(base_url, image_path):
    """Test with actual MNIST image"""
    from PIL import Image

    # Load and preprocess image
    img = Image.open(image_path).convert("L").resize((28, 28))
    pixels = np.array(img).flatten() / 255.0

    r = requests.post(
        f"{base_url}/v1/predict",
        json={"input": pixels.tolist()}
    )
    result = r.json()

    output = np.array(result["output"])
    predicted = np.argmax(output)

    print(f"Image: {image_path}")
    print(f"Predicted digit: {predicted}")
    print(f"Probabilities: {output}")
    return predicted

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8084")
    parser.add_argument("--image", help="Path to test image")
    args = parser.parse_args()

    print(f"Testing deployment at {args.url}\n")

    # Run tests
    assert test_health(args.url), "Health check failed"
    print()
    assert test_model_info(args.url), "Model info failed"
    print()
    test_predict(args.url)

    if args.image:
        print()
        test_mnist_image(args.url, args.image)

    print("\nAll tests passed!")
```

**Usage:**
```bash
# Basic test
python test_deployment.py --url http://localhost:8084

# Test with specific image
python test_deployment.py --url http://localhost:8084 --image digit_5.png
```

#### Batch Testing Script

```python
#!/usr/bin/env python3
"""batch_test.py - Benchmark inference throughput"""

import requests
import numpy as np
import time

def benchmark(base_url, num_requests=100, input_size=784, batch_size=1):
    """Run benchmark test"""

    # Warm up
    for _ in range(5):
        requests.post(
            f"{base_url}/v1/predict",
            json={"input": [0.0] * input_size}
        )

    # Benchmark
    latencies = []
    start = time.time()

    for i in range(num_requests):
        input_data = np.random.rand(batch_size, input_size).tolist()

        r = requests.post(
            f"{base_url}/v1/predict",
            json={"input": input_data if batch_size > 1 else input_data[0]}
        )

        result = r.json()
        latencies.append(result["latency_ms"])

    elapsed = time.time() - start

    print(f"Benchmark Results:")
    print(f"  Total requests: {num_requests}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Throughput: {num_requests/elapsed:.1f} req/s")
    print(f"  Avg latency: {np.mean(latencies):.2f}ms")
    print(f"  P50 latency: {np.percentile(latencies, 50):.2f}ms")
    print(f"  P95 latency: {np.percentile(latencies, 95):.2f}ms")
    print(f"  P99 latency: {np.percentile(latencies, 99):.2f}ms")

if __name__ == "__main__":
    benchmark("http://localhost:8084", num_requests=100)
```

### Testing with PowerShell (Windows)

```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:8084/health"

# Model info
Invoke-RestMethod -Uri "http://localhost:8084/v1/model"

# Predict with zeros
$input = @{ input = @(0.0) * 784 }
$body = $input | ConvertTo-Json -Compress
Invoke-RestMethod -Uri "http://localhost:8084/v1/predict" -Method Post -Body $body -ContentType "application/json"
```

### Comparing Deployment Types

| Feature | Embedded | Local Node | Network |
|---------|----------|------------|---------|
| Setup | One click | Start daemon | Configure cluster |
| Port | Custom (8084) | 8080 | Variable |
| Performance | Good | Better | Best (distributed) |
| Use Case | Quick testing | Production single-node | Production multi-node |
| Endpoints | /health, /v1/model, /v1/predict | Full API | Full API + gRPC |

### Verifying Model Architecture

After deployment, always verify the model loaded correctly:

```bash
# Check layer count matches expected
curl -s http://localhost:8084/v1/model | jq '.num_layers'

# Verify layer names
curl -s http://localhost:8084/v1/model | jq '.layers[].name'
```

**Expected for MNIST MLP (7 layers):**
```
Linear(784 -> 512)
ReLU
Dropout(p=0.200000)
Linear(512 -> 256)
ReLU
Dropout(p=0.200000)
Linear(256 -> 10)
```

### Common Test Scenarios

#### 1. Smoke Test
```bash
# Just verify server responds
curl -s http://localhost:8084/health | jq -e '.status == "healthy"'
```

#### 2. Model Validation
```bash
# Verify model loaded with correct architecture
layers=$(curl -s http://localhost:8084/v1/model | jq '.num_layers')
[ "$layers" -gt 0 ] && echo "Model loaded with $layers layers"
```

#### 3. Inference Test
```bash
# Test with known input, verify output shape
curl -s -X POST http://localhost:8084/v1/predict \
  -H "Content-Type: application/json" \
  -d "{\"input\": $(python -c 'print([0.0]*784)')}" \
  | jq '.shape'
# Expected: [1, 10]
```

#### 4. Load Test
```bash
# Simple load test with Apache Bench
ab -n 1000 -c 10 -p input.json -T "application/json" \
   http://localhost:8084/v1/predict
```

Create `input.json`:
```json
{"input": [0.0, 0.0, 0.0, ... ]}
```

### Debugging Failed Deployments

If the model fails to load:

1. **Check Engine Console** for error messages
2. **Verify .cyxmodel structure:**
   ```bash
   ls -la my.cyxmodel/
   # Should show: manifest.json, graph.cyxgraph, weights/
   ```
3. **Validate manifest.json:**
   ```bash
   cat my.cyxmodel/manifest.json | jq .
   ```
4. **Check graph.cyxgraph has valid nodes:**
   ```bash
   cat my.cyxmodel/graph.cyxgraph | jq '.nodes | length'
   ```

---

## API Reference

### HTTP REST API

Base URL: `http://localhost:8080`

#### Health Check

```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "server_type": "cyxwiz-server-node",
  "active_deployments": 1
}
```

#### List Models

```
GET /v1/models
```

Response:
```json
{
  "object": "list",
  "data": [
    {"id": "mnist-v1", "type": "model", "status": "running"}
  ]
}
```

#### Run Prediction

```
POST /v1/predict
Content-Type: application/json

{
  "deployment_id": "string",
  "input": [[float, ...], ...]
}
```

Response:
```json
{
  "deployment_id": "string",
  "output": [float, ...],
  "shape": [int, ...],
  "latency_ms": float
}
```

#### Get Deployment Info

```
GET /v1/deployments/:id
```

Response:
```json
{
  "id": "string",
  "status": int,
  "input_specs": [{"name": "string", "shape": [int], "dtype": "string"}],
  "output_specs": [{"name": "string", "shape": [int], "dtype": "string"}],
  "metrics": {"request_count": int, "avg_latency_ms": float}
}
```

### OpenAI-Compatible API

The Server Node provides OpenAI-compatible endpoints for easy integration with existing tools and libraries.

#### Text Completions

```
POST /v1/completions
Content-Type: application/json

{
  "deployment_id": "string",     // or "model": "string"
  "prompt": "string",
  "max_tokens": 256,             // optional, default: 256
  "temperature": 0.7,            // optional, default: 0.7
  "top_p": 0.9,                  // optional, default: 0.9
  "stop": ["\n"]                // optional, stop sequences
}
```

Response:
```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1702489200,
  "model": "dep_xxx",
  "choices": [{
    "text": "Generated text here...",
    "index": 0,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 50,
    "total_tokens": 55
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "deployment_id": "my-llm",
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 0.8
  }'
```

#### Chat Completions

```
POST /v1/chat/completions
Content-Type: application/json

{
  "deployment_id": "string",     // or "model": "string"
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 256,             // optional
  "temperature": 0.7             // optional
}
```

Response:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1702489200,
  "model": "dep_xxx",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you today?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 10,
    "total_tokens": 30
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "deployment_id": "my-llm",
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Write a Python function to calculate factorial."}
    ],
    "max_tokens": 200
  }'
```

**Prompt Format:**
The chat endpoint uses ChatML format internally:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
```

#### Embeddings

```
POST /v1/embeddings
Content-Type: application/json

{
  "deployment_id": "string",     // or "model": "string"
  "input": "Text to embed"       // string or array of strings
}
```

Response:
```json
{
  "object": "list",
  "data": [{
    "object": "embedding",
    "index": 0,
    "embedding": [0.123, -0.456, 0.789, ...]
  }],
  "model": "dep_xxx",
  "usage": {
    "prompt_tokens": 4,
    "total_tokens": 4
  }
}
```

**Example:**
```bash
# Single text
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "deployment_id": "my-embedding-model",
    "input": "Hello world"
  }'

# Multiple texts
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "deployment_id": "my-embedding-model",
    "input": ["Hello world", "How are you?"]
  }'
```

#### Python Client Example (OpenAI SDK Compatible)

```python
import openai

# Point to CyxWiz Server Node
openai.api_base = "http://localhost:8080/v1"
openai.api_key = "not-needed"  # No auth required currently

# Chat completion
response = openai.ChatCompletion.create(
    model="my-deployment-id",
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=100
)
print(response.choices[0].message.content)

# Text completion
response = openai.Completion.create(
    model="my-deployment-id",
    prompt="Once upon a time",
    max_tokens=100
)
print(response.choices[0].text)

# Embeddings
response = openai.Embedding.create(
    model="my-embedding-model",
    input="Hello world"
)
print(response.data[0].embedding[:5])  # First 5 dimensions
```

### gRPC InferenceService

Service: `cyxwiz.protocol.InferenceService`
Address: `localhost:50057`

#### Infer

```protobuf
rpc Infer(InferRequest) returns (InferResponse);

message InferRequest {
  string deployment_id = 1;
  repeated TensorData inputs = 2;
}

message TensorData {
  string name = 1;
  repeated int64 shape = 2;
  DataType dtype = 3;
  bytes data = 4;  // Raw float32 bytes
}

message InferResponse {
  StatusCode status = 1;
  repeated TensorData outputs = 2;
  double latency_ms = 3;
  Error error = 4;
}
```

#### GetModelInfo

```protobuf
rpc GetModelInfo(GetModelInfoRequest) returns (GetModelInfoResponse);

message GetModelInfoRequest {
  string deployment_id = 1;
}

message GetModelInfoResponse {
  StatusCode status = 1;
  repeated TensorInfo input_specs = 2;
  repeated TensorInfo output_specs = 3;
  Error error = 4;
}
```

---

## Model Format Specification

### .cyxmodel Directory Structure

```
model_name.cyxmodel/
├── manifest.json       # Model metadata (required)
├── graph.cyxgraph      # Node graph JSON (optional)
├── config.json         # Training configuration
├── history.json        # Training history
└── weights/
    ├── manifest.json   # Tensor metadata
    └── *.bin           # Binary tensor files
```

### Binary Tensor Format

Each `.bin` file contains:
```
[4 bytes]  uint32_t ndims        // Number of dimensions
[8 bytes]  int64_t  shape[ndims] // Shape array
[4 bytes]  uint32_t dtype        // Data type (1=float32)
[N bytes]  float[]  data         // Raw tensor data (row-major)
```

### Supported Layer Types for Deployment

| Layer Type | Supported | Notes |
|------------|-----------|-------|
| Dense | Yes | Fully connected layer |
| ReLU | Yes | Activation |
| Sigmoid | Yes | Activation |
| Tanh | Yes | Activation |
| Softmax | Yes | Activation |
| LeakyReLU | Yes | Activation (alpha param) |
| ELU | Yes | Activation (alpha param) |
| GELU | Yes | Activation |
| Swish | Yes | Activation |
| Mish | Yes | Activation |
| Dropout | No | Disabled at inference |
| BatchNorm | Partial | Fixed running stats |
| Conv2D | Not Yet | Planned |
| LSTM/GRU | Not Yet | Planned |

---

## Troubleshooting

### "Deployment not found"

```json
{"error": {"message": "Deployment not found: mnist-v1", "code": "deployment_not_found"}}
```

**Solution:**
1. Verify deployment was created successfully
2. Check deployment ID spelling
3. Ensure model path is correct and accessible

### "Model loading failed"

**Solution:**
1. Verify `.cyxmodel` directory exists and has correct structure
2. Check `manifest.json` is valid JSON
3. Ensure all weight files are present in `weights/` directory

### "Input shape mismatch"

```json
{"error": {"message": "Expected input shape [1, 784], got [1, 728]"}}
```

**Solution:**
1. Flatten input image to correct size (28x28 = 784 for MNIST)
2. Verify batch dimension is included

### "Server connection refused"

```
curl: (7) Failed to connect to localhost port 8080
```

**Solution:**
1. Verify daemon is running: `ps aux | grep cyxwiz-server-daemon`
2. Check port is not in use: `netstat -an | grep 8080`
3. Try with explicit address: `curl http://127.0.0.1:8080/health`

### "gRPC deadline exceeded"

**Solution:**
1. Increase client timeout
2. Check model loading hasn't timed out
3. Verify deployment status is "running" (status=5)

### High Latency

**Possible causes:**
1. Model running on CPU instead of GPU
2. Large batch sizes
3. Network overhead

**Solutions:**
1. Enable CUDA backend: Build with `CYXWIZ_ENABLE_CUDA=ON`
2. Reduce batch size for real-time inference
3. Use gRPC instead of HTTP for lower latency

---

## Next Steps

After successful deployment:

1. **Monitor metrics** - Track request count, latency, errors
2. **Scale horizontally** - Deploy to multiple Server Nodes
3. **Add TLS** - Use `--tls-auto` or provide certificates
4. **Connect to Central Server** - For distributed workload management
5. **Build client SDKs** - Python, JavaScript, mobile

---

*Document Version: 1.0*
*Last Updated: 2025-12-15*
*CyxWiz Version: 0.2.x*

