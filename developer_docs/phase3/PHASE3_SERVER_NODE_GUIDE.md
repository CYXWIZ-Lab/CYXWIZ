# Phase 3: Server Node Implementation Guide

**Version:** 1.0
**Date:** November 7, 2025
**Status:** Implementation Ready
**Estimated Effort:** 2-3 days (16-24 hours)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component 1: Deployment Handler](#component-1-deployment-handler)
3. [Component 2: Model Loader Abstraction](#component-2-model-loader-abstraction)
4. [Component 3: Terminal Stream Handler](#component-3-terminal-stream-handler)
5. [Component 4: Metrics Collector](#component-4-metrics-collector)
6. [Integration Points](#integration-points)
7. [Dependencies](#dependencies)
8. [Testing Strategy](#testing-strategy)
9. [Implementation Roadmap](#implementation-roadmap)

---

## Architecture Overview

### Current State

The Server Node (`cyxwiz-server-node`) is a compute worker that:
- Receives deployment jobs from Central Server
- Loads and executes ML models using cyxwiz-backend
- Provides terminal access for debugging
- Reports metrics back to Central Server

**Current Files:**
```
cyxwiz-server-node/
├── src/
│   ├── main.cpp                    (28 lines, minimal skeleton)
│   ├── job_executor.cpp            (2 lines, TODO only)
│   ├── node_server.cpp             (1 line, TODO only)
│   └── metrics_collector.cpp       (2 lines, TODO only)
├── CMakeLists.txt
└── README.md
```

**Dependencies Already Linked:**
- `cyxwiz-backend` - ML compute library
- `cyxwiz-protocol` - gRPC protocol definitions
- `spdlog` - Logging
- `grpc`, `protobuf` - Network communication

---

### Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CyxWiz Server Node                       │
│                                                             │
│  ┌───────────────────────────────────────────────────┐    │
│  │              gRPC Server (Port 50052)             │    │
│  │  - DeploymentHandler (receives jobs)              │    │
│  │  - TerminalHandler (PTY access)                   │    │
│  │  - MetricsReporter (sends stats)                  │    │
│  └─────────────┬─────────────────────────────────────┘    │
│                │                                            │
│  ┌─────────────▼─────────────────────────────────────┐    │
│  │           Deployment Manager                      │    │
│  │  - Job Queue (pending deployments)                │    │
│  │  - Active Deployments Map                         │    │
│  │  - Resource Monitor                               │    │
│  └─────────────┬─────────────────────────────────────┘    │
│                │                                            │
│  ┌─────────────▼─────────────────────────────────────┐    │
│  │          Model Loader (Strategy Pattern)          │    │
│  │  ┌─────────────────────────────────────────────┐ │    │
│  │  │ ONNXLoader  │ GGUFLoader  │ PyTorchLoader  │ │    │
│  │  └─────────────────────────────────────────────┘ │    │
│  │  - Model validation                               │    │
│  │  - Memory allocation                              │    │
│  │  - Inference runtime setup                        │    │
│  └─────────────┬─────────────────────────────────────┘    │
│                │                                            │
│  ┌─────────────▼─────────────────────────────────────┐    │
│  │          Inference Engine                         │    │
│  │  - Request queue                                  │    │
│  │  - Batch processing                               │    │
│  │  - Output formatting                              │    │
│  └───────────────────────────────────────────────────┘    │
│                                                             │
│  ┌───────────────────────────────────────────────────┐    │
│  │       Terminal Service (PTY)                      │    │
│  │  - Shell spawning (bash/sh)                       │    │
│  │  - I/O streaming                                  │    │
│  │  - Session management                             │    │
│  └───────────────────────────────────────────────────┘    │
│                                                             │
│  ┌───────────────────────────────────────────────────┐    │
│  │        Metrics Collector                          │    │
│  │  - CPU/GPU monitoring (1s interval)               │    │
│  │  - Memory tracking                                │    │
│  │  - Network I/O                                    │    │
│  │  - Report to Central Server                       │    │
│  └───────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   Central Server        │
              │   (Receives metrics)    │
              └─────────────────────────┘
```

---

## Component 1: Deployment Handler

### Purpose
Receives deployment jobs from Central Server via gRPC and manages their lifecycle.

### File Structure
```
cyxwiz-server-node/src/
├── deployment_handler.h        (NEW)
├── deployment_handler.cpp      (NEW)
└── deployment_manager.h/.cpp   (NEW)
```

### Code Template: `deployment_handler.h`

```cpp
#pragma once

#include <grpc++/grpc++.h>
#include "deployment.grpc.pb.h"
#include "deployment_manager.h"
#include <memory>

namespace cyxwiz {

class DeploymentHandler {
public:
    DeploymentHandler(std::shared_ptr<DeploymentManager> manager);

    // Start gRPC server
    void StartServer(const std::string& address);

    // Shutdown server gracefully
    void Shutdown();

private:
    std::shared_ptr<DeploymentManager> deployment_manager_;
    std::unique_ptr<grpc::Server> server_;
};

// gRPC Service Implementation
class DeploymentServiceImpl final
    : public cyxwiz::protocol::DeploymentService::Service {
public:
    DeploymentServiceImpl(std::shared_ptr<DeploymentManager> manager);

    // RPC: Receive deployment from Central Server
    grpc::Status AssignDeployment(
        grpc::ServerContext* context,
        const cyxwiz::protocol::AssignDeploymentRequest* request,
        cyxwiz::protocol::AssignDeploymentResponse* response
    ) override;

    // RPC: Report deployment status
    grpc::Status ReportDeploymentStatus(
        grpc::ServerContext* context,
        const cyxwiz::protocol::DeploymentStatusRequest* request,
        cyxwiz::protocol::DeploymentStatusResponse* response
    ) override;

private:
    std::shared_ptr<DeploymentManager> manager_;
};

} // namespace cyxwiz
```

### Code Template: `deployment_manager.h`

```cpp
#pragma once

#include "model_loader.h"
#include <string>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <thread>

namespace cyxwiz {

enum class DeploymentStatus {
    Pending,
    Loading,
    Ready,
    Running,
    Stopped,
    Failed
};

struct DeploymentInfo {
    std::string deployment_id;
    std::string model_id;
    std::string model_path;
    DeploymentStatus status;
    std::string status_message;
    std::unique_ptr<ModelLoader> loader;
    std::thread worker_thread;
};

class DeploymentManager {
public:
    DeploymentManager();
    ~DeploymentManager();

    // Accept new deployment
    bool AcceptDeployment(
        const std::string& deployment_id,
        const std::string& model_path,
        const std::string& format
    );

    // Stop deployment
    void StopDeployment(const std::string& deployment_id);

    // Get deployment status
    DeploymentStatus GetStatus(const std::string& deployment_id);

    // Get all active deployments
    std::vector<std::string> GetActiveDeployments();

private:
    void ExecuteDeployment(const std::string& deployment_id);
    void UpdateStatus(const std::string& deployment_id,
                     DeploymentStatus status,
                     const std::string& message);

    std::unordered_map<std::string, std::unique_ptr<DeploymentInfo>> deployments_;
    std::mutex mutex_;
    bool running_;
};

} // namespace cyxwiz
```

### Implementation Pseudocode

```cpp
// deployment_handler.cpp

bool DeploymentManager::AcceptDeployment(
    const std::string& deployment_id,
    const std::string& model_path,
    const std::string& format
) {
    std::lock_guard<std::mutex> lock(mutex_);

    // 1. Check if deployment already exists
    if (deployments_.find(deployment_id) != deployments_.end()) {
        return false;
    }

    // 2. Create deployment info
    auto info = std::make_unique<DeploymentInfo>();
    info->deployment_id = deployment_id;
    info->model_path = model_path;
    info->status = DeploymentStatus::Pending;

    // 3. Create appropriate model loader
    info->loader = ModelLoaderFactory::Create(format);
    if (!info->loader) {
        return false;
    }

    // 4. Start worker thread
    info->worker_thread = std::thread(
        &DeploymentManager::ExecuteDeployment,
        this,
        deployment_id
    );

    deployments_[deployment_id] = std::move(info);
    return true;
}

void DeploymentManager::ExecuteDeployment(const std::string& deployment_id) {
    // 1. Update status to Loading
    UpdateStatus(deployment_id, DeploymentStatus::Loading, "Loading model...");

    // 2. Load model
    auto& info = deployments_[deployment_id];
    if (!info->loader->Load(info->model_path)) {
        UpdateStatus(deployment_id, DeploymentStatus::Failed, "Model loading failed");
        return;
    }

    // 3. Update status to Ready
    UpdateStatus(deployment_id, DeploymentStatus::Ready, "Model loaded, ready for inference");

    // 4. Wait for inference requests (event loop)
    while (info->status == DeploymentStatus::Ready ||
           info->status == DeploymentStatus::Running) {
        // Process inference queue
        // (Will be implemented with actual inference engine)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}
```

### Integration with Central Server

```cpp
// In main.cpp
auto deployment_manager = std::make_shared<DeploymentManager>();
auto deployment_handler = std::make_unique<DeploymentHandler>(deployment_manager);

// Start gRPC server on port 50052
deployment_handler->StartServer("0.0.0.0:50052");
```

---

## Component 2: Model Loader Abstraction

### Purpose
Unified interface for loading different model formats (ONNX, GGUF, PyTorch, TensorFlow).

### File Structure
```
cyxwiz-server-node/src/
├── model_loader.h          (NEW - Abstract base class)
├── onnx_loader.h/.cpp      (NEW - ONNX Runtime integration)
├── gguf_loader.h/.cpp      (NEW - llama.cpp integration)
├── pytorch_loader.h/.cpp   (NEW - LibTorch integration)
└── tensorflow_loader.h/.cpp (NEW - TensorFlow Lite)
```

### Code Template: `model_loader.h`

```cpp
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cyxwiz/tensor.h>

namespace cyxwiz {

enum class ModelFormat {
    ONNX,
    GGUF,
    PyTorch,
    TensorFlow,
    SafeTensors,
    TFLite,
    TorchScript
};

// Abstract base class for all model loaders
class ModelLoader {
public:
    virtual ~ModelLoader() = default;

    // Load model from file
    virtual bool Load(const std::string& model_path) = 0;

    // Run inference
    virtual bool Infer(
        const std::vector<cyxwiz::Tensor>& inputs,
        std::vector<cyxwiz::Tensor>& outputs
    ) = 0;

    // Get input tensor specs
    virtual std::vector<TensorInfo> GetInputSpecs() const = 0;

    // Get output tensor specs
    virtual std::vector<TensorInfo> GetOutputSpecs() const = 0;

    // Get memory usage (bytes)
    virtual size_t GetMemoryUsage() const = 0;

    // Unload model
    virtual void Unload() = 0;

    // Get model metadata
    virtual std::string GetModelInfo() const = 0;

protected:
    std::string model_path_;
    bool is_loaded_ = false;
};

// Factory for creating loaders
class ModelLoaderFactory {
public:
    static std::unique_ptr<ModelLoader> Create(const std::string& format);
    static std::unique_ptr<ModelLoader> Create(ModelFormat format);
};

} // namespace cyxwiz
```

### Code Template: `onnx_loader.h`

```cpp
#pragma once

#include "model_loader.h"
#include <onnxruntime_cxx_api.h>

namespace cyxwiz {

class ONNXLoader : public ModelLoader {
public:
    ONNXLoader();
    ~ONNXLoader() override;

    bool Load(const std::string& model_path) override;

    bool Infer(
        const std::vector<cyxwiz::Tensor>& inputs,
        std::vector<cyxwiz::Tensor>& outputs
    ) override;

    std::vector<TensorInfo> GetInputSpecs() const override;
    std::vector<TensorInfo> GetOutputSpecs() const override;
    size_t GetMemoryUsage() const override;
    void Unload() override;
    std::string GetModelInfo() const override;

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
};

} // namespace cyxwiz
```

### Implementation Pseudocode: ONNX Loader

```cpp
// onnx_loader.cpp

bool ONNXLoader::Load(const std::string& model_path) {
    try {
        // 1. Create ONNX Runtime environment
        env_ = std::make_unique<Ort::Env>(
            ORT_LOGGING_LEVEL_WARNING,
            "CyxWizServerNode"
        );

        // 2. Configure session options
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(4);

        // Enable GPU if available
        #ifdef USE_CUDA
        OrtCUDAProviderOptions cuda_options;
        session_options_->AppendExecutionProvider_CUDA(cuda_options);
        #endif

        // 3. Load model
        session_ = std::make_unique<Ort::Session>(
            *env_,
            model_path.c_str(),
            *session_options_
        );

        // 4. Extract input/output metadata
        Ort::AllocatorWithDefaultOptions allocator;

        size_t num_inputs = session_->GetInputCount();
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(name.get());

            auto type_info = session_->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_shapes_.push_back(tensor_info.GetShape());
        }

        size_t num_outputs = session_->GetOutputCount();
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(name.get());

            auto type_info = session_->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            output_shapes_.push_back(tensor_info.GetShape());
        }

        model_path_ = model_path;
        is_loaded_ = true;
        return true;

    } catch (const Ort::Exception& e) {
        spdlog::error("ONNX loading failed: {}", e.what());
        return false;
    }
}

bool ONNXLoader::Infer(
    const std::vector<cyxwiz::Tensor>& inputs,
    std::vector<cyxwiz::Tensor>& outputs
) {
    if (!is_loaded_) return false;

    try {
        // 1. Prepare input tensors
        std::vector<Ort::Value> input_tensors;
        // Convert cyxwiz::Tensor to Ort::Value
        // (Implementation depends on tensor format)

        // 2. Prepare output tensor names
        std::vector<const char*> output_names;
        for (const auto& name : output_names_) {
            output_names.push_back(name.c_str());
        }

        // 3. Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names.data(),
            output_names.size()
        );

        // 4. Convert output tensors back to cyxwiz::Tensor
        // (Implementation depends on tensor format)

        return true;

    } catch (const Ort::Exception& e) {
        spdlog::error("ONNX inference failed: {}", e.what());
        return false;
    }
}
```

### Dependencies for Model Loaders

**ONNX Runtime:**
```cmake
# In CMakeLists.txt
find_package(onnxruntime REQUIRED)
target_link_libraries(cyxwiz-server-node PRIVATE onnxruntime::onnxruntime)
```

**llama.cpp (for GGUF):**
```bash
# Clone as submodule
git submodule add https://github.com/ggerganov/llama.cpp external/llama.cpp

# In CMakeLists.txt
add_subdirectory(external/llama.cpp)
target_link_libraries(cyxwiz-server-node PRIVATE llama)
```

**LibTorch (for PyTorch):**
```cmake
find_package(Torch REQUIRED)
target_link_libraries(cyxwiz-server-node PRIVATE "${TORCH_LIBRARIES}")
```

---

## Component 3: Terminal Stream Handler

### Purpose
Provides SSH-like terminal access to deployments for debugging and model interaction.

### File Structure
```
cyxwiz-server-node/src/
├── terminal_handler.h/.cpp     (NEW - gRPC service)
├── pty_session.h/.cpp          (NEW - PTY management)
└── terminal_manager.h/.cpp     (NEW - Session lifecycle)
```

### Code Template: `terminal_handler.h`

```cpp
#pragma once

#include <grpc++/grpc++.h>
#include "deployment.grpc.pb.h"
#include "terminal_manager.h"
#include <memory>

namespace cyxwiz {

class TerminalServiceImpl final
    : public cyxwiz::protocol::TerminalService::Service {
public:
    TerminalServiceImpl(std::shared_ptr<TerminalManager> manager);

    // Bidirectional streaming for terminal I/O
    grpc::Status StreamTerminal(
        grpc::ServerContext* context,
        grpc::ServerReaderWriter<
            cyxwiz::protocol::TerminalData,
            cyxwiz::protocol::TerminalData
        >* stream
    ) override;

    // Resize terminal
    grpc::Status ResizeTerminal(
        grpc::ServerContext* context,
        const cyxwiz::protocol::TerminalResize* request,
        cyxwiz::protocol::SimpleResponse* response
    ) override;

private:
    std::shared_ptr<TerminalManager> manager_;
};

} // namespace cyxwiz
```

### Code Template: `pty_session.h`

```cpp
#pragma once

#include <string>
#include <functional>
#include <thread>

namespace cyxwiz {

// Callback for receiving PTY output
using PtyOutputCallback = std::function<void(const std::string& data)>;

class PtySession {
public:
    PtySession();
    ~PtySession();

    // Start PTY with shell
    bool Start(const std::string& shell = "/bin/bash");

    // Write data to PTY
    void Write(const std::string& data);

    // Resize PTY
    void Resize(int rows, int cols);

    // Set output callback
    void SetOutputCallback(PtyOutputCallback callback);

    // Close PTY
    void Close();

    // Check if PTY is running
    bool IsRunning() const;

private:
    void ReadLoop();

    int master_fd_;
    int slave_fd_;
    pid_t child_pid_;
    std::thread read_thread_;
    PtyOutputCallback output_callback_;
    bool running_;
};

} // namespace cyxwiz
```

### Implementation Pseudocode: PTY Session

```cpp
// pty_session.cpp (Unix/Linux)

#include <pty.h>
#include <unistd.h>
#include <sys/wait.h>

bool PtySession::Start(const std::string& shell) {
    // 1. Create pseudo-terminal
    if (openpty(&master_fd_, &slave_fd_, nullptr, nullptr, nullptr) == -1) {
        return false;
    }

    // 2. Fork process
    child_pid_ = fork();
    if (child_pid_ == -1) {
        close(master_fd_);
        close(slave_fd_);
        return false;
    }

    if (child_pid_ == 0) {
        // Child process
        close(master_fd_);

        // Set controlling terminal
        setsid();
        ioctl(slave_fd_, TIOCSCTTY, 0);

        // Redirect stdin/stdout/stderr to slave PTY
        dup2(slave_fd_, STDIN_FILENO);
        dup2(slave_fd_, STDOUT_FILENO);
        dup2(slave_fd_, STDERR_FILENO);
        close(slave_fd_);

        // Execute shell
        execl(shell.c_str(), shell.c_str(), nullptr);
        _exit(1); // If execl fails
    }

    // Parent process
    close(slave_fd_);
    running_ = true;

    // Start read thread
    read_thread_ = std::thread(&PtySession::ReadLoop, this);

    return true;
}

void PtySession::ReadLoop() {
    char buffer[4096];
    while (running_) {
        ssize_t n = read(master_fd_, buffer, sizeof(buffer));
        if (n > 0 && output_callback_) {
            output_callback_(std::string(buffer, n));
        } else if (n <= 0) {
            running_ = false;
            break;
        }
    }
}

void PtySession::Write(const std::string& data) {
    if (running_) {
        write(master_fd_, data.c_str(), data.size());
    }
}

void PtySession::Resize(int rows, int cols) {
    struct winsize ws;
    ws.ws_row = rows;
    ws.ws_col = cols;
    ioctl(master_fd_, TIOCSWINSZ, &ws);
}
```

### Windows Implementation Note

For Windows, use `ConPTY` (Windows Pseudo Console) API:

```cpp
// pty_session_windows.cpp

#include <windows.h>

bool PtySession::Start(const std::string& shell) {
    // 1. Create pseudo console
    COORD size = {80, 24};
    HANDLE hPipeIn, hPipeOut;

    CreatePipe(&hPipeIn, &master_write_, nullptr, 0);
    CreatePipe(&master_read_, &hPipeOut, nullptr, 0);

    HRESULT hr = CreatePseudoConsole(
        size,
        hPipeIn,
        hPipeOut,
        0,
        &hpc_
    );

    // 2. Create process with pseudo console
    STARTUPINFOEX siEx = {};
    siEx.StartupInfo.cb = sizeof(STARTUPINFOEX);

    // Attach pseudo console to process
    // (Full implementation in Windows SDK docs)

    return SUCCEEDED(hr);
}
```

---

## Component 4: Metrics Collector

### Purpose
Collect real-time CPU/GPU/memory/network metrics and report to Central Server.

### File Structure
```
cyxwiz-server-node/src/
├── metrics_collector.h/.cpp    (EXTEND existing file)
├── system_monitor.h/.cpp       (NEW - OS-level metrics)
└── gpu_monitor.h/.cpp          (NEW - GPU metrics)
```

### Code Template: `metrics_collector.h`

```cpp
#pragma once

#include <grpc++/grpc++.h>
#include "node.grpc.pb.h"
#include <memory>
#include <thread>
#include <atomic>
#include <chrono>

namespace cyxwiz {

struct SystemMetrics {
    double cpu_usage_percent;
    double memory_usage_percent;
    uint64_t memory_used_bytes;
    uint64_t memory_total_bytes;
    uint64_t network_rx_bytes;
    uint64_t network_tx_bytes;

    // GPU metrics (if available)
    double gpu_usage_percent;
    double gpu_memory_usage_percent;
    uint64_t gpu_memory_used_bytes;
    uint64_t gpu_memory_total_bytes;
    double gpu_temperature_celsius;
    double gpu_power_watts;
};

class MetricsCollector {
public:
    MetricsCollector(const std::string& central_server_address);
    ~MetricsCollector();

    // Start metrics collection
    void Start();

    // Stop metrics collection
    void Stop();

    // Get current metrics
    SystemMetrics GetCurrentMetrics();

private:
    void CollectionLoop();
    void ReportToCentralServer(const SystemMetrics& metrics);

    SystemMetrics CollectSystemMetrics();
    SystemMetrics CollectGPUMetrics();

    std::string central_server_address_;
    std::unique_ptr<cyxwiz::protocol::NodeService::Stub> stub_;

    std::thread collection_thread_;
    std::atomic<bool> running_;
    std::chrono::seconds collection_interval_{1};
};

} // namespace cyxwiz
```

### Implementation Pseudocode: System Metrics (Linux)

```cpp
// system_monitor.cpp

#include <fstream>
#include <sstream>

SystemMetrics CollectSystemMetrics() {
    SystemMetrics metrics = {};

    // 1. CPU Usage (from /proc/stat)
    static uint64_t prev_idle = 0, prev_total = 0;

    std::ifstream stat_file("/proc/stat");
    std::string line;
    std::getline(stat_file, line);

    std::istringstream ss(line);
    std::string cpu;
    uint64_t user, nice, system, idle, iowait, irq, softirq, steal;
    ss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;

    uint64_t total = user + nice + system + idle + iowait + irq + softirq + steal;
    uint64_t diff_idle = idle - prev_idle;
    uint64_t diff_total = total - prev_total;

    metrics.cpu_usage_percent =
        100.0 * (diff_total - diff_idle) / diff_total;

    prev_idle = idle;
    prev_total = total;

    // 2. Memory Usage (from /proc/meminfo)
    std::ifstream meminfo("/proc/meminfo");
    uint64_t mem_total = 0, mem_available = 0;

    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal:") == 0) {
            sscanf(line.c_str(), "MemTotal: %lu kB", &mem_total);
        } else if (line.find("MemAvailable:") == 0) {
            sscanf(line.c_str(), "MemAvailable: %lu kB", &mem_available);
        }
    }

    metrics.memory_total_bytes = mem_total * 1024;
    metrics.memory_used_bytes = (mem_total - mem_available) * 1024;
    metrics.memory_usage_percent =
        100.0 * metrics.memory_used_bytes / metrics.memory_total_bytes;

    // 3. Network I/O (from /proc/net/dev)
    std::ifstream netdev("/proc/net/dev");
    uint64_t rx_bytes = 0, tx_bytes = 0;

    while (std::getline(netdev, line)) {
        if (line.find(':') != std::string::npos) {
            uint64_t rx, tx;
            // Parse interface stats
            // Sum all interfaces
            rx_bytes += rx;
            tx_bytes += tx;
        }
    }

    metrics.network_rx_bytes = rx_bytes;
    metrics.network_tx_bytes = tx_bytes;

    return metrics;
}
```

### Implementation Pseudocode: GPU Metrics (NVIDIA)

```cpp
// gpu_monitor.cpp (Using NVML)

#include <nvml.h>

SystemMetrics CollectGPUMetrics() {
    SystemMetrics metrics = {};

    // 1. Initialize NVML
    if (nvmlInit() != NVML_SUCCESS) {
        return metrics;
    }

    // 2. Get device handle (assuming first GPU)
    nvmlDevice_t device;
    if (nvmlDeviceGetHandleByIndex(0, &device) != NVML_SUCCESS) {
        nvmlShutdown();
        return metrics;
    }

    // 3. GPU utilization
    nvmlUtilization_t utilization;
    if (nvmlDeviceGetUtilizationRates(device, &utilization) == NVML_SUCCESS) {
        metrics.gpu_usage_percent = utilization.gpu;
        metrics.gpu_memory_usage_percent = utilization.memory;
    }

    // 4. GPU memory
    nvmlMemory_t memory;
    if (nvmlDeviceGetMemoryInfo(device, &memory) == NVML_SUCCESS) {
        metrics.gpu_memory_total_bytes = memory.total;
        metrics.gpu_memory_used_bytes = memory.used;
    }

    // 5. GPU temperature
    unsigned int temp;
    if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp)
        == NVML_SUCCESS) {
        metrics.gpu_temperature_celsius = temp;
    }

    // 6. GPU power
    unsigned int power;
    if (nvmlDeviceGetPowerUsage(device, &power) == NVML_SUCCESS) {
        metrics.gpu_power_watts = power / 1000.0;
    }

    nvmlShutdown();
    return metrics;
}
```

---

## Integration Points

### 1. gRPC Server Registration

```cpp
// In main.cpp

// Build gRPC server
grpc::ServerBuilder builder;

// Add deployment service
auto deployment_service = std::make_unique<DeploymentServiceImpl>(
    deployment_manager
);
builder.AddListeningPort("0.0.0.0:50052", grpc::InsecureServerCredentials());
builder.RegisterService(deployment_service.get());

// Add terminal service
auto terminal_service = std::make_unique<TerminalServiceImpl>(
    terminal_manager
);
builder.RegisterService(terminal_service.get());

// Start server
std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
spdlog::info("Server Node listening on 0.0.0.0:50052");

// Start metrics collector
MetricsCollector metrics("localhost:50051"); // Central Server
metrics.Start();

// Wait for shutdown signal
server->Wait();

metrics.Stop();
```

### 2. Central Server Communication

**Node Registration:**
```cpp
// On startup, register with Central Server
auto stub = NodeService::NewStub(
    grpc::CreateChannel(
        "localhost:50051",
        grpc::InsecureChannelCredentials()
    )
);

RegisterNodeRequest request;
request.mutable_info()->set_node_id(node_id);
request.mutable_info()->set_name("server-node-1");
// ... set hardware info

RegisterNodeResponse response;
grpc::ClientContext context;
stub->RegisterNode(&context, request, &response);
```

**Heartbeat:**
```cpp
// Every 30 seconds, send heartbeat
HeartbeatRequest request;
request.set_node_id(node_id);
// ... set current status

HeartbeatResponse response;
grpc::ClientContext context;
stub->Heartbeat(&context, request, &response);
```

---

## Dependencies

### Required Libraries

**System Libraries:**
- gRPC and Protocol Buffers (already linked)
- NVML (NVIDIA Management Library) for GPU monitoring
- PTY library (Linux: `libutil`, Windows: ConPTY API)

**Model Runtime Libraries:**

1. **ONNX Runtime** (vcpkg):
```bash
./vcpkg install onnxruntime
```

2. **llama.cpp** (Git submodule):
```bash
git submodule add https://github.com/ggerganov/llama.cpp external/llama.cpp
```

3. **LibTorch** (PyTorch C++):
```bash
# Download from pytorch.org
# Link in CMakeLists.txt
```

4. **TensorFlow Lite** (vcpkg):
```bash
./vcpkg install tensorflow-lite
```

### CMakeLists.txt Updates

```cmake
# Find required packages
find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

# Optional: Model runtimes
find_package(onnxruntime)
find_package(Torch)
find_package(tensorflow-lite)

# Server Node executable
add_executable(cyxwiz-server-node
    src/main.cpp
    src/deployment_handler.cpp
    src/deployment_manager.cpp
    src/model_loader.cpp
    src/onnx_loader.cpp
    src/gguf_loader.cpp
    src/pytorch_loader.cpp
    src/terminal_handler.cpp
    src/pty_session.cpp
    src/terminal_manager.cpp
    src/metrics_collector.cpp
    src/system_monitor.cpp
    src/gpu_monitor.cpp
)

target_link_libraries(cyxwiz-server-node PRIVATE
    cyxwiz-backend
    cyxwiz-protocol
    gRPC::grpc++
    protobuf::libprotobuf
    spdlog::spdlog
)

# Link model runtimes if available
if(onnxruntime_FOUND)
    target_link_libraries(cyxwiz-server-node PRIVATE onnxruntime::onnxruntime)
    target_compile_definitions(cyxwiz-server-node PRIVATE HAS_ONNX=1)
endif()

if(Torch_FOUND)
    target_link_libraries(cyxwiz-server-node PRIVATE "${TORCH_LIBRARIES}")
    target_compile_definitions(cyxwiz-server-node PRIVATE HAS_PYTORCH=1)
endif()

# Platform-specific libraries
if(UNIX)
    target_link_libraries(cyxwiz-server-node PRIVATE util pthread)
elseif(WIN32)
    target_link_libraries(cyxwiz-server-node PRIVATE ws2_32)
endif()

# NVML for GPU monitoring
find_library(NVML_LIBRARY nvidia-ml)
if(NVML_LIBRARY)
    target_link_libraries(cyxwiz-server-node PRIVATE ${NVML_LIBRARY})
    target_compile_definitions(cyxwiz-server-node PRIVATE HAS_NVML=1)
endif()
```

---

## Testing Strategy

### Unit Tests

**Test 1: Model Loader**
```cpp
TEST(ModelLoader, LoadONNXModel) {
    auto loader = ModelLoaderFactory::Create("onnx");
    ASSERT_TRUE(loader->Load("test_model.onnx"));
    ASSERT_TRUE(loader->is_loaded());

    auto input_specs = loader->GetInputSpecs();
    ASSERT_GT(input_specs.size(), 0);
}

TEST(ModelLoader, InvalidModel) {
    auto loader = ModelLoaderFactory::Create("onnx");
    ASSERT_FALSE(loader->Load("invalid.onnx"));
}
```

**Test 2: Deployment Manager**
```cpp
TEST(DeploymentManager, AcceptDeployment) {
    DeploymentManager manager;
    ASSERT_TRUE(manager.AcceptDeployment(
        "deploy-123",
        "model.onnx",
        "onnx"
    ));

    ASSERT_EQ(manager.GetStatus("deploy-123"),
              DeploymentStatus::Pending);
}
```

**Test 3: Metrics Collector**
```cpp
TEST(MetricsCollector, CollectMetrics) {
    SystemMetrics metrics = CollectSystemMetrics();
    ASSERT_GE(metrics.cpu_usage_percent, 0.0);
    ASSERT_LE(metrics.cpu_usage_percent, 100.0);
    ASSERT_GT(metrics.memory_total_bytes, 0);
}
```

### Integration Tests

**Test 4: End-to-End Deployment**
```cpp
// 1. Start Server Node
// 2. Send deployment via gRPC
// 3. Verify model loads
// 4. Send inference request
// 5. Verify output
```

**Test 5: Terminal Session**
```cpp
// 1. Create terminal session
// 2. Send command via gRPC
// 3. Verify output received
// 4. Close session
```

---

## Implementation Roadmap

### Day 1: Core Infrastructure (6-8 hours)

**Morning (4 hours):**
1. Set up CMakeLists.txt with all dependencies (1 hour)
2. Implement `DeploymentHandler` and `DeploymentManager` (2 hours)
3. Implement `ModelLoader` base class and factory (1 hour)

**Afternoon (4 hours):**
4. Implement `ONNXLoader` (3 hours)
5. Write unit tests for deployment and ONNX (1 hour)

**Deliverables:**
- Deployment handler accepting jobs
- ONNX models can be loaded
- Basic test coverage

---

### Day 2: Additional Loaders & Terminal (6-8 hours)

**Morning (4 hours):**
1. Implement `GGUFLoader` (llama.cpp integration) (3 hours)
2. Implement `PyTorchLoader` skeleton (1 hour)

**Afternoon (4 hours):**
3. Implement `TerminalHandler` and `PtySession` (3 hours)
4. Test terminal streaming (1 hour)

**Deliverables:**
- GGUF models can be loaded (LLMs)
- Terminal access working
- Can execute shell commands remotely

---

### Day 3: Metrics & Integration (4-6 hours)

**Morning (3 hours):**
1. Implement `MetricsCollector` (2 hours)
2. Implement GPU monitoring with NVML (1 hour)

**Afternoon (3 hours):**
3. Integrate all components in `main.cpp` (1 hour)
4. End-to-end testing (2 hours)

**Deliverables:**
- Real-time metrics reporting
- Full integration tested
- Ready for Phase 4 (Engine GUI)

---

## Code Style Guidelines

**Naming Conventions:**
- Classes: `PascalCase` (e.g., `DeploymentHandler`)
- Functions: `PascalCase` (e.g., `LoadModel()`)
- Variables: `snake_case_` with trailing underscore for members
- Constants: `UPPER_SNAKE_CASE`

**Error Handling:**
- Use `spdlog` for logging
- Return `bool` for success/failure
- Use exceptions only for truly exceptional cases

**Threading:**
- Use `std::thread` for background tasks
- Protect shared data with `std::mutex`
- Use `std::atomic` for simple flags

**Memory Management:**
- Use `std::unique_ptr` for ownership
- Use `std::shared_ptr` for shared ownership
- Avoid raw pointers where possible

---

## Troubleshooting

### Common Issues

**Issue 1: ONNX Runtime Not Found**
```bash
# Solution: Install via vcpkg
./vcpkg/vcpkg install onnxruntime
```

**Issue 2: llama.cpp Build Fails**
```bash
# Solution: Update submodule
git submodule update --init --recursive
cd external/llama.cpp
mkdir build && cd build
cmake .. && make
```

**Issue 3: PTY Not Working on Windows**
```cpp
// Solution: Use ConPTY API (Windows 10+)
// See: https://docs.microsoft.com/en-us/windows/console/creating-a-pseudoconsole-session
```

**Issue 4: GPU Metrics Show 0%**
```bash
# Solution: Install NVIDIA drivers and NVML
# Ubuntu:
sudo apt-get install nvidia-cuda-toolkit

# Verify:
nvidia-smi
```

---

## Next Steps After Phase 3

Once Server Node is complete, proceed to **Phase 4: Engine GUI**:

1. **Deployment Panel** - GUI for creating deployments
2. **Terminal Panel** - Integrated terminal in Engine
3. **Wallet Connector** - Payment integration
4. **Asset Browser Extensions** - Model marketplace UI

See `PHASE4_ENGINE_GUI_GUIDE.md` (to be created).

---

## References

- ONNX Runtime C++ API: https://onnxruntime.ai/docs/api/c/
- llama.cpp: https://github.com/ggerganov/llama.cpp
- LibTorch: https://pytorch.org/cppdocs/
- gRPC C++ Guide: https://grpc.io/docs/languages/cpp/
- NVML API: https://developer.nvidia.com/nvidia-management-library-nvml

---

**Document Version:** 1.0
**Last Updated:** November 7, 2025
**Status:** Ready for Implementation
**Estimated Completion:** 2-3 days with focused development
