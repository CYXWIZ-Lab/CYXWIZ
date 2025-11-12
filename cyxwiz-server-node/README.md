# CyxWiz Server Node (Miner)

**Distributed compute worker for the CyxWiz ML training network**

## Overview

The **CyxWiz Server Node** (also called **Miner**) is a C++20 application that provides GPU/CPU compute resources to the CyxWiz decentralized ML training network. It receives model deployment assignments from the Central Server, executes training jobs, and earns CYXWIZ tokens for completed work.

### Key Features

✓ **Multi-Format Model Support** - ONNX, GGUF (LLMs), PyTorch
✓ **Remote Terminal Access** - SSH-like pseudo-terminal streaming
✓ **Hardware Auto-Detection** - CPU cores, RAM, GPU capabilities
✓ **Automatic Registration** - Connects to Central Server on startup
✓ **Real-Time Metrics** - CPU, GPU, memory, network monitoring
✓ **Cross-Platform** - Windows, Linux, macOS

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────┐
│            CyxWiz Server Node (Miner)               │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐     ┌──────────────┐            │
│  │ Node Client  │────▶│Central Server│            │
│  │  (Register)  │     │  (gRPC)      │            │
│  └──────────────┘     └──────────────┘            │
│         │                                          │
│         ▼                                          │
│  ┌──────────────────────────────────┐             │
│  │   Deployment Handler (Port 50052)│             │
│  │  - CreateDeployment               │             │
│  │  - StopDeployment                │             │
│  │  - GetDeploymentStatus            │             │
│  │  - GetDeploymentMetrics           │             │
│  └──────────────────────────────────┘             │
│         │                                          │
│         ▼                                          │
│  ┌──────────────────────────────────┐             │
│  │   Deployment Manager              │             │
│  │  - Thread-safe lifecycle          │             │
│  │  - Model loading                  │             │
│  │  - Inference execution            │             │
│  │  - Metrics tracking               │             │
│  └──────────────────────────────────┘             │
│         │                                          │
│         ▼                                          │
│  ┌──────────────────────────────────┐             │
│  │   Model Loaders (Factory)         │             │
│  │  ├─ ONNXLoader                   │             │
│  │  ├─ GGUFLoader (llama.cpp)       │             │
│  │  └─ PyTorchLoader (LibTorch)     │             │
│  └──────────────────────────────────┘             │
│                                                     │
│  ┌──────────────────────────────────┐             │
│  │   Terminal Handler (Port 50053)   │             │
│  │  - CreateSession                  │             │
│  │  - StreamTerminal (Bidirectional) │             │
│  │  - ResizeTerminal                 │             │
│  │  - CloseSession                   │             │
│  └──────────────────────────────────┘             │
│         │                                          │
│         ▼                                          │
│  ┌──────────────────────────────────┐             │
│  │   PTY Sessions (Unix/Windows)     │             │
│  │  - openpty/fork (Unix)            │             │
│  │  - ConPTY (Windows)               │             │
│  │  - Real-time I/O streaming        │             │
│  └──────────────────────────────────┘             │
│                                                     │
│  ┌──────────────────────────────────┐             │
│  │   Hardware Detector               │             │
│  │  - CPU cores, RAM                 │             │
│  │  - GPU detection (CUDA/OpenCL)    │             │
│  │  - Network interface              │             │
│  └──────────────────────────────────┘             │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Communication Flow

```
Central Server                Server Node
      │                            │
      │  1. RegisterNode()         │
      │◄───────────────────────────┤
      │                            │
      │  2. RegisterNodeResponse   │
      ├───────────────────────────▶│
      │     (node_id, session)     │
      │                            │
      │  3. Heartbeat() [10s]      │
      │◄───────────────────────────┤
      │◄───────────────────────────┤
      │◄───────────────────────────┤
      │                            │
      │  4. CreateDeployment()     │
      ├───────────────────────────▶│
      │                            │
      │         ┌──────────────────┤
      │         │ Load Model       │
      │         │ Execute Job      │
      │         │ Track Metrics    │
      │         └──────────────────┤
      │                            │
      │  5. ReportProgress()       │
      │◄───────────────────────────┤
      │                            │
      │  6. ReportCompletion()     │
      │◄───────────────────────────┤
      │     (results, signature)   │
      │                            │
      │  7. Payment TX             │
      ├───────────────────────────▶│
      │                            │
```

---

## File Structure

```
cyxwiz-server-node/
├── src/
│   ├── main.cpp                    # Entry point, service initialization
│   ├── node_client.h/cpp           # Central Server registration & heartbeat
│   ├── deployment_handler.h/cpp    # gRPC deployment service (port 50052)
│   ├── deployment_manager.h/cpp    # Deployment lifecycle management
│   ├── model_loader.h/cpp          # Model format abstraction
│   ├── terminal_handler.h/cpp      # gRPC terminal service (port 50053)
│   ├── metrics_collector.h/cpp     # System metrics tracking
│   ├── node_server.cpp             # Legacy (to be refactored)
│   └── job_executor.cpp            # Legacy (to be refactored)
│
├── CMakeLists.txt                  # Build configuration
└── README.md                       # This file
```

---

## Building from Source

### Prerequisites

**Required:**
- **C++20 Compiler** - GCC 10+, Clang 12+, MSVC 2019+
- **CMake 3.20+**
- **vcpkg** - Dependency management
- **gRPC & Protobuf** - Installed via vcpkg
- **spdlog** - Logging library

**Optional (for GPU support):**
- **ArrayFire 3.8+** - GPU acceleration
- **CUDA Toolkit** - NVIDIA GPU support
- **OpenCL SDK** - AMD/Intel GPU support

### Build Commands

**Linux/macOS:**
```bash
cd cyxwiz-server-node
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/cyxwiz-server-node
```

**Windows:**
```powershell
cd cyxwiz-server-node
cmake -B build -S . -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
.\build\Release\cyxwiz-server-node.exe
```

**Full Project Build (from root):**
```bash
cd cyxwiz-root
cmake --preset windows-release  # or linux-release, macos-release
cmake --build build/windows-release --target cyxwiz-server-node
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CYXWIZ_CENTRAL_SERVER` | `localhost:50051` | Central Server address |
| `CYXWIZ_NODE_ID` | Auto-generated | Unique node identifier |
| `CYXWIZ_DEPLOYMENT_PORT` | `50052` | Deployment service port |
| `CYXWIZ_TERMINAL_PORT` | `50053` | Terminal service port |
| `CYXWIZ_WALLET_ADDRESS` | `` | Solana wallet for payments |

### Command Line Arguments

```bash
cyxwiz-server-node --help

Options:
  --central-server <address>    Central Server address (default: localhost:50051)
  --node-id <id>                Node identifier (default: auto-generated)
  --deployment-port <port>      Deployment service port (default: 50052)
  --terminal-port <port>        Terminal service port (default: 50053)
  --wallet <address>            Solana wallet address
  --log-level <level>           Logging level: debug, info, warn, error (default: info)
  --standalone                  Run without Central Server registration
```

**Note:** Command-line parsing is TODO - currently uses hardcoded defaults.

---

## Hardware Requirements

### Minimum

- **CPU:** 4 cores
- **RAM:** 8 GB
- **Storage:** 50 GB free
- **Network:** 10 Mbps upload/download

### Recommended

- **CPU:** 16+ cores (AMD Ryzen/Intel Xeon)
- **RAM:** 32 GB+
- **GPU:** NVIDIA RTX 3090/4090 (24GB VRAM) or AMD equivalent
- **Storage:** 500 GB+ NVMe SSD
- **Network:** 100 Mbps+ with stable connection

### Supported GPUs

**NVIDIA (CUDA):**
- Compute Capability 6.1+ (GTX 1080 Ti+, Tesla P40+, RTX series)
- Tested: RTX 3090, RTX 4090, A100, H100

**AMD (OpenCL):**
- Radeon RX 6000/7000 series
- Tested: RX 6900 XT, RX 7900 XTX

**Apple (Metal):**
- M1/M2/M3 chips with 16GB+ unified memory

---

## Deployment Workflow

### 1. Startup & Registration

```cpp
// Pseudo-code for main.cpp
Initialize cyxwiz::backend
Create DeploymentManager
Start DeploymentHandler (port 50052)
Start TerminalHandler (port 50053)

// Register with Central Server
NodeClient client("localhost:50051", node_id);
if (client.Register()) {
    client.StartHeartbeat(10);  // 10 second interval
}

// Main loop
while (!shutdown) {
    sleep(1);
}
```

### 2. Accepting Deployments

**Central Server calls `CreateDeployment`:**
```protobuf
message CreateDeploymentRequest {
  DeploymentConfig config = 1;
}

message DeploymentConfig {
  ModelInfo model = 1;
  DeploymentType type = 2;  // TRAINING, INFERENCE, FINE_TUNING
  map<string, string> hyperparameters = 3;
}
```

**Server Node responds:**
```protobuf
message CreateDeploymentResponse {
  Deployment deployment = 1;  // Contains deployment_id, status
  StatusCode status = 2;
  Error error = 3;
}
```

### 3. Model Loading

```cpp
// Deployment Manager loads model
ModelLoader* loader = ModelLoaderFactory::Create(format);
if (!loader->Load(model_path)) {
    status = DEPLOYMENT_STATUS_FAILED;
    return;
}
```

### 4. Execution

```cpp
// Main inference loop
while (!should_stop) {
    // Wait for inference requests
    // Process batch
    // Update metrics (request_count, latency)
    // Report progress to Central Server
}
```

### 5. Completion & Payment

```cpp
// Report completion
ReportCompletionRequest req;
req.set_job_id(job_id);
req.set_result(job_result);
req.set_signature(crypto_signature);

ReportCompletionResponse resp;
if (resp.payment_released) {
    log("Payment TX: {}", resp.payment_tx_hash);
}
```

---

## Model Format Support

### ONNX (ONNXLoader)

**Status:** Stub implementation
**Runtime:** ONNX Runtime
**Use Cases:** Image classification, object detection, NLP models

**TODO:**
- Link against ONNX Runtime library
- Implement `Load()`, `Infer()`, `GetInputSpecs()`, `GetOutputSpecs()`

### GGUF (GGUFLoader)

**Status:** Stub implementation
**Runtime:** llama.cpp
**Use Cases:** Large Language Models (LLaMA, Mistral, Qwen, etc.)

**TODO:**
- Integrate llama.cpp as submodule or library
- Implement context management
- Add quantization support (Q4, Q8)

### PyTorch (PyTorchLoader)

**Status:** Stub implementation
**Runtime:** LibTorch (C++ API)
**Use Cases:** Custom PyTorch models, research

**TODO:**
- Link against LibTorch
- Implement TorchScript loading
- Add autograd support

---

## Terminal Access

### Unix/Linux (PTY - IMPLEMENTED)

**Implementation:**
```cpp
// Using openpty() and fork()
int master_fd, slave_fd;
openpty(&master_fd, &slave_fd, nullptr, nullptr, &winsize);

pid_t child = fork();
if (child == 0) {
    // Child process
    setsid();
    ioctl(slave_fd, TIOCSCTTY, 0);
    dup2(slave_fd, STDIN/STDOUT/STDERR);
    execl("/bin/bash", "bash", nullptr);
}
```

**Features:**
- Full PTY emulation with `/bin/bash`
- ANSI escape code support
- Terminal resizing via `TIOCSWINSZ`
- Non-blocking I/O

### Windows (ConPTY - STUB)

**Implementation:** TODO
**API:** Windows Pseudo Console (ConPTY) - Windows 10 1809+

**Required:**
```cpp
#include <windows.h>
// TODO: Include conpty.h when implementing

HPCON hPC;
CreatePseudoConsole(size, hPipeIn, hPipeOut, 0, &hPC);
```

---

## Metrics & Monitoring

### Collected Metrics

**System Metrics:**
- CPU usage (per-core and aggregate)
- RAM usage (total, available, cached)
- GPU usage (VRAM, utilization, temperature)
- Network throughput (TX/RX bytes)

**Deployment Metrics:**
- Request count
- Average latency (ms)
- Throughput (requests/sec)
- Error rate

**Node Metrics:**
- Uptime percentage
- Jobs completed/failed
- Total compute hours
- Reputation score

### Reporting

**Heartbeat (every 10 seconds):**
```protobuf
message HeartbeatRequest {
  string node_id = 1;
  NodeInfo current_status = 2;  // Updated hardware info
  repeated string active_jobs = 3;
}
```

**Progress Reports:**
```protobuf
message ReportProgressRequest {
  string node_id = 1;
  string job_id = 2;
  JobStatus status = 3;
  map<string, double> current_metrics = 4;  // loss, accuracy, etc.
}
```

---

## Development Guide

### Adding a New Model Format

1. **Create Loader Class:**
```cpp
// In model_loader.h
class MyFormatLoader : public ModelLoader {
public:
    bool Load(const std::string& path) override;
    bool Infer(const InputMap& inputs, OutputMap& outputs) override;
    // ... implement other virtual methods
};
```

2. **Register in Factory:**
```cpp
// In model_loader.cpp
std::unique_ptr<ModelLoader> ModelLoaderFactory::Create(const std::string& format) {
    if (format == "myformat") {
        return std::make_unique<MyFormatLoader>();
    }
    // ...
}
```

3. **Add to Supported Formats:**
```cpp
// In node_client.cpp - HardwareDetector::DetectHardwareInfo()
info.add_supported_formats("MyFormat");
info.add_available_runtimes("myformat-runtime");
```

### Adding Hardware Detection

**GPU Detection Example:**
```cpp
// In node_client.cpp
std::vector<protocol::DeviceCapabilities> HardwareDetector::DetectDevices() {
    #ifdef CYXWIZ_HAS_MY_GPU_SDK
    int gpu_count = MyGPUSDK::GetDeviceCount();
    for (int i = 0; i < gpu_count; ++i) {
        auto gpu_info = MyGPUSDK::GetDeviceInfo(i);
        protocol::DeviceCapabilities gpu;
        gpu.set_device_type(protocol::DEVICE_MY_GPU);
        gpu.set_device_name(gpu_info.name);
        gpu.set_memory_total(gpu_info.memory);
        devices.push_back(gpu);
    }
    #endif
}
```

### Debugging

**Enable Debug Logging:**
```bash
export SPDLOG_LEVEL=debug
./cyxwiz-server-node
```

**gRPC Debug:**
```bash
export GRPC_VERBOSITY=debug
export GRPC_TRACE=all
```

**Valgrind (Memory Leaks):**
```bash
valgrind --leak-check=full ./cyxwiz-server-node
```

---

## API Reference

### NodeClient

```cpp
class NodeClient {
public:
    NodeClient(const std::string& central_server, const std::string& node_id);

    bool Register();                     // Register with Central Server
    bool StartHeartbeat(int interval);   // Start heartbeat loop
    void StopHeartbeat();                // Stop heartbeat loop
    bool SendHeartbeat();                // Send single heartbeat
    void SetActiveJobs(const std::vector<std::string>& ids);

    std::string GetNodeId() const;
    bool IsRegistered() const;
};
```

### DeploymentManager

```cpp
class DeploymentManager {
public:
    std::string AcceptDeployment(
        const std::string& model_id,
        protocol::DeploymentType type,
        const protocol::DeploymentConfig& config
    );

    void StopDeployment(const std::string& deployment_id);
    protocol::DeploymentStatus GetDeploymentStatus(const std::string& id) const;
    std::vector<DeploymentMetrics> GetDeploymentMetrics(const std::string& id) const;

    size_t GetActiveDeploymentCount() const;
    bool HasDeployment(const std::string& id) const;
};
```

### ModelLoader

```cpp
class ModelLoader {
public:
    virtual bool Load(const std::string& path) = 0;
    virtual bool Infer(const InputMap& inputs, OutputMap& outputs) = 0;
    virtual std::vector<TensorSpec> GetInputSpecs() const = 0;
    virtual std::vector<TensorSpec> GetOutputSpecs() const = 0;
    virtual uint64_t GetMemoryUsage() const = 0;
    virtual void Unload() = 0;
    virtual bool IsLoaded() const = 0;
    virtual std::string GetFormat() const = 0;
};
```

---

## Testing

### Unit Tests

```bash
cd build
ctest --output-on-failure

# Run specific tests
./cyxwiz-tests "[node_client]"
./cyxwiz-tests "[deployment_manager]"
```

### Integration Tests

```bash
# Start Central Server in one terminal
cd cyxwiz-central-server
cargo run

# Start Server Node in another terminal
cd cyxwiz-server-node/build
./cyxwiz-server-node

# Check registration in Central Server TUI
# Navigate to "Nodes" tab - should see newly registered node
```

---

## Troubleshooting

### Common Issues

**Problem:** "Failed to register with Central Server"
```
Solution: Ensure Central Server is running on localhost:50051
Check: netstat -an | grep 50051
```

**Problem:** "Failed to create PTY" (Linux)
```
Solution: Ensure /dev/ptmx permissions are correct
Check: ls -la /dev/ptmx
```

**Problem:** "Model loading failed"
```
Solution: Check model file exists and format is correct
Check: file ./models/model.onnx
```

**Problem:** "GPU not detected"
```
Solution: Install GPU drivers and verify ArrayFire can detect device
Check: af_info (ArrayFire utility)
```

---

## License

CyxWiz Server Node is part of the CyxWiz project.
**License:** MIT (see root LICENSE file)

---

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Implement changes following C++20 style guide
4. Add tests for new functionality
5. Run `clang-format` on modified files
6. Submit pull request

**Code Style:**
- C++20 features preferred
- Smart pointers (`std::unique_ptr`, `std::shared_ptr`)
- RAII for resource management
- spdlog for logging
- Follow existing naming conventions

---

## Contact & Support

- **GitHub Issues:** https://github.com/cyxwiz/cyxwiz/issues
- **Discord:** https://discord.gg/cyxwiz
- **Docs:** https://docs.cyxwiz.com

---

**Last Updated:** 2025-11-12
**Version:** 0.1.0 (Alpha)
