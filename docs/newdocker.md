# Docker Integration for CyxWiz Server Node

## Overview

This document outlines the Docker integration strategy for CyxWiz Server Node, designed to be **fully compatible** with the existing architecture while adding sandboxing, security, and multi-tenancy capabilities.

### Goals

1. **Sandboxing/Security** - Isolate untrusted training jobs from the host system
2. **Dependency Management** - Each job runs with specific ML framework versions
3. **Resource Isolation** - Limit CPU, memory, GPU allocation per job
4. **Reproducibility** - Consistent environment across different Server Nodes
5. **Multi-tenancy** - Run multiple jobs safely on the same node
6. **Backward Compatibility** - Native execution remains the default fallback

---

## Current Architecture Analysis

### Existing Components (Unchanged)

```
cyxwiz-server-node/
├── daemon_main.cpp           # Entry point - creates all services
├── job_executor.h/cpp        # Core training execution
├── job_execution_service.cpp # P2P gRPC service (Engine→Node)
├── node_client.cpp           # Central Server communication
├── core/
│   ├── device_pool.h/cpp     # Multi-GPU management (KEEP AS-IS)
│   ├── backend_manager.cpp   # ArrayFire backend selection
│   └── state_manager.cpp     # Node state persistence
└── ...
```

### Key Integration Points

| Component | Current | With Docker |
|-----------|---------|-------------|
| **JobExecutor** | Runs training in worker threads | Adds ContainerExecutor mode |
| **DevicePool** | Manages GPU allocation | Extended for container GPU passthrough |
| **JobExecutionService** | P2P streaming with Engine | No changes needed |
| **NodeClient** | Reports to Central Server | Reports container capabilities |

---

## Architecture: Daemon Outside, Jobs Inside

```
┌─────────────────────────────────────────────────────────────────┐
│                         HOST SYSTEM                              │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Server Node Daemon (Native)                  │   │
│  │                                                           │   │
│  │  ┌─────────────────┐    ┌──────────────────────────────┐ │   │
│  │  │  JobExecutor    │    │  JobExecutionService         │ │   │
│  │  │                 │    │  (P2P gRPC - unchanged)      │ │   │
│  │  │  ┌───────────┐  │    │                              │ │   │
│  │  │  │ Native    │  │    │  - ConnectToNode()           │ │   │
│  │  │  │ Execution │  │    │  - SendJob()                 │ │   │
│  │  │  └───────────┘  │    │  - StreamTrainingMetrics()   │ │   │
│  │  │  ┌───────────┐  │    │                              │ │   │
│  │  │  │ Container │  │    └──────────────────────────────┘ │   │
│  │  │  │ Execution │──┼──────────────────┐                  │   │
│  │  │  └───────────┘  │                  │                  │   │
│  │  └─────────────────┘                  │                  │   │
│  │                                       │                  │   │
│  │  ┌─────────────────┐    ┌─────────────┴──────────────┐   │   │
│  │  │  DevicePool     │    │  ContainerManager          │   │   │
│  │  │  (GPU mgmt)     │    │  (Docker/Podman API)       │   │   │
│  │  └─────────────────┘    └────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                    Docker Socket / Podman                        │
│                              │                                   │
│         ┌────────────────────┼────────────────────┐             │
│         ▼                    ▼                    ▼             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │   Job 1     │     │   Job 2     │     │   Job 3     │       │
│  │ Container   │     │ Container   │     │ Container   │       │
│  │             │     │             │     │             │       │
│  │ cyxwiz-job- │     │ cyxwiz-job- │     │ cyxwiz-job- │       │
│  │ executor    │     │ executor    │     │ executor    │       │
│  │ + CUDA 12   │     │ + CUDA 11   │     │ CPU only    │       │
│  │             │     │             │     │             │       │
│  │ GPU: 0      │     │ GPU: 1      │     │ No GPU      │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Execution Mode Abstraction

**New Files:**
- `cyxwiz-server-node/src/execution/execution_mode.h`
- `cyxwiz-server-node/src/execution/native_executor.cpp`
- `cyxwiz-server-node/src/execution/container_executor.cpp`
- `cyxwiz-server-node/src/execution/container_manager.h/cpp`

**Changes to Existing:**
- `job_executor.h/cpp` - Add execution mode selection

```cpp
// execution/execution_mode.h
#pragma once

namespace cyxwiz {
namespace servernode {
namespace execution {

enum class ExecutionMode {
    Native,     // Direct execution (current behavior)
    Docker,     // Docker container isolation
    Podman,     // Rootless Podman (more secure)
};

enum class SandboxLevel {
    None = 0,   // Native only - trusted jobs
    Basic = 1,  // Podman rootless
    Full = 2,   // Docker with NVIDIA toolkit
};

struct ExecutionCapabilities {
    bool docker_available = false;
    bool podman_available = false;
    bool nvidia_toolkit = false;
    SandboxLevel max_sandbox_level = SandboxLevel::None;
    std::vector<std::string> available_images;
};

// Detect available execution modes
ExecutionCapabilities DetectCapabilities();

// Select best execution mode for a job
ExecutionMode SelectExecutionMode(
    const ExecutionCapabilities& caps,
    bool job_requires_sandbox,
    bool prefer_container);

} // namespace execution
} // namespace servernode
} // namespace cyxwiz
```

### Phase 2: Container Manager

```cpp
// execution/container_manager.h
#pragma once

#include <string>
#include <memory>
#include <functional>
#include <vector>
#include <optional>
#include "job.pb.h"

namespace cyxwiz {
namespace servernode {
namespace execution {

struct ContainerConfig {
    std::string image;
    std::vector<int> gpu_ids;           // GPUs to passthrough
    size_t memory_limit_mb = 0;         // 0 = unlimited
    int cpu_cores = 0;                  // 0 = unlimited
    size_t shm_size_mb = 2048;          // Shared memory for CUDA IPC
    bool read_only_root = true;
    bool no_network = true;             // Security: disable network
    std::string job_config_base64;      // Serialized JobConfig
    std::string data_mount_path;        // Host path for /job/data
    std::string output_mount_path;      // Host path for /job/output
    std::string checkpoint_mount_path;  // Host path for /job/checkpoints
};

struct ContainerStatus {
    std::string container_id;
    bool running = false;
    int exit_code = -1;
    std::string error_message;
};

// Progress callback from container stdout
using ContainerProgressCallback = std::function<void(
    const std::string& job_id,
    double progress,
    const std::map<std::string, double>& metrics,
    const std::string& log_line
)>;

class ContainerManager {
public:
    ContainerManager();
    ~ContainerManager();

    // Initialize with Docker/Podman socket
    bool Initialize(const std::string& socket_path = "/var/run/docker.sock");

    // Check if Docker/Podman is available
    bool IsAvailable() const;
    bool HasNvidiaToolkit() const;

    // Image management
    bool PullImage(const std::string& image, std::function<void(double)> progress_cb = nullptr);
    bool ImageExists(const std::string& image) const;
    std::vector<std::string> ListImages(const std::string& prefix = "cyxwiz/") const;

    // Container lifecycle
    std::optional<std::string> CreateContainer(const ContainerConfig& config);
    bool StartContainer(const std::string& container_id);
    bool StopContainer(const std::string& container_id, int timeout_seconds = 10);
    bool RemoveContainer(const std::string& container_id);
    ContainerStatus GetContainerStatus(const std::string& container_id) const;

    // Stream container logs (for progress reporting)
    // Calls callback for each line of stdout (expects JSON progress)
    void StreamLogs(const std::string& container_id,
                   ContainerProgressCallback callback,
                   std::atomic<bool>& stop_flag);

    // Execute command in running container
    std::pair<int, std::string> Exec(const std::string& container_id,
                                      const std::vector<std::string>& command);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace execution
} // namespace servernode
} // namespace cyxwiz
```

### Phase 3: Integration with JobExecutor

**Modify `job_executor.h`:**

```cpp
// Add to JobExecutor class
class JobExecutor {
public:
    // Existing interface unchanged...

    // New: Set execution mode preference
    void SetExecutionMode(execution::ExecutionMode mode);
    execution::ExecutionMode GetExecutionMode() const;

    // New: Get execution capabilities
    execution::ExecutionCapabilities GetCapabilities() const;

private:
    // Existing members...

    // New: Container support
    execution::ExecutionMode execution_mode_ = execution::ExecutionMode::Native;
    std::unique_ptr<execution::ContainerManager> container_manager_;
    execution::ExecutionCapabilities capabilities_;

    // New: Execute job in container
    void ExecuteJobInContainer(const std::string& job_id);

    // Existing: Execute job natively (renamed for clarity)
    void ExecuteJobNative(const std::string& job_id);
};
```

**Modify `job_executor.cpp` - ExecuteJob():**

```cpp
void JobExecutor::ExecuteJob(const std::string& job_id) {
    // Select execution mode based on job requirements and capabilities
    auto mode = execution_mode_;

    // If container requested but not available, fall back to native
    if (mode != execution::ExecutionMode::Native && !container_manager_->IsAvailable()) {
        spdlog::warn("Container execution requested but not available, falling back to native");
        mode = execution::ExecutionMode::Native;
    }

    // Execute based on mode
    switch (mode) {
        case execution::ExecutionMode::Docker:
        case execution::ExecutionMode::Podman:
            ExecuteJobInContainer(job_id);
            break;
        case execution::ExecutionMode::Native:
        default:
            ExecuteJobNative(job_id);
            break;
    }
}
```

### Phase 4: Container Job Executor Binary

Create a minimal binary for execution inside containers:

**New File: `cyxwiz-server-node/src/container/job_executor_main.cpp`**

```cpp
// Standalone binary for container execution
// This is what runs inside Docker/Podman containers

#include <iostream>
#include <fstream>
#include <chrono>
#include <nlohmann/json.hpp>
#include <cyxwiz/cyxwiz.h>

// Progress is reported via stdout JSON
void ReportProgress(const std::string& job_id, int epoch, int total_epochs,
                    double loss, double accuracy) {
    nlohmann::json j;
    j["type"] = "progress";
    j["job_id"] = job_id;
    j["epoch"] = epoch;
    j["total_epochs"] = total_epochs;
    j["loss"] = loss;
    j["accuracy"] = accuracy;
    j["timestamp"] = std::chrono::system_clock::now().time_since_epoch().count();

    std::cout << j.dump() << std::endl;
    std::cout.flush();
}

void ReportComplete(const std::string& job_id, bool success,
                   const std::string& model_path, const std::string& error = "") {
    nlohmann::json j;
    j["type"] = "complete";
    j["job_id"] = job_id;
    j["success"] = success;
    j["model_path"] = model_path;
    j["error"] = error;

    std::cout << j.dump() << std::endl;
    std::cout.flush();
}

int main(int argc, char* argv[]) {
    // Parse job config from argument or environment
    std::string config_base64;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--job-config=") == 0) {
            config_base64 = arg.substr(13);
        }
    }

    if (config_base64.empty()) {
        config_base64 = std::getenv("CYXWIZ_JOB_CONFIG") ?: "";
    }

    if (config_base64.empty()) {
        std::cerr << "No job config provided" << std::endl;
        return 1;
    }

    // Decode and parse job config
    // ... (decode base64, parse protobuf)

    // Initialize ArrayFire with available GPU
    cyxwiz::Initialize();

    // Run training loop
    // ... (similar to JobExecutor::RunTraining but simpler)

    // Report completion
    ReportComplete(job_id, true, "/job/output/model.cyxmodel");

    return 0;
}
```

---

## Container Images

### Base Image Hierarchy

```
cyxwiz/base:latest
    ├── cyxwiz/executor:cuda12-latest
    ├── cyxwiz/executor:cuda11-latest
    ├── cyxwiz/executor:rocm-latest
    └── cyxwiz/executor:cpu-latest
```

### Dockerfile for CUDA 12

```dockerfile
# cyxwiz/executor:cuda12-latest
FROM nvidia/cuda:12.2-cudnn8-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    libfftw3-dev \
    liblapack-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ArrayFire (CUDA build)
COPY --from=arrayfire/arrayfire:3.8.3-cuda /opt/arrayfire /opt/arrayfire
ENV AF_PATH=/opt/arrayfire
ENV LD_LIBRARY_PATH=/opt/arrayfire/lib64:$LD_LIBRARY_PATH

# CyxWiz Backend Library
COPY lib/libcyxwiz-backend.so /usr/local/lib/
COPY include/cyxwiz/ /usr/local/include/cyxwiz/
RUN ldconfig

# Job Executor Binary
COPY bin/cyxwiz-job-executor /usr/local/bin/
RUN chmod +x /usr/local/bin/cyxwiz-job-executor

# Create non-root user
RUN useradd -m -u 1000 cyxwiz
USER cyxwiz

# Working directory
WORKDIR /job
VOLUME ["/job/data", "/job/checkpoints", "/job/output"]

ENTRYPOINT ["cyxwiz-job-executor"]
```

### Image Selection Logic

```cpp
std::string SelectImage(const protocol::JobConfig& config,
                        const std::vector<std::string>& available_images) {
    // Priority: exact match > cuda12 > cuda11 > cpu

    if (config.required_device() == protocol::DEVICE_TYPE_CUDA) {
        // Check for CUDA version preference
        for (const auto& img : available_images) {
            if (img.find("cuda12") != std::string::npos) {
                return img;
            }
        }
        for (const auto& img : available_images) {
            if (img.find("cuda11") != std::string::npos) {
                return img;
            }
        }
    }

    // Fallback to CPU
    for (const auto& img : available_images) {
        if (img.find("cpu") != std::string::npos) {
            return img;
        }
    }

    return "cyxwiz/executor:cpu-latest";  // Default
}
```

---

## Security Configuration

### Container Security Flags

```cpp
std::vector<std::string> GetSecurityFlags(bool strict = true) {
    std::vector<std::string> flags = {
        "--security-opt=no-new-privileges",  // Prevent privilege escalation
        "--cap-drop=ALL",                     // Drop all capabilities
        "--read-only",                        // Read-only root filesystem
        "--user=1000:1000",                   // Non-root user
        "--pids-limit=1024",                  // Limit processes
    };

    if (strict) {
        flags.push_back("--network=none");   // No network access
    }

    return flags;
}
```

### GPU Passthrough

```cpp
std::string BuildGPUFlags(const std::vector<int>& gpu_ids) {
    if (gpu_ids.empty()) {
        return "";
    }

    std::stringstream ss;
    ss << "--gpus '\"device=";
    for (size_t i = 0; i < gpu_ids.size(); ++i) {
        if (i > 0) ss << ",";
        ss << gpu_ids[i];
    }
    ss << "\"'";

    return ss.str();
}
```

### Volume Mounts

```cpp
std::vector<std::string> BuildVolumeMounts(const ContainerConfig& config) {
    return {
        "-v " + config.data_mount_path + ":/job/data:ro",          // Read-only data
        "-v " + config.checkpoint_mount_path + ":/job/checkpoints", // Read-write checkpoints
        "-v " + config.output_mount_path + ":/job/output",          // Read-write output
        "--tmpfs /tmp:size=1g",                                      // Ephemeral temp space
        "--shm-size=" + std::to_string(config.shm_size_mb) + "m",   // Shared memory for CUDA
    };
}
```

---

## Communication: Daemon ↔ Container

### Progress via stdout JSON Streaming (Recommended)

The simplest and most compatible approach:

1. Container writes JSON to stdout
2. Daemon captures via Docker API log streaming
3. Daemon parses JSON and updates progress

```cpp
// In ContainerManager::StreamLogs()
void ContainerManager::StreamLogs(const std::string& container_id,
                                  ContainerProgressCallback callback,
                                  std::atomic<bool>& stop_flag) {
    // Use Docker API to stream logs
    // Parse each line as JSON
    // Call callback with parsed progress

    while (!stop_flag) {
        std::string line = docker_api_->ReadLogLine(container_id);
        if (line.empty()) break;

        try {
            auto j = nlohmann::json::parse(line);

            if (j["type"] == "progress") {
                std::map<std::string, double> metrics;
                metrics["loss"] = j["loss"];
                metrics["accuracy"] = j["accuracy"];

                double progress = static_cast<double>(j["epoch"]) / j["total_epochs"];
                callback(j["job_id"], progress, metrics, line);
            }
            else if (j["type"] == "complete") {
                // Handle completion
            }
        } catch (...) {
            // Non-JSON line, treat as log
            callback("", 0, {}, line);
        }
    }
}
```

---

## Integration with P2P Flow

### No Changes to JobExecutionService

The P2P service (`job_execution_service.cpp`) **does not change**. It continues to:
1. Receive jobs from Engine
2. Call `JobExecutor::ExecuteJobAsync()`
3. Stream progress via `StreamTrainingMetrics()`

The only change is internal to `JobExecutor` - whether it runs training natively or in a container.

```
Engine                    Server Node Daemon              Container
  │                             │                            │
  │  SendJob()                  │                            │
  ├────────────────────────────>│                            │
  │                             │                            │
  │                             │  JobExecutor.ExecuteJobAsync()
  │                             │        │                   │
  │                             │        ▼                   │
  │                             │  [Mode = Docker?]          │
  │                             │        │                   │
  │                             │        ├──► docker run     │
  │                             │        │    cyxwiz-job-executor
  │                             │        │         │         │
  │                             │        │         ▼         │
  │                             │        │    [Training Loop]│
  │                             │        │         │         │
  │  StreamTrainingMetrics()    │◄───────┼─────────┤ stdout  │
  │◄────────────────────────────│        │    {"progress":..}│
  │                             │        │         │         │
  │                             │        │         ▼         │
  │                             │◄───────┼─────────┤ exit 0  │
  │  TrainingComplete           │        │                   │
  │◄────────────────────────────│        │                   │
```

---

## Configuration

### Server Node Config (YAML)

```yaml
# /etc/cyxwiz/server-node.yaml

# Execution settings
execution:
  # Preferred mode: native, docker, podman, auto
  # auto = use container if available, fallback to native
  mode: auto

  # Allow native execution for untrusted jobs?
  # If false and container unavailable, job is rejected
  allow_untrusted_native: false

# Docker settings (only used if mode = docker or auto)
docker:
  enabled: true
  socket: /var/run/docker.sock

  # Image settings
  registry: docker.io/cyxwiz
  pull_policy: if-not-present  # always, never, if-not-present

  # Pre-pull these images on startup
  preload_images:
    - cyxwiz/executor:cuda12-latest
    - cyxwiz/executor:cpu-latest

  # Default resource limits
  default_limits:
    memory_mb: 16384
    cpu_cores: 4
    shm_size_mb: 2048
    pids: 1024

  # Security
  security:
    read_only_root: true
    no_network: true
    drop_capabilities: true

  # Cleanup
  cleanup:
    remove_containers: true
    keep_logs_hours: 24

# GPU settings
gpu:
  # How to allocate GPUs to containers
  # exclusive = one GPU per container
  # shared = multiple containers can share GPUs (with memory limits)
  allocation: exclusive

  # Memory limit per container (0 = unlimited)
  memory_limit_mb: 0
```

---

## Node Registration Update

### Protocol Changes

```protobuf
// In node.proto - extend NodeCapabilities
message NodeCapabilities {
    // Existing fields...

    // Container support (new)
    bool docker_available = 20;
    bool podman_available = 21;
    bool nvidia_container_toolkit = 22;

    // Sandbox level
    enum SandboxLevel {
        SANDBOX_NONE = 0;    // Native only
        SANDBOX_BASIC = 1;   // Podman rootless
        SANDBOX_FULL = 2;    // Docker + NVIDIA toolkit
    }
    SandboxLevel sandbox_level = 23;

    // Available container images
    repeated string container_images = 24;
}
```

### Central Server Job Routing

Central Server can use sandbox_level for job routing:
- Untrusted jobs → only route to nodes with `sandbox_level >= BASIC`
- GPU jobs → only route to nodes with `nvidia_container_toolkit = true`

---

## Implementation Tasks

### Phase 1: Core Infrastructure (Week 1)
- [ ] Create `execution/` directory structure
- [ ] Implement `ExecutionMode` enum and `DetectCapabilities()`
- [ ] Add ContainerManager with Docker socket detection
- [ ] Implement `IsAvailable()` and `HasNvidiaToolkit()` checks

### Phase 2: Container Lifecycle (Week 2)
- [ ] Implement `CreateContainer()` with security flags
- [ ] Implement `StartContainer()`, `StopContainer()`, `RemoveContainer()`
- [ ] Implement `StreamLogs()` for progress capture
- [ ] Add GPU passthrough support

### Phase 3: JobExecutor Integration (Week 3)
- [ ] Add execution mode to JobExecutor
- [ ] Implement `ExecuteJobInContainer()`
- [ ] Wire up progress callbacks from container logs
- [ ] Test native ↔ container fallback

### Phase 4: Container Job Binary (Week 4)
- [ ] Create `cyxwiz-job-executor` standalone binary
- [ ] Build Docker images for CUDA 12, CUDA 11, CPU
- [ ] Test image with sample training job
- [ ] Set up image registry (Docker Hub or private)

### Phase 5: Node Registration (Week 5)
- [ ] Extend NodeCapabilities protobuf
- [ ] Update NodeClient to report container capabilities
- [ ] Update Central Server job routing (optional)

### Phase 6: Testing & Hardening (Week 6)
- [ ] End-to-end tests: native, docker, fallback
- [ ] Security audit of container config
- [ ] Performance benchmarks: container vs native overhead
- [ ] Documentation and deployment guide

---

## Migration Path

### For Existing Deployments

1. **No action required** - Default mode is `auto` with fallback to native
2. **Optional** - Install Docker and NVIDIA Container Toolkit for sandboxing
3. **Recommended** - Pull executor images: `docker pull cyxwiz/executor:cuda12-latest`

### For New Deployments

1. Install Docker + NVIDIA Container Toolkit
2. Configure `execution.mode: docker` in server-node.yaml
3. Pull required images
4. Run daemon normally - jobs will execute in containers

---

## Open Questions (Resolved)

| Question | Decision |
|----------|----------|
| **Image Updates** | Use semantic versioning, Central Server can specify required version in job config |
| **Data Transfer** | Mount host path read-only (compatible with current lazy loading) |
| **Checkpoint Storage** | Local host volume mounted into container |
| **Windows Support** | Native execution primary on Windows; Docker Desktop WSL2 as optional |
| **Rootless Docker** | Support Podman rootless as `SandboxLevel::Basic` |

---

## Security Considerations

1. **Network Isolation** - Containers have `--network=none` by default
2. **No Privilege Escalation** - `--security-opt=no-new-privileges`
3. **Minimal Capabilities** - `--cap-drop=ALL`
4. **Read-only Root** - `--read-only` with tmpfs for temp files
5. **Non-root User** - Containers run as UID 1000
6. **Resource Limits** - CPU, memory, PIDs all limited

---

## References

- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [Podman Rootless](https://github.com/containers/podman/blob/main/docs/tutorials/rootless_tutorial.md)
