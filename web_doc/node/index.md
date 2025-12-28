# CyxWiz Server Node

The Server Node (also called "miner") is the distributed compute worker that executes ML training jobs. It can run in GUI mode for interactive use or as a background daemon.

## Overview

The Server Node provides:
- **Job Execution** - Train models using GPU/CPU
- **Hardware Monitoring** - Track resource utilization
- **OpenAI-Compatible API** - Serve models via REST
- **Central Server Communication** - Register and receive jobs
- **Pool Mining** - Collaborative training (future)
- **GUI and Daemon Modes** - Flexible deployment

## Architecture

```
cyxwiz-server-node/
├── src/
│   ├── main.cpp              # Entry point (mode selection)
│   ├── gui_main.cpp          # GUI mode entry
│   ├── daemon_main.cpp       # Daemon mode entry
│   ├── core/
│   │   ├── state_manager.cpp    # Application state
│   │   ├── config_manager.cpp   # Configuration
│   │   ├── backend_manager.cpp  # cyxwiz-backend integration
│   │   ├── metrics_collector.cpp # Hardware metrics
│   │   ├── metrics_storage.cpp  # Metrics history
│   │   └── device_pool.cpp      # GPU/CPU pool
│   ├── gui/
│   │   ├── server_application.cpp
│   │   ├── server_main_window.cpp
│   │   ├── server_panel.cpp     # Base panel class
│   │   └── panels/
│   │       ├── dashboard_panel.cpp
│   │       ├── hardware_panel.cpp
│   │       ├── job_monitor_panel.cpp
│   │       ├── deployment_panel.cpp
│   │       ├── api_keys_panel.cpp
│   │       ├── settings_panel.cpp
│   │       └── wallet_panel.cpp
│   ├── http/
│   │   ├── openai_api_server.cpp
│   │   └── routes/
│   │       ├── chat_route.cpp
│   │       ├── completions_route.cpp
│   │       └── models_route.cpp
│   ├── ipc/
│   │   ├── daemon_client.cpp    # GUI→Daemon communication
│   │   └── daemon_service.cpp   # Daemon IPC server
│   ├── security/
│   │   ├── tls_config.cpp
│   │   ├── api_key_manager.cpp
│   │   ├── audit_logger.cpp
│   │   └── docker_manager.cpp
│   ├── auth/
│   │   └── auth_manager.cpp
│   ├── node_client.cpp          # Central Server client
│   ├── node_service.cpp         # gRPC service impl
│   ├── job_executor.cpp         # Job execution engine
│   └── job_execution_service.cpp
└── tui/
    ├── tui_application.cpp      # Terminal UI
    └── components/
        └── resource_gauges.cpp
```

## Documentation Sections

| Section | Description |
|---------|-------------|
| [GUI Mode](gui.md) | Interactive desktop application |
| [Daemon Mode](daemon.md) | Background service operation |
| [Job Execution](jobs.md) | How jobs are executed |
| [Hardware Monitoring](monitoring.md) | Resource tracking |
| [OpenAI API](api.md) | Model serving interface |
| [Security](security.md) | Sandboxing and auth |
| [Configuration](configuration.md) | Settings and options |

## Running Modes

### GUI Mode (Default)

Interactive application with visual dashboard:

```bash
# Windows
.\cyxwiz-server-node.exe

# Linux
./cyxwiz-server-node
```

### Daemon Mode

Background service for production deployment:

```bash
# Windows
.\cyxwiz-server-node.exe --daemon

# Linux
./cyxwiz-server-node --daemon
```

### TUI Mode

Terminal-based interface:

```bash
./cyxwiz-server-node --tui
```

## GUI Interface

```
+------------------------------------------------------------------+
|  CyxWiz Server Node v0.1.0                      [_] [x]          |
+------------------------------------------------------------------+
|  File  View  Settings  Help                                       |
+------------------------------------------------------------------+
|                                                                   |
|  DASHBOARD                                                        |
|  +----------------------------+  +----------------------------+   |
|  | STATUS: Online             |  | JOBS                       |   |
|  | Node ID: abc123...         |  | Active: 2                  |   |
|  | Uptime: 12h 34m            |  | Queued: 5                  |   |
|  | Central: Connected         |  | Completed: 156             |   |
|  +----------------------------+  +----------------------------+   |
|                                                                   |
|  HARDWARE                                                         |
|  +-----------------------------------------------------------+   |
|  | CPU:  [============        ] 58%   Cores: 8               |   |
|  | RAM:  [========            ] 42%   12.4/32 GB             |   |
|  | GPU:  [================    ] 78%   RTX 4060 8GB           |   |
|  | VRAM: [==============      ] 71%   5.7/8 GB               |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  ACTIVE JOBS                                                      |
|  +-----------------------------------------------------------+   |
|  | Job ID     | Status  | Progress | ETA      | Client       |   |
|  +-----------------------------------------------------------+   |
|  | j-abc123   | Running |   45%    | 2h 15m   | user@...     |   |
|  | j-def456   | Running |   12%    | 5h 30m   | user@...     |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
+------------------------------------------------------------------+
```

### GUI Panels

| Panel | Description |
|-------|-------------|
| **Dashboard** | Overview of node status |
| **Hardware** | CPU, GPU, memory monitoring |
| **Jobs** | Active and queued jobs |
| **Deployment** | Model deployment management |
| **API Keys** | Manage API access keys |
| **Analytics** | Historical metrics |
| **Settings** | Node configuration |
| **Wallet** | CYXWIZ token management |

## Central Server Connection

### Registration Flow

```
1. Node starts
        |
        v
2. Collect hardware info
   - GPU devices
   - CPU cores
   - RAM
   - Network info
        |
        v
3. Send RegisterNodeRequest
   to Central Server (50051)
        |
        v
4. Receive node_id and session_token
        |
        v
5. Start heartbeat loop (every 10s)
        |
        v
6. Wait for job assignments
```

### Network Mode vs Standalone Mode

| Feature | Network Mode | Standalone Mode |
|---------|--------------|-----------------|
| Central Server | Connected | Not connected |
| Job Source | Network jobs | Local only |
| Heartbeat | Active | Disabled |
| Payments | Enabled | Disabled |
| Model Serving | Available | Available |

## Job Execution

### Job Lifecycle

```
1. Receive AssignJobRequest
        |
        v
2. Validate authorization
        |
        v
3. Download dataset (if needed)
        |
        v
4. Initialize training
   - Load model config
   - Create optimizer
   - Setup data loaders
        |
        v
5. Training loop
   - Forward pass
   - Compute loss
   - Backward pass
   - Optimizer step
   - Report progress (every N batches)
        |
        v
6. Save model weights
        |
        v
7. Upload to storage
        |
        v
8. Send ReportCompletionRequest
```

### Resource Management

```cpp
// Device pool manages GPU allocation
class DevicePool {
public:
    DeviceInfo AcquireDevice(const JobRequirements& req);
    void ReleaseDevice(int device_id);
    bool IsDeviceAvailable(int device_id);
    size_t GetAvailableMemory(int device_id);
};
```

## OpenAI-Compatible API

The Server Node can serve deployed models via a REST API compatible with OpenAI's API format.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/models` | List available models |
| POST | `/v1/chat/completions` | Chat completion |
| POST | `/v1/completions` | Text completion |
| POST | `/v1/embeddings` | Generate embeddings |

### Example Usage

```bash
# List models
curl http://localhost:8000/v1/models

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deployed-model-1",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Configuration

```toml
[api]
enabled = true
port = 8000
max_concurrent_requests = 10
rate_limit_per_minute = 60

[api.tls]
enabled = false
cert_path = "cert.pem"
key_path = "key.pem"
```

## Security

### Docker Sandboxing

Untrusted training code runs in Docker containers:

```cpp
class DockerManager {
public:
    std::string CreateContainer(const JobConfig& config);
    void StartContainer(const std::string& container_id);
    void StopContainer(const std::string& container_id);
    void RemoveContainer(const std::string& container_id);

    void SetResourceLimits(
        const std::string& container_id,
        size_t memory_limit,
        int cpu_cores
    );
};
```

### API Key Management

```cpp
class APIKeyManager {
public:
    std::string GenerateKey();
    bool ValidateKey(const std::string& key);
    void RevokeKey(const std::string& key);
    KeyPermissions GetPermissions(const std::string& key);
};
```

### Audit Logging

All security events are logged:
- API key usage
- Job execution start/end
- Authentication attempts
- Resource limit violations

## Metrics Collection

### Hardware Metrics

Collected every 5 seconds:

```cpp
struct HardwareMetrics {
    float cpu_usage;           // 0-100%
    float cpu_temperature;     // Celsius
    size_t ram_used;           // Bytes
    size_t ram_total;          // Bytes

    // Per-GPU metrics
    std::vector<GPUMetrics> gpus;
};

struct GPUMetrics {
    int device_id;
    std::string name;
    float utilization;         // 0-100%
    float temperature;         // Celsius
    size_t memory_used;        // Bytes
    size_t memory_total;       // Bytes
    float power_draw;          // Watts
};
```

### Metrics Storage

Metrics are stored locally for visualization:

```cpp
class MetricsStorage {
public:
    void Store(const HardwareMetrics& metrics);
    std::vector<HardwareMetrics> GetRange(
        std::chrono::time_point start,
        std::chrono::time_point end
    );
    void Cleanup(int days_to_keep = 7);
};
```

## Configuration

### Config File

Located at `~/.cyxwiz/server-node/config.toml`:

```toml
[node]
name = "my-node"
max_concurrent_jobs = 3

[central_server]
address = "localhost:50051"
heartbeat_interval_seconds = 10
reconnect_delay_seconds = 5

[devices]
# Specific GPU selection (empty = all)
gpu_ids = []
# Maximum memory per job
max_gpu_memory_mb = 8000

[api]
enabled = true
port = 8000

[logging]
level = "info"
file = "server-node.log"
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `CYXWIZ_NODE_NAME` | Override node name |
| `CYXWIZ_CENTRAL_SERVER` | Central server address |
| `CYXWIZ_LOG_LEVEL` | Logging level |

## Deployment

### Systemd Service

```ini
[Unit]
Description=CyxWiz Server Node
After=network.target

[Service]
Type=simple
User=cyxwiz
ExecStart=/usr/local/bin/cyxwiz-server-node --daemon
Restart=always
Environment=RUST_LOG=info

[Install]
WantedBy=multi-user.target
```

### Docker Deployment

```dockerfile
FROM nvidia/cuda:12.0-base
COPY cyxwiz-server-node /usr/local/bin/
COPY config.toml /etc/cyxwiz/
EXPOSE 50052 50053 8000
CMD ["cyxwiz-server-node", "--daemon"]
```

### Requirements

- **GPU**: NVIDIA with CUDA 11+ (optional)
- **RAM**: 16GB minimum
- **Storage**: 100GB for models/datasets
- **Network**: Stable connection to Central Server

---

**Next**: [GUI Mode](gui.md) | [Job Execution](jobs.md)
