# CyxWiz Full Connection Architecture

## Overview

This document extends the base connection architecture to support:
1. **Hardware Detection** - Enumerate all compute devices (CPU, GPUs) on the user's machine
2. **Resource Allocation** - User selects which devices to share and how much
3. **Per-Device Registration** - Each allocated device becomes a compute unit on the network
4. **Website Dashboard** - Display all devices with active/inactive status

## Shared Database Architecture

The Website and Central Server share the same MongoDB database. No separate reporting mechanism is needed - data flows through the shared database.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW ARCHITECTURE                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐                              ┌──────────────────┐
    │  Server Node GUI │                              │   Website (Next) │
    │   (C++ Client)   │                              │   Dashboard      │
    └────────┬─────────┘                              └────────┬─────────┘
             │                                                  │
             │ Heartbeats                                       │ Read
             │ POST /api/nodes/:id/heartbeat                    │ GET /api/users/:id/devices
             │                                                  │
             v                                                  v
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         CYXWIZ RUST API (Central Server)                     │
    │                              localhost:8080                                  │
    │                                                                              │
    │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
    │   │  Heartbeat  │    │   Node      │    │   Auth      │                     │
    │   │  Handler    │    │   CRUD      │    │   Handler   │                     │
    │   └──────┬──────┘    └──────┬──────┘    └─────────────┘                     │
    │          │                  │                                                │
    │          └────────┬─────────┘                                                │
    │                   │                                                          │
    │                   v                                                          │
    │          ┌─────────────────┐                                                 │
    │          │  MongoDB Driver │                                                 │
    │          └────────┬────────┘                                                 │
    └───────────────────┼──────────────────────────────────────────────────────────┘
                        │
                        │ Read/Write
                        v
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                              MONGODB DATABASE                                │
    │                                                                              │
    │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
    │   │   nodes     │    │   users     │    │  machines   │    │ compute_    │ │
    │   │ collection  │    │ collection  │    │ collection  │    │   units     │ │
    │   │             │    │             │    │             │    │ collection  │ │
    │   │ - node_id   │    │ - user_id   │    │ - machine_id│    │ - unit_id   │ │
    │   │ - status    │    │ - email     │    │ - hostname  │    │ - device_id │ │
    │   │ - last_seen │    │ - wallet    │    │ - hardware  │    │ - allocation│ │
    │   │ - specs     │    │             │    │ - status    │    │ - earnings  │ │
    │   └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
    └─────────────────────────────────────────────────────────────────────────────┘
                        │
                        │ Read (via Mongoose)
                        v
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         NEXT.JS WEB APPLICATION                              │
    │                              localhost:3000                                  │
    │                                                                              │
    │   ┌─────────────────────────────────────────────────────────────────────┐   │
    │   │  Server Components (RSC) - Direct MongoDB Access via Mongoose       │   │
    │   │                                                                      │   │
    │   │   DevicesPage                                                        │   │
    │   │     └── getUserNodes(userId)  ──────> nodes collection               │   │
    │   │         └── Returns devices with real-time status                    │   │
    │   │                                                                      │   │
    │   │   Status calculated from last_seen timestamp:                        │   │
    │   │     - last_seen < 15 sec ago  → ONLINE                               │   │
    │   │     - last_seen > 15 sec ago  → OFFLINE                              │   │
    │   └─────────────────────────────────────────────────────────────────────┘   │
    │                                                                              │
    │   ┌─────────────────────────────────────────────────────────────────────┐   │
    │   │  Auto-Refresh (Client-Side)                                          │   │
    │   │     - Polls every 10 seconds                                         │   │
    │   │     - Pauses after 2 min inactivity                                  │   │
    │   │     - router.refresh() → Re-fetches RSC data                         │   │
    │   └─────────────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

| Action | Source | Database | Consumer |
|--------|--------|----------|----------|
| Node heartbeat | Server Node GUI | `nodes` → updates `status`, `last_seen` | - |
| Register node | Server Node GUI | `nodes` → inserts new document | - |
| View devices | Website Dashboard | `nodes` → reads all user's nodes | User |
| Status calculation | Rust API & Website | Both calculate from `last_seen` | User |

### Key Points

1. **Single Source of Truth**: MongoDB is the shared database
2. **Rust API Writes**: Handles all writes (heartbeats, registration, updates)
3. **Website Reads**: Next.js reads directly from MongoDB via Mongoose
4. **Status Calculation**: Both Rust API and Website calculate status from `last_seen` timestamp
5. **No Message Queue Needed**: Simple polling + shared DB is sufficient for this scale

---

## REST API vs gRPC: Hybrid Communication

The system uses both REST API and gRPC, each for specific purposes. This hybrid approach provides the best of both worlds.

### Communication Protocol Summary

| Protocol | Use Case | Direction | Connection |
|----------|----------|-----------|------------|
| **REST API** | Auth, Registration, Heartbeats | Node → Server | Stateless, HTTP |
| **gRPC** | Task Assignment, Job Execution | Server ↔ Node | Persistent, Bidirectional |

### When to Use Each

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PROTOCOL SELECTION GUIDE                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

    REST API (HTTP)                              gRPC (HTTP/2)
    ───────────────                              ─────────────
    ✓ User Authentication (login)               ✓ Task Assignment
    ✓ Node Registration                         ✓ Job Streaming (model weights, data)
    ✓ Heartbeat Status Updates                  ✓ Real-time Progress Updates
    ✓ Allocation Changes                        ✓ Bidirectional Communication
    ✓ Earnings Queries                          ✓ Cancel/Abort Commands
    ✓ Dashboard Data                            ✓ Low-latency Operations

    [Stateless, Request-Response]               [Stateful, Stream-based]
    [Client initiates all requests]             [Server can push to client]
```

### Architecture with Both Protocols

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              HYBRID COMMUNICATION                                │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────────┐
    │                         SERVER NODE GUI (C++)                              │
    │                                                                            │
    │  ┌────────────────────┐              ┌────────────────────────────────┐  │
    │  │   REST API Client  │              │       gRPC Client              │  │
    │  │   (libcurl/cpp-httplib)           │       (grpc++)                 │  │
    │  │                    │              │                                │  │
    │  │  • Login/Logout    │              │  • Connect to Central Server   │  │
    │  │  • Register Node   │              │  • Receive task assignments    │  │
    │  │  • Send Heartbeats │              │  • Stream results back         │  │
    │  │  • Update Allocation │            │  • Handle abort/pause commands │  │
    │  └─────────┬──────────┘              └───────────────┬────────────────┘  │
    │            │                                          │                    │
    └────────────┼──────────────────────────────────────────┼────────────────────┘
                 │                                          │
                 │ HTTP/HTTPS                               │ gRPC (HTTP/2)
                 │ (Port 8080)                              │ (Port 50051)
                 │                                          │
                 v                                          v
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                      CYXWIZ CENTRAL SERVER (Rust)                           │
    │                                                                              │
    │   ┌──────────────────────────┐    ┌──────────────────────────────────────┐ │
    │   │     Axum REST API        │    │        Tonic gRPC Server             │ │
    │   │     (Port 8080)          │    │        (Port 50051)                  │ │
    │   │                          │    │                                      │ │
    │   │  • /api/auth/*           │    │  • NodeService                       │ │
    │   │  • /api/nodes/*          │    │    - StreamTasks()                   │ │
    │   │  • /api/compute-units/*  │    │    - ReportProgress()                │ │
    │   │  • /api/users/*          │    │    - StreamResults()                 │ │
    │   │                          │    │                                      │ │
    │   └───────────┬──────────────┘    └───────────────┬──────────────────────┘ │
    │               │                                    │                        │
    │               └──────────────┬─────────────────────┘                        │
    │                              │                                              │
    │                              v                                              │
    │                    ┌─────────────────┐                                      │
    │                    │  MongoDB Driver │                                      │
    │                    └────────┬────────┘                                      │
    └─────────────────────────────┼────────────────────────────────────────────────┘
                                  │
                                  v
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                              MONGODB DATABASE                                │
    │                                                                              │
    │   nodes | compute_units | tasks | task_results | users | earnings           │
    └─────────────────────────────────────────────────────────────────────────────┘
```

### Connection Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         FULL CONNECTION LIFECYCLE                                │
└─────────────────────────────────────────────────────────────────────────────────┘

     Server Node                                         Central Server
         │                                                      │
         │  1. POST /api/auth/login (email, password)           │
         │ ─────────────────────────────────────────────────────>│
         │                                                      │
         │                     JWT Token                        │
         │ <─────────────────────────────────────────────────────│
         │                                                      │
         │  2. POST /api/nodes/register (hardware, allocations) │
         │ ─────────────────────────────────────────────────────>│
         │                                                      │
         │                node_id, api_key, unit_ids            │
         │ <─────────────────────────────────────────────────────│
         │                                                      │
         │  3. gRPC Connect(node_id, api_key)                   │
         │ ════════════════════════════════════════════════════>│
         │                                                      │
         │  4. Heartbeat loop starts                            │
         │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─>│
         │     POST /api/nodes/:id/heartbeat (every 15s)        │
         │                                                      │
         │  5. gRPC: StreamTasks() opens                        │
         │ <════════════════════════════════════════════════════│
         │                                                      │
         │  6. Server pushes task assignment                    │
         │ <════════════════════════════════════════════════════│
         │     Task { id, type, model_url, input_data }         │
         │                                                      │
         │  7. Node executes task, streams progress             │
         │ ════════════════════════════════════════════════════>│
         │     Progress { task_id, percent, metrics }           │
         │                                                      │
         │  8. Node streams results                             │
         │ ════════════════════════════════════════════════════>│
         │     Result { task_id, output_data, stats }           │
         │                                                      │
         │  9. Server confirms completion, credits earnings     │
         │ <════════════════════════════════════════════════════│
         │     Confirmation { earnings, next_task? }            │
         │                                                      │

    Legend: ──────> REST API (stateless)
            ══════> gRPC (persistent stream)
            ─ ─ ─ > Periodic/background
```

### Existing gRPC Implementation

The gRPC services are already implemented in `D:\Dev\CyxWiz_Claude\cyxwiz-protocol\proto\`:

#### Proto Files

| File | Purpose |
|------|---------|
| `node.proto` | Node registration and heartbeat |
| `job.proto` | Job submission and management |
| `execution.proto` | P2P job execution between Engine and Node |
| `compute.proto` | Direct compute operations |
| `deployment.proto` | Model deployment |
| `common.proto` | Shared enums and messages |

#### Implemented gRPC Services

**1. NodeService** (Central Server ↔ Server Node)
```protobuf
service NodeService {
  rpc RegisterNode(RegisterNodeRequest) returns (RegisterNodeResponse);
  rpc Heartbeat(HeartbeatRequest) returns (HeartbeatResponse);
  rpc AssignJob(AssignJobRequest) returns (AssignJobResponse);
  rpc ReportProgress(ReportProgressRequest) returns (ReportProgressResponse);
  rpc ReportCompletion(ReportCompletionRequest) returns (ReportCompletionResponse);
  rpc GetNodeMetrics(GetNodeMetricsRequest) returns (GetNodeMetricsResponse);
}
```

**2. NodeDiscoveryService** (Find available nodes)
```protobuf
service NodeDiscoveryService {
  rpc FindNodes(FindNodesRequest) returns (FindNodesResponse);
  rpc ListNodes(ListNodesRequest) returns (ListNodesResponse);
  rpc GetNodeInfo(GetNodeInfoRequest) returns (GetNodeInfoResponse);
}
```

**3. JobExecutionService** (P2P: Engine ↔ Server Node on port 50052)
```protobuf
service JobExecutionService {
  rpc ConnectToNode(ConnectRequest) returns (ConnectResponse);
  rpc SendJob(SendJobRequest) returns (SendJobResponse);
  rpc StreamTrainingMetrics(stream TrainingCommand) returns (stream TrainingUpdate);
  rpc DownloadWeights(DownloadWeightsRequest) returns (stream WeightsChunk);
}
```

**4. ComputeService** (Direct compute operations)
```protobuf
service ComputeService {
  rpc Execute(ExecuteRequest) returns (ExecuteResponse);
  rpc ExecuteGraph(ExecuteGraphRequest) returns (ExecuteGraphResponse);
  rpc StreamExecute(stream ExecuteRequest) returns (stream ExecuteResponse);
}
```

#### C++ Implementation (Server Node)

**Node Client** (`cyxwiz-server-node/src/node_client.h`):
- `Register()` - Register with Central Server
- `StartHeartbeat()` / `StopHeartbeat()` - Background heartbeat thread
- `UpdateJobStatus()` - Report progress with metrics
- `ReportJobResult()` - Report final results

**Node Service** (`cyxwiz-server-node/src/node_service.h`):
- `AssignJob()` - Receives job assignments from Central Server
- `GetNodeMetrics()` - Provides node metrics
- `ValidateJobConfig()` - Pre-execution validation

### Why Both Protocols?

**REST API Strengths:**
1. **Simplicity** - Easy to debug, test with curl, log
2. **Stateless** - No connection state to manage
3. **Firewall-friendly** - Works through proxies/CDNs
4. **Browser-compatible** - Website can call directly

**gRPC Strengths:**
1. **Server Push** - Server initiates task assignments
2. **Streaming** - Efficient for large data (model weights, results)
3. **Low Latency** - HTTP/2 multiplexing, binary protocol
4. **Bidirectional** - Real-time control and monitoring
5. **Typed Contracts** - Proto definitions ensure compatibility

### Integration Pattern

```rust
// Central server with both REST and gRPC

#[tokio::main]
async fn main() {
    let db = init_mongodb().await;

    // Shared state for both servers
    let state = Arc::new(AppState {
        db: db.clone(),
        active_nodes: RwLock::new(HashMap::new()),
    });

    // REST API on port 8080
    let rest_app = Router::new()
        .nest("/api/auth", auth_routes())
        .nest("/api/nodes", node_routes())
        .nest("/api/compute-units", compute_unit_routes())
        .with_state(state.clone());

    // gRPC server on port 50051
    let grpc_service = NodeServiceImpl::new(state.clone());

    // Run both concurrently
    tokio::select! {
        _ = axum::serve(rest_listener, rest_app) => {},
        _ = tonic::transport::Server::builder()
            .add_service(NodeServiceServer::new(grpc_service))
            .serve(grpc_addr) => {},
    }
}
```

### C++ Client Integration (Actual Implementation)

The Server Node GUI uses both protocols through these existing classes:

**REST API Client** (`auth/auth_manager.h`):
```cpp
// AuthManager handles REST API calls for authentication
class AuthManager {
    // REST API calls
    std::future<AuthResult> LoginWithEmail(const std::string& email, const std::string& password);
    std::future<NodeRegistrationResult> RegisterNodeWithApi(const std::string& node_name, ...);
    bool SendHeartbeatToApi();  // POST /api/nodes/:id/heartbeat

    // Tokens
    std::string GetJwtToken() const;
    std::string GetNodeToken() const;
};
```

**gRPC Client** (`node_client.h`):
```cpp
// NodeClient handles gRPC communication with Central Server
class NodeClient {
    // gRPC calls
    bool Register(const NodeInfo& info);
    void StartHeartbeat();
    void StopHeartbeat();
    bool UpdateJobStatus(const std::string& job_id, JobStatus status, float progress);
    bool ReportJobResult(const std::string& job_id, const JobResult& result);
};
```

**Integration Flow**:
```cpp
// 1. Login via REST API (get JWT)
AuthManager::Instance().LoginWithEmail(email, password);

// 2. Register node via REST API (get node_id, api_key)
AuthManager::Instance().RegisterNodeWithApi("my-server-node");

// 3. Connect via gRPC for job execution
NodeClient client(grpc_server_address);
client.Register(hardware_info);
client.StartHeartbeat();

// 4. REST heartbeats run in parallel for website status
// (AuthManager sends heartbeats to REST API every 15s)

// 5. gRPC receives job assignments
// NodeService::AssignJob() callback triggered when job arrives
```

### Summary: REST vs gRPC Responsibilities

| Operation | Protocol | Implementation | Purpose |
|-----------|----------|----------------|---------|
| **User Login** | REST | `AuthManager::LoginWithEmail()` | JWT token for dashboard access |
| **Node Registration (Dashboard)** | REST | `AuthManager::RegisterNodeWithApi()` | Website shows device in dashboard |
| **Heartbeat (Dashboard)** | REST | `AuthManager::SendHeartbeatToApi()` | Website status (online/offline) |
| **Node Registration (Jobs)** | gRPC | `NodeClient::Register()` | Central Server job assignment |
| **Heartbeat (Jobs)** | gRPC | `NodeClient::StartHeartbeat()` | Keep gRPC connection alive |
| **Job Assignment** | gRPC | `NodeService::AssignJob()` | Server pushes job to node |
| **Progress Reporting** | gRPC | `NodeClient::UpdateJobStatus()` | Real-time job progress |
| **Job Results** | gRPC | `NodeClient::ReportJobResult()` | Send final results |
| **Model Weights** | gRPC | `JobExecutionService::DownloadWeights()` | Stream large model files |
| **Training Metrics** | gRPC | `JobExecutionService::StreamTrainingMetrics()` | Bidirectional streaming |

### Key Integration Points

1. **Dual Registration**: Node registers via both REST (for website dashboard) and gRPC (for job execution)
2. **Dual Heartbeat**: REST heartbeats update website status; gRPC heartbeats keep job connection alive
3. **Shared Credentials**: Both use the same `node_id` and authentication tokens
4. **MongoDB Sync**: Both protocols write to the same MongoDB database

---

## Complete Connection Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                USER'S MACHINE                                        │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                        Server Node GUI Application                            │   │
│  │                                                                               │   │
│  │   ┌─────────────┐    ┌─────────────────┐    ┌──────────────────────────┐    │   │
│  │   │   Login     │───>│ Hardware Detect │───>│   Resource Allocation    │    │   │
│  │   │   Panel     │    │     Panel       │    │        Panel             │    │   │
│  │   └─────────────┘    └─────────────────┘    └──────────────────────────┘    │   │
│  │                              │                          │                     │   │
│  │                              v                          v                     │   │
│  │                    ┌─────────────────┐        ┌─────────────────────┐        │   │
│  │                    │ CPU: AMD 5800X  │        │ Register Selected   │        │   │
│  │                    │ GPU0: RTX 4090  │───────>│ Devices to Server   │        │   │
│  │                    │ GPU1: RTX 3080  │        └─────────────────────┘        │   │
│  │                    └─────────────────┘                  │                     │   │
│  └─────────────────────────────────────────────────────────┼─────────────────────┘   │
└────────────────────────────────────────────────────────────┼─────────────────────────┘
                                                             │
                                                             v
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              CYXWIZ CENTRAL API                                      │
│                                                                                      │
│   POST /api/nodes/register                                                           │
│   {                                                                                  │
│     "user_id": "...",                                                               │
│     "machine_id": "unique-machine-id",                                              │
│     "compute_units": [                                                              │
│       { "type": "gpu", "device_id": 0, "name": "RTX 4090", "allocated_vram": 22GB },│
│       { "type": "cpu", "cores_allocated": 12, "name": "AMD 5800X" }                 │
│     ]                                                                               │
│   }                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                                             │
                                                             v
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                WEBSITE DASHBOARD                                     │
│                                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────┐      │
│   │  My Devices                                                              │      │
│   │  ───────────────────────────────────────────────────────────────────────│      │
│   │                                                                          │      │
│   │  [MACHINE: Gaming-PC]                                                    │      │
│   │  ├── CPU: AMD Ryzen 7 5800X                                             │      │
│   │  │   └── Status: INACTIVE (not shared)                                  │      │
│   │  │                                                                       │      │
│   │  ├── GPU 0: NVIDIA RTX 4090 (24GB)                                      │      │
│   │  │   └── Status: ACTIVE ● Sharing 22GB | Earnings: 1,234 CYXWIZ         │      │
│   │  │                                                                       │      │
│   │  └── GPU 1: NVIDIA RTX 3080 (10GB)                                      │      │
│   │      └── Status: ACTIVE ● Sharing 8GB | Earnings: 567 CYXWIZ            │      │
│   │                                                                          │      │
│   └─────────────────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Data Models

### Machine (Physical Computer)

```typescript
interface Machine {
  machine_id: string;         // Unique identifier for the physical machine
  owner_id: string;           // User who owns this machine
  hostname: string;           // Computer name
  os: string;                 // "Windows 11", "Ubuntu 22.04", etc.
  ip_address: string;         // Current IP
  last_seen: DateTime;        // Last heartbeat
  status: "online" | "offline";

  // All detected hardware (whether shared or not)
  hardware: {
    cpus: CpuInfo[];
    gpus: GpuInfo[];
    ram_total_gb: number;
    storage_total_gb: number;
  };
}

interface CpuInfo {
  device_id: number;          // 0, 1, etc. for multi-CPU systems
  name: string;               // "AMD Ryzen 7 5800X"
  cores: number;              // 8
  threads: number;            // 16
  base_clock_mhz: number;     // 3800
  boost_clock_mhz: number;    // 4700
}

interface GpuInfo {
  device_id: number;          // 0, 1, 2, etc.
  name: string;               // "NVIDIA GeForce RTX 4090"
  vendor: "nvidia" | "amd" | "intel";
  vram_mb: number;            // 24576 (24GB)
  driver_version: string;     // "546.33"
  compute_capability?: string; // "8.9" for CUDA
  is_integrated: boolean;     // false for discrete GPUs
}
```

### Compute Unit (Shared Resource)

```typescript
interface ComputeUnit {
  unit_id: string;            // Unique ID for this compute unit
  machine_id: string;         // Parent machine
  owner_id: string;           // User

  // Device identification
  device_type: "cpu" | "gpu";
  device_id: number;          // Which CPU/GPU on the machine
  device_name: string;        // "RTX 4090"

  // Resource allocation
  allocation: {
    // For GPU
    vram_allocated_mb?: number;     // 22528 (22GB of 24GB)
    vram_reserved_mb?: number;      // 2048 (2GB kept for system)

    // For CPU
    cores_allocated?: number;       // 12 of 16 cores
    cores_reserved?: number;        // 4 cores kept for system

    // Common
    priority: "low" | "normal" | "high";  // Task priority
    max_power_watts?: number;       // Power limit
  };

  // Status
  status: "active" | "paused" | "offline" | "error";
  current_task_id?: string;
  current_load_percent: number;

  // Stats
  stats: {
    uptime_seconds: number;
    tasks_completed: number;
    total_earnings: number;
    earnings_today: number;
  };

  // Timestamps
  registered_at: DateTime;
  last_heartbeat: DateTime;
}
```

## GUI Implementation

### 1. Hardware Detection Panel

Located in the Server Node GUI, this panel shows all available compute resources.

```cpp
// hardware_panel.h
namespace cyxwiz::servernode::gui {

struct DetectedCpu {
    int device_id;
    std::string name;
    int cores;
    int threads;
    int base_clock_mhz;
    int boost_clock_mhz;
};

struct DetectedGpu {
    int device_id;
    std::string name;
    std::string vendor;           // "NVIDIA", "AMD", "Intel"
    size_t vram_total_mb;
    size_t vram_available_mb;     // Currently free VRAM
    std::string driver_version;
    std::string compute_capability;
    bool is_integrated;
};

struct DetectedHardware {
    std::vector<DetectedCpu> cpus;
    std::vector<DetectedGpu> gpus;
    size_t ram_total_mb;
    size_t ram_available_mb;
};

class HardwarePanel : public ServerPanel {
public:
    void Render() override;
    void RefreshHardware();

    const DetectedHardware& GetHardware() const { return hardware_; }

private:
    void RenderCpuSection();
    void RenderGpuSection();
    void RenderMemorySection();

    DetectedHardware hardware_;
    bool is_scanning_ = false;
};

} // namespace
```

### 2. Resource Allocation Panel

User interface for selecting which resources to share.

```cpp
// allocation_panel.h
namespace cyxwiz::servernode::gui {

struct ResourceAllocation {
    // Device identification
    enum class DeviceType { Cpu, Gpu };
    DeviceType device_type;
    int device_id;
    std::string device_name;

    // Allocation settings
    bool is_enabled = false;          // Whether to share this device

    // GPU-specific
    size_t vram_allocated_mb = 0;     // VRAM to share
    size_t vram_reserved_mb = 2048;   // Keep 2GB for system by default

    // CPU-specific
    int cores_allocated = 0;
    int cores_reserved = 2;           // Keep 2 cores for system

    // Common settings
    int priority = 1;                 // 0=low, 1=normal, 2=high
    int max_power_percent = 100;      // Power limit percentage
    bool allow_overnight_only = false; // Only run when idle
};

class AllocationPanel : public ServerPanel {
public:
    AllocationPanel(HardwarePanel* hardware_panel);

    void Render() override;
    void Update() override;

    // Get current allocations
    std::vector<ResourceAllocation> GetAllocations() const;

    // Apply allocations (register with server)
    void ApplyAllocations();

private:
    void RenderDeviceCard(ResourceAllocation& alloc, const DetectedGpu& gpu);
    void RenderDeviceCard(ResourceAllocation& alloc, const DetectedCpu& cpu);
    void RenderVramSlider(ResourceAllocation& alloc, size_t max_vram);
    void RenderCoreSlider(ResourceAllocation& alloc, int max_cores);
    void RenderCommonSettings(ResourceAllocation& alloc);

    HardwarePanel* hardware_panel_;
    std::vector<ResourceAllocation> allocations_;

    // Registration state
    bool is_registering_ = false;
    std::string registration_error_;
    std::future<bool> registration_future_;
};

} // namespace
```

### 3. GUI Layout

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  CyxWiz Server Node                                              [─] [□] [×]   │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐                                                               │
│  │ User: John   │  Logged in as john@example.com                                │
│  │ ● Connected  │  Machine: GAMING-PC | IP: 192.168.1.100                       │
│  └──────────────┘                                                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  COMPUTE RESOURCES                                            [Refresh] [Apply]│
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  [□] CPU: AMD Ryzen 7 5800X (8 cores / 16 threads)                      │   │
│  │      ├── Cores to share: [====────────] 4 / 8 cores                     │   │
│  │      ├── Priority: [Low] [Normal] [High]                                │   │
│  │      └── Status: Not Shared                                             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  [✓] GPU 0: NVIDIA GeForce RTX 4090 (24 GB VRAM)           ● ACTIVE     │   │
│  │      ├── VRAM to share: [================────] 22 GB / 24 GB            │   │
│  │      │   (Keeping 2 GB for Windows/display)                             │   │
│  │      ├── Priority: [Low] [●Normal] [High]                               │   │
│  │      ├── Power Limit: [==============────] 80%                          │   │
│  │      ├── Current Load: 45% | Temp: 62°C                                 │   │
│  │      └── Earnings Today: 23.5 CYXWIZ | Total: 1,234.5 CYXWIZ            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  [✓] GPU 1: NVIDIA GeForce RTX 3080 (10 GB VRAM)           ● ACTIVE     │   │
│  │      ├── VRAM to share: [============────────] 8 GB / 10 GB             │   │
│  │      │   (Keeping 2 GB for system)                                      │   │
│  │      ├── Priority: [●Low] [Normal] [High]                               │   │
│  │      ├── Power Limit: [================────] 100%                       │   │
│  │      ├── Current Load: 78% | Temp: 71°C                                 │   │
│  │      └── Earnings Today: 12.3 CYXWIZ | Total: 567.8 CYXWIZ              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ACTIVITY LOG                                                                   │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  [07:15:32] GPU 0: Started task ml_inference_job_12345                          │
│  [07:15:30] GPU 1: Completed task training_batch_9876 (+2.3 CYXWIZ)             │
│  [07:14:45] Heartbeat sent successfully                                         │
│  [07:12:00] GPU 1: Started task training_batch_9876                             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## API Endpoints

### Complete API Reference

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/nodes/register` | POST | JWT | Register machine with compute units |
| `/api/nodes/:machine_id/allocations` | PUT | JWT | Update resource allocations |
| `/api/nodes/:node_id/heartbeat` | POST | API Key | Node heartbeat with compute unit status |
| `/api/nodes/:node_id` | GET | JWT | Get node details |
| `/api/nodes/:node_id` | DELETE | JWT | Unregister node |
| `/api/users/:user_id/devices` | GET | JWT | Get user's machines and devices |
| `/api/users/:user_id/devices/summary` | GET | JWT | Get earnings summary |
| `/api/compute-units/:unit_id` | GET | JWT | Get compute unit details |
| `/api/compute-units/:unit_id/pause` | POST | API Key | Pause compute unit |
| `/api/compute-units/:unit_id/resume` | POST | API Key | Resume compute unit |

---

### Register Machine with Compute Units

```
POST /api/nodes/register
Authorization: Bearer <jwt_token>

Request:
{
  "machine_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "hostname": "GAMING-PC",
  "os": "Windows 11 Pro",
  "ip_address": "192.168.1.100",
  "hardware": {
    "cpus": [{
      "device_id": 0,
      "name": "AMD Ryzen 7 5800X",
      "cores": 8,
      "threads": 16,
      "base_clock_mhz": 3800,
      "boost_clock_mhz": 4700
    }],
    "gpus": [{
      "device_id": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "vendor": "nvidia",
      "vram_mb": 24576,
      "driver_version": "546.33",
      "compute_capability": "8.9"
    }, {
      "device_id": 1,
      "name": "NVIDIA GeForce RTX 3080",
      "vendor": "nvidia",
      "vram_mb": 10240,
      "driver_version": "546.33",
      "compute_capability": "8.6"
    }],
    "ram_total_gb": 64,
    "storage_total_gb": 2000
  },
  "compute_units": [{
    "device_type": "gpu",
    "device_id": 0,
    "allocation": {
      "vram_allocated_mb": 22528,
      "vram_reserved_mb": 2048,
      "priority": "normal",
      "max_power_percent": 80
    }
  }, {
    "device_type": "gpu",
    "device_id": 1,
    "allocation": {
      "vram_allocated_mb": 8192,
      "vram_reserved_mb": 2048,
      "priority": "low",
      "max_power_percent": 100
    }
  }]
}

Response:
{
  "success": true,
  "machine_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "api_key": "node_api_key_here",
  "compute_units": [{
    "unit_id": "unit_gpu0_xyz123",
    "device_type": "gpu",
    "device_id": 0,
    "status": "active"
  }, {
    "unit_id": "unit_gpu1_abc456",
    "device_type": "gpu",
    "device_id": 1,
    "status": "active"
  }]
}
```

### Update Allocations

```
PUT /api/nodes/:machine_id/allocations
Authorization: Bearer <jwt_token>

Request:
{
  "compute_units": [{
    "unit_id": "unit_gpu0_xyz123",
    "allocation": {
      "vram_allocated_mb": 20480,  // Changed from 22GB to 20GB
      "priority": "high"           // Changed priority
    }
  }]
}

Response:
{
  "success": true,
  "updated_units": ["unit_gpu0_xyz123"]
}
```

### Heartbeat with Compute Unit Status

```
POST /api/nodes/:node_id/heartbeat
Authorization: Bearer <api_key>

Request:
{
  "timestamp": "2025-12-09T07:30:00Z",
  "compute_units": [{
    "unit_id": "unit_gpu0_xyz123",
    "status": "active",
    "current_load_percent": 45,
    "current_task_id": "task_12345",
    "temperature_celsius": 62,
    "power_draw_watts": 280,
    "vram_used_mb": 18432
  }, {
    "unit_id": "unit_gpu1_abc456",
    "status": "active",
    "current_load_percent": 78,
    "current_task_id": "task_67890",
    "temperature_celsius": 71,
    "power_draw_watts": 220,
    "vram_used_mb": 7680
  }]
}

Response:
{
  "success": true,
  "commands": []  // Any pending commands from server
}
```

### Get User's Devices (Website)

```
GET /api/users/:user_id/devices
Authorization: Bearer <jwt_token>

Response:
{
  "machines": [{
    "machine_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "hostname": "GAMING-PC",
    "os": "Windows 11 Pro",
    "status": "online",
    "last_seen": "2025-12-09T07:30:00Z",
    "hardware": {
      "cpus": [{
        "device_id": 0,
        "name": "AMD Ryzen 7 5800X",
        "cores": 8,
        "is_shared": false
      }],
      "gpus": [{
        "device_id": 0,
        "name": "NVIDIA GeForce RTX 4090",
        "vram_mb": 24576,
        "is_shared": true,
        "compute_unit": {
          "unit_id": "unit_gpu0_xyz123",
          "status": "active",
          "vram_allocated_mb": 22528,
          "current_load_percent": 45,
          "earnings_today": 23.5,
          "total_earnings": 1234.5
        }
      }, {
        "device_id": 1,
        "name": "NVIDIA GeForce RTX 3080",
        "vram_mb": 10240,
        "is_shared": true,
        "compute_unit": {
          "unit_id": "unit_gpu1_abc456",
          "status": "active",
          "vram_allocated_mb": 8192,
          "current_load_percent": 78,
          "earnings_today": 12.3,
          "total_earnings": 567.8
        }
      }]
    }
  }],
  "summary": {
    "total_machines": 1,
    "online_machines": 1,
    "active_gpus": 2,
    "active_cpus": 0,
    "total_earnings": 1802.3,
    "earnings_today": 35.8
  }
}
```

## Website Dashboard Updates

### Enhanced Device Card Component

```tsx
// devices-list.tsx

interface ComputeDevice {
  device_id: number;
  name: string;
  type: "cpu" | "gpu";

  // Hardware specs
  cores?: number;           // CPU
  vram_mb?: number;         // GPU

  // Sharing status
  is_shared: boolean;
  compute_unit?: {
    unit_id: string;
    status: "active" | "paused" | "offline" | "error";

    // Allocation
    vram_allocated_mb?: number;
    cores_allocated?: number;

    // Current state
    current_load_percent: number;
    current_task_id?: string;
    temperature_celsius?: number;

    // Earnings
    earnings_today: number;
    total_earnings: number;
  };
}

interface Machine {
  machine_id: string;
  hostname: string;
  os: string;
  status: "online" | "offline";
  last_seen: string;
  devices: ComputeDevice[];
}

function DeviceCard({ device, machineOnline }: { device: ComputeDevice; machineOnline: boolean }) {
  const isActive = device.is_shared && device.compute_unit?.status === "active" && machineOnline;

  return (
    <div className={`p-4 rounded-lg border ${isActive ? "border-green-500 bg-green-50" : "border-gray-200"}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {device.type === "gpu" ? <GpuIcon /> : <CpuIcon />}
          <div>
            <p className="font-medium">{device.name}</p>
            <p className="text-sm text-muted-foreground">
              {device.type === "gpu"
                ? `${(device.vram_mb! / 1024).toFixed(0)} GB VRAM`
                : `${device.cores} cores`
              }
            </p>
          </div>
        </div>

        <Badge variant={isActive ? "success" : device.is_shared ? "warning" : "secondary"}>
          {isActive ? "Active" : device.is_shared ? "Paused" : "Not Shared"}
        </Badge>
      </div>

      {device.is_shared && device.compute_unit && (
        <div className="mt-4 space-y-2">
          {/* Resource allocation bar */}
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span>Allocated</span>
              <span>
                {device.type === "gpu"
                  ? `${(device.compute_unit.vram_allocated_mb! / 1024).toFixed(0)} GB`
                  : `${device.compute_unit.cores_allocated} cores`
                }
              </span>
            </div>
            <Progress
              value={device.type === "gpu"
                ? (device.compute_unit.vram_allocated_mb! / device.vram_mb!) * 100
                : (device.compute_unit.cores_allocated! / device.cores!) * 100
              }
            />
          </div>

          {/* Current load */}
          {isActive && (
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span>Current Load</span>
                <span>{device.compute_unit.current_load_percent}%</span>
              </div>
              <Progress
                value={device.compute_unit.current_load_percent}
                className={device.compute_unit.current_load_percent > 80 ? "bg-yellow-500" : ""}
              />
            </div>
          )}

          {/* Earnings */}
          <div className="flex justify-between text-sm pt-2 border-t">
            <span>Today: {device.compute_unit.earnings_today.toFixed(2)} CYXWIZ</span>
            <span>Total: {device.compute_unit.total_earnings.toFixed(2)} CYXWIZ</span>
          </div>
        </div>
      )}
    </div>
  );
}

function MachineCard({ machine }: { machine: Machine }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <ComputerIcon />
            <div>
              <CardTitle>{machine.hostname}</CardTitle>
              <CardDescription>{machine.os}</CardDescription>
            </div>
          </div>
          <Badge variant={machine.status === "online" ? "success" : "secondary"}>
            {machine.status === "online" ? "Online" : "Offline"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {machine.devices.map(device => (
            <DeviceCard
              key={`${device.type}-${device.device_id}`}
              device={device}
              machineOnline={machine.status === "online"}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
```

---

### Pause Compute Unit

```
POST /api/compute-units/:unit_id/pause
Authorization: Bearer <api_key>

Request:
{
  "reason": "user_requested"  // Optional: "user_requested", "maintenance", "overheating"
}

Response:
{
  "success": true,
  "unit_id": "unit_gpu0_xyz123",
  "status": "paused",
  "paused_at": "2025-12-09T08:00:00Z"
}
```

---

### Resume Compute Unit

```
POST /api/compute-units/:unit_id/resume
Authorization: Bearer <api_key>

Response:
{
  "success": true,
  "unit_id": "unit_gpu0_xyz123",
  "status": "active",
  "resumed_at": "2025-12-09T08:30:00Z"
}
```

---

### Get Compute Unit Details

```
GET /api/compute-units/:unit_id
Authorization: Bearer <jwt_token>

Response:
{
  "unit_id": "unit_gpu0_xyz123",
  "machine_id": "a1b2c3d4-...",
  "owner_id": "user_123",
  "device_type": "gpu",
  "device_id": 0,
  "device_name": "NVIDIA GeForce RTX 4090",
  "allocation": {
    "vram_allocated_mb": 22528,
    "vram_reserved_mb": 2048,
    "priority": "normal",
    "max_power_percent": 80
  },
  "status": "active",
  "current_task_id": "task_12345",
  "current_load_percent": 45,
  "stats": {
    "uptime_seconds": 86400,
    "tasks_completed": 127,
    "total_earnings": 1234.56,
    "earnings_today": 23.45
  },
  "registered_at": "2025-12-01T00:00:00Z",
  "last_heartbeat": "2025-12-09T07:30:00Z"
}
```

---

### Get User Devices Summary

```
GET /api/users/:user_id/devices/summary
Authorization: Bearer <jwt_token>

Response:
{
  "user_id": "user_123",
  "summary": {
    "total_machines": 2,
    "online_machines": 1,
    "offline_machines": 1,
    "total_compute_units": 5,
    "active_compute_units": 3,
    "paused_compute_units": 1,
    "offline_compute_units": 1,
    "total_vram_shared_gb": 40,
    "total_cores_shared": 8,
    "total_earnings": 5678.90,
    "earnings_today": 45.67,
    "earnings_this_week": 312.45,
    "earnings_this_month": 1234.56
  },
  "by_device_type": {
    "gpu": {
      "count": 4,
      "active": 3,
      "total_vram_gb": 68,
      "shared_vram_gb": 40,
      "total_earnings": 5123.45
    },
    "cpu": {
      "count": 1,
      "active": 0,
      "total_cores": 16,
      "shared_cores": 0,
      "total_earnings": 555.45
    }
  }
}
```

---

### Unregister Machine

```
DELETE /api/nodes/:machine_id
Authorization: Bearer <jwt_token>

Response:
{
  "success": true,
  "machine_id": "a1b2c3d4-...",
  "compute_units_removed": 2,
  "message": "Machine and all compute units unregistered"
}
```

---

### Error Responses

All endpoints return errors in this format:

```json
{
  "error": "Error message describing what went wrong",
  "code": "ERROR_CODE",           // Optional error code
  "details": {}                   // Optional additional details
}
```

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 400 | `INVALID_REQUEST` | Request body validation failed |
| 401 | `UNAUTHORIZED` | Invalid or missing auth token |
| 403 | `FORBIDDEN` | Not allowed to access this resource |
| 404 | `NOT_FOUND` | Resource not found |
| 409 | `CONFLICT` | Resource already exists or state conflict |
| 422 | `INVALID_ALLOCATION` | Resource allocation exceeds hardware limits |
| 429 | `RATE_LIMITED` | Too many requests |
| 500 | `SERVER_ERROR` | Internal server error |

---

## Hardware Detection Implementation

### Windows (NVML + WMI)

```cpp
// hardware_detector_win.cpp
#include <nvml.h>
#include <Windows.h>
#include <wbemidl.h>

namespace cyxwiz::servernode::hardware {

DetectedHardware DetectHardware() {
    DetectedHardware result;

    // Detect GPUs using NVML
    nvmlInit();
    unsigned int device_count;
    nvmlDeviceGetCount(&device_count);

    for (unsigned int i = 0; i < device_count; i++) {
        nvmlDevice_t device;
        nvmlDeviceGetHandleByIndex(i, &device);

        DetectedGpu gpu;
        gpu.device_id = i;

        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
        nvmlDeviceGetName(device, name, sizeof(name));
        gpu.name = name;
        gpu.vendor = "nvidia";

        nvmlMemory_t memory;
        nvmlDeviceGetMemoryInfo(device, &memory);
        gpu.vram_total_mb = memory.total / (1024 * 1024);
        gpu.vram_available_mb = memory.free / (1024 * 1024);

        int major, minor;
        nvmlDeviceGetCudaComputeCapability(device, &major, &minor);
        gpu.compute_capability = std::to_string(major) + "." + std::to_string(minor);

        gpu.is_integrated = false;

        result.gpus.push_back(gpu);
    }
    nvmlShutdown();

    // Detect CPUs using WMI
    // ... WMI code for CPU detection

    // Detect RAM
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    GlobalMemoryStatusEx(&statex);
    result.ram_total_mb = statex.ullTotalPhys / (1024 * 1024);
    result.ram_available_mb = statex.ullAvailPhys / (1024 * 1024);

    return result;
}

} // namespace
```

### AMD GPU Detection

```cpp
// For AMD GPUs, use AMD GPU Services (AGS) or ROCm SMI
#ifdef AMD_GPU_SUPPORT
#include <amd_ags.h>

void DetectAmdGpus(DetectedHardware& result) {
    AGSContext* ags_context = nullptr;
    AGSGPUInfo gpu_info;

    if (agsInit(&ags_context, nullptr, &gpu_info) == AGS_SUCCESS) {
        for (int i = 0; i < gpu_info.numDevices; i++) {
            DetectedGpu gpu;
            gpu.device_id = result.gpus.size();
            gpu.name = gpu_info.devices[i].adapterString;
            gpu.vendor = "amd";
            gpu.vram_total_mb = gpu_info.devices[i].localMemoryInBytes / (1024 * 1024);
            gpu.is_integrated = gpu_info.devices[i].isAPU;

            result.gpus.push_back(gpu);
        }
        agsDeInit(ags_context);
    }
}
#endif
```

## State Transitions

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           COMPUTE UNIT STATE MACHINE                             │
└─────────────────────────────────────────────────────────────────────────────────┘

                          User enables sharing
           ┌─────────────────────────────────────────────┐
           │                                             │
           v                                             │
    ┌──────────────┐                             ┌───────┴──────┐
    │   INACTIVE   │                             │  NOT_SHARED  │
    │  (detected   │<────────────────────────────│  (hardware   │
    │   but not    │   User disables sharing     │   detected)  │
    │   shared)    │                             └──────────────┘
    └──────┬───────┘
           │ Registration API call
           v
    ┌──────────────┐
    │  REGISTERING │
    │  (API call   │
    │   in flight) │
    └──────┬───────┘
           │ Success
           v
    ┌──────────────┐     Heartbeat timeout      ┌──────────────┐
    │    ACTIVE    │────────────────────────────>│   OFFLINE    │
    │  (accepting  │                             │  (no comms)  │
    │    tasks)    │<────────────────────────────│              │
    └──────┬───────┘     Heartbeat received      └──────────────┘
           │
           │ User pauses or
           │ system override
           v
    ┌──────────────┐
    │    PAUSED    │
    │  (temporary  │
    │    stop)     │
    └──────────────┘
```

## Earnings Tracking

Each compute unit tracks earnings independently:

```typescript
interface EarningsRecord {
  unit_id: string;
  date: string;              // "2025-12-09"

  // Task-based earnings
  tasks_completed: number;
  task_earnings: number;     // CYXWIZ earned from tasks

  // Time-based earnings (availability rewards)
  uptime_seconds: number;
  uptime_earnings: number;

  // Total
  total_earnings: number;
}

// Earnings formula (example)
// GPU earnings = (VRAM_allocated_GB * hours_active * rate) + (tasks_completed * task_reward)
// CPU earnings = (cores_allocated * hours_active * rate) + (tasks_completed * task_reward)
```

## Security Considerations

1. **Machine ID**: Generated from hardware fingerprint (CPU ID, MAC address hash)
2. **Resource Limits**: Server validates allocations don't exceed hardware caps
3. **Abuse Prevention**: Rate limiting on registration, cooldown on allocation changes
4. **Verification**: Optional proof-of-work to verify hardware claims

## Implementation Phases

### Phase 1: Hardware Detection
- [ ] Implement NVML-based GPU detection
- [ ] Implement WMI-based CPU detection
- [ ] Add AMD GPU support (AGS/ROCm)
- [ ] Create HardwarePanel GUI

### Phase 2: Resource Allocation
- [ ] Create AllocationPanel GUI
- [ ] Implement VRAM/Core sliders
- [ ] Add priority and power limit settings
- [ ] Save/load allocation preferences

### Phase 3: Registration
- [ ] Update API to accept compute_units array
- [ ] Create machine and compute_unit database models
- [ ] Implement per-unit heartbeat tracking
- [ ] Update node registration flow

### Phase 4: Website Dashboard
- [ ] Create machine/device hierarchy display
- [ ] Show per-device earnings
- [ ] Add shared vs not-shared indicators
- [ ] Real-time status updates

### Phase 5: Earnings & Tasks
- [ ] Implement per-unit earnings tracking
- [ ] Task assignment respects resource allocation
- [ ] Power/priority settings affect task scheduling
