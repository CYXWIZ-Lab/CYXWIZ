# System Architecture

This document provides a comprehensive overview of CyxWiz's system architecture, component relationships, and data flows.

## High-Level Architecture

```
                                    CYXWIZ PLATFORM ARCHITECTURE

    +------------------+                                        +------------------+
    |                  |                                        |                  |
    |  CyxWiz Engine   |        +--------------------+          |  Server Node 1   |
    |  (Desktop GUI)   |<------>|                    |<-------->|  (GPU Worker)    |
    |                  |  gRPC  |  CyxWiz Central    |  gRPC    |                  |
    +------------------+        |     Server         |          +------------------+
            |                   |                    |                   |
            |                   | (Rust/Tokio)       |                   |
            v                   |                    |                   v
    +------------------+        +--------------------+          +------------------+
    |                  |               |  |  |                  |                  |
    | cyxwiz-backend   |               |  |  |                  |  Server Node 2   |
    |   (Shared DLL)   |               |  |  |                  |  (CPU Worker)    |
    |                  |               |  |  |                  |                  |
    +------------------+               |  |  |                  +------------------+
                                       |  |  |
                          +------------+  |  +------------+
                          |               |               |
                          v               v               v
                   +-----------+   +-----------+   +-----------+
                   |PostgreSQL |   |   Redis   |   |  Solana   |
                   |  SQLite   |   |   Cache   |   |Blockchain |
                   +-----------+   +-----------+   +-----------+
```

## Component Architecture

### CyxWiz Engine Architecture

```
cyxwiz-engine/
+-------------------+
|   Application     |  <- Main entry point, GLFW window management
+-------------------+
         |
         v
+-------------------+
|   Main Window     |  <- Dockable window layout, panel management
+-------------------+
         |
    +----+----+----+----+----+----+----+----+
    |    |    |    |    |    |    |    |    |
    v    v    v    v    v    v    v    v    v
+------+------+------+------+------+------+------+------+------+
|Node  |Script|Console|Props |Asset |View- |Train |Dataset|Tool |
|Editor|Editor|       |Panel |Brows.|port  |Dashbd|Panel  |Panels|
+------+------+------+------+------+------+------+------+------+
    |                    |
    v                    v
+-------------------+  +-------------------+
|  Code Generation  |  |  Training         |
|  (PyTorch/TF/etc) |  |  Executor         |
+-------------------+  +-------------------+
                              |
                              v
                       +-------------------+
                       |  cyxwiz-backend   |
                       |  (DLL/SO)         |
                       +-------------------+
```

**Key Components:**

| Component | File(s) | Purpose |
|-----------|---------|---------|
| Application | `application.cpp/h` | Window creation, ImGui init, main loop |
| MainWindow | `main_window.cpp/h` | Docking layout, panel coordination |
| NodeEditor | `node_editor.cpp/h` | Visual ML pipeline builder |
| ScriptEditor | `script_editor.cpp/h` | Python/CyxWiz code editing |
| Console | `console.cpp/h` | Python REPL, log output |
| Properties | `properties.cpp/h` | Node parameter editing |
| TrainingExecutor | `training_executor.cpp/h` | Local model training |

### Central Server Architecture

```
cyxwiz-central-server/
+-------------------+
|      main.rs      |  <- Entry point, service initialization
+-------------------+
         |
    +----+----+----+----+
    |    |    |    |    |
    v    v    v    v    v
+------+------+------+------+------+
| API  |Sched-|Data- |Cache |Block-|
|Module|uler  |base  |Module|chain |
+------+------+------+------+------+
    |      |      |      |      |
    v      v      v      v      v
+------+------+------+------+------+
|gRPC  |Job   |SQLx  |Redis |Solana|
|Servs |Queue |Pool  |Client|Client|
+------+------+------+------+------+
```

**Rust Modules:**

| Module | Path | Purpose |
|--------|------|---------|
| `api::grpc` | `src/api/grpc/` | gRPC service implementations |
| `api::rest` | `src/api/rest/` | REST API endpoints |
| `scheduler` | `src/scheduler/` | Job queue, node matcher |
| `database` | `src/database/` | Models, queries, migrations |
| `cache` | `src/cache/` | Redis integration |
| `blockchain` | `src/blockchain/` | Solana client, escrow |
| `auth` | `src/auth/` | JWT token management |
| `tui` | `src/tui/` | Terminal UI dashboard |

### Server Node Architecture

```
cyxwiz-server-node/
+-------------------+
|     main.cpp      |  <- Entry point, mode selection
+-------------------+
         |
    +----+----+
    |         |
    v         v
+--------+ +--------+
|  GUI   | | Daemon |
|  Mode  | |  Mode  |
+--------+ +--------+
    |         |
    v         v
+-------------------+
|   Core Services   |
|                   |
| +---------------+ |
| | Job Executor  | |
| +---------------+ |
| | Node Client   | |
| +---------------+ |
| | Metrics       | |
| | Collector     | |
| +---------------+ |
| | HTTP/API      | |
| | Server        | |
| +---------------+ |
+-------------------+
         |
         v
+-------------------+
|  cyxwiz-backend   |
|  (ArrayFire)      |
+-------------------+
```

**Key Components:**

| Component | File(s) | Purpose |
|-----------|---------|---------|
| ServerApplication | `server_application.cpp/h` | GUI mode main class |
| DaemonService | `daemon_service.cpp/h` | Headless service mode |
| JobExecutor | `job_executor.cpp/h` | Training job execution |
| NodeClient | `node_client.cpp/h` | Central Server communication |
| MetricsCollector | `metrics_collector.cpp/h` | Hardware monitoring |
| OpenAIAPIServer | `openai_api_server.cpp/h` | Model serving API |

## Data Flows

### Job Submission Flow

```
1. User creates model in Engine
         |
         v
2. Engine submits job via gRPC
   SubmitJobRequest {
     model_definition,
     dataset_uri,
     hyperparameters
   }
         |
         v
3. Central Server receives request
   - Validates job configuration
   - Creates escrow on Solana
   - Adds to job queue
         |
         v
4. Scheduler matches job to node
   - Checks device requirements
   - Considers node reputation
   - Selects best available node
         |
         v
5. Central Server assigns job
   AssignJobRequest -> Server Node
         |
         v
6. Server Node executes job
   - Downloads dataset
   - Runs training loop
   - Reports progress
         |
         v
7. Server Node reports completion
   ReportCompletionRequest {
     model_weights_uri,
     final_metrics
   }
         |
         v
8. Central Server releases payment
   - Verifies results
   - Transfers tokens from escrow
         |
         v
9. Engine receives notification
   - Downloads trained model
   - Updates UI
```

### Node Registration Flow

```
1. Server Node starts
         |
         v
2. Collects hardware info
   - GPU devices (CUDA/OpenCL)
   - CPU cores, RAM
   - Network capabilities
         |
         v
3. Sends RegisterNodeRequest
   NodeInfo {
     devices: [...],
     cpu_cores,
     ram_total,
     compute_score,
     wallet_address
   }
         |
         v
4. Central Server validates
   - Generates node_id
   - Creates session_token
   - Stores in database
         |
         v
5. Returns RegisterNodeResponse
   {
     node_id,
     session_token
   }
         |
         v
6. Node starts heartbeat loop
   Every 10 seconds:
   HeartbeatRequest {
     node_id,
     current_status,
     active_jobs
   }
```

### Training Progress Flow

```
Training Loop (Server Node)
         |
    +----+----+----+
    |    |    |    |
    v    v    v    v
 Epoch 1  2   3   ...N
    |    |    |    |
    +----+----+----+
         |
         v
For each epoch:
1. Calculate loss/accuracy
2. Send ReportProgressRequest
   {
     job_id,
     progress: 0.0-1.0,
     metrics: {loss, accuracy},
     current_epoch
   }
         |
         v
3. Central Server updates
   - Database job status
   - Redis cache
   - Streams to Engine
         |
         v
4. Engine receives stream
   JobUpdateStream {
     live_metrics,
     log_message
   }
         |
         v
5. UI updates in real-time
   - Training Dashboard plots
   - Viewport live metrics
```

## Network Topology

### Centralized Orchestration, Distributed Compute

```
                    +---------------------+
                    |   Central Server    |
                    |   (Single Point)    |
                    +---------------------+
                           /|\
                          / | \
                         /  |  \
                        /   |   \
                       /    |    \
                      v     v     v
               +------+ +------+ +------+
               |Node 1| |Node 2| |Node 3|
               |      | |      | |      |
               +------+ +------+ +------+
                  |        |        |
                  v        v        v
               +------+ +------+ +------+
               | GPU  | | GPU  | | CPU  |
               +------+ +------+ +------+
```

**Communication Patterns:**

| Pattern | Protocol | Use Case |
|---------|----------|----------|
| Request-Response | gRPC Unary | Job submission, status queries |
| Server Streaming | gRPC Stream | Real-time job updates |
| Heartbeat | gRPC Unary | Node keep-alive |
| P2P (Future) | WebRTC/QUIC | Direct Engine-Node transfer |

### Port Allocation

| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| Central Server gRPC | 50051 | gRPC/HTTP2 | Main API |
| Central Server REST | 8080 | HTTP/1.1 | Web dashboard |
| Server Node Deployment | 50052 | gRPC/HTTP2 | Model deployment |
| Server Node Terminal | 50053 | gRPC/HTTP2 | Remote terminal |
| Server Node OpenAI API | 8000 | HTTP/1.1 | Model inference |

## Security Architecture

### Authentication Flow

```
1. Node Registration
   - Node generates key pair
   - Sends public key to Central Server
   - Receives session token (JWT)

2. Job Authorization
   - Central Server generates job-specific token
   - Token includes: job_id, node_id, expiration
   - Node validates token before execution

3. Payment Authorization
   - Escrow created on Solana
   - Only Central Server can release
   - Multi-sig for large payments (future)
```

### Sandboxing Strategy

```
Untrusted Code Execution:
+-----------------------------------+
|         Server Node Host          |
|                                   |
|  +-----------------------------+  |
|  |     Docker Container        |  |
|  |                             |  |
|  |  +---------------------+    |  |
|  |  | Training Script     |    |  |
|  |  | (Sandboxed)         |    |  |
|  |  +---------------------+    |  |
|  |                             |  |
|  |  Limited: Network, FS,     |  |
|  |  Memory, CPU Time          |  |
|  +-----------------------------+  |
|                                   |
+-----------------------------------+
```

## Database Schema Overview

### Core Tables

```sql
-- Nodes table
nodes (
    id          UUID PRIMARY KEY,
    name        VARCHAR(255),
    ip_address  VARCHAR(45),
    port        INTEGER,
    cpu_cores   INTEGER,
    ram_total   BIGINT,
    status      VARCHAR(20),
    reputation  DECIMAL(3,2),
    created_at  TIMESTAMP,
    last_seen   TIMESTAMP
)

-- Jobs table
jobs (
    id              UUID PRIMARY KEY,
    user_id         UUID,
    job_type        VARCHAR(20),
    status          VARCHAR(20),
    priority        INTEGER,
    model_config    JSONB,
    assigned_node   UUID REFERENCES nodes(id),
    progress        DECIMAL(5,4),
    payment_amount  DECIMAL(20,8),
    escrow_tx       VARCHAR(88),
    created_at      TIMESTAMP,
    started_at      TIMESTAMP,
    completed_at    TIMESTAMP
)

-- Metrics table (time series)
metrics (
    id          BIGSERIAL PRIMARY KEY,
    node_id     UUID REFERENCES nodes(id),
    timestamp   TIMESTAMP,
    cpu_usage   REAL,
    gpu_usage   REAL,
    memory_used BIGINT,
    jobs_active INTEGER
)
```

## Scalability Considerations

### Horizontal Scaling

| Component | Scaling Strategy |
|-----------|------------------|
| Server Nodes | Add more nodes; Central Server auto-discovers |
| Central Server | Currently single instance; future: sharding |
| Database | Read replicas, connection pooling |
| Redis Cache | Redis Cluster for high availability |

### Performance Optimizations

1. **Connection Pooling**: SQLx maintains connection pool
2. **Caching**: Redis for frequently accessed data
3. **Async I/O**: Tokio runtime for concurrent requests
4. **Batch Processing**: Group small jobs for efficiency
5. **Data Locality**: Prefer nodes close to data source

---

**Next**: [Technology Stack](technology-stack.md) | [Data Flow](data-flow.md)
