# CyxWiz Central Server

The Central Server is the network orchestrator for the CyxWiz distributed computing platform. Written in Rust, it manages job scheduling, node registry, and blockchain payments.

## Overview

The Central Server provides:
- **gRPC Services** - Job submission, node management
- **Job Scheduler** - Intelligent job-to-node matching
- **Node Registry** - Track and monitor compute nodes
- **Payment Processing** - Solana blockchain integration
- **TUI Dashboard** - Real-time monitoring interface
- **REST API** - Web dashboard support

## Architecture

```
cyxwiz-central-server/
├── src/
│   ├── main.rs              # Entry point
│   ├── config.rs            # Configuration loading
│   ├── error.rs             # Error types
│   ├── lib.rs               # Library exports
│   ├── pb.rs                # Generated protobuf code
│   ├── api/
│   │   ├── grpc/            # gRPC service implementations
│   │   │   ├── job_service.rs
│   │   │   ├── node_service.rs
│   │   │   ├── job_status_service.rs
│   │   │   └── wallet_service.rs
│   │   └── rest/            # REST API endpoints
│   │       └── dashboard.rs
│   ├── auth/
│   │   ├── mod.rs
│   │   └── jwt.rs           # JWT token management
│   ├── blockchain/
│   │   ├── mod.rs
│   │   ├── client.rs        # Blockchain client trait
│   │   ├── solana_client.rs # Solana implementation
│   │   ├── escrow.rs        # Payment escrow
│   │   ├── payment_processor.rs
│   │   └── reputation.rs    # Node reputation
│   ├── cache/
│   │   └── mod.rs           # Redis caching
│   ├── database/
│   │   ├── mod.rs
│   │   ├── models.rs        # Database models
│   │   └── queries.rs       # SQL queries
│   ├── scheduler/
│   │   ├── mod.rs
│   │   ├── job_queue.rs     # Job queue management
│   │   ├── matcher.rs       # Job-node matching
│   │   └── node_monitor.rs  # Node health monitoring
│   └── tui/
│       ├── mod.rs
│       ├── app.rs           # TUI application state
│       ├── runner.rs        # TUI event loop
│       ├── updater.rs       # Data updates
│       └── views/           # TUI views
│           ├── dashboard.rs
│           ├── jobs.rs
│           ├── nodes.rs
│           └── blockchain.rs
├── config.toml              # Default configuration
└── Cargo.toml               # Dependencies
```

## Documentation Sections

| Section | Description |
|---------|-------------|
| [Configuration](configuration.md) | Server configuration options |
| [gRPC Services](grpc/index.md) | Service implementations |
| [Job Scheduler](scheduler.md) | Job matching and queue |
| [Database](database.md) | Schema and queries |
| [Blockchain](../blockchain/index.md) | Payment integration |
| [TUI Dashboard](tui.md) | Terminal interface |
| [REST API](rest-api.md) | HTTP endpoints |
| [Deployment](deployment.md) | Production setup |

## Quick Start

### Prerequisites

- Rust 1.70+
- PostgreSQL 13+ (or SQLite for development)
- Redis 6+ (optional, falls back to mock)
- Solana CLI (for blockchain features)

### Running the Server

```bash
cd cyxwiz-central-server

# Set PROTOC for protobuf compilation
export PROTOC="../vcpkg/packages/protobuf_x64-windows/tools/protobuf/protoc.exe"

# Build
cargo build --release

# Run in gRPC/REST mode (default)
cargo run --release

# Run in TUI mode
cargo run --release -- --tui
```

### Default Ports

| Service | Port | Protocol |
|---------|------|----------|
| gRPC | 50051 | HTTP/2 |
| REST | 8080 | HTTP/1.1 |

## Configuration

Configuration via `config.toml`:

```toml
[server]
grpc_address = "0.0.0.0:50051"
rest_address = "0.0.0.0:8080"

[database]
url = "postgresql://user:pass@localhost/cyxwiz"
# Or for SQLite:
# url = "sqlite:cyxwiz.db?mode=rwc"
max_connections = 10

[redis]
url = "redis://127.0.0.1:6379"

[blockchain]
network = "devnet"
solana_rpc_url = "https://api.devnet.solana.com"
payer_keypair_path = "~/.config/solana/id.json"
program_id = "11111111111111111111111111111111"

[scheduler]
polling_interval_ms = 5000
max_jobs_per_node = 5
timeout_seconds = 3600

[jwt]
secret = "your-secret-key-here"
p2p_token_expiration_seconds = 3600
```

## gRPC Services

### JobService

Client-to-server job management:

```protobuf
service JobService {
  rpc SubmitJob(SubmitJobRequest) returns (SubmitJobResponse);
  rpc GetJobStatus(GetJobStatusRequest) returns (GetJobStatusResponse);
  rpc CancelJob(CancelJobRequest) returns (CancelJobResponse);
  rpc StreamJobUpdates(GetJobStatusRequest) returns (stream JobUpdateStream);
  rpc ListJobs(ListJobsRequest) returns (ListJobsResponse);
}
```

### NodeService

Node registration and management:

```protobuf
service NodeService {
  rpc RegisterNode(RegisterNodeRequest) returns (RegisterNodeResponse);
  rpc Heartbeat(HeartbeatRequest) returns (HeartbeatResponse);
  rpc AssignJob(AssignJobRequest) returns (AssignJobResponse);
  rpc ReportProgress(ReportProgressRequest) returns (ReportProgressResponse);
  rpc ReportCompletion(ReportCompletionRequest) returns (ReportCompletionResponse);
}
```

### JobStatusService

Node-to-server status reporting:

```protobuf
service JobStatusService {
  rpc UpdateJobStatus(UpdateJobStatusRequest) returns (UpdateJobStatusResponse);
  rpc ReportJobResult(ReportJobResultRequest) returns (ReportJobResultResponse);
}
```

## Scheduler

The job scheduler matches jobs to nodes based on:

1. **Device Requirements** - GPU type, memory
2. **Node Capacity** - Available resources
3. **Reputation Score** - Historical performance
4. **Location** - Network latency (future)
5. **Queue Length** - Load balancing

### Scheduling Algorithm

```rust
fn score_node(job: &Job, node: &Node) -> f64 {
    let mut score = 0.0;

    // Device match
    if node.has_device(job.required_device) {
        score += 100.0;
    }

    // Memory availability
    if node.available_memory >= job.estimated_memory {
        score += 50.0;
    }

    // Reputation bonus
    score += node.reputation * 30.0;

    // Penalize busy nodes
    score -= node.active_jobs as f64 * 10.0;

    score
}
```

## Database Schema

### Core Tables

```sql
-- Nodes
CREATE TABLE nodes (
    id UUID PRIMARY KEY,
    name VARCHAR(255),
    ip_address VARCHAR(45),
    port INTEGER,
    status VARCHAR(20),
    reputation DECIMAL(3,2),
    created_at TIMESTAMP,
    last_seen TIMESTAMP
);

-- Jobs
CREATE TABLE jobs (
    id UUID PRIMARY KEY,
    user_id UUID,
    job_type VARCHAR(20),
    status VARCHAR(20),
    priority INTEGER,
    model_config JSONB,
    assigned_node UUID,
    progress DECIMAL(5,4),
    payment_amount DECIMAL(20,8),
    created_at TIMESTAMP
);

-- Metrics
CREATE TABLE metrics (
    id BIGSERIAL PRIMARY KEY,
    node_id UUID,
    timestamp TIMESTAMP,
    cpu_usage REAL,
    gpu_usage REAL,
    memory_used BIGINT
);
```

## TUI Dashboard

The TUI provides real-time monitoring:

```
+------------------------------------------------------------------+
|  CyxWiz Central Server v0.1.0                    [Tab: Dashboard] |
+------------------------------------------------------------------+
|  NETWORK STATS                    |  SYSTEM HEALTH               |
|  +----------------------------+   |  +------------------------+  |
|  | Active Nodes:    12        |   |  | Database:    OK        |  |
|  | Pending Jobs:    45        |   |  | Redis:       MOCK      |  |
|  | Running Jobs:    8         |   |  | Solana:      Devnet    |  |
|  | Completed (24h): 234       |   |  | Uptime:      3h 42m    |  |
|  +----------------------------+   |  +------------------------+  |
|                                                                   |
|  JOB THROUGHPUT (Last Hour)                                       |
|  [========================================                  ]     |
|   0    10    20    30    40    50    60                           |
|        Submitted        Completed        Failed                   |
|                                                                   |
|  TOP NODES BY REPUTATION                                          |
|  +-------------------+--------+--------+--------+                 |
|  | Node              | Jobs   | Rep.   | Status |                 |
|  +-------------------+--------+--------+--------+                 |
|  | node-alpha        |   156  |  0.98  | Online |                 |
|  | node-beta         |   143  |  0.95  | Online |                 |
|  | node-gamma        |    89  |  0.91  | Online |                 |
|  +-------------------+--------+--------+--------+                 |
+------------------------------------------------------------------+
|  [q] Quit  [Tab] Switch View  [j/k] Navigate  [Enter] Details    |
+------------------------------------------------------------------+
```

### TUI Navigation

| Key | Action |
|-----|--------|
| `Tab` | Switch between views |
| `j/k` | Navigate up/down |
| `Enter` | View details |
| `q` | Quit |
| `r` | Refresh |

### TUI Views

| View | Content |
|------|---------|
| Dashboard | Network overview, stats |
| Jobs | Job list, filtering |
| Nodes | Node list, details |
| Blockchain | Wallet, transactions |
| Logs | Server logs |
| Settings | Configuration |

## REST API

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/stats` | Network statistics |
| GET | `/api/jobs` | List jobs |
| GET | `/api/jobs/:id` | Job details |
| GET | `/api/nodes` | List nodes |
| GET | `/api/nodes/:id` | Node details |

### Example Requests

```bash
# Health check
curl http://localhost:8080/api/health

# Get stats
curl http://localhost:8080/api/stats

# List jobs
curl http://localhost:8080/api/jobs?status=running&limit=10
```

## Logging

The server uses `tracing` for structured logging:

```bash
# Default logging
RUST_LOG=info cargo run

# Debug logging
RUST_LOG=debug cargo run

# Component-specific
RUST_LOG=cyxwiz_central_server::scheduler=debug cargo run
```

## Monitoring

### Metrics Collected

- Job throughput (submissions/completions per hour)
- Node online/offline events
- Average job duration
- Payment volumes
- Error rates

### Health Checks

The `/api/health` endpoint returns:

```json
{
  "status": "healthy",
  "components": {
    "database": "ok",
    "redis": "ok",
    "solana": "ok"
  },
  "uptime_seconds": 13420
}
```

## Production Deployment

### Docker

```dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl-dev ca-certificates
COPY --from=builder /app/target/release/cyxwiz-central-server /usr/local/bin/
COPY config.toml /etc/cyxwiz/config.toml
EXPOSE 50051 8080
CMD ["cyxwiz-central-server"]
```

### Systemd Service

```ini
[Unit]
Description=CyxWiz Central Server
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=cyxwiz
ExecStart=/usr/local/bin/cyxwiz-central-server
Restart=always
Environment=RUST_LOG=info

[Install]
WantedBy=multi-user.target
```

---

**Next**: [Configuration](configuration.md) | [gRPC Services](grpc/index.md)
