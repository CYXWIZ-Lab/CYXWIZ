# How to Use CyxWiz Central Server and Server Node

This guide explains how to set up, run, and use the CyxWiz Central Server (Rust orchestrator) and Server Node (C++ compute worker) for distributed ML training.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Setup Instructions](#setup-instructions)
4. [Running the Central Server](#running-the-central-server)
5. [Running the Server Node](#running-the-server-node)
6. [Verifying the Connection](#verifying-the-connection)
7. [Common Operations](#common-operations)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Configuration](#advanced-configuration)

---

## Overview

### What is the Central Server?

The **Central Server** is a Rust-based orchestrator that:
- Manages node registration and discovery
- Schedules ML training jobs to available nodes
- Handles payment processing via Solana blockchain
- Maintains job queues and node metrics
- Provides gRPC and REST APIs

### What is the Server Node?

The **Server Node** is a C++ compute worker that:
- Registers with the Central Server
- Executes ML training jobs using ArrayFire
- Reports progress and metrics
- Supports GPU acceleration (CUDA/OpenCL)
- Can run in standalone mode if disconnected

### Architecture Flow

```
┌─────────────────┐      gRPC       ┌──────────────────┐
│  Central Server │◄────────────────┤   Server Node    │
│   (Rust/Tonic)  │                 │  (C++/ArrayFire) │
│                 │  RegisterNode   │                  │
│  - Job Queue    │◄────────────────┤  - GPU Training  │
│  - Scheduler    │                 │  - Job Executor  │
│  - Database     │  AssignJob      │  - Metrics       │
│  - Redis Cache  ├────────────────►│                  │
└─────────────────┘                 └──────────────────┘
        │                                    │
        │ Heartbeat (every 10s)             │
        │◄───────────────────────────────────┤
        │                                    │
        │ ReportProgress                     │
        │◄───────────────────────────────────┤
        │                                    │
        │ ReportCompletion                   │
        │◄───────────────────────────────────┤
```

---

## Prerequisites

### System Requirements

**For Central Server:**
- Operating System: Windows, Linux, or macOS
- Rust toolchain (1.70+)
- PostgreSQL 14+ or SQLite (for development)
- Redis 6+ (optional, for caching)
- 2 GB RAM minimum
- 100 MB disk space

**For Server Node:**
- Operating System: Windows, Linux, or macOS
- C++ compiler (MSVC 2019+, GCC 9+, or Clang 10+)
- ArrayFire 3.8+ (with CUDA/OpenCL support)
- GPU with CUDA 11+ or OpenCL 2.0+ (optional, CPU mode available)
- 4 GB RAM minimum
- 1 GB disk space

### Software Dependencies

**Central Server Dependencies:**
```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install PostgreSQL (example for Ubuntu)
sudo apt-get install postgresql postgresql-contrib

# Install Redis (example for Ubuntu)
sudo apt-get install redis-server

# Or use Docker
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=password postgres:14
docker run -d -p 6379:6379 redis:alpine
```

**Server Node Dependencies:**
```bash
# Install ArrayFire
# Download from: https://arrayfire.com/download

# Set environment variable
export ArrayFire_DIR=/path/to/arrayfire

# Install vcpkg dependencies (handled automatically by build script)
```

---

## Setup Instructions

### 1. Build the Central Server

```bash
# Navigate to Central Server directory
cd D:/Dev/CyxWiz_Claude/cyxwiz-central-server

# Build in release mode
cargo build --release

# Binary will be at: target/release/cyxwiz-central-server.exe (Windows)
# or target/release/cyxwiz-central-server (Linux/macOS)
```

**Build Time:** ~60 seconds (first build may take longer)

### 2. Build the Server Node

```bash
# Navigate to project root
cd D:/Dev/CyxWiz_Claude

# Run the build script
./build.bat        # Windows
# or
./scripts/build.sh # Linux/macOS

# Binary will be at: build/windows-release/bin/Release/cyxwiz-server-node.exe
```

**Build Time:** ~5-10 minutes (first build may take longer due to vcpkg)

### 3. Configure the Central Server

Create a configuration file at `cyxwiz-central-server/config.toml`:

```toml
[server]
grpc_address = "0.0.0.0:50051"
rest_address = "0.0.0.0:8080"

[database]
# For SQLite (development/testing)
url = "sqlite://./cyxwiz.db?mode=rwc"

# For PostgreSQL (production)
# url = "postgresql://username:password@localhost:5432/cyxwiz"

max_connections = 10
min_connections = 2

[redis]
url = "redis://127.0.0.1:6379"
max_connections = 10

[scheduler]
poll_interval_secs = 5
max_jobs_per_node = 4
job_timeout_secs = 3600

[blockchain]
network = "devnet"
solana_rpc_url = "https://api.devnet.solana.com"
payer_keypair_path = "~/.config/solana/id.json"
program_id = "11111111111111111111111111111111"
```

**Note:** The Central Server works with default configuration if `config.toml` is not provided.

### 4. Configure the Server Node (Optional)

The Server Node uses command-line arguments and defaults. No configuration file needed for basic usage.

---

## Running the Central Server

### Basic Usage

```bash
# Navigate to Central Server directory
cd D:/Dev/CyxWiz_Claude/cyxwiz-central-server

# Run in normal mode (gRPC + scheduler)
./target/release/cyxwiz-central-server.exe

# Run in TUI mode (for monitoring)
./target/release/cyxwiz-central-server.exe --tui
```

### Expected Output

```
INFO  CyxWiz Central Server v0.1.0
INFO  ========================================
INFO  Loading configuration...
INFO  Connecting to database: sqlite://./cyxwiz.db?mode=rwc
INFO  Running database migrations...
INFO  Migrations completed
INFO  Attempting to connect to Redis: redis://127.0.0.1:6379
INFO  ✓ Redis connected successfully
INFO  Starting job scheduler...
INFO  Job scheduler started
INFO  Initializing Solana client...
ERROR Solana keypair file not found: ~/.config/solana/id.json
ERROR Payment processing will be disabled
INFO  Starting gRPC server on 0.0.0.0:50051
INFO  ========================================
INFO  🚀 gRPC Server ready!
INFO     gRPC endpoint: 0.0.0.0:50051
INFO     NodeService: ENABLED (RegisterNode, Heartbeat, ReportProgress, ReportCompletion)
INFO     JobStatusService: ENABLED (UpdateJobStatus, ReportJobResult)
INFO     REST API: DISABLED (requires JobService fix)
INFO  ========================================
```

### Running in TUI Mode

TUI (Text User Interface) mode provides a live dashboard:

```bash
./target/release/cyxwiz-central-server.exe --tui
```

**TUI Features:**
- Real-time node status
- Active job monitoring
- Database statistics
- Live log streaming

**TUI Controls:**
- `q` - Quit
- `↑/↓` - Scroll logs
- `Tab` - Switch panels

### Running as a Service (Production)

**Windows (using NSSM):**
```bash
# Download NSSM from https://nssm.cc/download
nssm install CyxWizCentralServer "D:\Dev\CyxWiz_Claude\cyxwiz-central-server\target\release\cyxwiz-central-server.exe"
nssm start CyxWizCentralServer
```

**Linux (systemd):**
```bash
# Create service file: /etc/systemd/system/cyxwiz-central-server.service
[Unit]
Description=CyxWiz Central Server
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=cyxwiz
WorkingDirectory=/opt/cyxwiz/central-server
ExecStart=/opt/cyxwiz/central-server/cyxwiz-central-server
Restart=always

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable cyxwiz-central-server
sudo systemctl start cyxwiz-central-server
```

---

## Running the Server Node

### Basic Usage

```bash
# Navigate to project root
cd D:/Dev/CyxWiz_Claude

# Run the Server Node
./build/windows-release/bin/Release/cyxwiz-server-node.exe
```

### Expected Output

```
[info] CyxWiz Server Node v0.1.0
[info] ========================================
[info] Initializing CyxWiz Backend v0.1.0
[info] ArrayFire initialized successfully
[info] OpenCL backend available
[info] Found discrete GPU: NVIDIA_GeForce_GTX_1050_Ti (device 0)
[info] Using OpenCL device 0: NVIDIA_GeForce_GTX_1050_Ti
[info] Node ID: node_1763450697
[info] Deployment service: 0.0.0.0:50052
[info] Terminal service: 0.0.0.0:50053
[info] Node service: 0.0.0.0:50054
[info] JobExecutor initialized for node: node_1763450697
[info] NodeServiceImpl created for node: node_1763450697
[info] NodeService started on 0.0.0.0:50054
[info] DeploymentManager initialized for node: node_1763450697
[info] DeploymentHandler started successfully on 0.0.0.0:50052
[info] TerminalHandler started successfully on 0.0.0.0:50053
[info] Connecting to Central Server at localhost:50051...
[info] NodeClient created for Central Server: localhost:50051
[info] Registering node node_1763450697 with Central Server...
[info] Node registered successfully!
[info]   Node ID: f7c75722-7368-4e09-bf3e-de3c84aec8d3
[info]   Session Token: session_f7c75722-7368-4e09-bf3e-de3c84aec8d3
[info] Successfully registered with Central Server
[info] Heartbeat started (interval: 10s)
[info] ========================================
[info] Server Node is ready!
[info]   Deployment endpoint:  0.0.0.0:50052
[info]   Terminal endpoint:    0.0.0.0:50053
[info]   Node service:         0.0.0.0:50054
[info]   Active jobs:          0
[info] ========================================
[info] Press Ctrl+C to shutdown
```

### Command-Line Options

```bash
# Run with custom Central Server address
./cyxwiz-server-node.exe --server=192.168.1.100:50051

# Run in standalone mode (no Central Server connection)
./cyxwiz-server-node.exe --standalone

# Specify GPU device
./cyxwiz-server-node.exe --device=cuda:0

# Set resource limits
./cyxwiz-server-node.exe --max-jobs=4 --max-memory=8G
```

### Running Multiple Server Nodes

To run multiple nodes on the same machine:

```bash
# Node 1 (default ports)
./cyxwiz-server-node.exe

# Node 2 (custom ports)
./cyxwiz-server-node.exe --deployment-port=50062 --terminal-port=50063 --node-port=50064

# Node 3 (custom ports)
./cyxwiz-server-node.exe --deployment-port=50072 --terminal-port=50073 --node-port=50074
```

### Running as a Service

**Windows:**
```bash
nssm install CyxWizServerNode "D:\Dev\CyxWiz_Claude\build\windows-release\bin\Release\cyxwiz-server-node.exe"
nssm start CyxWizServerNode
```

**Linux:**
```bash
# Create service file: /etc/systemd/system/cyxwiz-server-node.service
[Unit]
Description=CyxWiz Server Node
After=network.target

[Service]
Type=simple
User=cyxwiz
WorkingDirectory=/opt/cyxwiz/server-node
ExecStart=/opt/cyxwiz/server-node/cyxwiz-server-node
Restart=always

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable cyxwiz-server-node
sudo systemctl start cyxwiz-server-node
```

---

## Verifying the Connection

### 1. Check Central Server Logs

Look for this log entry when a Server Node connects:

```
INFO  cyxwiz_central_server::api::grpc::node_service: Registering node: CyxWiz-Node-node_176
INFO  cyxwiz_central_server::api::grpc::node_service: Node f7c75722-7368-4e09-bf3e-de3c84aec8d3 registered successfully
```

### 2. Check Server Node Logs

Successful registration shows:

```
[info] Node registered successfully!
[info]   Node ID: f7c75722-7368-4e09-bf3e-de3c84aec8d3
[info]   Session Token: session_f7c75722-7368-4e09-bf3e-de3c84aec8d3
[info] Successfully registered with Central Server
[info] Heartbeat started (interval: 10s)
```

### 3. Query Registered Nodes (PostgreSQL)

```sql
-- Connect to PostgreSQL
psql -U username -d cyxwiz

-- List all registered nodes
SELECT id, name, wallet_address, status, last_heartbeat, gpu_model
FROM nodes
ORDER BY last_heartbeat DESC;
```

### 4. Query Registered Nodes (SQLite)

```bash
# Open SQLite database
sqlite3 cyxwiz-central-server/cyxwiz.db

# List all registered nodes
SELECT id, name, wallet_address, status, last_heartbeat, gpu_model
FROM nodes
ORDER BY last_heartbeat DESC;
```

### 5. Test gRPC Connection

Use `grpcurl` to test the connection:

```bash
# Install grpcurl
go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest

# List available services
grpcurl -plaintext localhost:50051 list

# Expected output:
# cyxwiz.protocol.JobStatusService
# cyxwiz.protocol.NodeService
# grpc.reflection.v1alpha.ServerReflection
```

---

## Common Operations

### Starting the Full System

**Step 1:** Start Redis (if using)
```bash
# Docker
docker run -d -p 6379:6379 redis:alpine

# Or native
redis-server
```

**Step 2:** Start PostgreSQL (if using)
```bash
# Docker
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=password postgres:14

# Or native
pg_ctl -D /usr/local/var/postgres start
```

**Step 3:** Start Central Server
```bash
cd cyxwiz-central-server
./target/release/cyxwiz-central-server.exe
```

**Step 4:** Start Server Node(s)
```bash
cd D:/Dev/CyxWiz_Claude
./build/windows-release/bin/Release/cyxwiz-server-node.exe
```

**Step 5:** Verify all components are running
```bash
# Check processes
ps aux | grep cyxwiz

# Check ports
netstat -an | grep 50051   # Central Server gRPC
netstat -an | grep 50052   # Server Node Deployment
netstat -an | grep 50053   # Server Node Terminal
netstat -an | grep 50054   # Server Node Service
```

### Stopping the System

```bash
# Stop all CyxWiz processes
taskkill /F /IM cyxwiz-central-server.exe  # Windows
taskkill /F /IM cyxwiz-server-node.exe     # Windows

# Or
pkill cyxwiz-central-server  # Linux/macOS
pkill cyxwiz-server-node     # Linux/macOS
```

### Monitoring Node Heartbeats

Heartbeats are sent every 10 seconds. Check Central Server logs:

```
# Successful heartbeat (no log entry, silence is success)

# Failed heartbeat (node marked offline after 30s timeout)
WARN  Node f7c75722-7368-4e09-bf3e-de3c84aec8d3 heartbeat timeout, marking as offline
```

### Submitting a Job (Future - requires Engine)

```bash
# Using the CyxWiz Engine GUI (future)
# 1. Launch Engine
./build/windows-release/bin/Release/cyxwiz-engine.exe

# 2. Create ML pipeline in node editor
# 3. Click "Submit Job to Network"
# 4. Select Central Server endpoint (localhost:50051)
# 5. Monitor job progress in real-time
```

---

## Troubleshooting

### Issue: Server Node fails to register with error code 12

**Symptoms:**
```
[error] gRPC error during registration:  (code: 12)
[error] Failed to register with Central Server
[warning] Server Node will run in standalone mode
```

**Solution:**
- Ensure Central Server is running and accessible
- Check that NodeService is enabled in Central Server logs
- Verify network connectivity: `ping localhost`
- Check firewall settings for port 50051

### Issue: Central Server fails to start (database connection error)

**Symptoms:**
```
ERROR Failed to connect to database
ERROR Please ensure PostgreSQL is running and the database exists
```

**Solution:**
```bash
# For PostgreSQL
createdb cyxwiz

# For SQLite (no action needed, auto-created)
# Just ensure directory is writable
```

### Issue: Redis connection failed

**Symptoms:**
```
WARN  ⚠ Redis connection failed: Connection refused
WARN    Running in MOCK MODE - Redis features disabled
```

**Solution:**
```bash
# Start Redis
redis-server

# Or use Docker
docker run -d -p 6379:6379 redis:alpine

# Or continue without Redis (mock mode works fine for testing)
```

### Issue: Port already in use

**Symptoms:**
```
ERROR Address already in use (os error 48)
```

**Solution:**
```bash
# Find process using the port
netstat -ano | findstr :50051  # Windows
lsof -i :50051                 # Linux/macOS

# Kill the process
taskkill /F /PID <PID>         # Windows
kill -9 <PID>                  # Linux/macOS
```

### Issue: ArrayFire not found

**Symptoms:**
```
[error] Failed to initialize ArrayFire
[error] ArrayFire library not found
```

**Solution:**
```bash
# Set environment variable
export ArrayFire_DIR=/path/to/arrayfire      # Linux/macOS
set ArrayFire_DIR=C:\path\to\arrayfire       # Windows

# Or install ArrayFire
# Download from: https://arrayfire.com/download
```

### Issue: Node shows as offline despite heartbeats

**Solution:**
```sql
-- Check last heartbeat timestamp
SELECT id, name, last_heartbeat, status FROM nodes;

-- Manually mark node as online (debugging only)
UPDATE nodes SET status = 'online', last_heartbeat = NOW() WHERE id = 'your-node-id';
```

### Enable Debug Logging

**Central Server:**
```bash
RUST_LOG=debug ./target/release/cyxwiz-central-server.exe
```

**Server Node:**
```bash
# Rebuild with debug symbols
cmake --build build/windows-release --config Debug
./build/windows-release/bin/Debug/cyxwiz-server-node.exe
```

---

## Advanced Configuration

### Load Balancing Multiple Server Nodes

The Central Server's job scheduler automatically load-balances jobs across nodes based on:
- Current node load (active jobs)
- Hardware capabilities (GPU model, RAM, CPU cores)
- Node reputation score
- Geographic location (future feature)

**Example:** 3 nodes with different specs
```
Node 1: 4 CPU cores, 8GB RAM, NVIDIA RTX 3090
Node 2: 8 CPU cores, 16GB RAM, AMD RX 6800
Node 3: 16 CPU cores, 32GB RAM, No GPU

Scheduler assigns:
- GPU-intensive jobs → Node 1 (NVIDIA RTX 3090)
- OpenCL jobs → Node 2 (AMD RX 6800)
- CPU-only jobs → Node 3 (16 cores)
```

### Custom Node Metadata

Nodes automatically report:
- Hardware specs (CPU, GPU, RAM)
- Network region
- Uptime percentage
- Completed jobs count

Future support for custom metadata:
```bash
./cyxwiz-server-node.exe --metadata="region=us-west,tier=premium"
```

### Database Migrations

Migrations run automatically on startup. To manually manage:

```bash
cd cyxwiz-central-server

# Check migration status
sqlx migrate info

# Run pending migrations
sqlx migrate run

# Revert last migration
sqlx migrate revert
```

### Backup and Restore

**PostgreSQL:**
```bash
# Backup
pg_dump cyxwiz > cyxwiz_backup.sql

# Restore
psql cyxwiz < cyxwiz_backup.sql
```

**SQLite:**
```bash
# Backup
cp cyxwiz.db cyxwiz_backup.db

# Restore
cp cyxwiz_backup.db cyxwiz.db
```

---

## Performance Tuning

### Central Server

**Database Connection Pool:**
```toml
[database]
max_connections = 20  # Increase for high load
min_connections = 5
```

**Scheduler Interval:**
```toml
[scheduler]
poll_interval_secs = 2  # Check for jobs every 2 seconds (faster)
```

**Redis Connection Pool:**
```toml
[redis]
max_connections = 20  # Increase for high cache usage
```

### Server Node

**Concurrent Jobs:**
```bash
./cyxwiz-server-node.exe --max-jobs=8  # Allow up to 8 concurrent jobs
```

**Memory Limits:**
```bash
./cyxwiz-server-node.exe --max-memory=16G  # Limit to 16GB RAM usage
```

**GPU Device Selection:**
```bash
# Use specific GPU
./cyxwiz-server-node.exe --device=cuda:1  # Use second CUDA GPU

# Use CPU only
./cyxwiz-server-node.exe --device=cpu
```

---

## Security Considerations

### Production Deployment

1. **Use TLS for gRPC:**
   ```toml
   [server]
   tls_cert_path = "/path/to/cert.pem"
   tls_key_path = "/path/to/key.pem"
   ```

2. **Restrict Database Access:**
   ```toml
   [database]
   url = "postgresql://readonly_user:password@localhost:5432/cyxwiz"
   ```

3. **Enable Authentication:**
   ```toml
   [auth]
   jwt_secret = "your-secret-key"
   token_expiry_secs = 3600
   ```

4. **Firewall Rules:**
   ```bash
   # Allow only trusted IPs to connect to Central Server
   ufw allow from 192.168.1.0/24 to any port 50051
   ```

---

## Next Steps

1. **Enable JobService:** Fix compilation errors to enable job submission
2. **Launch Engine:** Use the desktop client to submit training jobs
3. **Monitor Dashboard:** Access web dashboard (future) at http://localhost:8080
4. **Blockchain Integration:** Set up Solana keypair for payment processing
5. **Production Deployment:** Use systemd/NSSM for auto-restart and logging

---

## Quick Reference

### Port Summary

| Service              | Port  | Protocol |
|---------------------|-------|----------|
| Central Server gRPC | 50051 | gRPC     |
| Central Server REST | 8080  | HTTP     |
| Server Node Deployment | 50052 | gRPC |
| Server Node Terminal | 50053 | gRPC |
| Server Node Service | 50054 | gRPC |
| PostgreSQL          | 5432  | TCP      |
| Redis               | 6379  | TCP      |

### Command Cheat Sheet

```bash
# Build Central Server
cd cyxwiz-central-server && cargo build --release

# Build Server Node
cd D:/Dev/CyxWiz_Claude && ./build.bat

# Run Central Server
cd cyxwiz-central-server && ./target/release/cyxwiz-central-server.exe

# Run Server Node
cd D:/Dev/CyxWiz_Claude && ./build/windows-release/bin/Release/cyxwiz-server-node.exe

# Check processes
ps aux | grep cyxwiz  # Linux/macOS
tasklist | findstr cyxwiz  # Windows

# Kill all
pkill cyxwiz  # Linux/macOS
taskkill /F /IM cyxwiz*.exe  # Windows

# View logs
tail -f cyxwiz-central-server.log  # Central Server
tail -f cyxwiz-server-node.log     # Server Node
```

---

## Support

For issues, bugs, or feature requests:
- GitHub Issues: https://github.com/your-repo/cyxwiz/issues
- Documentation: See `CLAUDE.md` for architecture details
- Success Report: See `NODESERVICE_IMPLEMENTATION_SUCCESS.md`

**Version:** 0.1.0
**Last Updated:** 2025-11-18
