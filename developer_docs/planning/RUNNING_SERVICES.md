# Running CyxWiz Services

This guide explains how to properly start, stop, and manage CyxWiz services.

## Quick Start

### Option 1: Automated Scripts (Recommended)

**Start Central Server:**
```bash
start-central-server.bat
```

**Start Server Node (in a new terminal):**
```bash
start-server-node.bat
```

**Stop All Services:**
```bash
stop-all.bat
```

### Option 2: Manual Commands

**Central Server:**
```bash
cd cyxwiz-central-server
cargo run --release
```

**Server Node:**
```bash
.\build\windows-release\bin\Release\cyxwiz-server-node.exe
```

## Port Allocation

| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| Central Server | 50051 | gRPC | NodeService, JobService |
| Central Server | 8080 | HTTP | REST API |
| Server Node | 50052 | gRPC | Deployment service |
| Server Node | 50053 | gRPC | Terminal service |
| Server Node | 50054 | gRPC | NodeService (job assignments) |

## Startup Order

**IMPORTANT**: Always start services in this order:

1. **Central Server** (ports 50051, 8080)
   - Starts gRPC server
   - Starts REST API
   - Initializes job scheduler
   - Connects to Redis and SQLite

2. **Server Node** (ports 50052, 50053, 50054)
   - Initializes ArrayFire/GPU
   - Creates JobExecutor
   - Starts all gRPC services
   - Registers with Central Server
   - Starts heartbeat

## Troubleshooting

### Error: "AddrInUse - only one usage of each socket"

**Cause**: Port is already in use by a previous instance.

**Solution 1 - Use stop-all.bat:**
```bash
stop-all.bat
```

**Solution 2 - Manual cleanup:**
```bash
# Find processes using CyxWiz ports
netstat -ano | findstr ":50051 :8080 :50052 :50053 :50054"

# Kill the process (replace PID with actual process ID)
taskkill /F /PID <PID>
```

### Error: "Failed to start gRPC server"

**Causes:**
1. Port conflict (another process using the port)
2. Previous instance not shut down properly
3. Firewall blocking the port

**Solutions:**
```bash
# 1. Stop all services
stop-all.bat

# 2. Check if ports are free
netstat -ano | findstr ":50051 :50052 :50053 :50054"

# 3. If ports are free, try starting again
start-central-server.bat
start-server-node.bat
```

### Error: "Failed to connect to Central Server"

**Cause**: Central Server is not running or not accessible.

**Solution:**
```bash
# 1. Make sure Central Server is running
netstat -ano | findstr ":50051.*LISTENING"

# 2. If not running, start it first
start-central-server.bat

# 3. Wait 2-3 seconds for initialization

# 4. Then start Server Node
start-server-node.bat
```

## Proper Shutdown

### Using Scripts (Recommended)

```bash
stop-all.bat
```

This will:
- Find all CyxWiz processes
- Send termination signals
- Clean up port bindings

### Manual Shutdown

Press `Ctrl+C` in each terminal window running a service. This triggers the graceful shutdown handlers:

**Central Server:**
- Stops accepting new jobs
- Waits for current operations
- Closes database connections
- Shuts down gRPC server

**Server Node:**
- Cancels all active jobs
- Waits for worker threads to finish
- Stops heartbeat
- Closes connections to Central Server
- Shuts down all gRPC servers

## Health Checks

### Central Server

**gRPC (port 50051):**
```bash
# Using grpcurl
grpcurl -plaintext localhost:50051 list
```

**REST API (port 8080):**
```bash
curl http://localhost:8080/api/health
```

### Server Node

**Check if services are running:**
```bash
netstat -ano | findstr ":50052 :50053 :50054"
```

**Expected output:**
```
TCP    0.0.0.0:50052          0.0.0.0:0              LISTENING       <PID>
TCP    0.0.0.0:50053          0.0.0.0:0              LISTENING       <PID>
TCP    0.0.0.0:50054          0.0.0.0:0              LISTENING       <PID>
```

All three ports should show the same PID.

## Logging

### Central Server (Rust)

Logs are printed to stdout with colors:
- `INFO` - Green
- `WARN` - Yellow
- `ERROR` - Red

**Log location:** Console output only (use `> logfile.txt` to capture)

### Server Node (C++)

Logs via spdlog to stdout:
- `[info]` - Informational messages
- `[warn]` - Warnings
- `[error]` - Errors

**Example startup log:**
```
[2025-11-17 08:15:25.893] [info] CyxWiz Server Node v0.1.0
[2025-11-17 08:15:27.163] [info] JobExecutor initialized for node: node_XXX
[2025-11-17 08:15:27.163] [info] NodeServiceImpl created for node: node_XXX
[2025-11-17 08:15:27.171] [info] NodeService started on 0.0.0.0:50054
```

## Development Workflow

### Rebuilding After Code Changes

**Server Node (C++):**
```bash
# Stop running instance
stop-all.bat

# Rebuild
cmake --build build/windows-release --config Release --target cyxwiz-server-node

# Restart
start-central-server.bat
start-server-node.bat
```

**Central Server (Rust):**
```bash
# Stop running instance
stop-all.bat

# Rebuild and run
cd cyxwiz-central-server
cargo run --release
```

### Running Multiple Server Nodes

To run multiple Server Nodes on the same machine, you need to modify port assignments:

**Node 1:** ports 50052, 50053, 50054 (default)
**Node 2:** ports 50055, 50056, 50057
**Node 3:** ports 50058, 50059, 50060

Edit `main.cpp` to change port numbers, then rebuild.

## Common Issues

### Issue: Server Node shows "Active jobs: 0" but job is running

**Diagnosis:** Check if job was properly assigned via AssignJob RPC.

**Solution:** Verify Central Server job assignment is implemented.

### Issue: GPU not detected

**Check:**
```
[info] ArrayFire initialized successfully
[info] OpenCL backend available
```

If you see "CPU backend" instead of "OpenCL", your GPU drivers may need updating.

### Issue: Node registration fails

**Symptoms:**
```
[error] Failed to register with Central Server
[warn] Server Node will run in standalone mode
```

**Causes:**
1. Central Server not running
2. Network firewall blocking connection
3. gRPC client connection failure

**Solutions:**
1. Verify Central Server is running on port 50051
2. Check firewall settings
3. Try connecting to `127.0.0.1:50051` instead of `localhost:50051`

## Performance Monitoring

### Central Server

**Check registered nodes:**
```bash
curl http://localhost:8080/api/nodes
```

**Check active jobs:**
```bash
curl http://localhost:8080/api/jobs
```

### Server Node

**Check active jobs:**
Watch the console output for:
```
[info]   Active jobs:          N
```

**Monitor job progress:**
```
[info] Job job_123 progress: 50.0% - Epoch 5/10, Loss: 0.5234
```

## Scripts Reference

### start-central-server.bat

- Checks if Central Server is already running
- Starts Central Server on ports 50051 (gRPC) and 8080 (REST)
- Shows startup logs
- Press Ctrl+C to stop

### start-server-node.bat

- Checks if Server Node is already running
- Warns if Central Server is not running
- Starts Server Node on ports 50052, 50053, 50054
- Shows startup logs
- Press Ctrl+C to stop

### stop-all.bat

- Finds all CyxWiz processes
- Kills Central Server (PID on port 50051)
- Kills Server Node (PID on port 50052)
- Cleans up port bindings

## Next Steps

Once both services are running:

1. **Verify Connection:**
   - Check Server Node logs for "Successfully registered with Central Server"
   - Check Central Server logs for "Node <ID> registered successfully"

2. **Submit Test Job:**
   - Use Python gRPC client (see `PHASE5_3_COMPLETE.md` for example)
   - Or use `grpcurl` to test job submission

3. **Monitor Execution:**
   - Watch Server Node logs for job progress
   - Check Central Server for job status updates

---

## Quick Reference Commands

```bash
# Start services
start-central-server.bat              # Terminal 1
start-server-node.bat                 # Terminal 2

# Stop services
stop-all.bat                          # Or Ctrl+C in each terminal

# Check status
netstat -ano | findstr ":50051"       # Central Server
netstat -ano | findstr ":50052"       # Server Node

# Health check
curl http://localhost:8080/api/health

# View logs
# Just watch the console output
```

---

For more information, see:
- `PHASE5_3_COMPLETE.md` - Implementation details
- `PHASE5_SESSION_SUMMARY.md` - Development progress
- `README.md` - Project overview
