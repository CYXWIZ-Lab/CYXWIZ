# Testing Node Disconnection Detection

## Overview
This guide explains how to test the NodeMonitor feature that detects disconnected Server Nodes and logs the disconnection events.

## What Was Implemented

The NodeMonitor (cyxwiz-central-server/src/scheduler/node_monitor.rs) runs in the background and:
- Checks every **10 seconds** for nodes that haven't sent heartbeats
- Marks nodes as **Offline** if no heartbeat received for **30 seconds**
- Logs disconnection events with node ID, name, and time since last heartbeat

## Test Procedure

### Step 1: Start the Central Server

```bash
cd D:\Dev\CyxWiz_Claude
.\start-central-server.bat
```

**Expected log messages:**
```
INFO  cyxwiz_central_server: CyxWiz Central Server v0.1.0
INFO  cyxwiz_central_server: ========================================
INFO  cyxwiz_central_server: Loading configuration...
INFO  cyxwiz_central_server: Connecting to database: sqlite://cyxwiz.db
INFO  cyxwiz_central_server: Running database migrations...
INFO  cyxwiz_central_server: Migrations completed
INFO  cyxwiz_central_server: ✓ Redis connected successfully
INFO  cyxwiz_central_server: Starting job scheduler...
INFO  cyxwiz_central_server: Job scheduler started
INFO  cyxwiz_central_server: Starting node monitor...         <-- NodeMonitor starting
INFO  cyxwiz_central_server: Node monitor started             <-- NodeMonitor ready
INFO  cyxwiz_central_server: Starting gRPC server on 0.0.0.0:50051
INFO  cyxwiz_central_server: 🚀 gRPC Server ready!
```

### Step 2: Register a Test Node

Use one of the following methods:

**Option A: Using Python test script (if available)**
```bash
cd D:\Dev\CyxWiz_Claude
python test_node_disconnection.py
```

**Option B: Start a Server Node**
```bash
cd D:\Dev\CyxWiz_Claude
.\start-server-node.bat
```

**Option C: Use the insert script**
```bash
cd D:\Dev\CyxWiz_Claude\cyxwiz-central-server
python insert_test_data.py
```

###Step 3: Verify Node is Registered

You should see a log message in the Central Server:
```
INFO  cyxwiz_central_server::api::grpc::node_service: Node registered: [node_id] (TestNode-DisconnectionTest)
```

### Step 4: Stop Sending Heartbeats

- If using a Server Node: **Stop the server node process** (Ctrl+C)
- If using Python script: **The script will automatically stop** sending heartbeats after 20 seconds

### Step 5: Watch for Disconnection Detection

**Within 40 seconds**, you should see these log messages in the Central Server:

```
WARN  cyxwiz_central_server::scheduler::node_monitor: Node [node_id] (TestNode-DisconnectionTest) disconnected - no heartbeat for 32 seconds
INFO  cyxwiz_central_server::scheduler::node_monitor: Node [node_id] (TestNode-DisconnectionTest) marked as OFFLINE
```

**Timeline:**
- T+0s: Node stops sending heartbeats
- T+10s: NodeMonitor checks, node still within 30s timeout
- T+20s: NodeMonitor checks, node still within 30s timeout
- T+30s: NodeMonitor checks, node NOW beyond 30s timeout → **DISCONNECTION DETECTED**
- Disconnection is logged immediately

## Configuration

The NodeMonitor timeouts are configured in **cyxwiz-central-server/src/main.rs:103-106**:

```rust
let node_monitor = NodeMonitor::new(
    db_pool.clone(),
    30,  // 30 seconds timeout - mark offline if no heartbeat
    10   // Check every 10 seconds
);
```

To change the behavior:
- **Decrease timeout** (e.g., 15) for faster detection
- **Increase check interval** (e.g., 20) to reduce database queries

## Expected Log Output (Full Example)

```
[20:23:49] INFO  CyxWiz Central Server v0.1.0
[20:23:49] INFO  Starting node monitor...
[20:23:49] INFO  Node monitor started (timeout: 30s, check interval: 10s)
[20:23:49] INFO  🚀 gRPC Server ready!

[20:24:10] INFO  Node registered: abc123 (TestNode-DisconnectionTest)
[20:24:15] DEBUG Heartbeat received from node abc123
[20:24:20] DEBUG Heartbeat received from node abc123
[20:24:25] DEBUG Heartbeat received from node abc123
[20:24:30] DEBUG Heartbeat received from node abc123

... [Node stops sending heartbeats] ...

[20:25:00] DEBUG NodeMonitor: Checking nodes (0 offline detected)
[20:25:10] DEBUG NodeMonitor: Checking nodes (0 offline detected)
[20:25:20] WARN  Node abc123 (TestNode-DisconnectionTest) disconnected - no heartbeat for 32 seconds
[20:25:20] INFO  Node abc123 (TestNode-DisconnectionTest) marked as OFFLINE
```

## Troubleshooting

**Issue: "Node monitor started" message not appearing**
- The NodeMonitor code may not be integrated in main.rs
- Check that lines 101-111 in main.rs contain the NodeMonitor initialization

**Issue: No disconnection detected**
- Verify the node was actually registered (check `nodes` table in database)
- Ensure the node stopped sending heartbeats
- Wait at least 40 seconds (10s check interval + 30s timeout + buffer)

**Issue: Database errors**
- Run migrations: `cd cyxwiz-central-server && sqlx database reset -y`
- Or delete database and restart: `rm cyxwiz.db && cargo run --release`

## Verification Queries

Check node status directly in the database:

```sql
-- View all nodes and their last heartbeat
SELECT id, name, status, last_heartbeat,
       julianday('now') - julianday(last_heartbeat) as hours_since_heartbeat
FROM nodes;

-- Count nodes by status
SELECT status, COUNT(*) as count
FROM nodes
GROUP BY status;
```

## Success Criteria

✅ NodeMonitor starts successfully with Central Server
✅ Registered nodes appear in database with status 'online'
✅ Nodes that stop sending heartbeats are detected within 40 seconds
✅ Disconnection events are logged with WARN level
✅ Node status is updated to 'offline' in database
✅ Confirmation message logged at INFO level

## Implementation Details

**Files Modified:**
- `cyxwiz-central-server/src/scheduler/node_monitor.rs` - NodeMonitor implementation
- `cyxwiz-central-server/src/database/queries.rs` - Added `get_all_online_nodes()` query
- `cyxwiz-central-server/src/scheduler/mod.rs` - Exported NodeMonitor module
- `cyxwiz-central-server/src/main.rs:101-111` - NodeMonitor initialization and spawning

**Key Functions:**
- `NodeMonitor::new()` - Create monitor with timeout and check interval
- `NodeMonitor::run()` - Background loop that checks nodes every N seconds
- `NodeMonitor::check_disconnected_nodes()` - Query online nodes and mark stale ones as offline
- `queries::get_all_online_nodes()` - Fetch all nodes with status='online'
- `queries::update_node_status()` - Update node status to 'offline'
