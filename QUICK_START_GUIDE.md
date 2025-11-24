# CyxWiz Quick Start Guide

## Testing the JobStatusPanel GUI

This guide will help you start the Central Server and test the JobStatusPanel visualization.

### Prerequisites

1. **Central Server** (Rust/Cargo installed)
2. **CyxWiz Engine** (C++/CMake built)
3. **Python 3.8+** with grpcio

### Step 1: Start the Central Server

**First Time Setup:**

If this is your first time starting the server or if you don't have a `cyxwiz.db` file:

```bash
cd cyxwiz-central-server

# Check if database exists
ls cyxwiz.db*

# If cyxwiz.db doesn't exist but cyxwiz.db.backup does:
cp cyxwiz.db.backup cyxwiz.db

# OR if no database exists at all, create empty file:
# Windows:
type nul > cyxwiz.db
# Linux/Mac:
touch cyxwiz.db

# Now start the server
cargo run --release
```

**Normal Startup (database already exists):**

```bash
cd cyxwiz-central-server
cargo run --release
```

**Expected output:**
```
INFO CyxWiz Central Server v0.1.0
INFO Loading configuration...
INFO Connecting to database: sqlite://cyxwiz.db
INFO Running database migrations...
INFO Migrations completed
INFO ✓ Redis connected successfully
INFO Starting job scheduler...
INFO Job scheduler started
INFO Starting node monitor...
INFO Node monitor started
INFO Initializing Solana client...
ERROR Solana keypair file not found: ~/.config/solana/id.json  (⬅ Expected for dev)
ERROR Payment processing will be disabled  (⬅ Expected for dev)
INFO Initializing JWT manager...
INFO JWT manager initialized (token expiration: 3600s)
INFO Starting gRPC server on 0.0.0.0:50051
INFO 🚀 gRPC Server ready!
INFO    gRPC endpoint: 0.0.0.0:50051
INFO    JobService: ENABLED
INFO    NodeService: ENABLED
INFO    JobStatusService: ENABLED
INFO    REST API: ENABLED
```

**Important:** Keep this terminal open. The server must keep running while you test the GUI.

### Step 2: Launch the Engine GUI

```bash
# From project root
cd build/bin/Release
./cyxwiz-engine.exe
```

The Engine will open with the JobStatusPanel visible in the main window.

### Step 3: Submit a Test Job

In a new terminal:

```bash
# From project root
python test_gui_simple.py
```

**Expected output:**
```
============================================================
 JobStatusPanel GUI Test
============================================================

This script will submit a P2P job to the Central Server.
Watch the Engine GUI's 'Job Status & Orchestration' panel!

[1/4] Connecting to Central Server...
      Connected to localhost:50051

[2/4] Creating P2P training job...

[3/4] Submitting job to Central Server...
      Job submitted successfully!
      Job ID: 5f3e8a2b-...

>>> CHECK THE ENGINE GUI NOW <<<
    The job should appear in the left panel!
```

### Step 4: Watch the GUI Update

In the Engine GUI, look for the **"Job Status & Orchestration"** panel:

**Left Panel (Job List):**
- Job appears with status color:
  - 🟡 Yellow = PENDING (waiting for node assignment)
  - 🔵 Blue = IN PROGRESS
  - 🟢 Green = COMPLETED
  - 🔴 Red = FAILED/ERROR

**Right Panel (Job Details):**
- Job ID
- Status and progress
- **P2P Orchestration Status** section shows:
  - "Waiting for node assignment..." (initially)
  - When assigned: "Node Assignment Received!"
    - Node endpoint
    - JWT authentication token (truncated, hover for full token)
  - P2P connection status

**Auto-Refresh:** The panel updates every second automatically.

### Troubleshooting

#### Central Server Database Error

**Problem:** `Error: Database(SqliteError { code: 14, message: "unable to open database file" })`

This error occurs when the SQLite database file `cyxwiz.db` is missing or inaccessible.

**Root Cause:**
- The database file doesn't exist yet
- SQLite migrations try to run before the database file is created
- File permissions issue (rare on Windows)

**Solution 1: Restore from Backup (if available)**

If you have a `cyxwiz.db.backup` file:

```bash
cd cyxwiz-central-server
cp cyxwiz.db.backup cyxwiz.db
cargo run --release
```

**Solution 2: Create Fresh Database**

If no backup exists, create an empty database and let migrations populate it:

```bash
cd cyxwiz-central-server

# Create empty database file (Windows)
type nul > cyxwiz.db

# OR on Linux/Mac:
touch cyxwiz.db

# Now run the server
cargo run --release
```

**Solution 3: Check Configuration**

Ensure `config.toml` has the correct database URL:

```toml
[database]
url = "sqlite://cyxwiz.db"  # Relative path, runs in cyxwiz-central-server/
```

**Verification:**

After fixing, you should see:

```
INFO Connecting to database: sqlite://cyxwiz.db
INFO Running database migrations...
INFO Migrations completed
INFO 🚀 gRPC Server ready!
```

Check the server is listening:

```bash
# Windows
netstat -an | findstr "50051"

# Linux/Mac
netstat -an | grep 50051
```

Expected output: `TCP    0.0.0.0:50051          0.0.0.0:0              LISTENING`

#### Missing JWT Configuration

**Problem:**
```
ERROR Failed to load config: missing field `jwt`, using defaults
INFO Connecting to database: postgres://cyxwiz:cyxwiz@localhost/cyxwiz
```

**Root Cause:** The `config.toml` file is missing the `[jwt]` section, causing the server to use default config with PostgreSQL.

**Solution:**

Add the JWT section to `cyxwiz-central-server/config.toml`:

```toml
[jwt]
secret = "cyxwiz_development_secret_key_change_in_production"
p2p_token_expiration_seconds = 3600
```

The complete config should include:
- `[server]` - gRPC and REST addresses
- `[database]` - SQLite URL
- `[redis]` - Cache settings
- `[scheduler]` - Job scheduling intervals
- `[blockchain]` - Solana settings
- `[jwt]` - Authentication (REQUIRED)

#### Solana Payment Processing Disabled

**Warning (Expected):**
```
ERROR Solana keypair file not found: ~/.config/solana/id.json
ERROR Payment processing will be disabled
```

**This is normal for development!**

The Central Server requires a Solana wallet keypair for blockchain payment processing. For local testing and development, you can safely ignore these warnings. The server will continue to operate with mock payment processing.

**To enable real payments (production only):**
1. Install Solana CLI: https://docs.solana.com/cli/install-solana-cli-tools
2. Generate keypair: `solana-keygen new`
3. Update `config.toml` with your keypair path
4. Fund the wallet with devnet SOL for testing

#### Redis Connection (Optional)

**Warning (Safe to ignore):**
```
ERROR Failed to connect to Redis: Connection refused
```

Redis is used for caching and rate limiting. The Central Server will work without Redis, but with reduced performance. For local testing, you can safely ignore this warning.

**To install Redis (optional):**

**Windows:**
1. Download from: https://github.com/tporadowski/redis/releases
2. Extract and run `redis-server.exe`

**Linux:**
```bash
sudo apt install redis-server
sudo systemctl start redis
```

**Mac:**
```bash
brew install redis
brew services start redis
```

After starting Redis, restart the Central Server to use caching.

#### Test script can't connect

**Problem:** `gRPC Error - StatusCode.UNAVAILABLE - failed to connect`

**Solution:**
1. Check Central Server is running: `netstat -an | findstr "50051"` (Windows) or `netstat -an | grep 50051` (Linux/Mac)
2. If nothing, the server isn't listening. Check server output for errors.
3. Verify you're in the `cyxwiz-central-server` directory when starting the server

#### No jobs appear in GUI

**Problem:** JobStatusPanel is empty.

**Solution:**
1. Make sure you clicked "Connect to Server" in the Engine toolbar first
2. The Engine's `job_manager_` might not be initialized
3. Try restarting the Engine

---

### Quick Troubleshooting Checklist

Before asking for help, verify:

1. ✅ **Database exists**: `cyxwiz-central-server/cyxwiz.db` file is present
2. ✅ **Config complete**: `config.toml` has all sections including `[jwt]`
3. ✅ **Server running**: `netstat -an | findstr "50051"` shows LISTENING
4. ✅ **In correct directory**: You're in `cyxwiz-central-server/` when starting server
5. ✅ **Port not blocked**: Firewall allows port 50051
6. ⚠️ **Redis warning**: Safe to ignore for local testing
7. ⚠️ **Solana warning**: Safe to ignore for local testing

**Common mistakes:**
- Running `cargo run` from project root instead of `cyxwiz-central-server/`
- Forgetting to restore/create `cyxwiz.db` file
- Missing `[jwt]` section in `config.toml`
- Not keeping the server terminal open while testing

### Configuration Files

**Central Server config:** `cyxwiz-central-server/config.toml`
```toml
[server]
grpc_address = "0.0.0.0:50051"

[database]
url = "sqlite://cyxwiz.db"

[jwt]
secret = "cyxwiz_development_secret_key_change_in_production"
p2p_token_expiration_seconds = 3600
```

### What the Test Does

`test_gui_simple.py` performs these steps:

1. Connects to Central Server on localhost:50051
2. Creates a JobConfig with training parameters
3. Submits the job via gRPC `SubmitJob()`
4. Polls `GetJobStatus()` every 2 seconds for 15 iterations
5. Checks if a NodeAssignment was received
6. Displays the assignment details (node endpoint, JWT token)

The Engine's JobStatusPanel monitors these jobs via the JobManager, which polls the Central Server.

### Next Steps

Once you see jobs appearing in the GUI:

1. **Register a Server Node** - Start a Server Node to actually execute jobs
2. **Watch the full P2P workflow** - See job assignment → P2P connection → training execution
3. **Add GUI job submission** - Submit jobs directly from the Engine GUI (coming soon!)

## Summary

✅ **JobStatusPanel** - Displays job status with real-time updates
✅ **Test Scripts** - `test_gui_simple.py` and `test_gui_visualization.py`
✅ **Configuration** - config.toml with JWT settings
🔲 **Server Node** - Not yet started (needed for actual job execution)
🔲 **GUI Submission** - Not yet implemented (use Python scripts for now)

Enjoy testing the P2P orchestration workflow!
