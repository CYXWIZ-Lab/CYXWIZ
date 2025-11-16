# gRPC Server Enablement Guide

This guide documents the steps required to enable the full gRPC server in the CyxWiz Central Server.

## Current Status

The Central Server currently runs in **TUI-only mode** by default. The full gRPC/REST server implementation exists but is **partially disabled** due to compilation issues.

**Working components:**
- ✅ Database layer (migrations, queries, models)
- ✅ Job scheduler with intelligent node matching
- ✅ Redis cache integration (with mock fallback)
- ✅ TUI dashboard (Ratatui)
- ✅ All gRPC service implementations (deployment, job, node, terminal, model)
- ✅ Blockchain/Solana client stub

**Issues preventing full gRPC enablement:**
- ❌ Protobuf module (`pb`) not accessible across all service files
- ❌ Some service files reference undefined types
- ❌ Build system needs restructuring for dual-mode operation

## Architecture Overview

```
cyxwiz-central-server/
├── src/
│   ├── main.rs          # Entry point with mode selection
│   ├── api/
│   │   ├── grpc/        # gRPC service implementations
│   │   │   ├── job_service.rs
│   │   │   ├── node_service.rs
│   │   │   ├── deployment_service.rs
│   │   │   ├── terminal_service.rs
│   │   │   └── model_service.rs
│   │   └── rest/        # REST API (Axum)
│   ├── blockchain/      # Solana client & payment processor
│   ├── cache/           # Redis cache (with mock mode)
│   ├── database/        # SQLite/PostgreSQL + migrations
│   ├── scheduler/       # Job scheduler & matcher
│   └── tui/             # Terminal dashboard
└── build.rs             # Protobuf code generation
```

## Changes Made

### 1. Mode Selection in main.rs

Added command-line argument parsing to support two modes:

```rust
// Default: TUI mode
cargo run

// Server mode with gRPC/REST
cargo run -- --server
```

### 2. Enabled Modules

Uncommented the following modules in `src/main.rs`:

```rust
mod api;           // gRPC + REST API
mod blockchain;    // Solana integration
mod scheduler;     // Job scheduling
```

### 3. Protobuf Module Definition

```rust
pub mod pb {
    tonic::include_proto!("cyxwiz.protocol");
}
```

## Remaining Compilation Errors

### Error 1: Protobuf Module Not Accessible

**Issue:** `pb::` module defined in `main.rs` is not accessible from service files.

**Affected files:**
- `src/api/grpc/job_service.rs` (line 45, 155, 199, 206, ...)
- `src/api/grpc/node_service.rs`
- `src/api/grpc/deployment_service.rs`
- `src/api/grpc/terminal_service.rs`
- `src/api/grpc/model_service.rs`

**Current pattern:**
```rust
// In service files
use crate::pb::{...};  // ❌ pb not in scope

type StreamJobUpdatesStream = tokio_stream::wrappers::ReceiverStream<
    std::result::Result<pb::JobUpdateStream, tonic::Status>
>;
```

**Solution options:**

**Option A: Move pb to separate module**
```rust
// Create src/pb.rs
pub mod pb {
    tonic::include_proto!("cyxwiz.protocol");
}

// In main.rs
mod pb;
pub use pb::pb;

// In service files
use crate::pb::{...};  // ✅ Now works
```

**Option B: Re-export from main**
```rust
// In main.rs
pub mod pb {
    tonic::include_proto!("cyxwiz.protocol");
}

// Service files already import from crate::pb, so this should work
// But currently doesn't due to visibility rules
```

**Option C: Local pb module in each service file**
```rust
// In each service file
mod pb {
    include!(concat!(env!("OUT_DIR"), "/cyxwiz.protocol.rs"));
}
```

### Error 2: JobServiceImpl Trait Confusion

**Issue:** Rust is confusing the impl block signature.

**Current code:**
```rust
impl crate::pb::job_service_server::JobServiceImpl {  // ❌ Wrong type
    pub fn new(...) -> Self { ... }
}
```

**Should be:**
```rust
impl JobServiceImpl {  // ✅ Struct name
    pub fn new(...) -> Self { ... }
}

#[tonic::async_trait]
impl crate::pb::job_service_server::JobService for JobServiceImpl {  // ✅ Trait
    // ...
}
```

### Error 3: Duplicate `#[tonic::async_trait]` Annotations

Found in `job_service.rs:42-43`:
```rust
#[tonic::async_trait]
#[tonic::async_trait]  // ❌ Duplicate
impl crate::pb::job_service_server::JobService for JobServiceImpl {
```

**Fix:** Remove one annotation.

## Quick Start (Automated)

We've created helper scripts to assist with Phase 1 fixes:

**Linux/macOS:**
```bash
cd cyxwiz-central-server
chmod +x fix_grpc.sh
./fix_grpc.sh
```

**Windows:**
```bash
cd cyxwiz-central-server
fix_grpc.bat
```

These scripts will guide you through the manual edits needed. They include:
- Backing up main.rs
- Notes on what to edit in each file
- Instructions for enabling modules

**Note:** These scripts provide guidance but require manual editing. Full automation is not possible due to the complexity of the changes.

---

## Step-by-Step Fix Procedure (Manual)

### Phase 1: Fix Protobuf Module Access

1. **Create `src/pb.rs` (Already Done):**
```rust
//! Generated protobuf code module

pub use pb_inner::*;

mod pb_inner {
    tonic::include_proto!("cyxwiz.protocol");
}
```

2. **Update `src/main.rs`:**
```rust
mod pb;  // Add this line

// Remove old pb definition:
// pub mod pb { ... }
```

3. **Verify all service files can import:**
```rust
use crate::pb::{...};
```

### Phase 2: Fix Service Implementations

For **each** service file (`job_service.rs`, `node_service.rs`, etc.):

1. **Fix struct impl blocks:**
```rust
// OLD (wrong):
impl crate::pb::xxx_service_server::XxxServiceImpl {
    pub fn new(...) -> Self { ... }
}

// NEW (correct):
impl XxxServiceImpl {
    pub fn new(...) -> Self { ... }
}
```

2. **Remove duplicate async_trait annotations**

3. **Fix pb:: references:**
```rust
// Ensure all pb:: types are imported at the top
use crate::pb::{
    Type1, Type2, ...
};
```

### Phase 3: Fix Blockchain/Solana Errors

The `SolanaClient` may have API mismatches:

1. Check `src/blockchain/solana.rs` for compilation errors
2. Update Solana SDK calls to match current version
3. Handle keypair file loading properly

### Phase 4: Build and Test

```bash
# Try building with fixes
cd cyxwiz-central-server
cargo build --release 2>&1 | tee build.log

# If successful, test TUI mode
cargo run

# Then test server mode
cargo run -- --server
```

### Phase 5: Integration Testing

Once compilation succeeds:

1. **Start Central Server in server mode:**
```bash
cargo run -- --server
```

2. **Verify gRPC endpoints are listening:**
```bash
# Should see:
# gRPC server on 0.0.0.0:50051
# REST API on 0.0.0.0:8080
```

3. **Test with Server Node:**
```bash
cd ../build/windows-release/bin/Release
./cyxwiz-server-node.exe
```

4. **Verify node registration:**
```bash
# Central Server logs should show:
# "Node registered: <node_id>"
```

## Current Workaround

Until all compilation issues are resolved, use **TUI-only mode**:

```bash
cargo run  # No --server flag
```

This mode provides:
- Database connectivity
- Redis cache (or mock)
- Visual dashboard with network stats
- No gRPC/REST endpoints (Server Nodes will run in standalone mode)

## Testing Plan

Once gRPC is enabled, test the following workflows:

### 1. Node Registration
- Server Node connects to Central Server
- Hardware capabilities are detected and sent
- Node appears in Central Server TUI dashboard

### 2. Job Submission
- Engine submits job via gRPC
- Central Server creates job record
- Scheduler assigns job to best node
- Node receives job via gRPC callback

### 3. Job Execution
- Node executes model training
- Progress updates stream back to Central Server
- Central Server relays updates to Engine

### 4. Terminal Streaming
- Engine requests terminal access to Node
- Central Server brokers bidirectional stream
- Commands execute on Node, output streams back

### 5. Model Upload/Download
- Engine uploads model to Central Server
- Central Server stores in `./storage/models`
- Node downloads model for deployment

## Performance Considerations

With gRPC enabled:

- **Memory:** Scheduler + multiple streaming connections = ~200-300 MB baseline
- **CPU:** Job matching algorithm runs every 100ms (configurable)
- **Network:** gRPC uses HTTP/2 multiplexing (efficient for many connections)
- **Database:** SQLite may bottleneck at ~50 concurrent nodes (consider PostgreSQL)

## Configuration

Edit `config.toml` to adjust server settings:

```toml
[server]
grpc_address = "0.0.0.0:50051"
rest_address = "0.0.0.0:8080"

[scheduler]
job_poll_interval_ms = 100
max_retries = 3
node_heartbeat_timeout_ms = 30000

[database]
url = "sqlite://cyxwiz.db"  # Or PostgreSQL

[blockchain]
network = "devnet"
solana_rpc_url = "https://api.devnet.solana.com"
```

## Debugging Tips

### View Compilation Errors in Detail
```bash
cargo build 2>&1 | tee errors.txt
# Open errors.txt to see all errors
```

### Enable Debug Logging
```bash
RUST_LOG=debug cargo run -- --server
```

### Check Generated Protobuf Code
```bash
# Generated files are in target/release/build/cyxwiz-central-server-*/out/
ls target/release/build/cyxwiz-central-server-*/out/
cat target/release/build/cyxwiz-central-server-*/out/cyxwiz.protocol.rs
```

### Test gRPC Endpoint Manually
```bash
# Install grpcurl
go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest

# List services
grpcurl -plaintext localhost:50051 list

# Call a method
grpcurl -plaintext -d '{"node_id": "test"}' \
  localhost:50051 cyxwiz.protocol.NodeService/Heartbeat
```

## Success Criteria

gRPC enablement is complete when:

1. ✅ `cargo build --release` succeeds with no errors
2. ✅ `cargo run -- --server` starts without panics
3. ✅ gRPC server listens on port 50051
4. ✅ REST API listens on port 8080
5. ✅ Server Node can register with Central Server
6. ✅ Jobs can be submitted and assigned to nodes
7. ✅ Terminal streaming works end-to-end
8. ✅ TUI mode still works with `cargo run` (no --server)

## Next Steps After gRPC Works

1. **Blockchain Integration** - Connect real Solana payment processing
2. **Load Testing** - Test with 100+ simulated nodes
3. **Security** - Add TLS encryption for gRPC
4. **Authentication** - Implement JWT tokens for API
5. **Monitoring** - Add Prometheus metrics export
6. **Docker** - Containerize for easy deployment

## Questions or Issues?

- Check `cyxwiz-central-server/README.md` for architecture details
- Review protobuf definitions in `cyxwiz-protocol/proto/`
- See Server Node implementation in `cyxwiz-server-node/`
- Consult Engine GUI implementation in `cyxwiz-engine/`

---

**Last Updated:** 2025-01-13
**Status:** Compilation errors preventing gRPC server mode
**Priority:** High - Required for distributed training network
