# Node Endpoint Registration Fix

## Problem
Two server nodes running on different PCs were registering with the same address in the Central Server, making them indistinguishable.

## Root Cause
1. The `Node` database model did not store IP address and port information
2. Nodes without wallet addresses (empty string "") could not be distinguished from each other
3. Duplicate detection only checked wallet_address, which was empty for test nodes

## Solution Implemented

### 1. Database Model Updates
**File**: `cyxwiz-central-server/src/database/models.rs`
- Added `ip_address: String` field to Node struct
- Added `port: i32` field to Node struct

### 2. Database Migration
**File**: `cyxwiz-central-server/migrations/20250118000001_add_node_endpoint.sql`
- Created migration to add `ip_address` and `port` columns to nodes table
- Added index on `(ip_address, port)` for efficient lookups
- Set defaults: `ip_address = '0.0.0.0'`, `port = 50052`

### 3. Registration Logic Updates
**File**: `cyxwiz-central-server/src/api/grpc/node_service.rs`

**Improved Duplicate Detection:**
- **Primary check** (for authenticated nodes): Check by `wallet_address`
  - If found, update the IP:port endpoint (handles IP changes)
- **Fallback check** (for unauthenticated nodes): Check by `ip_address:port` combination
  - Prevents duplicate registration of nodes from different IPs
- **Store endpoint**: Now saves IP address and port during node registration

### 4. Database Query Functions
**File**: `cyxwiz-central-server/src/database/queries.rs`

**New Functions:**
```rust
// Find node by IP:port endpoint
pub async fn get_node_by_endpoint(pool: &DbPool, ip_address: &str, port: i32) -> Result<Option<Node>>

// Update node's IP and port
pub async fn update_node_endpoint(pool: &DbPool, node_id: &DbId, ip_address: &str, port: i32) -> Result<()>
```

**Updated:**
- `create_node()` - Now includes ip_address and port in INSERT statement

### 5. Server Node Size_t Warnings Fixed
**Files**:
- `cyxwiz-backend/src/core/tensor.cpp`
- `cyxwiz-backend/src/core/device.cpp`

**Changes:**
- Added explicit `static_cast<unsigned int>()` for all ArrayFire dim4 array indexing
- Added explicit `static_cast<size_t>()` for memory calculations involving floating point

## How It Works Now

### Scenario 1: Node with Wallet Address
```
1. Node registers with wallet_address="abc123", ip="192.168.1.10", port=50052
2. Central Server checks: Does wallet "abc123" exist?
3. If YES → Update existing node's IP to 192.168.1.10:50052
4. If NO → Create new node entry
```

### Scenario 2: Node without Wallet Address (Test Mode)
```
1. Node registers with wallet_address="", ip="192.168.1.10", port=50052
2. Central Server checks: Does IP 192.168.1.10:50052 exist?
3. If YES → Return existing node ID
4. If NO → Create new node entry
```

## Benefits
✅ **Unique Identification**: Each node from a different IP is uniquely registered
✅ **Dynamic IP Handling**: Nodes with wallets can update their IP addresses
✅ **Test-Friendly**: Nodes without wallets can still be distinguished by IP:port
✅ **No Duplicate Nodes**: Two nodes on different PCs will never share the same registration

## Migration Steps

1. **Backup database** (done automatically):
   ```bash
   cp cyxwiz-central-server/cyxwiz.db cyxwiz-central-server/cyxwiz.db.backup
   ```

2. **Restart Central Server** - Migration runs automatically on startup:
   ```bash
   cd cyxwiz-central-server
   cargo run --release
   ```

3. **Restart Server Nodes** - They will re-register with their IP addresses:
   ```bash
   build/bin/Release/cyxwiz-server-node.exe
   ```

## Testing
Run two server nodes from different machines:
- **PC 1**: 192.168.1.10 → Registers as Node A
- **PC 2**: 192.168.1.20 → Registers as Node B
- Both will have unique entries in the database with distinct IP addresses

## Database Schema After Migration
```sql
CREATE TABLE nodes (
    id TEXT PRIMARY KEY,
    wallet_address TEXT NOT NULL,  -- UNIQUE removed (now handled by partial index)
    name TEXT NOT NULL,
    ...
    ip_address TEXT NOT NULL DEFAULT '0.0.0.0',  -- NEW
    port INTEGER NOT NULL DEFAULT 50052,           -- NEW
    ...
);

-- Indexes
CREATE INDEX idx_nodes_endpoint ON nodes(ip_address, port);  -- NEW

-- Partial unique index: Only enforce uniqueness for NON-EMPTY wallet addresses
CREATE UNIQUE INDEX idx_nodes_wallet_unique ON nodes(wallet_address) WHERE wallet_address != '';  -- NEW
```

## Fix for Error 2067 (UNIQUE Constraint Violation)

If you get error code 2067 when registering a second node, it means the old UNIQUE constraint is still active.

**Migration 20250118000002** fixes this by:
1. Removing the inline UNIQUE constraint from `wallet_address`
2. Adding a **partial unique index** that only enforces uniqueness on non-empty wallet addresses
3. This allows multiple test nodes with `wallet_address=""` while preventing duplicate real wallet addresses

## Files Changed
1. ✅ `cyxwiz-central-server/src/database/models.rs`
2. ✅ `cyxwiz-central-server/src/api/grpc/node_service.rs`
3. ✅ `cyxwiz-central-server/src/database/queries.rs`
4. ✅ `cyxwiz-central-server/migrations/20250118000001_add_node_endpoint.sql`
5. ✅ `cyxwiz-central-server/migrations/20250118000002_fix_wallet_unique_constraint.sql` **(NEW - fixes error 2067)**
6. ✅ `cyxwiz-backend/src/core/tensor.cpp`
7. ✅ `cyxwiz-backend/src/core/device.cpp`

## Compile Status
- ✅ Central Server: **Built successfully** (Release mode)
- ✅ Server Node: **Built successfully** (Release mode)
- ✅ Warnings: Only unused imports (non-critical)

## Migrations Applied
1. `20250118000001_add_node_endpoint.sql` - Adds ip_address and port columns
2. `20250118000002_fix_wallet_unique_constraint.sql` - Fixes UNIQUE constraint to allow multiple empty wallet addresses
