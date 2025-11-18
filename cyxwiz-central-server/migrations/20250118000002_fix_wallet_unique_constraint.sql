-- Fix UNIQUE constraint on wallet_address to allow multiple empty strings
-- SQLite doesn't support dropping constraints, so we need to recreate the table

-- Step 1: Create new table without UNIQUE constraint on wallet_address
CREATE TABLE IF NOT EXISTS nodes_new (
    id TEXT PRIMARY KEY,
    wallet_address TEXT NOT NULL,  -- Removed UNIQUE constraint
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'offline' CHECK(status IN ('online', 'offline', 'busy', 'maintenance')),
    reputation_score REAL NOT NULL DEFAULT 0.5,
    stake_amount INTEGER NOT NULL DEFAULT 0,

    -- Capabilities
    cpu_cores INTEGER NOT NULL,
    ram_gb INTEGER NOT NULL,
    gpu_model TEXT,
    gpu_memory_gb INTEGER,
    has_cuda INTEGER NOT NULL DEFAULT 0,
    has_opencl INTEGER NOT NULL DEFAULT 0,

    -- Metrics
    total_jobs_completed INTEGER NOT NULL DEFAULT 0,
    total_jobs_failed INTEGER NOT NULL DEFAULT 0,
    uptime_percentage REAL NOT NULL DEFAULT 100.0,
    current_load REAL NOT NULL DEFAULT 0.0,

    -- Location
    country TEXT,
    region TEXT,

    -- Network endpoint
    ip_address TEXT NOT NULL DEFAULT '0.0.0.0',
    port INTEGER NOT NULL DEFAULT 50052,

    last_heartbeat TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    registered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Step 2: Copy existing data
INSERT INTO nodes_new
SELECT * FROM nodes;

-- Step 3: Drop old table
DROP TABLE nodes;

-- Step 4: Rename new table
ALTER TABLE nodes_new RENAME TO nodes;

-- Step 5: Recreate indexes
CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status);
CREATE INDEX IF NOT EXISTS idx_nodes_reputation ON nodes(reputation_score DESC);
CREATE INDEX IF NOT EXISTS idx_nodes_endpoint ON nodes(ip_address, port);

-- Step 6: Create partial unique index - only enforce uniqueness for NON-EMPTY wallet addresses
-- This allows multiple nodes with wallet_address='' but prevents duplicate real wallet addresses
CREATE UNIQUE INDEX IF NOT EXISTS idx_nodes_wallet_unique
ON nodes(wallet_address)
WHERE wallet_address != '';

-- Step 7: Create regular index for lookup on all wallet addresses (including empty)
CREATE INDEX IF NOT EXISTS idx_nodes_wallet ON nodes(wallet_address);
