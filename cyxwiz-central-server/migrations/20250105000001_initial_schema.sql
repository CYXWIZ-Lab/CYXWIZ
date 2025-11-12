-- SQLite schema for CyxWiz Central Server
-- Note: SQLite doesn't support ENUMs, so we use TEXT with CHECK constraints

-- Nodes table
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    wallet_address TEXT NOT NULL UNIQUE,
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

    last_heartbeat TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    registered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status);
CREATE INDEX IF NOT EXISTS idx_nodes_wallet ON nodes(wallet_address);
CREATE INDEX IF NOT EXISTS idx_nodes_reputation ON nodes(reputation_score DESC);

-- Jobs table
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    user_wallet TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'assigned', 'running', 'completed', 'failed', 'cancelled')),
    job_type TEXT NOT NULL,

    -- Requirements
    required_gpu INTEGER NOT NULL DEFAULT 0,
    required_gpu_memory_gb INTEGER,
    required_ram_gb INTEGER NOT NULL,
    estimated_duration_seconds INTEGER NOT NULL,

    -- Cost
    estimated_cost INTEGER NOT NULL,
    actual_cost INTEGER,

    -- Assignment
    assigned_node_id TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,

    -- Results
    result_hash TEXT,
    error_message TEXT,

    -- Metadata (JSON)
    metadata TEXT NOT NULL DEFAULT '{}',

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (assigned_node_id) REFERENCES nodes(id)
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_wallet);
CREATE INDEX IF NOT EXISTS idx_jobs_node ON jobs(assigned_node_id);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);

-- Payments table
CREATE TABLE IF NOT EXISTS payments (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    node_id TEXT,
    user_wallet TEXT NOT NULL,
    node_wallet TEXT,

    amount INTEGER NOT NULL,
    platform_fee INTEGER NOT NULL,
    node_reward INTEGER NOT NULL,

    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'locked', 'streaming', 'completed', 'failed', 'refunded')),

    -- Blockchain
    escrow_tx_hash TEXT,
    completion_tx_hash TEXT,
    escrow_account TEXT,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    locked_at TEXT,
    completed_at TEXT,

    FOREIGN KEY (job_id) REFERENCES jobs(id),
    FOREIGN KEY (node_id) REFERENCES nodes(id)
);

CREATE INDEX IF NOT EXISTS idx_payments_job ON payments(job_id);
CREATE INDEX IF NOT EXISTS idx_payments_node ON payments(node_id);
CREATE INDEX IF NOT EXISTS idx_payments_status ON payments(status);

-- Node metrics (time-series data)
CREATE TABLE IF NOT EXISTS node_metrics (
    id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,

    cpu_usage_percent REAL NOT NULL,
    ram_usage_percent REAL NOT NULL,
    gpu_usage_percent REAL,
    gpu_memory_usage_percent REAL,

    network_rx_bytes INTEGER NOT NULL DEFAULT 0,
    network_tx_bytes INTEGER NOT NULL DEFAULT 0,

    active_jobs INTEGER NOT NULL DEFAULT 0,

    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_metrics_node ON node_metrics(node_id);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON node_metrics(timestamp DESC);
