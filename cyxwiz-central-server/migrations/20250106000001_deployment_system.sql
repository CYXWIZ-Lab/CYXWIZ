-- Migration: Add deployment system tables and extend nodes table (SQLite version)
-- Created: 2025-01-06
-- Description: Adds support for model deployments, terminal sessions, and model marketplace

-- ============================================================================
-- EXTEND NODES TABLE
-- ============================================================================

-- Add model deployment capabilities to nodes table
ALTER TABLE nodes ADD COLUMN supported_formats TEXT DEFAULT '[]';  -- JSON array as text
ALTER TABLE nodes ADD COLUMN max_model_size_bytes INTEGER DEFAULT 0;
ALTER TABLE nodes ADD COLUMN supports_terminal_access INTEGER DEFAULT 0;  -- Boolean as 0/1
ALTER TABLE nodes ADD COLUMN available_runtimes TEXT DEFAULT '[]';  -- JSON array as text

-- Add detailed GPU information
ALTER TABLE nodes ADD COLUMN gpu_vram_total_bytes INTEGER DEFAULT 0;
ALTER TABLE nodes ADD COLUMN gpu_vram_available_bytes INTEGER DEFAULT 0;
ALTER TABLE nodes ADD COLUMN gpu_driver_version TEXT;
ALTER TABLE nodes ADD COLUMN cuda_version TEXT;
ALTER TABLE nodes ADD COLUMN compute_capability REAL;
ALTER TABLE nodes ADD COLUMN pcie_generation INTEGER;
ALTER TABLE nodes ADD COLUMN pcie_lanes INTEGER;

-- ============================================================================
-- MODELS TABLE (Model Registry/Marketplace)
-- ============================================================================

CREATE TABLE models (
    id TEXT PRIMARY KEY,  -- UUID as TEXT in SQLite
    name TEXT NOT NULL,
    description TEXT,
    owner_user_id TEXT NOT NULL,

    -- Model details
    format TEXT NOT NULL CHECK(format IN ('onnx', 'gguf', 'pytorch', 'tensorflow', 'safetensors', 'tflite', 'torchscript')),
    source TEXT NOT NULL CHECK(source IN ('local', 'huggingface', 'cyxwiz_hub', 'url')),
    source_url TEXT,
    size_bytes INTEGER NOT NULL,

    -- Requirements
    min_vram_bytes INTEGER NOT NULL DEFAULT 0,
    min_ram_bytes INTEGER NOT NULL DEFAULT 0,
    min_cpu_cores INTEGER NOT NULL DEFAULT 1,
    required_device_type TEXT,
    gpu_preference TEXT,

    -- Marketplace
    is_public INTEGER NOT NULL DEFAULT 0,  -- Boolean as 0/1
    price_per_download INTEGER DEFAULT 0,
    download_count INTEGER NOT NULL DEFAULT 0,
    rating REAL DEFAULT 0.0,
    rating_count INTEGER NOT NULL DEFAULT 0,
    tags TEXT DEFAULT '[]',  -- JSON array as text

    -- Storage
    storage_path TEXT NOT NULL,
    checksum_sha256 TEXT NOT NULL,

    -- Metadata
    metadata TEXT NOT NULL DEFAULT '{}',  -- JSON as text

    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_models_owner ON models(owner_user_id);
CREATE INDEX idx_models_format ON models(format);
CREATE INDEX idx_models_public ON models(is_public) WHERE is_public = 1;
CREATE INDEX idx_models_rating ON models(rating DESC);
CREATE INDEX idx_models_created ON models(created_at DESC);

-- ============================================================================
-- DEPLOYMENTS TABLE
-- ============================================================================

CREATE TABLE deployments (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,

    -- Deployment configuration
    type TEXT NOT NULL CHECK(type IN ('local', 'network')),
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'provisioning', 'loading', 'ready', 'running', 'stopped', 'failed', 'terminated')),
    status_message TEXT,

    -- Network deployment specific
    assigned_node_id TEXT REFERENCES nodes(id) ON DELETE SET NULL,
    max_price_per_hour INTEGER,
    actual_hourly_rate INTEGER,
    preferred_region TEXT,

    -- Runtime configuration
    environment_vars TEXT NOT NULL DEFAULT '{}',  -- JSON as text
    runtime_params TEXT NOT NULL DEFAULT '{}',    -- JSON as text
    port INTEGER,
    enable_terminal INTEGER NOT NULL DEFAULT 0,   -- Boolean as 0/1

    -- Endpoints
    endpoint_url TEXT,
    terminal_endpoint TEXT,

    -- Payment (network deployments)
    payment_escrow_address TEXT,
    payment_escrow_tx_hash TEXT,
    total_cost INTEGER NOT NULL DEFAULT 0,

    -- Metrics
    uptime_seconds INTEGER NOT NULL DEFAULT 0,
    total_requests INTEGER NOT NULL DEFAULT 0,
    avg_latency_ms REAL NOT NULL DEFAULT 0.0,

    -- Timestamps
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    started_at TEXT,
    stopped_at TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_deployments_user ON deployments(user_id);
CREATE INDEX idx_deployments_model ON deployments(model_id);
CREATE INDEX idx_deployments_node ON deployments(assigned_node_id);
CREATE INDEX idx_deployments_status ON deployments(status);
CREATE INDEX idx_deployments_type ON deployments(type);
CREATE INDEX idx_deployments_created ON deployments(created_at DESC);

-- ============================================================================
-- TERMINAL SESSIONS TABLE
-- ============================================================================

CREATE TABLE terminal_sessions (
    id TEXT PRIMARY KEY,
    deployment_id TEXT NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,

    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'closed', 'error')),

    -- Terminal configuration
    rows INTEGER NOT NULL DEFAULT 24,
    cols INTEGER NOT NULL DEFAULT 80,

    -- Activity tracking
    last_activity TEXT NOT NULL DEFAULT (datetime('now')),
    data_sent_bytes INTEGER NOT NULL DEFAULT 0,
    data_received_bytes INTEGER NOT NULL DEFAULT 0,

    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    closed_at TEXT
);

CREATE INDEX idx_terminal_sessions_deployment ON terminal_sessions(deployment_id);
CREATE INDEX idx_terminal_sessions_user ON terminal_sessions(user_id);
CREATE INDEX idx_terminal_sessions_status ON terminal_sessions(status);
CREATE INDEX idx_terminal_sessions_activity ON terminal_sessions(last_activity DESC);

-- ============================================================================
-- DEPLOYMENT METRICS TABLE (Time-series data)
-- ============================================================================

CREATE TABLE deployment_metrics (
    id TEXT PRIMARY KEY,
    deployment_id TEXT NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,

    -- Resource usage
    cpu_usage_percent REAL NOT NULL,
    gpu_usage_percent REAL,
    memory_usage_bytes INTEGER NOT NULL,
    vram_usage_bytes INTEGER,

    -- Request metrics
    request_count INTEGER NOT NULL DEFAULT 0,
    avg_latency_ms REAL NOT NULL DEFAULT 0.0,
    throughput_rps REAL NOT NULL DEFAULT 0.0,

    -- Error tracking
    error_count INTEGER NOT NULL DEFAULT 0,

    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_deployment_metrics_deployment ON deployment_metrics(deployment_id);
CREATE INDEX idx_deployment_metrics_timestamp ON deployment_metrics(timestamp DESC);
CREATE INDEX idx_deployment_metrics_composite ON deployment_metrics(deployment_id, timestamp DESC);

-- ============================================================================
-- MODEL DOWNLOADS TABLE (Track downloads for analytics)
-- ============================================================================

CREATE TABLE model_downloads (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,

    -- Payment (if model has a price)
    payment_amount INTEGER NOT NULL DEFAULT 0,
    payment_tx_hash TEXT,

    downloaded_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_model_downloads_model ON model_downloads(model_id);
CREATE INDEX idx_model_downloads_user ON model_downloads(user_id);
CREATE INDEX idx_model_downloads_date ON model_downloads(downloaded_at DESC);

-- ============================================================================
-- MODEL RATINGS TABLE
-- ============================================================================

CREATE TABLE model_ratings (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,

    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    review TEXT,

    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),

    UNIQUE(model_id, user_id)
);

CREATE INDEX idx_model_ratings_model ON model_ratings(model_id);
CREATE INDEX idx_model_ratings_user ON model_ratings(user_id);
CREATE INDEX idx_model_ratings_rating ON model_ratings(rating DESC);

-- ============================================================================
-- TRIGGERS (SQLite version)
-- ============================================================================

-- Trigger to update models.rating when a new rating is added/updated
CREATE TRIGGER trigger_update_model_rating_insert
AFTER INSERT ON model_ratings
BEGIN
    UPDATE models
    SET
        rating = (SELECT AVG(rating) FROM model_ratings WHERE model_id = NEW.model_id),
        rating_count = (SELECT COUNT(*) FROM model_ratings WHERE model_id = NEW.model_id),
        updated_at = datetime('now')
    WHERE id = NEW.model_id;
END;

CREATE TRIGGER trigger_update_model_rating_update
AFTER UPDATE ON model_ratings
BEGIN
    UPDATE models
    SET
        rating = (SELECT AVG(rating) FROM model_ratings WHERE model_id = NEW.model_id),
        rating_count = (SELECT COUNT(*) FROM model_ratings WHERE model_id = NEW.model_id),
        updated_at = datetime('now')
    WHERE id = NEW.model_id;
END;

-- Trigger to update models.download_count
CREATE TRIGGER trigger_increment_download_count
AFTER INSERT ON model_downloads
BEGIN
    UPDATE models
    SET
        download_count = download_count + 1,
        updated_at = datetime('now')
    WHERE id = NEW.model_id;
END;

-- Trigger to update deployment uptime and timestamps
CREATE TRIGGER trigger_update_deployment_uptime
AFTER UPDATE ON deployments
WHEN NEW.status != OLD.status
BEGIN
    UPDATE deployments
    SET
        started_at = CASE
            WHEN NEW.status = 'running' AND OLD.status != 'running' THEN datetime('now')
            ELSE started_at
        END,
        stopped_at = CASE
            WHEN NEW.status IN ('stopped', 'failed', 'terminated') AND OLD.status = 'running' THEN datetime('now')
            ELSE stopped_at
        END,
        uptime_seconds = CASE
            WHEN NEW.status IN ('stopped', 'failed', 'terminated') AND OLD.status = 'running' AND started_at IS NOT NULL
            THEN (julianday(datetime('now')) - julianday(started_at)) * 86400
            ELSE uptime_seconds
        END,
        updated_at = datetime('now')
    WHERE id = NEW.id;
END;

-- Trigger to update terminal session activity
CREATE TRIGGER trigger_update_terminal_activity
AFTER UPDATE ON terminal_sessions
WHEN NEW.data_sent_bytes != OLD.data_sent_bytes OR NEW.data_received_bytes != OLD.data_received_bytes
BEGIN
    UPDATE terminal_sessions
    SET last_activity = datetime('now')
    WHERE id = NEW.id;
END;

-- Trigger to automatically update updated_at timestamps
CREATE TRIGGER trigger_update_models_timestamp
AFTER UPDATE ON models
BEGIN
    UPDATE models SET updated_at = datetime('now') WHERE id = NEW.id;
END;

CREATE TRIGGER trigger_update_deployments_timestamp
AFTER UPDATE ON deployments
BEGIN
    UPDATE deployments SET updated_at = datetime('now') WHERE id = NEW.id;
END;

CREATE TRIGGER trigger_update_model_ratings_timestamp
AFTER UPDATE ON model_ratings
BEGIN
    UPDATE model_ratings SET updated_at = datetime('now') WHERE id = NEW.id;
END;
