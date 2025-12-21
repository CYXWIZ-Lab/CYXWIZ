-- Node devices table for multi-device support
-- Each node can have multiple devices (GPU0, GPU1, CPU, etc.)
-- Devices are populated when node registers with Central Server

CREATE TABLE IF NOT EXISTS node_devices (
    id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,

    -- Device identification
    device_type TEXT NOT NULL CHECK(device_type IN ('cuda', 'opencl', 'cpu')),
    device_index INTEGER NOT NULL,
    device_name TEXT NOT NULL,

    -- Allocation settings (user-specified in GUI)
    is_enabled INTEGER NOT NULL DEFAULT 1,
    vram_allocated_mb INTEGER DEFAULT 0,
    cores_allocated INTEGER DEFAULT 0,

    -- Hardware info
    memory_total_bytes INTEGER DEFAULT 0,
    memory_available_bytes INTEGER DEFAULT 0,
    compute_units INTEGER DEFAULT 0,
    supports_fp64 INTEGER DEFAULT 0,
    supports_fp16 INTEGER DEFAULT 0,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Each device is unique per node
    UNIQUE(node_id, device_type, device_index),
    FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_node_devices_node ON node_devices(node_id);
CREATE INDEX IF NOT EXISTS idx_node_devices_enabled ON node_devices(is_enabled);
CREATE INDEX IF NOT EXISTS idx_node_devices_type ON node_devices(device_type);
