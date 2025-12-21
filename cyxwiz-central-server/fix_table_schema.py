#!/usr/bin/env python3
"""
Recreate nodes table without UNIQUE constraint on wallet_address
"""
import sqlite3

db_path = "cyxwiz.db"

print(f"Opening database: {db_path}")
conn = sqlite3.connect(db_path)
conn.execute("PRAGMA foreign_keys=OFF")
cursor = conn.cursor()

try:
    print("Creating new nodes table without UNIQUE constraint...")
    cursor.execute("""
        CREATE TABLE nodes_new (
            id TEXT PRIMARY KEY,
            wallet_address TEXT NOT NULL,
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
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

            -- Extended fields
            supported_formats TEXT DEFAULT '[]',
            max_model_size_bytes INTEGER DEFAULT 0,
            supports_terminal_access INTEGER DEFAULT 0,
            available_runtimes TEXT DEFAULT '[]',
            gpu_vram_total_bytes INTEGER DEFAULT 0,
            gpu_vram_available_bytes INTEGER DEFAULT 0,
            gpu_driver_version TEXT,
            cuda_version TEXT,
            compute_capability REAL,
            pcie_generation INTEGER,
            pcie_lanes INTEGER,
            ip_address TEXT NOT NULL DEFAULT '0.0.0.0',
            port INTEGER NOT NULL DEFAULT 50052
        )
    """)

    print("Copying data from old table...")
    cursor.execute("INSERT INTO nodes_new SELECT * FROM nodes")

    print("Dropping old table...")
    cursor.execute("DROP TABLE nodes")

    print("Renaming new table...")
    cursor.execute("ALTER TABLE nodes_new RENAME TO nodes")

    print("Creating indexes...")
    # Status index
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status)")

    # Reputation index
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_reputation ON nodes(reputation_score DESC)")

    # Partial UNIQUE index for non-empty wallet addresses
    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_nodes_wallet_unique
        ON nodes(wallet_address)
        WHERE wallet_address != ''
    """)

    # Regular wallet index
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_wallet ON nodes(wallet_address)")

    # Endpoint index
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_endpoint ON nodes(ip_address, port)")

    conn.execute("PRAGMA foreign_keys=ON")
    conn.commit()
    print("SUCCESS: Table recreated without UNIQUE constraint!")

except Exception as e:
    print(f"ERROR: {e}")
    conn.rollback()
finally:
    conn.close()

print("\nVerifying schema...")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='nodes'")
print("New table schema:")
print(cursor.fetchone()[0])
print("\nIndexes:")
cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='nodes'")
for name, sql in cursor.fetchall():
    print(f"  {name}")
    if sql:
        print(f"    {sql}")
conn.close()
print("\nDone!")
