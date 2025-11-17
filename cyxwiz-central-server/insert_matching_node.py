#!/usr/bin/env python3
import sqlite3
import uuid
from datetime import datetime, timezone

# Connect to database
conn = sqlite3.connect('cyxwiz.db')
cursor = conn.cursor()

# Use a specific UUID that matches what we'll track
# We'll use a deterministic UUID for the node
node_id_str = "node_1763366902"  # This should match the Server Node's ID format

# For SQLite, we need to store UUID as BLOB
# Generate a UUID from the node string
node_id = uuid.uuid5(uuid.NAMESPACE_DNS, node_id_str)

print(f"Inserting node with ID: {node_id}")
print(f"Node string: {node_id_str}")

# Insert node matching Server Node's capabilities
cursor.execute('''
INSERT INTO nodes (
    id, wallet_address, name, status, reputation_score, stake_amount,
    cpu_cores, ram_gb, gpu_model, gpu_memory_gb, has_cuda, has_opencl,
    total_jobs_completed, total_jobs_failed, uptime_percentage, current_load,
    country, region, last_heartbeat, registered_at, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', (
    node_id.bytes,  # Store as binary BLOB
    'node_wallet_' + node_id_str,
    'Local Test Node - GTX 1050 Ti',
    'online',  # Must be online for scheduler
    100.0,  # Perfect reputation
    1000000000,  # 1 SOL stake
    8,  # CPU cores (estimate)
    16,  # 16GB RAM (estimate)
    'NVIDIA GeForce GTX 1050 Ti',
    4,  # 4GB GPU memory (GTX 1050 Ti spec)
    0,  # has CUDA (OpenCL mode)
    1,  # has OpenCL
    0,  # completed jobs
    0,  # failed jobs
    100.0,  # uptime percentage
    0.0,  # current load (idle)
    'USA',
    'Local',
    datetime.now(timezone.utc).isoformat(),  # last heartbeat (now)
    datetime.now(timezone.utc).isoformat(),
    datetime.now(timezone.utc).isoformat()
))

conn.commit()

print("Node inserted successfully!")
print(f"\nNode details:")
print(f"  ID (UUID): {node_id}")
print(f"  Name: Local Test Node - GTX 1050 Ti")
print(f"  Status: online")
print(f"  GPU: NVIDIA GeForce GTX 1050 Ti (4GB)")
print(f"  Load: 0.0 (idle)")

# Verify insertion
cursor.execute('SELECT COUNT(*) FROM nodes WHERE status = "online"')
online_nodes = cursor.fetchone()[0]
print(f"\nOnline nodes in database: {online_nodes}")

conn.close()
print("\nReady for job assignment!")
