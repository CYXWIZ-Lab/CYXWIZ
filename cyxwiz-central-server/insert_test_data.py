#!/usr/bin/env python3
import sqlite3
import uuid
from datetime import datetime, timezone

# Connect to database
conn = sqlite3.connect('cyxwiz.db')
cursor = conn.cursor()

# Generate UUIDs as binary
node_id = uuid.uuid4()
job_id = uuid.uuid4()

print(f"Inserting test node: {node_id}")
print(f"Inserting test job: {job_id}")

# Insert test node (must be online and available for scheduler to find it)
cursor.execute('''
INSERT INTO nodes (
    id, wallet_address, name, status, reputation_score, stake_amount,
    cpu_cores, ram_gb, gpu_model, gpu_memory_gb, has_cuda, has_opencl,
    total_jobs_completed, total_jobs_failed, uptime_percentage, current_load,
    country, region, last_heartbeat, registered_at, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', (
    node_id.bytes,  # Store as binary BLOB
    'test_wallet_abc123',
    'Test GPU Node',
    'online',  # Must be online for scheduler
    95.5,  # High reputation
    1000000000,  # 1 SOL stake
    16,  # 16 CPU cores
    64,  # 64GB RAM
    'NVIDIA RTX 4090',
    24,  # 24GB GPU memory
    1,  # has CUDA
    1,  # has OpenCL
    42,  # completed jobs
    2,   # failed jobs
    99.8,  # uptime percentage
    0.3,  # current load (30% - below 90% threshold)
    'USA',
    'California',
    datetime.now(timezone.utc).isoformat(),  # last heartbeat (recent)
    datetime.now(timezone.utc).isoformat(),
    datetime.now(timezone.utc).isoformat()
))

print("✓ Test node inserted")

# Insert test job (pending status for scheduler to pick up)
cursor.execute('''
INSERT INTO jobs (
    id, user_wallet, status, job_type,
    required_gpu, required_gpu_memory_gb, required_ram_gb,
    estimated_duration_seconds, estimated_cost, actual_cost,
    assigned_node_id, retry_count, result_hash, error_message,
    metadata, created_at, started_at, completed_at, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', (
    job_id.bytes,  # Store as binary BLOB
    'user_wallet_xyz789',
    'pending',  # Must be pending for scheduler
    'training',
    1,  # requires GPU
    8,  # 8GB GPU memory
    16, # 16GB RAM
    3600,  # 1 hour estimate
    5000000,  # cost in tokens
    None,  # no actual cost yet
    None,  # not assigned yet
    0,  # retry count
    None,  # no result yet
    None,  # no error
    '{"model": "resnet50", "dataset": "cifar10", "epochs": 10, "batch_size": 64}',
    datetime.now(timezone.utc).isoformat(),
    None,  # not started
    None,  # not completed
    datetime.now(timezone.utc).isoformat()
))

conn.commit()

print("✓ Test job inserted")
print(f"\nNode ID (UUID): {node_id}")
print(f"Job ID (UUID): {job_id}")
print(f"\nScheduler should now:")
print("  1. Find the pending job")
print("  2. Find the available online node")
print("  3. Assign the job to the node")
print(f"  4. Update job status from 'pending' to 'assigned'")

# Verify insertion
cursor.execute('SELECT COUNT(*) FROM nodes WHERE status = "online"')
online_nodes = cursor.fetchone()[0]
print(f"\nOnline nodes in database: {online_nodes}")

cursor.execute('SELECT COUNT(*) FROM jobs WHERE status = "pending"')
pending_jobs = cursor.fetchone()[0]
print(f"Pending jobs in database: {pending_jobs}")

conn.close()
print("\n✓ Database setup complete - scheduler should process job within 1 second")
