#!/usr/bin/env python3
import sqlite3
import uuid
import json
from datetime import datetime, timezone

# Connect to database
conn = sqlite3.connect('cyxwiz.db')
cursor = conn.cursor()

job_id = uuid.uuid4()

print(f"Inserting training job: {job_id}")

# Create proper metadata JSON for the job
metadata = {
    "model_definition": '{"type": "resnet", "layers": 50}',
    "dataset_uri": "mock://cifar10",
    "batch_size": 64,
    "epochs": 10,
    "hyperparameters": {
        "learning_rate": "0.001",
        "optimizer": "adam",
        "weight_decay": "0.0001"
    }
}

# Insert training job with proper metadata
cursor.execute('''
INSERT INTO jobs (
    id, user_wallet, status, job_type,
    required_gpu, required_gpu_memory_gb, required_ram_gb,
    estimated_duration_seconds, estimated_cost, actual_cost,
    assigned_node_id, retry_count, result_hash, error_message,
    metadata, created_at, started_at, completed_at, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', (
    job_id.bytes,
    'user_wallet_xyz789',
    'pending',
    'training',
    1,  # requires GPU
    4,  # 4GB GPU memory (matches GTX 1050 Ti)
    16, # 16GB RAM
    3600,  # 1 hour estimate
    5000000,  # cost in tokens
    None,  # no actual cost yet
    None,  # not assigned yet
    0,  # retry count
    None,  # no result yet
    None,  # no error
    json.dumps(metadata),  # Proper metadata JSON
    datetime.now(timezone.utc).isoformat(),
    None,  # not started
    None,  # not completed
    datetime.now(timezone.utc).isoformat()
))

conn.commit()

print("Training job inserted successfully!")
print(f"Job ID (UUID): {job_id}")
print(f"\nMetadata:")
print(json.dumps(metadata, indent=2))

# Verify
cursor.execute('SELECT COUNT(*) FROM jobs WHERE status = "pending"')
pending_count = cursor.fetchone()[0]
print(f"\nPending jobs in database: {pending_count}")

conn.close()
print("\nReady for scheduler to assign job to node!")
