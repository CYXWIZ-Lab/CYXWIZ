#!/usr/bin/env python3
import sqlite3
import uuid
from datetime import datetime

# Connect to database
conn = sqlite3.connect('cyxwiz.db')
cursor = conn.cursor()

# Generate job ID
job_id = str(uuid.uuid4())

# Insert test job
cursor.execute('''
INSERT INTO jobs (
    id,
    user_wallet,
    status,
    job_type,
    required_gpu,
    required_gpu_memory_gb,
    required_ram_gb,
    estimated_duration_seconds,
    estimated_cost,
    metadata,
    created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', (
    job_id,
    'test_user_wallet_123',
    'pending',
    'training',
    1,  # requires GPU
    8,  # 8GB GPU memory
    16, # 16GB RAM
    3600,  # 1 hour
    1000000,  # cost in lamports
    '{"model": "resnet50", "dataset": "imagenet", "epochs": 10}',
    datetime.now().isoformat()
))

conn.commit()
print(f"Test job inserted successfully!")
print(f"Job ID: {job_id}")
print(f"Status: pending")
print(f"Type: training")
print(f"Requirements: GPU (8GB), RAM (16GB)")

# Verify insertion
cursor.execute('SELECT COUNT(*) FROM jobs WHERE status = "pending"')
count = cursor.fetchone()[0]
print(f"\nTotal pending jobs: {count}")

conn.close()
