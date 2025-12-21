#!/usr/bin/env python3
"""
Mark migrations as already applied
"""
import sqlite3
import hashlib

db_path = "cyxwiz.db"

print(f"Opening database: {db_path}")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create _sqlx_migrations table if it doesn't exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS _sqlx_migrations (
        version BIGINT PRIMARY KEY,
        description TEXT NOT NULL,
        installed_on TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        success BOOLEAN NOT NULL,
        checksum BLOB NOT NULL,
        execution_time BIGINT NOT NULL
    )
""")

# Get checksums for migrations
migrations = [
    (20250118000001, "add_node_endpoint", "migrations/20250118000001_add_node_endpoint.sql"),
    (20250118000002, "fix_wallet_unique_constraint", "migrations/20250118000002_fix_wallet_unique_constraint.sql"),
]

for version, description, filepath in migrations:
    # Check if already applied
    cursor.execute("SELECT version FROM _sqlx_migrations WHERE version = ?", (version,))
    if cursor.fetchone():
        print(f"Migration {version} already applied, skipping")
        continue

    # Read migration file and compute checksum
    try:
        with open(filepath, "rb") as f:
            content = f.read()
        checksum = hashlib.sha384(content).digest()

        # Insert migration record
        cursor.execute("""
            INSERT INTO _sqlx_migrations (version, description, success, checksum, execution_time)
            VALUES (?, ?, 1, ?, 0)
        """, (version, description, checksum))

        print(f"Marked migration {version} ({description}) as applied")
    except FileNotFoundError:
        print(f"Warning: Migration file not found: {filepath}")

conn.commit()
conn.close()

print("\nDone! Migrations marked as applied.")
