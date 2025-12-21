#!/usr/bin/env python3
"""
Add ip_address and port columns to nodes table
"""
import sqlite3

db_path = "cyxwiz.db"

print(f"Opening database: {db_path}")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    # Check if columns already exist
    cursor.execute("PRAGMA table_info(nodes)")
    columns = [row[1] for row in cursor.fetchall()]
    print(f"Existing columns: {columns}")

    # Add ip_address if it doesn't exist
    if 'ip_address' not in columns:
        print("Adding ip_address column...")
        cursor.execute("ALTER TABLE nodes ADD COLUMN ip_address TEXT NOT NULL DEFAULT '0.0.0.0'")
        print("  Added ip_address")
    else:
        print("  ip_address already exists")

    # Add port if it doesn't exist
    if 'port' not in columns:
        print("Adding port column...")
        cursor.execute("ALTER TABLE nodes ADD COLUMN port INTEGER NOT NULL DEFAULT 50052")
        print("  Added port")
    else:
        print("  port already exists")

    # Create endpoint index
    print("Creating endpoint index...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_endpoint ON nodes(ip_address, port)")
    print("  Created idx_nodes_endpoint")

    conn.commit()
    print("\nSUCCESS: Columns added!")
except Exception as e:
    print(f"\nERROR: {e}")
    conn.rollback()
finally:
    conn.close()

print("\nVerifying schema...")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(nodes)")
columns = cursor.fetchall()
print("Nodes table columns:")
for col in columns:
    print(f"  {col[1]} ({col[2]})")
conn.close()
print("\nDone!")
