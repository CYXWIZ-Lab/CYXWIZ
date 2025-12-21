#!/usr/bin/env python3
"""
Simple fix: Drop UNIQUE index and create partial UNIQUE index for wallet_address
"""
import sqlite3

db_path = "cyxwiz.db"

print(f"Opening database: {db_path}")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    # First, check existing indexes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='nodes' AND sql LIKE '%wallet_address%'")
    existing = cursor.fetchall()
    print(f"Existing wallet indexes: {existing}")

    # Drop existing UNIQUE index if it exists
    for idx in existing:
        idx_name = idx[0]
        if idx_name:
            print(f"Dropping index: {idx_name}")
            cursor.execute(f"DROP INDEX IF EXISTS {idx_name}")

    # Create partial unique index - only enforce uniqueness for non-empty wallet addresses
    print("Creating partial UNIQUE index...")
    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_nodes_wallet_unique
        ON nodes(wallet_address)
        WHERE wallet_address != ''
    """)

    # Create regular index for all wallet addresses
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_wallet ON nodes(wallet_address)")

    # Create endpoint index if it doesn't exist
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_endpoint ON nodes(ip_address, port)")

    conn.commit()
    print("SUCCESS: Migration applied!")
except Exception as e:
    print(f"ERROR: Migration failed: {e}")
    conn.rollback()
finally:
    conn.close()

print("\nVerifying indexes...")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='nodes'")
indexes = cursor.fetchall()
for idx_name, idx_sql in indexes:
    print(f"  {idx_name}")
    if idx_sql:
        print(f"    SQL: {idx_sql}")
conn.close()
print("\nDone!")
