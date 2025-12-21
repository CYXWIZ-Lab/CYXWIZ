#!/usr/bin/env python3
"""
Fix the UNIQUE constraint on wallet_address to allow multiple empty strings.
"""
import sqlite3

db_path = "cyxwiz.db"

print(f"Opening database: {db_path}")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Read and execute the migration SQL
with open("migrations/20250118000002_fix_wallet_unique_constraint.sql", "r") as f:
    sql = f.read()

print("Executing migration...")
try:
    cursor.executescript(sql)
    conn.commit()
    print("✓ Migration applied successfully!")
except Exception as e:
    print(f"✗ Migration failed: {e}")
    conn.rollback()
finally:
    conn.close()

print("\nVerifying schema...")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check indexes
cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='nodes'")
indexes = cursor.fetchall()
print(f"Indexes on nodes table: {[idx[0] for idx in indexes]}")

conn.close()
print("\nDone!")
