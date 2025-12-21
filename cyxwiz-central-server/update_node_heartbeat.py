import sqlite3
from datetime import datetime, timedelta

# Connect to database
conn = sqlite3.connect('cyxwiz.db')
cursor = conn.cursor()

# Get the test node
cursor.execute("SELECT id, name FROM nodes LIMIT 1")
node = cursor.fetchone()

if node:
    node_id, node_name = node
    # Set last_heartbeat to 60 seconds ago and status to online
    old_time = (datetime.utcnow() - timedelta(seconds=60)).strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("UPDATE nodes SET last_heartbeat = ?, status = 'online' WHERE id = ?", (old_time, node_id))
    conn.commit()

    print(f"Updated node {node_id} ({node_name})")
    print(f"  - Status: online")
    print(f"  - Last heartbeat: {old_time} (60 seconds ago)")
    print(f"\nNodeMonitor should detect this as disconnected within 10 seconds")
    print(f"Watch the Central Server logs for:")
    print(f"  WARN  Node {node_id} ({node_name}) disconnected - no heartbeat for XX seconds")
    print(f"  INFO  Node {node_id} ({node_name}) marked as OFFLINE")
else:
    print("No nodes found in database")

conn.close()
