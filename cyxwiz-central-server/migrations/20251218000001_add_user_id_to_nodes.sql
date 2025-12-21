-- Add user_id to nodes table for linking to Website's MongoDB users
-- This allows one user to operate multiple nodes across different devices

-- Add user_id column (nullable for backward compatibility with existing nodes)
ALTER TABLE nodes ADD COLUMN user_id TEXT;

-- Add device_id for hardware-based unique identification
-- Note: SQLite doesn't support UNIQUE in ALTER TABLE, so we add it without constraint
-- Uniqueness will be enforced at application level
ALTER TABLE nodes ADD COLUMN device_id TEXT;

-- Create index for fast user lookups
CREATE INDEX IF NOT EXISTS idx_nodes_user_id ON nodes(user_id);

-- Create index for device_id lookups (helps with uniqueness checks)
CREATE INDEX IF NOT EXISTS idx_nodes_device_id ON nodes(device_id);
