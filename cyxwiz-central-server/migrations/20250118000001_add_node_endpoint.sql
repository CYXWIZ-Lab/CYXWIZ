-- Add IP address and port to nodes table
ALTER TABLE nodes ADD COLUMN ip_address TEXT NOT NULL DEFAULT '0.0.0.0';
ALTER TABLE nodes ADD COLUMN port INTEGER NOT NULL DEFAULT 50052;

-- Create index for IP:port lookups
CREATE INDEX IF NOT EXISTS idx_nodes_endpoint ON nodes(ip_address, port);

-- Update the wallet_address UNIQUE constraint to allow empty strings for unauthenticated nodes
-- SQLite doesn't support modifying constraints directly, but empty strings won't be duplicated
-- because we check by IP:port for nodes without wallet addresses
