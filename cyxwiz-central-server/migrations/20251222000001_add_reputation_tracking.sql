-- Add reputation tracking columns to nodes table for ban system
-- See docs/CLAUDE.md for reputation tier documentation

-- Strike tracking: how many times the node dropped below probation threshold
ALTER TABLE nodes ADD COLUMN strike_count INTEGER NOT NULL DEFAULT 0;

-- Ban expiration: NULL means not banned, otherwise contains UTC timestamp
ALTER TABLE nodes ADD COLUMN banned_until TEXT DEFAULT NULL;

-- Total lifetime bans: used for escalating ban durations
ALTER TABLE nodes ADD COLUMN total_bans INTEGER NOT NULL DEFAULT 0;

-- Timestamp of last strike: when the node last dropped below threshold
ALTER TABLE nodes ADD COLUMN last_strike_at TEXT DEFAULT NULL;

-- Create index for finding banned nodes
CREATE INDEX IF NOT EXISTS idx_nodes_banned_until ON nodes(banned_until);

-- Create index for finding nodes by strike count
CREATE INDEX IF NOT EXISTS idx_nodes_strike_count ON nodes(strike_count);
