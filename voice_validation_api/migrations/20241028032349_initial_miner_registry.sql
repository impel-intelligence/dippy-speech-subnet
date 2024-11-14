-- Migration: initial_miner_registry

-- Write your migration SQL here
CREATE TABLE IF NOT EXISTS miner_entries (
    hotkey VARCHAR NOT NULL,
    hash VARCHAR NOT NULL,
    block INTEGER NOT NULL,
    uid INTEGER NOT NULL,
    created_at timestamptz
);

CREATE TABLE IF NOT EXISTS hash_entries (
    hash VARCHAR PRIMARY KEY,
    total_score FLOAT, 
    alpha_score FLOAT, 
    beta_score FLOAT, 
    gamma_score FLOAT, 
    notes TEXT, 
    repo_namespace TEXT,
    repo_name TEXT,
    timestamp timestamptz,
    safetensors_hash TEXT,
    status TEXT
);

CREATE INDEX IF NOT EXISTS idx_miner_entries_hotkey 
ON miner_entries(hotkey);