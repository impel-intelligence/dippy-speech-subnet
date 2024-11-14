-- Migration: initial_miner_registry

CREATE TABLE IF NOT EXISTS miner_entries (
    uid INTEGER PRIMARY KEY,
    hotkey VARCHAR NOT NULL,
    hash VARCHAR NOT NULL,
    block INTEGER NOT NULL
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
