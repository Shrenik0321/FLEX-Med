-- ==========================================
-- FLEX-Med Database Schema (FINAL, FIXED)
-- ==========================================

-- ------------------------------------------
-- 0. DROP TABLES
-- ------------------------------------------

DROP TABLE IF EXISTS client_simulation_metrics CASCADE;
DROP TABLE IF EXISTS fl_simulations CASCADE;
DROP TABLE IF EXISTS clients CASCADE;
DROP TABLE IF EXISTS system_config CASCADE;

-- ------------------------------------------
-- 1. ENUM TYPES (IDEMPOTENT)
-- ------------------------------------------

DO $$
BEGIN
    -- Client status
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'client_status') THEN
        CREATE TYPE client_status AS ENUM ('Active', 'Inactive');
    END IF;

    -- Model types (CNN architectures)
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'model_type') THEN
        CREATE TYPE model_type AS ENUM (
            'efficientnet_b0',
            'efficientnet_b1',
            'efficientnet_b2'
        );
    END IF;

    -- FL simulation lifecycle status
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'simulation_status') THEN
        CREATE TYPE simulation_status AS ENUM (
            'pending', 'running', 'completed', 'failed'
        );
    END IF;

    -- Per-client simulation status
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'client_simulation_status') THEN
        CREATE TYPE client_simulation_status AS ENUM (
            'pending', 'training', 'completed', 'failed'
        );
    END IF;
END
$$;

-- ------------------------------------------
-- 2. CLIENTS TABLE
-- ------------------------------------------

CREATE TABLE clients (
    id SERIAL PRIMARY KEY,
    client_name TEXT NOT NULL UNIQUE,
    model_type model_type NOT NULL,
    model_path TEXT,
    status client_status NOT NULL DEFAULT 'Inactive',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_clients_status ON clients(status);
CREATE INDEX idx_clients_model_type ON clients(model_type);

-- ------------------------------------------
-- fl_simulations TABLE
-- ------------------------------------------
-- configs JSONB structure:
-- {
--   "num_rounds": 10,
--   "local_epochs": 1,
--   "learning_rate": 0.001,
--   "weight_decay": 0.02,
--   ...
-- }
-- aggregate_metrics JSONB structure:
-- {
--   "rounds": [
--     {
--       "round": 1,
--       "avg_accuracy": 0.85,
--       "avg_precision": 0.82,
--       "avg_recall": 0.88,
--       "avg_f1": 0.85,
--       "avg_loss": 0.45,
--       "avg_train_loss": 0.08,    -- Average training loss across clients
--       "avg_val_loss": 0.07,      -- Average validation loss across clients
--       "evaluated_at": "2026-01-31T..."
--     }
--   ],
--   "improvement": { ... }  -- Calculated from Round 1 vs Final Round
-- }

CREATE TABLE fl_simulations (
    id SERIAL PRIMARY KEY,
    configs JSONB NOT NULL DEFAULT '{}'::jsonb,
    aggregate_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    status simulation_status NOT NULL DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration INTEGER
);

CREATE INDEX idx_fl_simulations_status
    ON fl_simulations(status);

CREATE INDEX idx_fl_simulations_created_at
    ON fl_simulations(created_at DESC);

-- ------------------------------------------
-- 4. CLIENT_SIMULATION_METRICS TABLE
-- ------------------------------------------
-- metrics JSONB structure:
-- {
--   "global": {
--     "post_fl": { accuracy, precision, recall, f1_score, loss, ... }
--   },
--   "data_heterogeneity": {          -- Partition statistics from Dirichlet partitioning
--     "total_samples": 2500,         -- Total samples in this client's partition
--     "train_samples": 2125,         -- 85% for training
--     "val_samples": 375,            -- 15% for validation
--     "class_distribution": {
--       "leukemia": 1800,            -- ALL (class 0) sample count
--       "healthy": 700,              -- Healthy (class 1) sample count
--       "leukemia_pct": 72.0,        -- Percentage of leukemia samples
--       "healthy_pct": 28.0          -- Percentage of healthy samples
--     },
--     "imbalance_ratio": 2.57,       -- Ratio of majority to minority class
--     "partition_id": 0              -- Client's partition ID in Dirichlet partitioner
--   },
--   "rounds": [
--     {
--       "round": 1,
--       "validation": {            -- Evaluation on validation set after training
--         "accuracy": 0.83,
--         "precision": 0.80,
--         "recall": 0.86,
--         "f1_score": 0.83,
--         "loss": 0.55,
--         "evaluated_at": "..."
--       },
--       "training": {              -- Metrics from local training phase
--         "train_loss": 0.1158,    -- Training loss during local training
--         "val_loss": 0.0743,      -- Validation loss during local training
--         "distill_loss": 0.0346,  -- Knowledge distillation loss (null for Round 1)
--         "num_examples": 2179,    -- Number of training samples
--         "training_time": 3501.1  -- Training duration in seconds
--       }
--     }
--   ]
-- }

CREATE TABLE client_simulation_metrics (
    id SERIAL PRIMARY KEY,
    simulation_id INTEGER NOT NULL
        REFERENCES fl_simulations(id) ON DELETE CASCADE,
    client_id INTEGER NOT NULL
        REFERENCES clients(id) ON DELETE CASCADE,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    status client_simulation_status NOT NULL DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    UNIQUE (simulation_id, client_id)
);

CREATE INDEX idx_client_sim_metrics_status
    ON client_simulation_metrics(status);

CREATE INDEX idx_client_sim_metrics_simulation
    ON client_simulation_metrics(simulation_id);

CREATE INDEX idx_client_sim_metrics_client
    ON client_simulation_metrics(client_id);

-- ------------------------------------------
-- 5. SYSTEM_CONFIG TABLE
-- ------------------------------------------
-- Stores system-wide configuration for FL simulations
-- Uses singleton pattern (only one row exists)

CREATE TABLE system_config (
    id SERIAL PRIMARY KEY,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- Single configuration — no presets. dirichlet_alpha is a direct top-level field.
INSERT INTO system_config (config) VALUES ('{
    "dirichlet_alpha": 1.5,
    "consensus_momentum": 0.35,
    "minority_boost": 0.90,
    "distill_weight_base": 0.80,
    "distill_decay_rate": 0.10,
    "weight_decay": 0.03,
    "train_loss_weight": 0.70,
    "distill_loss_weight": 0.30,
    "lr_decay": 0.95,
    "learning_rate": 0.001,
    "num_rounds": 10,
    "local_epochs": 1,
    "batch_size": 32,
    "distill_lr": 0.0005,
    "distill_epochs": 1,
    "temperature": 3.0,
    "dirichlet_seed": 42,
    "dirichlet_min_partition_size": 400,
    "public_anchor_dataset_path": "/content/drive/MyDrive/College/FLEX-Med/backend/datasets/cnmc/cnmc_public_anchor",
    "public_test_dataset_path": "/content/drive/MyDrive/College/FLEX-Med/backend/datasets/cnmc/cnmc_public_test",
    "local_train_dataset_path": "/content/drive/MyDrive/College/FLEX-Med/backend/datasets/cnmc/cnmc_local_train",
    "ngrok_url": "https://intraspinal-agape-deidra.ngrok-free.dev"
}'::jsonb);

CREATE INDEX idx_system_config_id ON system_config(id);
