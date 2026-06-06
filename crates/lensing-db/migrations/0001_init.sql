-- Metadata/parameter state for the lensing server.
--
-- Timestamps are stored as TEXT (RFC 3339, exactly as the Rust structs carry
-- them): Postgres TIMESTAMPTZ truncates to microseconds, which would change
-- the strings on round-trip — and byte-identical round-trips are what makes
-- the file backfill verifiably lossless.
--
-- No hard foreign keys on purpose: runs may reference deleted datasets and
-- promoted models may outlive their runs (delete_run / dataset deletion are
-- features, and a model dir is a self-contained snapshot). FK constraints
-- would reject that history and violate the zero-data-loss backfill.

CREATE TABLE IF NOT EXISTS definitions (
    name         TEXT PRIMARY KEY,
    predictor    TEXT NOT NULL,
    hyperparams  JSONB NOT NULL,
    dataset_tags JSONB NOT NULL DEFAULT '[]',
    notes        TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT
);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    created_at TEXT,
    -- Full manifest.json (columns, pca, target, split, feature/quality/
    -- currency config). The binary artifacts stay under data/datasets/.
    manifest   JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    dataset_id       TEXT NOT NULL,
    predictor        TEXT NOT NULL,
    hyperparams      JSONB NOT NULL,
    status           TEXT NOT NULL,
    started_at       TEXT NOT NULL,
    finished_at      TEXT,
    exit_code        INTEGER,
    stderr_tail      TEXT,
    metrics          JSONB,
    contract_version INTEGER NOT NULL DEFAULT 1,
    has_checkpoint   BOOLEAN NOT NULL DEFAULT FALSE,
    from_definition  TEXT
);
CREATE INDEX IF NOT EXISTS runs_dataset_idx ON runs (dataset_id);
CREATE INDEX IF NOT EXISTS runs_predictor_idx ON runs (predictor);
CREATE INDEX IF NOT EXISTS runs_definition_idx ON runs (from_definition) WHERE from_definition IS NOT NULL;

-- progress.jsonl lines, in order. Stored as TEXT (the raw line) so the
-- mirror is byte-faithful to the file.
CREATE TABLE IF NOT EXISTS run_events (
    run_id TEXT NOT NULL,
    seq    INTEGER NOT NULL,
    line   TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);

CREATE TABLE IF NOT EXISTS models (
    name       TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL,
    predictor  TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    notes      TEXT,
    -- Frozen featurization contract (contract.json), for queryability; the
    -- authoritative copy lives in the model dir snapshot.
    contract   JSONB
);

-- Hyperparameter presets used by the CLI-runner pattern (data/cli-runs/hp/).
CREATE TABLE IF NOT EXISTS hp_presets (
    name        TEXT PRIMARY KEY,
    hyperparams JSONB NOT NULL
);

-- ---- Distributed roles (hub / training worker / inference node) ----
-- Workers claim queued runs from this table (FOR UPDATE SKIP LOCKED) and
-- write results back; inference nodes materialize promoted models from
-- model_artifacts. Datasets travel over the hub's archive endpoint, so a
-- worker needs only DATABASE_URL + the hub URL (+ the predictor toolchain).
ALTER TABLE runs ADD COLUMN IF NOT EXISTS claimed_by TEXT;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS heartbeat_at TEXT;

-- Predictor-written run outputs (checkpoints, metrics, predictions, viz)
-- uploaded by workers so promotion/inspection works from anywhere.
CREATE TABLE IF NOT EXISTS run_artifacts (
    run_id TEXT NOT NULL,
    path   TEXT NOT NULL,
    bytes  BYTEA NOT NULL,
    PRIMARY KEY (run_id, path)
);

-- Promoted model snapshots (record.json, contract.json, weights, …) so an
-- inference node can serve a model with no shared filesystem.
CREATE TABLE IF NOT EXISTS model_artifacts (
    name  TEXT NOT NULL,
    path  TEXT NOT NULL,
    bytes BYTEA NOT NULL,
    PRIMARY KEY (name, path)
);

-- The best-models group: a flat snapshot of the current member set, rewritten
-- wholesale on each recompute (definitions-style full swap). The JSON mirror
-- under data/best-models.json is the file copy on the backfill path and also
-- carries the group's pinned/excluded curation lists.
CREATE TABLE IF NOT EXISTS best_models (
    name         TEXT PRIMARY KEY,
    rank         INTEGER NOT NULL,
    metric       TEXT NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    run_id       TEXT NOT NULL,
    predictor    TEXT NOT NULL,
    dataset_id   TEXT NOT NULL,
    source       TEXT NOT NULL,            -- 'auto' | 'pinned'
    selected_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS best_models_rank_idx ON best_models (rank);
