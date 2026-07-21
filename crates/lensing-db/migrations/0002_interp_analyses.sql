-- Persisted interpretability analyses (SAE / layer-probe / embedding-probe).
-- A transactional MIRROR of the interp jobs whose raw output also lives under
-- data/interp/<id>/. The full analysis document is kept in `result` (JSONB,
-- on par with datasets.manifest) so a past analysis is served without
-- recomputation and survives a data/interp cleanup; the large dataset-SAE codes
-- intermediate is never stored. No FKs (an analysis may outlive its model or
-- dataset), timestamps as TEXT — matching every other table here. Applied as an
-- idempotent additive migration after 0001_init.sql on every connect.
CREATE TABLE IF NOT EXISTS interp_analyses (
    id          TEXT PRIMARY KEY,   -- the interp job id (…-msae/-sae/-lp/-cmp/-ep)
    tool        TEXT NOT NULL,      -- 'model-sae'|'sae'|'layer-probe'|'layer-probe-compare'|'embedding-probe'
    model       TEXT,               -- model dir name; NULL for dataset-centric tools
    dataset_id  TEXT NOT NULL,
    predictor   TEXT,               -- predictor family of `model`, when applicable
    config      JSONB NOT NULL,     -- request knobs (layers, n_atoms, l1, epochs, compare_embedding, …)
    status      TEXT NOT NULL,      -- 'running'|'done'|'failed'
    error       TEXT,
    source      TEXT NOT NULL DEFAULT 'manual',  -- 'manual'|'auto'
    created_at  TEXT NOT NULL,
    finished_at TEXT,
    result      JSONB               -- full analysis document; NULL until done
);
CREATE INDEX IF NOT EXISTS interp_analyses_model_idx ON interp_analyses (model);
CREATE INDEX IF NOT EXISTS interp_analyses_tool_idx  ON interp_analyses (tool);
