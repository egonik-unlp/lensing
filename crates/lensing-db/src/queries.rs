//! Typed query layer over tokio-postgres.

use anyhow::{Context, Result};
use lensing_core::{BestModelEntry, InterpAnalysis, ModelDefinition, ModelRecord, RunMeta};

use crate::Db;

// ---------------- definitions (database-authoritative) ----------------

pub async fn load_definitions(db: &Db) -> Result<Vec<ModelDefinition>> {
    let client = db.client().await?;
    let rows = client
        .query(
            "SELECT name, predictor, hyperparams, dataset_tags, notes, created_at, updated_at
             FROM definitions ORDER BY created_at, name",
            &[],
        )
        .await
        .context("load definitions")?;
    rows.iter()
        .map(|r| {
            Ok(ModelDefinition {
                name: r.get("name"),
                predictor: r.get("predictor"),
                hyperparams: r.get::<_, serde_json::Value>("hyperparams"),
                dataset_tags: serde_json::from_value(r.get::<_, serde_json::Value>("dataset_tags"))
                    .context("dataset_tags")?,
                notes: r.get("notes"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
            })
        })
        .collect()
}

/// Replace the whole definitions table with the given set, atomically. The
/// set is tiny (tens of rows) and `models.toml` semantics are whole-file, so
/// a full swap is the simplest write that can't drift.
pub async fn replace_definitions(db: &Db, defs: &[ModelDefinition]) -> Result<()> {
    let mut client = db.client().await?;
    let tx = client.transaction().await?;
    tx.execute("DELETE FROM definitions", &[]).await?;
    for d in defs {
        tx.execute(
            "INSERT INTO definitions (name, predictor, hyperparams, dataset_tags, notes, created_at, updated_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7)",
            &[
                &d.name,
                &d.predictor,
                &d.hyperparams,
                &serde_json::to_value(&d.dataset_tags)?,
                &d.notes,
                &d.created_at,
                &d.updated_at,
            ],
        )
        .await
        .with_context(|| format!("insert definition {}", d.name))?;
    }
    tx.commit().await.context("commit definitions")?;
    Ok(())
}

/// Seed a definition only if absent (first backfill from models.toml).
pub async fn seed_definition(db: &Db, d: &ModelDefinition) -> Result<bool> {
    let client = db.client().await?;
    let n = client
        .execute(
            "INSERT INTO definitions (name, predictor, hyperparams, dataset_tags, notes, created_at, updated_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT (name) DO NOTHING",
            &[
                &d.name,
                &d.predictor,
                &d.hyperparams,
                &serde_json::to_value(&d.dataset_tags)?,
                &d.notes,
                &d.created_at,
                &d.updated_at,
            ],
        )
        .await?;
    Ok(n > 0)
}

// ---------------- runs (mirror of data/runs/*/meta.json) ----------------

pub async fn upsert_run(db: &Db, m: &RunMeta) -> Result<()> {
    let client = db.client().await?;
    client
        .execute(
            "INSERT INTO runs (run_id, dataset_id, predictor, hyperparams, status, started_at,
                               finished_at, exit_code, stderr_tail, metrics, contract_version,
                               has_checkpoint, from_definition)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
             ON CONFLICT (run_id) DO UPDATE SET
               dataset_id = EXCLUDED.dataset_id, predictor = EXCLUDED.predictor,
               hyperparams = EXCLUDED.hyperparams, status = EXCLUDED.status,
               started_at = EXCLUDED.started_at, finished_at = EXCLUDED.finished_at,
               exit_code = EXCLUDED.exit_code, stderr_tail = EXCLUDED.stderr_tail,
               metrics = EXCLUDED.metrics, contract_version = EXCLUDED.contract_version,
               has_checkpoint = EXCLUDED.has_checkpoint, from_definition = EXCLUDED.from_definition",
            &[
                &m.run_id,
                &m.dataset_id,
                &m.predictor,
                &m.hyperparams,
                &status_str(m.status),
                &m.started_at,
                &m.finished_at,
                &m.exit_code,
                &m.stderr_tail,
                &m.metrics.as_ref().map(serde_json::to_value).transpose()?,
                &(m.contract_version as i32),
                &m.has_checkpoint,
                &m.from_definition,
            ],
        )
        .await
        .with_context(|| format!("upsert run {}", m.run_id))?;
    Ok(())
}

fn meta_from_row(row: &tokio_postgres::Row) -> Result<RunMeta> {
    let status: String = row.get("status");
    Ok(RunMeta {
        run_id: row.get("run_id"),
        dataset_id: row.get("dataset_id"),
        predictor: row.get("predictor"),
        hyperparams: row.get::<_, serde_json::Value>("hyperparams"),
        status: serde_json::from_value(serde_json::Value::String(status)).context("run status")?,
        started_at: row.get("started_at"),
        finished_at: row.get("finished_at"),
        exit_code: row.get("exit_code"),
        stderr_tail: row.get("stderr_tail"),
        metrics: row
            .get::<_, Option<serde_json::Value>>("metrics")
            .map(serde_json::from_value)
            .transpose()
            .context("run metrics")?,
        contract_version: row.get::<_, i32>("contract_version") as u32,
        has_checkpoint: row.get("has_checkpoint"),
        from_definition: row.get("from_definition"),
    })
}

/// All runs, newest first (run ids embed their timestamp).
pub async fn load_runs(db: &Db) -> Result<Vec<RunMeta>> {
    let client = db.client().await?;
    let rows = client
        .query("SELECT * FROM runs ORDER BY run_id DESC", &[])
        .await
        .context("load runs")?;
    rows.iter().map(meta_from_row).collect()
}

pub async fn load_run(db: &Db, run_id: &str) -> Result<Option<RunMeta>> {
    let client = db.client().await?;
    let row = client
        .query_opt("SELECT * FROM runs WHERE run_id = $1", &[&run_id])
        .await?;
    row.as_ref().map(meta_from_row).transpose()
}

/// Atomically claim the oldest queued run for a training worker. The claim
/// flips status to running under FOR UPDATE SKIP LOCKED, so concurrent
/// workers never double-claim.
pub async fn claim_queued_run(db: &Db, worker_id: &str, now: &str) -> Result<Option<RunMeta>> {
    let client = db.client().await?;
    let row = client
        .query_opt(
            "UPDATE runs SET status = 'running', claimed_by = $1, heartbeat_at = $2,
                             started_at = $2
             WHERE run_id = (SELECT run_id FROM runs WHERE status = 'queued'
                             ORDER BY run_id ASC LIMIT 1 FOR UPDATE SKIP LOCKED)
             RETURNING *",
            &[&worker_id, &now],
        )
        .await
        .context("claim queued run")?;
    row.as_ref().map(meta_from_row).transpose()
}

/// Progress lines for a run, ordered, optionally from a seq offset.
pub async fn load_events(db: &Db, run_id: &str, from_seq: i32) -> Result<Vec<(i32, String)>> {
    let client = db.client().await?;
    let rows = client
        .query(
            "SELECT seq, line FROM run_events WHERE run_id = $1 AND seq >= $2 ORDER BY seq",
            &[&run_id, &from_seq],
        )
        .await?;
    Ok(rows.iter().map(|r| (r.get("seq"), r.get("line"))).collect())
}

// ---------------- run/model artifacts (worker + inference transport) ----------------

pub async fn put_run_artifacts(db: &Db, run_id: &str, files: &[(String, Vec<u8>)]) -> Result<()> {
    let mut client = db.client().await?;
    let tx = client.transaction().await?;
    for (path, bytes) in files {
        tx.execute(
            "INSERT INTO run_artifacts (run_id, path, bytes) VALUES ($1, $2, $3)
             ON CONFLICT (run_id, path) DO UPDATE SET bytes = EXCLUDED.bytes",
            &[&run_id, &path, &bytes],
        )
        .await
        .with_context(|| format!("upload run artifact {path}"))?;
    }
    tx.commit().await?;
    Ok(())
}

pub async fn load_run_artifacts(db: &Db, run_id: &str) -> Result<Vec<(String, Vec<u8>)>> {
    let client = db.client().await?;
    let rows = client
        .query("SELECT path, bytes FROM run_artifacts WHERE run_id = $1", &[&run_id])
        .await?;
    Ok(rows.iter().map(|r| (r.get("path"), r.get("bytes"))).collect())
}

pub async fn load_run_artifact(db: &Db, run_id: &str, path: &str) -> Result<Option<Vec<u8>>> {
    let client = db.client().await?;
    let row = client
        .query_opt(
            "SELECT bytes FROM run_artifacts WHERE run_id = $1 AND path = $2",
            &[&run_id, &path],
        )
        .await?;
    Ok(row.map(|r| r.get("bytes")))
}

pub async fn put_model_artifacts(db: &Db, name: &str, files: &[(String, Vec<u8>)]) -> Result<()> {
    let mut client = db.client().await?;
    let tx = client.transaction().await?;
    tx.execute("DELETE FROM model_artifacts WHERE name = $1", &[&name]).await?;
    for (path, bytes) in files {
        tx.execute(
            "INSERT INTO model_artifacts (name, path, bytes) VALUES ($1, $2, $3)",
            &[&name, &path, &bytes],
        )
        .await
        .with_context(|| format!("upload model artifact {path}"))?;
    }
    tx.commit().await?;
    Ok(())
}

pub async fn model_artifact_names(db: &Db) -> Result<Vec<String>> {
    let client = db.client().await?;
    let rows = client.query("SELECT DISTINCT name FROM model_artifacts ORDER BY name", &[]).await?;
    Ok(rows.iter().map(|r| r.get("name")).collect())
}

pub async fn load_model_artifacts(db: &Db, name: &str) -> Result<Vec<(String, Vec<u8>)>> {
    let client = db.client().await?;
    let rows = client
        .query("SELECT path, bytes FROM model_artifacts WHERE name = $1", &[&name])
        .await?;
    Ok(rows.iter().map(|r| (r.get("path"), r.get("bytes"))).collect())
}

pub async fn delete_model_artifacts(db: &Db, name: &str) -> Result<()> {
    let client = db.client().await?;
    client.execute("DELETE FROM model_artifacts WHERE name = $1", &[&name]).await?;
    Ok(())
}

pub async fn rename_model_artifacts(db: &Db, old: &str, new: &str) -> Result<()> {
    let client = db.client().await?;
    client
        .execute("UPDATE model_artifacts SET name = $2 WHERE name = $1", &[&old, &new])
        .await?;
    Ok(())
}

pub async fn delete_run(db: &Db, run_id: &str) -> Result<()> {
    let mut client = db.client().await?;
    let tx = client.transaction().await?;
    tx.execute("DELETE FROM run_events WHERE run_id = $1", &[&run_id]).await?;
    tx.execute("DELETE FROM run_artifacts WHERE run_id = $1", &[&run_id]).await?;
    tx.execute("DELETE FROM runs WHERE run_id = $1", &[&run_id]).await?;
    tx.commit().await?;
    Ok(())
}

pub async fn next_event_seq(db: &Db, run_id: &str) -> Result<i32> {
    let client = db.client().await?;
    let row = client
        .query_one(
            "SELECT COALESCE(MAX(seq), -1) + 1 AS next FROM run_events WHERE run_id = $1",
            &[&run_id],
        )
        .await?;
    Ok(row.get::<_, i32>("next"))
}

/// Insert progress lines starting at `seq`, in chunks (the backfill pushes
/// hundreds of thousands of lines).
pub async fn insert_events(db: &Db, run_id: &str, start_seq: i32, lines: &[String]) -> Result<()> {
    let client = db.client().await?;
    for (chunk_idx, chunk) in lines.chunks(1000).enumerate() {
        let base = start_seq + (chunk_idx * 1000) as i32;
        let seqs: Vec<i32> = (0..chunk.len() as i32).map(|i| base + i).collect();
        client
            .execute(
                "INSERT INTO run_events (run_id, seq, line)
                 SELECT $1, t.s, t.l FROM UNNEST($2::int4[], $3::text[]) AS t(s, l)
                 ON CONFLICT (run_id, seq) DO NOTHING",
                &[&run_id, &seqs, &chunk.to_vec()],
            )
            .await
            .with_context(|| format!("insert events for {run_id}"))?;
    }
    Ok(())
}

// ---------------- models (mirror of data/models/*/record.json) ----------------

pub async fn upsert_model(db: &Db, r: &ModelRecord, contract: Option<&serde_json::Value>) -> Result<()> {
    let client = db.client().await?;
    client
        .execute(
            "INSERT INTO models (name, run_id, predictor, dataset_id, created_at, notes, contract)
             VALUES ($1, $2, $3, $4, $5, $6, $7)
             ON CONFLICT (name) DO UPDATE SET
               run_id = EXCLUDED.run_id, predictor = EXCLUDED.predictor,
               dataset_id = EXCLUDED.dataset_id, created_at = EXCLUDED.created_at,
               notes = EXCLUDED.notes,
               contract = COALESCE(EXCLUDED.contract, models.contract)",
            &[&r.name, &r.run_id, &r.predictor, &r.dataset_id, &r.created_at, &r.notes, &contract],
        )
        .await
        .with_context(|| format!("upsert model {}", r.name))?;
    Ok(())
}

pub async fn rename_model(db: &Db, old: &str, r: &ModelRecord) -> Result<()> {
    let mut client = db.client().await?;
    let tx = client.transaction().await?;
    // PK change: carry the contract over, then drop the old row.
    let contract: Option<serde_json::Value> = tx
        .query_opt("SELECT contract FROM models WHERE name = $1", &[&old])
        .await?
        .and_then(|row| row.get("contract"));
    tx.execute("DELETE FROM models WHERE name = $1", &[&old]).await?;
    tx.execute(
        "INSERT INTO models (name, run_id, predictor, dataset_id, created_at, notes, contract)
         VALUES ($1, $2, $3, $4, $5, $6, $7)
         ON CONFLICT (name) DO UPDATE SET
           run_id = EXCLUDED.run_id, predictor = EXCLUDED.predictor,
           dataset_id = EXCLUDED.dataset_id, created_at = EXCLUDED.created_at,
           notes = EXCLUDED.notes, contract = EXCLUDED.contract",
        &[&r.name, &r.run_id, &r.predictor, &r.dataset_id, &r.created_at, &r.notes, &contract],
    )
    .await?;
    tx.commit().await?;
    Ok(())
}

pub async fn delete_model(db: &Db, name: &str) -> Result<()> {
    let client = db.client().await?;
    client.execute("DELETE FROM models WHERE name = $1", &[&name]).await?;
    Ok(())
}

// ---------------- best-models group (mirror of data/best-models.json) ----------------

/// Replace the whole best_models table with the given member set, atomically.
/// The group is tiny (≈12 rows) and the JSON mirror's semantics are
/// whole-file, so a full swap is the simplest write that can't drift.
pub async fn replace_best_models(db: &Db, entries: &[BestModelEntry]) -> Result<()> {
    let mut client = db.client().await?;
    let tx = client.transaction().await?;
    tx.execute("DELETE FROM best_models", &[]).await?;
    for e in entries {
        tx.execute(
            "INSERT INTO best_models (name, rank, metric, metric_value, run_id, predictor,
                                      dataset_id, source, selected_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            &[
                &e.name,
                &(e.rank as i32),
                &e.metric,
                &e.metric_value,
                &e.run_id,
                &e.predictor,
                &e.dataset_id,
                &source_str(e.source),
                &e.selected_at,
            ],
        )
        .await
        .with_context(|| format!("insert best model {}", e.name))?;
    }
    tx.commit().await.context("commit best models")?;
    Ok(())
}

pub async fn load_best_models(db: &Db) -> Result<Vec<BestModelEntry>> {
    let client = db.client().await?;
    let rows = client
        .query("SELECT * FROM best_models ORDER BY rank", &[])
        .await
        .context("load best models")?;
    rows.iter()
        .map(|r| {
            let source: String = r.get("source");
            Ok(BestModelEntry {
                name: r.get("name"),
                rank: r.get::<_, i32>("rank") as u32,
                metric: r.get("metric"),
                metric_value: r.get("metric_value"),
                run_id: r.get("run_id"),
                predictor: r.get("predictor"),
                dataset_id: r.get("dataset_id"),
                source: serde_json::from_value(serde_json::Value::String(source))
                    .context("best model source")?,
                selected_at: r.get("selected_at"),
            })
        })
        .collect()
}

fn source_str(s: lensing_core::BestModelSource) -> &'static str {
    match s {
        lensing_core::BestModelSource::Auto => "auto",
        lensing_core::BestModelSource::Pinned => "pinned",
    }
}

// ---------------- datasets / presets (mirror) ----------------

pub async fn upsert_dataset(
    db: &Db,
    dataset_id: &str,
    created_at: Option<&str>,
    manifest: &serde_json::Value,
) -> Result<()> {
    let client = db.client().await?;
    client
        .execute(
            "INSERT INTO datasets (dataset_id, created_at, manifest) VALUES ($1, $2, $3)
             ON CONFLICT (dataset_id) DO UPDATE SET
               created_at = EXCLUDED.created_at, manifest = EXCLUDED.manifest",
            &[&dataset_id, &created_at, &manifest],
        )
        .await
        .with_context(|| format!("upsert dataset {dataset_id}"))?;
    Ok(())
}

pub async fn delete_dataset(db: &Db, dataset_id: &str) -> Result<()> {
    let client = db.client().await?;
    client.execute("DELETE FROM datasets WHERE dataset_id = $1", &[&dataset_id]).await?;
    Ok(())
}

pub async fn upsert_preset(db: &Db, name: &str, hyperparams: &serde_json::Value) -> Result<()> {
    let client = db.client().await?;
    client
        .execute(
            "INSERT INTO hp_presets (name, hyperparams) VALUES ($1, $2)
             ON CONFLICT (name) DO UPDATE SET hyperparams = EXCLUDED.hyperparams",
            &[&name, &hyperparams],
        )
        .await?;
    Ok(())
}

// ---------------- interp analyses (mirror of data/interp/*) ----------------

fn interp_from_row(row: &tokio_postgres::Row, with_result: bool) -> InterpAnalysis {
    InterpAnalysis {
        id: row.get("id"),
        tool: row.get("tool"),
        model: row.get("model"),
        dataset_id: row.get("dataset_id"),
        predictor: row.get("predictor"),
        config: row.get::<_, serde_json::Value>("config"),
        status: row.get("status"),
        error: row.get("error"),
        source: row.get("source"),
        created_at: row.get("created_at"),
        finished_at: row.get("finished_at"),
        result: if with_result { row.get::<_, Option<serde_json::Value>>("result") } else { None },
    }
}

pub async fn upsert_interp_analysis(db: &Db, a: &InterpAnalysis) -> Result<()> {
    let client = db.client().await?;
    client
        .execute(
            "INSERT INTO interp_analyses
               (id, tool, model, dataset_id, predictor, config, status, error, source,
                created_at, finished_at, result)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
             ON CONFLICT (id) DO UPDATE SET
               tool = EXCLUDED.tool, model = EXCLUDED.model, dataset_id = EXCLUDED.dataset_id,
               predictor = EXCLUDED.predictor, config = EXCLUDED.config, status = EXCLUDED.status,
               error = EXCLUDED.error, source = EXCLUDED.source, created_at = EXCLUDED.created_at,
               finished_at = EXCLUDED.finished_at,
               result = COALESCE(EXCLUDED.result, interp_analyses.result)",
            &[
                &a.id,
                &a.tool,
                &a.model,
                &a.dataset_id,
                &a.predictor,
                &a.config,
                &a.status,
                &a.error,
                &a.source,
                &a.created_at,
                &a.finished_at,
                &a.result,
            ],
        )
        .await
        .with_context(|| format!("upsert interp analysis {}", a.id))?;
    Ok(())
}

/// All analyses, newest first, WITHOUT the heavy `result` document.
pub async fn load_interp_analyses(db: &Db) -> Result<Vec<InterpAnalysis>> {
    let client = db.client().await?;
    let rows = client
        .query(
            "SELECT id, tool, model, dataset_id, predictor, config, status, error, source,
                    created_at, finished_at
             FROM interp_analyses ORDER BY created_at DESC, id DESC",
            &[],
        )
        .await
        .context("load interp analyses")?;
    Ok(rows.iter().map(|r| interp_from_row(r, false)).collect())
}

/// One analysis, WITH its `result` document.
pub async fn load_interp_analysis(db: &Db, id: &str) -> Result<Option<InterpAnalysis>> {
    let client = db.client().await?;
    let row = client
        .query_opt("SELECT * FROM interp_analyses WHERE id = $1", &[&id])
        .await?;
    Ok(row.as_ref().map(|r| interp_from_row(r, true)))
}

pub async fn delete_interp_analysis(db: &Db, id: &str) -> Result<()> {
    let client = db.client().await?;
    client.execute("DELETE FROM interp_analyses WHERE id = $1", &[&id]).await?;
    Ok(())
}

/// The ids already present, so the backfill can skip them (never clobber a live
/// row's source/curation with a disk-derived one).
pub async fn interp_analysis_ids(db: &Db) -> Result<std::collections::HashSet<String>> {
    let client = db.client().await?;
    let rows = client.query("SELECT id FROM interp_analyses", &[]).await?;
    Ok(rows.iter().map(|r| r.get::<_, String>("id")).collect())
}

/// True when a non-failed per-model SAE analysis already exists for `model`
/// (the auto-queue dedup: promotion should queue at most one).
pub async fn model_has_interp_analysis(db: &Db, model: &str) -> Result<bool> {
    let client = db.client().await?;
    let row = client
        .query_one(
            "SELECT EXISTS(
               SELECT 1 FROM interp_analyses
               WHERE model = $1 AND tool = 'model-sae' AND status <> 'failed'
             ) AS present",
            &[&model],
        )
        .await?;
    Ok(row.get::<_, bool>("present"))
}

pub async fn count(db: &Db, table: &str) -> Result<i64> {
    // Table names come from our own code, never from input.
    let client = db.client().await?;
    let row = client.query_one(&format!("SELECT COUNT(*) AS n FROM {table}"), &[]).await?;
    Ok(row.get::<_, i64>("n"))
}

pub fn status_str(s: lensing_core::RunStatus) -> &'static str {
    match s {
        lensing_core::RunStatus::Queued => "queued",
        lensing_core::RunStatus::Running => "running",
        lensing_core::RunStatus::Succeeded => "succeeded",
        lensing_core::RunStatus::Failed => "failed",
        lensing_core::RunStatus::Interrupted => "interrupted",
        lensing_core::RunStatus::Stopped => "stopped",
    }
}
