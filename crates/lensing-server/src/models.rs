//! Named models: promotion of succeeded runs into self-contained model
//! directories, and one-shot predict invocations against them.
//!
//! `data/models/<name>/` layout:
//!   record.json          ModelRecord (name, run_id, predictor, dataset_id, …)
//!   contract.json        frozen featurization contract (lensing_core::Contract)
//!   pca_components.f32   copied from the training dataset
//!   hyperparams.json     copy of the run's hp.json (predictors rebuild their
//!                        architecture from it before loading weights)
//!   <everything else the predictor wrote into the run dir: checkpoints,
//!    scaler.json, model.json, …>
//!
//! A model dir is a snapshot, not a reference: it stays invocable after the
//! training dataset or run directory is deleted.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, ensure, Context, Result};
use chrono::Utc;
use lensing_core::{
    Contract, InferencePrediction, InputFields, Manifest, ModelRecord, RunStatus,
    CONTRACT_VERSION,
};
use lensing_pipeline::features::Encoder;
use lensing_pipeline::inference::{Featurizer, RawItem};
use lensing_pipeline::qdrant::{self, RawPoint};

use crate::registry;
use crate::runs;
use crate::state::AppState;

/// Files in a run dir owned by the server / training contract; everything
/// else is predictor-written state and gets copied into the model dir.
/// `STOP` is the graceful-stop marker — deleted when a run finishes, listed
/// here so a race can never copy it into a model dir.
const RUN_OWNED: &[&str] =
    &["meta.json", "hp.json", "progress.jsonl", "metrics.json", "predictions.json", "STOP"];

pub fn validate_name(name: &str) -> Result<()> {
    let ok = !name.is_empty()
        && name.len() <= 64
        && name.starts_with(|c: char| c.is_ascii_lowercase() || c.is_ascii_digit())
        && name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
    ensure!(ok, "model name must match ^[a-z0-9][a-z0-9-]{{0,63}}$");
    Ok(())
}

pub fn read_record(model_dir: &Path) -> Result<ModelRecord> {
    let text = std::fs::read_to_string(model_dir.join("record.json"))
        .with_context(|| format!("read {}/record.json", model_dir.display()))?;
    Ok(serde_json::from_str(&text)?)
}

pub fn read_contract(model_dir: &Path) -> Result<Contract> {
    let text = std::fs::read_to_string(model_dir.join("contract.json"))
        .with_context(|| format!("read {}/contract.json", model_dir.display()))?;
    Ok(serde_json::from_str(&text)?)
}

pub fn list_models(state: &AppState) -> Vec<ModelRecord> {
    let mut records = Vec::new();
    if let Ok(entries) = std::fs::read_dir(state.models_dir()) {
        for e in entries.flatten() {
            if let Ok(r) = read_record(&e.path()) {
                records.push(r);
            }
        }
    }
    records.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    records
}

/// Promote a run into `data/models/<name>/`. Succeeded and stopped runs are
/// always promotable; failed/interrupted runs only when a periodic
/// checkpoint landed (`has_checkpoint`) — the model is then built from the
/// params as they were, without final test metrics. Fails if the name is
/// taken, the predictor is train-only, or the training dataset (needed for
/// the contract snapshot) is gone.
pub fn promote(
    state: &AppState,
    name: &str,
    run_id: &str,
    notes: Option<String>,
) -> Result<ModelRecord> {
    validate_name(name)?;

    let run_dir = state.runs_dir().join(run_id);
    let meta = runs::read_meta(&run_dir).with_context(|| format!("run {run_id} not found"))?;
    ensure!(
        matches!(meta.status, RunStatus::Succeeded | RunStatus::Stopped) || meta.has_checkpoint,
        "run is {:?} and has no saved checkpoint to promote",
        meta.status
    );
    ensure!(
        meta.status != RunStatus::Running,
        "run is still running; stop it first"
    );
    let predictor = state
        .registry
        .get(&meta.predictor)
        .with_context(|| format!("predictor {} is not in the registry", meta.predictor))?;
    ensure!(
        predictor.supports_predict(),
        "predictor {} is train-only (no predict_args in registry.toml)",
        meta.predictor
    );

    let dataset_dir = state.datasets_dir().join(&meta.dataset_id);
    let manifest: Manifest = serde_json::from_str(
        &std::fs::read_to_string(dataset_dir.join("manifest.json")).with_context(|| {
            format!(
                "training dataset {} is gone; promotion snapshots its manifest",
                meta.dataset_id
            )
        })?,
    )?;

    let model_dir = state.models_dir().join(name);
    // create_dir (not create_dir_all) doubles as the atomic name-taken check.
    std::fs::create_dir(&model_dir)
        .map_err(|_| anyhow::anyhow!("model name {name:?} is already taken"))?;

    let result = (|| -> Result<ModelRecord> {
        // Contract: the frozen featurization spec.
        let encoder = Encoder::from_columns(&manifest.columns)?;
        let contract = Contract {
            contract_version: CONTRACT_VERSION,
            n_cols: manifest.n_cols,
            columns: manifest.columns.clone(),
            pca: manifest.pca.clone(),
            target: manifest.target.clone(),
            feature_config: manifest.feature_config.clone(),
            input_fields: InputFields {
                required_numeric: encoder.numeric_fields(),
                required_categorical: encoder.categorical_groups(),
                embedding_dim: manifest.pca.components_shape[1],
            },
            numerics_collection: manifest
                .numerics
                .as_ref()
                .and_then(|n| n.reconcile_collection.clone()),
            imputation: manifest.imputation.clone(),
        };
        std::fs::write(model_dir.join("contract.json"), serde_json::to_vec_pretty(&contract)?)?;

        // Featurization binary + hyperparams snapshot.
        std::fs::copy(
            dataset_dir.join("pca_components.f32"),
            model_dir.join("pca_components.f32"),
        )
        .context("copy pca_components.f32")?;
        std::fs::copy(run_dir.join("hp.json"), model_dir.join("hyperparams.json"))
            .context("copy hp.json")?;

        // Everything the predictor wrote (checkpoints, scaler, …), including
        // subdirectories — meta-predictors nest member model dirs (e.g. the
        // blend's members/<i>/, whose inner metrics.json etc. are
        // predictor-written state, so RUN_OWNED applies at top level only).
        let copied = copy_predictor_files(&run_dir, &model_dir, true)?;
        ensure!(copied > 0, "run {run_id} has no checkpoint files to promote");

        let record = ModelRecord {
            name: name.to_string(),
            run_id: run_id.to_string(),
            predictor: meta.predictor.clone(),
            dataset_id: meta.dataset_id.clone(),
            created_at: Utc::now().to_rfc3339(),
            notes,
        };
        std::fs::write(model_dir.join("record.json"), serde_json::to_vec_pretty(&record)?)?;
        Ok(record)
    })();

    match &result {
        Ok(record) => {
            if let Some(sink) = &state.db_sink {
                let contract = read_contract(&model_dir)
                    .ok()
                    .and_then(|c| serde_json::to_value(c).ok());
                sink.model_upsert(record, contract);
                // Full snapshot into the database so inference nodes can
                // materialize this model with no shared filesystem.
                match collect_dir_files(&model_dir) {
                    Ok(files) => sink.model_artifacts(name, files),
                    Err(e) => eprintln!("[lensing-server] model artifact upload failed: {e:#}"),
                }
            }
        }
        Err(_) => {
            let _ = std::fs::remove_dir_all(&model_dir);
        }
    }
    result
}

/// Every file under `dir` as (relative path, bytes).
fn collect_dir_files(dir: &Path) -> Result<Vec<(String, Vec<u8>)>> {
    fn rec(base: &Path, dir: &Path, out: &mut Vec<(String, Vec<u8>)>) -> Result<()> {
        for e in std::fs::read_dir(dir)?.flatten() {
            let path = e.path();
            let ft = e.file_type()?;
            if ft.is_symlink() {
                continue;
            }
            if ft.is_dir() {
                rec(base, &path, out)?;
            } else {
                let rel = path.strip_prefix(base).unwrap().to_string_lossy().into_owned();
                out.push((rel, std::fs::read(&path)?));
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    rec(dir, dir, &mut out)?;
    Ok(out)
}

/// Recursively copy predictor-written files from `src` into `dst`.
/// `RUN_OWNED` names are excluded at the top level only; atomic-write temps
/// (`.tmp`/`-tmp` in the name) are excluded at any depth — a kill
/// mid-checkpoint can leave one behind (model-tmp.mpk, model.jld2.tmp) and a
/// meta-predictor's member training artifacts (ds-tmp/) are train-time-only.
/// Symlinks are skipped so a model dir stays a self-contained snapshot.
fn copy_predictor_files(src: &Path, dst: &Path, top_level: bool) -> Result<usize> {
    let mut copied = 0usize;
    for e in std::fs::read_dir(src)?.flatten() {
        let file_name = e.file_name();
        let fname = file_name.to_string_lossy();
        if top_level && RUN_OWNED.contains(&fname.as_ref()) {
            continue;
        }
        if fname.contains(".tmp") || fname.contains("-tmp") {
            continue;
        }
        let ft = e.file_type()?;
        if ft.is_symlink() {
            continue;
        }
        if ft.is_dir() {
            let sub = dst.join(&file_name);
            std::fs::create_dir_all(&sub)?;
            copied += copy_predictor_files(&e.path(), &sub, false)?;
        } else {
            std::fs::copy(e.path(), dst.join(&file_name))
                .with_context(|| format!("copy {fname}"))?;
            copied += 1;
        }
    }
    Ok(copied)
}

/// Rename a promoted model: move the directory and rewrite `record.name`.
/// Best-effort against concurrent predicts (they resolve the model dir once
/// up front, so an in-flight predict may fail against the old path).
pub fn rename_model(state: &AppState, old: &str, new: &str) -> Result<ModelRecord> {
    validate_name(new)?;
    let old_dir = state.models_dir().join(old);
    ensure!(old_dir.join("record.json").is_file(), "model {old} not found");
    let new_dir = state.models_dir().join(new);
    ensure!(!new_dir.exists(), "model name {new:?} is already taken");
    std::fs::rename(&old_dir, &new_dir)
        .with_context(|| format!("rename {old} -> {new}"))?;
    let mut record = read_record(&new_dir)?;
    record.name = new.to_string();
    std::fs::write(new_dir.join("record.json"), serde_json::to_vec_pretty(&record)?)?;
    if let Some(sink) = &state.db_sink {
        sink.model_rename(old, &record);
        sink.model_artifacts_rename(old, new);
        // The renamed dir's record.json changed; refresh the snapshot.
        match collect_dir_files(&new_dir) {
            Ok(files) => sink.model_artifacts(new, files),
            Err(e) => eprintln!("[lensing-server] model artifact refresh failed: {e:#}"),
        }
    }
    Ok(record)
}

pub fn delete_model(state: &AppState, name: &str) -> Result<()> {
    let model_dir = state.models_dir().join(name);
    ensure!(model_dir.join("record.json").is_file(), "model {name} not found");
    std::fs::remove_dir_all(&model_dir)?;
    if let Some(sink) = &state.db_sink {
        sink.model_delete(name);
        sink.model_artifacts_delete(name);
    }
    Ok(())
}

/// One predict request: raw items and/or Qdrant point ids.
#[derive(Debug, serde::Deserialize)]
pub struct PredictRequest {
    #[serde(default)]
    pub items: Vec<serde_json::Value>,
    #[serde(default)]
    pub point_ids: Vec<u64>,
}

#[derive(Debug, serde::Serialize)]
pub struct PredictResponse {
    pub predictions: Vec<InferencePrediction>,
    pub warnings: Vec<String>,
}

/// Predict failures, separated so the API layer can map status codes.
pub enum PredictError {
    /// Unknown model name → 404.
    NotFound,
    /// Caller-fixable input problem (bad item, wrong embedding dim) → 400.
    BadInput(String),
    /// Everything else (predictor crash, IO) → 500.
    Internal(anyhow::Error),
}

impl From<anyhow::Error> for PredictError {
    fn from(e: anyhow::Error) -> Self {
        PredictError::Internal(e)
    }
}

/// Run a one-shot predict against a promoted model. Featurizes inputs under
/// the model's frozen contract, writes a temp input mini-artifact, spawns
/// the predictor's predict subcommand under the training semaphore, returns
/// target-space predictions.
pub async fn predict(
    state: Arc<AppState>,
    name: String,
    req: PredictRequest,
) -> Result<PredictResponse, PredictError> {
    let model_dir = state.models_dir().join(&name);
    let record = read_record(&model_dir).map_err(|_| PredictError::NotFound)?;
    let predictor = state
        .registry
        .get(&record.predictor)
        .with_context(|| format!("predictor {} is not in the registry", record.predictor))?
        .clone();
    let (command, args_template) = predictor
        .predict_invocation()
        .with_context(|| format!("predictor {} is train-only", record.predictor))?;
    let (command, args_template) = (command.to_string(), args_template.to_vec());

    if req.items.is_empty() && req.point_ids.is_empty() {
        return Err(PredictError::BadInput("request must contain items and/or point_ids".into()));
    }

    // Resolve + featurize on a blocking thread (Qdrant fetch is blocking
    // reqwest; PCA projection is CPU work). Errors here are caller-fixable.
    let st = state.clone();
    let model_dir2 = model_dir.clone();
    let prepared = tokio::task::spawn_blocking(move || -> Result<(PathBuf, Vec<String>)> {
        let featurizer = Featurizer::from_model_dir(&model_dir2)?;

        let mut points: Vec<RawPoint> = if req.point_ids.is_empty() {
            Vec::new()
        } else {
            qdrant::fetch_points(&st.qdrant_url, &st.collection, &req.point_ids, &st.domain)?
        };
        // Reconciled contracts: repeat the build-time join for collection
        // points (the build collection dropped the size fields). Only the
        // point_ids-sourced points — caller items get fallback ids that must
        // not be joined against real companion rows.
        if let Some(companion) =
            featurizer.numerics_collection(st.domain.companion_collection().as_deref())
        {
            if !points.is_empty() {
                lensing_pipeline::numerics::reconcile(
                    &mut points,
                    &companion,
                    &st.qdrant_url,
                    &st.domain,
                )?;
            }
        }
        for (i, item) in req.items.into_iter().enumerate() {
            let raw: RawItem = serde_json::from_value(item)
                .map_err(|e| anyhow::anyhow!("item {i}: {e}"))?;
            points.push(raw.into_point(i));
        }
        // Same per-point cleanup training applied (decimal repair + optional
        // text backfill) so predict features cannot drift from trained ones.
        if featurizer.raw_numerics() {
            lensing_pipeline::numerics::normalize(
                &mut points,
                featurizer.area_content_backfill(),
                &st.domain,
            );
        }

        // Warnings first (they report what was missing), then the same
        // frozen-median fill training applied.
        let warnings = featurizer.warnings(&points);
        if let Some(imp) = featurizer.imputation() {
            lensing_pipeline::numerics::impute(&mut points, imp);
        }
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let input_dir = model_dir2.join("tmp").join(format!("in-{nanos:x}"));
        featurizer.write_input_dir(&input_dir, &points)?;
        Ok((input_dir, warnings))
    })
    .await
    .context("featurize task panicked")?;
    let (input_dir, warnings) = match prepared {
        Ok(p) => p,
        Err(e) => return Err(PredictError::BadInput(format!("{e:#}"))),
    };

    // Always clean the temp dir up, success or failure.
    let result = run_predict_process(&state, &command, &args_template, &model_dir, &input_dir).await;
    let _ = std::fs::remove_dir_all(&input_dir);

    Ok(PredictResponse { predictions: result?, warnings })
}

async fn run_predict_process(
    state: &AppState,
    command: &str,
    args_template: &[String],
    model_dir: &Path,
    input_dir: &Path,
) -> Result<Vec<InferencePrediction>> {
    let output_file = input_dir.join("out.json");
    let args = registry::substitute(
        args_template,
        &[
            ("model", model_dir.to_string_lossy().into_owned()),
            ("input", input_dir.to_string_lossy().into_owned()),
            ("output", output_file.to_string_lossy().into_owned()),
        ],
    );

    // Same slots as training: a burst of predicts can't oversubscribe CPU.
    let _permit = state.run_slots.clone().acquire_owned().await?;
    let (exit_code, stderr_tail) =
        runs::spawn_and_capture(command, &args, &state.root, &|_line| {}).await?;
    if exit_code != 0 {
        bail!("predictor exited {exit_code}: {}", stderr_tail.trim());
    }

    let text = std::fs::read_to_string(&output_file)
        .context("predictor exited 0 but wrote no output file")?;
    Ok(serde_json::from_str(&text).context("parse predictor output")?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nested member dirs (blend) are copied; RUN_OWNED applies at top level
    /// only; tmp-named entries are skipped at any depth.
    #[test]
    fn promote_copy_recurses_with_top_level_exclusions() {
        let src = std::env::temp_dir().join(format!("lensing-copy-src-{}", std::process::id()));
        let dst = std::env::temp_dir().join(format!("lensing-copy-dst-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&src);
        let _ = std::fs::remove_dir_all(&dst);
        std::fs::create_dir_all(src.join("members/0/ds-tmp")).unwrap();
        std::fs::write(src.join("model.ubj"), b"m").unwrap();
        std::fs::write(src.join("metrics.json"), b"{}").unwrap(); // RUN_OWNED, top level
        std::fs::write(src.join("model-tmp.mpk"), b"t").unwrap(); // tmp, top level
        std::fs::write(src.join("members/0/metrics.json"), b"{}").unwrap(); // nested: copied
        std::fs::write(src.join("members/0/scaler.json"), b"{}").unwrap();
        std::fs::write(src.join("members/0/ds-tmp/features.f32"), b"x").unwrap(); // tmp dir: skipped
        std::fs::create_dir_all(&dst).unwrap();

        let copied = copy_predictor_files(&src, &dst, true).unwrap();
        assert_eq!(copied, 3, "model.ubj + 2 nested member files");
        assert!(dst.join("model.ubj").is_file());
        assert!(!dst.join("metrics.json").exists());
        assert!(!dst.join("model-tmp.mpk").exists());
        assert!(dst.join("members/0/metrics.json").is_file());
        assert!(dst.join("members/0/scaler.json").is_file());
        assert!(!dst.join("members/0/ds-tmp").exists());

        let _ = std::fs::remove_dir_all(&src);
        let _ = std::fs::remove_dir_all(&dst);
    }
}
