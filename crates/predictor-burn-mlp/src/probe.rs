//! Layer-wise linear probes (Alain & Bengio 2016, arXiv:1610.01644) for the
//! burn MLP — the family-specific glue around the shared `lensing_interp` engine.
//!
//! Here we load the trained MLP, capture the representation at each stage (the
//! raw input, then each hidden layer's post-activation output) using the model's
//! own scaler, and hand those activation matrices to `lensing_interp`, which fits a
//! cheap closed-form ridge probe per stage and reports how linearly decodable
//! the target is at that depth. The rising curve is a window on how the
//! architecture builds its prediction up; the gap to the model's own output is
//! what the final (trained) layer still adds.

use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use serde_json::json;

use lensing_core::Dataset;
use lensing_interp::{
    feature_group_stages, fit_stages, gather_targets, reference_stage, Stage, StageKind,
};

use crate::model::{self, Activation};
use crate::scaler::Scaler;

/// Minimal view of a model dir's `hyperparams.json` — just what we need to
/// rebuild and run the architecture.
#[derive(Deserialize)]
struct HpSnapshot {
    #[serde(default = "default_hidden")]
    hidden: Vec<usize>,
    #[serde(default)]
    dropout: f64,
    #[serde(default)]
    activation: Activation,
    #[serde(default = "default_batch")]
    batch_size: usize,
}
fn default_hidden() -> Vec<usize> {
    vec![256, 128]
}
fn default_batch() -> usize {
    256
}

/// Run the analysis end-to-end and write a JSON result to `output`.
pub fn run(
    model_dir: &Path,
    dataset_dir: &Path,
    output: &Path,
    lambda_override: f64,
    emit: &dyn Fn(serde_json::Value),
) -> Result<()> {
    let hp: HpSnapshot = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("hyperparams.json"))
            .context("read hyperparams.json from model dir")?,
    )
    .context("parse hyperparams.json")?;
    let scaler = Scaler::load(&model_dir.join("scaler.json"))
        .context("load scaler.json from model dir")?;
    let n_cols = scaler.mean.len();

    let ds = Dataset::load(dataset_dir)?;
    ensure!(
        ds.manifest.n_cols == n_cols,
        "model was fit on {n_cols} columns, dataset has {}",
        ds.manifest.n_cols
    );
    let bs = hp.batch_size.max(1);
    emit(json!({"event":"log","msg":format!(
        "layer-probe: model {:?} ({} activation), dataset {} ({} train / {} test, {} cols)",
        hp.hidden, hp.activation.name(), ds.manifest.dataset_id,
        ds.train_idx.len(), ds.test_idx.len(), n_cols)}));

    let mlp = model::load(&model_dir.join("model"), n_cols, &hp.hidden, hp.dropout, hp.activation)?;

    // Standardize all rows with the model's own scaler, then materialize the
    // representation at every stage (input + each hidden layer).
    let xs = scaler.transform(&ds.features);
    emit(json!({"event":"log","msg":"capturing per-stage activations"}));
    let captured = model::capture_stage_activations(&mlp, &xs, n_cols, bs);
    // The model's own pipeline: input → each hidden layer. This is the clean
    // depth axis the UI plots (the model-output reference is appended after
    // fitting); the feature-family probes go on a separate axis below.
    let net_stages: Vec<Stage> = captured
        .into_iter()
        .enumerate()
        .map(|(k, la)| Stage {
            name: la.name,
            kind: if k == 0 { StageKind::Input } else { StageKind::Hidden },
            dim: la.dim,
            values: la.values,
        })
        .collect();

    // Targets (transformed/log space) + the same ≈2× max-train-target clamp the
    // model uses when projecting back to target space.
    let transform = ds.manifest.target.transform;
    let ytr = gather_targets(&ds.target, &ds.train_idx);
    let yte = gather_targets(&ds.target, &ds.test_idx);
    let ytr_max = ytr.iter().cloned().fold(f64::MIN, f64::max);
    let clamp_hi = ytr_max + std::f64::consts::LN_2;

    let mut results = fit_stages(
        &net_stages, &ds.train_idx, &ds.test_idx, &ytr, &yte, transform, clamp_hi, lambda_override,
        emit,
    );

    // Reference endpoint: the model's OWN prediction (the trained output head),
    // not a fitted probe — the curve should approach this from below.
    let model_pred: Vec<f64> = model::predict_loaded(&mlp, &xs, n_cols, bs)
        .into_iter()
        .map(|v| v as f64)
        .collect();
    results.push(reference_stage(
        results.len(),
        "output (model)",
        &model_pred,
        &ds.train_idx,
        &ds.test_idx,
        &ytr,
        &yte,
        transform,
        clamp_hi,
    ));

    // Feature-family attribution probes on the input columns (PCA vs metadata,
    // per one-hot group / numeric field) — a separate axis from the depth curve,
    // so a probe on each family shows where the decodable target signal lives.
    emit(json!({"event":"log","msg":"fitting feature-group probes"}));
    let group_stages = feature_group_stages(&xs, n_cols, &ds.manifest.columns);
    let feature_groups = fit_stages(
        &group_stages, &ds.train_idx, &ds.test_idx, &ytr, &yte, transform, clamp_hi,
        lambda_override, emit,
    );

    let out = json!({
        "tool": "layer-probe",
        "method": "Alain & Bengio 2016 (arXiv:1610.01644) — ridge probe per stage",
        "model_dir": model_dir.file_name().and_then(|s| s.to_str()).unwrap_or(""),
        "dataset_id": ds.manifest.dataset_id,
        "n_train": ds.train_idx.len(),
        "n_test": ds.test_idx.len(),
        "hidden": hp.hidden,
        "activation": hp.activation.name(),
        "target_transform": transform,
        "stages": results,
        "feature_groups": feature_groups,
    });
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(output, serde_json::to_vec_pretty(&out)?)
        .with_context(|| format!("write {}", output.display()))?;
    emit(json!({"event":"log","msg":format!("wrote {}", output.display())}));
    emit(json!({"event":"done"}));
    Ok(())
}
