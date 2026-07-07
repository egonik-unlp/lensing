//! Stages, per-stage probe results, and the orchestration that fits a ridge
//! probe on each stage of a model's forward pass.
//!
//! A *stage* is a named `[n_rows, dim]` activation matrix (all rows, row-major);
//! the predictor crate captures these from its own forward pass and hands them
//! here. This module is fully family-agnostic — it never references `burn`, a
//! specific architecture, or the domain — so any model that can emit named
//! stages is probed for free.

use serde::Serialize;
use serde_json::json;

use lensing_core::{compute_metrics, ColumnDesc, ColumnKind, TargetTransform};

use crate::ridge::fit_layer_probe;

/// What a stage represents, so the UI can group/colour it. `Input` is the raw
/// feature vector the model consumes; `FeatureGroup` is a subset of the input
/// columns (a feature-family attribution probe); `Hidden` is a post-activation
/// layer/block representation; `Output` is the model's own prediction (a
/// reference anchor, not a fitted probe).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageKind {
    Input,
    FeatureGroup,
    Hidden,
    Output,
}

/// One captured stage: row-major `[n_rows, dim]` activations to fit a probe on.
pub struct Stage {
    pub name: String,
    pub kind: StageKind,
    pub dim: usize,
    /// All rows (train and test), row-major `[n_rows, dim]`. Train/test are
    /// selected by index in [`fit_stages`].
    pub values: Vec<f32>,
}

/// The decodability of the target at one stage. `*_log` are in transformed
/// (log) target space — the probe's native objective; the target-space metrics
/// invert the transform so they compare directly against the model's own MAE.
#[derive(Debug, Clone, Serialize)]
pub struct StageResult {
    pub index: usize,
    pub name: String,
    pub kind: StageKind,
    pub dim: usize,
    /// `null` for a non-fitted reference stage (the model output).
    pub lambda: Option<f64>,
    pub train_r2_log: f64,
    pub test_r2_log: f64,
    pub test_r2_target: f64,
    pub test_mae: f64,
    pub test_medape: f64,
}

/// Gather `idx` rows out of a row-major `[n_rows, dim]` f32 matrix into f64.
fn gather_rows(values: &[f32], dim: usize, idx: &[u32]) -> Vec<f64> {
    let mut out = Vec::with_capacity(idx.len() * dim);
    for &i in idx {
        let r = i as usize * dim;
        out.extend(values[r..r + dim].iter().map(|&v| v as f64));
    }
    out
}

/// Gather the transformed target for `idx` rows as f64 — a convenience for
/// callers that hold the dataset's `target` (length n_rows).
pub fn gather_targets(target: &[f32], idx: &[u32]) -> Vec<f64> {
    idx.iter().map(|&i| target[i as usize] as f64).collect()
}

/// A directly-usable ridge probe fit — the same closed-form solver
/// [`fit_stages`] uses, exposed for callers (e.g. the embedding probe) that hold
/// arbitrary `[n, dim]` row-major f64 design matrices rather than stages.
pub struct RidgeFit {
    pub lambda: f64,
    pub train_pred: Vec<f64>,
    pub test_pred: Vec<f64>,
}

/// Fit ridge `X_train → y_train` (log space), λ by internal hold-out CV unless
/// `lambda_override > 0`, and return log-space predictions for both splits.
pub fn ridge(xtr: &[f64], ytr: &[f64], xte: &[f64], dim: usize, lambda_override: f64) -> RidgeFit {
    let f = crate::ridge::fit_layer_probe(xtr, ytr, xte, dim, lambda_override);
    RidgeFit { lambda: f.lambda, train_pred: f.train_pred_log, test_pred: f.test_pred_log }
}

fn r2(pred: &[f64], actual: &[f64]) -> f64 {
    let n = actual.len().max(1) as f64;
    let mean = actual.iter().sum::<f64>() / n;
    let ss_tot: f64 = actual.iter().map(|a| (a - mean).powi(2)).sum::<f64>().max(1e-12);
    let ss_res: f64 = pred.iter().zip(actual).map(|(p, a)| (a - p).powi(2)).sum();
    1.0 - ss_res / ss_tot
}

/// Build a result: log-space R² (train + test) plus target-space MAE/medAPE/R²
/// after inverting the target transform (test predictions clamped like the
/// model, then floored at 0).
#[allow(clippy::too_many_arguments)]
fn build_result(
    index: usize,
    name: &str,
    kind: StageKind,
    dim: usize,
    lambda: Option<f64>,
    train_pred_log: &[f64],
    ytr_log: &[f64],
    test_pred_log: &[f64],
    yte_log: &[f64],
    clamp_hi: f64,
    transform: TargetTransform,
) -> StageResult {
    let pairs: Vec<(f64, f64)> = test_pred_log
        .iter()
        .zip(yte_log)
        .map(|(&p, &a)| (transform.invert(a), transform.invert(p.min(clamp_hi)).max(0.0)))
        .collect();
    let m = compute_metrics(&pairs);
    StageResult {
        index,
        name: name.to_string(),
        kind,
        dim,
        lambda,
        train_r2_log: r2(train_pred_log, ytr_log),
        test_r2_log: r2(test_pred_log, yte_log),
        test_r2_target: m.r2.unwrap_or(f64::NAN),
        test_mae: m.mae.unwrap_or(f64::NAN),
        test_medape: m.medape.map(|v| v * 100.0).unwrap_or(f64::NAN),
    }
}

/// Fit a ridge probe on each stage and score it. Results are indexed from
/// `start_index` (so callers can append reference stages after). `ytr_log` /
/// `yte_log` are the transformed targets for the train / test splits (see
/// [`gather_targets`]); `lambda_override > 0` forces λ, else it's chosen per
/// stage by internal CV.
#[allow(clippy::too_many_arguments)]
pub fn fit_stages(
    stages: &[Stage],
    train_idx: &[u32],
    test_idx: &[u32],
    ytr_log: &[f64],
    yte_log: &[f64],
    transform: TargetTransform,
    clamp_hi: f64,
    lambda_override: f64,
    emit: &dyn Fn(serde_json::Value),
) -> Vec<StageResult> {
    let mut out = Vec::with_capacity(stages.len());
    for (k, stage) in stages.iter().enumerate() {
        let index = start_index(stages, k);
        let htr = gather_rows(&stage.values, stage.dim, train_idx);
        let hte = gather_rows(&stage.values, stage.dim, test_idx);
        let probe = fit_layer_probe(&htr, ytr_log, &hte, stage.dim, lambda_override);
        let r = build_result(
            index,
            &stage.name,
            stage.kind,
            stage.dim,
            Some(probe.lambda),
            &probe.train_pred_log,
            ytr_log,
            &probe.test_pred_log,
            yte_log,
            clamp_hi,
            transform,
        );
        emit(json!({"event":"log","msg":format!(
            "stage {index} {:<18} dim {:>4}  test logR² {:>6.3}  MAE {:>8.0}  medAPE {:>5.1}%  (λ={})",
            stage.name, stage.dim, r.test_r2_log, r.test_mae, r.test_medape, probe.lambda)}));
        out.push(r);
    }
    out
}

// Stages are numbered from 0 in fit order; `start_index` is just `k` today but
// kept as a seam in case callers want to offset.
fn start_index(_stages: &[Stage], k: usize) -> usize {
    k
}

/// A non-fitted reference stage built from the model's own per-row predictions
/// (transformed/log space, length n_rows) — the curve should approach this from
/// below. `kind` is [`StageKind::Output`].
#[allow(clippy::too_many_arguments)]
pub fn reference_stage(
    index: usize,
    name: &str,
    pred_log_all: &[f64],
    train_idx: &[u32],
    test_idx: &[u32],
    ytr_log: &[f64],
    yte_log: &[f64],
    transform: TargetTransform,
    clamp_hi: f64,
) -> StageResult {
    let tr: Vec<f64> = train_idx.iter().map(|&i| pred_log_all[i as usize]).collect();
    let te: Vec<f64> = test_idx.iter().map(|&i| pred_log_all[i as usize]).collect();
    build_result(index, name, StageKind::Output, 1, None, &tr, ytr_log, &te, yte_log, clamp_hi, transform)
}

/// Build feature-family probe stages from the (already-assembled) input matrix
/// and the dataset's column descriptors: one stage for all PCA columns, one for
/// all metadata columns, and one per one-hot source group / numeric field.
/// Empty groups are skipped. The ridge engine re-standardizes each stage, so
/// whether `input_all` is raw or scaled does not change the result.
pub fn feature_group_stages(input_all: &[f32], n_cols: usize, columns: &[ColumnDesc]) -> Vec<Stage> {
    use std::collections::BTreeMap;

    let mut pca: Vec<usize> = Vec::new();
    let mut metadata: Vec<usize> = Vec::new();
    let mut onehot: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut numeric: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, c) in columns.iter().enumerate() {
        match &c.kind {
            ColumnKind::Pca { .. } => pca.push(i),
            ColumnKind::Numeric { field, group } => {
                metadata.push(i);
                numeric.entry(group.clone().unwrap_or_else(|| field.clone())).or_default().push(i);
            }
            ColumnKind::Onehot { group, .. } => {
                metadata.push(i);
                onehot.entry(group.clone()).or_default().push(i);
            }
        }
    }

    let n_rows = if n_cols == 0 { 0 } else { input_all.len() / n_cols };
    let project = |cols: &[usize]| -> Vec<f32> {
        let mut out = Vec::with_capacity(n_rows * cols.len());
        for r in 0..n_rows {
            let base = r * n_cols;
            for &c in cols {
                out.push(input_all[base + c]);
            }
        }
        out
    };
    let stage = |name: String, cols: &[usize]| Stage {
        name,
        kind: StageKind::FeatureGroup,
        dim: cols.len(),
        values: project(cols),
    };

    let mut stages = Vec::new();
    if !pca.is_empty() {
        stages.push(stage("input:pca".into(), &pca));
    }
    // Only worth a combined "metadata" stage when it isn't identical to a single
    // group below (i.e. there is more than one metadata family).
    if !metadata.is_empty() && (onehot.len() + numeric.len()) > 1 {
        stages.push(stage("input:metadata".into(), &metadata));
    }
    for (g, cols) in &onehot {
        stages.push(stage(format!("input:onehot:{g}"), cols));
    }
    for (f, cols) in &numeric {
        stages.push(stage(format!("input:numeric:{f}"), cols));
    }
    stages
}
