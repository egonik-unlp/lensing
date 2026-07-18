use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// `meta.json` in a run directory. Owned by lensing-server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMeta {
    pub run_id: String,
    pub dataset_id: String,
    pub predictor: String,
    pub hyperparams: serde_json::Value,
    pub status: RunStatus,
    /// RFC 3339, UTC.
    pub started_at: String,
    pub finished_at: Option<String>,
    pub exit_code: Option<i32>,
    /// Tail of stderr, kept for the UI's failed-run state.
    pub stderr_tail: Option<String>,
    /// Present once the run succeeds.
    pub metrics: Option<Metrics>,
    /// Predictor contract version the run was produced under. Runs written
    /// before the predict subcommand existed deserialize as 1.
    #[serde(default = "default_contract_version")]
    pub contract_version: u32,
    /// True when the run directory holds a loadable checkpoint: set on
    /// finish for predict-capable predictors, or mid-run on the first
    /// periodic `checkpoint` event. Gates promotion for non-succeeded runs.
    #[serde(default)]
    pub has_checkpoint: bool,
    /// Model definition (`models.toml`) this run was launched from, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_definition: Option<String>,
}

fn default_contract_version() -> u32 {
    1
}

/// The current predictor contract version written into new run metas.
pub const CONTRACT_VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    /// Enqueued for a remote training worker; not yet claimed.
    Queued,
    Running,
    Succeeded,
    Failed,
    Interrupted,
    /// Stopped early on user request; the predictor still evaluated and wrote
    /// its checkpoint, so the run is promotable like a succeeded one.
    Stopped,
}

/// `metrics.json` written by a predictor. `n_test` is universal; the regression
/// fields populate for regression runs, the classification fields for
/// binary/multiclass runs. All `Option` so a run carries only its task's
/// metrics, and old (regression-only) `metrics.json` / JSONB rows still
/// deserialize unchanged (the five fields read back as `Some`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metrics {
    pub n_test: usize,
    // ---- regression (target space) ----
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mae: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rmse: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub r2: Option<f64>,
    /// Mean absolute percentage error, as a fraction (0.25 = 25%).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mape: Option<f64>,
    /// Median absolute percentage error, as a fraction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub medape: Option<f64>,
    // ---- classification (binary + multiclass) ----
    /// Fraction correct at argmax / 0.5 threshold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accuracy: Option<f64>,
    /// Mean negative log-likelihood (cross-entropy), natural log.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logloss: Option<f64>,
    /// Binary ROC-AUC, or multiclass one-vs-rest macro-AUC.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auc: Option<f64>,
    /// Binary Brier score (mean squared error of P(class==1)).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brier: Option<f64>,
    /// Multiclass macro-averaged F1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub macro_f1: Option<f64>,
    // ---- ranking (next-item retrieval; task = ranking) ----
    /// Recall@k: fraction of test cases whose true next item appears in the
    /// top-k retrieved candidates. `k` is fixed by the run config; higher-better.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recall_at_k: Option<f64>,
    /// Mean reciprocal rank of the true next item over the ranked candidate
    /// list (0 if outside the evaluated top-N); higher-better.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mrr: Option<f64>,
    /// Hit-rate@k: fraction of cases with at least one relevant item in the
    /// top-k (== recall@k for a single held-out next item; distinct for
    /// multi-target sequences); higher-better.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hit_rate: Option<f64>,
    // ---- graded-relevance ranking (keyed by relevance-field name) ----
    /// Graded-relevance recall@k, keyed by the item field it credits (a
    /// top-k candidate sharing the truth's value on that field counts as a
    /// partial hit). Keys come from `[sequence].relevance_fields`; the
    /// framework hardcodes no field name. Absent for non-graded runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graded_recall_at_k: Option<BTreeMap<String, f64>>,
    /// Graded-relevance MRR, keyed by relevance-field name (see above).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graded_mrr: Option<BTreeMap<String, f64>>,
}

/// One element of `predictions.json` written by a predictor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub row_id: u64,
    /// Regression: target space. Classification: the true class id as f64.
    pub actual: f64,
    /// Regression: target-space value. Binary: P(class==1). Multiclass: argmax
    /// class id.
    pub predicted: f64,
    /// Class probabilities — binary `[p0, p1]`, multiclass the K-vector;
    /// absent for regression.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proba: Option<Vec<f64>>,
    /// Ranking (next-item) predictors: the top-k retrieved item ids,
    /// best-first for this case. Absent for pointwise predictors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k_ids: Option<Vec<u64>>,
}

/// One element of a predict subcommand's output file. There is no ground truth
/// at inference time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferencePrediction {
    pub row_id: u64,
    /// Regression: target-space value. Binary: P(class==1). Multiclass: argmax
    /// class id.
    pub predicted: f64,
    /// Class probabilities (binary `[p0, p1]` / multiclass K-vector); absent
    /// for regression.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proba: Option<Vec<f64>>,
    /// Ranking (next-item) predictors: the top-k retrieved item ids,
    /// best-first for this case. Absent for pointwise predictors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k_ids: Option<Vec<u64>>,
}

/// One JSON line on a predictor's stdout.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum ProgressEvent {
    Epoch { epoch: usize, total_epochs: usize, train_loss: f64, val_loss: f64 },
    Log { msg: String },
    /// Periodic checkpoint saved (`checkpoint_every` hyperparam); the run dir
    /// now holds loadable params even if training dies later.
    Checkpoint { epoch: usize },
    /// The predictor saw the STOP file and is finishing up (eval + final
    /// checkpoint) before exiting.
    Stopping,
    Done,
}

/// Compute price-space metrics from (actual, predicted) pairs.
pub fn compute_metrics(pairs: &[(f64, f64)]) -> Metrics {
    let n = pairs.len();
    assert!(n > 0, "no predictions");
    let mut abs_err = 0.0;
    let mut sq_err = 0.0;
    let mut apes: Vec<f64> = Vec::with_capacity(n);
    let mean_actual = pairs.iter().map(|(a, _)| a).sum::<f64>() / n as f64;
    let mut ss_tot = 0.0;
    for &(actual, predicted) in pairs {
        let e = predicted - actual;
        abs_err += e.abs();
        sq_err += e * e;
        ss_tot += (actual - mean_actual).powi(2);
        // APE is undefined when actual == 0 (binary / zero-valued targets, e.g.
        // a classification target). Accumulate it only over nonzero actuals;
        // for an all-positive target this is every row, so behavior is
        // unchanged — it only stops producing NaN when zeros are present.
        if actual != 0.0 {
            apes.push((e / actual).abs());
        }
    }
    apes.sort_by(|a, b| a.total_cmp(b));
    let m = apes.len();
    let medape = if m == 0 {
        0.0
    } else if m % 2 == 1 {
        apes[m / 2]
    } else {
        (apes[m / 2 - 1] + apes[m / 2]) / 2.0
    };
    Metrics {
        n_test: n,
        mae: Some(abs_err / n as f64),
        rmse: Some((sq_err / n as f64).sqrt()),
        r2: Some(if ss_tot > 0.0 { 1.0 - sq_err / ss_tot } else { 0.0 }),
        mape: Some(if m == 0 { 0.0 } else { apes.iter().sum::<f64>() / m as f64 }),
        medape: Some(medape),
        ..Default::default()
    }
}

/// Rank-based ROC-AUC (Mann–Whitney U) from (label∈{0,1}, score) pairs, with
/// tie-corrected average ranks. Returns 0.5 when one class is absent.
pub fn roc_auc(label_scores: &[(f64, f64)]) -> f64 {
    let n_pos = label_scores.iter().filter(|(y, _)| *y > 0.5).count();
    let n_neg = label_scores.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return 0.5;
    }
    let mut idx: Vec<usize> = (0..label_scores.len()).collect();
    idx.sort_by(|&a, &b| label_scores[a].1.total_cmp(&label_scores[b].1));
    // Average ranks (1-based) over groups of tied scores.
    let mut ranks = vec![0.0f64; label_scores.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && label_scores[idx[j]].1 == label_scores[idx[i]].1 {
            j += 1;
        }
        let avg = ((i + 1 + j) as f64) / 2.0; // mean of ranks (i+1)..=j
        for &k in &idx[i..j] {
            ranks[k] = avg;
        }
        i = j;
    }
    let sum_pos_ranks: f64 = label_scores
        .iter()
        .zip(&ranks)
        .filter(|((y, _), _)| *y > 0.5)
        .map(|(_, r)| *r)
        .sum();
    let u = sum_pos_ranks - (n_pos * (n_pos + 1)) as f64 / 2.0;
    u / (n_pos as f64 * n_neg as f64)
}

/// Binary classification metrics from (actual∈{0,1}, p1=P(class==1)) pairs.
/// Used by the blend / sweep recompute paths; ordinary runs report their own.
pub fn compute_binary_metrics(pairs: &[(f64, f64)]) -> Metrics {
    let n = pairs.len();
    assert!(n > 0, "no predictions");
    const EPS: f64 = 1e-15;
    let mut correct = 0usize;
    let mut logloss = 0.0;
    let mut brier = 0.0;
    for &(y, p1) in pairs {
        let p = p1.clamp(EPS, 1.0 - EPS);
        logloss -= y * p.ln() + (1.0 - y) * (1.0 - p).ln();
        brier += (p1 - y).powi(2);
        let pred = if p1 >= 0.5 { 1.0 } else { 0.0 };
        if (pred - y).abs() < 0.5 {
            correct += 1;
        }
    }
    Metrics {
        n_test: n,
        accuracy: Some(correct as f64 / n as f64),
        logloss: Some(logloss / n as f64),
        auc: Some(roc_auc(pairs)),
        brier: Some(brier / n as f64),
        ..Default::default()
    }
}

/// Multiclass metrics from true class ids + per-row probability vectors (each
/// length `k`). Reports accuracy, mean cross-entropy, macro-F1 and one-vs-rest
/// macro-AUC. Used by the blend / sweep recompute paths.
pub fn compute_multiclass_metrics(actual: &[usize], proba: &[Vec<f64>], k: usize) -> Metrics {
    let n = actual.len();
    assert!(n > 0 && n == proba.len() && k >= 2, "bad multiclass inputs");
    const EPS: f64 = 1e-15;
    let mut correct = 0usize;
    let mut logloss = 0.0;
    // Per-class confusion tallies for macro-F1.
    let mut tp = vec![0usize; k];
    let mut fp = vec![0usize; k];
    let mut fn_ = vec![0usize; k];
    for (i, &y) in actual.iter().enumerate() {
        let row = &proba[i];
        logloss -= row.get(y).copied().unwrap_or(0.0).clamp(EPS, 1.0).ln();
        let pred = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(c, _)| c)
            .unwrap_or(0);
        if pred == y {
            correct += 1;
            tp[y] += 1;
        } else {
            fp[pred] += 1;
            fn_[y] += 1;
        }
    }
    let macro_f1 = (0..k)
        .map(|c| {
            let denom = 2 * tp[c] + fp[c] + fn_[c];
            if denom == 0 {
                0.0
            } else {
                2.0 * tp[c] as f64 / denom as f64
            }
        })
        .sum::<f64>()
        / k as f64;
    let macro_auc = (0..k)
        .map(|c| {
            let ls: Vec<(f64, f64)> = actual
                .iter()
                .zip(proba)
                .map(|(&y, row)| (if y == c { 1.0 } else { 0.0 }, row.get(c).copied().unwrap_or(0.0)))
                .collect();
            roc_auc(&ls)
        })
        .sum::<f64>()
        / k as f64;
    Metrics {
        n_test: n,
        accuracy: Some(correct as f64 / n as f64),
        logloss: Some(logloss / n as f64),
        auc: Some(macro_auc),
        macro_f1: Some(macro_f1),
        ..Default::default()
    }
}

/// Ranking (next-item) metrics from per-case ranked candidate id lists
/// (best-first) and the true next-item id for each case. `k` is the cutoff.
/// `recall_at_k` = fraction of cases whose true item is within the top-k;
/// `mrr` = mean reciprocal rank of the true item (0 when absent from the list);
/// `hit_rate` = fraction of cases with the true item anywhere in the list.
/// All three are higher-better. Graded-relevance metrics are computed by the
/// predictor (it alone knows each item's relevance-field values).
pub fn compute_ranking_metrics(ranked: &[Vec<u64>], truth: &[u64], k: usize) -> Metrics {
    let n = ranked.len();
    assert!(n > 0 && n == truth.len() && k >= 1, "bad ranking inputs");
    let (mut recall, mut mrr, mut hit) = (0.0, 0.0, 0.0);
    for (cands, &t) in ranked.iter().zip(truth) {
        if let Some(pos) = cands.iter().position(|&c| c == t) {
            hit += 1.0;
            mrr += 1.0 / (pos as f64 + 1.0);
            if pos < k {
                recall += 1.0;
            }
        }
    }
    Metrics {
        n_test: n,
        recall_at_k: Some(recall / n as f64),
        mrr: Some(mrr / n as f64),
        hit_rate: Some(hit / n as f64),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranking_metrics_hand_checked() {
        // case 0: truth 7 at rank 1 (pos 0) → recall@2 hit, rr 1.0
        // case 1: truth 3 at rank 3 (pos 2) → outside k=2 (no recall), rr 1/3, hit
        // case 2: truth 9 absent → no recall, rr 0, no hit
        let ranked = vec![vec![7, 1, 2], vec![5, 8, 3], vec![1, 2, 4]];
        let truth = vec![7u64, 3, 9];
        let m = compute_ranking_metrics(&ranked, &truth, 2);
        assert!((m.recall_at_k.unwrap() - 1.0 / 3.0).abs() < 1e-9);
        assert!((m.mrr.unwrap() - (1.0 + 1.0 / 3.0) / 3.0).abs() < 1e-9);
        assert!((m.hit_rate.unwrap() - 2.0 / 3.0).abs() < 1e-9);
        let s = serde_json::to_string(&m).unwrap();
        assert!(s.contains("recall_at_k") && s.contains("mrr") && s.contains("hit_rate"));
        assert!(!s.contains("auc") && !s.contains("mae"));
    }

    /// meta.json written before contract v2 (no contract_version /
    /// has_checkpoint fields) must keep deserializing.
    #[test]
    fn old_run_meta_deserializes() {
        let old = r#"{
            "run_id": "run-20260101-000000-1-burn-mlp",
            "dataset_id": "ds-x",
            "predictor": "burn-mlp",
            "hyperparams": {"epochs": 50},
            "status": "succeeded",
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:05:00Z",
            "exit_code": 0,
            "stderr_tail": null,
            "metrics": {"mae":1.0,"rmse":2.0,"r2":0.5,"mape":0.2,"medape":0.1,"n_test":10}
        }"#;
        let meta: RunMeta = serde_json::from_str(old).unwrap();
        assert_eq!(meta.contract_version, 1);
        assert!(!meta.has_checkpoint);
        assert!(meta.from_definition.is_none());
        assert_eq!(meta.status, RunStatus::Succeeded);
    }

    #[test]
    fn stopped_status_round_trips() {
        let s = serde_json::to_string(&RunStatus::Stopped).unwrap();
        assert_eq!(s, "\"stopped\"");
        let back: RunStatus = serde_json::from_str(&s).unwrap();
        assert_eq!(back, RunStatus::Stopped);
    }

    #[test]
    fn old_regression_metrics_json_still_deserializes() {
        // The five regression fields read back as Some; classification fields None.
        let m: Metrics =
            serde_json::from_str(r#"{"mae":1.0,"rmse":2.0,"r2":0.5,"mape":0.2,"medape":0.1,"n_test":10}"#)
                .unwrap();
        assert_eq!(m.mae, Some(1.0));
        assert_eq!(m.auc, None);
        // round-trips without leaking null classification keys
        let s = serde_json::to_string(&m).unwrap();
        assert!(!s.contains("auc"));
        assert!(s.contains("\"mae\":1.0"));
    }

    #[test]
    fn roc_auc_ranks_perfectly_and_handles_ties() {
        // perfect separation → AUC 1.0
        let perfect = [(0.0, 0.1), (0.0, 0.2), (1.0, 0.8), (1.0, 0.9)];
        assert!((roc_auc(&perfect) - 1.0).abs() < 1e-9);
        // reversed → 0.0
        let rev = [(1.0, 0.1), (1.0, 0.2), (0.0, 0.8), (0.0, 0.9)];
        assert!((roc_auc(&rev) - 0.0).abs() < 1e-9);
        // all-ties → 0.5; single class → 0.5
        let ties = [(0.0, 0.5), (1.0, 0.5)];
        assert!((roc_auc(&ties) - 0.5).abs() < 1e-9);
        assert!((roc_auc(&[(1.0, 0.3), (1.0, 0.9)]) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn binary_metrics_sane() {
        let m = compute_binary_metrics(&[(1.0, 0.9), (0.0, 0.2), (1.0, 0.6), (0.0, 0.4)]);
        assert_eq!(m.n_test, 4);
        assert_eq!(m.accuracy, Some(1.0)); // all on the correct side of 0.5
        assert!(m.auc.unwrap() > 0.99);
        assert!(m.logloss.unwrap() > 0.0 && m.brier.unwrap() > 0.0);
        assert!(m.mae.is_none());
    }

    #[test]
    fn multiclass_metrics_sane() {
        // 3 classes, perfect argmax predictions
        let proba = vec![
            vec![0.8, 0.1, 0.1],
            vec![0.1, 0.7, 0.2],
            vec![0.2, 0.2, 0.6],
        ];
        let m = compute_multiclass_metrics(&[0, 1, 2], &proba, 3);
        assert_eq!(m.accuracy, Some(1.0));
        assert!((m.macro_f1.unwrap() - 1.0).abs() < 1e-9);
        assert!(m.logloss.unwrap() > 0.0);
        assert_eq!(m.n_test, 3);
    }

    #[test]
    fn new_progress_events_round_trip() {
        let ck: ProgressEvent = serde_json::from_str(r#"{"event":"checkpoint","epoch":50}"#).unwrap();
        assert!(matches!(ck, ProgressEvent::Checkpoint { epoch: 50 }));
        let st: ProgressEvent = serde_json::from_str(r#"{"event":"stopping"}"#).unwrap();
        assert!(matches!(st, ProgressEvent::Stopping));
    }
}
