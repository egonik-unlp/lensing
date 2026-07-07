//! P1 — the embedding probe (the decisive "absent vs unused" diagnostic).
//!
//! Reads ONE dataset artifact (no model, no Qdrant) and asks: is a hard segment's
//! target error *information-limited* (signal genuinely ABSENT from the embeddings
//! → go scrape more fields) or merely *information-unused* (signal PRESENT but
//! discarded by PCA-128 or by a global fit → recover it)? It decides by fitting
//! linear (ridge) + nonlinear (MLP) probes on the embedding block — raw vs the
//! first 128 PCA columns — sliced into segments, against a per-segment-median
//! floor, plus a segment-class logistic decode.
//!
//! The split axis is configurable: any one-hot categorical group in the dataset
//! (any one-hot categorical group; default: the domain's interp segment field). Segments
//! are built data-driven — a pooled `all` reference plus every value of the chosen
//! group with enough test rows — and the segment with the worst per-segment-median
//! floor (the hardest tail) drives the absent-vs-unused verdict.
//!
//! A full-rank PCA (pca_dims=1536) is a lossless rotation of the raw embeddings,
//! so "raw" = all PCA columns and "pca128" = the first 128. Linear/MLP probes are
//! rotation-invariant, so this is the canonical raw-vs-PCA-128 contrast.
//!
//! The linear arm reuses `lensing_interp`'s ridge solver; the MLP/logistic
//! arms use burn.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::json;

use lensing_core::{ColumnKind, Dataset};

use crate::nn_probe;

/// Name of the pooled, all-rows reference segment.
const POOLED: &str = "all (reference)";

/// A segment needs at least this many test rows to fit a probe worth trusting.
const MIN_TEST: usize = 15;

// Decision thresholds (heuristics; mirror the script).
const LIFT_PRESENT: f64 = 0.10; // ≥10% target-MAE cut vs segment-median ⇒ "present"
const LOGR2_PRESENT: f64 = 0.10; // AND log-space R² ≥ 0.10 on the segment
const GAP_DISCARDED: f64 = 0.05; // raw beats pca128 by ≥5% MAE ⇒ "PCA discards it"

#[derive(Serialize, Clone)]
pub struct Metrics {
    pub mae: f64,
    pub medape: f64,
    pub target_r2: f64,
    pub log_r2: f64,
}

#[derive(Serialize, Default)]
struct ViewResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    type_median: Option<Metrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    global_linear: Option<Metrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seg_linear: Option<Metrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seg_mlp: Option<Metrics>,
    n_test: usize,
    n_train_seg: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    skip: Option<String>,
}

#[derive(Serialize)]
struct ViewOut {
    view: String,
    #[serde(flatten)]
    result: ViewResult,
}

#[derive(Serialize)]
struct SegmentOut {
    name: String,
    views: Vec<ViewOut>,
}

#[derive(Serialize)]
struct SegmentDecode {
    accuracy: f64,
    macro_f1: f64,
    majority_acc: f64,
    /// AUC of the hardest segment's value vs. all others, from its softmax column.
    #[serde(skip_serializing_if = "Option::is_none")]
    hardest_vs_rest_auc: Option<f64>,
    /// The value the AUC above is computed for (the hardest segment).
    #[serde(skip_serializing_if = "Option::is_none")]
    hardest_value: Option<String>,
    n_test: usize,
}

#[derive(Serialize)]
struct Report {
    tool: &'static str,
    dataset: String,
    /// The one-hot group the rows were sliced by.
    split_by: String,
    /// All one-hot groups present in the dataset (so a caller can re-run on another).
    available_splits: Vec<String>,
    emb_dim: usize,
    n_rows: usize,
    n_train: usize,
    n_test: usize,
    views: Vec<String>,
    value_counts: BTreeMap<String, usize>,
    segments: Vec<SegmentOut>,
    #[serde(skip_serializing_if = "Option::is_none")]
    segment_decode: Option<SegmentDecode>,
    verdict: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_vs_pca128_gap: Option<f64>,
}

// ---------------------------------------------------------------------------
// math helpers
// ---------------------------------------------------------------------------

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Mirror the lab's clamp_output so a probe's tail can't expm1-explode and wreck
/// target-space metrics (~2× max / 0.5× min train target).
fn clamp_log(pred: &[f64], ytr_log: &[f64]) -> Vec<f64> {
    let (mut pmax, mut pmin) = (f64::MIN, f64::MAX);
    for &y in ytr_log {
        let p = y.exp_m1();
        pmax = pmax.max(p);
        pmin = pmin.min(p);
    }
    let lo = (pmin * 0.5).max(1.0).ln_1p();
    let hi = (pmax * 2.0).ln_1p();
    pred.iter().map(|&p| p.clamp(lo, hi)).collect()
}

/// Log-space R² (robust information read) + target-space MAE/medAPE/R²
/// (leaderboard-comparable), on clamped predictions.
fn metrics(y_log: &[f64], pred_log: &[f64], ytr_log: &[f64]) -> Metrics {
    let pred = clamp_log(pred_log, ytr_log);
    let n = y_log.len() as f64;
    let ymean = y_log.iter().sum::<f64>() / n;
    let ss_res: f64 = y_log.iter().zip(&pred).map(|(y, p)| (y - p).powi(2)).sum();
    let ss_tot: f64 = y_log.iter().map(|y| (y - ymean).powi(2)).sum::<f64>().max(1.0);
    let log_r2 = 1.0 - ss_res / ss_tot;

    let yp: Vec<f64> = y_log.iter().map(|y| y.exp_m1()).collect();
    let pp: Vec<f64> = pred.iter().map(|p| p.exp_m1()).collect();
    let mae = yp.iter().zip(&pp).map(|(y, p)| (y - p).abs()).sum::<f64>() / n;
    let mut ape: Vec<f64> = yp.iter().zip(&pp).map(|(y, p)| (y - p).abs() / y.max(1.0)).collect();
    let medape = median(&mut ape) * 100.0;
    let ypmean = yp.iter().sum::<f64>() / n;
    let ss_res_p: f64 = yp.iter().zip(&pp).map(|(y, p)| (y - p).powi(2)).sum();
    let ss_tot_p: f64 = yp.iter().map(|y| (y - ypmean).powi(2)).sum::<f64>().max(1.0);
    let target_r2 = 1.0 - ss_res_p / ss_tot_p;
    Metrics { mae, medape, target_r2, log_r2 }
}

/// Predict each test row = train median log-target of its segment value (global
/// median fallback). This is the per-segment "floor" the probes must beat.
fn segment_median_pred(y: &[f64], seg: &[String], tr: &[usize], te_seg: &[usize]) -> Vec<f64> {
    let mut all: Vec<f64> = tr.iter().map(|&i| y[i]).collect();
    let global_med = median(&mut all);
    let mut by_value: HashMap<&str, Vec<f64>> = HashMap::new();
    for &i in tr {
        by_value.entry(seg[i].as_str()).or_default().push(y[i]);
    }
    let meds: HashMap<&str, f64> =
        by_value.into_iter().map(|(k, mut v)| (k, median(&mut v))).collect();
    te_seg.iter().map(|&i| *meds.get(seg[i].as_str()).unwrap_or(&global_med)).collect()
}

/// Gather a row-major `[rows.len(), cols.len()]` f64 sub-matrix from the dataset
/// feature block.
fn gather(features: &[f32], n_cols: usize, rows: &[usize], cols: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows.len() * cols.len());
    for &r in rows {
        let base = r * n_cols;
        for &c in cols {
            out.push(features[base + c] as f64);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// engine
// ---------------------------------------------------------------------------

/// A view (column subset) of the embedding block, with its global-linear probe
/// already fit (train on all rows) and evaluated on every test row.
struct ViewData {
    name: String,
    cols: Vec<usize>,
    dim: usize,
    /// global-linear log-space prediction for each test row (in `te` order).
    global_test_pred: Vec<f64>,
}

/// The categorical group to slice by: the caller's `--split-by` when non-empty,
/// else the domain's `[interp].segment_field` (falling back to the first
/// categorical field). Empty only if the domain declares no categorical field,
/// which the caller then reports as a "no one-hot columns" error.
fn resolve_split_by(requested: &str) -> String {
    if !requested.is_empty() {
        return requested.to_string();
    }
    lensing_core::domain::Domain::load_or_default(std::path::Path::new("."))
        .ok()
        .and_then(|d| d.interp_segment_field().map(|s| s.to_string()))
        .unwrap_or_default()
}

pub fn run(
    dataset_dir: &Path,
    output: &Path,
    want_mlp: bool,
    split_by: &str,
    emit: &dyn Fn(serde_json::Value),
) -> Result<()> {
    let resolved = resolve_split_by(split_by);
    let split_by = resolved.as_str();
    let ds = Dataset::load(dataset_dir).context("load dataset artifact")?;
    let n_cols = ds.manifest.n_cols;
    let features = &ds.features;
    let y: Vec<f64> = ds.target.iter().map(|&v| v as f64).collect();
    let tr: Vec<usize> = ds.train_idx.iter().map(|&i| i as usize).collect();
    let te: Vec<usize> = ds.test_idx.iter().map(|&i| i as usize).collect();

    // PCA column indices and the one-hot block for the chosen split group; also
    // collect the names of every one-hot group so the report can advertise the
    // other axes a caller could re-run on.
    let mut pca_idx: Vec<usize> = Vec::new();
    let mut split_cols: Vec<(String, usize)> = Vec::new();
    let mut available_splits: BTreeSet<String> = BTreeSet::new();
    for (i, c) in ds.manifest.columns.iter().enumerate() {
        match &c.kind {
            ColumnKind::Pca { .. } => pca_idx.push(i),
            ColumnKind::Onehot { group, value } => {
                available_splits.insert(group.clone());
                if group == split_by {
                    split_cols.push((value.clone(), i));
                }
            }
            _ => {}
        }
    }
    let available_splits: Vec<String> = available_splits.into_iter().collect();
    ensure!(!pca_idx.is_empty(), "dataset lacks PCA columns — wrong artifact?");
    ensure!(
        !split_cols.is_empty(),
        "dataset has no one-hot columns for split group {:?}; available: {}",
        split_by,
        if available_splits.is_empty() { "<none>".into() } else { available_splits.join(", ") }
    );

    // Decode the segment label per row by argmax over the split group's one-hot block.
    let seg: Vec<String> = (0..ds.manifest.n_rows)
        .map(|r| {
            let base = r * n_cols;
            let mut best = ("__none__", 0.0f32);
            for (name, ci) in &split_cols {
                let v = features[base + ci];
                if v > best.1 {
                    best = (name.as_str(), v);
                }
            }
            if best.1 < 0.5 { "__none__".to_string() } else { best.0.to_string() }
        })
        .collect();

    let mut value_counts: BTreeMap<String, usize> = BTreeMap::new();
    for s in &seg {
        *value_counts.entry(s.clone()).or_default() += 1;
    }

    let emb_dim = pca_idx.len();
    emit(json!({"event":"log","msg":format!(
        "embedding probe — {}: {} rows, emb(PCA) {}, split by '{}', {} train / {} test",
        ds.manifest.dataset_id, ds.manifest.n_rows, emb_dim, split_by, tr.len(), te.len())}));

    // Views: pca128 always; raw (all PCA columns) only when there is more to see.
    let mut views: Vec<(String, Vec<usize>)> = Vec::new();
    views.push(("pca128".to_string(), pca_idx[..emb_dim.min(128)].to_vec()));
    if emb_dim > 128 {
        views.push((format!("raw({emb_dim})"), pca_idx.clone()));
    }

    // y for train (all rows) — clamp anchor + global-linear target.
    let ytr_all: Vec<f64> = tr.iter().map(|&i| y[i]).collect();

    // Fit each view's global-linear probe ONCE (train all rows → predict all test).
    let mut view_data: Vec<ViewData> = Vec::with_capacity(views.len());
    for (name, cols) in &views {
        let dim = cols.len();
        emit(json!({"event":"log","msg":format!("[{name}] global-linear ridge (dim {dim})")}));
        let xtr = gather(features, n_cols, &tr, cols);
        let xte = gather(features, n_cols, &te, cols);
        let fit = lensing_interp::ridge(&xtr, &ytr_all, &xte, dim, 0.0);
        view_data.push(ViewData {
            name: name.clone(),
            cols: cols.clone(),
            dim,
            global_test_pred: fit.test_pred,
        });
    }

    // Segments: a pooled `all` reference, then every value of the split group with
    // ≥MIN_TEST test rows, most-populous first (so a 40-value group stays bounded).
    let mut te_counts: HashMap<&str, usize> = HashMap::new();
    for &i in &te {
        *te_counts.entry(seg[i].as_str()).or_default() += 1;
    }
    let mut values: Vec<String> = te_counts
        .iter()
        .filter(|(v, &n)| **v != "__none__" && n >= MIN_TEST)
        .map(|(v, _)| (*v).to_string())
        .collect();
    values.sort_by(|a, b| {
        te_counts[b.as_str()].cmp(&te_counts[a.as_str()]).then_with(|| a.cmp(b))
    });

    // (segment display name, Some(value) to match | None = pooled all-rows).
    let mut seg_specs: Vec<(String, Option<String>)> = vec![(POOLED.to_string(), None)];
    seg_specs.extend(values.iter().map(|v| (v.clone(), Some(v.clone()))));

    let mut segment_outs: Vec<SegmentOut> = Vec::new();
    for (seg_name, value) in &seg_specs {
        // test rows (and their positions in `te`) and train rows in this segment.
        let in_seg = |i: usize| value.as_ref().map_or(true, |v| &seg[i] == v);
        let te_seg_pos: Vec<usize> = (0..te.len()).filter(|&p| in_seg(te[p])).collect();
        let te_seg: Vec<usize> = te_seg_pos.iter().map(|&p| te[p]).collect();
        let tr_seg: Vec<usize> = tr.iter().copied().filter(|&i| in_seg(i)).collect();

        let mut view_outs: Vec<ViewOut> = Vec::new();
        for vd in &view_data {
            let mut vr =
                ViewResult { n_test: te_seg.len(), n_train_seg: tr_seg.len(), ..Default::default() };
            if te_seg.len() < MIN_TEST {
                vr.skip = Some(format!("only {} test rows", te_seg.len()));
                view_outs.push(ViewOut { view: vd.name.clone(), result: vr });
                continue;
            }
            let yte_seg: Vec<f64> = te_seg.iter().map(|&i| y[i]).collect();

            // per-segment-median floor
            let base = segment_median_pred(&y, &seg, &tr, &te_seg);
            vr.type_median = Some(metrics(&yte_seg, &base, &ytr_all));

            // global-linear: slice the precomputed all-test prediction
            let gp: Vec<f64> = te_seg_pos.iter().map(|&p| vd.global_test_pred[p]).collect();
            vr.global_linear = Some(metrics(&yte_seg, &gp, &ytr_all));

            // segment-only probes (purest "is in-segment signal present")
            if tr_seg.len() >= 40 {
                let ytr_seg: Vec<f64> = tr_seg.iter().map(|&i| y[i]).collect();
                let xtr = gather(features, n_cols, &tr_seg, &vd.cols);
                let xte = gather(features, n_cols, &te_seg, &vd.cols);
                let fit = lensing_interp::ridge(&xtr, &ytr_seg, &xte, vd.dim, 0.0);
                vr.seg_linear = Some(metrics(&yte_seg, &fit.test_pred, &ytr_seg));

                // nonlinear ceiling — only trustworthy with enough rows.
                if want_mlp && tr_seg.len() >= 800 {
                    emit(json!({"event":"log","msg":format!(
                        "[{}] {} seg-mlp ({} train rows)", vd.name, seg_name, tr_seg.len())}));
                    let pred = nn_probe::mlp_regress(&xtr, &ytr_seg, &xte, vd.dim, 42);
                    vr.seg_mlp = Some(metrics(&yte_seg, &pred, &ytr_seg));
                }
            }
            view_outs.push(ViewOut { view: vd.name.clone(), result: vr });
        }

        segment_outs.push(SegmentOut {
            name: seg_name.clone(),
            views: view_outs,
        });
    }

    // The hardest tail: the non-pooled segment with the worst per-segment-median
    // floor that actually got a probe fit. Drives the verdict + decode AUC.
    let hardest = hardest_segment(&segment_outs);

    // Probe B: segment-class decode (raw view if available else pca128), with the
    // hardest segment's value as the one-vs-rest AUC target.
    let decode_cols = view_data.last().map(|v| v.cols.clone()).unwrap_or_default();
    let segment_decode = if want_mlp {
        emit(json!({"event":"log","msg":format!("segment-class decode ('{}')", split_by)}));
        Some(segment_decode(
            features,
            n_cols,
            &decode_cols,
            &seg,
            &tr,
            &te,
            hardest.map(|s| s.name.as_str()),
        ))
    } else {
        None
    };

    // Verdict + raw-vs-pca128 gap on the hardest segment.
    let (verdict, raw_vs_pca128_gap) = verdict(emb_dim, hardest);

    let report = Report {
        tool: "embedding-probe",
        dataset: ds.manifest.dataset_id.clone(),
        split_by: split_by.to_string(),
        available_splits,
        emb_dim,
        n_rows: ds.manifest.n_rows,
        n_train: tr.len(),
        n_test: te.len(),
        views: views.iter().map(|(n, _)| n.clone()).collect(),
        value_counts,
        segments: segment_outs,
        segment_decode,
        verdict: verdict.clone(),
        raw_vs_pca128_gap,
    };

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(output, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write {}", output.display()))?;
    emit(json!({"event":"log","msg":format!("verdict: {verdict}")}));
    emit(json!({"event":"log","msg":format!("wrote {}", output.display())}));
    emit(json!({"event":"done"}));
    Ok(())
}

/// The hardest non-pooled segment: the one with the worst per-segment-median floor
/// (highest medAPE) that has a fitted segment-linear probe to interpret. The raw
/// view (`views.last()`) is used as the reference read.
fn hardest_segment(segments: &[SegmentOut]) -> Option<&SegmentOut> {
    segments
        .iter()
        .filter(|s| s.name != POOLED)
        .filter_map(|s| {
            let r = &s.views.last()?.result;
            r.seg_linear.as_ref()?;
            Some((s, r.type_median.as_ref()?.medape))
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(s, _)| s)
}

/// Apply the decision rule to the hardest segment and compute the raw-vs-pca128
/// seg-linear MAE gap. Returns (verdict string, gap or None).
fn verdict(emb_dim: usize, hardest: Option<&SegmentOut>) -> (String, Option<f64>) {
    let hard = match hardest {
        Some(s) => s,
        None => return ("INCONCLUSIVE — no segment had enough rows to fit a probe.".into(), None),
    };
    let key = if emb_dim > 128 { format!("raw({emb_dim})") } else { "pca128".to_string() };
    let v = match hard.views.iter().find(|v| v.view == key) {
        Some(v) => &v.result,
        None => return ("INCONCLUSIVE — no probe view for the hardest segment.".into(), None),
    };
    let (base, seg_lin) = match (&v.type_median, &v.seg_linear) {
        (Some(b), Some(s)) => (b.mae, s),
        _ => return (format!("INCONCLUSIVE — too few '{}' rows to fit a segment probe.", hard.name), None),
    };
    // Use only well-behaved probes (positive log-R²) so an overfit MLP can't
    // drive the verdict; the linear probe is the anchor.
    let mut best_raw = seg_lin.mae;
    let mut best_logr2 = seg_lin.log_r2;
    if let Some(m) = &v.seg_mlp {
        if m.log_r2 > 0.0 {
            best_raw = best_raw.min(m.mae);
            best_logr2 = best_logr2.max(m.log_r2);
        }
    }
    let lift = (base - best_raw) / base;
    let verdict = if lift >= LIFT_PRESENT && best_logr2 >= LOGR2_PRESENT {
        format!(
            "PRESENT (probe cuts '{}' MAE {:.1}% vs segment-median, logR² {:.2}) → signal is in the \
             embeddings but UNUSED. See raw-vs-pca128 below for whether PCA-128 is the bottleneck.",
            hard.name, lift * 100.0, best_logr2
        )
    } else {
        format!(
            "ABSENT (hardest segment '{}': probe lift {:.1}% < {:.0}% and/or logR² {:.2} < {:.2}) → \
             information ceiling looks REAL; redirect to corpus-level fixes (scrape more fields).",
            hard.name, lift * 100.0, LIFT_PRESENT * 100.0, best_logr2, LOGR2_PRESENT
        )
    };

    // raw-vs-pca128 gap on the hardest segment's seg-linear MAE.
    let gap = if emb_dim > 128 {
        let raw_mae = seg_lin.mae;
        hard.views
            .iter()
            .find(|v| v.view == "pca128")
            .and_then(|v| v.result.seg_linear.as_ref())
            .map(|m128| (m128.mae - raw_mae) / m128.mae)
    } else {
        None
    };
    let verdict = match gap {
        Some(g) if g >= GAP_DISCARDED => format!(
            "{verdict} raw BEATS pca128 by {:.1}% MAE → PCA-128 is discarding signal (cheaper fix: more dims).",
            g * 100.0
        ),
        Some(g) => format!(
            "{verdict} raw ≈ pca128 ({:+.1}% MAE) → if present, the signal is already in PCA-128 and unused by the global fit.",
            g * 100.0
        ),
        None => verdict,
    };
    (verdict, gap)
}

/// Probe B — multinomial segment-class decode from the embeddings: accuracy /
/// macro-F1 / hardest-vs-rest AUC vs the majority baseline.
fn segment_decode(
    features: &[f32],
    n_cols: usize,
    cols: &[usize],
    seg: &[String],
    tr: &[usize],
    te: &[usize],
    hardest: Option<&str>,
) -> SegmentDecode {
    // Drop the __none__ bucket from train/test.
    let tr2: Vec<usize> = tr.iter().copied().filter(|&i| seg[i] != "__none__").collect();
    let te2: Vec<usize> = te.iter().copied().filter(|&i| seg[i] != "__none__").collect();

    // Stable class index map from the train labels.
    let mut classes: Vec<String> = tr2.iter().map(|&i| seg[i].clone()).collect();
    classes.sort();
    classes.dedup();
    let cls_idx: HashMap<&str, usize> = classes.iter().enumerate().map(|(j, c)| (c.as_str(), j)).collect();

    let xtr = gather(features, n_cols, &tr2, cols);
    let xte = gather(features, n_cols, &te2, cols);
    let ytr: Vec<usize> = tr2.iter().map(|&i| cls_idx[seg[i].as_str()]).collect();

    let out = nn_probe::logistic_decode(&xtr, &ytr, &xte, cols.len(), classes.len(), 42);

    // Accuracy + macro-F1 against the true test labels.
    let yte_true: Vec<usize> = te2.iter().map(|&i| *cls_idx.get(seg[i].as_str()).unwrap_or(&usize::MAX)).collect();
    let n = yte_true.len().max(1);
    let acc = yte_true.iter().zip(&out.pred).filter(|(t, p)| **t == **p).count() as f64 / n as f64;

    let k = classes.len();
    let mut tp = vec![0usize; k];
    let mut fp = vec![0usize; k];
    let mut fnn = vec![0usize; k];
    for (&t, &p) in yte_true.iter().zip(&out.pred) {
        if t == usize::MAX {
            continue;
        }
        if t == p {
            tp[t] += 1;
        } else {
            fp[p] += 1;
            fnn[t] += 1;
        }
    }
    let mut f1_sum = 0.0;
    for c in 0..k {
        let prec = tp[c] as f64 / (tp[c] + fp[c]).max(1) as f64;
        let rec = tp[c] as f64 / (tp[c] + fnn[c]).max(1) as f64;
        f1_sum += if prec + rec > 0.0 { 2.0 * prec * rec / (prec + rec) } else { 0.0 };
    }
    let macro_f1 = f1_sum / k as f64;

    // Majority baseline.
    let mut counts = vec![0usize; k];
    for &c in &ytr {
        counts[c] += 1;
    }
    let maj = counts.iter().enumerate().max_by_key(|(_, &c)| c).map(|(j, _)| j).unwrap_or(0);
    let majority_acc = yte_true.iter().filter(|&&t| t == maj).count() as f64 / n as f64;

    // hardest-vs-rest AUC from the hardest segment's softmax column.
    let mut hardest_vs_rest_auc = None;
    let mut hardest_value = None;
    if let Some((v, ci)) = hardest.and_then(|v| cls_idx.get(v).map(|&ci| (v, ci))) {
        let scores: Vec<f64> = (0..te2.len()).map(|r| out.proba[r * k + ci]).collect();
        let labels: Vec<bool> = yte_true.iter().map(|&t| t == ci).collect();
        if let Some(a) = auc(&scores, &labels) {
            hardest_vs_rest_auc = Some(a);
            hardest_value = Some(v.to_string());
        }
    }

    SegmentDecode {
        accuracy: acc,
        macro_f1,
        majority_acc,
        hardest_vs_rest_auc,
        hardest_value,
        n_test: te2.len(),
    }
}

/// Rank-based ROC AUC (Mann–Whitney). None if a class is empty.
fn auc(scores: &[f64], labels: &[bool]) -> Option<f64> {
    let n_pos = labels.iter().filter(|&&l| l).count();
    let n_neg = labels.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return None;
    }
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());
    // average ranks (1-based), handling ties
    let mut ranks = vec![0.0f64; scores.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && scores[idx[j]] == scores[idx[i]] {
            j += 1;
        }
        let avg = ((i + 1 + j) as f64) / 2.0; // mean of ranks i+1..=j
        for &k in &idx[i..j] {
            ranks[k] = avg;
        }
        i = j;
    }
    let sum_pos: f64 = (0..labels.len()).filter(|&r| labels[r]).map(|r| ranks[r]).sum();
    Some((sum_pos - n_pos as f64 * (n_pos as f64 + 1.0) / 2.0) / (n_pos as f64 * n_neg as f64))
}
