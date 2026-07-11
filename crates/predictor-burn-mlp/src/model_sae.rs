//! Per-model sparse autoencoder — dictionary learning on a *trained MLP's* own
//! hidden activations, the nonlinear sibling of the layer probe (`probe.rs`).
//!
//! Where `lensing-sae analyze` runs the SAE on a DATASET's (frozen, shared) embedding
//! block, this runs the SAME engine (`lensing_sae::sae`) on the activations a promoted
//! checkpoint produces at each hidden layer — so the atoms describe what THAT
//! model has carved into its representation, not what the embedding affords.
//!
//! Because a hidden basis rotates with the training seed, the result is tied to
//! one checkpoint: atom *identities* are per-model, not per-architecture. What it
//! buys — the reason to run it over the plain layer probe — is four reads the
//! linear probe can't give:
//!   1. capacity: how much of each layer's width the model actually uses (dead /
//!      rare atoms) — an over- or under-provisioning signal for the layer size.
//!   2. segment representation: for EVERY one-hot group value, whether the model
//!      built dedicated atoms for that slice or smeared it across shared ones.
//!   3. concept-vs-decodability depth: the layer where the target becomes linearly
//!      decodable (from the probe) vs. the layer where interpretable concepts
//!      first form (from the SAE) — where the depth is doing representational work
//!      vs. generic transformation.
//!   4. dropped signal: target-relevant concepts present in the embedding
//!      (a prior dataset-SAE run) that no model atom tracks — signal the trained
//!      model is leaving on the table. Optional (needs `--dataset-sae`).
//!
//! Applies to the MLP family only; tree ensembles have no activation space.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};

use lensing_core::{ColumnKind, Dataset};
use lensing_interp::{fit_stages, gather_targets, reference_stage, Stage, StageKind};
use lensing_sae::sae;

use crate::model::{self, Activation};
use crate::scaler::Scaler;

/// Everything the `model-sae` subcommand takes. Grouped into a struct so the
/// `main.rs` match arm stays readable.
pub struct Args {
    pub model_dir: PathBuf,
    pub dataset_dir: PathBuf,
    pub output: PathBuf,
    /// 1-based hidden-layer indices to analyze; empty ⇒ every hidden layer.
    pub layers: Vec<usize>,
    /// SAE atom count; 0 ⇒ 2× the layer width (an overcomplete dictionary).
    pub n_atoms: usize,
    pub l1: f64,
    pub epochs: usize,
    pub lr: f64,
    pub batch: usize,
    pub seed: u64,
    /// Ridge penalty for the depth/raw-vs-code probes; 0 ⇒ internal CV.
    pub lambda: f64,
    /// Skip a one-hot group whose slice has fewer than this many rows.
    pub min_segment: usize,
    /// A prior `lensing-sae analyze --emit-codes` result JSON, to diff for dropped
    /// signal (its target atoms must carry `code` vectors).
    pub dataset_sae: Option<PathBuf>,
    /// Label the surfaced atoms with an LLM (auto-interp, Bills et al. 2023).
    /// No-op unless OPENAI_API_KEY is set.
    pub label_atoms: bool,
    /// Cache dir for trained SAEs (content-addressed by model+layer+config).
    pub cache_dir: Option<PathBuf>,
}

/// A concept is "interpretable" if it correlates with (log) target at least this
/// much and isn't vanishingly rare — the bar for counting it toward depth.
const CONCEPT_CORR: f64 = 0.10;
const MIN_FREQ: f64 = 0.005;
/// A segment is "represented" if some atom's mean activation separates it from
/// the rest by at least this many (pooled) SDs.
const SEG_SEP: f64 = 0.5;

/// Minimal view of a model dir's `hyperparams.json` — enough to rebuild and run
/// the architecture (mirrors `probe.rs`).
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

/// Bump when the recipe changes so stale SAE caches are not reused.
const ENGINE_VERSION: u32 = 1;

pub fn run(args: Args, emit: &dyn Fn(Value)) -> Result<()> {
    let Args {
        model_dir,
        dataset_dir,
        output,
        layers,
        n_atoms,
        l1,
        epochs,
        lr,
        batch,
        seed,
        lambda,
        min_segment,
        dataset_sae,
        label_atoms,
        cache_dir,
    } = args;

    let hp: HpSnapshot = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("hyperparams.json"))
            .context("read hyperparams.json from model dir")?,
    )
    .context("parse hyperparams.json")?;
    let scaler = Scaler::load(&model_dir.join("scaler.json"))
        .context("load scaler.json from model dir")?;
    let n_cols = scaler.mean.len();

    let ds = Dataset::load(&dataset_dir)?;
    ensure!(
        ds.manifest.n_cols == n_cols,
        "model was fit on {n_cols} columns, dataset has {}",
        ds.manifest.n_cols
    );
    let n_rows = ds.manifest.n_rows;
    let bs = hp.batch_size.max(1);
    let model_id = model_dir.file_name().and_then(|s| s.to_str()).unwrap_or("model").to_string();
    emit(json!({"event":"log","msg":format!(
        "model-sae: model {model_id} {:?} ({} activation), dataset {} ({} train / {} test)",
        hp.hidden, hp.activation.name(), ds.manifest.dataset_id, ds.train_idx.len(),
        ds.test_idx.len())}));

    let mlp = model::load(&model_dir.join("model"), n_cols, &hp.hidden, hp.dropout, hp.activation)?;
    let xs = scaler.transform(&ds.features);
    let captured = model::capture_stage_activations(&mlp, &xs, n_cols, bs);

    // ---- Depth axis: linear decodability at each stage + the model's own head.
    // The rising curve (input → hidden_1 → … → output) is the layer probe; the
    // SAE below asks *where in that curve interpretable concepts appear*.
    let transform = ds.manifest.target.transform;
    let ytr = gather_targets(&ds.target, &ds.train_idx);
    let yte = gather_targets(&ds.target, &ds.test_idx);
    let clamp_hi = ytr.iter().cloned().fold(f64::MIN, f64::max) + std::f64::consts::LN_2;
    let net_stages: Vec<Stage> = captured
        .iter()
        .enumerate()
        .map(|(k, la)| Stage {
            name: la.name.clone(),
            kind: if k == 0 { StageKind::Input } else { StageKind::Hidden },
            dim: la.dim,
            values: la.values.clone(),
        })
        .collect();
    let mut depth = fit_stages(
        &net_stages, &ds.train_idx, &ds.test_idx, &ytr, &yte, transform, clamp_hi, lambda, emit,
    );
    let model_pred: Vec<f64> =
        model::predict_loaded(&mlp, &xs, n_cols, bs).into_iter().map(|v| v as f64).collect();
    depth.push(reference_stage(
        depth.len(), "output (model)", &model_pred, &ds.train_idx, &ds.test_idx, &ytr, &yte,
        transform, clamp_hi,
    ));

    // ---- Segment masks: one per one-hot column value, in manifest order.
    let segments: Vec<(String, usize)> = ds
        .manifest
        .columns
        .iter()
        .enumerate()
        .filter_map(|(ci, c)| match &c.kind {
            ColumnKind::Onehot { group, value } => Some((format!("{group}={value}"), ci)),
            _ => None,
        })
        .collect();

    // ---- Per hidden layer: train the SAE on its activations and analyze.
    // captured[0] is the input; hidden layers are captured[1..], 1-based.
    let n_hidden = captured.len().saturating_sub(1);
    let want: BTreeSet<usize> = if layers.is_empty() {
        (1..=n_hidden).collect()
    } else {
        layers.into_iter().filter(|&k| k >= 1 && k <= n_hidden).collect()
    };
    ensure!(!want.is_empty(), "no valid hidden layers to analyze (model has {n_hidden})");

    // Atom auto-interp labeling (Bills et al. 2023) is a deferred upstream
    // follow-up; atoms are surfaced unlabeled here. `label_atoms` is accepted for
    // API stability so the request/CLI surface is stable when it lands.
    if label_atoms {
        emit(json!({"event":"log","msg":
            "atom auto-labeling is not available in this build; atoms returned unlabeled"}));
    }

    // Embedding diff: parse the dataset SAE's target atoms once (standardized).
    // The costly step is training the per-layer SAEs; the diff itself is a cheap
    // correlation, so it's computed for EVERY analyzed layer — the layer to read
    // can then be chosen in the UI without retraining anything.
    let (ds_atoms, ds_missing_codes) = match &dataset_sae {
        Some(p) => load_ds_atoms(p, n_rows)?,
        None => (Vec::new(), 0),
    };
    if dataset_sae.is_some() {
        emit(json!({"event":"log","msg":format!(
            "embedding diff: {} target-relevant atoms loaded ({} lacked codes)",
            ds_atoms.len(), ds_missing_codes)}));
    }

    let mut layer_reports: Vec<Value> = Vec::new();

    for k in &want {
        let la = &captured[*k];
        let d = la.dim;
        let m = if n_atoms == 0 { 2 * d } else { n_atoms };

        // Standardize this layer's activations on TRAIN stats, then train/encode.
        let (mean, std) = col_stats(&la.values, &ds.train_idx, d);
        let mut act = la.values.clone();
        for r in 0..n_rows {
            for j in 0..d {
                act[r * d + j] = ((act[r * d + j] as f64 - mean[j]) / std[j]) as f32;
            }
        }
        let train_x = gather(&act, &ds.train_idx, d);

        let cfg = sae::SaeConfig { n_atoms: m, l1, epochs, lr, batch_size: batch, seed };
        let cache = cache_dir.as_ref().map(|c| {
            c.join(cache_key(&model_id, &ds.manifest.dataset_id, *k, d, m, l1, epochs, lr, seed))
        });
        let hit = cache.as_ref().map(|c| c.join("model.mpk").exists()).unwrap_or(false);
        let sae_model = if hit {
            emit(json!({"event":"log","msg":format!("layer {k}: loaded cached SAE")}));
            sae::load(cache.as_ref().unwrap(), d, m)?
        } else {
            emit(json!({"event":"log","msg":format!(
                "layer {k}: {d}-d activations → {m} atoms, l1 {l1}, {epochs} epochs")}));
            let (sm, _loss) = sae::train(&train_x, d, &cfg, emit);
            if let Some(c) = &cache {
                let _ = sae::save(&sm, c);
            }
            sm
        };
        let out = sae::encode_all(&sae_model, &act, d, batch);

        // (1) Capacity read: how much of the layer's width the model uses.
        let (freq, _amean, sd, corr) = atom_global(&out.code_all, m, &ds.target, n_rows);
        let active = freq.iter().filter(|&&f| f > 0.0).count();
        let rare = freq.iter().filter(|&&f| f > 0.0 && f < MIN_FREQ).count();
        let capacity = json!({
            "n_atoms": m,
            "active_atoms": active,
            "dead_atoms": m - active,
            "rare_atoms": rare,
            "utilization": active as f64 / m as f64,
            "l0_mean": out.l0_mean,
            "l0_frac": out.l0_mean / m as f64,
            "var_explained": out.var_explained,
        });

        // Atom → target-correlation ranking (interpretable concepts).
        let mut by_target: Vec<usize> = (0..m).filter(|&i| freq[i] >= MIN_FREQ).collect();
        by_target.sort_by(|&a, &b| corr[b].abs().partial_cmp(&corr[a].abs()).unwrap());
        let n_concepts = by_target.iter().filter(|&&i| corr[i].abs() >= CONCEPT_CORR).count();
        let shown: Vec<usize> = by_target.iter().take(12).copied().collect();
        // Atoms are surfaced unlabeled (auto-interp labeling is a deferred
        // upstream follow-up); the `label` field stays null.
        let labels: HashMap<usize, String> = HashMap::new();
        let atoms_by_target: Vec<Value> = shown
            .iter()
            .map(|&i| json!({
                "atom": i, "target_corr": corr[i], "freq": freq[i],
                "label": labels.get(&i),
                "top_rows": top_k_rows(&out.code_all, m, i, 12),
            }))
            .collect();

        // Does sparsifying the layer decode the target BETTER than the raw layer?
        let probe_stages = vec![
            Stage { name: format!("hidden_{k}:raw"), kind: StageKind::Input, dim: d, values: act },
            Stage {
                name: format!("hidden_{k}:sae"),
                kind: StageKind::Hidden,
                dim: m,
                values: out.code_all.clone(),
            },
        ];
        let probe = fit_stages(
            &probe_stages, &ds.train_idx, &ds.test_idx, &ytr, &yte, transform, clamp_hi, lambda,
            emit,
        );

        // (2) Segment representation, generalized over every one-hot value.
        let seg_reports: Vec<Value> = segments
            .iter()
            .filter_map(|(label, ci)| {
                let mask: Vec<bool> =
                    (0..n_rows).map(|r| ds.features[r * ds.manifest.n_cols + ci] > 0.5).collect();
                let nseg = mask.iter().filter(|&&b| b).count();
                if nseg < min_segment || nseg == n_rows {
                    return None;
                }
                let sep = segment_sep(&out.code_all, m, &mask, &sd, n_rows);
                let mut order: Vec<usize> = (0..m).filter(|&i| freq[i] >= MIN_FREQ).collect();
                order.sort_by(|&a, &b| sep[b].abs().partial_cmp(&sep[a].abs()).unwrap());
                let best = order.first().copied();
                let top: Vec<Value> = order
                    .iter()
                    .take(3)
                    .map(|&i| json!({"atom": i, "separation": sep[i], "freq": freq[i]}))
                    .collect();
                Some(json!({
                    "segment": label,
                    "n_rows": nseg,
                    "represented": best.map(|i| sep[i].abs() >= SEG_SEP).unwrap_or(false),
                    "top_atoms": top,
                }))
            })
            .collect();

        // (4) Dropped signal FOR THIS LAYER: target-relevant embedding concepts
        // whose best correlation with any of this layer's atoms is below the bar
        // (cheap given the codes are already in hand).
        let dropped_vs_embedding = if ds_atoms.is_empty() {
            Value::Null
        } else {
            dropped_for_layer(&ds_atoms, &out.code_all, m, n_rows)
        };

        layer_reports.push(json!({
            "layer": k,
            "dim": d,
            "capacity": capacity,
            "n_interpretable_concepts": n_concepts,
            "probe": probe,
            "atoms_by_target_corr": atoms_by_target,
            "segments": seg_reports,
            "dropped_vs_embedding": dropped_vs_embedding,
        }));
    }

    // (3) Concept-vs-decodability depth: join the linear target R² at each hidden
    // stage with the count of interpretable SAE concepts formed there.
    let depth_summary: Vec<Value> = layer_reports
        .iter()
        .map(|lr_| {
            let k = lr_["layer"].as_u64().unwrap_or(0) as usize;
            // net_stages[k] is hidden layer k (stage 0 is the input).
            let lin = depth.get(k);
            json!({
                "layer": k,
                "linear_r2_target": lin.map(|s| s.test_r2_target),
                "linear_r2_log": lin.map(|s| s.test_r2_log),
                "n_interpretable_concepts": lr_["n_interpretable_concepts"],
            })
        })
        .collect();

    if dataset_sae.is_none() {
        emit(json!({"event":"log","msg":
            "no --dataset-sae given; skipping dropped-signal diff"}));
    }

    let result = json!({
        "tool": "model-sae",
        "method": "sparse autoencoder (dictionary learning) on a trained MLP's hidden activations",
        "engine_version": ENGINE_VERSION,
        "model_dir": model_id,
        "dataset_id": ds.manifest.dataset_id,
        "n_train": ds.train_idx.len(),
        "n_test": ds.test_idx.len(),
        "hidden": hp.hidden,
        "activation": hp.activation.name(),
        "config": {"n_atoms": n_atoms, "l1": l1, "epochs": epochs, "lr": lr, "seed": seed},
        "depth_linear_probe": depth,
        "layers": layer_reports,
        "concept_vs_decodability": depth_summary,
        // Per-layer dropped-signal lives on each layer (`dropped_vs_embedding`);
        // this records the diff parameters + whether it ran.
        "embedding_diff": if dataset_sae.is_some() {
            json!({
                "ran": true,
                "match_corr_threshold": DROPPED_MATCH_CORR,
                "n_target_atoms": ds_atoms.len(),
                "n_missing_codes": ds_missing_codes,
            })
        } else {
            json!({"ran": false})
        },
    });

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&output, serde_json::to_vec_pretty(&result)?)
        .with_context(|| format!("write {}", output.display()))?;
    emit(json!({"event":"log","msg":format!("wrote {}", output.display())}));
    emit(json!({"event":"done"}));
    Ok(())
}

/// Per-atom frequency, mean, SD, and correlation with the (log) target, over all
/// rows. `code` is row-major `[n_rows, m]`.
fn atom_global(code: &[f32], m: usize, y: &[f32], n_rows: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let nf = n_rows as f64;
    let ybar = y.iter().map(|&v| v as f64).sum::<f64>() / nf;
    let ysd = (y.iter().map(|&v| (v as f64 - ybar).powi(2)).sum::<f64>() / nf).sqrt().max(1e-9);
    let (mut cnt, mut s, mut ss, mut sy) =
        (vec![0f64; m], vec![0f64; m], vec![0f64; m], vec![0f64; m]);
    for r in 0..n_rows {
        let yr = y[r] as f64;
        let base = r * m;
        for i in 0..m {
            let v = code[base + i] as f64;
            if v > 1e-6 {
                cnt[i] += 1.0;
            }
            s[i] += v;
            ss[i] += v * v;
            sy[i] += v * yr;
        }
    }
    let (mut freq, mut mean, mut sd, mut corr) =
        (vec![0f64; m], vec![0f64; m], vec![0f64; m], vec![0f64; m]);
    for i in 0..m {
        mean[i] = s[i] / nf;
        sd[i] = (ss[i] / nf - mean[i] * mean[i]).max(0.0).sqrt();
        let cov = sy[i] / nf - mean[i] * ybar;
        corr[i] = if sd[i] > 1e-9 { cov / (sd[i] * ysd) } else { 0.0 };
        freq[i] = cnt[i] / nf;
    }
    (freq, mean, sd, corr)
}

/// Per-atom segment-vs-rest mean-activation separation, in units of the atom's
/// pooled SD (`sd`, from [`atom_global`]).
fn segment_sep(code: &[f32], m: usize, mask: &[bool], sd: &[f64], n_rows: usize) -> Vec<f64> {
    let (mut sseg, mut srest) = (vec![0f64; m], vec![0f64; m]);
    let (mut nseg, mut nrest) = (0f64, 0f64);
    for r in 0..n_rows {
        let base = r * m;
        if mask[r] {
            nseg += 1.0;
            for i in 0..m {
                sseg[i] += code[base + i] as f64;
            }
        } else {
            nrest += 1.0;
            for i in 0..m {
                srest[i] += code[base + i] as f64;
            }
        }
    }
    (0..m)
        .map(|i| {
            if nseg > 0.0 && nrest > 0.0 && sd[i] > 1e-9 {
                (sseg[i] / nseg - srest[i] / nrest) / sd[i]
            } else {
                0.0
            }
        })
        .collect()
}

/// A target-relevant embedding concept from a dataset SAE, with its per-row
/// activation vector pre-standardized (ready to correlate against any layer).
struct DsAtom {
    atom: Value,
    target_corr: f64,
    label: Option<Value>,
    code_std: Vec<f64>,
}

/// A concept whose best correlation with any model atom is below this is
/// "dropped" — the embedding carries a target-relevant direction the model at
/// that layer never tracks.
const DROPPED_MATCH_CORR: f64 = 0.30;

/// Parse a dataset SAE (`lensing-sae analyze --emit-codes`) ONCE into the
/// target-relevant atoms, each with its standardized code. Returns the atoms and
/// how many were skipped for lacking a usable `code` vector.
///
/// Row alignment: both SAEs encode the SAME dataset in the SAME row order, so
/// index `r` is the same item in the dataset code and every model layer code.
fn load_ds_atoms(path: &Path, n_rows: usize) -> Result<(Vec<DsAtom>, usize)> {
    let ds_sae: Value = serde_json::from_str(
        &std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?,
    )?;
    let atoms = ds_sae["atoms_by_target_corr"].as_array().cloned().unwrap_or_default();
    let mut out = Vec::new();
    let mut missing = 0usize;
    for a in &atoms {
        let corr = a["target_corr"].as_f64().unwrap_or(0.0);
        if corr.abs() < 0.15 {
            continue;
        }
        let code: Vec<f32> = match a["code"].as_array() {
            Some(v) if v.len() == n_rows => {
                v.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect()
            }
            _ => {
                missing += 1;
                continue;
            }
        };
        out.push(DsAtom {
            atom: a["atom"].clone(),
            target_corr: corr,
            label: a.get("label").cloned(),
            code_std: standardize_col(&code, 1, 0, n_rows),
        });
    }
    Ok((out, missing))
}

/// The dropped-signal diff for ONE model layer: for each embedding concept, the
/// best absolute Pearson correlation with any of the layer's atoms. Cheap —
/// standardizes the layer's atoms once, then a dot product per (concept, atom).
fn dropped_for_layer(ds_atoms: &[DsAtom], model_code: &[f32], model_m: usize, n_rows: usize) -> Value {
    let model_cols: Vec<Vec<f64>> =
        (0..model_m).map(|j| standardize_col(model_code, model_m, j, n_rows)).collect();
    let mut dropped: Vec<Value> = Vec::new();
    for a in ds_atoms {
        let best = model_cols
            .iter()
            .map(|mc| pearson_pre(&a.code_std, mc, n_rows).abs())
            .fold(0.0f64, f64::max);
        if best < DROPPED_MATCH_CORR {
            dropped.push(json!({
                "dataset_atom": a.atom,
                "target_corr": a.target_corr,
                "label": a.label,
                "best_match_corr": best,
            }));
        }
    }
    json!({
        "n_checked": ds_atoms.len(),
        "n_dropped": dropped.len(),
        "dropped": dropped,
    })
}

/// Standardize column `j` of a row-major `[n_rows, stride]` block to zero mean /
/// unit variance (returns zeros if the column is constant).
fn standardize_col(x: &[f32], stride: usize, j: usize, n_rows: usize) -> Vec<f64> {
    let mut v: Vec<f64> = (0..n_rows).map(|r| x[r * stride + j] as f64).collect();
    let mean = v.iter().sum::<f64>() / n_rows.max(1) as f64;
    let sd = (v.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / n_rows.max(1) as f64).sqrt();
    if sd < 1e-9 {
        return vec![0.0; n_rows];
    }
    for e in &mut v {
        *e = (*e - mean) / sd;
    }
    v
}

/// Pearson correlation of two already-standardized vectors: just their mean
/// product.
fn pearson_pre(a: &[f64], b: &[f64], n_rows: usize) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() / n_rows.max(1) as f64
}

/// The rows that most strongly activate `atom`, highest first (active only).
fn top_k_rows(code: &[f32], m: usize, atom: usize, k: usize) -> Vec<usize> {
    let n = code.len() / m;
    let mut v: Vec<(f32, usize)> =
        (0..n).map(|r| (code[r * m + atom], r)).filter(|(x, _)| *x > 1e-6).collect();
    v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    v.into_iter().take(k).map(|(_, r)| r).collect()
}

/// Column mean/std over the train split (for standardizing an activation block).
fn col_stats(x: &[f32], train_idx: &[u32], d: usize) -> (Vec<f64>, Vec<f64>) {
    let n = train_idx.len().max(1) as f64;
    let mut mean = vec![0f64; d];
    for &r in train_idx {
        let base = r as usize * d;
        for k in 0..d {
            mean[k] += x[base + k] as f64;
        }
    }
    for mk in mean.iter_mut() {
        *mk /= n;
    }
    let mut var = vec![0f64; d];
    for &r in train_idx {
        let base = r as usize * d;
        for k in 0..d {
            let e = x[base + k] as f64 - mean[k];
            var[k] += e * e;
        }
    }
    let std = var.iter().map(|v| (v / n).sqrt().max(1e-8)).collect();
    (mean, std)
}

fn gather(x: &[f32], idx: &[u32], d: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(idx.len() * d);
    for &r in idx {
        out.extend_from_slice(&x[r as usize * d..(r as usize + 1) * d]);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn cache_key(
    model_id: &str,
    dataset_id: &str,
    layer: usize,
    d: usize,
    m: usize,
    l1: f64,
    epochs: usize,
    lr: f64,
    seed: u64,
) -> String {
    let f = |x: f64| format!("{x}").replace('.', "p").replace('-', "m");
    format!(
        "{model_id}__{dataset_id}__L{layer}_d{d}_a{m}_l1{}_e{epochs}_lr{}_s{seed}_v{ENGINE_VERSION}",
        f(l1),
        f(lr)
    )
}
