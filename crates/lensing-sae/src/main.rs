//! `lensing-sae analyze` — P2 of the interpretability program.
//!
//! Trains a sparse autoencoder on a dataset's embedding (PCA) block, then asks
//! two questions with the burn-free `lensing-interp` ridge engine:
//!   1. does the SAE code decode target better than the raw embedding block —
//!      overall, and on the hard segment (the P1 hot spot)?
//!   2. are the learned atoms interpretable — do any track target or separate the
//!      segment from the rest?
//! Output is a single JSON result; lensing-server's /api/interp/sae runs it as a job.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;

use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand};
use serde_json::{json, Value};

use lensing_core::domain::Domain;
use lensing_core::{ColumnKind, Dataset};
use lensing_interp::{fit_stages, gather_targets, Stage, StageKind};
use lensing_sae::{autointerp, llm, sae};

#[derive(Parser)]
#[command(name = "lensing-sae")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train an SAE on a dataset's embeddings and write the analysis JSON.
    Analyze {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        output: PathBuf,
        /// Cap the embedding dims fed to the SAE (the signal P2 targets lives in
        /// the full tail, so the default uses all PCA columns up to this).
        #[arg(long, default_value_t = 1536)]
        max_dims: usize,
        /// Atom count; 0 ⇒ 2× the input dim (an overcomplete dictionary).
        #[arg(long, default_value_t = 0)]
        n_atoms: usize,
        #[arg(long, default_value_t = 0.005)]
        l1: f64,
        #[arg(long, default_value_t = 50)]
        epochs: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f64,
        #[arg(long, default_value_t = 256)]
        batch: usize,
        #[arg(long, default_value_t = 1337)]
        seed: u64,
        /// Categorical value to slice for the segment probe; empty ⇒ pooled only.
        #[arg(long, default_value = "")]
        segment: String,
        /// Label the top atoms with an LLM (auto-interp, Bills et al. 2023).
        /// No-op unless OPENAI_API_KEY is set.
        #[arg(long)]
        label_atoms: bool,
        /// Emit each surfaced target atom's full per-row activation vector
        /// (`code`) into the result — lets a per-model SAE correlate its own
        /// atoms against these embedding concepts (the dropped-signal diff).
        /// Off by default (the vectors are large and the UI doesn't need them).
        #[arg(long)]
        emit_codes: bool,
        /// Cache dir for trained SAEs (content-addressed by dataset + config).
        /// A hit skips training; omit to always train fresh.
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
}

/// Bump when the SAE training recipe changes, so stale caches are not reused.
const ENGINE_VERSION: u32 = 1;

/// Content-address a trained SAE by everything that determines its weights.
/// Readable (debuggable) rather than hashed; floats are dot-/minus-escaped.
fn cache_key(dataset_id: &str, d: usize, m: usize, l1: f64, epochs: usize, lr: f64, seed: u64) -> String {
    let f = |x: f64| format!("{x}").replace('.', "p").replace('-', "m");
    format!(
        "{dataset_id}__d{d}_a{m}_l1{}_e{epochs}_lr{}_s{seed}_v{ENGINE_VERSION}",
        f(l1),
        f(lr)
    )
}

fn emit(v: Value) {
    let mut o = std::io::stdout().lock();
    let _ = writeln!(o, "{v}");
    let _ = o.flush();
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Analyze {
            dataset,
            output,
            max_dims,
            n_atoms,
            l1,
            epochs,
            lr,
            batch,
            seed,
            segment,
            label_atoms,
            emit_codes,
            cache_dir,
        } => analyze(
            dataset, output, max_dims, n_atoms, l1, epochs, lr, batch, seed, segment, label_atoms,
            emit_codes, cache_dir,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn analyze(
    dataset: PathBuf,
    output: PathBuf,
    max_dims: usize,
    n_atoms: usize,
    l1: f64,
    epochs: usize,
    lr: f64,
    batch: usize,
    seed: u64,
    segment: String,
    label_atoms: bool,
    emit_codes: bool,
    cache_dir: Option<PathBuf>,
) -> Result<()> {
    let ds = Dataset::load(&dataset)?;
    let nc = ds.manifest.n_cols;
    let n_rows = ds.manifest.n_rows;
    let cols = &ds.manifest.columns;

    // Embedding (PCA) columns, capped to max_dims.
    let pca_idx: Vec<usize> = cols
        .iter()
        .enumerate()
        .filter(|(_, c)| matches!(c.kind, ColumnKind::Pca { .. }))
        .map(|(i, _)| i)
        .collect();
    ensure!(!pca_idx.is_empty(), "dataset has no PCA columns");
    let d = pca_idx.len().min(max_dims.max(1));
    let use_idx = &pca_idx[..d];

    // Assemble the [n_rows, d] embedding block and standardize on train stats.
    let mut emb = vec![0f32; n_rows * d];
    for r in 0..n_rows {
        let base = r * nc;
        for (k, &c) in use_idx.iter().enumerate() {
            emb[r * d + k] = ds.features[base + c];
        }
    }
    let (mean, std) = col_stats(&emb, &ds.train_idx, d);
    for r in 0..n_rows {
        for k in 0..d {
            emb[r * d + k] = ((emb[r * d + k] as f64 - mean[k]) / std[k]) as f32;
        }
    }
    let train_x = gather(&emb, &ds.train_idx, d);
    let m = if n_atoms == 0 { 2 * d } else { n_atoms };

    // Content-addressed SAE cache: a hit (same dataset + config) skips the slow
    // training step; encode/probe/label always re-run (they're cheap).
    let cache = cache_dir
        .as_ref()
        .map(|c| c.join(cache_key(&ds.manifest.dataset_id, d, m, l1, epochs, lr, seed)));
    let cached = cache.as_ref().map(|c| c.join("model.mpk").exists()).unwrap_or(false);
    let cfg = sae::SaeConfig { n_atoms: m, l1, epochs, lr, batch_size: batch, seed };
    let model = if cached {
        let c = cache.as_ref().unwrap();
        emit(json!({"event":"log","msg":format!("loaded cached SAE from {}", c.display())}));
        sae::load(c, d, m)?
    } else {
        emit(json!({"event":"log","msg":format!(
            "SAE: {d} embedding dims → {m} atoms, l1 {l1}, {epochs} epochs on {} train rows (no cache hit)",
            ds.train_idx.len())}));
        let (model, _final_loss) = sae::train(&train_x, d, &cfg, &emit);
        if let Some(c) = &cache {
            match sae::save(&model, c) {
                Ok(()) => emit(json!({"event":"log","msg":format!("cached SAE to {}", c.display())})),
                Err(e) => emit(json!({"event":"log","msg":format!("cache save failed: {e:#}")})),
            }
        }
        model
    };
    let out = sae::encode_all(&model, &emb, d, batch);
    emit(json!({"event":"log","msg":format!(
        "SAE {}: var-explained {:.3}, mean active atoms {:.1}/{m} ({:.1}%)",
        if cached { "loaded" } else { "trained" },
        out.var_explained, out.l0_mean, 100.0 * out.l0_mean / m as f64)}));
    // Write a human-readable cache manifest next to a freshly-trained model.
    if !cached {
        if let Some(c) = &cache {
            let _ = std::fs::write(
                c.join("meta.json"),
                serde_json::to_vec_pretty(&json!({
                    "dataset_id": ds.manifest.dataset_id, "input_dims": d, "n_atoms": m,
                    "l1": l1, "epochs": epochs, "lr": lr, "seed": seed,
                    "engine_version": ENGINE_VERSION,
                    "var_explained": out.var_explained, "l0_mean": out.l0_mean,
                }))
                .unwrap_or_default(),
            );
        }
    }

    // Segment mask (the requested categorical value) from the one-hot column.
    let seg_col = cols.iter().position(
        |c| matches!(&c.kind, ColumnKind::Onehot { value, .. } if value == &segment),
    );
    let seg_mask: Option<Vec<bool>> = seg_col.map(|sc| {
        (0..n_rows).map(|r| ds.features[r * nc + sc] > 0.5).collect()
    });

    // Atom interpretability (over all rows) before we move the code into a stage.
    let (target, sep) = atom_stats(&out.code_all, m, &ds.target, seg_mask.as_deref(), n_rows);

    // Auto-interp (Bills et al. 2023): optionally label the top atoms with an LLM
    // from their top-activating items. Gated on --label-atoms AND a key.
    let labels = if label_atoms {
        match llm::LlmConfig::from_env() {
            Some(cfg) => {
                let mut seen = HashSet::new();
                let atoms: Vec<usize> =
                    target.iter().chain(sep.iter()).map(|t| t.0).filter(|a| seen.insert(*a)).collect();
                emit(json!({"event":"log","msg":format!(
                    "auto-interp: labeling {} atoms via {}", atoms.len(), cfg.model)}));
                let content = autointerp::load_content(&dataset, &ds.row_ids, n_rows);
                let vocab = llm::Vocab::from_domain(
                    &Domain::load_or_default(std::path::Path::new(".")).unwrap_or_default(),
                );
                let labels = autointerp::label_atoms_concurrent(
                    &cfg, &vocab, &out.code_all, m, &atoms, &content,
                );
                emit(json!({"event":"log","msg":format!("auto-interp: labeled {}", labels.len())}));
                labels
            }
            None => {
                emit(json!({"event":"log","msg":
                    "--label-atoms set but OPENAI_API_KEY missing; skipping atom labels"}));
                HashMap::new()
            }
        }
    } else {
        HashMap::new()
    };
    // `top_rows`: the dataset row indices whose code most strongly activates each
    // surfaced atom (same order label_atoms_concurrent uses). Downstream tools
    // join these to the corpus metadata to build a per-atom exemplar set.
    // `code` (per-row activation vector) is included only under --emit-codes:
    // a per-model SAE reads it to correlate its own atoms against these
    // embedding concepts (the dropped-signal diff), and it's large.
    let atom_code = |a: usize| -> Vec<f32> {
        (0..n_rows).map(|r| out.code_all[r * m + a]).collect()
    };
    let atoms_by_target: Vec<Value> = target
        .iter()
        .map(|(a, c, f)| {
            let mut v = json!({
                "atom": a, "target_corr": c, "freq": f, "label": labels.get(a),
                "top_rows": autointerp::top_k_rows(&out.code_all, m, *a, 12),
            });
            if emit_codes {
                v["code"] = json!(atom_code(*a));
            }
            v
        })
        .collect();
    let atoms_by_segment: Vec<Value> = sep
        .iter()
        .map(|(a, s, f)| json!({
            "atom": a, "separation": s, "freq": f, "label": labels.get(a),
            "top_rows": autointerp::top_k_rows(&out.code_all, m, *a, 12),
        }))
        .collect();

    // Two stages to probe: the raw embedding block vs the SAE code.
    let stages = vec![
        Stage { name: "input:pca".into(), kind: StageKind::Input, dim: d, values: emb },
        Stage { name: "sae:code".into(), kind: StageKind::Hidden, dim: m, values: out.code_all },
    ];
    let transform = ds.manifest.target.transform;

    let mut probes: Vec<Value> = Vec::new();
    probe_into(&mut probes, &stages, &ds.train_idx, &ds.test_idx, &ds.target, transform, "pooled");

    // Segment probe (the P1 hot spot): restrict the split to segment rows.
    if let Some(mask) = &seg_mask {
        let str_: Vec<u32> = ds.train_idx.iter().copied().filter(|&r| mask[r as usize]).collect();
        let ste: Vec<u32> = ds.test_idx.iter().copied().filter(|&r| mask[r as usize]).collect();
        if str_.len() >= 40 && ste.len() >= 15 {
            emit(json!({"event":"log","msg":format!(
                "segment '{segment}': {} train / {} test rows", str_.len(), ste.len())}));
            probe_into(&mut probes, &stages, &str_, &ste, &ds.target, transform, &segment);
        } else {
            emit(json!({"event":"log","msg":format!(
                "segment '{segment}' too small to probe ({} train / {} test)", str_.len(), ste.len())}));
        }
    }

    // Query-time encoder for the surfaced atoms (union of the two rankings): a
    // client standardizes a query's PCA vector with (mean,std), then
    // relu(z·W[:,j] + b[j]) gives atom j's activation — lighting up the concepts
    // a novel composed query fires.
    let mut surfaced: Vec<usize> = Vec::new();
    {
        let mut seen = HashSet::new();
        for t in target.iter().chain(sep.iter()) {
            if seen.insert(t.0) {
                surfaced.push(t.0);
            }
        }
    }
    let (enc_w, enc_b) = model.encoder_params(); // enc_w row-major [d, m]
    let enc_atoms: serde_json::Map<String, Value> = surfaced
        .iter()
        .map(|&j| {
            let row: Vec<f32> = (0..d).map(|i| enc_w[i * m + j]).collect();
            (j.to_string(), json!({"w": row, "b": enc_b.get(j).copied().unwrap_or(0.0)}))
        })
        .collect();
    let encoder = json!({"mean": mean, "std": std, "atoms": enc_atoms});

    let result = json!({
        "tool": "sae",
        "method": "sparse autoencoder (dictionary learning) on the embedding block; lensing-interp ridge probe of the code",
        "dataset_id": ds.manifest.dataset_id,
        "n_train": ds.train_idx.len(),
        "n_test": ds.test_idx.len(),
        "segment": segment,
        "cached": cached,
        "config": {"input_dims": d, "n_atoms": m, "l1": l1, "epochs": epochs, "lr": lr},
        "recon": {
            "var_explained": out.var_explained,
            "l0_mean": out.l0_mean,
            "l0_frac": out.l0_mean / m as f64,
        },
        "probes": probes,
        "atoms_by_target_corr": atoms_by_target,
        "atoms_by_segment_separation": atoms_by_segment,
        "encoder": encoder,
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

/// Fit ridge probes on [input:pca, sae:code] for the given split and push the
/// two results (tagged with `segment`) into `probes`.
fn probe_into(
    probes: &mut Vec<Value>,
    stages: &[Stage],
    train_idx: &[u32],
    test_idx: &[u32],
    target: &[f32],
    transform: lensing_core::TargetTransform,
    segment: &str,
) {
    let ytr = gather_targets(target, train_idx);
    let yte = gather_targets(target, test_idx);
    let clamp_hi = ytr.iter().cloned().fold(f64::MIN, f64::max) + std::f64::consts::LN_2;
    let res = fit_stages(stages, train_idx, test_idx, &ytr, &yte, transform, clamp_hi, 0.0, &emit);
    for r in &res {
        probes.push(json!({
            "segment": segment,
            "stage": r.name,
            "dim": r.dim,
            "test_r2_log": r.test_r2_log,
            "test_r2_target": r.test_r2_target,
            "test_mae": r.test_mae,
            "test_medape": r.test_medape,
        }));
    }
}

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

/// Per-atom statistics over all rows: activation frequency, correlation with the
/// (log) target, and segment-vs-rest mean-activation separation. Returns the
/// top-12 atoms `(atom, score, freq)` by |target correlation| and by segment
/// separation (score is the corr / the separation respectively).
#[allow(clippy::type_complexity)]
fn atom_stats(
    code: &[f32],
    m: usize,
    y: &[f32],
    seg_mask: Option<&[bool]>,
    n_rows: usize,
) -> (Vec<(usize, f64, f64)>, Vec<(usize, f64, f64)>) {
    let nf = n_rows as f64;
    let ybar = y.iter().map(|&v| v as f64).sum::<f64>() / nf;
    let ysd = (y.iter().map(|&v| (v as f64 - ybar).powi(2)).sum::<f64>() / nf).sqrt().max(1e-9);

    let mut cnt = vec![0u32; m];
    let mut s = vec![0f64; m];
    let mut ss = vec![0f64; m];
    let mut sy = vec![0f64; m];
    let mut sseg = vec![0f64; m];
    let mut srest = vec![0f64; m];
    let mut nseg = 0f64;
    for r in 0..n_rows {
        let yr = y[r] as f64;
        let is_seg = seg_mask.map(|sm| sm[r]).unwrap_or(false);
        if is_seg {
            nseg += 1.0;
        }
        let base = r * m;
        for i in 0..m {
            let v = code[base + i] as f64;
            if v > 1e-6 {
                cnt[i] += 1;
            }
            s[i] += v;
            ss[i] += v * v;
            sy[i] += v * yr;
            if is_seg {
                sseg[i] += v;
            } else {
                srest[i] += v;
            }
        }
    }
    let nrest = nf - nseg;

    let mut target: Vec<(usize, f64, f64)> = Vec::with_capacity(m);
    let mut sep: Vec<(usize, f64, f64)> = Vec::with_capacity(m);
    for i in 0..m {
        let mean = s[i] / nf;
        let sd = (ss[i] / nf - mean * mean).max(0.0).sqrt();
        let cov = sy[i] / nf - mean * ybar;
        let corr = if sd > 1e-9 { cov / (sd * ysd) } else { 0.0 };
        let freq = cnt[i] as f64 / nf;
        target.push((i, corr, freq));
        if seg_mask.is_some() && nseg > 0.0 && nrest > 0.0 && sd > 1e-9 {
            let d = (sseg[i] / nseg - srest[i] / nrest) / sd;
            sep.push((i, d, freq));
        }
    }
    target.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    sep.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    target.truncate(12);
    sep.truncate(12);
    (target, sep)
}
