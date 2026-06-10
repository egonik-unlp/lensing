//! Blend meta-predictor: combines member models' target-space predictions
//! (weighted mean or median vote). Members are frozen promoted models,
//! models.toml definitions trained inside the blend run, or inline
//! predictor+hyperparams recipes. Rides the ordinary contract v2 — the
//! server and UI never learn blends exist.

mod child;
mod columns;
mod combine;
mod spec;
mod viz;

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use serde_json::json;

use lensing_core::registry::{Predictor, Registry};
use lensing_core::{
    artifact, compute_metrics, ColumnDesc, Contract, Dataset, InferenceInput,
    InferencePrediction, InputManifest, Metrics, ModelDefinition, ModelRecord, Prediction,
    CONTRACT_VERSION,
};

use spec::{BlendFile, BlendMember, Hyperparams, MemberSpec, Rule, WeightFit};

#[derive(Parser)]
#[command(name = "predictor-blend")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train/evaluate the blend members on a dataset directory and write
    /// blended metrics/predictions to a run dir.
    Train {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        hyperparams: PathBuf,
    },
    /// Predict target values by fanning a server-featurized input out to the member
    /// models nested in the blend model directory.
    Predict {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Export the ensemble: export each included member to ONNX under
    /// `members/<i>/model.onnx` and write a `combination.json` describing how
    /// the consumer recombines them.
    Export {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}

/// Contract: progress as JSON-lines on stdout.
pub fn emit(v: serde_json::Value) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{v}");
    let _ = out.flush();
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Train { dataset, output, hyperparams } => train(dataset, output, hyperparams),
        Command::Predict { model, input, output } => predict(model, input, output),
        Command::Export { model, output } => export(model, output),
    }
}

/// `combination.json`: how a consumer recombines the exported member ONNX
/// graphs. Each member's `model.onnx` takes its own column subset of the
/// blend's assembled feature vector (selected by `columns`), is inverted to
/// target space (per the shared featurize.json target), then combined by
/// `rule` with the given weights.
#[derive(serde::Serialize)]
struct Combination {
    rule: Rule,
    members: Vec<CombinationMember>,
}

#[derive(serde::Serialize)]
struct CombinationMember {
    index: usize,
    predictor: String,
    /// Export-relative directory holding this member's `model.onnx`.
    dir: String,
    /// Names (in order) of the blend feature columns this member consumes.
    columns: Vec<String>,
    n_cols: usize,
    /// Combination weight (mean rule); 1 for every voter under median.
    weight: f64,
}

/// Export the blend: recursively export each included member to ONNX, then
/// write `combination.json`. Members that cannot be exported (their predictor
/// has no `export` support) fail the whole export with a clear message.
fn export(model_dir: PathBuf, output: PathBuf) -> Result<()> {
    let blend: BlendFile = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("blend.json"))
            .context("read blend.json from model dir")?,
    )
    .context("parse blend.json")?;
    let registry = Registry::load(Path::new("registry.toml"))?;
    std::fs::create_dir_all(&output)?;

    let mut members = Vec::new();
    for m in blend.members.iter().filter(|m| m.included) {
        let label = format!("m{} {}", m.index, m.source.as_deref().unwrap_or(&m.predictor));
        let predictor = registry
            .get(&m.predictor)
            .with_context(|| format!("member predictor {:?} not in registry", m.predictor))?;
        ensure!(
            predictor.supports_export(),
            "member {label}: predictor {} has no ONNX export — the blend cannot be exported \
             until every included member's family supports export",
            m.predictor
        );
        let member_model_dir = model_dir.join("members").join(m.index.to_string());
        let dir = format!("members/{}", m.index);
        let member_out_dir = output.join("members").join(m.index.to_string());
        child::export_member(&label, predictor, &member_model_dir, &member_out_dir)?;
        emit(json!({"event": "log", "msg": format!("[{label}] exported {dir}/model.onnx")}));
        members.push(CombinationMember {
            index: m.index,
            predictor: m.predictor.clone(),
            dir,
            columns: m.columns.clone(),
            n_cols: m.n_cols,
            weight: blend.weights[m.index],
        });
    }
    ensure!(!members.is_empty(), "blend.json has no included members to export");

    let combination = Combination { rule: blend.rule, members };
    std::fs::write(
        output.join("combination.json"),
        serde_json::to_vec_pretty(&combination)?,
    )?;
    emit(json!({"event": "log", "msg": format!(
        "exported {} members, rule {:?}", combination.members.len(), blend.rule)}));
    emit(json!({"event": "done"}));
    Ok(())
}

/// `models.toml` at the repository root (cwd is the repo root by contract).
#[derive(Debug, Default, Deserialize)]
struct Definitions {
    #[serde(default)]
    definitions: Vec<ModelDefinition>,
}

fn load_definitions() -> Result<Definitions> {
    match std::fs::read_to_string("models.toml") {
        Ok(text) => toml::from_str(&text).context("parse models.toml"),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Definitions::default()),
        Err(e) => Err(e).context("read models.toml"),
    }
}

/// A member with its source resolved and its column view of the dataset
/// computed.
struct Resolved {
    index: usize,
    kind: &'static str,
    predictor: Predictor,
    source: Option<String>,
    weight: f64,
    exclude_blocks: Vec<String>,
    /// Indices into the blend dataset/input columns, in member order.
    col_idx: Vec<usize>,
    col_descs: Vec<ColumnDesc>,
    frozen: bool,
    /// Trained members: launch-ready hyperparams.
    hyperparams: Option<serde_json::Value>,
}

impl Resolved {
    fn label(&self) -> String {
        match &self.source {
            Some(s) => format!("m{} {}", self.index, s),
            None => format!("m{} {}", self.index, self.predictor.name),
        }
    }
}

fn resolve_members(
    hp: &Hyperparams,
    registry: &Registry,
    defs: &Definitions,
    ds: &Dataset,
    dataset_dir: &Path,
    run_dir: &Path,
) -> Result<Vec<Resolved>> {
    let mut out = Vec::with_capacity(hp.members.len());
    for (i, m) in hp.members.iter().enumerate() {
        m.validate(i)?;
        let resolved = if let Some(model_name) = &m.model {
            resolve_frozen(i, m, model_name, registry, ds, dataset_dir, run_dir)?
        } else {
            let (kind, predictor_name, member_hp, source) = if let Some(def_name) = &m.definition
            {
                let def = defs
                    .definitions
                    .iter()
                    .find(|d| &d.name == def_name)
                    .with_context(|| {
                        format!("member {i}: definition {def_name:?} not in models.toml")
                    })?;
                ("definition", def.predictor.clone(), def.hyperparams.clone(), Some(def_name.clone()))
            } else {
                let p = m.predictor.clone().unwrap();
                ("inline", p, m.hyperparams.clone().unwrap_or(json!({})), None)
            };
            let predictor = registry
                .get(&predictor_name)
                .with_context(|| {
                    format!("member {i}: predictor {predictor_name:?} not in registry")
                })?
                .clone();
            ensure!(predictor.name != "blend", "member {i}: blends cannot nest blends");
            ensure!(
                predictor.supports_predict(),
                "member {i}: predictor {} is train-only — a promoted blend could not invoke it",
                predictor.name
            );
            let merged = predictor.merge_hyperparams(Some(member_hp))?;
            let col_idx = columns::mask_by_blocks(&ds.manifest.columns, &m.exclude_blocks)
                .with_context(|| format!("member {i}"))?;
            let col_descs = col_idx.iter().map(|&c| ds.manifest.columns[c].clone()).collect();
            Resolved {
                index: i,
                kind,
                predictor,
                source,
                weight: m.weight.unwrap_or(1.0),
                exclude_blocks: m.exclude_blocks.clone(),
                col_idx,
                col_descs,
                frozen: false,
                hyperparams: Some(merged),
            }
        };
        out.push(resolved);
    }
    Ok(out)
}

/// Frozen member: copy the promoted model dir into the run dir and verify it
/// can consume this dataset's features (column names + identical PCA fit —
/// `pca_5` from a different fit is a different direction under the same name).
fn resolve_frozen(
    i: usize,
    m: &MemberSpec,
    model_name: &str,
    registry: &Registry,
    ds: &Dataset,
    dataset_dir: &Path,
    run_dir: &Path,
) -> Result<Resolved> {
    let src = Path::new("data/models").join(model_name);
    ensure!(
        src.join("record.json").is_file(),
        "member {i}: promoted model {model_name:?} not found"
    );
    let record: ModelRecord =
        serde_json::from_str(&std::fs::read_to_string(src.join("record.json"))?)?;
    ensure!(record.predictor != "blend", "member {i}: blends cannot nest blends");
    let predictor = registry
        .get(&record.predictor)
        .with_context(|| format!("member {i}: predictor {:?} not in registry", record.predictor))?
        .clone();
    ensure!(
        predictor.supports_predict(),
        "member {i}: predictor {} is train-only",
        predictor.name
    );
    let contract: Contract =
        serde_json::from_str(&std::fs::read_to_string(src.join("contract.json"))?)
            .with_context(|| format!("member {i}: read contract.json of {model_name}"))?;

    // PCA guard: same shape, same mean, byte-identical components.
    let mp = &contract.pca;
    let dp = &ds.manifest.pca;
    let components_equal = || -> Result<bool> {
        let a = std::fs::read(src.join("pca_components.f32"))?;
        let b = std::fs::read(dataset_dir.join("pca_components.f32"))?;
        Ok(a == b)
    };
    if mp.components_shape != dp.components_shape || mp.mean != dp.mean || !components_equal()? {
        bail!(
            "member {i}: model {model_name:?} was trained on a different PCA fit than this \
             dataset — its pca_N columns are different directions under the same names. \
             Frozen members require an identical PCA (same corpus rows + split)."
        );
    }

    let names: Vec<String> = contract.columns.iter().map(|c| c.name.clone()).collect();
    let col_idx = columns::mask_by_names(&ds.manifest.columns, &names)
        .with_context(|| format!("member {i}: model {model_name:?}"))?;

    let member_dir = run_dir.join("members").join(i.to_string());
    copy_dir(&src, &member_dir)
        .with_context(|| format!("member {i}: copy model {model_name:?}"))?;

    Ok(Resolved {
        index: i,
        kind: "model",
        predictor,
        source: Some(model_name.to_string()),
        weight: m.weight.unwrap_or(1.0),
        exclude_blocks: vec![],
        col_idx,
        col_descs: contract.columns.clone(),
        frozen: true,
        hyperparams: None,
    })
}

/// Recursive copy, skipping atomic-write temps and tmp dirs.
fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for e in std::fs::read_dir(src)?.flatten() {
        let name = e.file_name();
        let s = name.to_string_lossy();
        if s.contains(".tmp") || s.contains("-tmp") || s == "tmp" {
            continue;
        }
        let ft = e.file_type()?;
        if ft.is_dir() {
            copy_dir(&e.path(), &dst.join(&name))?;
        } else if ft.is_file() {
            std::fs::copy(e.path(), dst.join(&name))?;
        }
    }
    Ok(())
}

/// Write a member-local input mini-artifact (the predict contract's format)
/// for the given dataset rows, column-selected for the member.
fn write_member_input(dir: &Path, ds: &Dataset, rows: &[u32], member: &Resolved) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let features = columns::subset(&ds.features, ds.manifest.n_cols, rows, &member.col_idx);
    artifact::write_f32(&dir.join("features.f32"), &features)?;
    let row_ids: Vec<u64> = rows.iter().map(|&r| ds.row_ids[r as usize]).collect();
    artifact::write_u64(&dir.join("row_ids.u64"), &row_ids)?;
    let manifest = InputManifest {
        n_rows: rows.len(),
        n_cols: member.col_idx.len(),
        columns: member.col_descs.clone(),
        target: ds.manifest.target.clone(),
    };
    std::fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&manifest)?)?;
    Ok(())
}

/// Write a member-local training dataset: the blend dataset column-selected
/// for the member, with the (possibly val-reduced) train split.
fn write_member_dataset(
    dir: &Path,
    ds: &Dataset,
    member: &Resolved,
    train_idx: &[u32],
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let all_rows: Vec<u32> = (0..ds.manifest.n_rows as u32).collect();
    let features = columns::subset(&ds.features, ds.manifest.n_cols, &all_rows, &member.col_idx);
    artifact::write_f32(&dir.join("features.f32"), &features)?;
    artifact::write_f32(&dir.join("target.f32"), &ds.target)?;
    artifact::write_u64(&dir.join("row_ids.u64"), &ds.row_ids)?;
    artifact::write_u32(&dir.join("train_idx.u32"), train_idx)?;
    artifact::write_u32(&dir.join("test_idx.u32"), &ds.test_idx)?;

    let mut manifest = ds.manifest.clone();
    manifest.n_cols = member.col_idx.len();
    manifest.columns = member.col_descs.clone();
    manifest.split.n_train = train_idx.len();
    std::fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&manifest)?)?;
    Ok(())
}

/// Predictions keyed by row id, asserted to cover exactly `rows`.
fn preds_for_rows(
    label: &str,
    by_row: &HashMap<u64, f64>,
    ds: &Dataset,
    rows: &[u32],
) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(rows.len());
    for &r in rows {
        let id = ds.row_ids[r as usize];
        let p = by_row
            .get(&id)
            .with_context(|| format!("member {label} predicted no value for row {id}"))?;
        out.push(*p);
    }
    Ok(out)
}

fn train(dataset_dir: PathBuf, run_dir: PathBuf, hp_path: PathBuf) -> Result<()> {
    let hp: Hyperparams =
        serde_json::from_str(&std::fs::read_to_string(&hp_path).context("read hyperparams")?)
            .context("parse hyperparams")?;
    ensure!(hp.members.len() >= 2, "a blend needs at least 2 members");
    if hp.rule == Rule::Median {
        ensure!(
            hp.members.iter().all(|m| m.weight.is_none()),
            "rule=median is an unweighted vote; remove member weights"
        );
        ensure!(hp.weight_fit == WeightFit::None, "rule=median has no weights to fit");
    }
    if hp.weight_fit == WeightFit::Grid {
        ensure!(
            hp.val_fraction > 0.0 && hp.val_fraction <= 0.5,
            "val_fraction must be in (0, 0.5]"
        );
    }

    let ds = Dataset::load(&dataset_dir)?;
    let registry = Registry::load(Path::new("registry.toml"))?;
    let defs = load_definitions()?;
    let members = resolve_members(&hp, &registry, &defs, &ds, &dataset_dir, &run_dir)?;

    emit(json!({"event": "log", "msg": format!(
        "blend: {} members ({}), rule {:?}, weight_fit {:?}",
        members.len(),
        members.iter().map(|m| m.label()).collect::<Vec<_>>().join(", "),
        hp.rule, hp.weight_fit
    )}));
    let _ = std::fs::write(run_dir.join("viz.svg"), viz::render(&members, hp.rule));

    // Validation carve-out for weight fitting: trained members see
    // train-minus-val, the test split stays untouched.
    let (fit_train_idx, val_idx): (Vec<u32>, Vec<u32>) = if hp.weight_fit == WeightFit::Grid {
        let (keep, val) = lensing_core::shuffle::train_test_split(
            ds.train_idx.len(),
            hp.val_fraction,
            ds.manifest.split.seed,
        );
        (
            keep.iter().map(|&p| ds.train_idx[p as usize]).collect(),
            val.iter().map(|&p| ds.train_idx[p as usize]).collect(),
        )
    } else {
        (ds.train_idx.clone(), Vec::new())
    };

    let blend_stop = run_dir.join(child::STOP_FILE);
    let mut stopped = false;
    // Per member: test predictions (target space), val predictions, solo metrics.
    let mut test_preds: Vec<Option<Vec<f64>>> = vec![None; members.len()];
    let mut val_preds: Vec<Option<Vec<f64>>> = vec![None; members.len()];
    let mut solo: Vec<Option<Metrics>> = vec![None; members.len()];
    let actual_test: Vec<f64> = ds
        .test_idx
        .iter()
        .map(|&r| ds.manifest.target.transform.invert(ds.target[r as usize] as f64))
        .collect();

    for member in &members {
        let label = member.label();
        let member_dir = run_dir.join("members").join(member.index.to_string());
        std::fs::create_dir_all(&member_dir)?;

        let by_row: HashMap<u64, f64> = if member.frozen {
            // Frozen: evaluate via the member's predict CLI on the test rows.
            let eval_dir = member_dir.join("eval-tmp");
            write_member_input(&eval_dir, &ds, &ds.test_idx, member)?;
            let preds = child::predict_member(
                &label,
                &member.predictor,
                &member_dir,
                &eval_dir,
                &eval_dir.join("out.json"),
            )?;
            preds.into_iter().map(|p| (p.row_id, p.predicted)).collect()
        } else {
            if stopped {
                emit(json!({"event": "log", "msg": format!(
                    "[{label}] skipped: stop requested before this member trained"
                )}));
                continue;
            }
            let ds_dir = member_dir.join("ds-tmp");
            write_member_dataset(&ds_dir, &ds, member, &fit_train_idx)?;
            let member_hp_path = member_dir.join("hp.json");
            std::fs::write(
                &member_hp_path,
                serde_json::to_vec_pretty(member.hyperparams.as_ref().unwrap())?,
            )?;
            emit(json!({"event": "log", "msg": format!(
                "[{label}] training {} on {} cols × {} train rows",
                member.predictor.name, member.col_idx.len(), fit_train_idx.len()
            )}));
            stopped |= child::train_member(
                &label,
                &member.predictor,
                &ds_dir,
                &member_dir,
                &member_hp_path,
                &blend_stop,
            )?;
            // The member's run-dir output is also its model dir at predict
            // time; predictors read hyperparams.json there (promotion's name).
            std::fs::copy(&member_hp_path, member_dir.join("hyperparams.json"))?;
            let preds: Vec<Prediction> = serde_json::from_str(
                &std::fs::read_to_string(member_dir.join("predictions.json"))
                    .with_context(|| format!("member {label} wrote no predictions.json"))?,
            )?;
            preds.into_iter().map(|p| (p.row_id, p.predicted)).collect()
        };

        let preds = preds_for_rows(&label, &by_row, &ds, &ds.test_idx)?;
        let pairs: Vec<(f64, f64)> =
            actual_test.iter().copied().zip(preds.iter().copied()).collect();
        let m = compute_metrics(&pairs);
        emit(json!({"event": "log", "msg": format!(
            "[{label}] solo test MAE {:.0} medAPE {:.1}% R² {:.3}",
            m.mae, m.medape * 100.0, m.r2
        )}));
        emit(json!({"event": "epoch", "epoch": member.index + 1,
            "total_epochs": members.len(), "train_loss": 0.0, "val_loss": m.mae}));
        solo[member.index] = Some(m);
        test_preds[member.index] = Some(preds);

        if hp.weight_fit == WeightFit::Grid {
            // Validation predictions via the member's predict CLI (works for
            // frozen and just-trained members alike).
            let val_dir = member_dir.join("val-eval-tmp");
            write_member_input(&val_dir, &ds, &val_idx, member)?;
            let preds = child::predict_member(
                &label,
                &member.predictor,
                &member_dir,
                &val_dir,
                &val_dir.join("out.json"),
            )?;
            let by_row: HashMap<u64, f64> =
                preds.into_iter().map(|p| (p.row_id, p.predicted)).collect();
            val_preds[member.index] = Some(preds_for_rows(&label, &by_row, &ds, &val_idx)?);
        }
    }

    // Stop can exclude trained members; the blend still saves over what
    // finished ("stopped runs must be saved").
    let included: Vec<usize> = (0..members.len()).filter(|&i| test_preds[i].is_some()).collect();
    ensure!(!included.is_empty(), "stop landed before any member finished");
    if included.len() < members.len() {
        emit(json!({"event": "log", "msg": format!(
            "blend combines {}/{} members (stop excluded the rest)",
            included.len(), members.len()
        )}));
    }

    // Final weights, parallel to ALL members (excluded → 0).
    let mut weights = vec![0.0f64; members.len()];
    match hp.rule {
        Rule::Median => {}
        Rule::Mean => {
            let included_w: Vec<f64> = included.iter().map(|&i| members[i].weight).collect();
            let fitted = if hp.weight_fit == WeightFit::Grid && included.len() >= 2 {
                let vp: Vec<Vec<f64>> =
                    included.iter().map(|&i| val_preds[i].clone().unwrap()).collect();
                let actual_val: Vec<f64> = val_idx
                    .iter()
                    .map(|&r| ds.manifest.target.transform.invert(ds.target[r as usize] as f64))
                    .collect();
                let w = combine::grid_fit(&vp, &actual_val)?;
                emit(json!({"event": "log", "msg": format!(
                    "grid weight fit on {} val rows: {:?}", val_idx.len(), w
                )}));
                w
            } else {
                combine::normalize(&included_w)?
            };
            for (&i, w) in included.iter().zip(fitted) {
                weights[i] = w;
            }
        }
    }

    let included_preds: Vec<Vec<f64>> =
        included.iter().map(|&i| test_preds[i].clone().unwrap()).collect();
    let included_weights: Vec<f64> = included.iter().map(|&i| weights[i]).collect();
    let blended = combine::combine(hp.rule, &included_preds, &included_weights);

    let pairs: Vec<(f64, f64)> =
        actual_test.iter().copied().zip(blended.iter().copied()).collect();
    let metrics = compute_metrics(&pairs);
    let predictions: Vec<Prediction> = ds
        .test_idx
        .iter()
        .zip(actual_test.iter().zip(&blended))
        .map(|(&r, (&actual, &predicted))| Prediction {
            row_id: ds.row_ids[r as usize],
            actual,
            predicted,
        })
        .collect();
    std::fs::write(run_dir.join("predictions.json"), serde_json::to_vec(&predictions)?)?;
    std::fs::write(run_dir.join("metrics.json"), serde_json::to_vec_pretty(&metrics)?)?;

    let blend_file = BlendFile {
        contract_version: CONTRACT_VERSION,
        rule: hp.rule,
        weight_fit: hp.weight_fit,
        weights,
        members: members
            .iter()
            .map(|m| BlendMember {
                index: m.index,
                kind: m.kind.to_string(),
                predictor: m.predictor.name.clone(),
                source: m.source.clone(),
                columns: m.col_descs.iter().map(|c| c.name.clone()).collect(),
                n_cols: m.col_idx.len(),
                exclude_blocks: m.exclude_blocks.clone(),
                frozen: m.frozen,
                included: test_preds[m.index].is_some(),
                solo_metrics: solo[m.index].clone(),
            })
            .collect(),
    };
    std::fs::write(run_dir.join("blend.json"), serde_json::to_vec_pretty(&blend_file)?)?;

    // Train-time-only artifacts; promotion would skip them by name anyway.
    for m in &members {
        let member_dir = run_dir.join("members").join(m.index.to_string());
        for tmp in ["ds-tmp", "eval-tmp", "val-eval-tmp"] {
            let _ = std::fs::remove_dir_all(member_dir.join(tmp));
        }
    }

    emit(json!({"event": "log", "msg": format!(
        "blend test MAE {:.0} medAPE {:.1}% R² {:.3} ({} members, rule {:?})",
        metrics.mae, metrics.medape * 100.0, metrics.r2, included.len(), hp.rule
    )}));
    emit(json!({"event": "done"}));
    Ok(())
}

fn predict(model_dir: PathBuf, input_dir: PathBuf, output: PathBuf) -> Result<()> {
    let blend: BlendFile = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("blend.json"))
            .context("read blend.json from model dir")?,
    )
    .context("parse blend.json")?;
    let registry = Registry::load(Path::new("registry.toml"))?;
    let input = InferenceInput::load(&input_dir)?;
    emit(json!({"event": "log", "msg": format!(
        "blend predict: {} rows through {} members",
        input.manifest.n_rows,
        blend.members.iter().filter(|m| m.included).count()
    )}));

    let tmp_base = std::env::temp_dir().join(format!(
        "lensing-blend-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    ));

    let result = (|| -> Result<Vec<InferencePrediction>> {
        let mut member_preds: Vec<Vec<f64>> = Vec::new();
        let mut member_weights: Vec<f64> = Vec::new();
        for m in blend.members.iter().filter(|m| m.included) {
            let label = format!("m{} {}", m.index, m.source.as_deref().unwrap_or(&m.predictor));
            let predictor = registry
                .get(&m.predictor)
                .with_context(|| format!("member predictor {:?} not in registry", m.predictor))?;
            let col_idx = columns::mask_by_names(&input.manifest.columns, &m.columns)
                .with_context(|| format!("member {label}"))?;

            let member_input = tmp_base.join(m.index.to_string());
            std::fs::create_dir_all(&member_input)?;
            let all_rows: Vec<u32> = (0..input.manifest.n_rows as u32).collect();
            let features =
                columns::subset(&input.features, input.manifest.n_cols, &all_rows, &col_idx);
            artifact::write_f32(&member_input.join("features.f32"), &features)?;
            artifact::write_u64(&member_input.join("row_ids.u64"), &input.row_ids)?;
            let manifest = InputManifest {
                n_rows: input.manifest.n_rows,
                n_cols: col_idx.len(),
                columns: col_idx.iter().map(|&c| input.manifest.columns[c].clone()).collect(),
                target: input.manifest.target.clone(),
            };
            std::fs::write(
                member_input.join("manifest.json"),
                serde_json::to_vec_pretty(&manifest)?,
            )?;

            let preds = child::predict_member(
                &label,
                predictor,
                &model_dir.join("members").join(m.index.to_string()),
                &member_input,
                &member_input.join("out.json"),
            )?;
            let by_row: HashMap<u64, f64> =
                preds.into_iter().map(|p| (p.row_id, p.predicted)).collect();
            let ordered: Vec<f64> = input
                .row_ids
                .iter()
                .map(|id| {
                    by_row
                        .get(id)
                        .copied()
                        .with_context(|| format!("member {label} predicted no value for row {id}"))
                })
                .collect::<Result<_>>()?;
            member_preds.push(ordered);
            member_weights.push(blend.weights[m.index]);
        }
        ensure!(!member_preds.is_empty(), "blend.json has no included members");

        let blended = combine::combine(blend.rule, &member_preds, &member_weights);
        Ok(input
            .row_ids
            .iter()
            .zip(blended)
            .map(|(&row_id, predicted)| InferencePrediction { row_id, predicted })
            .collect())
    })();
    let _ = std::fs::remove_dir_all(&tmp_base);

    let predictions = result?;
    std::fs::write(&output, serde_json::to_vec(&predictions)?)?;
    emit(json!({"event": "done"}));
    Ok(())
}
