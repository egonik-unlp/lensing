//! MLP target-value predictor on the language-neutral dataset artifact.
//! burn 0.21, ndarray CPU backend, hand-written training loop.

mod model;
mod onnx_export;
mod scaler;
mod viz;

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use serde_json::json;

use lensing_core::{compute_metrics, Dataset, InferenceInput, InferencePrediction, Prediction};

#[derive(Parser)]
#[command(name = "predictor-burn-mlp")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train on a dataset directory and write metrics/predictions to a run dir.
    Train {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        hyperparams: PathBuf,
    },
    /// Predict target values for a server-featurized input mini-artifact using a
    /// promoted model directory (checkpoint + scaler + hyperparams snapshot).
    Predict {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Export the trained model as a portable `model.onnx` into the output
    /// directory. The graph's input is the assembled feature vector (scaler +
    /// network baked in); output is the transformed-space target.
    Export {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Debug, Deserialize)]
struct Hyperparams {
    #[serde(default = "d_epochs")]
    epochs: usize,
    #[serde(default = "d_lr")]
    lr: f64,
    #[serde(default = "d_batch")]
    batch_size: usize,
    #[serde(default = "d_hidden")]
    hidden: Vec<usize>,
    #[serde(default = "d_dropout")]
    dropout: f64,
    /// Hidden-layer activation. Defaults to relu so promoted models whose
    /// hyperparams snapshot predates this knob keep working unchanged.
    #[serde(default)]
    activation: model::Activation,
    /// Save a loadable checkpoint every N epochs (0 = end only).
    #[serde(default)]
    checkpoint_every: usize,
    /// Early stopping: stop once val loss hasn't improved in this many
    /// epochs and restore the best-epoch weights (0 = off). The monitored
    /// split is a `val_fraction` carve-out of the train split — never the
    /// test split, which stays untouched for final metrics.
    #[serde(default)]
    patience: usize,
    /// Fraction of the train split held out as the early-stop validation
    /// set. Only used when `patience > 0`.
    #[serde(default = "d_val_fraction")]
    val_fraction: f64,
    /// Cap transformed-space predictions at the train-target max + ln(2)
    /// (≈ 2× the highest train target): defuses the expm1 blowup mode that
    /// makes shallow nets unusable. Off by default.
    #[serde(default)]
    clamp_output: bool,
}

fn d_epochs() -> usize { 50 }
fn d_lr() -> f64 { 1e-3 }
fn d_batch() -> usize { 256 }
fn d_hidden() -> Vec<usize> { vec![256, 128] }
fn d_dropout() -> f64 { 0.1 }
fn d_val_fraction() -> f64 { 0.15 }

/// Contract: progress as JSON-lines on stdout.
fn emit(v: serde_json::Value) {
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

/// Export the trained MLP to `<output>/model.onnx`: rebuild the architecture
/// from the hyperparams snapshot, load the weights and scaler, and bake the
/// standardizer + dense stack (+ optional output clamp) into one ONNX graph.
fn export(model_dir: PathBuf, output: PathBuf) -> Result<()> {
    let hp: Hyperparams = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("hyperparams.json"))
            .context("read hyperparams.json from model dir")?,
    )
    .context("parse hyperparams.json")?;

    let scaler = scaler::Scaler::load(&model_dir.join("scaler.json"))
        .context("load scaler.json from model dir")?;
    let n_cols = scaler.mean.len();
    let mlp = model::load(&model_dir.join("model"), n_cols, &hp.hidden, hp.dropout, hp.activation)?;

    let clamp_max_t = if hp.clamp_output {
        let clamp: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(model_dir.join("clamp.json"))
                .context("clamp_output model is missing clamp.json")?,
        )?;
        Some(clamp["max_t"].as_f64().context("clamp.json missing max_t")? as f32)
    } else {
        None
    };

    onnx_export::write_onnx(&mlp, &scaler, clamp_max_t, &output)?;
    emit(json!({"event":"log","msg":format!(
        "exported model.onnx: input [N, {n_cols}] (scaler + {} dense layers baked in)",
        hp.hidden.len() + 1)}));
    emit(json!({"event":"done"}));
    Ok(())
}

fn predict(model_dir: PathBuf, input_dir: PathBuf, output: PathBuf) -> Result<()> {
    let hp: Hyperparams = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("hyperparams.json"))
            .context("read hyperparams.json from model dir")?,
    )
    .context("parse hyperparams.json")?;

    let input = InferenceInput::load(&input_dir)?;
    let n_cols = input.manifest.n_cols;
    emit(json!({"event":"log","msg":format!(
        "predicting {} rows, {} features, mlp {:?}",
        input.manifest.n_rows, n_cols, hp.hidden)}));

    let scaler = scaler::Scaler::load(&model_dir.join("scaler.json"))
        .context("load scaler.json from model dir")?;
    anyhow::ensure!(
        scaler.mean.len() == n_cols,
        "scaler was fit on {} columns, input has {n_cols}",
        scaler.mean.len()
    );
    let x = scaler.transform(&input.features);

    let mlp = model::load(&model_dir.join("model"), n_cols, &hp.hidden, hp.dropout, hp.activation)?;
    let mut predicted_t = model::predict_loaded(&mlp, &x, n_cols, hp.batch_size);
    if hp.clamp_output {
        // Same bound training applied; promotion copies clamp.json verbatim.
        let clamp: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(model_dir.join("clamp.json"))
                .context("clamp_output model is missing clamp.json")?,
        )?;
        let max_t = clamp["max_t"].as_f64().context("clamp.json missing max_t")? as f32;
        for p in predicted_t.iter_mut() {
            *p = p.min(max_t);
        }
    }

    let transform = input.manifest.target.transform;
    let predictions: Vec<InferencePrediction> = input
        .row_ids
        .iter()
        .zip(&predicted_t)
        .map(|(&row_id, &p)| InferencePrediction {
            row_id,
            predicted: transform.invert(p as f64).max(0.0),
        })
        .collect();
    std::fs::write(&output, serde_json::to_vec(&predictions)?)
        .with_context(|| format!("write {}", output.display()))?;

    emit(json!({"event":"done"}));
    Ok(())
}

fn train(dataset_dir: PathBuf, run_dir: PathBuf, hp_path: PathBuf) -> Result<()> {
    let hp: Hyperparams = serde_json::from_str(
        &std::fs::read_to_string(&hp_path).context("read hyperparams")?,
    )
    .context("parse hyperparams")?;
    anyhow::ensure!(hp.epochs >= 1, "epochs must be >= 1");
    anyhow::ensure!(hp.batch_size >= 1, "batch_size must be >= 1");
    anyhow::ensure!(!hp.hidden.is_empty(), "hidden must have at least one layer");
    std::fs::create_dir_all(&run_dir)?;

    let ds = Dataset::load(&dataset_dir)?;
    let n_cols = ds.manifest.n_cols;
    emit(json!({"event":"log","msg":format!(
        "dataset {}: {} train / {} test rows, {} features",
        ds.manifest.dataset_id, ds.train_idx.len(), ds.test_idx.len(), n_cols)}));
    emit(json!({"event":"log","msg":format!(
        "mlp {:?}, {} activation, dropout {}, lr {}, batch {}, {} epochs",
        hp.hidden, hp.activation.name(), hp.dropout, hp.lr, hp.batch_size, hp.epochs)}));

    // Architecture diagram (registry capability `visualization`): written
    // before the first epoch so the live run view can show it immediately.
    viz::write_viz(&run_dir.join("viz.svg"), n_cols, &hp.hidden, hp.dropout, hp.activation.name())?;

    // Early stopping (patience > 0): carve a held-out validation set from
    // the TRAIN split to monitor — never the test split, which stays
    // untouched for final metrics. Deterministic carve via the same seeded
    // splitter that produced the dataset's train/test split.
    let (fit_idx, val_idx) = if hp.patience > 0 {
        anyhow::ensure!(
            hp.val_fraction > 0.0 && hp.val_fraction < 1.0,
            "val_fraction must be in (0, 1) when patience > 0"
        );
        let (fit_pos, val_pos) =
            lensing_core::shuffle::train_test_split(ds.train_idx.len(), hp.val_fraction, 1337);
        anyhow::ensure!(
            !val_pos.is_empty() && !fit_pos.is_empty(),
            "early-stop carve left an empty split; adjust val_fraction"
        );
        let map = |pos: Vec<u32>| -> Vec<u32> {
            pos.into_iter().map(|p| ds.train_idx[p as usize]).collect()
        };
        (map(fit_pos), map(val_pos))
    } else {
        (ds.train_idx.clone(), Vec::new())
    };
    if hp.patience > 0 {
        emit(json!({"event":"log","msg":format!(
            "early stopping: patience {}, monitoring {} held-out train rows \
             (val_fraction {}); test split untouched",
            hp.patience, val_idx.len(), hp.val_fraction)}));
    }

    let (train_x_raw, train_y) = ds.gather(&fit_idx);
    let (val_x_raw, val_y) = ds.gather(&val_idx);
    let (test_x_raw, test_y) = ds.gather(&ds.test_idx);

    // Standardize features with train-split (fit-slice) statistics only.
    let scaler = scaler::Scaler::fit(&train_x_raw, n_cols);
    scaler.save(&run_dir.join("scaler.json"))?;
    let train_x = scaler.transform(&train_x_raw);
    let val_x = scaler.transform(&val_x_raw);
    let test_x = scaler.transform(&test_x_raw);

    // Monitored split: the early-stop carve-out when patience > 0, else the
    // test split (legacy val curve).
    let (mon_x, mon_y): (&[f32], &[f32]) = if hp.patience > 0 {
        (&val_x, &val_y)
    } else {
        (&test_x, &test_y)
    };

    let trained = model::run_training(
        model::TrainInputs {
            train_x: &train_x,
            train_y: &train_y,
            val_x: mon_x,
            val_y: mon_y,
            n_cols,
        },
        model::TrainConfig {
            epochs: hp.epochs,
            lr: hp.lr,
            batch_size: hp.batch_size,
            hidden: hp.hidden.clone(),
            dropout: hp.dropout,
            activation: hp.activation,
            seed: 1337,
            checkpoint_every: hp.checkpoint_every,
            patience: hp.patience,
        },
        &run_dir,
        &emit,
    )?;

    // Predictions in target space (optionally clamped in transformed space).
    let transform = ds.manifest.target.transform;
    let mut predicted_t = model::predict(&trained, &test_x, n_cols, hp.batch_size);
    if hp.clamp_output {
        let max_t = train_y.iter().cloned().fold(f32::MIN, f32::max) + std::f32::consts::LN_2;
        std::fs::write(run_dir.join("clamp.json"), serde_json::to_vec(&json!({"max_t": max_t}))?)?;
        let n_clamped = predicted_t.iter().filter(|p| **p > max_t).count();
        for p in predicted_t.iter_mut() {
            *p = p.min(max_t);
        }
        emit(json!({"event":"log","msg":format!(
            "output clamp at t={max_t:.3} (~2x max train target): {n_clamped} test predictions capped")}));
    }
    let mut predictions = Vec::with_capacity(ds.test_idx.len());
    let mut pairs = Vec::with_capacity(ds.test_idx.len());
    for (j, &row) in ds.test_idx.iter().enumerate() {
        let actual = transform.invert(test_y[j] as f64);
        let predicted = transform.invert(predicted_t[j] as f64).max(0.0);
        pairs.push((actual, predicted));
        predictions.push(Prediction { row_id: ds.row_ids[row as usize], actual, predicted });
    }
    let metrics = compute_metrics(&pairs);
    std::fs::write(run_dir.join("metrics.json"), serde_json::to_vec(&metrics)?)?;
    std::fs::write(run_dir.join("predictions.json"), serde_json::to_vec(&predictions)?)?;
    model::save_atomic(&trained, &run_dir)?;

    emit(json!({"event":"log","msg":format!(
        "MAE {:.0}  RMSE {:.0}  medAPE {:.1}%  R² {:.3}",
        metrics.mae, metrics.rmse, metrics.medape * 100.0, metrics.r2)}));
    emit(json!({"event":"done"}));
    Ok(())
}
