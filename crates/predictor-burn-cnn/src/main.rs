//! 1D CNN target-value predictor on the language-neutral dataset artifact.
//! burn 0.21, ndarray CPU backend, hand-written training loop.
//!
//! The leading PCA columns of the feature vector are convolved as a
//! 1-channel signal; the metadata tail (numeric + one-hots) joins the dense
//! head after global pooling. Mirrored by predictors/torch-cnn (PyTorch) and
//! predictors/flux-cnn (Flux.jl).

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

use lensing_core::manifest::{ColumnDesc, ColumnKind};
use lensing_core::{compute_metrics, Dataset, InferenceInput, InferencePrediction, Prediction};

#[derive(Parser)]
#[command(name = "predictor-burn-cnn")]
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
    /// directory (scaler + conv/dense stack baked into one graph).
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
    #[serde(default = "d_channels")]
    channels: Vec<usize>,
    #[serde(default = "d_kernel")]
    kernel_size: usize,
    #[serde(default = "d_dense_hidden")]
    dense_hidden: usize,
    #[serde(default = "d_dropout")]
    dropout: f64,
    /// Save a loadable checkpoint every N epochs (0 = end only).
    #[serde(default)]
    checkpoint_every: usize,
}

fn d_epochs() -> usize { 50 }
fn d_lr() -> f64 { 1e-3 }
fn d_batch() -> usize { 256 }
fn d_channels() -> Vec<usize> { vec![32, 64] }
fn d_kernel() -> usize { 3 }
fn d_dense_hidden() -> usize { 64 }
fn d_dropout() -> f64 { 0.1 }

/// Count the leading PCA columns: the conv stack's signal width. The split is
/// re-derived from whichever manifest accompanies the features, so train and
/// predict always agree with the data on disk.
fn count_pca(columns: &[ColumnDesc]) -> usize {
    columns
        .iter()
        .take_while(|c| matches!(c.kind, ColumnKind::Pca { .. }))
        .count()
}

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

/// Export the trained CNN to `<output>/model.onnx`. The PCA/meta split is read
/// from the frozen contract's columns; the architecture is rebuilt from the
/// hyperparams snapshot before loading weights.
fn export(model_dir: PathBuf, output: PathBuf) -> Result<()> {
    let hp: Hyperparams = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("hyperparams.json"))
            .context("read hyperparams.json from model dir")?,
    )
    .context("parse hyperparams.json")?;

    // The PCA signal width comes from the frozen contract's column list.
    let contract: lensing_core::Contract = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("contract.json"))
            .context("read contract.json from model dir")?,
    )
    .context("parse contract.json")?;
    let n_cols = contract.n_cols;
    let n_pca = count_pca(&contract.columns);
    anyhow::ensure!(n_pca > 0, "contract has no pca columns; the CNN needs a signal");

    let scaler = scaler::Scaler::load(&model_dir.join("scaler.json"))
        .context("load scaler.json from model dir")?;
    anyhow::ensure!(
        scaler.mean.len() == n_cols,
        "scaler was fit on {} columns, contract has {n_cols}",
        scaler.mean.len()
    );

    let arch = arch_from(&hp, n_pca, n_cols);
    let cnn = model::load(&model_dir.join("model"), &arch)?;
    onnx_export::write_onnx(&cnn, &scaler, &output)?;
    emit(json!({"event":"log","msg":format!(
        "exported model.onnx: input [N, {n_cols}] ({n_pca} pca signal + {} meta)",
        n_cols - n_pca)}));
    emit(json!({"event":"done"}));
    Ok(())
}

fn arch_from(hp: &Hyperparams, n_pca: usize, n_cols: usize) -> model::Arch {
    model::Arch {
        n_pca,
        n_meta: n_cols - n_pca,
        channels: hp.channels.clone(),
        kernel_size: hp.kernel_size,
        dense_hidden: hp.dense_hidden,
        dropout: hp.dropout,
    }
}

fn predict(model_dir: PathBuf, input_dir: PathBuf, output: PathBuf) -> Result<()> {
    let hp: Hyperparams = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("hyperparams.json"))
            .context("read hyperparams.json from model dir")?,
    )
    .context("parse hyperparams.json")?;

    let input = InferenceInput::load(&input_dir)?;
    let n_cols = input.manifest.n_cols;
    let n_pca = count_pca(&input.manifest.columns);
    anyhow::ensure!(n_pca > 0, "input has no pca columns; the CNN needs a signal");
    emit(json!({"event":"log","msg":format!(
        "predicting {} rows, {} features ({} pca), cnn {:?} k{}",
        input.manifest.n_rows, n_cols, n_pca, hp.channels, hp.kernel_size)}));

    let scaler = scaler::Scaler::load(&model_dir.join("scaler.json"))
        .context("load scaler.json from model dir")?;
    anyhow::ensure!(
        scaler.mean.len() == n_cols,
        "scaler was fit on {} columns, input has {n_cols}",
        scaler.mean.len()
    );
    let x = scaler.transform(&input.features);

    let arch = arch_from(&hp, n_pca, n_cols);
    let cnn = model::load(&model_dir.join("model"), &arch)?;
    let predicted_t = model::predict_loaded(&cnn, &x, n_cols, hp.batch_size);

    let transform = input.manifest.target.transform;
    let predictions: Vec<InferencePrediction> = input
        .row_ids
        .iter()
        .zip(&predicted_t)
        .map(|(&row_id, &p)| InferencePrediction {
            row_id,
            predicted: transform.invert(p as f64).max(0.0),
            proba: None,
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
    anyhow::ensure!(!hp.channels.is_empty(), "channels must have at least one layer");
    anyhow::ensure!(hp.kernel_size >= 1, "kernel_size must be >= 1");
    std::fs::create_dir_all(&run_dir)?;

    let ds = Dataset::load(&dataset_dir)?;
    let n_cols = ds.manifest.n_cols;
    let n_pca = count_pca(&ds.manifest.columns);
    anyhow::ensure!(n_pca > 0, "dataset has no pca columns; the CNN needs a signal");
    emit(json!({"event":"log","msg":format!(
        "dataset {}: {} train / {} test rows, {} features ({} pca signal + {} meta)",
        ds.manifest.dataset_id, ds.train_idx.len(), ds.test_idx.len(),
        n_cols, n_pca, n_cols - n_pca)}));
    emit(json!({"event":"log","msg":format!(
        "cnn {:?} k{}, dense {}, dropout {}, lr {}, batch {}, {} epochs",
        hp.channels, hp.kernel_size, hp.dense_hidden, hp.dropout,
        hp.lr, hp.batch_size, hp.epochs)}));

    // Architecture diagram (registry capability `visualization`): written
    // before the first epoch so the live run view can show it immediately.
    viz::write_viz(
        &run_dir.join("viz.svg"),
        n_pca,
        n_cols - n_pca,
        &hp.channels,
        hp.kernel_size,
        hp.dense_hidden,
        hp.dropout,
    )?;

    let (train_x_raw, train_y) = ds.gather(&ds.train_idx);
    let (test_x_raw, test_y) = ds.gather(&ds.test_idx);

    // Standardize features with train-split statistics only.
    let scaler = scaler::Scaler::fit(&train_x_raw, n_cols);
    scaler.save(&run_dir.join("scaler.json"))?;
    let train_x = scaler.transform(&train_x_raw);
    let test_x = scaler.transform(&test_x_raw);

    let trained = model::run_training(
        model::TrainInputs {
            train_x: &train_x,
            train_y: &train_y,
            test_x: &test_x,
            test_y: &test_y,
            n_cols,
        },
        model::TrainConfig {
            epochs: hp.epochs,
            lr: hp.lr,
            batch_size: hp.batch_size,
            arch: arch_from(&hp, n_pca, n_cols),
            seed: 1337,
            checkpoint_every: hp.checkpoint_every,
        },
        &run_dir,
        &emit,
    )?;

    // Predictions in target space.
    let transform = ds.manifest.target.transform;
    let predicted_t = model::predict(&trained, &test_x, n_cols, hp.batch_size);
    let mut predictions = Vec::with_capacity(ds.test_idx.len());
    let mut pairs = Vec::with_capacity(ds.test_idx.len());
    for (j, &row) in ds.test_idx.iter().enumerate() {
        let actual = transform.invert(test_y[j] as f64);
        let predicted = transform.invert(predicted_t[j] as f64).max(0.0);
        pairs.push((actual, predicted));
        predictions.push(Prediction { row_id: ds.row_ids[row as usize], actual, predicted, proba: None });
    }
    let metrics = compute_metrics(&pairs);
    std::fs::write(run_dir.join("metrics.json"), serde_json::to_vec(&metrics)?)?;
    std::fs::write(run_dir.join("predictions.json"), serde_json::to_vec(&predictions)?)?;
    model::save_atomic(&trained, &run_dir)?;

    emit(json!({"event":"log","msg":format!(
        "MAE {:.0}  RMSE {:.0}  medAPE {:.1}%  R² {:.3}",
        metrics.mae.unwrap_or(f64::NAN), metrics.rmse.unwrap_or(f64::NAN),
        metrics.medape.unwrap_or(f64::NAN) * 100.0, metrics.r2.unwrap_or(f64::NAN))}));
    emit(json!({"event":"done"}));
    Ok(())
}
