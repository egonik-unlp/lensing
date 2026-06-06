//! Model definition and hand-written training loop (burn 0.21, ndarray CPU).
//! Loss values are MSE in transformed (log) target space.
//!
//! The feature vector is split at `n_pca`: the leading PCA components are
//! treated as a 1-channel 1D signal for the conv stack; the metadata tail
//! (numeric + one-hot columns) bypasses convolution and is concatenated into
//! the dense head after global pooling.

use std::path::Path;

use anyhow::Result;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::Autodiff;
use burn::module::{AutodiffModule, Module};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig, MaxPool1d, MaxPool1dConfig};
use burn::nn::{
    Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig1d, Relu,
};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::CompactRecorder;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use lensing_core::shuffle::SplitMix64;

pub type Inner = NdArray<f32>;
pub type Auto = Autodiff<Inner>;

#[derive(Module, Debug)]
pub struct Cnn<B: Backend> {
    convs: Vec<Conv1d<B>>,
    pool: MaxPool1d,
    global_pool: AdaptiveAvgPool1d,
    dense: Linear<B>,
    output: Linear<B>,
    relu: Relu,
    dropout: Dropout,
    /// Split point: leading PCA columns go through the conv stack.
    n_pca: usize,
}

/// Architecture knobs, shared by `Cnn::new` and `load`.
#[derive(Clone)]
pub struct Arch {
    pub n_pca: usize,
    pub n_meta: usize,
    pub channels: Vec<usize>,
    pub kernel_size: usize,
    pub dense_hidden: usize,
    pub dropout: f64,
}

impl<B: Backend> Cnn<B> {
    pub fn new(arch: &Arch, device: &B::Device) -> Self {
        let k = arch.kernel_size;
        let mut convs = Vec::with_capacity(arch.channels.len());
        let mut prev_ch = 1;
        for &ch in &arch.channels {
            // Symmetric k/2 padding (== PyTorch padding=k//2, Flux pad=k÷2),
            // so all three CNN implementations see identical lengths.
            convs.push(
                Conv1dConfig::new(prev_ch, ch, k)
                    .with_padding(PaddingConfig1d::Explicit(k / 2, k / 2))
                    .init(device),
            );
            prev_ch = ch;
        }
        Cnn {
            convs,
            pool: MaxPool1dConfig::new(2).init(), // stride defaults to kernel_size
            global_pool: AdaptiveAvgPool1dConfig::new(1).init(),
            dense: LinearConfig::new(prev_ch + arch.n_meta, arch.dense_hidden).init(device),
            output: LinearConfig::new(arch.dense_hidden, 1).init(device),
            relu: Relu::new(),
            dropout: DropoutConfig::new(arch.dropout).init(),
            n_pca: arch.n_pca,
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch, n_cols] = x.dims();
        let pca = x.clone().slice([0..batch, 0..self.n_pca]);
        let meta = x.slice([0..batch, self.n_pca..n_cols]);

        // burn Conv1d is NCL: [batch, channels, length].
        let mut signal = pca.reshape([batch, 1, self.n_pca]);
        for conv in &self.convs {
            signal = self.relu.forward(conv.forward(signal));
            // Halve the length, but never pool a length-1 signal away.
            if signal.dims()[2] >= 2 {
                signal = self.pool.forward(signal);
            }
            signal = self.dropout.forward(signal);
        }

        let [_, ch, _] = signal.dims();
        let pooled = self.global_pool.forward(signal).reshape([batch, ch]);
        let h = Tensor::cat(vec![pooled, meta], 1);
        let h = self.dropout.forward(self.relu.forward(self.dense.forward(h)));
        self.output.forward(h)
    }
}

pub struct TrainInputs<'a> {
    pub train_x: &'a [f32],
    pub train_y: &'a [f32],
    pub test_x: &'a [f32],
    pub test_y: &'a [f32],
    pub n_cols: usize,
}

pub struct TrainConfig {
    pub epochs: usize,
    pub lr: f64,
    pub batch_size: usize,
    pub arch: Arch,
    pub seed: u64,
    /// Save a loadable checkpoint every N epochs (0 = only at the end), so a
    /// killed run can still be promoted from its last saved params.
    pub checkpoint_every: usize,
}

fn batch_tensor<B: Backend>(
    x: &[f32],
    rows: &[usize],
    n_cols: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut flat = Vec::with_capacity(rows.len() * n_cols);
    for &r in rows {
        flat.extend_from_slice(&x[r * n_cols..(r + 1) * n_cols]);
    }
    Tensor::from_data(TensorData::new(flat, [rows.len(), n_cols]), device)
}

fn target_tensor<B: Backend>(y: &[f32], rows: &[usize], device: &B::Device) -> Tensor<B, 2> {
    let flat: Vec<f32> = rows.iter().map(|&r| y[r]).collect();
    Tensor::from_data(TensorData::new(flat, [rows.len(), 1]), device)
}

pub fn run_training(
    inputs: TrainInputs<'_>,
    cfg: TrainConfig,
    run_dir: &Path,
    emit: &dyn Fn(serde_json::Value),
) -> Result<Cnn<Auto>> {
    let device = NdArrayDevice::default();
    Auto::seed(&device, cfg.seed);

    let n_train = inputs.train_y.len();
    let mut model: Cnn<Auto> = Cnn::new(&cfg.arch, &device);
    let mut optimizer = AdamConfig::new().init();
    let loss_fn = MseLoss::new();

    let mut order: Vec<usize> = (0..n_train).collect();
    let mut rng = SplitMix64(cfg.seed ^ 0xA5A5_A5A5);

    for epoch in 1..=cfg.epochs {
        // Graceful stop (contract v2): the server drops a STOP file into the
        // run dir; finish early and let the caller run the normal
        // end-of-training path (eval + metrics + final checkpoint).
        if run_dir.join("STOP").exists() {
            emit(serde_json::json!({"event":"stopping"}));
            emit(serde_json::json!({"event":"log","msg":format!(
                "stop requested; evaluating with params as of epoch {}", epoch - 1)}));
            break;
        }

        // Fisher-Yates reshuffle each epoch.
        for i in (1..n_train).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }

        let mut loss_sum = 0.0f64;
        for batch in order.chunks(cfg.batch_size) {
            let x = batch_tensor::<Auto>(inputs.train_x, batch, inputs.n_cols, &device);
            let y = target_tensor::<Auto>(inputs.train_y, batch, &device);
            let pred = model.forward(x);
            let loss = loss_fn.forward(pred, y, Reduction::Mean);
            let loss_value = loss.clone().into_data().to_vec::<f32>().unwrap()[0] as f64;
            loss_sum += loss_value * batch.len() as f64;

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(cfg.lr, model, grads);
        }
        let train_loss = loss_sum / n_train as f64;

        // Validation on the inner backend: no autodiff, dropout inactive.
        let val_model = model.valid();
        let val_pred = predict_inner(&val_model, inputs.test_x, inputs.n_cols, cfg.batch_size);
        let val_loss = val_pred
            .iter()
            .zip(inputs.test_y)
            .map(|(p, a)| (*p as f64 - *a as f64).powi(2))
            .sum::<f64>()
            / inputs.test_y.len() as f64;

        emit(serde_json::json!({
            "event": "epoch",
            "epoch": epoch,
            "total_epochs": cfg.epochs,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }));

        // Periodic checkpoint: after this event the run is promotable even
        // if the process is later killed. The final epoch is skipped — the
        // caller saves right after training anyway.
        if cfg.checkpoint_every > 0 && epoch % cfg.checkpoint_every == 0 && epoch < cfg.epochs {
            save_atomic(&model, run_dir)?;
            emit(serde_json::json!({"event":"checkpoint","epoch":epoch}));
        }
    }

    Ok(model)
}

fn predict_inner(model: &Cnn<Inner>, x: &[f32], n_cols: usize, batch_size: usize) -> Vec<f32> {
    let device = NdArrayDevice::default();
    let n = x.len() / n_cols;
    let rows: Vec<usize> = (0..n).collect();
    let mut out = Vec::with_capacity(n);
    for batch in rows.chunks(batch_size.max(1)) {
        let t = batch_tensor::<Inner>(x, batch, n_cols, &device);
        let pred = model.forward(t);
        out.extend(pred.into_data().to_vec::<f32>().unwrap());
    }
    out
}

/// Predict transformed-space targets for all rows of `x` (eval mode).
pub fn predict(model: &Cnn<Auto>, x: &[f32], n_cols: usize, batch_size: usize) -> Vec<f32> {
    predict_inner(&model.valid(), x, n_cols, batch_size)
}

/// Atomically write the checkpoint as `<dir>/model.mpk`: record to a temp
/// stem first, then rename — a kill mid-write can never corrupt the loadable
/// file. Saves the eval-mode (inner) module; records are backend-agnostic,
/// so `load` reads it unchanged.
///
/// The temp stem is dot-free ("model-tmp") because the recorder
/// `set_extension`s the path: "model.tmp" would collapse onto "model.mpk".
pub fn save_atomic(model: &Cnn<Auto>, dir: &Path) -> Result<()> {
    let tmp = dir.join("model-tmp");
    model
        .valid()
        .save_file(tmp.to_string_lossy().as_ref(), &CompactRecorder::new())
        .map_err(|e| anyhow::anyhow!("save checkpoint: {e}"))?;
    std::fs::rename(dir.join("model-tmp.mpk"), dir.join("model.mpk"))?;
    Ok(())
}

/// Load a checkpoint for inference (inner backend, no autodiff). The module
/// must be rebuilt with the trained architecture BEFORE loading weights; the
/// arch comes from the model dir's hyperparams snapshot + input manifest.
pub fn load(path: &Path, arch: &Arch) -> Result<Cnn<Inner>> {
    let device = NdArrayDevice::default();
    Cnn::<Inner>::new(arch, &device)
        .load_file(path.to_string_lossy().as_ref(), &CompactRecorder::new(), &device)
        .map_err(|e| anyhow::anyhow!("load checkpoint {}: {e}", path.display()))
}

/// Predict transformed-space targets with a loaded inference model.
pub fn predict_loaded(model: &Cnn<Inner>, x: &[f32], n_cols: usize, batch_size: usize) -> Vec<f32> {
    predict_inner(model, x, n_cols, batch_size)
}
