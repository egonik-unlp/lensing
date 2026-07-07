//! Sparse autoencoder (dictionary learning) on a standardized feature block.
//! burn 0.21 / ndarray CPU, hand-written training loop.
//!
//! Overcomplete ReLU code with an L1 sparsity penalty: the net reconstructs the
//! input while keeping few atoms active, so each learned dictionary atom is a
//! candidate (sparse, rare) feature direction — the regime where variance-PCA
//! buries structure. This is the engine for P2 of the interpretability program.

use std::path::Path;

use anyhow::Result;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::Autodiff;
use burn::module::{AutodiffModule, Module};
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::CompactRecorder;
use burn::tensor::backend::Backend;
use burn::tensor::{activation, Tensor, TensorData};

use lensing_core::shuffle::SplitMix64;

pub type Inner = NdArray<f32>;
pub type Auto = Autodiff<Inner>;

#[derive(Module, Debug)]
pub struct Sae<B: Backend> {
    encoder: Linear<B>,
    decoder: Linear<B>,
}

impl<B: Backend> Sae<B> {
    pub fn new(d: usize, m: usize, device: &B::Device) -> Self {
        Sae {
            encoder: LinearConfig::new(d, m).init(device),
            decoder: LinearConfig::new(m, d).init(device),
        }
    }
    pub fn encode(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        activation::relu(self.encoder.forward(x))
    }
    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let code = self.encode(x);
        let recon = self.decoder.forward(code.clone());
        (code, recon)
    }
    /// Number of dictionary atoms (encoder output width).
    pub fn n_atoms(&self) -> usize {
        let [_d, m] = self.encoder.weight.dims();
        m
    }
}

pub struct SaeConfig {
    pub n_atoms: usize,
    pub l1: f64,
    pub epochs: usize,
    pub lr: f64,
    pub batch_size: usize,
    pub seed: u64,
}

pub struct SaeOutput {
    /// Row-major `[n_rows, n_atoms]` code for ALL rows.
    pub code_all: Vec<f32>,
    /// Fraction of input variance reconstructed (all rows).
    pub var_explained: f64,
    /// Mean number of active atoms (code > eps) per row.
    pub l0_mean: f64,
}

fn batch_tensor<B: Backend>(x: &[f32], rows: &[usize], d: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut flat = Vec::with_capacity(rows.len() * d);
    for &r in rows {
        flat.extend_from_slice(&x[r * d..(r + 1) * d]);
    }
    Tensor::from_data(TensorData::new(flat, [rows.len(), d]), device)
}

/// Train the SAE on `train_x` (standardized, `[n_train, d]`). Returns the trained
/// model on the eval (inner) backend, ready to save or encode with, plus the
/// final mean epoch loss. Deterministic for a fixed `(train_x, cfg)`.
pub fn train(
    train_x: &[f32],
    d: usize,
    cfg: &SaeConfig,
    emit: &dyn Fn(serde_json::Value),
) -> (Sae<Inner>, f64) {
    let device = NdArrayDevice::default();
    Auto::seed(&device, cfg.seed);
    let m = cfg.n_atoms;
    let n_train = train_x.len() / d;
    let mut sae = Sae::<Auto>::new(d, m, &device);
    let mut opt = AdamConfig::new().init();
    let mse = MseLoss::new();
    let mut order: Vec<usize> = (0..n_train).collect();
    let mut rng = SplitMix64(cfg.seed ^ 0x05AE_05AE);
    let mut final_loss = 0.0;

    for epoch in 1..=cfg.epochs {
        for i in (1..n_train).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        let mut loss_sum = 0.0f64;
        let mut nb = 0usize;
        for batch in order.chunks(cfg.batch_size.max(1)) {
            let x = batch_tensor::<Auto>(train_x, batch, d, &device);
            let (code, recon) = sae.forward(x.clone());
            let rec = mse.forward(recon, x, Reduction::Mean);
            // Canonical SAE sparsity: mean over the batch of the per-row sum of
            // atom activations (so λ is interpretable as a per-row L1 budget).
            let l1 = code.abs().sum_dim(1).mean();
            let loss = rec.add(l1.mul_scalar(cfg.l1 as f32));
            loss_sum += loss.clone().into_data().to_vec::<f32>().unwrap()[0] as f64;
            nb += 1;
            let grads = GradientsParams::from_grads(loss.backward(), &sae);
            sae = opt.step(cfg.lr, sae, grads);
        }
        final_loss = loss_sum / nb.max(1) as f64;
        if epoch == 1 || epoch % 10 == 0 || epoch == cfg.epochs {
            emit(serde_json::json!({"event":"log","msg":format!(
                "sae epoch {epoch}/{} loss {:.4}", cfg.epochs, final_loss)}));
        }
    }
    (sae.valid(), final_loss)
}

/// Encode ALL rows of `all_x` (`[n_rows, d]`) with a trained/loaded model and
/// report reconstruction quality + sparsity.
pub fn encode_all(model: &Sae<Inner>, all_x: &[f32], d: usize, batch_size: usize) -> SaeOutput {
    let device = NdArrayDevice::default();
    let m = model.n_atoms();
    let n_rows = all_x.len() / d;
    let mut code_all = Vec::with_capacity(n_rows * m);
    let rows: Vec<usize> = (0..n_rows).collect();
    let mut active = 0.0f64;
    let mut sse = 0.0f64;
    let mut sst = 0.0f64;
    for batch in rows.chunks(batch_size.max(1)) {
        let x = batch_tensor::<Inner>(all_x, batch, d, &device);
        let (code, recon) = model.forward(x.clone());
        let cv = code.into_data().to_vec::<f32>().unwrap();
        for &v in &cv {
            if v > 1e-6 {
                active += 1.0;
            }
        }
        code_all.extend(cv);
        let xv = x.into_data().to_vec::<f32>().unwrap();
        let rv = recon.into_data().to_vec::<f32>().unwrap();
        for k in 0..xv.len() {
            let e = (xv[k] - rv[k]) as f64;
            sse += e * e;
            sst += (xv[k] as f64) * (xv[k] as f64);
        }
    }
    SaeOutput { code_all, var_explained: 1.0 - sse / sst.max(1e-9), l0_mean: active / n_rows as f64 }
}

/// Persist the trained model to `<dir>/model.mpk` (atomic temp+rename, like the
/// MLP predictor) so a kill mid-write can't corrupt a cached checkpoint.
pub fn save(model: &Sae<Inner>, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join("model-tmp");
    model
        .clone()
        .save_file(tmp.to_string_lossy().as_ref(), &CompactRecorder::new())
        .map_err(|e| anyhow::anyhow!("save sae: {e}"))?;
    std::fs::rename(dir.join("model-tmp.mpk"), dir.join("model.mpk"))?;
    Ok(())
}

/// Load a cached model (`<dir>/model.mpk`); the architecture `(d, m)` must match
/// what it was trained with (the cache key encodes both, so it always does).
pub fn load(dir: &Path, d: usize, m: usize) -> Result<Sae<Inner>> {
    let device = NdArrayDevice::default();
    Sae::<Inner>::new(d, m, &device)
        .load_file(dir.join("model").to_string_lossy().as_ref(), &CompactRecorder::new(), &device)
        .map_err(|e| anyhow::anyhow!("load sae {}: {e}", dir.display()))
}
