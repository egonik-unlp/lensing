//! Autoencoder compressor (burn 0.21, ndarray CPU) — the nonlinear counterpart
//! to PCA, hoisted out of the `predictor-burn-ae-clf` predictor where the
//! reusable encoder/decoder was welded to a softmax classifier head.
//!
//! Here it stands alone as a [`Compressor`]: an encoder + mirror decoder trained
//! unsupervised by reconstruction. Two things the scattered versions each
//! re-derived are first-class config: a **block-wise** reconstruction loss (so a
//! wide modality doesn't drown a narrow one — the SongAE pattern) and an optional
//! **sparse** penalty on the latent (the `model_sae.rs` variant). The trained
//! encoder is extractable (`into_encoder`) so a predictor can still transplant it.

use std::path::Path;

use anyhow::{ensure, Result};
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::Autodiff;
use burn::module::{AutodiffModule, Module};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{activation, Tensor, TensorData};

use lensing_core::shuffle::SplitMix64;

use crate::blocks::BlockLayout;
use crate::onnx::DenseLayer;
use crate::Compressor;

pub type Inner = NdArray<f32>;
pub type Auto = Autodiff<Inner>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    #[default]
    Relu,
    Gelu,
    Silu,
    Tanh,
}

impl Activation {
    fn apply<B: Backend>(self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            Activation::Relu => activation::relu(x),
            Activation::Gelu => activation::gelu(x),
            Activation::Silu => activation::silu(x),
            Activation::Tanh => activation::tanh(x),
        }
    }

    fn onnx(self) -> lensing_onnx::Activation {
        match self {
            Activation::Relu => lensing_onnx::Activation::Relu,
            Activation::Gelu => lensing_onnx::Activation::Gelu,
            Activation::Silu => lensing_onnx::Activation::Silu,
            Activation::Tanh => lensing_onnx::Activation::Tanh,
        }
    }
}

/// Optional sparsity penalty on the latent (the sparse-AE variant): adds
/// `weight · mean(|z|)` (L1) to the reconstruction loss, pushing the latent
/// toward fewer active units.
#[derive(Clone, Copy, Debug)]
pub struct SparseConfig {
    pub weight: f64,
}

/// Autoencoder configuration. `enc_hidden`'s last entry is the latent dim.
#[derive(Clone, Debug)]
pub struct AeConfig {
    /// Encoder layer widths; the last is the latent (bottleneck) dim.
    pub enc_hidden: Vec<usize>,
    pub activation: Activation,
    pub dropout: f64,
    pub epochs: usize,
    pub lr: f64,
    pub batch_size: usize,
    pub seed: u64,
    /// `Some` = sparse AE; `None` = dense AE.
    pub sparse: Option<SparseConfig>,
    /// Block layout for the balanced reconstruction loss (defaults to a single
    /// block = plain global MSE when unset via [`AeConfig::single_block`]).
    pub blocks: BlockLayout,
}

impl AeConfig {
    /// A reasonable dense-AE config with one reconstruction block over the whole
    /// vector (no multimodal weighting).
    pub fn dense(latent: usize, d: usize) -> Self {
        AeConfig {
            enc_hidden: vec![256, latent],
            activation: Activation::Relu,
            dropout: 0.0,
            epochs: 200,
            lr: 1e-3,
            batch_size: 256,
            seed: 42,
            sparse: None,
            blocks: BlockLayout::single(d),
        }
    }

    fn latent(&self) -> usize {
        *self.enc_hidden.last().expect("enc_hidden non-empty")
    }
}

// ----------------------------------------------------------------------------
// The burn module: encoder + mirror decoder.
// ----------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct AeNet<B: Backend> {
    encoder: Vec<Linear<B>>,
    decoder: Vec<Linear<B>>,
    dropout: Dropout,
    #[module(skip)]
    activation: Activation,
}

impl<B: Backend> AeNet<B> {
    pub fn new(cfg: &AeConfig, d: usize, device: &B::Device) -> Self {
        let mut encoder = Vec::with_capacity(cfg.enc_hidden.len());
        let mut prev = d;
        for &h in &cfg.enc_hidden {
            encoder.push(LinearConfig::new(prev, h).init(device));
            prev = h;
        }
        // Decoder mirrors the encoder: latent → reversed hidden (minus the
        // bottleneck) → d.
        let mut decoder = Vec::new();
        let mut prev = cfg.latent();
        let mut rev: Vec<usize> = cfg.enc_hidden.iter().rev().skip(1).copied().collect();
        rev.push(d);
        for &h in &rev {
            decoder.push(LinearConfig::new(prev, h).init(device));
            prev = h;
        }
        AeNet {
            encoder,
            decoder,
            dropout: DropoutConfig::new(cfg.dropout).init(),
            activation: cfg.activation,
        }
    }

    /// Encode with activation + dropout after every encoder layer (incl. latent).
    fn encode_t(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut h = x;
        for layer in &self.encoder {
            h = self.dropout.forward(self.activation.apply(layer.forward(h)));
        }
        h
    }

    /// Reconstruct `x`: encode then decode (linear output on the final layer).
    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let z = self.encode_t(x);
        let mut h = z.clone();
        let n = self.decoder.len();
        for (i, layer) in self.decoder.iter().enumerate() {
            h = layer.forward(h);
            if i + 1 < n {
                h = self.activation.apply(h);
            }
        }
        (h, z)
    }

    /// Consume the trained AE, yielding the encoder layers for transplant into a
    /// downstream predictor (preserves the `ae-clf` contract).
    pub fn into_encoder(self) -> Vec<Linear<B>> {
        self.encoder
    }
}

/// Balanced per-block reconstruction MSE: sum over blocks of the block's mean
/// squared error, so each block contributes on equal footing (REQ-5).
fn block_loss<B: Backend>(recon: Tensor<B, 2>, target: Tensor<B, 2>, layout: &BlockLayout) -> Tensor<B, 1> {
    let [n, _d] = recon.dims();
    let mut loss: Option<Tensor<B, 1>> = None;
    for b in &layout.blocks {
        let r = recon.clone().slice([0..n, b.start..b.end]);
        let t = target.clone().slice([0..n, b.start..b.end]);
        let diff = r - t;
        let block_mse = (diff.clone() * diff).mean();
        loss = Some(match loss {
            Some(acc) => acc + block_mse,
            None => block_mse,
        });
    }
    loss.expect("block layout non-empty")
}

fn slab<B: Backend>(x: &[f32], rows: &[usize], d: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut flat = Vec::with_capacity(rows.len() * d);
    for &r in rows {
        flat.extend_from_slice(&x[r * d..(r + 1) * d]);
    }
    Tensor::from_data(TensorData::new(flat, [rows.len(), d]), device)
}

// ----------------------------------------------------------------------------
// The compressor.
// ----------------------------------------------------------------------------

/// Autoencoder wrapped as a [`Compressor`]. `fit` trains the net; `encode`/
/// `reconstruct` run the trained (eval-mode) net on plain f32 matrices.
pub struct Autoencoder {
    cfg: AeConfig,
    d: usize,
    model: Option<AeNet<Inner>>,
    /// Per-block reconstruction R² measured on the fit data (the AE's quality
    /// metric, analogous to PCA's EVR).
    last_r2: Vec<(String, f32)>,
}

impl Autoencoder {
    pub fn new(cfg: AeConfig) -> Self {
        Autoencoder { cfg, d: 0, model: None, last_r2: Vec::new() }
    }

    /// Per-block reconstruction R² from the last `fit` (empty until fitted).
    pub fn block_r2(&self) -> &[(String, f32)] {
        &self.last_r2
    }

    fn fitted(&self) -> Result<&AeNet<Inner>> {
        self.model.as_ref().ok_or_else(|| anyhow::anyhow!("autoencoder not fitted"))
    }

    /// The trained encoder layers, for transplant into a predictor.
    pub fn into_encoder(self) -> Result<Vec<Linear<Inner>>> {
        Ok(self.model.ok_or_else(|| anyhow::anyhow!("autoencoder not fitted"))?.into_encoder())
    }

    fn run_forward(&self, x: &[f32], d: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let model = self.fitted()?;
        let device = NdArrayDevice::default();
        let n = x.len() / d;
        let rows: Vec<usize> = (0..n).collect();
        let xb = slab::<Inner>(x, &rows, d, &device);
        let (recon, z) = model.forward(xb);
        let recon = recon.into_data().to_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let z = z.into_data().to_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?;
        Ok((recon, z))
    }
}

impl Compressor for Autoencoder {
    fn fit(&mut self, x: &[f32], d: usize, rows: &[u32]) -> Result<()> {
        ensure!(!rows.is_empty(), "no rows to fit autoencoder");
        self.cfg.blocks.validate(d)?;
        self.d = d;
        let device = NdArrayDevice::default();
        Auto::seed(&device, self.cfg.seed);

        let mut net: AeNet<Auto> = AeNet::new(&self.cfg, d, &device);
        let mut optimizer = AdamConfig::new().init();

        let fit_rows: Vec<usize> = rows.iter().map(|&r| r as usize).collect();
        let n = fit_rows.len();
        let mut order = fit_rows.clone();
        let mut rng = SplitMix64(self.cfg.seed ^ 0x5151_5151_5151_5151);

        for _epoch in 0..self.cfg.epochs {
            for i in (1..n).rev() {
                let j = (rng.next_u64() % (i as u64 + 1)) as usize;
                order.swap(i, j);
            }
            for batch in order.chunks(self.cfg.batch_size.max(1)) {
                let xb = slab::<Auto>(x, batch, d, &device);
                let (recon, z) = net.forward(xb.clone());
                let mut loss = block_loss(recon, xb, &self.cfg.blocks);
                if let Some(sp) = self.cfg.sparse {
                    // L1 sparsity on the latent.
                    let l1 = z.abs().mean();
                    loss = loss + l1.mul_scalar(sp.weight);
                }
                let grads = GradientsParams::from_grads(loss.backward(), &net);
                net = optimizer.step(self.cfg.lr, net, grads);
            }
        }

        let trained = net.valid();
        // Measure per-block reconstruction R² on the fit rows.
        let fit_x: Vec<f32> = fit_rows.iter().flat_map(|&r| x[r * d..(r + 1) * d].to_vec()).collect();
        self.model = Some(trained);
        let (recon, _z) = self.run_forward(&fit_x, d)?;
        self.last_r2 = self.cfg.blocks.reconstruction_r2(&recon, &fit_x, d);
        Ok(())
    }

    fn encode(&self, x: &[f32], d: usize) -> Result<Vec<f32>> {
        Ok(self.run_forward(x, d)?.1)
    }

    fn reconstruct(&self, z: &[f32]) -> Result<Vec<f32>> {
        // Decode-only path: run the decoder on given latents.
        let model = self.fitted()?;
        let device = NdArrayDevice::default();
        let k = self.latent_dim();
        let n = z.len() / k;
        let rows: Vec<usize> = (0..n).collect();
        let zt = slab::<Inner>(z, &rows, k, &device);
        let mut h = zt;
        let nlayers = model.decoder.len();
        for (i, layer) in model.decoder.iter().enumerate() {
            h = layer.forward(h);
            if i + 1 < nlayers {
                h = self.cfg.activation.apply(h);
            }
        }
        h.into_data().to_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))
    }

    fn latent_dim(&self) -> usize {
        self.cfg.latent()
    }

    fn quality(&self) -> serde_json::Value {
        if self.last_r2.is_empty() {
            return serde_json::Value::Null;
        }
        serde_json::json!({
            "kind": "block_r2",
            "blocks": self.last_r2,
        })
    }

    fn save(&self, dir: &Path) -> Result<()> {
        use burn::record::CompactRecorder;
        let model = self.fitted()?;
        std::fs::create_dir_all(dir).ok();
        model
            .clone()
            .save_file(dir.join("autoencoder").to_string_lossy().as_ref(), &CompactRecorder::new())
            .map_err(|e| anyhow::anyhow!("save autoencoder: {e}"))?;
        Ok(())
    }

    fn export_encoder_onnx(&self, dir: &Path) -> Result<()> {
        let model = self.fitted()?;
        let layers: Vec<DenseLayer> = model
            .encoder
            .iter()
            .map(|l| {
                let [w_in, w_out] = l.weight.dims();
                DenseLayer {
                    weight: l.weight.val().into_data().to_vec::<f32>().unwrap(),
                    w_in,
                    w_out,
                    bias: l.bias.as_ref().map(|b| b.val().into_data().to_vec::<f32>().unwrap()),
                }
            })
            .collect();
        crate::onnx::export_ae_encoder(dir, &layers, self.cfg.activation.onnx(), self.d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
        // Data on a low-rank manifold so an AE can compress it: two latent
        // factors drive all d columns.
        let mut rng = SplitMix64(seed);
        let mut x = Vec::with_capacity(n * d);
        for _ in 0..n {
            let a = (rng.next_u64() % 1000) as f32 / 500.0 - 1.0;
            let b = (rng.next_u64() % 1000) as f32 / 500.0 - 1.0;
            for j in 0..d {
                let w = (j as f32 + 1.0) / d as f32;
                x.push(a * w + b * (1.0 - w));
            }
        }
        x
    }

    #[test]
    fn fit_encode_reconstruct_shapes() {
        let d = 8;
        let n = 128;
        let x = toy_data(n, d, 3);
        let rows: Vec<u32> = (0..n as u32).collect();
        let mut cfg = AeConfig::dense(2, d);
        cfg.epochs = 40;
        cfg.enc_hidden = vec![8, 2];
        let mut ae = Autoencoder::new(cfg);
        ae.fit(&x, d, &rows).unwrap();
        let z = ae.encode(&x, d).unwrap();
        assert_eq!(z.len(), n * 2);
        let recon = ae.reconstruct(&z).unwrap();
        assert_eq!(recon.len(), n * d);
        // Correctness = the metric is well-formed (one block, finite R²).
        // Convergence quality is exercised by the e2e, not this smoke test.
        let r2 = ae.block_r2();
        assert_eq!(r2.len(), 1);
        assert!(r2[0].1.is_finite(), "block R² = {}", r2[0].1);
    }

    #[test]
    fn sparse_variant_fits() {
        let d = 6;
        let n = 96;
        let x = toy_data(n, d, 5);
        let rows: Vec<u32> = (0..n as u32).collect();
        let mut cfg = AeConfig::dense(3, d);
        cfg.epochs = 20;
        cfg.enc_hidden = vec![6, 3];
        cfg.sparse = Some(SparseConfig { weight: 0.01 });
        let mut ae = Autoencoder::new(cfg);
        ae.fit(&x, d, &rows).unwrap();
        assert_eq!(ae.encode(&x, d).unwrap().len(), n * 3);
    }

    #[test]
    fn into_encoder_transplant() {
        let d = 4;
        let n = 40;
        let x = toy_data(n, d, 9);
        let rows: Vec<u32> = (0..n as u32).collect();
        let mut cfg = AeConfig::dense(2, d);
        cfg.epochs = 5;
        cfg.enc_hidden = vec![4, 2];
        let mut ae = Autoencoder::new(cfg);
        ae.fit(&x, d, &rows).unwrap();
        let encoder = ae.into_encoder().unwrap();
        assert_eq!(encoder.len(), 2); // matches enc_hidden layers
    }
}
