//! ONNX export of a compressor's *encoder*, so a latent encoder is portable
//! outside the framework (Node, browser, Worker) — the same road the predictor
//! crates and `model-export` already take via `lensing-onnx`.
//!
//! Two shapes cover both compressors:
//! - PCA: mean-center then a single `Gemm` (`z = (x - mean)·componentsᵀ`).
//! - autoencoder: a stack of `Gemm` + activation, one per encoder layer.

use std::path::Path;

use anyhow::{Context, Result};
use lensing_onnx::{Activation, Dim, GraphBuilder};

/// One dense encoder layer, row-major `[w_in, w_out]` (burn's `Linear` layout →
/// `Gemm` with no transpose), matching `ExportLayer` in the ae-clf predictor.
pub struct DenseLayer {
    pub weight: Vec<f32>,
    pub w_in: usize,
    pub w_out: usize,
    pub bias: Option<Vec<f32>>,
}

fn write(dir: &Path, bytes: Vec<u8>) -> Result<()> {
    std::fs::create_dir_all(dir).ok();
    let path = dir.join("encoder.onnx");
    std::fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Export a PCA encoder: `z = (x - mean) · componentsᵀ`.
/// `components_t` is row-major `[d, k]` (the transpose of the k×d basis).
pub fn export_pca_encoder(
    dir: &Path,
    mean: &[f32],
    components_t: Vec<f32>,
    d: usize,
    k: usize,
) -> Result<()> {
    let mut g = GraphBuilder::new();
    g.add_weight("pca_mean", vec![d as i64], mean.to_vec());
    let centered = g.op("Sub", &["input", "pca_mean"], vec![]);
    let out = g.gemm(&centered, "pca_components_t", components_t, d, k, "", None);
    let bytes = g.build(
        "input",
        &[Dim::Param("batch".into()), Dim::Value(d as i64)],
        &out,
        &[Dim::Param("batch".into()), Dim::Value(k as i64)],
        "lensing-compression/pca",
    );
    write(dir, bytes)
}

/// Export an autoencoder encoder: a chain of `Gemm` + `activation`, one per
/// layer (activation after every encoder layer incl. the latent, matching the
/// burn encoder's `encode`).
pub fn export_ae_encoder(
    dir: &Path,
    layers: &[DenseLayer],
    activation: Activation,
    d: usize,
) -> Result<()> {
    let mut g = GraphBuilder::new();
    let mut cur = "input".to_string();
    let mut latent = d;
    for (i, l) in layers.iter().enumerate() {
        let w = g.gemm(&cur, &format!("enc_w{i}"), l.weight.clone(), l.w_in, l.w_out, &format!("enc_b{i}"), l.bias.clone());
        cur = g.activation(&w, activation);
        latent = l.w_out;
    }
    let bytes = g.build(
        "input",
        &[Dim::Param("batch".into()), Dim::Value(d as i64)],
        &cur,
        &[Dim::Param("batch".into()), Dim::Value(latent as i64)],
        "lensing-compression/autoencoder",
    );
    write(dir, bytes)
}
