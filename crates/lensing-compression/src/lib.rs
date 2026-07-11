//! `lensing-compression` — information compression / latent representations.
//!
//! One roof over the compression techniques that used to be scattered across the
//! ecosystem: PCA (hoisted out of `lensing-pipeline`), autoencoders (hoisted out
//! of the `predictor-burn-ae-clf` predictor, incl. a sparse variant), block-wise
//! multimodal reconstruction (the SongAE pattern), latent generation over a
//! vector store, and ONNX encoder export.
//!
//! The unifying contract is [`Compressor`]: fit → encode → reconstruct, plus
//! persistence and encoder export. A compressor produces *representation*, never
//! *prediction* — it is deliberately unsupervised (no target ever enters `fit`).

use std::path::Path;

use anyhow::Result;

pub mod autoencoder;
pub mod blocks;
pub mod onnx;
pub mod pca;
pub mod preprocess;
pub mod store;

pub use autoencoder::{AeConfig, Autoencoder, SparseConfig};
pub use blocks::{Block, BlockLayout};
pub use pca::{Pca, PcaModel, Spectrum};
pub use preprocess::PreprocessSpec;
pub use store::{build_latent_collection, Method, PipelineConfig};

/// The shared compression contract. Implemented by [`Pca`] and the autoencoder.
///
/// `fit` is unsupervised: it only ever sees the feature matrix, never a label or
/// target (REQ-1). `encode` maps data → latent; `reconstruct` maps latent → an
/// approximation of the original space.
pub trait Compressor {
    /// Fit the compressor on the row-major feature matrix `x` (width `d`),
    /// using only the selected `rows` (e.g. a train split).
    fn fit(&mut self, x: &[f32], d: usize, rows: &[u32]) -> Result<()>;

    /// Encode data (row-major, width `d`) to the latent space (width
    /// [`Compressor::latent_dim`]).
    fn encode(&self, x: &[f32], d: usize) -> Result<Vec<f32>>;

    /// Reconstruct the original space from latents (row-major, width
    /// [`Compressor::latent_dim`]).
    fn reconstruct(&self, z: &[f32]) -> Result<Vec<f32>>;

    /// Latent (bottleneck) dimension.
    fn latent_dim(&self) -> usize;

    /// A JSON quality metric describing how well the latent preserves the data:
    /// PCA reports its EVR curve, the autoencoder its per-block reconstruction
    /// R². `Null` until fitted. Consumed by the server/UI detail view.
    fn quality(&self) -> serde_json::Value {
        serde_json::Value::Null
    }

    /// Persist the fitted model + preprocessing spec under `dir`.
    fn save(&self, dir: &Path) -> Result<()>;

    /// Export the encoder to ONNX under `dir` for portable inference.
    fn export_encoder_onnx(&self, dir: &Path) -> Result<()>;
}
