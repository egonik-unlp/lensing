//! Dataset pipeline: Qdrant fetch → feature encoding → PCA → artifact.

pub mod build;
pub mod currency;
pub mod features;
pub mod inference;
pub mod numerics;
pub mod openai;
// PCA now lives in `lensing-compression` (one Compressor among PCA + autoencoder).
// Re-exported so every consumer (`build`, `inference`, the ONNX bundle) and the
// `pca_components.f32` artifact format stay unchanged.
pub use lensing_compression::pca;
pub mod qdrant;
pub mod quality;
pub mod redundancy;

pub use lensing_core::shuffle;

pub use build::{build_dataset, BuildConfig};
pub use inference::{Featurizer, RawItem};
