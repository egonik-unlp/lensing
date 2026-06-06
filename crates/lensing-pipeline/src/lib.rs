//! Dataset pipeline: Qdrant fetch → feature encoding → PCA → artifact.

pub mod build;
pub mod currency;
pub mod features;
pub mod inference;
pub mod numerics;
pub mod openai;
pub mod pca;
pub mod qdrant;
pub mod quality;
pub mod redundancy;

pub use lensing_core::shuffle;

pub use build::{build_dataset, BuildConfig};
pub use inference::{Featurizer, RawItem};
