use serde::{Deserialize, Serialize};

use crate::manifest::{ColumnDesc, FeatureConfig, PcaInfo, TargetInfo};

/// `model.json` in a model directory (`data/models/<name>/`). The record of a
/// promoted run. Owned by lensing-server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecord {
    pub name: String,
    pub run_id: String,
    pub predictor: String,
    pub dataset_id: String,
    /// RFC 3339, UTC.
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

/// `contract.json` in a model directory: the featurization contract frozen at
/// training time. Everything the server needs to featurize raw inputs into
/// the trained column order (plus `pca_components.f32` as a binary sibling),
/// and everything a predictor needs to interpret them (`n_cols`, `target`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contract {
    pub contract_version: u32,
    pub n_cols: usize,
    /// One descriptor per feature column, in on-disk order. Copied verbatim
    /// from the training dataset's manifest.
    pub columns: Vec<ColumnDesc>,
    pub pca: PcaInfo,
    pub target: TargetInfo,
    pub feature_config: FeatureConfig,
    /// What callers must send for inference. Derived from `columns`.
    pub input_fields: InputFields,
    /// Companion collection the raw-numerics fields were reconciled from at
    /// build time; the predict path repeats the join for point_ids inputs.
    /// Absent on contracts frozen before raw numerics existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numerics_collection: Option<String>,
    /// Frozen train-split medians for missing-numeric imputation; the predict
    /// path repeats the fill so features cannot drift from trained ones.
    /// Present only on contracts trained with `impute_numerics`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imputation: Option<crate::numerics::NumericImputation>,
}

/// Caller-facing summary of what an inference input item must contain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputFields {
    /// Numeric payload fields the model consumes (e.g. "bedrooms").
    pub required_numeric: Vec<String>,
    /// Categorical payload fields the model consumes (e.g. "propertyType").
    /// Unknown values fall into the trained `__other__` bucket.
    pub required_categorical: Vec<String>,
    /// Expected embedding dimension (the PCA input width).
    pub embedding_dim: usize,
}

/// `manifest.json` of an inference input mini-artifact: the trimmed manifest
/// the server writes next to `features.f32` for a predict invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputManifest {
    pub n_rows: usize,
    pub n_cols: usize,
    pub columns: Vec<ColumnDesc>,
    pub target: TargetInfo,
}
