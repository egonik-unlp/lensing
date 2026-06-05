//! Hyperparameter schema and the `blend.json` model record.

use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct Hyperparams {
    pub members: Vec<MemberSpec>,
    #[serde(default)]
    pub rule: Rule,
    #[serde(default)]
    pub weight_fit: WeightFit,
    #[serde(default = "d_val_fraction")]
    pub val_fraction: f64,
}

fn d_val_fraction() -> f64 {
    0.15
}

/// One member of the blend. Exactly one of `model` (frozen promoted model),
/// `definition` (models.toml recipe, trained together) or `predictor`+
/// `hyperparams` (inline recipe, trained together) must be set.
#[derive(Debug, Clone, Deserialize)]
pub struct MemberSpec {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub definition: Option<String>,
    #[serde(default)]
    pub predictor: Option<String>,
    #[serde(default)]
    pub hyperparams: Option<serde_json::Value>,
    #[serde(default)]
    pub weight: Option<f64>,
    /// Feature blocks this member must not see (trained members only):
    /// "pca", "coordinates", "raw_numerics", "bedrooms", or a one-hot group
    /// name ("propertyType", "neighborhood", …).
    #[serde(default)]
    pub exclude_blocks: Vec<String>,
}

impl MemberSpec {
    pub fn validate(&self, index: usize) -> Result<()> {
        let sources =
            [self.model.is_some(), self.definition.is_some(), self.predictor.is_some()];
        ensure!(
            sources.iter().filter(|s| **s).count() == 1,
            "member {index}: exactly one of model / definition / predictor must be set"
        );
        if self.hyperparams.is_some() && self.predictor.is_none() {
            bail!("member {index}: hyperparams only apply to an inline predictor member");
        }
        if self.model.is_some() && !self.exclude_blocks.is_empty() {
            bail!(
                "member {index}: exclude_blocks cannot apply to a frozen model \
                 (its columns are fixed by its contract)"
            );
        }
        if let Some(w) = self.weight {
            ensure!(w.is_finite() && w >= 0.0, "member {index}: weight must be ≥ 0");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Rule {
    #[default]
    Mean,
    /// Per-row median in price space — the voting committee. Weights are
    /// rejected: every voter counts once.
    Median,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WeightFit {
    #[default]
    None,
    /// Grid-search the mean weights on a validation split carved out of the
    /// train split (trained members then train on train-minus-val).
    Grid,
}

/// `blend.json` — everything predict needs to reassemble the blend from the
/// promoted model dir.
#[derive(Debug, Serialize, Deserialize)]
pub struct BlendFile {
    pub contract_version: u32,
    pub rule: Rule,
    pub weight_fit: WeightFit,
    /// Final (post-grid, renormalized) weights, parallel to `members`.
    pub weights: Vec<f64>,
    pub members: Vec<BlendMember>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BlendMember {
    pub index: usize,
    /// "model" | "definition" | "inline".
    pub kind: String,
    /// Predictor name in registry.toml.
    pub predictor: String,
    /// Model or definition name; absent for inline members.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Ordered column names this member consumes (indices are resolved
    /// against whatever manifest accompanies the features at hand).
    pub columns: Vec<String>,
    pub n_cols: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exclude_blocks: Vec<String>,
    pub frozen: bool,
    /// False when a graceful stop landed before this member trained; the
    /// blend then combines over the included members only.
    pub included: bool,
    /// Solo test metrics in price space (absent if excluded).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solo_metrics: Option<pg_core::Metrics>,
}
