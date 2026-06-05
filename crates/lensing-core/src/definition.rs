use serde::{Deserialize, Serialize};

/// One entry of `models.toml` at the repository root: a named, git-versionable
/// preset of predictor + concrete hyperparams. Distinct from a promoted model
/// (`data/models/<name>/`, a frozen weights snapshot); the two namespaces are
/// independent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDefinition {
    pub name: String,
    /// Predictor name in `registry.toml`.
    pub predictor: String,
    /// Hyperparams already merged over the predictor's schema defaults, so a
    /// definition is always launch-ready.
    pub hyperparams: serde_json::Value,
    /// Datasets this definition has been used with (auto-appended when a run
    /// is launched from it) or is intended for (tagged by hand).
    #[serde(default)]
    pub dataset_tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    /// RFC 3339, UTC.
    pub created_at: String,
    /// RFC 3339, UTC; bumped on every mutation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
}
