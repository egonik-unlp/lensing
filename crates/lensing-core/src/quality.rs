use serde::{Deserialize, Serialize};

/// Data-quality filter configuration for a dataset build. Each built-in rule
/// can be toggled; enabled rules exclude flagged rows from the artifact.
/// The applied config is recorded in the manifest (provenance: a filtered
/// dataset is what its models trained on).
///
/// Field names are domain-neutral; the pre-generalization real-estate names
/// (`nonpositive_price`, `price_outlier`, `price_range`, `price_min/max`,
/// `bedrooms_outlier`, `bedrooms_max`) are kept as `#[serde(alias …)]` so a
/// config written against the old schema still parses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityFilterConfig {
    /// target ≤ 0 or missing. Cheap and safe; on by default.
    #[serde(default = "default_true", alias = "nonpositive_price")]
    pub nonpositive_target: bool,
    /// Robust target outliers: MAD z-score on log1p(target), grouped by the
    /// domain's outlier group (global fallback for thin groups).
    #[serde(default, alias = "price_outlier")]
    pub target_outlier: bool,
    /// |MAD z| above this flags a row for `target-outlier`.
    #[serde(default = "default_mad_z", alias = "price_outlier_mad_z")]
    pub target_outlier_mad_z: f64,
    /// Empty critical categoricals. Off by default: training tolerates them
    /// via the __other__ bucket.
    #[serde(default)]
    pub missing_fields: bool,
    /// Manual hard target caps; complements the statistical MAD rule with
    /// domain knowledge (an absurdly small or large value is noise, not an
    /// outlier). Bounds default to open — a domain sets real caps.
    #[serde(default, alias = "price_range")]
    pub target_range: bool,
    /// Targets below this flag a row for `target-range`. Defaults to open
    /// (no lower cap) — domain-neutral; a domain sets its own floor.
    #[serde(default = "default_target_min", alias = "price_min")]
    pub target_min: f64,
    /// Targets above this flag a row for `target-range`. Defaults to open
    /// (no upper cap).
    #[serde(default = "default_target_max", alias = "price_max")]
    pub target_max: f64,
    /// The domain's capped numeric field negative or above the cap:
    /// data-entry noise. Zero means "unspecified" and is never flagged.
    #[serde(default, alias = "bedrooms_outlier")]
    pub capped_numeric_outlier: bool,
    /// Values above this flag a row for `capped-numeric-outlier`. Defaults to
    /// open (no cap).
    #[serde(default = "default_target_max", alias = "bedrooms_max")]
    pub capped_numeric_max: f64,
    /// Exact duplicate document text (first occurrence kept). Duplicates
    /// double-count in the PCA fit and leak across the split.
    #[serde(default)]
    pub duplicate_content: bool,
    /// Document text shorter than the floor: thin embedding signal.
    #[serde(default)]
    pub short_content: bool,
    /// Character floor for `short-content`.
    #[serde(default = "default_min_chars")]
    pub short_content_min_chars: usize,
}

fn default_true() -> bool {
    true
}
fn default_mad_z() -> f64 {
    3.5
}
/// Open lower bound: nothing is below it, so an unconfigured `target-range`
/// rule flags nothing. Finite (not −∞) so it round-trips through JSON.
fn default_target_min() -> f64 {
    f64::MIN
}
/// Open upper bound: nothing is above it. Finite so it round-trips through JSON.
fn default_target_max() -> f64 {
    f64::MAX
}
fn default_min_chars() -> usize {
    80
}

impl Default for QualityFilterConfig {
    fn default() -> Self {
        Self {
            nonpositive_target: true,
            target_outlier: false,
            target_outlier_mad_z: default_mad_z(),
            missing_fields: false,
            target_range: false,
            target_min: default_target_min(),
            target_max: default_target_max(),
            capped_numeric_outlier: false,
            capped_numeric_max: default_target_max(),
            duplicate_content: false,
            short_content: false,
            short_content_min_chars: default_min_chars(),
        }
    }
}

impl QualityFilterConfig {
    pub fn rule_enabled(&self, rule: &str) -> bool {
        match rule {
            "nonpositive-target" => self.nonpositive_target,
            "target-outlier" => self.target_outlier,
            "missing-fields" => self.missing_fields,
            "target-range" => self.target_range,
            "capped-numeric-outlier" => self.capped_numeric_outlier,
            "duplicate-content" => self.duplicate_content,
            "short-content" => self.short_content,
            _ => false,
        }
    }
}

/// Per-rule outcome of a quality evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleStats {
    pub rule: String,
    /// Rows the rule matched (regardless of whether it was enabled).
    pub n_flagged: usize,
    /// Rows actually excluded by this rule (0 when the rule is disabled).
    pub n_excluded: usize,
}

/// Recorded in the manifest of a filtered dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    pub config: QualityFilterConfig,
    pub rules: Vec<RuleStats>,
    /// Distinct rows excluded across all enabled rules.
    pub n_excluded_total: usize,
}
