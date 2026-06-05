use serde::{Deserialize, Serialize};

/// Data-quality filter configuration for a dataset build. Each built-in rule
/// can be toggled; enabled rules exclude flagged rows from the artifact.
/// The applied config is recorded in the manifest (provenance: a filtered
/// dataset is what its models trained on).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityFilterConfig {
    /// price ≤ 0 or missing target. Cheap and safe; on by default.
    #[serde(default = "default_true")]
    pub nonpositive_price: bool,
    /// Robust price outliers: MAD z-score on log1p(price), grouped per
    /// propertyType (global fallback for thin groups).
    #[serde(default)]
    pub price_outlier: bool,
    /// |MAD z| above this flags a row for `price_outlier`.
    #[serde(default = "default_mad_z")]
    pub price_outlier_mad_z: f64,
    /// Empty critical categoricals (propertyType, neighborhood). Off by
    /// default: training tolerates them via the __other__ bucket.
    #[serde(default)]
    pub missing_fields: bool,
    /// Manual hard price caps; complements the statistical MAD rule with
    /// domain knowledge (a $30 or $900M "sale" is noise, not an outlier).
    #[serde(default)]
    pub price_range: bool,
    /// Prices below this flag a row for `price_range`.
    #[serde(default = "default_price_min")]
    pub price_min: f64,
    /// Prices above this flag a row for `price_range`.
    #[serde(default = "default_price_max")]
    pub price_max: f64,
    /// Bedrooms negative or above the cap: data-entry noise on a real
    /// feature. Zero is "unspecified" in this corpus and is never flagged.
    #[serde(default)]
    pub bedrooms_outlier: bool,
    /// Bedrooms above this flag a row for `bedrooms-outlier`.
    #[serde(default = "default_bedrooms_max")]
    pub bedrooms_max: f64,
    /// Exact duplicate listing text (first occurrence kept). Relisted
    /// properties double-count in the PCA fit and leak across the split.
    #[serde(default)]
    pub duplicate_content: bool,
    /// Listing text shorter than the floor: thin embedding signal.
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
fn default_price_min() -> f64 {
    1_000.0
}
fn default_price_max() -> f64 {
    50_000_000.0
}
fn default_bedrooms_max() -> f64 {
    15.0
}
fn default_min_chars() -> usize {
    80
}

impl Default for QualityFilterConfig {
    fn default() -> Self {
        Self {
            nonpositive_price: true,
            price_outlier: false,
            price_outlier_mad_z: default_mad_z(),
            missing_fields: false,
            price_range: false,
            price_min: default_price_min(),
            price_max: default_price_max(),
            bedrooms_outlier: false,
            bedrooms_max: default_bedrooms_max(),
            duplicate_content: false,
            short_content: false,
            short_content_min_chars: default_min_chars(),
        }
    }
}

impl QualityFilterConfig {
    pub fn rule_enabled(&self, rule: &str) -> bool {
        match rule {
            "nonpositive-price" => self.nonpositive_price,
            "price-outlier" => self.price_outlier,
            "missing-fields" => self.missing_fields,
            "price-range" => self.price_range,
            "bedrooms-outlier" => self.bedrooms_outlier,
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
