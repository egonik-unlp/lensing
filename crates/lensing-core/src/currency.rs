use serde::{Deserialize, Serialize};

/// How non-`keep` currencies are handled at dataset build time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CurrencyMode {
    /// Currency is ignored entirely (no reconcile fetch either).
    Off,
    /// Hard filter: rows whose currency differs from `keep` are excluded
    /// (via the `foreign-currency` quality rule).
    Filter,
    /// Convert foreign target values into `keep` using per-date exchange rates.
    Convert,
}

/// Currency handling for a dataset build. The corpus mixes ARS and USD
/// prices; sale listings are conventionally priced in USD, so the default
/// hard-filters to USD.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrencyConfig {
    #[serde(default = "default_mode")]
    pub mode: CurrencyMode,
    /// The currency the target is expressed in.
    #[serde(default = "default_keep")]
    pub keep: String,
    /// Companion collection holding `metadata.currency` / `metadata.createdAt`
    /// for the same point ids (the build collection dropped them). `None`
    /// relies on currency already inline in the source collection.
    #[serde(default = "default_reconcile")]
    pub reconcile_collection: Option<String>,
    /// Exchange-rate series for `Convert`, substituted into the domain's
    /// `rate_url_template` (e.g. "blue" or "oficial" in the example domain).
    #[serde(default = "default_rate_source")]
    pub rate_source: String,
}

fn default_mode() -> CurrencyMode {
    CurrencyMode::Filter
}
fn default_keep() -> String {
    "USD".into()
}
fn default_reconcile() -> Option<String> {
    Some("properties".into())
}
fn default_rate_source() -> String {
    "blue".into()
}

impl Default for CurrencyConfig {
    fn default() -> Self {
        Self {
            mode: default_mode(),
            keep: default_keep(),
            reconcile_collection: default_reconcile(),
            rate_source: default_rate_source(),
        }
    }
}

/// Recorded in the manifest of a dataset built with currency handling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrencyReport {
    pub config: CurrencyConfig,
    /// Rows whose currency differed from `keep` (before any conversion).
    pub n_foreign: usize,
    /// Rows with no currency after reconciliation (kept, never dropped).
    pub n_missing: usize,
    /// Rows whose target value was converted into `keep` (Convert mode only).
    pub n_converted: usize,
    /// Range of exchange rates actually applied (Convert mode only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_max: Option<f64>,
}
