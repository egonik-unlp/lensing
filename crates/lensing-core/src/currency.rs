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

/// Currency handling for a dataset build (only meaningful for domains with
/// a `[currency]` section — single-currency domains force mode Off). The
/// per-build overrides default to the domain's bindings via the
/// `effective_*` accessors: an empty/absent value means "use the domain's".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrencyConfig {
    #[serde(default = "default_mode")]
    pub mode: CurrencyMode,
    /// The currency the target is expressed in; "" → the domain's
    /// `[currency].keep`.
    #[serde(default)]
    pub keep: String,
    /// Companion collection holding the currency/date fields for the same
    /// point ids (when the build collection dropped them). Absent → the
    /// domain's `[currency].reconcile_collection`; `""` relies on currency
    /// already inline in the source collection.
    #[serde(default)]
    pub reconcile_collection: Option<String>,
    /// Exchange-rate series for `Convert`, substituted into the domain's
    /// `rate_url_template`; "" → the domain's `[currency].rate_source`.
    #[serde(default)]
    pub rate_source: String,
}

fn default_mode() -> CurrencyMode {
    CurrencyMode::Filter
}

impl Default for CurrencyConfig {
    fn default() -> Self {
        Self {
            mode: default_mode(),
            keep: String::new(),
            reconcile_collection: None,
            rate_source: String::new(),
        }
    }
}

impl CurrencyConfig {
    /// The kept currency, falling back to the domain's binding.
    pub fn effective_keep<'a>(&'a self, dc: &'a crate::domain::CurrencyDomain) -> &'a str {
        if self.keep.is_empty() {
            &dc.keep
        } else {
            &self.keep
        }
    }

    /// The rate series, falling back to the domain's binding.
    pub fn effective_rate_source<'a>(&'a self, dc: &'a crate::domain::CurrencyDomain) -> &'a str {
        if self.rate_source.is_empty() {
            &dc.rate_source
        } else {
            &self.rate_source
        }
    }

    /// The reconcile companion: explicit value, `""` disables the join,
    /// absent falls back to the domain's binding.
    pub fn effective_reconcile<'a>(
        &'a self,
        dc: &'a crate::domain::CurrencyDomain,
    ) -> Option<&'a str> {
        match &self.reconcile_collection {
            Some(c) if c.is_empty() => None,
            Some(c) => Some(c),
            None => dc.reconcile_collection.as_deref(),
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
