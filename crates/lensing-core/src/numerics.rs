use serde::{Deserialize, Serialize};

/// Recorded in the manifest of a dataset built with reconciled numeric
/// features: coverage of each reconciled field over the post-quality rows.
/// The build collections dropped these fields; they are joined back from the
/// raw scrape collection by point id (like currency), with an optional
/// document-text backfill for areas.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NumericsReport {
    /// Companion collection the fields were reconciled from (None = fields
    /// were already inline in the source collection).
    pub reconcile_collection: Option<String>,
    /// Whether missing areas were backfilled from the document text.
    pub content_backfill: bool,
    /// Rows with each field present (> 0; zero means unspecified) after
    /// reconciliation and backfill, keyed by field name. Legacy manifests
    /// recorded fixed `n_<snake_case>` keys instead; those are accepted on
    /// read via the catch-all below.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub coverage: std::collections::BTreeMap<String, usize>,
    /// Rows whose area came from the document text (subset of coverage).
    #[serde(default)]
    pub n_area_backfilled: usize,
    /// Scraper-mangled areas repaired (fractional sub-15 m² values whose
    /// ×1000 is plausible — European thousands dots parsed as decimals).
    #[serde(default)]
    pub n_area_repaired: usize,
    /// Rows with sanity-bounded coordinates present (mis-geocodes count as
    /// missing).
    #[serde(default)]
    pub n_coordinates: usize,
    /// Legacy fixed-field counts from pre-domain manifests (display-only;
    /// an empty map flattens to nothing on serialize).
    #[serde(flatten)]
    pub legacy: std::collections::BTreeMap<String, serde_json::Value>,
}

/// Frozen imputation values for the reconciled numeric fields, computed from
/// the train split only (the build owns the split, so test rows never leak
/// into the medians). Recorded in the manifest when `impute_numerics` is on
/// and copied into the promoted contract so predict imputes identically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericImputation {
    /// Categorical field whose value selects the median group. Pre-domain
    /// contracts deserialize to the original grouping field.
    #[serde(default = "legacy_group_field")]
    pub group_field: String,
    /// Per-group medians over train rows with the field present. Groups with
    /// no present value for a field fall back to `global`. (The JSON key
    /// keeps its historical name for contract compatibility.)
    pub by_property_type: std::collections::BTreeMap<String, NumericMedians>,
    /// Medians over all train rows (fallback for unseen groups).
    pub global: NumericMedians,
}

fn legacy_group_field() -> String {
    "propertyType".into()
}

/// Median of each reconciled numeric field, keyed by FIELD NAME (in source
/// units, not log space). A zero means the field had no present value
/// anywhere — imputed rows then keep the zero-as-unspecified convention.
///
/// Pre-domain contracts stored a fixed snake_case struct; deserialization
/// accepts both and normalizes the legacy keys to field names.
#[derive(Debug, Clone, Default, Serialize)]
pub struct NumericMedians(pub std::collections::BTreeMap<String, f64>);

impl NumericMedians {
    pub fn get(&self, field: &str) -> Option<f64> {
        self.0.get(field).copied()
    }
}

impl<'de> Deserialize<'de> for NumericMedians {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = std::collections::BTreeMap::<String, f64>::deserialize(d)?;
        // Legacy fixed-struct keys → original field names.
        let map = raw
            .into_iter()
            .map(|(k, v)| {
                let k = match k.as_str() {
                    "total_area" => "totalArea".to_string(),
                    "covered_area" => "coveredArea".to_string(),
                    _ => k,
                };
                (k, v)
            })
            .collect();
        Ok(NumericMedians(map))
    }
}
