use serde::{Deserialize, Serialize};

/// `manifest.json` at the root of a dataset directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub dataset_id: String,
    /// Optional human-assigned display name. The slug `dataset_id` stays the
    /// stable identity (lineage keys off it); this is presentation only.
    /// Absent on datasets built before naming existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// RFC 3339, UTC.
    pub created_at: String,
    pub source: Source,
    pub n_rows: usize,
    pub n_cols: usize,
    /// One descriptor per feature column, in on-disk order.
    pub columns: Vec<ColumnDesc>,
    pub pca: PcaInfo,
    pub target: TargetInfo,
    pub split: SplitInfo,
    /// Echo of the feature configuration the dataset was built with.
    pub feature_config: FeatureConfig,
    /// Quality filters applied at build time, with per-rule counts. Absent
    /// on datasets built before quality filtering existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<crate::quality::QualityReport>,
    /// Cumulative explained-variance curve over the full eigen-spectrum
    /// (train split), truncated for size. Lets the detail view show how much
    /// variance more components would have captured. Empty on older datasets.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cumulative_evr: Vec<f32>,
    /// Feature-redundancy findings computed at build time. Absent on datasets
    /// built before redundancy analysis existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub redundancy: Option<crate::redundancy::RedundancyReport>,
    /// Currency handling applied at build time (reconcile / filter / convert).
    /// Absent on datasets built before currency handling existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub currency: Option<crate::currency::CurrencyReport>,
    /// Raw-metadata numeric reconciliation applied at build time (areas,
    /// bathrooms, garages, rooms). Absent unless `raw_numerics` was on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numerics: Option<crate::numerics::NumericsReport>,
    /// Frozen train-split medians used to fill missing numeric fields.
    /// Absent unless `impute_numerics` was on; promotion copies it into the
    /// contract so predict imputes identically.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imputation: Option<crate::numerics::NumericImputation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub qdrant_url: String,
    pub collection: String,
    /// Human-readable description of the point filter.
    pub filter: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDesc {
    pub name: String,
    pub kind: ColumnKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ColumnKind {
    /// PCA component of the embedding vector.
    Pca { component: usize },
    /// Raw numeric payload field.
    Numeric { field: String },
    /// One-hot indicator. `group` is the source field, `value` the category.
    Onehot { group: String, value: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PcaInfo {
    pub dims: usize,
    /// Mean of the original 1536-dim vectors over the train split (inline).
    pub mean: Vec<f32>,
    /// Shape of `pca_components.f32`: [dims, original_dim], row-major.
    pub components_shape: [usize; 2],
    pub explained_variance_ratio: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetInfo {
    pub field: String,
    pub transform: TargetTransform,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetTransform {
    Log1p,
    None,
}

impl TargetTransform {
    /// Map a transformed target value back to price space.
    pub fn invert(&self, y: f64) -> f64 {
        match self {
            TargetTransform::Log1p => y.exp_m1(),
            TargetTransform::None => y,
        }
    }

    pub fn apply(&self, price: f64) -> f64 {
        match self {
            TargetTransform::Log1p => price.ln_1p(),
            TargetTransform::None => price,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitInfo {
    pub test_ratio: f64,
    pub seed: u64,
    pub n_train: usize,
    pub n_test: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    pub pca_dims: usize,
    /// Generic per-field enables, keyed by domain field name (or toggle-group
    /// name). When non-empty this is authoritative and the legacy named flags
    /// below are ignored. Written by every new build; legacy-only configs
    /// come from old manifests/contracts and pre-domain API clients.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub fields: std::collections::BTreeMap<String, bool>,
    /// Per-categorical vocabulary-size overrides (field name → top-N).
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub vocab_top_n: std::collections::BTreeMap<String, usize>,
    /// Coordinate plausibility bounds frozen into the dataset/contract:
    /// `[[lat_min, lat_max], [lon_min, lon_max]]`. Absent on pre-domain
    /// artifacts → [`legacy_coordinate_bounds`] (the original corpus bounds).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coordinate_bounds: Option<[[f64; 2]; 2]>,
    #[serde(default = "default_flag_true")]
    pub bedrooms: bool,
    #[serde(default = "default_flag_true")]
    pub property_type: bool,
    /// 0 disables neighborhood one-hots; otherwise top-N + "other" bucket.
    #[serde(default = "default_top_n_40")]
    pub neighborhood_top_n: usize,
    #[serde(default)]
    pub city: bool,
    #[serde(default)]
    pub province: bool,
    #[serde(default)]
    pub cluster: bool,
    /// Numeric payload fields from the raw scrape collection (totalArea,
    /// coveredArea, bathrooms, garages, rooms), reconciled by point id at
    /// build time: log1p columns for areas, raw counts for the rest, each
    /// with a missing indicator. Absent on datasets built before this existed.
    #[serde(default)]
    pub raw_numerics: bool,
    /// With `raw_numerics`: rows lacking both areas get totalArea extracted
    /// from "… m²" mentions in the listing text.
    #[serde(default)]
    pub area_content_backfill: bool,
    /// Latitude/longitude from the raw collection's `metadata.coordinates`,
    /// reconciled by point id like the numeric fields: raw-degree lat/lon
    /// columns plus a single pair-missing indicator. Out-of-Argentina values
    /// are treated as missing (mis-geocodes).
    #[serde(default)]
    pub coordinates: bool,
    /// With `raw_numerics`: fill missing numeric fields with per-propertyType
    /// train-split medians and drop the missing-indicator columns. Kernel
    /// models can't branch on indicators the way trees do; this gives them
    /// meaningful distances instead (the frozen medians live in the manifest
    /// and the promoted contract so predict imputes identically).
    #[serde(default)]
    pub impute_numerics: bool,
}

fn default_flag_true() -> bool {
    true
}
fn default_top_n_40() -> usize {
    40
}

/// Coordinate bounds of artifacts frozen before the domain configuration
/// existed (the original corpus: Argentina). Pre-domain contracts encoded
/// out-of-bounds geocodes as missing against exactly these ranges, so the
/// inference path must keep reproducing them.
pub fn legacy_coordinate_bounds() -> [[f64; 2]; 2] {
    [[-56.0, -21.0], [-74.0, -53.0]]
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            pca_dims: 32,
            fields: Default::default(),
            vocab_top_n: Default::default(),
            coordinate_bounds: None,
            bedrooms: true,
            property_type: true,
            neighborhood_top_n: 40,
            city: false,
            province: false,
            cluster: false,
            raw_numerics: false,
            area_content_backfill: false,
            coordinates: false,
            impute_numerics: false,
        }
    }
}
