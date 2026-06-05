use serde::{Deserialize, Serialize};

/// Feature-redundancy findings over the non-PCA columns (numeric + one-hot).
/// PCA components are orthogonal by construction, so they are excluded from
/// every check here. Computed at analyze time and persisted in the manifest.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RedundancyReport {
    /// Columns whose variance is at or below the near-zero threshold: they
    /// carry almost no signal.
    pub near_zero_variance: Vec<NearZeroColumn>,
    /// Per one-hot group occupancy (how the categories are spread).
    pub onehot_groups: Vec<OnehotGroupStat>,
    /// Column pairs with `|corr| > threshold` (numeric + one-hot only).
    pub correlated_pairs: Vec<CorrelatedPair>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NearZeroColumn {
    pub column: String,
    pub variance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnehotGroupStat {
    pub group: String,
    /// Distinct categories in the vocabulary (excludes the `__other__` bucket).
    pub n_values: usize,
    /// Fraction of rows that fell into the `__other__` bucket, if the group
    /// has one.
    pub other_fraction: f64,
    /// Categories occupied by fewer than `rare_threshold` rows.
    pub rare_buckets: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelatedPair {
    pub a: String,
    pub b: String,
    pub corr: f64,
}
