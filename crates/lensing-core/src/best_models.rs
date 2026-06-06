use serde::{Deserialize, Serialize};

/// How a model earned its slot in the best-models group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BestModelSource {
    /// Selected by the deterministic primary-metric ranking.
    Auto,
    /// Pinned by an operator/agent; always included regardless of rank.
    Pinned,
}

/// One member of the best-models group, with its ranking provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestModelEntry {
    /// Promoted model name (`data/models/<name>/`).
    pub name: String,
    /// 1 = best.
    pub rank: u32,
    /// Primary metric label the ranking used (e.g. "MAE").
    pub metric: String,
    /// Primary metric value, target space.
    pub metric_value: f64,
    pub run_id: String,
    pub predictor: String,
    pub dataset_id: String,
    pub source: BestModelSource,
    /// RFC 3339, UTC.
    pub selected_at: String,
}

/// The best-models group document (`data/best-models.json`, mirrored to the
/// Postgres `best_models` table). Rewritten wholesale on each recompute.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestModelGroup {
    /// Primary metric label from the domain at recompute time.
    pub primary_metric: String,
    /// Configured group-size cap (pins may exceed it).
    pub size: usize,
    /// Members ordered by rank.
    pub entries: Vec<BestModelEntry>,
    /// Model names force-included by curation.
    #[serde(default)]
    pub pinned: Vec<String>,
    /// Run ids / model names force-excluded by curation.
    #[serde(default)]
    pub excluded: Vec<String>,
    /// RFC 3339, UTC; last recompute.
    pub updated_at: String,
}

impl BestModelGroup {
    pub fn empty(primary_metric: &str, size: usize) -> Self {
        BestModelGroup {
            primary_metric: primary_metric.to_string(),
            size,
            entries: Vec::new(),
            pinned: Vec::new(),
            excluded: Vec::new(),
            updated_at: String::new(),
        }
    }
}
