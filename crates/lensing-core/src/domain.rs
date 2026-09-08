//! The domain configuration: everything that ties this framework to one
//! concrete prediction problem (corpus schema, target variable, feature
//! fields, quality-rule bindings, currency handling, UI vocabulary), parsed
//! from `domain.toml` at the repository root.
//!
//! Design rules:
//! - The contract/manifest types stay self-describing and never reference
//!   the domain — frozen models keep predicting if `domain.toml` changes.
//! - Quality rule KEYS ("price-range", "bedrooms-outlier", …) are stable
//!   internal identifiers shared by manifests, the API and the UI; the
//!   domain binds them to ITS fields and provides display labels. A new
//!   domain reuses the keys with different bindings/labels.
//! - The repository ships `domain.toml` as a NEUTRAL PLACEHOLDER (the
//!   template is unconfigured until /bootstrap); [`Domain::default`] embeds
//!   that file, so a missing file behaves exactly like the shipped state.
//!   The worked example (Argentine real-estate prices, exercising every
//!   lever) is the separate fixture behind [`Domain::example`].

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::manifest::TargetTransform;

/// The embedded copy of the repository's `domain.toml` (single source: the
/// file at the workspace root — the neutral placeholder a template ships).
pub const DEFAULT_DOMAIN_TOML: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../domain.toml"));

/// The worked-example domain: the framework's original problem (Argentine
/// real-estate sale prices), exercising every lever — [currency],
/// [coordinates], reconciled fields, vocab caps, quality bindings. Used as
/// the rich fixture by tests across the workspace and referenced from
/// BOOTSTRAP.md; NOT the shipped configuration.
pub const EXAMPLE_DOMAIN_TOML: &str = include_str!("example-domain.toml");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Domain {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    pub project: Project,
    pub corpus: Corpus,
    pub target: TargetSpec,
    /// Field descriptors in declared order. Order is load-bearing: feature
    /// columns are emitted numerics-first then categoricals, each in the
    /// order written here.
    pub fields: Vec<FieldDesc>,
    /// Plausible bounds for the coordinates field; values outside count as
    /// missing (mis-geocodes). Required iff a `coordinates`-role field exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coordinates: Option<CoordinateBounds>,
    /// Currency handling. Absent means a single-currency domain: currency
    /// mode is forced Off and the foreign-currency rule never excludes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub currency: Option<CurrencyDomain>,
    #[serde(default)]
    pub quality: QualityBindings,
    pub metrics: MetricsSpec,
    /// Sequence / next-item configuration. Required iff `[target].task =
    /// "ranking"`; absent for pointwise domains.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence: Option<SequenceSpec>,
    /// Interpretability-subsystem parameters (layer/embedding/SAE probes).
    #[serde(default)]
    pub interp: Interp,
}

/// Next-item / session configuration for a `ranking` task. The dataset is
/// built as ordered per-session item sequences (by an external sequences
/// producer), not a flat feature matrix; the model predicts the next item and
/// is ranked by Recall@k / MRR / hit-rate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceSpec {
    /// Payload field grouping events into a session (e.g. "session_id").
    pub session_field: String,
    /// Payload field giving intra-session order (a timestamp, e.g. "ts").
    pub order_field: String,
    /// Session-boundary gap in minutes.
    #[serde(default = "default_gap_minutes")]
    pub gap_minutes: u32,
    /// Events shorter than this (ms) count as skips and are dropped.
    #[serde(default = "default_skip_threshold_ms")]
    pub skip_threshold_ms: u32,
    /// How the next-item label is chosen. Currently only "last_item".
    #[serde(default = "default_seq_label")]
    pub label: String,
    /// Collection whose vectors are the per-item latents used for retrieval.
    pub latent_source: String,
    /// Item-metadata field holding the canonical identity token a serving
    /// request's prefix is matched against (e.g. a URI or slug). The predictor
    /// also accepts integer vocab indices regardless of this. Framework-
    /// neutral: the value is whatever `items.json` uses for identity.
    #[serde(default = "default_identity_field")]
    pub identity_field: String,
    /// Graded-relevance item fields the eval ALSO credits (`<field>@k`): a
    /// top-k candidate sharing the truth's value on one of these counts as a
    /// partial hit. A generic list — the framework hardcodes no field name.
    #[serde(default)]
    pub relevance_fields: Vec<String>,
}

fn default_gap_minutes() -> u32 {
    30
}
fn default_skip_threshold_ms() -> u32 {
    30_000
}
fn default_seq_label() -> String {
    "last_item".to_string()
}
fn default_identity_field() -> String {
    "id".to_string()
}

/// Interpretability parameters, consumed by the `/api/interp/*` tools and the
/// predictor probe subcommands. Entirely optional — every field has a neutral
/// default derived from the domain's own fields.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Interp {
    /// Categorical field the embedding / SAE probes slice rows by (the segment
    /// axis). Empty ⇒ the first `categorical`-role field.
    #[serde(default)]
    pub segment_field: String,
}

fn default_schema_version() -> u32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    /// Machine name (kebab-case).
    pub name: String,
    /// Human title for UI headers and reports.
    pub title: String,
    /// What one corpus row is called ("listing", "vehicle", …).
    pub entity_noun: String,
    pub entity_noun_plural: String,
    /// What the target variable is called ("price", "salary", …).
    pub target_noun: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Corpus {
    /// Defaults for the CLI/server flags (flags still override).
    pub qdrant_url: String,
    pub collection: String,
    pub manual_collection: String,
    pub embedding_dim: usize,
    /// JSON key under each point's payload where the metadata fields live;
    /// empty string means fields sit at the payload root.
    #[serde(default = "default_metadata_root")]
    pub metadata_root: String,
    /// Payload-root key holding the embedded document text (duplicate /
    /// short-content rules, area backfill).
    #[serde(default = "default_content_field")]
    pub content_field: String,
    pub filter: CorpusFilter,
}

fn default_metadata_root() -> String {
    "metadata".into()
}
fn default_content_field() -> String {
    "content".into()
}

/// The corpus point filter, translated 1:1 into a Qdrant filter. Field names
/// are metadata-rooted (the root prefix is applied automatically).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusFilter {
    /// Human-readable description, recorded in manifests.
    pub desc: String,
    #[serde(default)]
    pub must: Vec<FieldMatch>,
    #[serde(default)]
    pub must_not: Vec<FieldMatch>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldMatch {
    pub field: String,
    pub equals: serde_json::Value,
}

/// The learning task. Drives which metrics are computed/ranked and how the
/// target column + predictions are interpreted. Defaults to `Regression` so
/// existing domains (with no `task` key) are unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    #[default]
    Regression,
    /// Two classes (labels {0,1}); predictions carry P(class==1).
    Binary,
    /// K>2 classes (softmax); predictions carry the K-vector.
    Multiclass,
    /// Next-item retrieval over ordered sequences. Uses the `[sequence]`
    /// block instead of a scalar `[target]`; ranked by Recall@k / MRR / hit.
    Ranking,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetSpec {
    /// Metadata field holding the target value.
    pub field: String,
    pub transform: TargetTransform,
    pub format: TargetFormat,
    /// Learning task. Absent ⇒ regression (back-compat).
    #[serde(default)]
    pub task: Task,
    /// Class labels for `multiclass` (ordered; index = class id). Absent ⇒
    /// derived from the target field's distinct values at build time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub classes: Option<Vec<String>>,
}

/// How the UI renders target-space values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetFormat {
    /// "money" | "number"
    pub style: String,
    #[serde(default)]
    pub symbol: String,
    #[serde(default = "default_locale")]
    pub locale: String,
}

fn default_locale() -> String {
    "en-US".into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldRole {
    /// The prediction target; never a feature.
    Target,
    /// One-hot encoded feature with a frozen vocabulary.
    Categorical,
    /// Numeric feature column (optionally log1p-encoded, with indicator).
    Numeric,
    /// Geo pair encoded as lat/lon + pair-missing indicator.
    Coordinates,
    /// Used by the corpus filter only; never a feature.
    FilterOnly,
    /// RFC 3339 date (currency rate key); never a feature.
    Timestamp,
    /// Stored/displayed on manual entries; never a feature.
    Display,
}

/// Where a field lives inside the point payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldPath {
    /// Under `corpus.metadata_root` (the normal case).
    #[default]
    Metadata,
    /// Directly at the payload root (e.g. cluster tags).
    Payload,
}

/// Categorical vocabulary policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum VocabSpec {
    /// `vocab = "all"` — every distinct value, sorted, + "other" bucket.
    Named(VocabAll),
    /// `vocab = { top_n = 40 }` — top-N by frequency + "other" bucket.
    TopN { top_n: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VocabAll {
    All,
}

impl Default for VocabSpec {
    fn default() -> Self {
        VocabSpec::Named(VocabAll::All)
    }
}

impl VocabSpec {
    pub fn top_n(&self) -> Option<usize> {
        match self {
            VocabSpec::TopN { top_n } => Some(*top_n),
            VocabSpec::Named(_) => None,
        }
    }
}

/// Numeric encoding for a `Numeric`-role field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericEncode {
    /// Raw value column named after the field.
    #[default]
    Raw,
    /// `ln(1+x)` column named `<field>_log`.
    Log1p,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDesc {
    /// Payload JSON key (also the feature/manifest column basename and the
    /// `items.json` key — predictors read these names).
    pub name: String,
    /// Display label; defaults to `name`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub role: FieldRole,
    #[serde(default)]
    pub path: FieldPath,
    #[serde(default)]
    pub vocab: VocabSpec,
    #[serde(default)]
    pub encode: NumericEncode,
    /// Emit a `<field>_missing` indicator column next to the value column.
    #[serde(default)]
    pub indicator: bool,
    /// A value of 0 means "unspecified" (scraper convention).
    #[serde(default)]
    pub zero_is_missing: bool,
    /// Counted by the `missing-fields` quality rule when empty.
    #[serde(default)]
    pub critical: bool,
    /// Fetched from a companion collection by point id at build time
    /// (derived collections dropped these payload fields).
    #[serde(default)]
    pub reconcile: bool,
    /// Member of a named toggle group (e.g. "raw_numerics": fields that
    /// enable/disable together via one feature flag).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    /// Enabled in the default feature configuration.
    #[serde(default)]
    pub default_on: bool,
    /// UI input suggestions (datalist) for manual entries.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<String>,
    /// Required on manual entry forms.
    #[serde(default)]
    pub required: bool,
}

impl FieldDesc {
    pub fn label(&self) -> &str {
        self.label.as_deref().unwrap_or(&self.name)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinateBounds {
    /// The `Coordinates`-role field these bounds apply to.
    pub field: String,
    pub lat_range: [f64; 2],
    pub lon_range: [f64; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrencyDomain {
    /// The currency the target is expressed in.
    pub keep: String,
    /// Metadata field carrying a row's currency code.
    pub currency_field: String,
    /// Timestamp field keying the per-date exchange rate (Convert mode).
    pub date_field: String,
    /// Companion collection for currency/date reconciliation by point id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reconcile_collection: Option<String>,
    /// Default rate series name; substituted into `rate_url_template`.
    pub rate_source: String,
    /// `{source}` is replaced by the rate series. The endpoint must return
    /// `[{"fecha": "YYYY-MM-DD", "venta": <rate>}, …]`.
    pub rate_url_template: String,
    /// `[foreign, kept]` — the convertible pair (both directions).
    pub pair: [String; 2],
}

/// Bindings of the built-in quality rules to this domain's fields. Rule keys
/// are stable identifiers; labels are presentation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityBindings {
    /// Grouping field for the target-outlier MAD rule. Defaults to the first
    /// critical categorical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outlier_group: Option<String>,
    /// Numeric field checked by the cap rule (key "bedrooms-outlier").
    /// Absent disables that rule for the domain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capped_numeric: Option<String>,
    /// Rule key → display label, served to the UI.
    #[serde(default)]
    pub rule_labels: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSpec {
    /// Metric tables sort by this column.
    pub primary: String,
    /// Metric column order for reports/UI.
    pub columns: Vec<String>,
    /// Metrics rendered as percentages.
    pub percent: Vec<String>,
    /// Unit label for target-space metric axes ("USD", "points", …).
    pub value_unit: String,
    /// Size cap of the auto-maintained best-models group. Defaults to 12.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_models_size: Option<usize>,
    /// Per-metric ranking direction override (metric name → lower-is-better).
    /// Absent names fall back to the built-in defaults in [`Self::lower_is_better`].
    #[serde(default)]
    pub directions: BTreeMap<String, bool>,
}

impl MetricsSpec {
    pub const DEFAULT_BEST_MODELS_SIZE: usize = 12;

    /// Effective best-models group size.
    pub fn best_models_size(&self) -> usize {
        self.best_models_size.unwrap_or(Self::DEFAULT_BEST_MODELS_SIZE)
    }

    /// Read a named metric off a [`crate::Metrics`] blob (case-insensitive).
    /// `None` when the metric is unknown or absent on this run — callers then
    /// skip the run for ranking.
    pub fn extract(&self, name: &str, m: &crate::Metrics) -> Option<f64> {
        let lname = name.to_lowercase();
        match lname.as_str() {
            "mae" => m.mae,
            "rmse" => m.rmse,
            "mape" => m.mape,
            "medape" => m.medape,
            "r²" | "r2" => m.r2,
            "accuracy" | "acc" => m.accuracy,
            "logloss" | "log_loss" => m.logloss,
            "auc" | "roc_auc" | "macro_auc" => m.auc,
            "brier" => m.brier,
            "macro_f1" | "f1" => m.macro_f1,
            "mrr" => m.mrr,
            // ranking cutoff metrics carry an `@k` suffix (recall@10, hit@10).
            _ if lname.starts_with("recall") => m.recall_at_k,
            _ if lname.starts_with("hit") => m.hit_rate,
            // graded-relevance: "<field>_mrr" and "<field>@k" route to the
            // keyed graded maps (field ∈ [sequence].relevance_fields). No field
            // name is hardcoded in the framework.
            _ if lname.ends_with("_mrr") => m
                .graded_mrr
                .as_ref()
                .and_then(|g| g.get(lname.trim_end_matches("_mrr")).copied()),
            _ => m
                .graded_recall_at_k
                .as_ref()
                .and_then(|g| g.get(lname.split('@').next().unwrap_or("")).copied()),
        }
    }

    /// `Some(lower_is_better)` for the configured primary metric, or `None`
    /// when the primary names no known metric (ranking is then skipped).
    pub fn primary_direction(&self) -> Option<bool> {
        if self.extract_name_known(&self.primary) {
            Some(self.lower_is_better(&self.primary))
        } else {
            None
        }
    }

    /// Whether a smaller value of `name` ranks better. Error-style metrics
    /// (mae/rmse/mape/medape/logloss/brier) are lower-better; score-style
    /// metrics (r2/accuracy/auc/f1) are higher-better. `[metrics].directions`
    /// overrides per name.
    pub fn lower_is_better(&self, name: &str) -> bool {
        if let Some(&b) = self.directions.get(&name.to_lowercase()) {
            return b;
        }
        matches!(
            name.to_lowercase().as_str(),
            "mae" | "rmse" | "mape" | "medape" | "logloss" | "log_loss" | "brier"
        )
    }

    fn extract_name_known(&self, name: &str) -> bool {
        let lname = name.to_lowercase();
        lname == "mrr"
            || lname.contains('@')      // ranking cutoff metrics: recall@k, hit@k, <field>@k
            || lname.ends_with("_mrr") // graded per-field MRR
            || matches!(
                lname.as_str(),
                "mae" | "rmse" | "mape" | "medape" | "r²" | "r2" | "accuracy" | "acc"
                    | "logloss" | "log_loss" | "auc" | "roc_auc" | "macro_auc" | "brier"
                    | "macro_f1" | "f1"
            )
    }
}

impl Default for Domain {
    fn default() -> Self {
        toml::from_str(DEFAULT_DOMAIN_TOML).expect("embedded domain.toml is valid")
    }
}

impl Domain {
    /// The worked-example domain (see [`EXAMPLE_DOMAIN_TOML`]) — the rich
    /// fixture for tests that need coordinates/currency/reconcile/vocab
    /// behavior. The shipped default is the neutral placeholder.
    pub fn example() -> Self {
        toml::from_str(EXAMPLE_DOMAIN_TOML).expect("example-domain.toml is valid")
    }

    /// Default companion collection for reconciled-field joins (numerics
    /// and currency reconcile from the same raw collection): the
    /// `[currency].reconcile_collection` binding. `None` when the domain
    /// declares no companion — reconcile then relies on inline fields.
    pub fn companion_collection(&self) -> Option<String> {
        self.currency.as_ref().and_then(|c| c.reconcile_collection.clone())
    }

    /// Load `domain.toml` from `root`, falling back to the embedded default
    /// when the file is absent. A present-but-invalid file is an error.
    pub fn load_or_default(root: &Path) -> Result<Domain> {
        let path = root.join("domain.toml");
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Domain::default());
            }
            Err(e) => return Err(e).with_context(|| format!("read {}", path.display())),
        };
        let domain: Domain =
            toml::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        domain.validate().with_context(|| format!("validate {}", path.display()))?;
        Ok(domain)
    }

    pub fn validate(&self) -> Result<()> {
        let mut seen = std::collections::BTreeSet::new();
        for f in &self.fields {
            if !seen.insert(&f.name) {
                bail!("duplicate field {:?}", f.name);
            }
        }
        // A ranking task predicts the next sequence item, not a scalar payload
        // field, so it uses [sequence] instead of a role="target" field.
        if self.target.task != Task::Ranking {
            match self.field(&self.target.field) {
                Some(f) if f.role == FieldRole::Target => {}
                Some(f) => bail!("[target].field {:?} must have role \"target\", not {:?}", f.name, f.role),
                None => bail!("[target].field {:?} is not declared in [[fields]]", self.target.field),
            }
        }
        let coord_fields: Vec<&FieldDesc> =
            self.fields.iter().filter(|f| f.role == FieldRole::Coordinates).collect();
        match (&self.coordinates, coord_fields.as_slice()) {
            (Some(b), [f]) if b.field == f.name => {}
            (None, []) => {}
            (Some(b), _) => bail!(
                "[coordinates].field {:?} must name exactly one coordinates-role field",
                b.field
            ),
            (None, _) => bail!("a coordinates-role field requires a [coordinates] bounds section"),
        }
        if let Some(c) = &self.currency {
            if c.pair.iter().all(|p| p != &c.keep) {
                bail!("[currency].keep {:?} must be one of pair {:?}", c.keep, c.pair);
            }
        }
        if let Some(g) = &self.quality.outlier_group {
            if !self.fields.iter().any(|f| &f.name == g && f.role == FieldRole::Categorical) {
                bail!("[quality].outlier_group {:?} is not a categorical field", g);
            }
        }
        if let Some(n) = &self.quality.capped_numeric {
            if !self.fields.iter().any(|f| &f.name == n && f.role == FieldRole::Numeric) {
                bail!("[quality].capped_numeric {:?} is not a numeric field", n);
            }
        }
        if !self.metrics.columns.contains(&self.metrics.primary) {
            bail!("[metrics].primary {:?} is not in [metrics].columns", self.metrics.primary);
        }
        if self.metrics.primary_direction().is_none() {
            bail!(
                "[metrics].primary {:?} is not a known metric (mae/rmse/mape/medape/r2/accuracy/logloss/auc/brier/macro_f1)",
                self.metrics.primary
            );
        }
        // Classification targets are class labels, not continuous values:
        // a target transform (e.g. log1p) on a class id is meaningless.
        if self.target.task != Task::Regression && self.target.transform != TargetTransform::None {
            bail!(
                "[target].transform must be \"none\" for a {:?} task, not {:?}",
                self.target.task, self.target.transform
            );
        }
        // A ranking task requires a [sequence] block whose session/order/
        // latent fields are populated; a pointwise task must not carry one.
        // Those fields come from an external event stream, not [[fields]], so
        // we only require the block to be present + non-empty.
        match (self.target.task, &self.sequence) {
            (Task::Ranking, None) => bail!("[target].task = \"ranking\" requires a [sequence] block"),
            (Task::Ranking, Some(seq)) => {
                for (key, val) in [
                    ("session_field", &seq.session_field),
                    ("order_field", &seq.order_field),
                    ("latent_source", &seq.latent_source),
                ] {
                    if val.trim().is_empty() {
                        bail!("[sequence].{key} must be a non-empty field/collection name");
                    }
                }
            }
            (_, Some(_)) => bail!("[sequence] is only valid with [target].task = \"ranking\""),
            (_, None) => {}
        }
        Ok(())
    }

    // ---------- derived accessors ----------

    pub fn field(&self, name: &str) -> Option<&FieldDesc> {
        self.fields.iter().find(|f| f.name == name)
    }

    pub fn numeric_fields(&self) -> impl Iterator<Item = &FieldDesc> {
        self.fields.iter().filter(|f| f.role == FieldRole::Numeric)
    }

    pub fn categorical_fields(&self) -> impl Iterator<Item = &FieldDesc> {
        self.fields.iter().filter(|f| f.role == FieldRole::Categorical)
    }

    /// The categorical field the interpretability probes slice by: the explicit
    /// `[interp].segment_field` binding when set and present, else the first
    /// `categorical`-role field, else `None` (probes run pooled-only).
    pub fn interp_segment_field(&self) -> Option<&str> {
        let explicit = self.interp.segment_field.trim();
        if !explicit.is_empty() {
            if self.fields.iter().any(|f| f.name == explicit && f.role == FieldRole::Categorical) {
                return Some(explicit);
            }
        }
        self.categorical_fields().next().map(|f| f.name.as_str())
    }

    pub fn coordinates_field(&self) -> Option<&FieldDesc> {
        self.fields.iter().find(|f| f.role == FieldRole::Coordinates)
    }

    /// Fields reconciled from a companion collection (numeric + coordinates).
    pub fn reconcile_fields(&self) -> impl Iterator<Item = &FieldDesc> {
        self.fields.iter().filter(|f| f.reconcile)
    }

    /// Fields counted by the missing-fields rule.
    pub fn critical_fields(&self) -> impl Iterator<Item = &FieldDesc> {
        self.fields.iter().filter(|f| f.critical)
    }

    /// Grouping field for the target-outlier rule (explicit binding, else
    /// the first critical categorical, else none → global stats only).
    pub fn outlier_group(&self) -> Option<&str> {
        self.quality
            .outlier_group
            .as_deref()
            .or_else(|| self.critical_fields().find(|f| f.role == FieldRole::Categorical).map(|f| f.name.as_str()))
    }

    /// Whether `field` is enabled under `cfg`. The generic `cfg.fields` map
    /// (field name or toggle-group name → bool) is authoritative when
    /// non-empty; otherwise the legacy named flags apply — a compatibility
    /// shim for pre-domain manifests/requests, keyed by the original
    /// example-domain field names (other domains always use the map).
    pub fn field_enabled(&self, cfg: &crate::FeatureConfig, f: &FieldDesc) -> bool {
        if !cfg.fields.is_empty() {
            if let Some(&on) = cfg.fields.get(&f.name) {
                return on;
            }
            if let Some(group) = &f.group {
                if let Some(&on) = cfg.fields.get(group) {
                    return on;
                }
            }
            return false;
        }
        // Legacy flag mapping (pre-domain configs).
        match (f.name.as_str(), f.group.as_deref(), f.role) {
            (_, Some("raw_numerics"), _) => cfg.raw_numerics,
            (_, _, FieldRole::Coordinates) => cfg.coordinates,
            ("bedrooms", _, _) => cfg.bedrooms,
            ("propertyType", _, _) => cfg.property_type,
            ("neighborhood", _, _) => cfg.neighborhood_top_n > 0,
            ("city", _, _) => cfg.city,
            ("province", _, _) => cfg.province,
            ("cluster", _, _) => cfg.cluster,
            _ => f.default_on,
        }
    }

    /// Effective vocabulary size for a categorical under `cfg` (None = all
    /// distinct values).
    pub fn effective_top_n(&self, cfg: &crate::FeatureConfig, f: &FieldDesc) -> Option<usize> {
        if let Some(&n) = cfg.vocab_top_n.get(&f.name) {
            return Some(n);
        }
        if cfg.fields.is_empty() && f.name == "neighborhood" && cfg.neighborhood_top_n > 0 {
            return Some(cfg.neighborhood_top_n);
        }
        f.vocab.top_n()
    }

    /// Resolve a config into its fully-explicit form before a build: the
    /// generic `fields`/`vocab_top_n` maps are populated from the effective
    /// enable set, coordinate bounds are frozen in, and the legacy flags are
    /// synced for fields they describe (so old readers of new manifests see
    /// truthful values).
    pub fn normalize_config(&self, cfg: &mut crate::FeatureConfig) {
        let enabled: Vec<(String, bool, Option<usize>)> = self
            .fields
            .iter()
            .filter(|f| {
                matches!(
                    f.role,
                    FieldRole::Numeric | FieldRole::Categorical | FieldRole::Coordinates
                )
            })
            .map(|f| (f.name.clone(), self.field_enabled(cfg, f), self.effective_top_n(cfg, f)))
            .collect();
        for (name, on, top_n) in &enabled {
            cfg.fields.insert(name.clone(), *on);
            if let (true, Some(n)) = (*on, top_n) {
                cfg.vocab_top_n.insert(name.clone(), *n);
            }
        }
        if let Some(b) = &self.coordinates {
            cfg.coordinate_bounds = Some([b.lat_range, b.lon_range]);
        }
        let on = |name: &str| enabled.iter().any(|(n, e, _)| n == name && *e);
        cfg.bedrooms = on("bedrooms");
        cfg.property_type = on("propertyType");
        cfg.neighborhood_top_n = if on("neighborhood") {
            self.field("neighborhood")
                .and_then(|f| self.effective_top_n(cfg, f))
                .unwrap_or(0)
        } else {
            0
        };
        cfg.city = on("city");
        cfg.province = on("province");
        cfg.cluster = on("cluster");
        cfg.raw_numerics = self
            .fields
            .iter()
            .any(|f| f.group.as_deref() == Some("raw_numerics") && on(&f.name));
        cfg.coordinates = self.coordinates_field().map(|f| on(&f.name)).unwrap_or(false);
    }

    /// Bounds for this domain's coordinates field. Domains without one fall
    /// back to the legacy bounds (only reachable from pre-domain artifacts).
    pub fn coordinate_bounds_arr(&self) -> [[f64; 2]; 2] {
        self.coordinates
            .as_ref()
            .map(|b| [b.lat_range, b.lon_range])
            .unwrap_or_else(crate::manifest::legacy_coordinate_bounds)
    }

    /// The fully-qualified payload key of a metadata field (root applied).
    pub fn qualified_key(&self, f: &FieldDesc) -> String {
        match f.path {
            FieldPath::Payload => f.name.clone(),
            FieldPath::Metadata if self.corpus.metadata_root.is_empty() => f.name.clone(),
            FieldPath::Metadata => format!("{}.{}", self.corpus.metadata_root, f.name),
        }
    }

    /// The Qdrant filter for the corpus scroll/count, from `[corpus.filter]`.
    pub fn qdrant_filter(&self) -> serde_json::Value {
        let prefix = |field: &str| {
            if self.corpus.metadata_root.is_empty() {
                field.to_string()
            } else {
                format!("{}.{}", self.corpus.metadata_root, field)
            }
        };
        let clause = |m: &FieldMatch| {
            serde_json::json!({ "key": prefix(&m.field), "match": { "value": m.equals } })
        };
        serde_json::json!({
            "must": self.corpus.filter.must.iter().map(clause).collect::<Vec<_>>(),
            "must_not": self.corpus.filter.must_not.iter().map(clause).collect::<Vec<_>>(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_domain_parses_and_validates() {
        // The shipped default is the neutral placeholder a blank template
        // boots with — structurally valid, no example-domain bindings.
        let d = Domain::default();
        d.validate().unwrap();
        assert_eq!(d.target.field, "target");
        assert_eq!(d.corpus.metadata_root, "metadata");
        let numeric: Vec<&str> = d.numeric_fields().map(|f| f.name.as_str()).collect();
        assert_eq!(numeric, ["numeric_example"]);
        let cats: Vec<&str> = d.categorical_fields().map(|f| f.name.as_str()).collect();
        assert_eq!(cats, ["category_example"]);
        assert_eq!(d.coordinates_field().map(|f| f.name.as_str()), None);
        assert_eq!(d.outlier_group(), None);
        assert_eq!(d.quality.capped_numeric.as_deref(), None);
        assert!(d.currency.is_none());
        assert_eq!(d.qualified_key(d.field("target").unwrap()), "metadata.target");
    }

    #[test]
    fn example_domain_parses_and_validates() {
        let d = Domain::example();
        d.validate().unwrap();
        assert_eq!(d.target.field, "price");
        // Declared order is load-bearing for column order.
        let numeric: Vec<&str> = d.numeric_fields().map(|f| f.name.as_str()).collect();
        assert_eq!(numeric, ["bedrooms", "totalArea", "coveredArea", "bathrooms", "garages", "rooms"]);
        let cats: Vec<&str> = d.categorical_fields().map(|f| f.name.as_str()).collect();
        assert_eq!(cats, ["propertyType", "neighborhood", "city", "province", "cluster"]);
        assert_eq!(d.coordinates_field().map(|f| f.name.as_str()), Some("coordinates"));
        assert_eq!(d.outlier_group(), Some("propertyType"));
        assert_eq!(d.quality.capped_numeric.as_deref(), Some("bedrooms"));
        // cluster lives at the payload root.
        let cluster = d.field("cluster").unwrap();
        assert_eq!(cluster.path, FieldPath::Payload);
        assert_eq!(d.qualified_key(cluster), "cluster");
        assert_eq!(d.qualified_key(d.field("price").unwrap()), "metadata.price");
    }

    #[test]
    fn metric_extract_and_direction() {
        let m = crate::Metrics {
            n_test: 10,
            mae: Some(1.0),
            rmse: Some(2.0),
            r2: Some(0.5),
            mape: Some(0.2),
            medape: Some(0.1),
            auc: Some(0.77),
            ..Default::default()
        };
        let spec = |primary: &str| MetricsSpec {
            primary: primary.into(),
            columns: vec![primary.into()],
            percent: vec![],
            value_unit: "USD".into(),
            best_models_size: None,
            directions: std::collections::BTreeMap::new(),
        };
        let s = spec("MAE");
        assert_eq!(s.extract("MAE", &m), Some(1.0));
        assert!(s.lower_is_better("MAE"));
        assert_eq!(spec("medAPE").extract("medAPE", &m), Some(0.1));
        assert!(spec("medAPE").lower_is_better("medAPE"));
        // score-style metrics rank higher-is-better
        assert_eq!(spec("R²").extract("R²", &m), Some(0.5));
        assert!(!spec("R²").lower_is_better("R²"));
        assert_eq!(spec("AUC").extract("auc", &m), Some(0.77));
        assert!(!spec("AUC").lower_is_better("AUC"));
        // unknown metric → no direction, absent metric → None
        assert!(spec("ELO").primary_direction().is_none());
        assert_eq!(s.extract("logloss", &m), None);
        // direction override
        let mut over = spec("AUC");
        over.directions.insert("auc".into(), true);
        assert!(over.lower_is_better("AUC"));
        // embedded domain: known primary, default group size
        let d = Domain::default();
        assert!(d.metrics.primary_direction().is_some());
        assert_eq!(d.metrics.best_models_size(), 12);
    }

    #[test]
    fn qdrant_filter_matches_legacy_sale_filter() {
        let d = Domain::example();
        assert_eq!(
            d.qdrant_filter(),
            serde_json::json!({
                "must": [{ "key": "metadata.operation", "match": { "value": "sale" } }],
                "must_not": [{ "key": "metadata.price", "match": { "value": 0 } }]
            })
        );
    }
}
