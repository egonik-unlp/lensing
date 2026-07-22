//! The domain configuration: everything that ties this framework to one
//! concrete prediction problem (corpus schema, target variable, feature
//! fields, quality-rule bindings, currency handling, UI vocabulary), parsed
//! from `domain.toml` at the repository root.
//!
//! Design rules:
//! - The contract/manifest types stay self-describing and never reference
//!   the domain — frozen models keep predicting if `domain.toml` changes.
//! - Quality rule KEYS ("target-range", "capped-numeric-outlier", …) are stable
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
    /// Interpretability-subsystem parameters (layer/embedding/SAE probes).
    #[serde(default)]
    pub interp: Interp,
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
    /// Continuous target on time-ordered rows: the train/test split is
    /// chronological (train = past, test = future) instead of random, so a
    /// model is only ever evaluated on rows later than everything it trained
    /// on. Target space and metrics are otherwise regression-like.
    TimeSeries,
}

impl Task {
    /// Classification tasks carry class *labels*, not continuous values, so a
    /// target transform (e.g. log1p) is meaningless on them.
    pub fn is_classification(self) -> bool {
        matches!(self, Task::Binary | Task::Multiclass)
    }
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
    /// For `time_series`: the field the chronological split orders by. Absent ⇒
    /// the domain's single `timestamp`-role field. Ignored for other tasks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_field: Option<String>,
    /// For `time_series`: forecast horizon (rows ahead), stored on the manifest
    /// for predictors and used as an embargo gap between train and test so the
    /// model cannot peek at rows within one horizon of the split. Ignored for
    /// other tasks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub horizon: Option<u32>,
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
    /// Numeric field checked by the cap rule (key "capped-numeric-outlier").
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

    /// Read a named metric off a [`crate::Metrics`] blob (case- and
    /// separator-insensitive). Resolves built-in metrics from their typed
    /// fields and instance-defined metrics from the run's `extra` map, so a
    /// domain can rank by any metric its predictors emit. `None` when the
    /// metric is absent on this run — callers then skip it for ranking.
    pub fn extract(&self, name: &str, m: &crate::Metrics) -> Option<f64> {
        m.get(name)
    }

    /// `Some(lower_is_better)` for the configured primary metric. Always
    /// resolvable now that direction is data-driven (explicit
    /// `[metrics].directions`, else a name heuristic), so ranking is never
    /// skipped for lack of a known name; `Option` is kept for callers.
    pub fn primary_direction(&self) -> Option<bool> {
        Some(self.lower_is_better(&self.primary))
    }

    /// Whether a smaller value of `name` ranks better. An explicit
    /// `[metrics].directions` entry wins (matched name-insensitively);
    /// otherwise error/loss-style metrics are lower-better and score-style
    /// higher-better — see [`crate::run::metric_lower_is_better_default`].
    pub fn lower_is_better(&self, name: &str) -> bool {
        let norm = crate::run::normalize_metric_name(name);
        if let Some((_, &b)) =
            self.directions.iter().find(|(k, _)| crate::run::normalize_metric_name(k) == norm)
        {
            return b;
        }
        crate::run::metric_lower_is_better_default(name)
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
        match self.field(&self.target.field) {
            Some(f) if f.role == FieldRole::Target => {}
            Some(f) => bail!("[target].field {:?} must have role \"target\", not {:?}", f.name, f.role),
            None => bail!("[target].field {:?} is not declared in [[fields]]", self.target.field),
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
        // Metric names are open: built-ins resolve from typed fields, anything
        // else from a predictor's `extra` map. A metric only needs a ranking
        // direction, which is always derivable ([metrics].directions or the
        // name heuristic) — so no name is rejected here.
        // Classification targets are class labels, not continuous values:
        // a target transform (e.g. log1p) on a class id is meaningless.
        // Regression and time_series targets are continuous and may transform.
        if self.target.task.is_classification() && self.target.transform != TargetTransform::None {
            bail!(
                "[target].transform must be \"none\" for a {:?} task, not {:?}",
                self.target.task, self.target.transform
            );
        }
        // A time_series target needs a resolvable ordering field: either an
        // explicit [target].time_field naming a timestamp-role field, or
        // exactly one timestamp-role field to fall back on.
        if self.target.task == Task::TimeSeries {
            self.time_field()?;
        }
        Ok(())
    }

    // ---------- derived accessors ----------

    pub fn field(&self, name: &str) -> Option<&FieldDesc> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// The field a `time_series` build orders its chronological split by:
    /// `[target].time_field` when set (must name a `timestamp`-role field),
    /// else the domain's sole `timestamp`-role field. `Err` when the choice is
    /// ambiguous or missing — called from [`Self::validate`] and the builder.
    pub fn time_field(&self) -> Result<&FieldDesc> {
        if let Some(name) = &self.target.time_field {
            return match self.field(name) {
                Some(f) if f.role == FieldRole::Timestamp => Ok(f),
                Some(f) => bail!(
                    "[target].time_field {:?} must have role \"timestamp\", not {:?}",
                    name, f.role
                ),
                None => bail!("[target].time_field {:?} is not declared in [[fields]]", name),
            };
        }
        let ts: Vec<&FieldDesc> =
            self.fields.iter().filter(|f| f.role == FieldRole::Timestamp).collect();
        match ts.as_slice() {
            [f] => Ok(f),
            [] => bail!(
                "a time_series target needs a timestamp-role field (or set [target].time_field)"
            ),
            _ => bail!(
                "several timestamp-role fields exist; set [target].time_field to pick the ordering field"
            ),
        }
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
        // Fallback for pre-domain configs (empty `fields` map): the generic
        // mechanism groups still resolve; every other field honors its own
        // `default_on`. (The old domain's named field flags used to be matched
        // here — retired now that `fields` is authoritative for every build.)
        match (f.group.as_deref(), f.role) {
            (Some("raw_numerics"), _) => cfg.raw_numerics,
            (_, FieldRole::Coordinates) => cfg.coordinates,
            _ => f.default_on,
        }
    }

    /// Effective vocabulary size for a categorical under `cfg` (None = all
    /// distinct values).
    pub fn effective_top_n(&self, cfg: &crate::FeatureConfig, f: &FieldDesc) -> Option<usize> {
        if let Some(&n) = cfg.vocab_top_n.get(&f.name) {
            return Some(n);
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
        // Sync the domain-generic mechanism toggles from the effective enable
        // set. (The old domain's named field flags used to be mirrored here
        // too — retired; `fields` is the single authoritative enable map.)
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
        let mut extra = std::collections::BTreeMap::new();
        // an instance-defined metric a predictor emitted (hyphenated name)
        extra.insert("alarm_pr_auc".to_string(), serde_json::json!(0.83));
        let m = crate::Metrics {
            n_test: 10,
            mae: Some(1.0),
            rmse: Some(2.0),
            r2: Some(0.5),
            mape: Some(0.2),
            medape: Some(0.1),
            auc: Some(0.77),
            extra,
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
        // instance-defined metric: resolves from `extra`, ranks higher-better,
        // and now always yields a direction (data-driven, no closed set)
        assert_eq!(spec("alarm_pr_auc").extract("alarm_pr_auc", &m), Some(0.83));
        assert!(!spec("alarm_pr_auc").lower_is_better("alarm_pr_auc"));
        assert!(spec("alarm_pr_auc").primary_direction().is_some());
        // a *_loss custom name defaults to lower-is-better
        assert!(spec("val_loss").lower_is_better("val_loss"));
        // absent metric → None
        assert_eq!(s.extract("logloss", &m), None);
        // direction override, matched name-insensitively (hyphen vs underscore)
        let mut over = spec("ROC-AUC");
        over.directions.insert("roc_auc".into(), true);
        assert!(over.lower_is_better("ROC-AUC"));
        // embedded domain: primary resolves, default group size
        let d = Domain::default();
        assert!(d.metrics.primary_direction().is_some());
        assert_eq!(d.metrics.best_models_size(), 12);
    }

    #[test]
    fn time_series_resolves_ordering_field_and_validates() {
        let mut d = Domain::example();
        // Flip the worked example to a forecast: continuous target keeps its
        // transform, and the sole timestamp field orders the split.
        d.target.task = Task::TimeSeries;
        assert_eq!(d.time_field().unwrap().name, "createdAt");
        d.validate().expect("time_series with one timestamp field validates");

        // An explicit but wrong-role time_field is rejected.
        d.target.time_field = Some("price".into());
        assert!(d.validate().is_err());

        // A classification target still cannot carry a transform.
        let mut c = Domain::example();
        c.target.task = Task::Binary;
        assert!(c.validate().is_err()); // example uses log1p
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
