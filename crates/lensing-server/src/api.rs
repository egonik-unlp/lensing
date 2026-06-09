use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use chrono::Utc;
use futures::stream::{self, Stream, StreamExt};
use lensing_core::Manifest;
use serde::Deserialize;
use serde_json::json;
use tokio_stream::wrappers::BroadcastStream;

use crate::best_models;
use crate::definitions::{self, DefError};
use crate::models;
use crate::runs;
use crate::state::{AppState, BuildStatus, JobStatus};

/// Anyhow-backed error → JSON 500/4xx.
pub struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}"))
    }
}

fn not_found(what: &str) -> ApiError {
    ApiError(StatusCode::NOT_FOUND, format!("{what} not found"))
}

fn bad_request(msg: String) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, msg)
}

fn conflict(msg: String) -> ApiError {
    ApiError(StatusCode::CONFLICT, msg)
}

impl From<DefError> for ApiError {
    fn from(e: DefError) -> Self {
        match e {
            DefError::NotFound => not_found("definition"),
            DefError::Conflict(msg) => conflict(msg),
            DefError::Bad(msg) => bad_request(msg),
            DefError::Internal(e) => e.into(),
        }
    }
}

// ---------- API docs ----------

/// `docs/openapi.yaml`, served from disk so spec edits don't need a rebuild.
pub async fn openapi_spec(State(state): State<Arc<AppState>>) -> Result<Response, ApiError> {
    let path = state.root.join("docs/openapi.yaml");
    let bytes = tokio::fs::read(&path).await.map_err(|_| not_found("openapi spec"))?;
    Ok(([(header::CONTENT_TYPE, "application/yaml")], bytes).into_response())
}

/// Swagger UI shell (assets from the CDN) pointed at /api/openapi.yaml.
pub async fn swagger_ui() -> Response {
    const PAGE: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>lensing API docs</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
  <style>body { margin: 0; }</style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.ui = SwaggerUIBundle({
      url: '/api/openapi.yaml',
      dom_id: '#swagger-ui',
      deepLinking: true,
      tryItOutEnabled: true,
    });
  </script>
</body>
</html>"#;
    ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], PAGE).into_response()
}

// ---------- predictors ----------

pub async fn list_predictors(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(json!({ "predictors": state.registry.predictors }))
}

// ---------- datasets ----------

/// Strip the 1536-float PCA mean for API responses; the UI never needs it.
fn manifest_for_api(mut m: Manifest) -> Manifest {
    m.pca.mean = Vec::new();
    m
}

fn load_manifest(state: &AppState, id: &str) -> Option<Manifest> {
    let text = std::fs::read_to_string(state.datasets_dir().join(id).join("manifest.json")).ok()?;
    serde_json::from_str(&text).ok()
}

pub async fn list_datasets(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let mut manifests: Vec<Manifest> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(state.datasets_dir()) {
        for e in entries.flatten() {
            if let Some(id) = e.file_name().to_str() {
                if let Some(m) = load_manifest(&state, id) {
                    manifests.push(manifest_for_api(m));
                }
            }
        }
    }
    manifests.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Json(json!({ "datasets": manifests }))
}

pub async fn get_dataset(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Manifest>, ApiError> {
    load_manifest(&state, &id)
        .map(manifest_for_api)
        .map(Json)
        .ok_or_else(|| not_found("dataset"))
}

pub async fn get_dataset_items(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let path = state.datasets_dir().join(&id).join("items.json");
    let bytes = tokio::fs::read(&path).await.map_err(|_| not_found("dataset items"))?;
    Ok((
        [
            (header::CONTENT_TYPE, "application/json"),
            // items.json is immutable per dataset; let the browser keep it.
            (header::CACHE_CONTROL, "private, max-age=86400"),
        ],
        bytes,
    )
        .into_response())
}

/// Train/test membership as row ids (not indices), so clients can join
/// directly with `items.json` without knowing the on-disk row order.
pub async fn get_dataset_split(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let dir = state.datasets_dir().join(&id);
    if !dir.join("manifest.json").is_file() {
        return Err(not_found("dataset"));
    }
    let body = tokio::task::spawn_blocking(move || -> anyhow::Result<serde_json::Value> {
        let row_ids = lensing_core::artifact::read_u64(&dir.join("row_ids.u64"))?;
        let resolve = |idx: Vec<u32>| -> anyhow::Result<Vec<u64>> {
            idx.iter()
                .map(|&i| {
                    row_ids
                        .get(i as usize)
                        .copied()
                        .ok_or_else(|| anyhow::anyhow!("split index {i} out of range"))
                })
                .collect()
        };
        let train = resolve(lensing_core::artifact::read_u32(&dir.join("train_idx.u32"))?)?;
        let test = resolve(lensing_core::artifact::read_u32(&dir.join("test_idx.u32"))?)?;
        Ok(json!({ "train": train, "test": test }))
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("split read panicked: {e}")))??;
    Ok((
        [
            (header::CONTENT_TYPE, "application/json"),
            // Split membership is immutable per dataset, same as items.json.
            (header::CACHE_CONTROL, "private, max-age=86400"),
        ],
        body.to_string(),
    )
        .into_response())
}

#[derive(Deserialize)]
pub struct BuildRequest {
    #[serde(default = "default_pca_dims")]
    pub pca_dims: usize,
    #[serde(default = "default_test_ratio")]
    pub test_ratio: f64,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_true")]
    pub log_target: bool,
    /// Generic per-field enables, keyed by domain field name (or toggle-group
    /// name). When non-empty, authoritative; the legacy named flags below are
    /// then ignored (kept for pre-domain clients).
    #[serde(default)]
    pub fields: std::collections::BTreeMap<String, bool>,
    /// Per-categorical vocabulary-size overrides (field name → top-N).
    #[serde(default)]
    pub vocab_top_n: std::collections::BTreeMap<String, usize>,
    #[serde(default = "default_true")]
    pub bedrooms: bool,
    #[serde(default = "default_true")]
    pub property_type: bool,
    #[serde(default = "default_top_n")]
    pub neighborhood_top_n: usize,
    #[serde(default)]
    pub city: bool,
    #[serde(default)]
    pub province: bool,
    #[serde(default)]
    pub cluster: bool,
    /// Quality filters (rule toggles + thresholds); defaults exclude only
    /// nonpositive target values.
    #[serde(default)]
    pub quality: lensing_core::QualityFilterConfig,
    /// Currency handling (reconcile / hard filter / convert); the default
    /// reconciles from the domain's companion collection and hard-filters
    /// to the kept currency.
    #[serde(default)]
    pub currency: lensing_core::CurrencyConfig,
    /// Source Qdrant collection to build/analyze from. Defaults to the
    /// server's configured collection; set to build off an exported clean one.
    #[serde(default)]
    pub collection: Option<String>,
    /// Raw-metadata numeric features (the domain's reconcile-flagged
    /// numeric fields), reconciled from `numerics_collection` by point id.
    #[serde(default)]
    pub raw_numerics: bool,
    /// With `raw_numerics`: backfill missing area-like fields from unit
    /// mentions (e.g. "… m²") in the entry text.
    #[serde(default)]
    pub area_content_backfill: bool,
    /// Lat/lon from the raw collection's coordinates field (raw-degree
    /// columns + pair-missing indicator), reconciled like the numerics.
    #[serde(default)]
    pub coordinates: bool,
    /// With `raw_numerics`: fill missing numeric fields with per-group
    /// train-split medians (grouped by the domain's outlier group) and drop
    /// the missing-indicator columns.
    #[serde(default)]
    pub impute_numerics: bool,
    /// Companion collection for the raw-numerics reconcile join. Absent →
    /// the domain's companion collection; empty string → no join (inline
    /// fields only).
    #[serde(default)]
    pub numerics_collection: Option<String>,
}

/// Companion collection for a request: explicit value, `""` disables the
/// join, absent falls back to the domain's companion collection.
fn resolve_numerics_collection(state: &AppState, requested: &Option<String>) -> Option<String> {
    match requested {
        Some(c) if c.is_empty() => None,
        Some(c) => Some(c.clone()),
        None => state.domain.companion_collection(),
    }
}

/// The source collection a request targets, defaulting to the server's.
fn resolve_collection(state: &AppState, requested: &Option<String>) -> String {
    requested.clone().unwrap_or_else(|| state.collection.clone())
}

fn default_pca_dims() -> usize { 32 }
fn default_test_ratio() -> f64 { 0.2 }
fn default_seed() -> u64 { 42 }
fn default_true() -> bool { true }
fn default_top_n() -> usize { 40 }

impl BuildRequest {
    /// Feature config from the request (generic maps + legacy flags); the
    /// build resolves it against the domain via `normalize_config`.
    fn feature_config(&self, state: &AppState) -> Result<lensing_core::FeatureConfig, ApiError> {
        // Unknown field names are caller typos, not silently-ignored toggles.
        for name in self.fields.keys().chain(self.vocab_top_n.keys()) {
            let known = state.domain.field(name).is_some()
                || state.domain.fields.iter().any(|f| f.group.as_deref() == Some(name));
            if !known {
                return Err(bad_request(format!("unknown field {name:?} (see GET /api/domain)")));
            }
        }
        Ok(lensing_core::FeatureConfig {
            pca_dims: self.pca_dims,
            fields: self.fields.clone(),
            vocab_top_n: self.vocab_top_n.clone(),
            coordinate_bounds: None, // frozen in by normalize_config
            bedrooms: self.bedrooms,
            property_type: self.property_type,
            neighborhood_top_n: self.neighborhood_top_n,
            city: self.city,
            province: self.province,
            cluster: self.cluster,
            raw_numerics: self.raw_numerics,
            area_content_backfill: self.area_content_backfill,
            coordinates: self.coordinates,
            impute_numerics: self.impute_numerics,
        })
    }
}

pub async fn build_dataset(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BuildRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if req.pca_dims < 1 || req.pca_dims > 1536 {
        return Err(bad_request("pca_dims must be in 1..=1536".into()));
    }
    if !(0.05..=0.5).contains(&req.test_ratio) {
        return Err(bad_request("test_ratio must be in 0.05..=0.5".into()));
    }
    if req.impute_numerics && !req.raw_numerics {
        return Err(bad_request("impute_numerics requires raw_numerics".into()));
    }

    let build_id = format!("build-{}", runs::new_run_id("ds"));
    state
        .builds
        .lock()
        .unwrap()
        .insert(build_id.clone(), BuildStatus::Building { stage: "queued".into() });

    let features = req.feature_config(&state)?;
    let cfg = lensing_pipeline::BuildConfig {
        domain: state.domain.as_ref().clone(),
        qdrant_url: state.qdrant_url.clone(),
        collection: resolve_collection(&state, &req.collection),
        out_root: state.datasets_dir(),
        test_ratio: req.test_ratio,
        seed: req.seed,
        log_target: req.log_target,
        features,
        quality: req.quality,
        currency: req.currency,
        numerics_collection: resolve_numerics_collection(&state, &req.numerics_collection),
    };

    let st = state.clone();
    let id = build_id.clone();
    tokio::spawn(async move {
        let _permit = st.build_slots.clone().acquire_owned().await.unwrap();
        let st2 = st.clone();
        let id2 = id.clone();
        let result = tokio::task::spawn_blocking(move || {
            lensing_pipeline::build_dataset(&cfg, &move |stage: &str| {
                st2.builds
                    .lock()
                    .unwrap()
                    .insert(id2.clone(), BuildStatus::Building { stage: stage.to_string() });
            })
        })
        .await;
        let status = match result {
            Ok(Ok(manifest)) => {
                if let (Some(sink), Ok(full)) = (&st.db_sink, serde_json::to_value(&manifest)) {
                    sink.dataset_upsert(&manifest.dataset_id, Some(manifest.created_at.clone()), full);
                }
                BuildStatus::Done { dataset_id: manifest.dataset_id }
            }
            Ok(Err(e)) => BuildStatus::Failed { error: format!("{e:#}") },
            Err(e) => BuildStatus::Failed { error: format!("build task panicked: {e}") },
        };
        st.builds.lock().unwrap().insert(id, status);
    });

    Ok(Json(json!({ "build_id": build_id })))
}

pub async fn get_build(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<BuildStatus>, ApiError> {
    state
        .builds
        .lock()
        .unwrap()
        .get(&id)
        .cloned()
        .map(Json)
        .ok_or_else(|| not_found("build"))
}

pub async fn get_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<JobStatus>, ApiError> {
    state
        .jobs
        .lock()
        .unwrap()
        .get(&id)
        .cloned()
        .map(Json)
        .ok_or_else(|| not_found("job"))
}

#[derive(Deserialize)]
pub struct RenameDatasetRequest {
    pub name: String,
}

/// Validate a human display name (not the slug `dataset_id`): trimmed,
/// non-empty, bounded, no control characters.
fn validate_display_name(name: &str) -> Result<(), ApiError> {
    if name.is_empty() {
        return Err(bad_request("name must not be empty".into()));
    }
    if name.chars().count() > 80 {
        return Err(bad_request("name must be at most 80 characters".into()));
    }
    if name.chars().any(|c| c.is_control()) {
        return Err(bad_request("name must not contain control characters".into()));
    }
    Ok(())
}

/// Set a dataset's display name. The slug `dataset_id` (and the directory)
/// stay put, so run/definition lineage is unaffected.
pub async fn rename_dataset(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<RenameDatasetRequest>,
) -> Result<Json<Manifest>, ApiError> {
    let name = req.name.trim();
    validate_display_name(name)?;
    // Read the FULL manifest (with pca.mean): we write it straight back, so it
    // must never have passed through `manifest_for_api`.
    let mut m = load_manifest(&state, &id).ok_or_else(|| not_found("dataset"))?;
    m.name = Some(name.to_string());
    let path = state.datasets_dir().join(&id).join("manifest.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&m).map_err(anyhow::Error::from)?)
        .map_err(anyhow::Error::from)?;
    if let (Some(sink), Ok(full)) = (&state.db_sink, serde_json::to_value(&m)) {
        sink.dataset_upsert(&m.dataset_id, Some(m.created_at.clone()), full);
    }
    Ok(Json(manifest_for_api(m)))
}

/// Async analyze job: the full eigen-spectrum (EVR preview) + the feature
/// redundancy report, sharing the build's expensive prologue (scroll → quality
/// → split). Returns a `job_id`; poll `GET /api/jobs/{id}`.
pub async fn analyze_dataset(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BuildRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if !(0.05..=0.5).contains(&req.test_ratio) {
        return Err(bad_request("test_ratio must be in 0.05..=0.5".into()));
    }

    let job_id = format!("analyze-{}", runs::new_run_id("an"));
    state
        .jobs
        .lock()
        .unwrap()
        .insert(job_id.clone(), JobStatus::Running { stage: "queued".into() });

    let qdrant_url = state.qdrant_url.clone();
    let collection = resolve_collection(&state, &req.collection);
    let test_ratio = req.test_ratio;
    let seed = req.seed;
    let quality = req.quality.clone();
    let currency = req.currency.clone();
    let mut features = req.feature_config(&state)?;
    let domain = state.domain.as_ref().clone();
    domain.normalize_config(&mut features);
    let numerics_collection = resolve_numerics_collection(&state, &req.numerics_collection);

    let st = state.clone();
    let id = job_id.clone();
    tokio::spawn(async move {
        let _permit = st.build_slots.clone().acquire_owned().await.unwrap();
        let st2 = st.clone();
        let id2 = id.clone();
        let set_stage = move |stage: String| {
            st2.jobs.lock().unwrap().insert(id2.clone(), JobStatus::Running { stage });
        };
        let result = tokio::task::spawn_blocking(move || -> anyhow::Result<serde_json::Value> {
            use lensing_pipeline::{features::Encoder, pca, qdrant, quality as q, redundancy};
            // Hard shape checks fail here with a clear message, not mid-scroll.
            set_stage("checking collection".into());
            qdrant::ensure_buildable(&qdrant_url, &collection, &domain)?;
            set_stage("fetching points".into());
            let s = set_stage.clone();
            let mut points = qdrant::scroll_all(&qdrant_url, &collection, &domain, &move |n| {
                s(format!("fetching points {n}"))
            })?;
            anyhow::ensure!(!points.is_empty(), "filter matches no points");

            // Same prologue as the build: reconcile/convert currency first so
            // the preview matches what a build would produce.
            let s = set_stage.clone();
            lensing_pipeline::currency::apply(
                &mut points,
                &currency,
                &qdrant_url,
                &domain,
                &move |stage: &str| s(stage.into()),
            )?;

            set_stage("applying quality filters".into());
            let analysis = q::evaluate(&points, &quality, &currency, &domain);
            let excluded = analysis.excluded(&quality);
            let n_excluded = excluded.len();
            if !excluded.is_empty() {
                let mut i = 0usize;
                points.retain(|_| {
                    let keep = !excluded.contains(&i);
                    i += 1;
                    keep
                });
            }
            anyhow::ensure!(points.len() > 1, "quality filters excluded the whole corpus");

            // Same as the build: reconcile raw numerics before encoding so
            // the redundancy preview sees the columns a build would produce.
            let s = set_stage.clone();
            lensing_pipeline::numerics::apply(
                &mut points,
                &features,
                &numerics_collection,
                &qdrant_url,
                &domain,
                &move |stage: &str| s(stage.into()),
            )?;

            let n = points.len();
            let d = points[0].vector.len();
            let (train_idx, _test_idx) = lensing_core::shuffle::train_test_split(n, test_ratio, seed);

            set_stage("computing eigen-spectrum (train rows)".into());
            let mut vectors = Vec::with_capacity(n * d);
            for p in &points {
                vectors.extend_from_slice(&p.vector);
            }
            let spectrum = pca::eigen_spectrum(&vectors, d, &train_idx)?;
            drop(vectors);

            set_stage("analyzing feature redundancy".into());
            let encoder = Encoder::build(&points, &domain, &features);
            let report = redundancy::analyze(&points, &encoder);

            Ok(json!({
                "evr": spectrum.evr(),
                "cumulative_evr": spectrum.cumulative_evr(),
                "redundancy": report,
                "n_rows": n,
                "n_excluded": n_excluded,
            }))
        })
        .await;
        let status = match result {
            Ok(Ok(result)) => JobStatus::Done { result },
            Ok(Err(e)) => JobStatus::Failed { error: format!("{e:#}") },
            Err(e) => JobStatus::Failed { error: format!("analyze task panicked: {e}") },
        };
        st.jobs.lock().unwrap().insert(id, status);
    });

    Ok(Json(json!({ "job_id": job_id })))
}

// ---------- collections (export clean subset + source selection) ----------

/// Every Qdrant collection, for the build form's source picker. Arbitrary
/// collections are allowed as sources; /collections/validate reports whether
/// one actually fits the product's expected shape.
pub async fn list_collections(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let qdrant_url = state.qdrant_url.clone();
    let source = state.collection.clone();
    let mut collections = tokio::task::spawn_blocking(move || lensing_pipeline::qdrant::list_collections(&qdrant_url))
        .await
        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("list collections panicked: {e}")))??;
    collections.sort();
    Ok(Json(json!({ "collections": collections, "source": source })))
}

#[derive(Deserialize)]
pub struct ValidateCollectionRequest {
    /// Collection to probe; defaults to the server's.
    #[serde(default)]
    pub collection: Option<String>,
}

/// Probe a collection's shape (vector config, filter match count, payload
/// field coverage). Shape problems come back as data in the report, always
/// 200; only transport failures error.
pub async fn validate_collection(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ValidateCollectionRequest>,
) -> Result<Json<lensing_pipeline::qdrant::CollectionValidation>, ApiError> {
    let qdrant_url = state.qdrant_url.clone();
    let collection = resolve_collection(&state, &req.collection);
    let domain = state.domain.clone();
    let report = tokio::task::spawn_blocking(move || {
        lensing_pipeline::qdrant::validate_collection(&qdrant_url, &collection, &domain)
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("validate collection panicked: {e}")))??;
    Ok(Json(report))
}

#[derive(Deserialize)]
pub struct ExportRequest {
    /// Slug appended to `<source-collection>-clean-`.
    pub name_suffix: String,
    #[serde(default)]
    pub quality: lensing_core::QualityFilterConfig,
    /// Currency handling. Reconciled currency (and Convert-mode target values) are
    /// stamped into the exported payloads, so clean collections are
    /// self-contained.
    #[serde(default)]
    pub currency: lensing_core::CurrencyConfig,
    /// Source collection to filter from; defaults to the server's.
    #[serde(default)]
    pub source: Option<String>,
}

/// Write quality-filter survivors to a NEW Qdrant collection, source untouched.
/// Returns a `job_id`; poll `GET /api/jobs/{id}`.
pub async fn export_collection(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExportRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if req.name_suffix.len() > 40 {
        return Err(bad_request("name suffix must be at most 40 characters".into()));
    }
    models::validate_name(&req.name_suffix).map_err(|e| bad_request(format!("{e:#}")))?;
    let source = resolve_collection(&state, &req.source);
    let target = format!("{source}-clean-{}", req.name_suffix);
    if target == source {
        return Err(bad_request("refusing to export onto the source collection".into()));
    }

    // 409 if the target already exists (get_collection_params succeeds).
    let probe_url = state.qdrant_url.clone();
    let probe_target = target.clone();
    let exists = tokio::task::spawn_blocking(move || {
        lensing_pipeline::qdrant::get_collection_params(&probe_url, &probe_target).is_ok()
    })
    .await
    .unwrap_or(false);
    if exists {
        return Err(conflict(format!("collection {target} already exists")));
    }

    let job_id = format!("export-{}", runs::new_run_id("ex"));
    state
        .jobs
        .lock()
        .unwrap()
        .insert(job_id.clone(), JobStatus::Running { stage: "queued".into() });

    let qdrant_url = state.qdrant_url.clone();
    let quality = req.quality.clone();
    let currency = req.currency.clone();
    let domain = state.domain.clone();
    let st = state.clone();
    let id = job_id.clone();
    tokio::spawn(async move {
        let _permit = st.build_slots.clone().acquire_owned().await.unwrap();
        let st2 = st.clone();
        let id2 = id.clone();
        let set_stage = move |stage: String| {
            st2.jobs.lock().unwrap().insert(id2.clone(), JobStatus::Running { stage });
        };
        let target_for_job = target.clone();
        let result = tokio::task::spawn_blocking(move || -> anyhow::Result<serde_json::Value> {
            use lensing_pipeline::qdrant;
            set_stage("reading source params".into());
            let (size, distance) = qdrant::get_collection_params(&qdrant_url, &source)?;

            set_stage("fetching source points".into());
            let s = set_stage.clone();
            let raw = qdrant::scroll_all_raw(&qdrant_url, &source, &domain, &move |n| {
                s(format!("fetching source points {n}"))
            })?;
            let n_source = raw.len();

            set_stage("applying quality filters".into());
            // Flattened view (vectors unneeded) just to run the quality
            // rules; the raw payloads are what actually get written back.
            let mut typed: Vec<qdrant::RawPoint> = raw
                .iter()
                .map(|p| qdrant::RawPoint {
                    id: p.id,
                    vector: Vec::new(),
                    payload: qdrant::Payload::from_raw(
                        p.payload.clone(),
                        &domain.corpus.metadata_root,
                    ),
                })
                .collect();
            let s = set_stage.clone();
            lensing_pipeline::currency::apply(
                &mut typed,
                &currency,
                &qdrant_url,
                &domain,
                &move |stage: &str| s(stage.into()),
            )?;
            let analysis = lensing_pipeline::quality::evaluate(&typed, &quality, &currency, &domain);
            let excluded = analysis.excluded(&quality);
            let n_excluded = excluded.len();
            // Stamp the reconciled currency (and Convert-mode target values)
            // into the exported payloads: the clean collection is
            // self-contained. Field paths follow the domain's metadata root.
            let convert = currency.mode == lensing_core::CurrencyMode::Convert;
            let root = domain.corpus.metadata_root.clone();
            let stamp = |payload: &mut serde_json::Value, key: &str, value: serde_json::Value| {
                let slot = if root.is_empty() { &mut *payload } else { &mut payload[root.as_str()] };
                slot[key] = value;
            };
            let survivors: Vec<qdrant::RawJsonPoint> = raw
                .into_iter()
                .zip(&typed)
                .enumerate()
                .filter(|(i, _)| !excluded.contains(i))
                .map(|(_, (mut p, t))| {
                    if let Some(dc) = &domain.currency {
                        let c = t.payload.get(&dc.currency_field).clone();
                        if c.is_string() {
                            stamp(&mut p.payload, &dc.currency_field, c);
                            if convert {
                                stamp(
                                    &mut p.payload,
                                    &domain.target.field,
                                    t.payload.get(&domain.target.field).clone(),
                                );
                            }
                        }
                    }
                    p
                })
                .collect();
            let n_written = survivors.len();

            set_stage(format!("creating {target_for_job}"));
            qdrant::create_collection(&qdrant_url, &target_for_job, size, &distance)?;

            set_stage(format!("writing {n_written} points"));
            if let Err(e) = qdrant::upsert_raw(&qdrant_url, &target_for_job, &survivors) {
                let _ = qdrant::delete_collection(&qdrant_url, &target_for_job);
                return Err(e);
            }

            Ok(json!({
                "collection": target_for_job,
                "n_source": n_source,
                "n_excluded": n_excluded,
                "n_written": n_written,
            }))
        })
        .await;
        let status = match result {
            Ok(Ok(result)) => JobStatus::Done { result },
            Ok(Err(e)) => JobStatus::Failed { error: format!("{e:#}") },
            Err(e) => JobStatus::Failed { error: format!("export task panicked: {e}") },
        };
        st.jobs.lock().unwrap().insert(id, status);
    });

    Ok(Json(json!({ "job_id": job_id })))
}

// ---------- runs ----------

pub async fn list_runs(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let mut metas = Vec::new();
    let mut seen = std::collections::HashSet::new();
    // Local run dirs first: a just-started local run reaches the file before
    // the async database mirror catches up.
    if let Ok(entries) = std::fs::read_dir(state.runs_dir()) {
        for e in entries.flatten() {
            if let Ok(meta) = runs::read_meta(&e.path()) {
                seen.insert(meta.run_id.clone());
                metas.push(meta);
            }
        }
    }
    // Database rows cover queued + remote-worker runs that have no local dir.
    if let Some(db) = &state.db {
        if let Ok(rows) = lensing_db::queries::load_runs(db).await {
            for meta in rows {
                if !seen.contains(&meta.run_id) {
                    metas.push(meta);
                }
            }
        }
    }
    metas.sort_by(|a, b| b.started_at.cmp(&a.started_at));
    Json(json!({ "runs": metas }))
}

pub async fn get_run(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<lensing_core::RunMeta>, ApiError> {
    if let Ok(meta) = runs::read_meta(&state.runs_dir().join(&id)) {
        return Ok(Json(meta));
    }
    if let Some(db) = &state.db {
        if let Ok(Some(meta)) = lensing_db::queries::load_run(db, &id).await {
            return Ok(Json(meta));
        }
    }
    Err(not_found("run"))
}

#[derive(Deserialize)]
pub struct StartRunRequest {
    pub dataset_id: String,
    /// Required unless `definition` is set (then it may override).
    #[serde(default)]
    pub predictor: Option<String>,
    /// Launch from a model definition: its predictor + hyperparams are the
    /// base; request hyperparams overlay on top.
    #[serde(default)]
    pub definition: Option<String>,
    /// Enqueue for a remote training worker instead of training locally
    /// (status `queued` until a `lensing-server worker` claims it).
    #[serde(default)]
    pub queue: bool,
    #[serde(default)]
    pub hyperparams: Option<serde_json::Value>,
}

pub async fn start_run(
    State(state): State<Arc<AppState>>,
    Json(req): Json<StartRunRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Resolve predictor + base hyperparams from the definition, if any.
    let (predictor_name, base_hp) = match &req.definition {
        Some(def_name) => {
            let def = definitions::get(&state, def_name).await.ok_or_else(|| not_found("definition"))?;
            (req.predictor.clone().unwrap_or(def.predictor), Some(def.hyperparams))
        }
        None => {
            let name = req
                .predictor
                .clone()
                .ok_or_else(|| bad_request("predictor or definition is required".into()))?;
            (name, None)
        }
    };
    let predictor = state
        .registry
        .get(&predictor_name)
        .ok_or_else(|| bad_request(format!("unknown predictor {predictor_name}")))?;

    // Request hyperparams overlay the definition's, then merge over defaults.
    let submitted = match (base_hp, req.hyperparams) {
        (Some(serde_json::Value::Object(mut base)), Some(serde_json::Value::Object(over))) => {
            base.extend(over);
            Some(serde_json::Value::Object(base))
        }
        (Some(base), None) => Some(base),
        (None, over) => over,
        (base, Some(_)) => base, // non-object overlay: ignore, merge will validate
    };
    let hp = predictor
        .merge_hyperparams(submitted)
        .map_err(|e| bad_request(format!("{e:#}")))?;

    let run_id = if req.queue {
        runs::enqueue_run(
            &state,
            req.dataset_id.clone(),
            predictor_name,
            hp,
            req.definition.clone(),
        )
        .await
        .map_err(|e| bad_request(format!("{e:#}")))?
    } else {
        runs::start_run(
            state.clone(),
            req.dataset_id.clone(),
            predictor_name,
            hp,
            req.definition.clone(),
        )
        .map_err(|e| bad_request(format!("{e:#}")))?
    };

    // Record that the definition has been used with this dataset.
    if let Some(def_name) = &req.definition {
        if let Err(e) = definitions::add_dataset_tag(&state, def_name, &req.dataset_id).await {
            let e: ApiError = e.into();
            eprintln!("[lensing-server] failed to tag dataset on definition {def_name}: {}", e.1);
        }
    }
    Ok(Json(json!({ "run_id": run_id })))
}

/// Delete a finished run's directory. Running runs must be stopped first;
/// models promoted from the run keep working (promotion copies the files).
pub async fn delete_run(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if state.live_runs.lock().unwrap().contains_key(&id) {
        return Err(conflict(format!("run {id} is still running; stop it first")));
    }
    // Belt and braces with the live_runs check: start_run writes the
    // `running` meta just before registering the handle, so a delete landing
    // in that window (or against a zombie the sweep hasn't seen) is caught
    // by the on-disk status.
    if let Ok(meta) = runs::read_meta(&state.runs_dir().join(&id)) {
        if meta.status == lensing_core::RunStatus::Running {
            return Err(conflict(format!("run {id} is still running; stop it first")));
        }
    }
    if state.runs_dir().join(&id).join("meta.json").is_file() {
        runs::delete_run(&state, &id).map_err(|_| not_found("run"))?;
        return Ok(Json(json!({ "ok": true })));
    }
    // Database-only run (queued, or executed by a remote worker).
    if let Some(db) = &state.db {
        match lensing_db::queries::load_run(db, &id).await {
            Ok(Some(meta)) if meta.status == lensing_core::RunStatus::Running => {
                return Err(conflict(format!("run {id} is running on a worker; let it finish")));
            }
            Ok(Some(_)) => {
                lensing_db::queries::delete_run(db, &id).await.map_err(ApiError::from)?;
                return Ok(Json(json!({ "ok": true })));
            }
            _ => {}
        }
    }
    Err(not_found("run"))
}

/// A run-dir file, falling back to the database-uploaded artifact for runs
/// executed by a remote worker (no local dir).
async fn run_file(state: &AppState, id: &str, name: &str) -> Option<Vec<u8>> {
    let path = state.runs_dir().join(id).join(name);
    if let Ok(bytes) = tokio::fs::read(&path).await {
        return Some(bytes);
    }
    let db = state.db.as_ref()?;
    lensing_db::queries::load_run_artifact(db, id, name).await.ok().flatten()
}

pub async fn get_predictions(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let bytes = run_file(&state, &id, "predictions.json")
        .await
        .ok_or_else(|| not_found("predictions"))?;
    Ok((
        [
            (header::CONTENT_TYPE, "application/json"),
            (header::CACHE_CONTROL, "private, max-age=86400"),
        ],
        bytes,
    )
        .into_response())
}

// ---------- blends ----------

#[derive(Deserialize)]
pub struct BlendRequest {
    /// Finished runs whose test-split predictions are combined. All runs
    /// must share the same test split (same dataset seed and test_ratio).
    pub run_ids: Vec<String>,
    /// Per-run blend weights, parallel to `run_ids`; equal weights when
    /// omitted. Normalized to sum to 1.
    pub weights: Option<Vec<f64>>,
    /// Two runs only: also sweep the first run's weight over 0..=1 in steps
    /// of 0.05 and report metrics per point plus the argmin-MAE weight.
    #[serde(default)]
    pub sweep: bool,
}

fn load_run_predictions(state: &AppState, id: &str) -> Result<Vec<lensing_core::Prediction>, ApiError> {
    let path = state.runs_dir().join(id).join("predictions.json");
    let bytes =
        std::fs::read(&path).map_err(|_| not_found(&format!("predictions for run {id}")))?;
    serde_json::from_slice(&bytes)
        .map_err(|e| bad_request(format!("run {id}: malformed predictions.json: {e}")))
}

/// Pearson correlation between two legs' residuals (predicted − actual).
fn residual_correlation(a: &[lensing_core::Prediction], b: &[lensing_core::Prediction]) -> f64 {
    let n = a.len() as f64;
    let ea: Vec<f64> = a.iter().map(|p| p.predicted - p.actual).collect();
    let eb: Vec<f64> = b.iter().map(|p| p.predicted - p.actual).collect();
    let (ma, mb) = (ea.iter().sum::<f64>() / n, eb.iter().sum::<f64>() / n);
    let cov = ea.iter().zip(&eb).map(|(x, y)| (x - ma) * (y - mb)).sum::<f64>();
    let (va, vb) = (
        ea.iter().map(|x| (x - ma).powi(2)).sum::<f64>(),
        eb.iter().map(|y| (y - mb).powi(2)).sum::<f64>(),
    );
    cov / (va.sqrt() * vb.sqrt())
}

fn blend_metrics(
    legs: &[Vec<lensing_core::Prediction>],
    weights: &[f64],
    task: lensing_core::domain::Task,
) -> lensing_core::Metrics {
    let pairs: Vec<(f64, f64)> = (0..legs[0].len())
        .map(|i| {
            let blended: f64 =
                legs.iter().zip(weights).map(|(l, w)| w * l[i].predicted).sum();
            (legs[0][i].actual, blended)
        })
        .collect();
    // Binary blends P(class==1); score by classification metrics, not MAE.
    if task == lensing_core::domain::Task::Binary {
        lensing_core::compute_binary_metrics(&pairs)
    } else {
        lensing_core::compute_metrics(&pairs)
    }
}

/// Blend the test-split predictions of two or more finished runs and report
/// price-space metrics for the combination, each leg alone, and the pairwise
/// residual correlations. Pure post-processing: nothing is trained or written.
pub async fn blend_runs(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BlendRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if req.run_ids.len() < 2 {
        return Err(bad_request("need at least 2 run_ids to blend".into()));
    }
    let weights = match &req.weights {
        Some(w) => {
            if w.len() != req.run_ids.len() {
                return Err(bad_request("weights must be parallel to run_ids".into()));
            }
            if w.iter().any(|x| !x.is_finite() || *x < 0.0) || w.iter().sum::<f64>() <= 0.0 {
                return Err(bad_request("weights must be non-negative with a positive sum".into()));
            }
            let sum: f64 = w.iter().sum();
            w.iter().map(|x| x / sum).collect::<Vec<_>>()
        }
        None => vec![1.0 / req.run_ids.len() as f64; req.run_ids.len()],
    };

    // Scalar blending of `predicted` works for regression (target value) and
    // binary (P(class==1)); multiclass `predicted` is an argmax class id that
    // can't be linearly combined — refuse it.
    let task = state.domain.target.task;
    if task == lensing_core::domain::Task::Multiclass {
        return Err(bad_request(
            "blend endpoint does not support multiclass targets (probability-vector blending pending)".into(),
        ));
    }

    let mut legs: Vec<Vec<lensing_core::Prediction>> = Vec::with_capacity(req.run_ids.len());
    for id in &req.run_ids {
        let mut preds = load_run_predictions(&state, id)?;
        preds.sort_by_key(|p| p.row_id);
        legs.push(preds);
    }

    // All legs must cover the identical test split: same row_ids, same actuals.
    // A mismatch means different dataset seed/ratio (or target) — refuse rather
    // than silently blend the intersection.
    let first = &legs[0];
    for (k, leg) in legs.iter().enumerate().skip(1) {
        let ids: std::collections::HashSet<u64> = leg.iter().map(|p| p.row_id).collect();
        let first_ids: std::collections::HashSet<u64> = first.iter().map(|p| p.row_id).collect();
        if ids != first_ids {
            let only_first = first_ids.difference(&ids).count();
            let only_this = ids.difference(&first_ids).count();
            return Err(bad_request(format!(
                "test splits differ: {} rows only in {}, {} rows only in {} — runs must share a dataset seed/ratio",
                only_first, req.run_ids[0], only_this, req.run_ids[k],
            )));
        }
        for (a, b) in first.iter().zip(leg) {
            if (a.actual - b.actual).abs() > 1e-6 * a.actual.abs().max(1.0) {
                return Err(bad_request(format!(
                    "row {} has different actuals in {} and {} — runs target different data",
                    a.row_id, req.run_ids[0], req.run_ids[k],
                )));
            }
        }
    }

    let leg_summaries: Vec<serde_json::Value> = legs
        .iter()
        .zip(&req.run_ids)
        .zip(&weights)
        .map(|((leg, id), w)| {
            let pairs: Vec<(f64, f64)> = leg.iter().map(|p| (p.actual, p.predicted)).collect();
            let m = if task == lensing_core::domain::Task::Binary {
                lensing_core::compute_binary_metrics(&pairs)
            } else {
                lensing_core::compute_metrics(&pairs)
            };
            json!({ "run_id": id, "weight": w, "metrics": m })
        })
        .collect();

    let correlations: Vec<Vec<f64>> = (0..legs.len())
        .map(|i| {
            (0..legs.len())
                .map(|j| if i == j { 1.0 } else { residual_correlation(&legs[i], &legs[j]) })
                .collect()
        })
        .collect();

    let sweep = if req.sweep {
        if legs.len() != 2 {
            return Err(bad_request("sweep requires exactly 2 run_ids".into()));
        }
        // Select the sweep optimum by the domain's PRIMARY metric + direction
        // (e.g. min MAE for regression, max AUC for binary), not a hardcoded MAE.
        let primary = state.domain.metrics.primary.clone();
        let lower_wins = state.domain.metrics.lower_is_better(&primary);
        let scored: Vec<(f64, Option<f64>, lensing_core::Metrics)> = (0..=20)
            .map(|step| {
                let w = step as f64 * 0.05;
                let m = blend_metrics(&legs, &[w, 1.0 - w], task);
                let key = state.domain.metrics.extract(&primary, &m);
                (w, key, m)
            })
            .collect();
        let points: Vec<serde_json::Value> =
            scored.iter().map(|(w, _, m)| json!({ "w": w, "metrics": m })).collect();
        let best = scored
            .iter()
            .filter(|(_, k, _)| k.is_some())
            .min_by(|a, b| {
                let (va, vb) = (a.1.unwrap(), b.1.unwrap());
                if lower_wins { va.total_cmp(&vb) } else { vb.total_cmp(&va) }
            })
            .map(|(w, _, m)| json!({ "w": w, "metrics": m }));
        Some(json!({ "points": points, "best": best }))
    } else {
        None
    };

    Ok(Json(json!({
        "n_test": legs[0].len(),
        "blend": blend_metrics(&legs, &weights, task),
        "legs": leg_summaries,
        "residual_correlation": correlations,
        "sweep": sweep,
    })))
}

/// How long a force stop waits for a STOP-honoring predictor to finish the
/// epoch, evaluate and save before the kill fires. Generous enough to cover
/// flux-mlp's ~15 s JIT warmup plus an epoch and the eval pass.
const FORCE_STOP_GRACE: std::time::Duration = std::time::Duration::from_secs(60);

#[derive(Deserialize)]
pub struct StopRunRequest {
    /// false = graceful (STOP file; predictor finishes the current epoch,
    /// evaluates and checkpoints). true = stop-and-save with escalation:
    /// kill at `FORCE_STOP_GRACE` if the process is still alive (immediate
    /// for predictors that ignore STOP, or on a repeated force request).
    #[serde(default)]
    pub force: bool,
}

pub async fn stop_run(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<StopRunRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let handle = state
        .live_runs
        .lock()
        .unwrap()
        .get(&id)
        .cloned()
        .ok_or_else(|| conflict(format!("run {id} is not running")))?;

    // Both modes drop the STOP marker and flag the stop: if a force-killed
    // child manages a clean exit anyway, the run still reads as `stopped`.
    let run_dir = state.runs_dir().join(&id);
    std::fs::write(run_dir.join(runs::STOP_FILE), b"").map_err(anyhow::Error::from)?;
    handle.request_stop();
    let mode = if !req.force {
        "graceful"
    } else {
        // Force is still a save-first stop: predictors that honor STOP get a
        // grace window to finish the epoch, evaluate and write the model;
        // the kill fires only if the process is still alive at the deadline.
        // Predictors that ignore STOP — and a repeated force request — are
        // killed immediately.
        let honors_stop = runs::read_meta(&run_dir)
            .ok()
            .and_then(|m| state.registry.get(&m.predictor).map(|p| p.supports_stop))
            .unwrap_or(false);
        if !honors_stop || !handle.set_force_requested() {
            handle.force_kill();
            "kill"
        } else {
            let state = state.clone();
            let id = id.clone();
            tokio::spawn(async move {
                tokio::time::sleep(FORCE_STOP_GRACE).await;
                let live = state.live_runs.lock().unwrap().get(&id).cloned();
                if let Some(h) = live {
                    h.force_kill();
                }
            });
            "force"
        }
    };
    Ok(Json(json!({ "ok": true, "mode": mode })))
}

/// Architecture SVG written by the predictor at training start (optional,
/// capability-gated by `visualization` in registry.toml).
pub async fn get_run_viz(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let bytes = run_file(&state, &id, "viz.svg")
        .await
        .ok_or_else(|| not_found("visualization"))?;
    Ok((
        [
            (header::CONTENT_TYPE, "image/svg+xml"),
            // The file appears shortly after a run starts; don't cache a 404
            // era or a placeholder.
            (header::CACHE_CONTROL, "no-cache"),
        ],
        bytes,
    )
        .into_response())
}

pub async fn get_model_viz(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Response, ApiError> {
    let path = state.models_dir().join(&name).join("viz.svg");
    let bytes = tokio::fs::read(&path).await.map_err(|_| not_found("visualization"))?;
    Ok((
        [
            (header::CONTENT_TYPE, "image/svg+xml"),
            // Model dirs are immutable snapshots.
            (header::CACHE_CONTROL, "private, max-age=86400"),
        ],
        bytes,
    )
        .into_response())
}

/// `blend.json` written by the blend predictor at the end of training:
/// per-member kind/predictor/source, fitted weights, exclude_blocks and solo
/// test metrics. Only blend runs have it; 404 otherwise.
pub async fn get_run_blend(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let bytes =
        run_file(&state, &id, "blend.json").await.ok_or_else(|| not_found("blend record"))?;
    Ok((
        [
            (header::CONTENT_TYPE, "application/json"),
            // Appears when the run finishes; don't cache the 404 era.
            (header::CACHE_CONTROL, "no-cache"),
        ],
        bytes,
    )
        .into_response())
}

pub async fn get_model_blend(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Response, ApiError> {
    let path = state.models_dir().join(&name).join("blend.json");
    let bytes = tokio::fs::read(&path).await.map_err(|_| not_found("blend record"))?;
    Ok((
        [
            (header::CONTENT_TYPE, "application/json"),
            // Model dirs are immutable snapshots.
            (header::CACHE_CONTROL, "private, max-age=86400"),
        ],
        bytes,
    )
        .into_response())
}

/// SSE: replay recorded progress, then stream live events until terminal.
/// Local runs stream from the in-memory handle (or the progress file);
/// queued/remote-worker runs replay the database rows, polling for new ones
/// while the run is still queued/running.
pub async fn run_events(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError> {
    use lensing_core::RunStatus;
    let run_dir = state.runs_dir().join(&id);
    let local_meta = runs::read_meta(&run_dir).ok();
    let live = state.live_runs.lock().unwrap().get(&id).cloned();

    let stream: futures::stream::BoxStream<'static, Result<Event, Infallible>> = match (live, local_meta) {
        (Some(handle), _) => {
            let (history, rx) = handle.snapshot_and_subscribe();
            let replay = stream::iter(history);
            let live = BroadcastStream::new(rx).filter_map(|r| async move { r.ok() });
            replay
                .chain(live)
                .map(|line| Ok(Event::default().data(line)))
                .boxed()
        }
        (None, Some(meta)) => {
            // Finished local run: replay the file, then a terminal status line.
            let mut lines = runs::read_progress(&run_dir);
            let has_terminal = lines.last().is_some_and(|l| l.contains("\"status\""));
            if !has_terminal {
                lines.push(json!({"event":"status","status":meta.status}).to_string());
            }
            stream::iter(lines)
                .map(|line| Ok(Event::default().data(line)))
                .boxed()
        }
        (None, None) => {
            // Database-backed run (queued / remote worker): replay rows and
            // poll for new ones until the run is terminal.
            let db = state.db.clone().ok_or_else(|| not_found("run"))?;
            let meta = lensing_db::queries::load_run(&db, &id)
                .await
                .ok()
                .flatten()
                .ok_or_else(|| not_found("run"))?;
            let in_flight = matches!(meta.status, RunStatus::Queued | RunStatus::Running);
            let run_id = id.clone();
            futures::stream::unfold(
                (db, run_id, 0i32, in_flight, false),
                |(db, run_id, mut seq, in_flight, done)| async move {
                    if done {
                        return None;
                    }
                    loop {
                        let rows =
                            lensing_db::queries::load_events(&db, &run_id, seq).await.unwrap_or_default();
                        if !rows.is_empty() {
                            let lines: Vec<String> =
                                rows.iter().map(|(_, line)| line.clone()).collect();
                            seq = rows.last().unwrap().0 + 1;
                            let terminal =
                                lines.iter().any(|l| l.contains("\"event\":\"status\""));
                            return Some((lines, (db, run_id, seq, in_flight, terminal)));
                        }
                        if !in_flight {
                            // Finished without a recorded terminal line: emit
                            // one so clients refetch the meta and close.
                            let status = lensing_db::queries::load_run(&db, &run_id)
                                .await
                                .ok()
                                .flatten()
                                .map(|m| m.status)
                                .unwrap_or(RunStatus::Interrupted);
                            let line = json!({"event":"status","status":status}).to_string();
                            return Some((vec![line], (db, run_id, seq, in_flight, true)));
                        }
                        // Still queued/running on a worker: poll.
                        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                        if let Ok(Some(m)) = lensing_db::queries::load_run(&db, &run_id).await {
                            if !matches!(m.status, RunStatus::Queued | RunStatus::Running) {
                                // Terminal now; loop once more to drain rows,
                                // then the empty-read path emits the status.
                                let rows = lensing_db::queries::load_events(&db, &run_id, seq)
                                    .await
                                    .unwrap_or_default();
                                let mut lines: Vec<String> =
                                    rows.iter().map(|(_, l)| l.clone()).collect();
                                let has_terminal =
                                    lines.iter().any(|l| l.contains("\"event\":\"status\""));
                                if !has_terminal {
                                    lines.push(
                                        json!({"event":"status","status":m.status}).to_string(),
                                    );
                                }
                                return Some((lines, (db, run_id, seq, in_flight, true)));
                            }
                        }
                    }
                },
            )
            .flat_map(stream::iter)
            .map(|line| Ok(Event::default().data(line)))
            .boxed()
        }
    };

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

// ---------- dataset archive (training-worker transport) ----------

/// Tar.gz of a dataset directory, for remote training workers
/// (`lensing-server worker --hub-url <this server>`).
pub async fn dataset_archive(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    if id.is_empty() || id.contains(['/', '\\']) || id.contains("..") {
        return Err(bad_request("invalid dataset id".into()));
    }
    let dir = state.datasets_dir().join(&id);
    if !dir.join("manifest.json").is_file() {
        return Err(not_found("dataset"));
    }
    let bytes = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<u8>> {
        let mut gz =
            flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        {
            let mut tar = tar::Builder::new(&mut gz);
            tar.append_dir_all(&id, &dir)?;
            tar.finish()?;
        }
        Ok(gz.finish()?)
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("archive task panicked: {e}")))?
    .map_err(ApiError::from)?;
    Ok((
        [
            (header::CONTENT_TYPE, "application/gzip"),
            // Dataset dirs are immutable artifacts.
            (header::CACHE_CONTROL, "private, max-age=86400"),
        ],
        bytes,
    )
        .into_response())
}

// ---------- quality preflight ----------

#[derive(Deserialize)]
pub struct PreflightRequest {
    #[serde(default)]
    pub quality: lensing_core::QualityFilterConfig,
    /// Currency handling; reconciled before the rules run so the preview
    /// matches what a build would produce.
    #[serde(default)]
    pub currency: lensing_core::CurrencyConfig,
    /// Sample flagged items returned per rule.
    #[serde(default = "default_sample")]
    pub sample: usize,
    /// Source collection to evaluate; defaults to the server's.
    #[serde(default)]
    pub collection: Option<String>,
}

fn default_sample() -> usize {
    8
}

/// Evaluate quality rules over the live corpus (payload-only scroll, no
/// vectors) WITHOUT building anything: per-rule counts + sample rows, so the
/// user sees what a filter config would drop before committing to a build.
pub async fn preflight(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PreflightRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let qdrant_url = state.qdrant_url.clone();
    let collection = resolve_collection(&state, &req.collection);
    let sample = req.sample.min(50);
    let cfg = req.quality;
    let currency = req.currency;
    let domain = state.domain.clone();
    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<serde_json::Value> {
        let mut points =
            lensing_pipeline::qdrant::scroll_payloads(&qdrant_url, &collection, &domain, &|_| {})?;
        let currency_report =
            lensing_pipeline::currency::apply(&mut points, &currency, &qdrant_url, &domain, &|_| {})?;
        let analysis = lensing_pipeline::quality::evaluate(&points, &cfg, &currency, &domain);
        let report = analysis.report(&cfg);
        Ok(json!({
            "n_total": points.len(),
            "n_excluded_total": report.n_excluded_total,
            "rules": report.rules,
            "samples": analysis.samples(&points, sample, &domain),
            "currency": currency_report,
        }))
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("preflight panicked: {e}")))??;
    Ok(Json(result))
}

// ---------- models ----------

pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(json!({ "models": models::list_models(&state) }))
}

#[derive(Deserialize)]
pub struct PromoteRequest {
    pub name: String,
    pub run_id: String,
    #[serde(default)]
    pub notes: Option<String>,
}

pub async fn promote_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PromoteRequest>,
) -> Result<(StatusCode, Json<lensing_core::ModelRecord>), ApiError> {
    if state.models_dir().join(&req.name).exists() {
        return Err(ApiError(
            StatusCode::CONFLICT,
            format!("model name {:?} is already taken", req.name),
        ));
    }
    // Remote-worker runs have no local dir; materialize it from the
    // database (run row + uploaded artifacts) so promotion's file-based
    // path works unchanged.
    if !state.runs_dir().join(&req.run_id).join("meta.json").is_file() {
        runs::materialize_run(&state, &req.run_id)
            .await
            .map_err(|e| bad_request(format!("{e:#}")))?;
    }
    let record = models::promote(&state, &req.name, &req.run_id, req.notes)
        .map_err(|e| bad_request(format!("{e:#}")))?;
    // A manual promotion may belong in the best-models group.
    best_models::spawn_recompute(state.clone());
    Ok((StatusCode::CREATED, Json(record)))
}

pub async fn get_model(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let dir = state.models_dir().join(&name);
    let record = models::read_record(&dir).map_err(|_| not_found("model"))?;
    let contract = models::read_contract(&dir).map_err(anyhow::Error::from)?;
    // The training hyperparams, so the UI can clone them into a definition.
    let hyperparams: Option<serde_json::Value> = std::fs::read_to_string(dir.join("hyperparams.json"))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok());
    Ok(Json(json!({
        "record": record,
        "contract": contract_summary(&contract),
        "hyperparams": hyperparams,
    })))
}

#[derive(Deserialize)]
pub struct RenameRequest {
    pub new_name: String,
}

pub async fn rename_model(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<RenameRequest>,
) -> Result<Json<lensing_core::ModelRecord>, ApiError> {
    if !state.models_dir().join(&name).join("record.json").is_file() {
        return Err(not_found("model"));
    }
    if state.models_dir().join(&req.new_name).exists() {
        return Err(conflict(format!("model name {:?} is already taken", req.new_name)));
    }
    let record = models::rename_model(&state, &name, &req.new_name)
        .map_err(|e| bad_request(format!("{e:#}")))?;
    Ok(Json(record))
}

pub async fn delete_model(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    models::delete_model(&state, &name).map_err(|_| not_found("model"))?;
    // A deleted group member gets backfilled by the next-best candidate.
    // (To keep a top-ranked run OUT of the group, exclude it via
    // PUT /api/best-models — deletion alone would just re-promote it.)
    best_models::spawn_recompute(state.clone());
    Ok(Json(json!({ "ok": true })))
}

/// What callers must send: required fields, embedding dim, target transform.
pub async fn get_model_contract(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let dir = state.models_dir().join(&name);
    let contract = models::read_contract(&dir).map_err(|_| not_found("model"))?;
    Ok(Json(contract_summary(&contract)))
}

/// The caller-relevant slice of a contract (the 1536-float PCA mean and the
/// full column list stay server-side).
fn contract_summary(c: &lensing_core::Contract) -> serde_json::Value {
    json!({
        "contract_version": c.contract_version,
        "n_cols": c.n_cols,
        "input_fields": c.input_fields,
        "target": c.target,
        "feature_config": c.feature_config,
        "pca_dims": c.pca.dims,
    })
}

pub async fn predict_model(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<models::PredictRequest>,
) -> Result<Json<models::PredictResponse>, ApiError> {
    match models::predict(state, name, req).await {
        Ok(resp) => Ok(Json(resp)),
        Err(models::PredictError::NotFound) => Err(not_found("model")),
        Err(models::PredictError::BadInput(msg)) => Err(bad_request(msg)),
        Err(models::PredictError::Internal(e)) => Err(e.into()),
    }
}

// ---------- best-models group ----------

pub async fn get_best_models(
    State(state): State<Arc<AppState>>,
) -> Json<lensing_core::BestModelGroup> {
    Json(best_models::read_group(&state))
}

pub async fn recompute_best_models(
    State(state): State<Arc<AppState>>,
) -> Result<Json<lensing_core::BestModelGroup>, ApiError> {
    Ok(Json(best_models::recompute(&state).await?))
}

/// Curation deltas for the group: pins force-include promoted models,
/// excludes (run ids or model names) bar candidates from auto-selection.
#[derive(Deserialize)]
pub struct CurateRequest {
    #[serde(default)]
    pub pin: Vec<String>,
    #[serde(default)]
    pub exclude: Vec<String>,
    #[serde(default)]
    pub unpin: Vec<String>,
    #[serde(default)]
    pub unexclude: Vec<String>,
}

pub async fn put_best_models(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CurateRequest>,
) -> Result<Json<lensing_core::BestModelGroup>, ApiError> {
    let group = best_models::curate(&state, req.pin, req.exclude, req.unpin, req.unexclude)
        .await
        .map_err(|e| bad_request(format!("{e:#}")))?;
    Ok(Json(group))
}

pub async fn predict_best_models(
    State(state): State<Arc<AppState>>,
    Json(req): Json<models::PredictRequest>,
) -> Result<Json<best_models::GroupPredictResponse>, ApiError> {
    match best_models::predict_group(state, req).await {
        Ok(resp) => Ok(Json(resp)),
        Err(models::PredictError::NotFound) => Err(not_found("model")),
        Err(models::PredictError::BadInput(msg)) => Err(bad_request(msg)),
        Err(models::PredictError::Internal(e)) => Err(e.into()),
    }
}

// ---------- manual listings ----------

/// A manually authored entry: the document text plus the domain's metadata
/// fields, flat by field name. Unknown keys (display extras like photo URLs
/// or the source page) are stored verbatim — they never enter a feature
/// contract; declared fields are type-checked against the domain.
#[derive(Deserialize)]
pub struct CreateListingRequest {
    /// Free-text description; this is the text that gets embedded.
    pub content: String,
    #[serde(flatten)]
    pub fields: serde_json::Map<String, serde_json::Value>,
}

/// The text we embed for a manual listing. Corpus content is the raw
/// free-text description, so we embed the description as-is; if the upstream
/// embedder turns out to prepend structured fields, change it here only.
fn listing_embed_text(content: &str) -> String {
    content.trim().to_string()
}

/// Corpus-shaped payload (`content_field` + metadata under the domain's
/// root). Declared fields type-check against the domain; null/absent fields
/// are omitted, matching how absent fields look on the corpus; the domain's
/// timestamp field is stamped now when absent.
fn listing_payload(
    domain: &lensing_core::domain::Domain,
    req: &CreateListingRequest,
    content: &str,
) -> Result<serde_json::Value, ApiError> {
    use lensing_core::domain::FieldRole;
    let mut md = serde_json::Map::new();
    for (key, value) in &req.fields {
        if value.is_null() {
            continue;
        }
        if let Some(f) = domain.field(key) {
            let ok = match f.role {
                FieldRole::Numeric | FieldRole::Target => value.is_number(),
                FieldRole::Categorical | FieldRole::FilterOnly | FieldRole::Timestamp => {
                    value.is_string() || value.is_number()
                }
                FieldRole::Coordinates => {
                    value.get("lat").is_some() && value.get("lon").is_some()
                }
                FieldRole::Display => true,
            };
            if !ok {
                return Err(bad_request(format!(
                    "field {key:?} has the wrong shape for its role {:?}",
                    f.role
                )));
            }
        }
        md.insert(key.clone(), value.clone());
    }
    if let Some(ts) = domain.fields.iter().find(|f| f.role == FieldRole::Timestamp) {
        md.entry(ts.name.clone()).or_insert_with(|| json!(Utc::now().to_rfc3339()));
    }
    let root = domain.corpus.metadata_root.as_str();
    let content_field = domain.corpus.content_field.as_str();
    Ok(if root.is_empty() {
        let mut obj = md;
        obj.insert(content_field.to_string(), json!(content));
        serde_json::Value::Object(obj)
    } else {
        json!({ content_field: content, root: md })
    })
}

/// The API view of a stored listing payload: `{id, content, metadata}`,
/// resolving the domain's metadata root and content field.
fn listing_view(
    domain: &lensing_core::domain::Domain,
    id: u64,
    payload: &serde_json::Value,
) -> serde_json::Value {
    let root = domain.corpus.metadata_root.as_str();
    let content_field = domain.corpus.content_field.as_str();
    let metadata = if root.is_empty() {
        let mut obj = payload.as_object().cloned().unwrap_or_default();
        obj.remove(content_field);
        serde_json::Value::Object(obj)
    } else {
        payload[root].clone()
    };
    json!({ "id": id, "content": payload[content_field], "metadata": metadata })
}

/// Unique-enough id for single-user manual authoring: epoch MICROS plus a
/// process counter against same-microsecond double submits. Stays u64 because
/// the whole point pipeline (`RawPoint`, scroll parsing) requires u64 ids;
/// stays under 2^53 (micros, not nanos) so the JS UI can hold it exactly.
fn new_listing_id() -> u64 {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let micros = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0);
    micros.wrapping_add(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

pub async fn create_listing(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateListingRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), ApiError> {
    let content = listing_embed_text(&req.content);
    if content.is_empty() {
        return Err(bad_request("content must not be empty".into()));
    }
    let api_key = state.openai_api_key.clone().ok_or_else(|| {
        ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "OPENAI_API_KEY is not configured on the server; cannot embed listings".into(),
        )
    })?;
    let model = state.embedding_model.clone();
    let openai_base_url = state.openai_base_url.clone();
    let qdrant_url = state.qdrant_url.clone();
    let collection = state.manual_collection.clone();
    let id = new_listing_id();
    // Required width of corpus embeddings; every model contract freezes it,
    // so a listing embedded at any other width would be unusable.
    let embedding_dim = state.domain.corpus.embedding_dim;
    let payload = listing_payload(&state.domain, &req, &content)?;
    let view = listing_view(&state.domain, id, &payload);

    let stored = tokio::task::spawn_blocking(move || -> Result<serde_json::Value, ApiError> {
        let vector = lensing_pipeline::openai::embed(&api_key, &model, &openai_base_url, &content)
            .map_err(|e| ApiError(StatusCode::BAD_GATEWAY, format!("{e:#}")))?;
        if vector.len() != embedding_dim {
            return Err(bad_request(format!(
                "embedding model {model} returned {} dims; models require {embedding_dim} \
                 (set LENSING_EMBEDDING_MODEL to a {embedding_dim}-dim model)",
                vector.len()
            )));
        }
        if lensing_pipeline::qdrant::get_collection_params(&qdrant_url, &collection).is_err() {
            lensing_pipeline::qdrant::create_collection(
                &qdrant_url,
                &collection,
                embedding_dim as u64,
                "Cosine",
            )
            .map_err(ApiError::from)?;
        }
        let point = lensing_pipeline::qdrant::RawJsonPoint { id, vector, payload: payload.clone() };
        lensing_pipeline::qdrant::upsert_raw(&qdrant_url, &collection, std::slice::from_ref(&point))
            .map_err(ApiError::from)?;
        Ok(view)
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("create listing panicked: {e}")))??;

    Ok((StatusCode::CREATED, Json(stored)))
}

/// Full-replace edit of a manual listing. The embedding is recomputed only
/// when the description text actually changed; metadata-only edits keep the
/// stored vector (and cost no OpenAI call). `createdAt` is preserved —
/// editing a listing does not make it newer.
pub async fn update_listing(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
    Json(req): Json<CreateListingRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let content = listing_embed_text(&req.content);
    if content.is_empty() {
        return Err(bad_request("content must not be empty".into()));
    }
    let api_key = state.openai_api_key.clone();
    let model = state.embedding_model.clone();
    let openai_base_url = state.openai_base_url.clone();
    let qdrant_url = state.qdrant_url.clone();
    let collection = state.manual_collection.clone();
    let embedding_dim = state.domain.corpus.embedding_dim;
    let domain = state.domain.clone();
    let mut payload = listing_payload(&domain, &req, &content)?;

    let stored = tokio::task::spawn_blocking(move || -> Result<serde_json::Value, ApiError> {
        let existing = lensing_pipeline::qdrant::get_point_raw(&qdrant_url, &collection, id)
            .map_err(ApiError::from)?
            .ok_or_else(|| not_found("listing"))?;
        // The timestamp field is preserved — editing does not make it newer.
        let root = domain.corpus.metadata_root.as_str();
        let content_field = domain.corpus.content_field.as_str();
        if let Some(ts) = domain
            .fields
            .iter()
            .find(|f| f.role == lensing_core::domain::FieldRole::Timestamp)
        {
            let key = ts.name.as_str();
            let created = if root.is_empty() {
                existing.payload[key].clone()
            } else {
                existing.payload[root][key].clone()
            };
            if created.is_string() {
                if root.is_empty() {
                    payload[key] = created;
                } else {
                    payload[root][key] = created;
                }
            }
        }
        let vector = if existing.payload[content_field].as_str() == Some(content.as_str()) {
            existing.vector
        } else {
            let api_key = api_key.ok_or_else(|| {
                ApiError(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "OPENAI_API_KEY is not configured on the server; cannot re-embed edited content"
                        .into(),
                )
            })?;
            let vector =
                lensing_pipeline::openai::embed(&api_key, &model, &openai_base_url, &content)
                    .map_err(|e| ApiError(StatusCode::BAD_GATEWAY, format!("{e:#}")))?;
            if vector.len() != embedding_dim {
                return Err(bad_request(format!(
                    "embedding model {model} returned {} dims; models require {embedding_dim} \
                     (set LENSING_EMBEDDING_MODEL to a {embedding_dim}-dim model)",
                    vector.len()
                )));
            }
            vector
        };
        let point = lensing_pipeline::qdrant::RawJsonPoint { id, vector, payload: payload.clone() };
        lensing_pipeline::qdrant::upsert_raw(&qdrant_url, &collection, std::slice::from_ref(&point))
            .map_err(ApiError::from)?;
        Ok(listing_view(&domain, id, &payload))
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("update listing panicked: {e}")))??;

    Ok(Json(stored))
}

pub async fn list_listings(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let qdrant_url = state.qdrant_url.clone();
    let collection = state.manual_collection.clone();
    let domain = state.domain.clone();
    let mut listings = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<serde_json::Value>> {
        // No collection yet (nothing ever created) is just an empty list.
        if lensing_pipeline::qdrant::get_collection_params(&qdrant_url, &collection).is_err() {
            return Ok(Vec::new());
        }
        let points = lensing_pipeline::qdrant::scroll_collection_raw(&qdrant_url, &collection)?;
        Ok(points
            .into_iter()
            .map(|p| listing_view(&domain, p.id, &p.payload))
            .collect())
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("list listings panicked: {e}")))??;
    // Ids are epoch nanos, so id desc == newest first.
    listings.sort_by(|a, b| b["id"].as_u64().cmp(&a["id"].as_u64()));
    Ok(Json(json!({ "listings": listings })))
}

pub async fn get_listing(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let qdrant_url = state.qdrant_url.clone();
    let collection = state.manual_collection.clone();
    let point = tokio::task::spawn_blocking(move || {
        lensing_pipeline::qdrant::get_point_raw(&qdrant_url, &collection, id)
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("get listing panicked: {e}")))??
    .ok_or_else(|| not_found("listing"))?;
    // The embedding rides along so the UI can predict via inline items.
    let mut view = listing_view(&state.domain, point.id, &point.payload);
    view["embedding"] = json!(point.vector);
    Ok(Json(view))
}

pub async fn delete_listing(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let qdrant_url = state.qdrant_url.clone();
    let collection = state.manual_collection.clone();
    tokio::task::spawn_blocking(move || -> Result<(), ApiError> {
        if lensing_pipeline::qdrant::get_point_raw(&qdrant_url, &collection, id)
            .map_err(ApiError::from)?
            .is_none()
        {
            return Err(not_found("listing"));
        }
        lensing_pipeline::qdrant::delete_point(&qdrant_url, &collection, id).map_err(ApiError::from)
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("delete listing panicked: {e}")))??;
    Ok(Json(json!({ "ok": true })))
}

// ---------- model definitions ----------

pub async fn list_definitions(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(json!({ "definitions": definitions::list(&state).await }))
}

pub async fn get_definition(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<lensing_core::ModelDefinition>, ApiError> {
    definitions::get(&state, &name)
        .await
        .map(Json)
        .ok_or_else(|| not_found("definition"))
}

#[derive(Deserialize)]
pub struct CreateDefinitionRequest {
    pub name: String,
    pub predictor: String,
    #[serde(default)]
    pub hyperparams: Option<serde_json::Value>,
    #[serde(default)]
    pub dataset_tags: Vec<String>,
    #[serde(default)]
    pub notes: Option<String>,
}

pub async fn create_definition(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateDefinitionRequest>,
) -> Result<(StatusCode, Json<lensing_core::ModelDefinition>), ApiError> {
    let def = definitions::create(
        &state,
        &req.name,
        &req.predictor,
        req.hyperparams,
        req.dataset_tags,
        req.notes,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(def)))
}

#[derive(Deserialize)]
pub struct UpdateDefinitionRequest {
    #[serde(default)]
    pub hyperparams: Option<serde_json::Value>,
    #[serde(default)]
    pub dataset_tags: Option<Vec<String>>,
    /// Empty string clears the notes.
    #[serde(default)]
    pub notes: Option<String>,
}

pub async fn update_definition(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<UpdateDefinitionRequest>,
) -> Result<Json<lensing_core::ModelDefinition>, ApiError> {
    let def = definitions::update(&state, &name, req.hyperparams, req.dataset_tags, req.notes).await?;
    Ok(Json(def))
}

pub async fn rename_definition(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<RenameRequest>,
) -> Result<Json<lensing_core::ModelDefinition>, ApiError> {
    Ok(Json(definitions::rename(&state, &name, &req.new_name).await?))
}

#[derive(Deserialize)]
pub struct CloneDefinitionRequest {
    pub new_name: String,
}

pub async fn clone_definition(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<CloneDefinitionRequest>,
) -> Result<(StatusCode, Json<lensing_core::ModelDefinition>), ApiError> {
    let def = definitions::clone_def(&state, &name, &req.new_name).await?;
    Ok((StatusCode::CREATED, Json(def)))
}

pub async fn delete_definition(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    definitions::delete(&state, &name).await?;
    Ok(Json(json!({ "ok": true })))
}

// ---------- domain ----------

/// The domain configuration (`domain.toml`): corpus schema, target spec,
/// field descriptors, quality-rule labels, metrics display, UI vocabulary.
/// The UI renders labels, forms and formatters from this.
pub async fn get_domain(State(state): State<Arc<AppState>>) -> Json<lensing_core::domain::Domain> {
    Json(state.domain.as_ref().clone())
}

// ---------- misc ----------

pub async fn health() -> Json<serde_json::Value> {
    Json(json!({ "ok": true, "now": Utc::now().to_rfc3339() }))
}
