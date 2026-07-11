//! `/api/representations` — latent representations as first-class objects.
//!
//! A representation is a compressor (PCA / autoencoder / sparse-AE) fit over a
//! source collection, producing a latent collection datasets can build on. This
//! mirrors the dataset-build job pattern (`build_dataset` → `BuildStatus` →
//! `get_build`): `POST /api/representations` starts a job, the UI polls
//! `GET /api/builds/{id}`, and the finished representation is persisted under
//! `data/representations/<id>/` (meta.json + compressor artifacts).

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::Json;
use chrono::Utc;
use lensing_compression::store::{CompressorConfig, Method, PipelineConfig, SinkConfig, SourceConfig};
use lensing_compression::{AeConfig, Autoencoder, Compressor, Pca, PcaModel, SparseConfig};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::api::{bad_request, not_found, ApiError};
use crate::runs;
use crate::state::{AppState, BuildStatus};

/// Persisted description of a built representation (`meta.json`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RepresentationMeta {
    pub id: String,
    pub name: String,
    pub method: Method,
    pub latent_dim: usize,
    pub source_collection: String,
    pub sink_collection: String,
    pub created_at: String,
    pub n_points: usize,
    /// EVR curve (PCA) or per-block reconstruction R² (AE).
    pub quality: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct BuildRepresentationRequest {
    pub name: String,
    pub method: Method,
    pub latent: usize,
    /// Sink (latent) collection name; the source defaults to the corpus.
    pub sink_collection: String,
    #[serde(default)]
    pub source_collection: Option<String>,
    #[serde(default)]
    pub epochs: Option<usize>,
    #[serde(default)]
    pub hidden: Option<Vec<usize>>,
    #[serde(default)]
    pub sparse_weight: Option<f64>,
    #[serde(default = "default_distance")]
    pub distance: String,
}

fn default_distance() -> String {
    "Cosine".into()
}

fn reps_dir(state: &AppState) -> std::path::PathBuf {
    state.root.join("data/representations")
}

pub async fn list_representations(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let mut metas: Vec<RepresentationMeta> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(reps_dir(&state)) {
        for e in entries.flatten() {
            let path = e.path().join("meta.json");
            if let Ok(text) = std::fs::read_to_string(&path) {
                if let Ok(m) = serde_json::from_str::<RepresentationMeta>(&text) {
                    metas.push(m);
                }
            }
        }
    }
    metas.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Json(json!({ "representations": metas }))
}

pub async fn get_representation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<RepresentationMeta>, ApiError> {
    let path = reps_dir(&state).join(&id).join("meta.json");
    let text = std::fs::read_to_string(&path).map_err(|_| not_found("representation"))?;
    let m: RepresentationMeta = serde_json::from_str(&text).map_err(anyhow::Error::from)?;
    Ok(Json(m))
}

/// Build a representation: fit a compressor over the source collection and write
/// the latent to the sink collection. Returns a `build_id`; poll
/// `GET /api/builds/{id}`.
pub async fn build_representation(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BuildRepresentationRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if req.latent < 1 || req.latent > 4096 {
        return Err(bad_request("latent must be in 1..=4096".into()));
    }
    if req.name.trim().is_empty() {
        return Err(bad_request("name must not be empty".into()));
    }
    if req.sink_collection.trim().is_empty() {
        return Err(bad_request("sink_collection must not be empty".into()));
    }

    let build_id = format!("rep-{}", runs::new_run_id("rep"));
    state
        .builds
        .lock()
        .unwrap()
        .insert(build_id.clone(), BuildStatus::Building { stage: "queued".into() });

    let source_collection = req.source_collection.clone().unwrap_or_else(|| state.collection.clone());
    let cfg = PipelineConfig {
        source: SourceConfig { url: state.qdrant_url.clone(), collection: source_collection.clone() },
        compressor: CompressorConfig {
            method: req.method.clone(),
            latent: req.latent,
            epochs: req.epochs,
            hidden: req.hidden.clone(),
            sparse_weight: req.sparse_weight,
        },
        sink: SinkConfig {
            url: state.qdrant_url.clone(),
            collection: req.sink_collection.clone(),
            distance: req.distance.clone(),
        },
    };

    let rep_id = format!("rep_{}", req.name.trim().to_lowercase().replace(|c: char| !c.is_alphanumeric(), "_"));
    let name = req.name.trim().to_string();
    let dir = reps_dir(&state).join(&rep_id);
    let st = state.clone();
    let id = build_id.clone();
    tokio::spawn(async move {
        let _permit = st.build_slots.clone().acquire_owned().await.unwrap();
        let result = tokio::task::spawn_blocking(move || -> anyhow::Result<RepresentationMeta> {
            std::fs::create_dir_all(&dir)?;
            let mut compressor: Box<dyn Compressor> = build_compressor(&cfg.compressor);
            // Passthrough over the stored corpus vectors (no multimodal spec yet).
            let report = lensing_compression::build_latent_collection(&cfg, compressor.as_mut(), None)?;
            compressor.save(&dir)?;
            compressor.export_encoder_onnx(&dir).ok();
            let meta = RepresentationMeta {
                id: rep_id.clone(),
                name,
                method: cfg.compressor.method.clone(),
                latent_dim: report.latent_dim,
                source_collection,
                sink_collection: cfg.sink.collection.clone(),
                created_at: Utc::now().to_rfc3339(),
                n_points: report.n_points,
                quality: report.quality,
            };
            std::fs::write(dir.join("meta.json"), serde_json::to_vec_pretty(&meta)?)?;
            Ok(meta)
        })
        .await;
        let status = match result {
            Ok(Ok(meta)) => BuildStatus::Done { dataset_id: meta.id },
            Ok(Err(e)) => BuildStatus::Failed { error: format!("{e:#}") },
            Err(e) => BuildStatus::Failed { error: format!("representation build panicked: {e}") },
        };
        st.builds.lock().unwrap().insert(id, status);
    });

    Ok(Json(json!({ "build_id": build_id })))
}

#[derive(Debug, Deserialize)]
pub struct EncodeRequest {
    /// The item's stored vector (the passthrough input the representation was fit on).
    pub vector: Vec<f32>,
}

/// One-shot encode: reload a fitted representation and project a new item into
/// its latent space. Supported for PCA (reloaded from artifacts); the AE reload
/// path is deferred (the burn arch needs to be rehydrated from config).
pub async fn encode_representation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<EncodeRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let dir = reps_dir(&state).join(&id);
    let text = std::fs::read_to_string(dir.join("meta.json")).map_err(|_| not_found("representation"))?;
    let meta: RepresentationMeta = serde_json::from_str(&text).map_err(anyhow::Error::from)?;

    match meta.method {
        Method::Pca => {
            let mean = read_f32(&dir.join("pca_mean.f32")).map_err(|_| not_found("pca artifacts"))?;
            let comp = read_f32(&dir.join("pca_components.f32")).map_err(|_| not_found("pca artifacts"))?;
            let d = mean.len();
            let k = meta.latent_dim;
            if req.vector.len() != d {
                return Err(bad_request(format!("vector has {} dims, representation expects {d}", req.vector.len())));
            }
            let evr: Vec<f32> = vec![0.0; k]; // EVR not needed for projection
            let model = PcaModel::from_parts(&mean, &comp, k, d, evr).map_err(ApiError::from)?;
            let latent = model.project_all(&req.vector, d);
            Ok(Json(json!({ "latent": latent })))
        }
        _ => Err(bad_request("one-shot encode is currently supported for PCA representations only".into())),
    }
}

fn build_compressor(cfg: &CompressorConfig) -> Box<dyn Compressor> {
    match cfg.method {
        Method::Pca => Box::new(Pca::new(cfg.latent)),
        Method::Autoencoder | Method::SparseAe => {
            // The input dim is discovered at fit time from the source vectors, so
            // seed the config with a placeholder d; `AeConfig::dense` only uses d
            // for its default single-block layout, which `fit` re-validates.
            let mut ae = AeConfig::dense(cfg.latent, cfg.latent);
            if let Some(h) = &cfg.hidden {
                let mut hidden = h.clone();
                *hidden.last_mut().unwrap() = cfg.latent;
                ae.enc_hidden = hidden;
            }
            if let Some(e) = cfg.epochs {
                ae.epochs = e;
            }
            if matches!(cfg.method, Method::SparseAe) {
                ae.sparse = Some(SparseConfig { weight: cfg.sparse_weight.unwrap_or(0.01) });
            }
            Box::new(Autoencoder::new(ae))
        }
    }
}

fn read_f32(path: &std::path::Path) -> anyhow::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    anyhow::ensure!(bytes.len() % 4 == 0, "f32 file has non-multiple-of-4 length");
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_parses_with_defaults() {
        let req: BuildRepresentationRequest = serde_json::from_value(json!({
            "name": "Song AE",
            "method": "autoencoder",
            "latent": 64,
            "sink_collection": "spotify_tracks_song_ae",
            "epochs": 200,
            "hidden": [256, 64]
        }))
        .unwrap();
        assert_eq!(req.method, Method::Autoencoder);
        assert_eq!(req.latent, 64);
        assert_eq!(req.distance, "Cosine");
        assert!(req.source_collection.is_none());
    }

    #[test]
    fn meta_round_trips() {
        let m = RepresentationMeta {
            id: "rep_song_ae".into(),
            name: "Song AE".into(),
            method: Method::Autoencoder,
            latent_dim: 64,
            source_collection: "spotify_tracks_content".into(),
            sink_collection: "spotify_tracks_song_ae".into(),
            created_at: "2026-07-11T00:00:00Z".into(),
            n_points: 1000,
            quality: json!({"kind": "block_r2", "blocks": [["text", 0.87]]}),
        };
        let text = serde_json::to_string(&m).unwrap();
        let back: RepresentationMeta = serde_json::from_str(&text).unwrap();
        assert_eq!(back.latent_dim, 64);
        assert_eq!(back.method, Method::Autoencoder);
    }

    #[test]
    fn pca_compressor_builds() {
        let c = build_compressor(&CompressorConfig {
            method: Method::Pca,
            latent: 32,
            epochs: None,
            hidden: None,
            sparse_weight: None,
        });
        assert_eq!(c.latent_dim(), 32);
    }
}
