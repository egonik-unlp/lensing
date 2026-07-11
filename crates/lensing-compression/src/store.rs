//! Latent generation over a vector store — the SongAE flow, generalized and
//! config-driven: read a source collection, fit a [`Compressor`], write the
//! latent to a destination collection with the same ids + payloads.
//!
//! Everything (URLs, collection names, method, latent dim) comes from config, so
//! nothing is hardcoded and every destination is per-instance (D1). The Qdrant
//! client here is intentionally self-contained (a small reqwest wrapper) rather
//! than reusing `lensing-pipeline`'s — depending on that crate would cycle, since
//! it already depends on this one for the re-exported PCA.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::preprocess::PreprocessSpec;
use crate::Compressor;

// ----------------------------------------------------------------------------
// Config (TOML): [source] / [compressor] / [sink].
// ----------------------------------------------------------------------------

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    Pca,
    Autoencoder,
    SparseAe,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SourceConfig {
    /// Qdrant base URL — per instance, e.g. `http://localhost:6335`.
    pub url: String,
    pub collection: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct CompressorConfig {
    pub method: Method,
    pub latent: usize,
    #[serde(default)]
    pub epochs: Option<usize>,
    /// Encoder hidden widths for the AE (last is overridden by `latent`).
    #[serde(default)]
    pub hidden: Option<Vec<usize>>,
    /// Sparsity weight for the sparse-AE variant.
    #[serde(default)]
    pub sparse_weight: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SinkConfig {
    pub url: String,
    pub collection: String,
    #[serde(default = "default_distance")]
    pub distance: String,
}

fn default_distance() -> String {
    "Cosine".into()
}

/// A full latent-generation pipeline declaration.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct PipelineConfig {
    pub source: SourceConfig,
    pub compressor: CompressorConfig,
    pub sink: SinkConfig,
}

impl PipelineConfig {
    pub fn from_toml(text: &str) -> Result<Self> {
        toml::from_str(text).context("parsing pipeline config")
    }
}

// ----------------------------------------------------------------------------
// Minimal Qdrant client (reqwest blocking).
// ----------------------------------------------------------------------------

/// One corpus point: id + stored vector + payload.
pub struct StorePoint {
    pub id: Value,
    pub vector: Vec<f32>,
    pub payload: Value,
}

struct Qdrant {
    base: String,
    http: reqwest::blocking::Client,
}

impl Qdrant {
    fn new(base: &str) -> Self {
        Qdrant { base: base.trim_end_matches('/').to_string(), http: reqwest::blocking::Client::new() }
    }

    /// Scroll every point (id, vector, payload) from a collection.
    fn scroll_all(&self, collection: &str) -> Result<Vec<StorePoint>> {
        let mut out = Vec::new();
        let mut offset: Option<Value> = None;
        loop {
            let mut body = json!({"limit": 2048, "with_payload": true, "with_vector": true});
            if let Some(o) = &offset {
                body["offset"] = o.clone();
            }
            let url = format!("{}/collections/{}/points/scroll", self.base, collection);
            let resp: Value = self
                .http
                .post(&url)
                .json(&body)
                .send()
                .with_context(|| format!("scroll {collection}"))?
                .error_for_status()?
                .json()?;
            let result = &resp["result"];
            for p in result["points"].as_array().cloned().unwrap_or_default() {
                let vector = p["vector"]
                    .as_array()
                    .map(|a| a.iter().filter_map(Value::as_f64).map(|v| v as f32).collect())
                    .unwrap_or_default();
                out.push(StorePoint { id: p["id"].clone(), vector, payload: p["payload"].clone() });
            }
            match result["next_page_offset"].clone() {
                Value::Null => break,
                next => offset = Some(next),
            }
        }
        Ok(out)
    }

    fn recreate_collection(&self, name: &str, size: usize, distance: &str) -> Result<()> {
        let url = format!("{}/collections/{}", self.base, name);
        // Delete-if-exists then create (recreate semantics).
        let _ = self.http.delete(&url).send();
        self.http
            .put(&url)
            .json(&json!({"vectors": {"size": size, "distance": distance}}))
            .send()
            .with_context(|| format!("create {name}"))?
            .error_for_status()?;
        Ok(())
    }

    fn upsert(&self, name: &str, points: &[Value]) -> Result<()> {
        let url = format!("{}/collections/{}/points", self.base, name);
        self.http
            .put(&url)
            .json(&json!({"points": points}))
            .send()
            .with_context(|| format!("upsert {name}"))?
            .error_for_status()?;
        Ok(())
    }
}

/// Outcome of a latent-generation run.
pub struct LatentReport {
    pub n_points: usize,
    pub latent_dim: usize,
    /// The fitted compressor's quality metric (EVR for PCA, block-R² for AE).
    pub quality: Value,
}

/// Fit `compressor` over the source collection and write the latent to the sink
/// collection, preserving ids and payloads. Generalizes `build_song_ae.py`.
///
/// When `spec` is `Some`, each row is assembled from the point's stored vector +
/// payload exactly as at serve time (`encode_one`) — the multimodal SongAE case.
/// When `spec` is `None`, the stored vector is used directly (the "compress the
/// embedding" case: PCA/AE straight over the corpus vectors), with the input dim
/// inferred from the first point.
pub fn build_latent_collection(
    cfg: &PipelineConfig,
    compressor: &mut dyn Compressor,
    spec: Option<&PreprocessSpec>,
) -> Result<LatentReport> {
    let src = Qdrant::new(&cfg.source.url);
    let points = src.scroll_all(&cfg.source.collection)?;
    anyhow::ensure!(!points.is_empty(), "source collection {} is empty", cfg.source.collection);

    // Assemble the feature matrix. With a spec: replay serve-time encoding.
    // Without: passthrough the stored vector (dim from the first point).
    let d = match spec {
        Some(s) => s.feature_dim(),
        None => points[0].vector.len(),
    };
    anyhow::ensure!(d > 0, "source vectors have zero dimension");
    let mut x = Vec::with_capacity(points.len() * d);
    for p in &points {
        let row = match spec {
            Some(s) => s.encode_one(&p.vector, &p.payload)?,
            None => p.vector.clone(),
        };
        anyhow::ensure!(row.len() == d, "row width {} != expected {}", row.len(), d);
        x.extend_from_slice(&row);
    }

    let rows: Vec<u32> = (0..points.len() as u32).collect();
    compressor.fit(&x, d, &rows)?;
    let latents = compressor.encode(&x, d)?;
    let k = compressor.latent_dim();

    // L2-normalize each latent (Cosine sink), then upsert with same id+payload.
    let sink = Qdrant::new(&cfg.sink.url);
    sink.recreate_collection(&cfg.sink.collection, k, &cfg.sink.distance)?;
    let mut batch = Vec::new();
    for (i, p) in points.iter().enumerate() {
        let z = &latents[i * k..(i + 1) * k];
        let norm = (z.iter().map(|v| v * v).sum::<f32>().sqrt()).max(1e-9);
        let zn: Vec<f32> = z.iter().map(|v| v / norm).collect();
        batch.push(json!({"id": p.id, "vector": zn, "payload": p.payload}));
        if batch.len() == 256 {
            sink.upsert(&cfg.sink.collection, &batch)?;
            batch.clear();
        }
    }
    if !batch.is_empty() {
        sink.upsert(&cfg.sink.collection, &batch)?;
    }

    Ok(LatentReport { n_points: points.len(), latent_dim: k, quality: compressor.quality() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_dual_config() {
        let toml = r#"
[source]
url = "http://localhost:6335"
collection = "spotify_tracks_content"

[compressor]
method = "autoencoder"
latent = 64
epochs = 200
hidden = [256, 64]

[sink]
url = "http://localhost:6335"
collection = "spotify_tracks_song_ae"
distance = "Cosine"
"#;
        let cfg = PipelineConfig::from_toml(toml).unwrap();
        assert_eq!(cfg.compressor.method, Method::Autoencoder);
        assert_eq!(cfg.compressor.latent, 64);
        assert_eq!(cfg.source.collection, "spotify_tracks_content");
        assert_eq!(cfg.sink.collection, "spotify_tracks_song_ae");
        assert_eq!(cfg.sink.distance, "Cosine");
    }

    #[test]
    fn distance_defaults_to_cosine() {
        let toml = r#"
[source]
url = "http://q:6333"
collection = "src"
[compressor]
method = "pca"
latent = 32
[sink]
url = "http://q:6333"
collection = "dst"
"#;
        let cfg = PipelineConfig::from_toml(toml).unwrap();
        assert_eq!(cfg.compressor.method, Method::Pca);
        assert_eq!(cfg.sink.distance, "Cosine");
    }

    #[test]
    fn source_and_sink_are_independent_urls() {
        // D1: nothing forces source and sink onto a shared/default destination.
        let toml = r#"
[source]
url = "http://localhost:6335"
collection = "src"
[compressor]
method = "sparse_ae"
latent = 128
sparse_weight = 0.01
[sink]
url = "http://localhost:6335"
collection = "dst"
"#;
        let cfg = PipelineConfig::from_toml(toml).unwrap();
        assert_eq!(cfg.compressor.method, Method::SparseAe);
        assert_eq!(cfg.compressor.sparse_weight, Some(0.01));
    }
}
