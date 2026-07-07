//! Interpretability subsystem — analyses that explain *how* a trained model
//! arrives at its predictions, served under `/api/interp/*` and surfaced in the
//! UI's Interpretability section. Built to house many tools over time; portable
//! to the upstream framework.
//!
//! Tool #1 — **layer-wise linear probes** (Alain & Bengio 2016, arXiv:1610.01644):
//! fit a cheap ridge probe on the activations at each stage of a network and
//! report how linearly decodable the target is by depth. The heavy lifting lives
//! in the predictor's `layer-probe` subcommand (keeps the `burn` dependency out
//! of the server); here we orchestrate it as an async job exactly like a dataset
//! build and poll it through the existing `GET /api/jobs/{id}`. Which families
//! can be probed is declared in `registry.toml` via `probe_args` (only the
//! native burn nets, whose activations are reachable from Rust, set it).

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::api::{bad_request, not_found, ApiError};
use crate::state::{AppState, JobStatus};
use crate::{models, registry, runs};

#[derive(Deserialize)]
pub struct LayerProbeRequest {
    /// Model directory name under `data/models/`.
    pub model: String,
    /// Dataset id to probe on; defaults to the model's training dataset. Must
    /// match the model's `n_cols`.
    #[serde(default)]
    pub dataset: Option<String>,
    /// Ridge penalty; `0`/absent ⇒ pick by a small internal hold-out CV.
    #[serde(default)]
    pub lambda: f64,
}

#[derive(Deserialize)]
pub struct CompareRequest {
    /// Two or more model directory names to probe and overlay.
    pub models: Vec<String>,
    /// Shared dataset id; defaults to the first model's training dataset. Every
    /// model is probed on this same dataset so the curves are comparable.
    #[serde(default)]
    pub dataset: Option<String>,
    /// Ridge penalty; `0`/absent ⇒ pick by a small internal hold-out CV.
    #[serde(default)]
    pub lambda: f64,
}

/// `GET /api/interp/models` — the promoted models the layer probe can analyze:
/// any whose predictor family declares `probe_args` in the registry (the native
/// burn nets). Each carries its predictor, training dataset, `n_cols`, and
/// hyperparams so the UI can describe it and offer width-matching datasets.
pub async fn list_models(State(state): State<Arc<AppState>>) -> Result<Json<Value>, ApiError> {
    let dir = state.models_dir();
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for e in rd.flatten() {
            let p = e.path();
            if !p.is_dir() {
                continue;
            }
            let record = match models::read_record(&p) {
                Ok(r) => r,
                Err(_) => continue,
            };
            // Only families with a layer-probe subcommand declared in the registry.
            let probeable =
                state.registry.get(&record.predictor).map(|pr| pr.supports_probe()).unwrap_or(false);
            if !probeable {
                continue;
            }
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
            let n_cols = models::read_contract(&p).ok().map(|c| c.n_cols);
            let hyperparams: Value = std::fs::read_to_string(p.join("hyperparams.json"))
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or(Value::Null);
            // `hidden` + `activation` are surfaced top-level (the UI's MLP
            // descriptor); other families leave `hidden` empty.
            let hidden = hyperparams.get("hidden").cloned().unwrap_or_else(|| json!([]));
            let activation =
                hyperparams.get("activation").and_then(|a| a.as_str()).unwrap_or("relu");
            out.push(json!({
                "name": name,
                "predictor": record.predictor,
                "dataset_id": record.dataset_id,
                "n_cols": n_cols,
                "hidden": hidden,
                "activation": activation,
                "hyperparams": hyperparams,
            }));
        }
    }
    out.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    Ok(Json(json!({ "models": out })))
}

/// `POST /api/interp/layer-probe` — start the analysis (async). Returns a
/// `{ job_id }`; poll `GET /api/jobs/{id}` until `done`, whose `result` is the
/// layer-probe JSON (per-stage probe metrics). Mirrors the dataset-build job
/// pattern, capped by the same `build_slots` semaphore.
pub async fn start_layer_probe(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LayerProbeRequest>,
) -> Result<Json<Value>, ApiError> {
    let model_dir = state.models_dir().join(&req.model);
    let record = models::read_record(&model_dir).map_err(|_| not_found("model"))?;
    let predictor = state
        .registry
        .get(&record.predictor)
        .ok_or_else(|| bad_request(format!("predictor {} is not registered", record.predictor)))?;
    let (command, args_template) = predictor
        .probe_invocation()
        .map(|(c, a)| (c.to_string(), a.to_vec()))
        .ok_or_else(|| {
            bad_request(format!(
                "predictor {} has no layer-probe support",
                record.predictor
            ))
        })?;

    let dataset = req.dataset.clone().unwrap_or_else(|| record.dataset_id.clone());
    let dataset_dir = state.datasets_dir().join(&dataset);
    if !dataset_dir.join("manifest.json").exists() {
        return Err(not_found("dataset"));
    }

    let job_id = format!("interp-{}", runs::new_run_id("lp"));
    state
        .jobs
        .lock()
        .unwrap()
        .insert(job_id.clone(), JobStatus::Running { stage: "queued".into() });

    let out_file = state.root.join("data/interp").join(&job_id).join("layer-probe.json");
    let mut args = registry::substitute(
        &args_template,
        &[
            ("model", model_dir.to_string_lossy().into_owned()),
            ("dataset", dataset_dir.to_string_lossy().into_owned()),
            ("output", out_file.to_string_lossy().into_owned()),
        ],
    );
    if req.lambda > 0.0 {
        args.push("--lambda".into());
        args.push(req.lambda.to_string());
    }
    let root = state.root.clone();

    let st = state.clone();
    let id = job_id.clone();
    tokio::spawn(async move {
        let _permit = st.build_slots.clone().acquire_owned().await.unwrap();
        let st2 = st.clone();
        let id2 = id.clone();
        // Mirror engine `{"event":"log","msg":...}` lines into the job stage so
        // the UI shows live progress through the polled job status.
        let on_line = move |line: &str| {
            if let Ok(v) = serde_json::from_str::<Value>(line) {
                if v.get("event").and_then(|e| e.as_str()) == Some("log") {
                    if let Some(msg) = v.get("msg").and_then(|m| m.as_str()) {
                        st2.jobs
                            .lock()
                            .unwrap()
                            .insert(id2.clone(), JobStatus::Running { stage: msg.to_string() });
                    }
                }
            }
        };
        let res = runs::spawn_and_capture(&command, &args, &root, &on_line).await;
        let status = match res {
            Ok((0, _)) => match std::fs::read_to_string(&out_file)
                .ok()
                .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            {
                Some(result) => JobStatus::Done { result },
                None => JobStatus::Failed {
                    error: "layer-probe finished but produced no result file".into(),
                },
            },
            Ok((code, stderr)) => JobStatus::Failed {
                error: format!("layer-probe engine exited {code}: {stderr}"),
            },
            Err(e) => JobStatus::Failed { error: format!("{e:#}") },
        };
        st.jobs.lock().unwrap().insert(id, status);
    });

    Ok(Json(json!({ "job_id": job_id })))
}

/// `POST /api/interp/layer-probe/compare` — probe several models on one shared
/// dataset in a single async job so their decodability-by-depth curves can be
/// overlaid. Returns `{ job_id }`; poll `GET /api/jobs/{id}`, whose `result` is
/// `{ dataset_id, reports: [{ model, report, error }] }` (a model whose probe
/// fails carries a non-null `error` and a null `report`, never sinking the rest).
pub async fn start_layer_probe_compare(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompareRequest>,
) -> Result<Json<Value>, ApiError> {
    if req.models.len() < 2 {
        return Err(bad_request("comparison needs at least two models".into()));
    }

    // Resolve every model to its (command, args template) up front so a bad
    // request fails synchronously rather than mid-job. The shared dataset
    // defaults to the first model's training dataset.
    struct Plan {
        model: String,
        command: String,
        args_template: Vec<String>,
    }
    let mut plans = Vec::with_capacity(req.models.len());
    let mut default_dataset: Option<String> = None;
    for name in &req.models {
        let model_dir = state.models_dir().join(name);
        let record = models::read_record(&model_dir).map_err(|_| not_found("model"))?;
        default_dataset.get_or_insert_with(|| record.dataset_id.clone());
        let predictor = state.registry.get(&record.predictor).ok_or_else(|| {
            bad_request(format!("predictor {} is not registered", record.predictor))
        })?;
        let (c, a) = predictor
            .probe_invocation()
            .map(|(c, a)| (c.to_string(), a.to_vec()))
            .ok_or_else(|| {
                bad_request(format!("predictor {} has no layer-probe support", record.predictor))
            })?;
        plans.push(Plan { model: name.clone(), command: c, args_template: a });
    }
    let dataset = req.dataset.clone().or(default_dataset).unwrap();
    let dataset_dir = state.datasets_dir().join(&dataset);
    if !dataset_dir.join("manifest.json").exists() {
        return Err(not_found("dataset"));
    }

    let job_id = format!("interp-{}", runs::new_run_id("cmp"));
    state
        .jobs
        .lock()
        .unwrap()
        .insert(job_id.clone(), JobStatus::Running { stage: "queued".into() });

    let lambda = req.lambda;
    let root = state.root.clone();
    let base = state.root.join("data/interp").join(&job_id);
    let dataset_dir_s = dataset_dir.to_string_lossy().into_owned();
    let dataset_id = dataset.clone();
    let st = state.clone();
    let id = job_id.clone();
    tokio::spawn(async move {
        let _permit = st.build_slots.clone().acquire_owned().await.unwrap();
        let n = plans.len();
        let mut reports: Vec<Value> = Vec::with_capacity(n);
        for (i, plan) in plans.iter().enumerate() {
            st.jobs.lock().unwrap().insert(
                id.clone(),
                JobStatus::Running { stage: format!("probing {} ({}/{})", plan.model, i + 1, n) },
            );
            // Model names are validated `^[a-z0-9][a-z0-9-]{0,63}$`, so they are
            // safe as a path segment.
            let out_file = base.join(format!("{}.json", plan.model));
            let mut args = registry::substitute(
                &plan.args_template,
                &[
                    ("model", st.models_dir().join(&plan.model).to_string_lossy().into_owned()),
                    ("dataset", dataset_dir_s.clone()),
                    ("output", out_file.to_string_lossy().into_owned()),
                ],
            );
            if lambda > 0.0 {
                args.push("--lambda".into());
                args.push(lambda.to_string());
            }
            let res = runs::spawn_and_capture(&plan.command, &args, &root, &|_| {}).await;
            let entry = match res {
                Ok((0, _)) => match std::fs::read_to_string(&out_file)
                    .ok()
                    .and_then(|s| serde_json::from_str::<Value>(&s).ok())
                {
                    Some(report) => json!({ "model": plan.model, "report": report, "error": Value::Null }),
                    None => json!({ "model": plan.model, "report": Value::Null,
                        "error": "probe produced no result file" }),
                },
                Ok((code, stderr)) => json!({ "model": plan.model, "report": Value::Null,
                    "error": format!("exited {code}: {stderr}") }),
                Err(e) => json!({ "model": plan.model, "report": Value::Null, "error": format!("{e:#}") }),
            };
            reports.push(entry);
        }
        st.jobs.lock().unwrap().insert(
            id,
            JobStatus::Done { result: json!({ "dataset_id": dataset_id, "reports": reports }) },
        );
    });

    Ok(Json(json!({ "job_id": job_id })))
}

// ---------- embedding probe (P1): absent vs unused diagnostic ----------

#[derive(Deserialize)]
pub struct EmbeddingProbeRequest {
    /// Dataset id whose embedding (PCA) block is probed (ideally full-rank).
    pub dataset: String,
    /// One-hot categorical group to slice rows by (must exist in the dataset).
    /// Defaults to the domain's `[interp].segment_field`.
    #[serde(default)]
    pub split_by: Option<String>,
    /// Skip the slower nonlinear MLP ceiling probe + segment-class decode.
    #[serde(default)]
    pub no_mlp: bool,
}

/// `POST /api/interp/embedding-probe` — P1: on a dataset artifact (no model),
/// fit linear + MLP probes on the raw embeddings vs PCA-128 sliced into segments
/// by a categorical one-hot group (`split_by`, default: the domain's interp segment field), against a
/// per-segment-median floor, and decide whether the hardest segment's error is
/// information-ABSENT or merely UNUSED. Returns `{ job_id }`; poll
/// `GET /api/jobs/{id}`. Runs in the `burn-mlp` engine binary (needs burn).
pub async fn start_embedding_probe(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingProbeRequest>,
) -> Result<Json<Value>, ApiError> {
    let dataset_dir = state.datasets_dir().join(&req.dataset);
    if !dataset_dir.join("manifest.json").exists() {
        return Err(not_found("dataset"));
    }
    let command = state
        .registry
        .get("burn-mlp")
        .map(|p| p.command.clone())
        .ok_or_else(|| bad_request("burn-mlp engine is not registered".into()))?;

    let job_id = format!("interp-{}", runs::new_run_id("ep"));
    state
        .jobs
        .lock()
        .unwrap()
        .insert(job_id.clone(), JobStatus::Running { stage: "queued".into() });

    let out_file = state.root.join("data/interp").join(&job_id).join("embedding-probe.json");
    let mut args = vec![
        "embedding-probe".to_string(),
        "--dataset".into(),
        dataset_dir.to_string_lossy().into_owned(),
        "--output".into(),
        out_file.to_string_lossy().into_owned(),
    ];
    if let Some(split_by) = req.split_by.as_deref().filter(|s| !s.is_empty()) {
        args.push("--split-by".into());
        args.push(split_by.to_string());
    }
    if req.no_mlp {
        args.push("--no-mlp".into());
    }
    let root = state.root.clone();

    let st = state.clone();
    let id = job_id.clone();
    tokio::spawn(async move {
        let _permit = st.build_slots.clone().acquire_owned().await.unwrap();
        let st2 = st.clone();
        let id2 = id.clone();
        let on_line = move |line: &str| {
            if let Ok(v) = serde_json::from_str::<Value>(line) {
                if v.get("event").and_then(|e| e.as_str()) == Some("log") {
                    if let Some(msg) = v.get("msg").and_then(|m| m.as_str()) {
                        st2.jobs
                            .lock()
                            .unwrap()
                            .insert(id2.clone(), JobStatus::Running { stage: msg.to_string() });
                    }
                }
            }
        };
        let res = runs::spawn_and_capture(&command, &args, &root, &on_line).await;
        let status = match res {
            Ok((0, _)) => match std::fs::read_to_string(&out_file)
                .ok()
                .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            {
                Some(result) => JobStatus::Done { result },
                None => JobStatus::Failed {
                    error: "embedding probe finished but produced no result file".into(),
                },
            },
            Ok((code, stderr)) => JobStatus::Failed {
                error: format!("embedding probe engine exited {code}: {stderr}"),
            },
            Err(e) => JobStatus::Failed { error: format!("{e:#}") },
        };
        st.jobs.lock().unwrap().insert(id, status);
    });

    Ok(Json(json!({ "job_id": job_id })))
}

// ---------- Tool #2: sparse-autoencoder (dictionary learning) ----------

#[derive(Deserialize)]
pub struct SaeRequest {
    /// Dataset id whose embedding (PCA) block the SAE is trained on.
    pub dataset: String,
    #[serde(default = "sae_max_dims")]
    pub max_dims: usize,
    /// Atom count. Default 2048: overcomplete enough to be a dictionary, but not
    /// so wide that the code probe over-fits a small segment's test rows.
    #[serde(default = "sae_n_atoms")]
    pub n_atoms: usize,
    #[serde(default = "sae_l1")]
    pub l1: f64,
    #[serde(default = "sae_epochs")]
    pub epochs: usize,
    /// Categorical value to slice for the segment probe; empty ⇒ pooled only.
    #[serde(default = "sae_segment")]
    pub segment: String,
    /// Label the top atoms with an LLM (auto-interp). No-op on the engine side
    /// unless OPENAI_API_KEY is set in the server's environment.
    #[serde(default)]
    pub label_atoms: bool,
}
fn sae_max_dims() -> usize {
    1536
}
fn sae_n_atoms() -> usize {
    2048
}
fn sae_l1() -> f64 {
    0.0015
}
fn sae_epochs() -> usize {
    40
}
fn sae_segment() -> String {
    String::new()
}

/// `POST /api/interp/sae` — P2: train a sparse autoencoder on a dataset's
/// embedding block, then probe its code against the raw embeddings (pooled + the
/// segment) and surface the most interpretable atoms. Returns `{ job_id }`; poll
/// `GET /api/jobs/{id}`, whose `result` is the SAE analysis JSON. Shells out to
/// the `lensing-sae` engine binary (keeps `burn` out of the server). This tool is
/// dataset-centric (not model-centric) — it trains on the embeddings directly.
pub async fn start_sae(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SaeRequest>,
) -> Result<Json<Value>, ApiError> {
    let dataset_dir = state.datasets_dir().join(&req.dataset);
    if !dataset_dir.join("manifest.json").exists() {
        return Err(not_found("dataset"));
    }
    let bin = state.root.join("target/release/lensing-sae");
    if !bin.exists() {
        return Err(bad_request(
            "lensing-sae engine not built — run `cargo build --release -p lensing-sae`".into(),
        ));
    }

    let job_id = format!("interp-{}", runs::new_run_id("sae"));
    state
        .jobs
        .lock()
        .unwrap()
        .insert(job_id.clone(), JobStatus::Running { stage: "queued".into() });

    let out_file = state.root.join("data/interp").join(&job_id).join("sae.json");
    let mut args: Vec<String> = vec![
        "analyze".into(),
        "--dataset".into(),
        dataset_dir.to_string_lossy().into_owned(),
        "--output".into(),
        out_file.to_string_lossy().into_owned(),
        "--max-dims".into(),
        req.max_dims.to_string(),
        "--n-atoms".into(),
        req.n_atoms.to_string(),
        "--l1".into(),
        req.l1.to_string(),
        "--epochs".into(),
        req.epochs.to_string(),
        "--segment".into(),
        req.segment.clone(),
        "--cache-dir".into(),
        state.root.join("data/interp/sae-cache").to_string_lossy().into_owned(),
    ];
    if req.label_atoms {
        args.push("--label-atoms".into());
    }
    let command = bin.to_string_lossy().into_owned();
    let root = state.root.clone();
    let st = state.clone();
    let id = job_id.clone();
    tokio::spawn(async move {
        let _permit = st.build_slots.clone().acquire_owned().await.unwrap();
        let st2 = st.clone();
        let id2 = id.clone();
        let on_line = move |line: &str| {
            if let Ok(v) = serde_json::from_str::<Value>(line) {
                if v.get("event").and_then(|e| e.as_str()) == Some("log") {
                    if let Some(msg) = v.get("msg").and_then(|m| m.as_str()) {
                        st2.jobs
                            .lock()
                            .unwrap()
                            .insert(id2.clone(), JobStatus::Running { stage: msg.to_string() });
                    }
                }
            }
        };
        let res = runs::spawn_and_capture(&command, &args, &root, &on_line).await;
        let status = match res {
            Ok((0, _)) => match std::fs::read_to_string(&out_file)
                .ok()
                .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            {
                Some(result) => JobStatus::Done { result },
                None => JobStatus::Failed {
                    error: "sae engine finished but produced no result file".into(),
                },
            },
            Ok((code, stderr)) => JobStatus::Failed {
                error: format!("sae engine exited {code}: {stderr}"),
            },
            Err(e) => JobStatus::Failed { error: format!("{e:#}") },
        };
        st.jobs.lock().unwrap().insert(id, status);
    });

    Ok(Json(json!({ "job_id": job_id })))
}
