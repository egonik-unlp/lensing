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

use std::collections::HashSet;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::Json;
use chrono::Utc;
use lensing_core::InterpAnalysis;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::api::{bad_request, not_found, ApiError};
use crate::state::{AppState, JobStatus};
use crate::{models, registry, runs};

// ---------- persistence helpers (shared by every interp tool) ----------

/// Register a starting interp job: the in-memory `Running` status the UI polls,
/// plus — when the database is up — a `running` row so the analysis is listable
/// the moment it is queued.
fn begin_interp_job(state: &AppState, analysis: &InterpAnalysis) {
    state
        .jobs
        .lock()
        .unwrap()
        .insert(analysis.id.clone(), JobStatus::Running { stage: "queued".into() });
    if let Some(sink) = &state.db_sink {
        sink.interp_upsert(analysis);
    }
}

/// Record a finished interp job: update the in-memory status (unchanged
/// live-polling behavior) and mirror the terminal row — `result`/`error` plus
/// `finished_at` — to the database. Callers pass the `running` analysis they
/// began with; this stamps the outcome onto it.
fn finish_interp_job(state: &AppState, mut analysis: InterpAnalysis, status: JobStatus) {
    match &status {
        JobStatus::Done { result } => {
            analysis.status = "done".into();
            analysis.result = Some(result.clone());
        }
        JobStatus::Failed { error } => {
            analysis.status = "failed".into();
            analysis.error = Some(error.clone());
        }
        JobStatus::Running { .. } => {}
    }
    analysis.finished_at = Some(Utc::now().to_rfc3339());
    if let Some(sink) = &state.db_sink {
        sink.interp_upsert(&analysis);
    }
    state.jobs.lock().unwrap().insert(analysis.id.clone(), status);
}

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
            let supports_model_sae = state
                .registry
                .get(&record.predictor)
                .map(|pr| pr.supports_model_sae())
                .unwrap_or(false);
            out.push(json!({
                "name": name,
                "predictor": record.predictor,
                "dataset_id": record.dataset_id,
                "n_cols": n_cols,
                "hidden": hidden,
                "activation": activation,
                "hyperparams": hyperparams,
                "supports_model_sae": supports_model_sae,
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
    let analysis = InterpAnalysis::running(
        job_id.clone(),
        "layer-probe",
        Some(req.model.clone()),
        dataset.clone(),
        Some(record.predictor.clone()),
        json!({ "lambda": req.lambda }),
        "manual",
        Utc::now().to_rfc3339(),
    );
    begin_interp_job(&state, &analysis);

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
        finish_interp_job(&st, analysis, status);
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
    let analysis = InterpAnalysis::running(
        job_id.clone(),
        "layer-probe-compare",
        None,
        dataset.clone(),
        None,
        json!({ "models": req.models, "lambda": req.lambda }),
        "manual",
        Utc::now().to_rfc3339(),
    );
    begin_interp_job(&state, &analysis);

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
        finish_interp_job(
            &st,
            analysis,
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
    let analysis = InterpAnalysis::running(
        job_id.clone(),
        "embedding-probe",
        None,
        req.dataset.clone(),
        None,
        json!({ "split_by": req.split_by, "no_mlp": req.no_mlp }),
        "manual",
        Utc::now().to_rfc3339(),
    );
    begin_interp_job(&state, &analysis);

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
        finish_interp_job(&st, analysis, status);
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
    let analysis = InterpAnalysis::running(
        job_id.clone(),
        "sae",
        None,
        req.dataset.clone(),
        None,
        json!({
            "max_dims": req.max_dims, "n_atoms": req.n_atoms, "l1": req.l1,
            "epochs": req.epochs, "segment": req.segment, "label_atoms": req.label_atoms,
        }),
        "manual",
        Utc::now().to_rfc3339(),
    );
    begin_interp_job(&state, &analysis);

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
        finish_interp_job(&st, analysis, status);
    });

    Ok(Json(json!({ "job_id": job_id })))
}

#[derive(Deserialize)]
pub struct ModelSaeRequest {
    /// Model directory name under `data/models/` (must be an MLP-family model
    /// whose predictor declares `model_sae_args`).
    pub model: String,
    /// Dataset id to analyze on; defaults to the model's training dataset. Must
    /// match the model's `n_cols`.
    #[serde(default)]
    pub dataset: Option<String>,
    /// Comma-separated 1-based hidden layers to analyze; empty ⇒ all.
    #[serde(default)]
    pub layers: String,
    /// SAE atom count; `0`/absent ⇒ 2× the layer width.
    #[serde(default)]
    pub n_atoms: usize,
    #[serde(default = "sae_l1")]
    pub l1: f64,
    #[serde(default = "sae_epochs")]
    pub epochs: usize,
    /// Label the surfaced atoms with an LLM (auto-interp). No-op on the engine
    /// side unless OPENAI_API_KEY is set in the server's environment.
    #[serde(default)]
    pub label_atoms: bool,
    /// Also diff against the embedding: first train a dataset SAE (with codes)
    /// on the same dataset, then report target-relevant embedding concepts the
    /// model drops. Roughly doubles the run time.
    #[serde(default)]
    pub compare_embedding: bool,
}

/// `POST /api/interp/model-sae` — the per-model sibling of `/sae`: train a sparse
/// autoencoder on a promoted model's OWN hidden activations (per layer) and
/// report capacity, per-segment representation, concept-vs-decodability depth,
/// and — with `compare_embedding` — the target-relevant signal the model drops
/// relative to the embedding. Returns `{ job_id }`; poll `GET /api/jobs/{id}`,
/// whose `result` is the model-SAE JSON. Shells out to the predictor's own
/// `model-sae` subcommand (keeps `burn` out of the server).
pub async fn start_model_sae(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ModelSaeRequest>,
) -> Result<Json<Value>, ApiError> {
    let job_id = spawn_model_sae(state, req, "manual").await?;
    Ok(Json(json!({ "job_id": job_id })))
}

/// Start a per-model SAE analysis and return its job id. Shared by the HTTP
/// handler (`source = "manual"`) and the promotion auto-queue
/// (`source = "auto"`): it validates synchronously, registers the job (in-memory
/// + a persisted `running` row), spawns the engine work, and mirrors the
/// terminal row on completion. Takes a plain `Arc<AppState>` / `ModelSaeRequest`
/// so it is callable off the HTTP path.
pub async fn spawn_model_sae(
    state: Arc<AppState>,
    req: ModelSaeRequest,
    source: &str,
) -> Result<String, ApiError> {
    let model_dir = state.models_dir().join(&req.model);
    let record = models::read_record(&model_dir).map_err(|_| not_found("model"))?;
    let predictor = state
        .registry
        .get(&record.predictor)
        .ok_or_else(|| bad_request(format!("predictor {} is not registered", record.predictor)))?;
    let (command, args_template) = predictor
        .model_sae_invocation()
        .map(|(c, a)| (c.to_string(), a.to_vec()))
        .ok_or_else(|| {
            bad_request(format!("predictor {} has no per-model SAE support", record.predictor))
        })?;

    let dataset = req.dataset.clone().unwrap_or_else(|| record.dataset_id.clone());
    let dataset_dir = state.datasets_dir().join(&dataset);
    if !dataset_dir.join("manifest.json").exists() {
        return Err(not_found("dataset"));
    }

    // The embedding diff needs the lensing-sae engine to produce a codes-bearing
    // dataset SAE first.
    let sae_bin = state.root.join("target/release/lensing-sae");
    if req.compare_embedding && !sae_bin.exists() {
        return Err(bad_request(
            "lensing-sae engine not built — run `cargo build --release -p lensing-sae`".into(),
        ));
    }

    let job_id = format!("interp-{}", runs::new_run_id("msae"));
    let analysis = InterpAnalysis::running(
        job_id.clone(),
        "model-sae",
        Some(req.model.clone()),
        dataset.clone(),
        Some(record.predictor.clone()),
        json!({
            "layers": req.layers, "n_atoms": req.n_atoms, "l1": req.l1, "epochs": req.epochs,
            "label_atoms": req.label_atoms, "compare_embedding": req.compare_embedding,
        }),
        source,
        Utc::now().to_rfc3339(),
    );
    begin_interp_job(&state, &analysis);

    let job_dir = state.root.join("data/interp").join(&job_id);
    let out_file = job_dir.join("model-sae.json");
    let ds_sae_file = job_dir.join("dataset-sae.json");
    let cache_dir = state.root.join("data/interp/sae-cache");

    let mut args = registry::substitute(
        &args_template,
        &[
            ("model", model_dir.to_string_lossy().into_owned()),
            ("dataset", dataset_dir.to_string_lossy().into_owned()),
            ("output", out_file.to_string_lossy().into_owned()),
        ],
    );
    if !req.layers.trim().is_empty() {
        args.push("--layers".into());
        args.push(req.layers.clone());
    }
    args.push("--n-atoms".into());
    args.push(req.n_atoms.to_string());
    args.push("--l1".into());
    args.push(req.l1.to_string());
    args.push("--epochs".into());
    args.push(req.epochs.to_string());
    args.push("--cache-dir".into());
    args.push(cache_dir.to_string_lossy().into_owned());
    if req.label_atoms {
        args.push("--label-atoms".into());
    }
    if req.compare_embedding {
        args.push("--dataset-sae".into());
        args.push(ds_sae_file.to_string_lossy().into_owned());
    }

    // Args for the prerequisite dataset SAE (only used when comparing). Label
    // its atoms too when labelling was requested, so the embedding concepts in
    // the "dropped signal vs. the embedding" section carry names, not just ids.
    let mut sae_args: Vec<String> = vec![
        "analyze".into(),
        "--dataset".into(),
        dataset_dir.to_string_lossy().into_owned(),
        "--output".into(),
        ds_sae_file.to_string_lossy().into_owned(),
        "--l1".into(),
        req.l1.to_string(),
        "--epochs".into(),
        req.epochs.to_string(),
        "--emit-codes".into(),
        "--cache-dir".into(),
        cache_dir.to_string_lossy().into_owned(),
    ];
    if req.label_atoms {
        sae_args.push("--label-atoms".into());
    }
    let sae_command = sae_bin.to_string_lossy().into_owned();
    let compare = req.compare_embedding;
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

        // Stage 1 (optional): the embedding SAE with codes, for the diff.
        if compare {
            match runs::spawn_and_capture(&sae_command, &sae_args, &root, &on_line).await {
                Ok((0, _)) => {}
                Ok((code, stderr)) => {
                    finish_interp_job(
                        &st,
                        analysis,
                        JobStatus::Failed {
                            error: format!("dataset-sae engine exited {code}: {stderr}"),
                        },
                    );
                    return;
                }
                Err(e) => {
                    finish_interp_job(&st, analysis, JobStatus::Failed { error: format!("{e:#}") });
                    return;
                }
            }
        }

        // Stage 2: the per-model SAE (reads the stage-1 codes when comparing).
        let res = runs::spawn_and_capture(&command, &args, &root, &on_line).await;
        let status = match res {
            Ok((0, _)) => match std::fs::read_to_string(&out_file)
                .ok()
                .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            {
                Some(result) => JobStatus::Done { result },
                None => JobStatus::Failed {
                    error: "model-sae engine finished but produced no result file".into(),
                },
            },
            Ok((code, stderr)) => JobStatus::Failed {
                error: format!("model-sae engine exited {code}: {stderr}"),
            },
            Err(e) => JobStatus::Failed { error: format!("{e:#}") },
        };
        finish_interp_job(&st, analysis, status);
    });

    Ok(job_id)
}

/// Queue a per-model SAE analysis for a freshly promoted model — fire-and-forget,
/// mirroring [`crate::best_models::spawn_recompute`]. It no-ops unless auto-queue
/// is enabled (`LENSING_AUTO_MODEL_SAE`) and the model's predictor supports model-SAE,
/// and dedups against an existing non-failed analysis so a model is analyzed at
/// most once. The auto config is the embedding-diff read WITHOUT GPT labels;
/// richer/labeled runs are launched manually and persist the same way.
pub fn auto_queue_model_sae(state: Arc<AppState>, model: String) {
    if !state.auto_model_sae {
        return;
    }
    tokio::spawn(async move {
        // Only MLP-family models expose a hidden activation space to decompose.
        let Ok(record) = models::read_record(&state.models_dir().join(&model)) else {
            return;
        };
        let supported = state
            .registry
            .get(&record.predictor)
            .map(|p| p.supports_model_sae())
            .unwrap_or(false);
        if !supported {
            return;
        }
        // Dedup: promotion should queue at most one analysis per model.
        if let Some(db) = &state.db {
            match lensing_db::queries::model_has_interp_analysis(db, &model).await {
                Ok(true) => return,
                Ok(false) => {}
                Err(e) => eprintln!("[lensing-server] auto model-sae dedup check for {model}: {e:#}"),
            }
        }
        let req = ModelSaeRequest {
            model: model.clone(),
            dataset: None,
            layers: String::new(),
            n_atoms: 0,
            l1: sae_l1(),
            epochs: sae_epochs(),
            label_atoms: false,
            compare_embedding: true,
        };
        if let Err(e) = spawn_model_sae(state.clone(), req, "auto").await {
            eprintln!("[lensing-server] auto model-sae for {model} did not start: {}", e.message());
        }
    });
}

// ---------- persisted analyses: list / fetch / delete ----------

#[derive(Deserialize)]
pub struct AnalysesQuery {
    #[serde(default)]
    pub tool: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub dataset: Option<String>,
}

/// `GET /api/interp/analyses` — persisted analyses (metadata only, newest first),
/// optionally filtered by `tool` / `model` / `dataset`. DB-first, with a disk
/// merge for any job dir not yet mirrored (degraded mode / pre-backfill).
pub async fn list_analyses(
    State(state): State<Arc<AppState>>,
    Query(q): Query<AnalysesQuery>,
) -> Result<Json<Value>, ApiError> {
    let mut out: Vec<InterpAnalysis> = Vec::new();
    let mut known: HashSet<String> = HashSet::new();
    if let Some(db) = &state.db {
        if let Ok(rows) = lensing_db::queries::load_interp_analyses(db).await {
            for a in rows {
                known.insert(a.id.clone());
                out.push(a);
            }
        }
    }
    let interp_root = state.root.join("data/interp");
    let models_root = state.models_dir();
    for id in interp_dir_ids(&interp_root) {
        if known.contains(&id) {
            continue;
        }
        if let Some(a) = InterpAnalysis::from_disk(&interp_root, &models_root, &id, false) {
            out.push(a);
        }
    }
    out.sort_by(|a, b| b.created_at.cmp(&a.created_at).then_with(|| b.id.cmp(&a.id)));
    if let Some(t) = q.tool.as_deref().filter(|s| !s.is_empty()) {
        out.retain(|a| a.tool == t);
    }
    if let Some(m) = q.model.as_deref().filter(|s| !s.is_empty()) {
        out.retain(|a| a.model.as_deref() == Some(m));
    }
    if let Some(d) = q.dataset.as_deref().filter(|s| !s.is_empty()) {
        out.retain(|a| a.dataset_id == d);
    }
    Ok(Json(json!({ "analyses": out })))
}

/// `GET /api/interp/analyses/{id}` — one analysis WITH its full result document.
/// DB-first; falls back to reading the on-disk result file.
pub async fn get_analysis(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<InterpAnalysis>, ApiError> {
    if let Some(db) = &state.db {
        if let Ok(Some(a)) = lensing_db::queries::load_interp_analysis(db, &id).await {
            return Ok(Json(a));
        }
    }
    let interp_root = state.root.join("data/interp");
    InterpAnalysis::from_disk(&interp_root, &state.models_dir(), &id, true)
        .map(Json)
        .ok_or_else(|| not_found("analysis"))
}

/// `DELETE /api/interp/analyses/{id}` — drop the row and its job dir.
pub async fn delete_analysis(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    if let Some(sink) = &state.db_sink {
        sink.interp_delete(&id);
    }
    let dir = state.root.join("data/interp").join(&id);
    if dir.is_dir() {
        let _ = std::fs::remove_dir_all(&dir);
    }
    Ok(Json(json!({ "ok": true })))
}

/// Job-dir ids under `data/interp` (excludes the `sae-cache` dir).
fn interp_dir_ids(interp_root: &std::path::Path) -> Vec<String> {
    let mut ids = Vec::new();
    if let Ok(rd) = std::fs::read_dir(interp_root) {
        for e in rd.flatten() {
            if !e.path().is_dir() {
                continue;
            }
            if let Some(name) = e.file_name().to_str() {
                if name.starts_with("interp-") {
                    ids.push(name.to_string());
                }
            }
        }
    }
    ids
}
