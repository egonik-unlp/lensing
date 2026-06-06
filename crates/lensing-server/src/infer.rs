//! Inference role: `lensing-server infer --port <p>`.
//!
//! A slim, separately-deployable prediction service: it materializes every
//! promoted model from the database (`model_artifacts` — record, contract,
//! hyperparams, PCA basis, weights) into a local cache and serves ONLY the
//! model/predict surface. No UI, no run orchestration, no dataset builds.
//!
//! Requirements on the inference host: this binary + `registry.toml` +
//! `domain.toml` + the predict toolchains of the model families it serves
//! (predictor binaries / venv), plus DATABASE_URL. Qdrant access is only
//! needed for `point_ids` predicts; inline-`items` predicts work without it.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use axum::routing::get;
use axum::Router;
use tokio::sync::Semaphore;

use crate::registry::Registry;
use crate::state::AppState;
use crate::{api, definitions};

pub async fn run(
    root: PathBuf,
    database_url: String,
    port: u16,
    qdrant_url: String,
    max_concurrent: usize,
) -> Result<()> {
    let db = lensing_db::connect(&database_url)
        .await
        .context("inference role requires the metadata database")?;
    let domain = lensing_core::domain::Domain::load_or_default(&root)?;
    let registry = Registry::load(&root.join("registry.toml"))?;

    // Materialize promoted models from the database into the local cache.
    let models_dir = root.join("data/models");
    std::fs::create_dir_all(&models_dir)?;
    let names = lensing_db::queries::model_artifact_names(&db).await?;
    let mut materialized = 0usize;
    for name in &names {
        let dir = models_dir.join(name);
        if dir.join("record.json").is_file() {
            continue; // already cached
        }
        let files = lensing_db::queries::load_model_artifacts(&db, name).await?;
        for (rel, bytes) in &files {
            // Artifact paths come from our own uploader, but never let a
            // stored path escape the model dir.
            anyhow::ensure!(
                !rel.contains("..") && !rel.starts_with('/'),
                "model {name}: unsafe artifact path {rel:?}"
            );
            let path = dir.join(rel);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, bytes)?;
        }
        materialized += 1;
    }
    eprintln!(
        "[lensing-server infer] {} models available ({materialized} materialized from the database)",
        names.len()
    );

    // Collection default from the domain (only used by point_ids predicts).
    let collection = domain.corpus.collection.clone();
    let manual_collection = domain.corpus.manual_collection.clone();
    let state = Arc::new(AppState {
        root: root.clone(),
        domain: Arc::new(domain),
        registry,
        qdrant_url,
        collection,
        manual_collection,
        openai_api_key: None,
        embedding_model: String::new(),
        openai_base_url: String::new(),
        db: Some(db),
        db_sink: None, // read-only role: never mirrors writes
        definitions: tokio::sync::Mutex::new(definitions::Definitions::default()),
        live_runs: Mutex::new(Default::default()),
        builds: Mutex::new(Default::default()),
        jobs: Mutex::new(Default::default()),
        run_slots: Arc::new(Semaphore::new(max_concurrent)),
        build_slots: Arc::new(Semaphore::new(1)),
        best_models_lock: tokio::sync::Mutex::new(()),
    });

    // Materialize the best-models group from the database so the predict
    // surface includes the consensus endpoint (read-only: no recompute here).
    if let Some(db) = &state.db {
        match lensing_db::queries::load_best_models(db).await {
            Ok(entries) if !entries.is_empty() => {
                let group = lensing_core::BestModelGroup {
                    primary_metric: state.domain.metrics.primary.clone(),
                    size: state.domain.metrics.best_models_size(),
                    updated_at: entries
                        .iter()
                        .map(|e| e.selected_at.clone())
                        .max()
                        .unwrap_or_default(),
                    entries,
                    pinned: Vec::new(),
                    excluded: Vec::new(),
                };
                eprintln!(
                    "[lensing-server infer] best-models group: {} members",
                    group.entries.len()
                );
                std::fs::write(
                    root.join("data/best-models.json"),
                    serde_json::to_vec_pretty(&group)?,
                )?;
            }
            Ok(_) => {}
            Err(e) => eprintln!("[lensing-server infer] best-models load failed: {e:#}"),
        }
    }

    let app = Router::new()
        .route("/health", get(api::health))
        .route("/domain", get(api::get_domain))
        .route("/models", get(api::list_models))
        .route("/models/{name}", get(api::get_model))
        .route("/models/{name}/contract", get(api::get_model_contract))
        .route("/models/{name}/viz", get(api::get_model_viz))
        .route("/models/{name}/predict", axum::routing::post(api::predict_model))
        .route("/best-models", get(api::get_best_models))
        .route("/best-models/predict", axum::routing::post(api::predict_best_models))
        .with_state(state);
    let app = Router::new().nest("/api", app);

    let addr = format!("0.0.0.0:{port}");
    eprintln!("[lensing-server infer] listening on http://localhost:{port} (predict-only)");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
