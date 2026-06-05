mod api;
mod definitions;
mod infer;
mod models;
mod registry;
mod runs;
mod state;
mod worker;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use axum::routing::get;
use axum::Router;
use clap::Parser;
use tokio::sync::Semaphore;
use tower_http::services::{ServeDir, ServeFile};

use crate::registry::Registry;
use crate::state::AppState;

#[derive(Parser)]
#[command(name = "lensing-server", about = "Price-guesser API + UI server")]
struct Cli {
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Repository root (registry.toml, data/, ui/dist live under it).
    #[arg(long, default_value = ".")]
    root: PathBuf,
    #[arg(long, default_value = "http://localhost:6333")]
    qdrant_url: String,
    #[arg(long, default_value = "properties-tagged")]
    collection: String,
    /// Qdrant collection for manually authored listings.
    #[arg(long, default_value = "manual-listings")]
    manual_collection: String,
    /// Max concurrent training runs.
    #[arg(long, default_value_t = 2)]
    max_runs: usize,
    /// Postgres connection string for the metadata database. Resolution:
    /// this flag, else $DATABASE_URL, else the docker-compose default.
    #[arg(long)]
    database_url: Option<String>,
    /// Run without the metadata database (file-only state, as before).
    #[arg(long)]
    no_db: bool,
    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(clap::Subcommand)]
enum Cmd {
    /// Backfill every file-based state artifact (models.toml, data/runs,
    /// data/models, data/datasets, data/cli-runs/hp) into Postgres, print a
    /// consistency report, and exit. Idempotent; never touches the files.
    MigrateData,
    /// Training worker role: claim queued runs from the database, download
    /// datasets from the hub, train, upload results. Deployable anywhere
    /// with the predictor toolchains + DATABASE_URL.
    Worker {
        /// Hub base URL for dataset archives (e.g. http://hub:8080).
        #[arg(long)]
        hub_url: String,
        /// Process one queued run (or nothing) and exit — batch/lambda mode.
        #[arg(long)]
        once: bool,
        /// Queue poll interval when idle.
        #[arg(long, default_value_t = 10)]
        poll_secs: u64,
    },
    /// Inference role: materialize promoted models from the database and
    /// serve ONLY the predict surface (no UI, no orchestration).
    Infer {
        #[arg(long, default_value_t = 8090)]
        infer_port: u16,
        /// Max concurrent predict subprocesses.
        #[arg(long, default_value_t = 2)]
        max_predicts: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env from CWD if present; variables already set in the environment win.
    dotenvy::dotenv().ok();
    let cli = Cli::parse();
    let root = cli.root.canonicalize().context("resolve --root")?;
    let database_url = cli
        .database_url
        .clone()
        .or_else(|| std::env::var("DATABASE_URL").ok().filter(|u| !u.is_empty()))
        .unwrap_or_else(|| pg_db::DEFAULT_DATABASE_URL.to_string());

    match &cli.command {
        Some(Cmd::MigrateData) => return migrate_data(&root, &database_url).await,
        Some(Cmd::Worker { hub_url, once, poll_secs }) => {
            return worker::run(root, database_url, hub_url.clone(), *once, *poll_secs).await;
        }
        Some(Cmd::Infer { infer_port, max_predicts }) => {
            return infer::run(
                root,
                database_url,
                *infer_port,
                cli.qdrant_url.clone(),
                *max_predicts,
            )
            .await;
        }
        None => {}
    }

    let domain = pg_core::domain::Domain::load_or_default(&root)?;
    eprintln!(
        "[lensing-server] domain: {} ({} → {}), {} fields, collection {}",
        domain.project.name,
        domain.project.entity_noun,
        domain.project.target_noun,
        domain.fields.len(),
        domain.corpus.collection,
    );

    let registry = Registry::load(&root.join("registry.toml"))?;
    eprintln!(
        "[lensing-server] {} predictors registered: {}",
        registry.predictors.len(),
        registry.predictors.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(", ")
    );

    // Stale-run sweep must precede the backfill so swept `interrupted`
    // statuses land in the mirror, not stale `running` ones.
    std::fs::create_dir_all(root.join("data/datasets"))?;
    std::fs::create_dir_all(root.join("data/runs"))?;
    std::fs::create_dir_all(root.join("data/models"))?;
    runs::sweep_stale_runs(&root.join("data/runs"));

    let db = if cli.no_db {
        eprintln!("[lensing-server] --no-db: metadata database disabled (file-only state)");
        None
    } else {
        match pg_db::connect(&database_url).await {
            Ok(pool) => {
                match pg_db::backfill::backfill(&pool, &root).await {
                    Ok(report) => {
                        let counts = pg_db::backfill::table_counts(&pool).await?;
                        eprint!("[lensing-server] {}", report.render(&counts));
                        if !report.consistent_with(&counts) {
                            eprintln!("[lensing-server] WARNING: database is missing rows for files on disk (see report)");
                        }
                    }
                    Err(e) => eprintln!("[lensing-server] backfill failed: {e:#}"),
                }
                eprintln!("[lensing-server] metadata database connected ({database_url})");
                Some(pool)
            }
            Err(e) => {
                eprintln!("[lensing-server] WARNING: metadata database unreachable, running file-only: {e:#}");
                eprintln!("[lensing-server]          start it with `zig build db-up` (docker compose)");
                None
            }
        }
    };

    // Definitions: database-authoritative when up (the backfill above seeded
    // it from models.toml); file fallback otherwise.
    let defs = match &db {
        Some(pool) => {
            let defs = definitions::Definitions {
                definitions: pg_db::queries::load_definitions(pool).await?,
            };
            // Refresh the git-diffable snapshot so file and database agree.
            if let Err(e) = definitions::export_file(&root, &defs) {
                eprintln!("[lensing-server] models.toml export failed: {e:#}");
            }
            eprintln!("[lensing-server] {} model definitions loaded from the database", defs.definitions.len());
            defs
        }
        None => {
            let defs = definitions::load_file(&definitions::definitions_path(&root))?;
            if !defs.definitions.is_empty() {
                eprintln!("[lensing-server] {} model definitions loaded from models.toml", defs.definitions.len());
            }
            defs
        }
    };

    // OpenAI embeddings config for manual listings; key absence only blocks
    // listing creation, the rest of the server is unaffected.
    let openai_api_key = std::env::var("OPENAI_API_KEY").ok().filter(|k| !k.is_empty());
    let embedding_model = std::env::var("PG_EMBEDDING_MODEL")
        .ok()
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| pg_pipeline::openai::DEFAULT_MODEL.to_string());
    let openai_base_url = std::env::var("PG_OPENAI_BASE_URL")
        .ok()
        .filter(|u| !u.is_empty())
        .unwrap_or_else(|| pg_pipeline::openai::DEFAULT_BASE_URL.to_string());
    if openai_api_key.is_none() {
        eprintln!("[lensing-server] OPENAI_API_KEY not set: manual listing creation disabled");
    } else {
        eprintln!("[lensing-server] manual listings: embedding model {embedding_model}");
    }

    let db_sink = db.clone().map(pg_db::sink::DbSink::spawn);
    let state = Arc::new(AppState {
        root: root.clone(),
        domain: Arc::new(domain),
        registry,
        qdrant_url: cli.qdrant_url,
        collection: cli.collection,
        manual_collection: cli.manual_collection,
        openai_api_key,
        embedding_model,
        openai_base_url,
        db,
        db_sink,
        definitions: tokio::sync::Mutex::new(defs),
        live_runs: Mutex::new(Default::default()),
        builds: Mutex::new(Default::default()),
        jobs: Mutex::new(Default::default()),
        run_slots: Arc::new(Semaphore::new(cli.max_runs)),
        build_slots: Arc::new(Semaphore::new(1)),
    });

    let ui_dist = root.join("ui/dist");
    let spa = ServeDir::new(&ui_dist).fallback(ServeFile::new(ui_dist.join("index.html")));

    let api = Router::new()
        .route("/health", get(api::health))
        .route("/domain", get(api::get_domain))
        .route("/openapi.yaml", get(api::openapi_spec))
        .route("/predictors", get(api::list_predictors))
        .route("/datasets", get(api::list_datasets).post(api::build_dataset))
        .route("/datasets/preflight", axum::routing::post(api::preflight))
        .route("/datasets/analyze", axum::routing::post(api::analyze_dataset))
        .route("/datasets/{id}", get(api::get_dataset))
        .route("/datasets/{id}/items", get(api::get_dataset_items))
        .route("/datasets/{id}/split", get(api::get_dataset_split))
        .route("/datasets/{id}/archive", get(api::dataset_archive))
        .route("/datasets/{id}/rename", axum::routing::post(api::rename_dataset))
        .route("/builds/{id}", get(api::get_build))
        .route("/jobs/{id}", get(api::get_job))
        .route("/collections", get(api::list_collections))
        .route("/collections/validate", axum::routing::post(api::validate_collection))
        .route("/collections/export", axum::routing::post(api::export_collection))
        .route("/runs", get(api::list_runs).post(api::start_run))
        .route("/runs/{id}", get(api::get_run).delete(api::delete_run))
        .route("/runs/{id}/predictions", get(api::get_predictions))
        .route("/blend", axum::routing::post(api::blend_runs))
        .route("/runs/{id}/events", get(api::run_events))
        .route("/runs/{id}/stop", axum::routing::post(api::stop_run))
        .route("/runs/{id}/viz", get(api::get_run_viz))
        .route("/models", get(api::list_models).post(api::promote_model))
        .route("/models/{name}", get(api::get_model).delete(api::delete_model))
        .route("/models/{name}/viz", get(api::get_model_viz))
        .route("/models/{name}/contract", get(api::get_model_contract))
        .route("/models/{name}/predict", axum::routing::post(api::predict_model))
        .route("/models/{name}/rename", axum::routing::post(api::rename_model))
        .route("/listings", get(api::list_listings).post(api::create_listing))
        .route(
            "/listings/{id}",
            get(api::get_listing).put(api::update_listing).delete(api::delete_listing),
        )
        .route(
            "/definitions",
            get(api::list_definitions).post(api::create_definition),
        )
        .route(
            "/definitions/{name}",
            get(api::get_definition)
                .patch(api::update_definition)
                .delete(api::delete_definition),
        )
        .route("/definitions/{name}/rename", axum::routing::post(api::rename_definition))
        .route("/definitions/{name}/clone", axum::routing::post(api::clone_definition))
        .with_state(state);

    let app = Router::new()
        .nest("/api", api)
        .route("/docs", get(api::swagger_ui))
        .fallback_service(spa);

    let addr = format!("0.0.0.0:{}", cli.port);
    eprintln!("[lensing-server] listening on http://localhost:{}", cli.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// `lensing-server migrate-data`: explicit one-shot backfill. Unlike the serve
/// path, an unreachable database here is a hard error.
async fn migrate_data(root: &PathBuf, database_url: &str) -> Result<()> {
    let pool = pg_db::connect(database_url)
        .await
        .context("metadata database unreachable; start it with `zig build db-up`")?;
    eprintln!("[migrate-data] connected to {database_url}");
    let report = pg_db::backfill::backfill(&pool, root).await?;
    let counts = pg_db::backfill::table_counts(&pool).await?;
    print!("{}", report.render(&counts));
    if report.consistent_with(&counts) {
        println!("consistency: OK — every file artifact is covered by the database");
        Ok(())
    } else {
        println!("consistency: MISMATCH — the database is missing rows for files on disk");
        std::process::exit(1);
    }
}
