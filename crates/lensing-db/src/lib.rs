//! PostgreSQL state for lensing-server: model definitions, run metas + metrics,
//! promoted-model records, a dataset index and hyperparameter presets.
//!
//! Direction of truth, per table:
//! - `definitions` — the database is AUTHORITATIVE. `models.toml` is
//!   re-exported on every mutation as the git-diffable snapshot (and seeds
//!   the table once, on first backfill).
//! - `runs` / `run_events` / `models` / `datasets` / `hp_presets` — the
//!   database is a transactional MIRROR of the on-disk artifacts. Files are
//!   written first (predictors and promotion read the run/model dirs
//!   directly), and the [`sink::DbSink`] applies the same writes to Postgres
//!   in order. Backfill upserts from files, so the mirror self-heals at
//!   startup.
//!
//! Binary artifacts (features.f32, model weights, pca_components.f32, …)
//! never enter the database.

pub mod backfill;
pub mod queries;
pub mod sink;

use std::str::FromStr;

use anyhow::{Context, Result};

/// Idempotent DDL (CREATE … IF NOT EXISTS), applied on every connect.
const SCHEMA: &str = include_str!("../migrations/0001_init.sql");

pub const DEFAULT_DATABASE_URL: &str = "postgres://pg:pg@localhost:5433/lensing";

/// Connection pool to the metadata database.
#[derive(Clone)]
pub struct Db {
    pool: deadpool_postgres::Pool,
}

impl Db {
    pub async fn client(&self) -> Result<deadpool_postgres::Object> {
        self.pool.get().await.context("get database connection")
    }
}

/// Connect (short timeout: the server degrades to file-only state when the
/// database is unreachable) and bring the schema up to date.
pub async fn connect(url: &str) -> Result<Db> {
    let pg_config = tokio_postgres::Config::from_str(url)
        .with_context(|| format!("parse database url {url}"))?;
    let mgr = deadpool_postgres::Manager::from_config(
        pg_config,
        tokio_postgres::NoTls,
        deadpool_postgres::ManagerConfig {
            recycling_method: deadpool_postgres::RecyclingMethod::Fast,
        },
    );
    let pool = deadpool_postgres::Pool::builder(mgr)
        .max_size(8)
        .create_timeout(Some(std::time::Duration::from_secs(4)))
        .wait_timeout(Some(std::time::Duration::from_secs(4)))
        .runtime(deadpool_postgres::Runtime::Tokio1)
        .build()
        .context("build connection pool")?;
    let db = Db { pool };
    let client = tokio::time::timeout(std::time::Duration::from_secs(5), db.client())
        .await
        .context("database connection timed out")??;
    client.batch_execute(SCHEMA).await.context("apply database schema")?;
    Ok(db)
}
