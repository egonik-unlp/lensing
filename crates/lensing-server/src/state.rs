use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use tokio::sync::{broadcast, Notify, Semaphore};

use crate::registry::Registry;

pub struct AppState {
    /// Repository root; predictor commands run from here.
    pub root: PathBuf,
    /// The domain configuration (`domain.toml`): corpus schema, target,
    /// feature fields, quality bindings, UI vocabulary.
    pub domain: Arc<pg_core::domain::Domain>,
    pub registry: Registry,
    pub qdrant_url: String,
    pub collection: String,
    /// Qdrant collection holding manually authored listings (separate from
    /// the corpus so hand-entered data never leaks into dataset builds).
    pub manual_collection: String,
    /// OpenAI embeddings config for manual listings. The key comes from the
    /// `OPENAI_API_KEY` env var; absent means listing creation is disabled.
    pub openai_api_key: Option<String>,
    pub embedding_model: String,
    pub openai_base_url: String,
    /// Metadata database. `None` means the server runs file-only (degraded:
    /// Postgres unreachable at startup); everything still works off `data/`
    /// and `models.toml`, and the next startup's backfill heals the mirror.
    pub db: Option<pg_db::Db>,
    /// Ordered async mirror for run/model/dataset file writes.
    pub db_sink: Option<pg_db::sink::DbSink>,
    /// Model definitions. The database is authoritative (when up); every
    /// mutation also re-exports `models.toml` as the git-diffable snapshot.
    /// tokio Mutex: held across the database write so mutations serialize.
    pub definitions: tokio::sync::Mutex<crate::definitions::Definitions>,
    /// Live (running) runs: run_id → handle.
    pub live_runs: Mutex<HashMap<String, Arc<RunHandle>>>,
    /// In-flight dataset builds: build_id → status.
    pub builds: Mutex<HashMap<String, BuildStatus>>,
    /// In-flight heavy async jobs (analyze, export): job_id → status.
    pub jobs: Mutex<HashMap<String, JobStatus>>,
    /// Caps concurrent training runs (CPU-bound on a laptop).
    pub run_slots: Arc<Semaphore>,
    /// Caps concurrent dataset builds.
    pub build_slots: Arc<Semaphore>,
}

impl AppState {
    pub fn datasets_dir(&self) -> PathBuf {
        self.root.join("data/datasets")
    }
    pub fn runs_dir(&self) -> PathBuf {
        self.root.join("data/runs")
    }
    pub fn models_dir(&self) -> PathBuf {
        self.root.join("data/models")
    }
}

/// Progress fan-out for one running run. `history` and the broadcast are
/// updated under the same lock that SSE subscribers snapshot under, so a
/// late joiner never misses or duplicates a line.
///
/// Also carries the run's stop/kill state, lock-free so the SSE lock is
/// never held across child-process control.
pub struct RunHandle {
    inner: Mutex<RunProgress>,
    /// A graceful stop was requested (STOP file written); exit 0 + metrics
    /// then resolves to `stopped` instead of `succeeded`.
    stop_requested: AtomicBool,
    /// A periodic `checkpoint` event arrived: the run dir holds loadable
    /// params even if the process dies later.
    has_checkpoint: AtomicBool,
    /// A force stop was requested; a second force request escalates to an
    /// immediate kill instead of waiting out the grace window.
    force_requested: AtomicBool,
    /// Force-kill signal awaited by the process loop.
    kill: Notify,
}

struct RunProgress {
    history: Vec<String>,
    tx: broadcast::Sender<String>,
}

impl RunHandle {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(1024);
        Self {
            inner: Mutex::new(RunProgress { history: Vec::new(), tx }),
            stop_requested: AtomicBool::new(false),
            has_checkpoint: AtomicBool::new(false),
            force_requested: AtomicBool::new(false),
            kill: Notify::new(),
        }
    }

    /// Append a progress line: recorded for late joiners, fanned out live.
    pub fn push(&self, line: String) {
        let mut p = self.inner.lock().unwrap();
        p.history.push(line.clone());
        let _ = p.tx.send(line);
    }

    /// Snapshot history and subscribe atomically.
    pub fn snapshot_and_subscribe(&self) -> (Vec<String>, broadcast::Receiver<String>) {
        let p = self.inner.lock().unwrap();
        (p.history.clone(), p.tx.subscribe())
    }

    pub fn request_stop(&self) {
        self.stop_requested.store(true, Ordering::SeqCst);
    }

    pub fn is_stop_requested(&self) -> bool {
        self.stop_requested.load(Ordering::SeqCst)
    }

    /// True only for the first caller; lets the progress loop persist
    /// `has_checkpoint` to meta.json exactly once.
    pub fn set_checkpoint(&self) -> bool {
        !self.has_checkpoint.swap(true, Ordering::SeqCst)
    }

    pub fn checkpoint_seen(&self) -> bool {
        self.has_checkpoint.load(Ordering::SeqCst)
    }

    /// True only for the first caller; a repeat force request means the user
    /// wants the process dead now, not after the grace window.
    pub fn set_force_requested(&self) -> bool {
        !self.force_requested.swap(true, Ordering::SeqCst)
    }

    pub fn force_kill(&self) {
        self.kill.notify_waiters();
        // A waiter that subscribes after this call must still see the kill.
        self.kill.notify_one();
    }

    pub async fn killed(&self) {
        self.kill.notified().await;
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum BuildStatus {
    Building { stage: String },
    Done { dataset_id: String },
    Failed { error: String },
}

/// Status of a heavy async job (analyze, export). Mirrors [`BuildStatus`] but
/// the terminal success carries an arbitrary JSON result.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum JobStatus {
    Running { stage: String },
    Done { result: serde_json::Value },
    Failed { error: String },
}
