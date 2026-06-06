//! Ordered async mirror: file-state writers (runs.rs, models.rs, api.rs) are
//! synchronous and must never block on the database, so they enqueue ops on
//! an unbounded channel and a single worker task applies them to Postgres in
//! order. A database hiccup is logged and the op dropped — the backfill
//! upserts from files at the next startup, so the mirror self-heals.

use std::collections::HashMap;

use lensing_core::{BestModelEntry, ModelRecord, RunMeta};
use tokio::sync::mpsc;

use crate::queries;

#[derive(Debug)]
enum Op {
    RunUpsert(Box<RunMeta>),
    RunEvent { run_id: String, line: String },
    RunDelete(String),
    ModelUpsert { record: ModelRecord, contract: Option<serde_json::Value> },
    ModelRename { old: String, record: ModelRecord },
    ModelDelete(String),
    DatasetUpsert { dataset_id: String, created_at: Option<String>, manifest: serde_json::Value },
    DatasetDelete(String),
    ModelArtifacts { name: String, files: Vec<(String, Vec<u8>)> },
    ModelArtifactsDelete(String),
    ModelArtifactsRename { old: String, new: String },
    BestModelsReplace(Vec<BestModelEntry>),
}

#[derive(Clone)]
pub struct DbSink {
    tx: mpsc::UnboundedSender<Op>,
}

impl DbSink {
    /// Spawn the worker on the current tokio runtime.
    pub fn spawn(db: crate::Db) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(worker(db, rx));
        Self { tx }
    }

    pub fn run_upsert(&self, meta: &RunMeta) {
        let _ = self.tx.send(Op::RunUpsert(Box::new(meta.clone())));
    }
    pub fn run_event(&self, run_id: &str, line: &str) {
        let _ = self.tx.send(Op::RunEvent { run_id: run_id.to_string(), line: line.to_string() });
    }
    pub fn run_delete(&self, run_id: &str) {
        let _ = self.tx.send(Op::RunDelete(run_id.to_string()));
    }
    pub fn model_upsert(&self, record: &ModelRecord, contract: Option<serde_json::Value>) {
        let _ = self.tx.send(Op::ModelUpsert { record: record.clone(), contract });
    }
    pub fn model_rename(&self, old: &str, record: &ModelRecord) {
        let _ = self.tx.send(Op::ModelRename { old: old.to_string(), record: record.clone() });
    }
    pub fn model_delete(&self, name: &str) {
        let _ = self.tx.send(Op::ModelDelete(name.to_string()));
    }
    pub fn dataset_upsert(&self, dataset_id: &str, created_at: Option<String>, manifest: serde_json::Value) {
        let _ = self.tx.send(Op::DatasetUpsert {
            dataset_id: dataset_id.to_string(),
            created_at,
            manifest,
        });
    }
    pub fn dataset_delete(&self, dataset_id: &str) {
        let _ = self.tx.send(Op::DatasetDelete(dataset_id.to_string()));
    }
    /// Upload a promoted model's full file set (replaces any previous set).
    pub fn model_artifacts(&self, name: &str, files: Vec<(String, Vec<u8>)>) {
        let _ = self.tx.send(Op::ModelArtifacts { name: name.to_string(), files });
    }
    pub fn model_artifacts_delete(&self, name: &str) {
        let _ = self.tx.send(Op::ModelArtifactsDelete(name.to_string()));
    }
    pub fn model_artifacts_rename(&self, old: &str, new: &str) {
        let _ = self.tx.send(Op::ModelArtifactsRename {
            old: old.to_string(),
            new: new.to_string(),
        });
    }
    /// Replace the best-models group member set (whole-table swap).
    pub fn best_models_replace(&self, entries: &[BestModelEntry]) {
        let _ = self.tx.send(Op::BestModelsReplace(entries.to_vec()));
    }
}

async fn worker(db: crate::Db, mut rx: mpsc::UnboundedReceiver<Op>) {
    // Next progress-line seq per run, lazily initialized from the table so a
    // restarted server appends after the backfilled tail.
    let mut next_seq: HashMap<String, i32> = HashMap::new();

    while let Some(op) = rx.recv().await {
        let result = match op {
            Op::RunUpsert(meta) => queries::upsert_run(&db, &meta).await,
            Op::RunEvent { run_id, line } => {
                let seq = match next_seq.get(&run_id) {
                    Some(&s) => s,
                    None => match queries::next_event_seq(&db, &run_id).await {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("[lensing-db] event seq for {run_id}: {e:#}");
                            continue;
                        }
                    },
                };
                next_seq.insert(run_id.clone(), seq + 1);
                queries::insert_events(&db, &run_id, seq, std::slice::from_ref(&line)).await
            }
            Op::RunDelete(id) => {
                next_seq.remove(&id);
                queries::delete_run(&db, &id).await
            }
            Op::ModelUpsert { record, contract } => {
                queries::upsert_model(&db, &record, contract.as_ref()).await
            }
            Op::ModelRename { old, record } => queries::rename_model(&db, &old, &record).await,
            Op::ModelDelete(name) => queries::delete_model(&db, &name).await,
            Op::DatasetUpsert { dataset_id, created_at, manifest } => {
                queries::upsert_dataset(&db, &dataset_id, created_at.as_deref(), &manifest).await
            }
            Op::DatasetDelete(id) => queries::delete_dataset(&db, &id).await,
            Op::ModelArtifacts { name, files } => {
                queries::put_model_artifacts(&db, &name, &files).await
            }
            Op::ModelArtifactsDelete(name) => queries::delete_model_artifacts(&db, &name).await,
            Op::ModelArtifactsRename { old, new } => {
                queries::rename_model_artifacts(&db, &old, &new).await
            }
            Op::BestModelsReplace(entries) => queries::replace_best_models(&db, &entries).await,
        };
        if let Err(e) = result {
            eprintln!("[lensing-db] mirror write failed (backfill will heal at next startup): {e:#}");
        }
    }
}
