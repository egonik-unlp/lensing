//! Model definitions: named presets (predictor + merged hyperparams +
//! dataset tags). The Postgres `definitions` table is the authoritative
//! store; `models.toml` at the repository root is re-exported on every
//! mutation as the git-versionable snapshot (and seeds the table once, on
//! first backfill). When the database is down the file is the fallback, so
//! the server keeps working degraded.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::Utc;
use lensing_core::ModelDefinition;
use serde::{Deserialize, Serialize};

use crate::models::validate_name;
use crate::state::AppState;

/// In-memory definitions cache (source: database, else `models.toml`).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Definitions {
    #[serde(default)]
    pub definitions: Vec<ModelDefinition>,
}

/// Definition failures, separated so the API layer can map status codes.
pub enum DefError {
    /// Unknown definition name → 404.
    NotFound,
    /// Name already taken → 409.
    Conflict(String),
    /// Caller-fixable problem (bad name, unknown predictor/param) → 400.
    Bad(String),
    /// Everything else (IO, database, serialization) → 500.
    Internal(anyhow::Error),
}

impl From<anyhow::Error> for DefError {
    fn from(e: anyhow::Error) -> Self {
        DefError::Internal(e)
    }
}

pub fn definitions_path(root: &Path) -> PathBuf {
    root.join("models.toml")
}

/// Load `models.toml`. A missing file is an empty registry; a corrupt one is
/// a startup error (never silently dropped).
pub fn load_file(path: &Path) -> Result<Definitions> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Definitions::default()),
        Err(e) => return Err(e).with_context(|| format!("read {}", path.display())),
    };
    toml::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

/// Write temp + rename so a crash mid-write never corrupts the snapshot.
fn save_atomic(path: &Path, defs: &Definitions) -> Result<()> {
    let text = toml::to_string_pretty(defs).context("serialize models.toml")?;
    let tmp = path.with_extension("toml.tmp");
    std::fs::write(&tmp, text).with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("rename onto {}", path.display()))?;
    Ok(())
}

/// Re-export the current set to `models.toml` (startup snapshot refresh
/// after the database, the authoritative store, has been loaded).
pub fn export_file(root: &Path, defs: &Definitions) -> Result<()> {
    save_atomic(&definitions_path(root), defs)
}

pub async fn list(state: &AppState) -> Vec<ModelDefinition> {
    state.definitions.lock().await.definitions.clone()
}

pub async fn get(state: &AppState, name: &str) -> Option<ModelDefinition> {
    state
        .definitions
        .lock()
        .await
        .definitions
        .iter()
        .find(|d| d.name == name)
        .cloned()
}

/// Run `f` against a copy of the definitions vec, then commit: database
/// first (authoritative), `models.toml` export second, memory last. The lock
/// is held across the awaits so concurrent mutations serialize, and a failed
/// database write leaves memory and file untouched.
async fn mutate<T>(
    state: &AppState,
    f: impl FnOnce(&mut Vec<ModelDefinition>) -> Result<T, DefError>,
) -> Result<T, DefError> {
    let mut defs = state.definitions.lock().await;
    let mut copy = defs.definitions.clone();
    let out = f(&mut copy)?;
    if let Some(pool) = &state.db {
        lensing_db::queries::replace_definitions(pool, &copy)
            .await
            .context("write definitions to database")?;
    }
    let snapshot = Definitions { definitions: copy };
    save_atomic(&definitions_path(&state.root), &snapshot)?;
    defs.definitions = snapshot.definitions;
    Ok(out)
}

/// Validate name + predictor, merge hyperparams over the predictor's schema
/// defaults, and append the new definition.
pub async fn create(
    state: &AppState,
    name: &str,
    predictor: &str,
    hyperparams: Option<serde_json::Value>,
    dataset_tags: Vec<String>,
    notes: Option<String>,
) -> Result<ModelDefinition, DefError> {
    validate_name(name).map_err(|e| DefError::Bad(format!("{e:#}")))?;
    let p = state
        .registry
        .get(predictor)
        .ok_or_else(|| DefError::Bad(format!("unknown predictor {predictor}")))?;
    let hyperparams = p
        .merge_hyperparams(hyperparams)
        .map_err(|e| DefError::Bad(format!("{e:#}")))?;

    let def = ModelDefinition {
        name: name.to_string(),
        predictor: predictor.to_string(),
        hyperparams,
        dataset_tags: dedup(dataset_tags),
        notes: notes.filter(|n| !n.is_empty()),
        created_at: Utc::now().to_rfc3339(),
        updated_at: None,
    };
    mutate(state, |defs| {
        if defs.iter().any(|d| d.name == name) {
            return Err(DefError::Conflict(format!("definition name {name:?} is already taken")));
        }
        defs.push(def.clone());
        Ok(def)
    })
    .await
}

/// Partial update: any of hyperparams / dataset_tags / notes. Hyperparams
/// are re-merged over the schema defaults (same validation as create).
pub async fn update(
    state: &AppState,
    name: &str,
    hyperparams: Option<serde_json::Value>,
    dataset_tags: Option<Vec<String>>,
    notes: Option<String>,
) -> Result<ModelDefinition, DefError> {
    // Validate hyperparams against the definition's predictor before locking.
    let merged = match hyperparams {
        Some(hp) => {
            let def = get(state, name).await.ok_or(DefError::NotFound)?;
            let p = state
                .registry
                .get(&def.predictor)
                .ok_or_else(|| DefError::Bad(format!("predictor {} is not in the registry", def.predictor)))?;
            Some(p.merge_hyperparams(Some(hp)).map_err(|e| DefError::Bad(format!("{e:#}")))?)
        }
        None => None,
    };
    mutate(state, |defs| {
        let def = defs.iter_mut().find(|d| d.name == name).ok_or(DefError::NotFound)?;
        if let Some(hp) = merged {
            def.hyperparams = hp;
        }
        if let Some(tags) = dataset_tags {
            def.dataset_tags = dedup(tags);
        }
        if let Some(n) = notes {
            def.notes = if n.is_empty() { None } else { Some(n) };
        }
        def.updated_at = Some(Utc::now().to_rfc3339());
        Ok(def.clone())
    })
    .await
}

pub async fn rename(state: &AppState, old: &str, new: &str) -> Result<ModelDefinition, DefError> {
    validate_name(new).map_err(|e| DefError::Bad(format!("{e:#}")))?;
    mutate(state, |defs| {
        if defs.iter().any(|d| d.name == new) {
            return Err(DefError::Conflict(format!("definition name {new:?} is already taken")));
        }
        let def = defs.iter_mut().find(|d| d.name == old).ok_or(DefError::NotFound)?;
        def.name = new.to_string();
        def.updated_at = Some(Utc::now().to_rfc3339());
        Ok(def.clone())
    })
    .await
}

pub async fn delete(state: &AppState, name: &str) -> Result<(), DefError> {
    mutate(state, |defs| {
        let before = defs.len();
        defs.retain(|d| d.name != name);
        if defs.len() == before {
            return Err(DefError::NotFound);
        }
        Ok(())
    })
    .await
}

/// Clone a definition's predictor + params into a new definition. Dataset
/// tags are NOT copied: they record usage, and the clone hasn't run yet.
pub async fn clone_def(state: &AppState, src: &str, new: &str) -> Result<ModelDefinition, DefError> {
    validate_name(new).map_err(|e| DefError::Bad(format!("{e:#}")))?;
    mutate(state, |defs| {
        if defs.iter().any(|d| d.name == new) {
            return Err(DefError::Conflict(format!("definition name {new:?} is already taken")));
        }
        let src = defs.iter().find(|d| d.name == src).ok_or(DefError::NotFound)?;
        let def = ModelDefinition {
            name: new.to_string(),
            predictor: src.predictor.clone(),
            hyperparams: src.hyperparams.clone(),
            dataset_tags: Vec::new(),
            notes: src.notes.clone(),
            created_at: Utc::now().to_rfc3339(),
            updated_at: None,
        };
        defs.push(def.clone());
        Ok(def)
    })
    .await
}

/// Record that a dataset was used with a definition (no-op if already
/// tagged). Called when a run is launched from the definition.
pub async fn add_dataset_tag(state: &AppState, name: &str, dataset_id: &str) -> Result<(), DefError> {
    mutate(state, |defs| {
        let def = defs.iter_mut().find(|d| d.name == name).ok_or(DefError::NotFound)?;
        if !def.dataset_tags.iter().any(|t| t == dataset_id) {
            def.dataset_tags.push(dataset_id.to_string());
            def.updated_at = Some(Utc::now().to_rfc3339());
        }
        Ok(())
    })
    .await
}

fn dedup(tags: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(tags.len());
    for t in tags {
        if !t.is_empty() && !out.contains(&t) {
            out.push(t);
        }
    }
    out
}
