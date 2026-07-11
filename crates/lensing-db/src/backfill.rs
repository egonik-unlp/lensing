//! One-shot, idempotent backfill of every file-based state artifact into
//! Postgres. Zero-data-loss rules:
//! - nothing on disk is ever modified or deleted;
//! - unreadable files are logged into the report and skipped, never fatal;
//! - definitions seed with ON CONFLICT DO NOTHING (the database is
//!   authoritative once seeded); runs / models / datasets / presets upsert
//!   with file-wins semantics (the files are the working artifacts).

use std::path::Path;

use anyhow::{Context, Result};
use lensing_core::{InterpAnalysis, ModelDefinition, ModelRecord, RunMeta};
use serde::Deserialize;


use crate::queries;

#[derive(Debug, Default)]
pub struct BackfillReport {
    pub definitions_found: usize,
    pub definitions_seeded: usize,
    pub runs_found: usize,
    pub run_events_found: usize,
    pub models_found: usize,
    pub datasets_found: usize,
    pub presets_found: usize,
    pub best_models_found: usize,
    pub interp_found: usize,
    /// Files that could not be read/parsed: (path, error). Skipped, kept on disk.
    pub problems: Vec<(String, String)>,
}

impl BackfillReport {
    pub fn render(&self, db: &TableCounts) -> String {
        let mut out = String::new();
        out.push_str("backfill consistency report (files -> postgres)\n");
        out.push_str(&format!(
            "  definitions: {:>6} in models.toml   ({} newly seeded) | {:>6} in db\n",
            self.definitions_found, self.definitions_seeded, db.definitions
        ));
        out.push_str(&format!(
            "  runs:        {:>6} on disk                            | {:>6} in db\n",
            self.runs_found, db.runs
        ));
        out.push_str(&format!(
            "  run events:  {:>6} on disk                            | {:>6} in db\n",
            self.run_events_found, db.run_events
        ));
        out.push_str(&format!(
            "  models:      {:>6} on disk                            | {:>6} in db\n",
            self.models_found, db.models
        ));
        out.push_str(&format!(
            "  datasets:    {:>6} on disk                            | {:>6} in db\n",
            self.datasets_found, db.datasets
        ));
        out.push_str(&format!(
            "  hp presets:  {:>6} on disk                            | {:>6} in db\n",
            self.presets_found, db.hp_presets
        ));
        out.push_str(&format!(
            "  best models: {:>6} on disk                            | {:>6} in db\n",
            self.best_models_found, db.best_models
        ));
        out.push_str(&format!(
            "  interp:      {:>6} on disk                            | {:>6} in db\n",
            self.interp_found, db.interp_analyses
        ));
        if self.problems.is_empty() {
            out.push_str("  problems: none\n");
        } else {
            out.push_str(&format!("  problems: {} file(s) skipped (left on disk untouched):\n", self.problems.len()));
            for (path, err) in &self.problems {
                out.push_str(&format!("    {path}: {err}\n"));
            }
        }
        out
    }

    /// Disk counts the database must cover (db may hold MORE: deleted files
    /// whose rows live on are history, not drift).
    pub fn consistent_with(&self, db: &TableCounts) -> bool {
        db.definitions >= self.definitions_found as i64
            && db.runs >= self.runs_found as i64
            && db.run_events >= self.run_events_found as i64
            && db.models >= self.models_found as i64
            && db.datasets >= self.datasets_found as i64
            && db.hp_presets >= self.presets_found as i64
            && db.best_models >= self.best_models_found as i64
            && db.interp_analyses >= self.interp_found as i64
    }
}

#[derive(Debug)]
pub struct TableCounts {
    pub definitions: i64,
    pub runs: i64,
    pub run_events: i64,
    pub models: i64,
    pub datasets: i64,
    pub hp_presets: i64,
    pub best_models: i64,
    pub interp_analyses: i64,
}

pub async fn table_counts(db: &crate::Db) -> Result<TableCounts> {
    Ok(TableCounts {
        definitions: queries::count(db, "definitions").await?,
        runs: queries::count(db, "runs").await?,
        run_events: queries::count(db, "run_events").await?,
        models: queries::count(db, "models").await?,
        datasets: queries::count(db, "datasets").await?,
        hp_presets: queries::count(db, "hp_presets").await?,
        best_models: queries::count(db, "best_models").await?,
        interp_analyses: queries::count(db, "interp_analyses").await?,
    })
}

#[derive(Deserialize)]
struct DefinitionsFile {
    #[serde(default)]
    definitions: Vec<ModelDefinition>,
}

/// Backfill everything under `root` (repo root: models.toml, data/).
pub async fn backfill(db: &crate::Db, root: &Path) -> Result<BackfillReport> {
    let mut report = BackfillReport::default();

    // -------- definitions (seed-once; db authoritative afterwards) --------
    let defs_path = root.join("models.toml");
    match std::fs::read_to_string(&defs_path) {
        Ok(text) => match toml::from_str::<DefinitionsFile>(&text) {
            Ok(file) => {
                report.definitions_found = file.definitions.len();
                for d in &file.definitions {
                    match queries::seed_definition(db, d).await {
                        Ok(true) => report.definitions_seeded += 1,
                        Ok(false) => {}
                        Err(e) => report.problems.push((format!("models.toml [{}]", d.name), format!("{e:#}"))),
                    }
                }
            }
            Err(e) => report.problems.push((defs_path.display().to_string(), format!("{e:#}"))),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => report.problems.push((defs_path.display().to_string(), format!("{e}"))),
    }

    // -------- runs + progress events (file wins) --------
    for dir in subdirs(&root.join("data/runs")) {
        let meta_path = dir.join("meta.json");
        let meta: RunMeta = match read_json(&meta_path) {
            Ok(m) => m,
            Err(e) => {
                report.problems.push((meta_path.display().to_string(), format!("{e:#}")));
                continue;
            }
        };
        report.runs_found += 1;
        if let Err(e) = queries::upsert_run(db, &meta).await {
            report.problems.push((meta_path.display().to_string(), format!("{e:#}")));
            continue;
        }
        let progress = dir.join("progress.jsonl");
        if progress.is_file() {
            match std::fs::read_to_string(&progress) {
                Ok(text) => {
                    let lines: Vec<String> = text.lines().map(str::to_string).collect();
                    report.run_events_found += lines.len();
                    // progress.jsonl is append-only: only the tail past the
                    // mirrored prefix needs inserting, so re-runs are cheap.
                    match queries::next_event_seq(db, &meta.run_id).await {
                        Ok(next) if (next as usize) < lines.len() => {
                            if let Err(e) =
                                queries::insert_events(db, &meta.run_id, next, &lines[next as usize..]).await
                            {
                                report.problems.push((progress.display().to_string(), format!("{e:#}")));
                            }
                        }
                        Ok(_) => {}
                        Err(e) => report.problems.push((progress.display().to_string(), format!("{e:#}"))),
                    }
                }
                Err(e) => report.problems.push((progress.display().to_string(), format!("{e}"))),
            }
        }
    }

    // -------- promoted models (file wins) --------
    for dir in subdirs(&root.join("data/models")) {
        let record_path = dir.join("record.json");
        let record: ModelRecord = match read_json(&record_path) {
            Ok(r) => r,
            Err(e) => {
                report.problems.push((record_path.display().to_string(), format!("{e:#}")));
                continue;
            }
        };
        report.models_found += 1;
        let contract: Option<serde_json::Value> = read_json(&dir.join("contract.json")).ok();
        if let Err(e) = queries::upsert_model(db, &record, contract.as_ref()).await {
            report.problems.push((record_path.display().to_string(), format!("{e:#}")));
        }
        // Snapshot the model files into model_artifacts (once) so inference
        // nodes can materialize this model with no shared filesystem.
        match queries::load_model_artifacts(db, &record.name).await {
            Ok(existing) if !existing.is_empty() => {}
            Ok(_) => match collect_files(&dir) {
                Ok(files) => {
                    if let Err(e) = queries::put_model_artifacts(db, &record.name, &files).await {
                        report.problems.push((dir.display().to_string(), format!("{e:#}")));
                    }
                }
                Err(e) => report.problems.push((dir.display().to_string(), format!("{e:#}"))),
            },
            Err(e) => report.problems.push((dir.display().to_string(), format!("{e:#}"))),
        }
    }

    // -------- datasets (file wins; full manifest as JSONB) --------
    for dir in subdirs(&root.join("data/datasets")) {
        let manifest_path = dir.join("manifest.json");
        let manifest: serde_json::Value = match read_json(&manifest_path) {
            Ok(m) => m,
            Err(e) => {
                report.problems.push((manifest_path.display().to_string(), format!("{e:#}")));
                continue;
            }
        };
        report.datasets_found += 1;
        let id = manifest
            .get("dataset_id")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| dir.file_name().unwrap_or_default().to_string_lossy().into_owned());
        let created_at = manifest.get("created_at").and_then(|v| v.as_str()).map(str::to_string);
        if let Err(e) = queries::upsert_dataset(db, &id, created_at.as_deref(), &manifest).await {
            report.problems.push((manifest_path.display().to_string(), format!("{e:#}")));
        }
    }

    // -------- best-models group (file wins; whole-document swap) --------
    let group_path = root.join("data/best-models.json");
    if group_path.is_file() {
        match read_json::<lensing_core::BestModelGroup>(&group_path) {
            Ok(group) => {
                report.best_models_found = group.entries.len();
                if let Err(e) = queries::replace_best_models(db, &group.entries).await {
                    report.problems.push((group_path.display().to_string(), format!("{e:#}")));
                }
            }
            Err(e) => report.problems.push((group_path.display().to_string(), format!("{e:#}"))),
        }
    }

    // -------- interp analyses (file wins for missing; live rows kept) --------
    // Reconstruct each completed analysis from its result file and upsert only
    // the ones not already mirrored, so a live row's source/curation is never
    // clobbered by a disk-derived one.
    let interp_root = root.join("data/interp");
    let models_root = root.join("data/models");
    let existing = queries::interp_analysis_ids(db).await.unwrap_or_default();
    for dir in subdirs(&interp_root) {
        let id = match dir.file_name().and_then(|s| s.to_str()) {
            Some(s) if s.starts_with("interp-") => s.to_string(),
            _ => continue, // sae-cache and anything not a job dir
        };
        let Some(analysis) = InterpAnalysis::from_disk(&interp_root, &models_root, &id, true) else {
            continue; // unrecognized, or no single result file (e.g. compare)
        };
        report.interp_found += 1;
        if existing.contains(&id) {
            continue;
        }
        if let Err(e) = queries::upsert_interp_analysis(db, &analysis).await {
            report.problems.push((dir.display().to_string(), format!("{e:#}")));
        }
    }

    // -------- CLI hyperparameter presets (file wins) --------
    let hp_dir = root.join("data/cli-runs/hp");
    if let Ok(entries) = std::fs::read_dir(&hp_dir) {
        for e in entries.flatten() {
            let path = e.path();
            if path.extension().is_none_or(|x| x != "json") {
                continue;
            }
            let name = path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
            match read_json::<serde_json::Value>(&path) {
                Ok(hp) => {
                    report.presets_found += 1;
                    if let Err(e) = queries::upsert_preset(db, &name, &hp).await {
                        report.problems.push((path.display().to_string(), format!("{e:#}")));
                    }
                }
                Err(e) => report.problems.push((path.display().to_string(), format!("{e:#}"))),
            }
        }
    }

    Ok(report)
}

fn subdirs(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut out: Vec<_> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                .map(|e| e.path())
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

/// Every file under `dir` as (relative path, bytes), symlinks skipped.
fn collect_files(dir: &Path) -> Result<Vec<(String, Vec<u8>)>> {
    fn rec(base: &Path, dir: &Path, out: &mut Vec<(String, Vec<u8>)>) -> Result<()> {
        for e in std::fs::read_dir(dir)?.flatten() {
            let path = e.path();
            let ft = e.file_type()?;
            if ft.is_symlink() {
                continue;
            }
            if ft.is_dir() {
                rec(base, &path, out)?;
            } else {
                let rel = path.strip_prefix(base).unwrap().to_string_lossy().into_owned();
                out.push((rel, std::fs::read(&path)?));
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    rec(dir, dir, &mut out)?;
    Ok(out)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}
