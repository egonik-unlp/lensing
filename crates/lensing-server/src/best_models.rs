//! The best-models group: a deterministic top-N selection over everything
//! trained so far, kept current by recompute hooks (run completion, model
//! promotion/deletion) and served behind `/api/best-models`.
//!
//! Selection ranks all promotable finished runs and already-promoted models
//! by the domain's primary metric; the top `[metrics].best_models_size`
//! candidates form the group. Selected runs nobody promoted are
//! auto-promoted (`best-<predictor>-<run-slug>`), so the group is always
//! predictable against. Curation overrides (pin/exclude) come in through
//! `PUT /api/best-models` and survive recomputes; models that fall out of
//! the top-N stay promoted but leave the group.
//!
//! Persistence follows the file+db dual pattern: `data/best-models.json` is
//! the whole-document mirror (also carrying pinned/excluded), the
//! `best_models` table is the queryable copy, swapped wholesale via DbSink.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::Utc;
use lensing_core::{
    BestModelEntry, BestModelGroup, BestModelSource, InferencePrediction, RunStatus,
};

use crate::models::{self, PredictError, PredictRequest};
use crate::runs;
use crate::state::AppState;

/// Marker note written onto auto-promoted models.
const AUTO_NOTES: &str = "source: best-models auto";

// ---------------- persistence ----------------

fn group_path(state: &AppState) -> std::path::PathBuf {
    state.root.join("data/best-models.json")
}

/// The current group document; an absent/unreadable file is an empty group.
pub fn read_group(state: &AppState) -> BestModelGroup {
    let path = group_path(state);
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_else(|| {
            BestModelGroup::empty(
                &state.domain.metrics.primary,
                state.domain.metrics.best_models_size(),
            )
        })
}

fn write_group(state: &AppState, group: &BestModelGroup) -> Result<()> {
    std::fs::write(group_path(state), serde_json::to_vec_pretty(group)?)
        .context("write data/best-models.json")?;
    if let Some(sink) = &state.db_sink {
        sink.best_models_replace(&group.entries);
    }
    Ok(())
}

// ---------------- recompute ----------------

/// Fire a recompute without blocking the caller (run-completion and model
/// promote/delete hooks). Failures are logged, never propagated.
pub fn spawn_recompute(state: Arc<AppState>) {
    tokio::spawn(async move {
        if let Err(e) = recompute(&state).await {
            eprintln!("[lensing-server] best-models recompute failed: {e:#}");
        }
    });
}

/// Recompute the group under the re-entrancy lock (concurrent triggers
/// serialize; the group file never races).
pub async fn recompute(state: &Arc<AppState>) -> Result<BestModelGroup> {
    let _guard = state.best_models_lock.lock().await;
    recompute_locked(state).await
}

/// Apply pin/exclude deltas to the persisted curation lists, then recompute.
/// Pinned names must be promoted models; pinning wins over exclusion.
pub async fn curate(
    state: &Arc<AppState>,
    pin: Vec<String>,
    exclude: Vec<String>,
    unpin: Vec<String>,
    unexclude: Vec<String>,
) -> Result<BestModelGroup> {
    let _guard = state.best_models_lock.lock().await;
    let mut group = read_group(state);
    for name in pin {
        anyhow::ensure!(
            state.models_dir().join(&name).join("record.json").is_file(),
            "cannot pin {name:?}: not a promoted model"
        );
        if !group.pinned.contains(&name) {
            group.pinned.push(name);
        }
    }
    group.pinned.retain(|n| !unpin.contains(n));
    for name in exclude {
        if !group.excluded.contains(&name) {
            group.excluded.push(name);
        }
    }
    group.excluded.retain(|n| !unexclude.contains(n));
    write_group(state, &group)?;
    recompute_locked(state).await
}

/// One rankable candidate: a finished run (optionally already promoted).
struct Candidate {
    run_id: String,
    predictor: String,
    dataset_id: String,
    value: f64,
    /// Name of a promoted model pointing at this run, if any.
    model_name: Option<String>,
}

async fn recompute_locked(state: &Arc<AppState>) -> Result<BestModelGroup> {
    let prev = read_group(state);
    let primary = state.domain.metrics.primary.clone();
    let Some(lower_wins) = state.domain.metrics.primary_direction() else {
        eprintln!(
            "[lensing-server] best-models: primary metric {:?} maps to no run metric; group left unchanged",
            primary
        );
        return Ok(prev);
    };
    let size = state.domain.metrics.best_models_size();

    // ---- candidate pool: finished runs (local dirs + database rows) ----
    let mut by_run: HashMap<String, Candidate> = HashMap::new();
    let mut add_run = |meta: &lensing_core::RunMeta| {
        if !matches!(meta.status, RunStatus::Succeeded | RunStatus::Stopped) {
            return;
        }
        let Some(metrics) = &meta.metrics else { return };
        // Skip runs that don't carry the primary metric (e.g. a regression run
        // when the domain ranks by AUC, or vice versa) — they aren't rankable.
        let Some(value) = state.domain.metrics.extract(&primary, metrics) else { return };
        let promotable = state
            .registry
            .get(&meta.predictor)
            .map(|p| p.supports_predict())
            .unwrap_or(false);
        if !promotable {
            return;
        }
        by_run.entry(meta.run_id.clone()).or_insert_with(|| Candidate {
            run_id: meta.run_id.clone(),
            predictor: meta.predictor.clone(),
            dataset_id: meta.dataset_id.clone(),
            value,
            model_name: None,
        });
    };
    if let Ok(entries) = std::fs::read_dir(state.runs_dir()) {
        for e in entries.flatten() {
            if let Ok(meta) = runs::read_meta(&e.path()) {
                add_run(&meta);
            }
        }
    }
    if let Some(db) = &state.db {
        if let Ok(rows) = lensing_db::queries::load_runs(db).await {
            for meta in &rows {
                add_run(meta);
            }
        }
    }

    // ---- promoted models join their source run's candidate ----
    // A model whose run is gone keeps its previous in-group metric (group
    // membership must survive run-dir cleanup), else it cannot be ranked.
    // A model whose run is gone keeps its previous in-group metric — but ONLY
    // if the group's metric regime is unchanged. When the primary metric flips
    // (e.g. regression MAE → classification AUC), the stored value is in the old
    // metric and is NOT comparable; carrying it would let a stale MAE ~0.4
    // masquerade as an AUC. On a regime change, drop the fallback so such models
    // fall out until they produce a run scored under the current metric.
    let regime_changed = !prev.primary_metric.eq_ignore_ascii_case(&primary);
    let prev_value: HashMap<&str, f64> = if regime_changed {
        HashMap::new()
    } else {
        prev.entries.iter().map(|e| (e.name.as_str(), e.metric_value)).collect()
    };
    for record in models::list_models(state) {
        match by_run.get_mut(&record.run_id) {
            Some(cand) => {
                // Prefer a pinned / previously-selected name when several
                // models point at the same run.
                let keep = cand.model_name.as_deref().is_some_and(|n| {
                    prev.pinned.iter().any(|p| p == n)
                        || prev.entries.iter().any(|e| e.name == n)
                });
                if !keep {
                    cand.model_name = Some(record.name.clone());
                }
            }
            None => {
                if let Some(&value) = prev_value.get(record.name.as_str()) {
                    by_run.insert(
                        record.run_id.clone(),
                        Candidate {
                            run_id: record.run_id.clone(),
                            predictor: record.predictor.clone(),
                            dataset_id: record.dataset_id.clone(),
                            value,
                            model_name: Some(record.name.clone()),
                        },
                    );
                }
            }
        }
    }

    // ---- rank, honoring curation overrides ----
    let candidates: Vec<Candidate> = by_run.into_values().collect();
    let order = rank(&candidates, &prev.pinned, &prev.excluded, size, lower_wins);

    // ---- materialize the group, auto-promoting unpromoted selections ----
    let now = Utc::now().to_rfc3339();
    let mut entries: Vec<BestModelEntry> = Vec::new();
    let mut auto_taken = 0usize;
    for (idx, pinned) in order {
        if !pinned && auto_taken >= size {
            continue;
        }
        let cand = &candidates[idx];
        let name = match &cand.model_name {
            Some(name) => name.clone(),
            None => match auto_promote(state, cand).await {
                Ok(name) => name,
                Err(e) => {
                    eprintln!(
                        "[lensing-server] best-models: cannot promote run {}: {e:#}",
                        cand.run_id
                    );
                    continue;
                }
            },
        };
        if !pinned {
            auto_taken += 1;
        }
        entries.push(BestModelEntry {
            name,
            rank: entries.len() as u32 + 1,
            metric: state.domain.metrics.primary.clone(),
            metric_value: cand.value,
            run_id: cand.run_id.clone(),
            predictor: cand.predictor.clone(),
            dataset_id: cand.dataset_id.clone(),
            source: if pinned { BestModelSource::Pinned } else { BestModelSource::Auto },
            selected_at: now.clone(),
        });
    }

    let group = BestModelGroup {
        primary_metric: state.domain.metrics.primary.clone(),
        size,
        entries,
        pinned: prev.pinned,
        excluded: prev.excluded,
        updated_at: now,
    };
    write_group(state, &group)?;
    Ok(group)
}

/// Pure selection: indices into `candidates`, metric-ordered, pinned-flagged.
/// The first `size` rankable candidates are taken; pinned candidates ride
/// along regardless of rank (pin wins over exclusion). Promotion failures are
/// handled by the caller taking more than `size` auto rows from this order.
fn rank(
    candidates: &[Candidate],
    pinned: &[String],
    excluded: &[String],
    size: usize,
    lower_wins: bool,
) -> Vec<(usize, bool)> {
    let excluded: HashSet<&str> = excluded.iter().map(String::as_str).collect();
    let is_pinned = |c: &Candidate| {
        c.model_name.as_deref().is_some_and(|n| pinned.iter().any(|p| p == n))
    };
    let mut idxs: Vec<usize> = candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            c.value.is_finite()
                && (is_pinned(c)
                    || (!excluded.contains(c.run_id.as_str())
                        && !c.model_name.as_deref().is_some_and(|n| excluded.contains(n))))
        })
        .map(|(i, _)| i)
        .collect();
    idxs.sort_by(|&a, &b| {
        let (va, vb) = (candidates[a].value, candidates[b].value);
        if lower_wins { va.total_cmp(&vb) } else { vb.total_cmp(&va) }
    });
    let mut out: Vec<(usize, bool)> = Vec::new();
    for &i in &idxs {
        let pin = is_pinned(&candidates[i]);
        if out.iter().filter(|(_, p)| !p).count() < size || pin {
            out.push((i, pin));
        }
    }
    out
}

/// Collision-safe deterministic name for an auto-promotion:
/// `best-<predictor>-<run-date-time-hex>`. Materializes remote-worker runs
/// (database rows without a local dir) before promoting.
async fn auto_promote(state: &Arc<AppState>, cand: &Candidate) -> Result<String> {
    if !state.runs_dir().join(&cand.run_id).join("meta.json").is_file() {
        runs::materialize_run(state, &cand.run_id)
            .await
            .with_context(|| format!("materialize run {}", cand.run_id))?;
    }
    let slug = cand
        .run_id
        .strip_prefix("run-")
        .unwrap_or(&cand.run_id)
        .strip_suffix(&format!("-{}", cand.predictor))
        .unwrap_or(&cand.run_id)
        .to_string();
    let base = format!("best-{}-{}", cand.predictor, slug);
    let base: String = base.chars().take(60).collect();
    for attempt in 0..10u32 {
        let name =
            if attempt == 0 { base.clone() } else { format!("{}-{}", base, attempt + 1) };
        match models::promote(state, &name, &cand.run_id, Some(AUTO_NOTES.into())) {
            Ok(record) => return Ok(record.name),
            Err(e) if e.to_string().contains("already taken") => continue,
            Err(e) => return Err(e),
        }
    }
    anyhow::bail!("no free auto-promotion name for run {}", cand.run_id)
}

// ---------------- group predict (fan-out + consensus) ----------------

#[derive(Debug, serde::Serialize)]
pub struct MemberPredictions {
    pub name: String,
    pub rank: u32,
    pub predictions: Vec<InferencePrediction>,
}

/// Consensus across members for one input row. Regression: median of the
/// members' values. Binary: mean P(class==1). Multiclass: argmax of the
/// element-wise mean probability vector.
#[derive(Debug, serde::Serialize)]
pub struct ConsensusPoint {
    pub row_id: u64,
    pub predicted: f64,
    /// Consensus class-probability vector (classification only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proba: Option<Vec<f64>>,
    /// Members that produced a prediction for this row.
    pub n_models: usize,
}

#[derive(Debug, serde::Serialize)]
pub struct GroupPredictResponse {
    pub members: Vec<MemberPredictions>,
    pub consensus: Vec<ConsensusPoint>,
    pub warnings: Vec<String>,
}

/// Predict with every group member (the existing per-model predict path,
/// paced by the run-slot semaphore) and aggregate a median consensus. A
/// failing member becomes a warning, not a failure; only an empty group or
/// all members failing is an error.
pub async fn predict_group(
    state: Arc<AppState>,
    req: PredictRequest,
) -> Result<GroupPredictResponse, PredictError> {
    let group = read_group(&state);
    if group.entries.is_empty() {
        return Err(PredictError::BadInput(
            "best-models group is empty; train runs or POST /api/best-models/recompute".into(),
        ));
    }

    let futures: Vec<_> = group
        .entries
        .iter()
        .map(|entry| {
            let st = state.clone();
            let name = entry.name.clone();
            let rank = entry.rank;
            let member_req = PredictRequest {
                items: req.items.clone(),
                point_ids: req.point_ids.clone(),
            };
            async move { (name.clone(), rank, models::predict(st, name, member_req).await) }
        })
        .collect();

    let mut members: Vec<MemberPredictions> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let mut first_err: Option<PredictError> = None;
    for (name, rank, result) in futures::future::join_all(futures).await {
        match result {
            Ok(resp) => {
                for w in resp.warnings {
                    let line = format!("{name}: {w}");
                    if !warnings.contains(&line) {
                        warnings.push(line);
                    }
                }
                members.push(MemberPredictions { name, rank, predictions: resp.predictions });
            }
            Err(e) => {
                let msg = match &e {
                    PredictError::NotFound => "model not found (stale group; recompute)".into(),
                    PredictError::BadInput(m) => m.clone(),
                    PredictError::Internal(err) => format!("{err:#}"),
                };
                warnings.push(format!("{name}: predict failed: {msg}"));
                first_err.get_or_insert(e);
            }
        }
    }
    if members.is_empty() {
        return Err(first_err.unwrap_or(PredictError::BadInput("no members produced predictions".into())));
    }

    // Aggregate per row, task-aware. Regression → median scalar; binary → mean
    // P(class==1); multiclass → element-wise mean probability vector + argmax.
    use lensing_core::domain::Task;
    let task = state.domain.target.task;
    let mut scalars: HashMap<u64, Vec<f64>> = HashMap::new();
    let mut probas: HashMap<u64, Vec<Vec<f64>>> = HashMap::new();
    for m in &members {
        for p in &m.predictions {
            scalars.entry(p.row_id).or_default().push(p.predicted);
            if let Some(pr) = &p.proba {
                probas.entry(p.row_id).or_default().push(pr.clone());
            }
        }
    }
    let mut consensus: Vec<ConsensusPoint> = scalars
        .into_iter()
        .map(|(row_id, mut vals)| match task {
            Task::Binary => {
                let n = vals.len();
                let mean = vals.iter().sum::<f64>() / n as f64;
                ConsensusPoint {
                    row_id,
                    predicted: mean,
                    proba: Some(vec![1.0 - mean, mean]),
                    n_models: n,
                }
            }
            Task::Multiclass => {
                let rows = probas.get(&row_id).cloned().unwrap_or_default();
                let n = rows.len().max(1);
                let k = rows.first().map(|v| v.len()).unwrap_or(0);
                let mut mean = vec![0.0f64; k];
                for r in &rows {
                    for (i, &v) in r.iter().enumerate() {
                        mean[i] += v / n as f64;
                    }
                }
                let argmax = mean
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(c, _)| c as f64)
                    .unwrap_or(0.0);
                ConsensusPoint { row_id, predicted: argmax, proba: Some(mean), n_models: rows.len() }
            }
            // A forecast is a continuous prediction like regression: consensus
            // is the median across models.
            Task::Regression | Task::TimeSeries => {
                vals.sort_by(|a, b| a.total_cmp(b));
                let n = vals.len();
                let predicted = if n % 2 == 1 {
                    vals[n / 2]
                } else {
                    (vals[n / 2 - 1] + vals[n / 2]) / 2.0
                };
                ConsensusPoint { row_id, predicted, proba: None, n_models: n }
            }
        })
        .collect();
    consensus.sort_by_key(|c| c.row_id);

    Ok(GroupPredictResponse { members, consensus, warnings })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(run_id: &str, value: f64, model: Option<&str>) -> Candidate {
        Candidate {
            run_id: run_id.into(),
            predictor: "burn-mlp".into(),
            dataset_id: "ds-x".into(),
            value,
            model_name: model.map(str::to_string),
        }
    }

    #[test]
    fn rank_lower_wins_takes_top_n() {
        let c = vec![cand("r1", 3.0, None), cand("r2", 1.0, None), cand("r3", 2.0, None)];
        let order = rank(&c, &[], &[], 2, true);
        let runs: Vec<&str> = order.iter().map(|&(i, _)| c[i].run_id.as_str()).collect();
        assert_eq!(runs, ["r2", "r3"]);
    }

    #[test]
    fn rank_higher_wins_flips_direction() {
        let c = vec![cand("r1", 0.3, None), cand("r2", 0.9, None), cand("r3", 0.6, None)];
        let order = rank(&c, &[], &[], 2, false);
        let runs: Vec<&str> = order.iter().map(|&(i, _)| c[i].run_id.as_str()).collect();
        assert_eq!(runs, ["r2", "r3"]);
    }

    #[test]
    fn rank_excludes_by_run_id_and_model_name() {
        let c = vec![
            cand("r1", 1.0, None),
            cand("r2", 2.0, Some("m2")),
            cand("r3", 3.0, None),
        ];
        let order = rank(&c, &[], &["r1".into(), "m2".into()], 3, true);
        let runs: Vec<&str> = order.iter().map(|&(i, _)| c[i].run_id.as_str()).collect();
        assert_eq!(runs, ["r3"]);
    }

    #[test]
    fn rank_pins_ride_along_beyond_cap_and_beat_exclusion() {
        let c = vec![
            cand("r1", 1.0, None),
            cand("r2", 2.0, None),
            cand("r3", 9.0, Some("pinned-model")),
        ];
        // cap 2: r1+r2 fill the auto slots, the pin overflows the cap.
        let order =
            rank(&c, &["pinned-model".into()], &["pinned-model".into()], 2, true);
        let picked: Vec<(&str, bool)> =
            order.iter().map(|&(i, p)| (c[i].run_id.as_str(), p)).collect();
        assert_eq!(picked, [("r1", false), ("r2", false), ("r3", true)]);
    }

    #[test]
    fn rank_skips_non_finite_values() {
        let c = vec![cand("r1", f64::NAN, None), cand("r2", 2.0, None)];
        let order = rank(&c, &[], &[], 2, true);
        assert_eq!(order.len(), 1);
        assert_eq!(c[order[0].0].run_id, "r2");
    }
}
