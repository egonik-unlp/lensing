use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::Utc;
use pg_core::{Metrics, ProgressEvent, RunMeta, RunStatus};
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::Command;

use crate::state::{AppState, RunHandle};

const STDERR_TAIL_BYTES: usize = 8 * 1024;

pub fn new_run_id(predictor: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    format!("run-{}-{:05x}-{}", Utc::now().format("%Y%m%d-%H%M%S"), nanos & 0xfffff, predictor)
}

pub fn read_meta(run_dir: &PathBuf) -> Result<RunMeta> {
    let text = std::fs::read_to_string(run_dir.join("meta.json"))
        .with_context(|| format!("read {}/meta.json", run_dir.display()))?;
    Ok(serde_json::from_str(&text)?)
}

fn write_meta(run_dir: &PathBuf, meta: &RunMeta) -> Result<()> {
    std::fs::write(run_dir.join("meta.json"), serde_json::to_vec_pretty(meta)?)?;
    Ok(())
}

/// Name of the graceful-stop marker file the server drops into a run dir.
/// Iterative predictors poll for it between epochs (contract v2).
pub const STOP_FILE: &str = "STOP";

/// Remove a run directory. Promoted models are unaffected — promotion copies
/// everything it needs into `data/models/<name>/`. The caller must reject
/// runs that are still live.
pub fn delete_run(state: &AppState, id: &str) -> Result<()> {
    // A run id is a single path segment (see new_run_id); refuse anything
    // that could traverse out of runs_dir before the join.
    anyhow::ensure!(
        !id.is_empty() && !id.contains(['/', '\\']) && !id.contains(".."),
        "invalid run id {id:?}"
    );
    let run_dir = state.runs_dir().join(id);
    anyhow::ensure!(run_dir.join("meta.json").is_file(), "run {id} not found");
    std::fs::remove_dir_all(&run_dir)?;
    if let Some(sink) = &state.db_sink {
        sink.run_delete(id);
    }
    Ok(())
}

/// Mark any `running` runs as `interrupted` (called once at startup).
pub fn sweep_stale_runs(runs_dir: &PathBuf) {
    let Ok(entries) = std::fs::read_dir(runs_dir) else { return };
    for e in entries.flatten() {
        let dir = e.path();
        // The child died with the server; a stale STOP marker must not leak
        // into a future promotion copy or confuse a re-read of the dir.
        let _ = std::fs::remove_file(dir.join(STOP_FILE));
        if let Ok(mut meta) = read_meta(&dir) {
            if meta.status == RunStatus::Running {
                meta.status = RunStatus::Interrupted;
                meta.finished_at = Some(Utc::now().to_rfc3339());
                let _ = write_meta(&dir, &meta);
            }
        }
    }
}

/// Enqueue a run for a remote training worker: a database row only (status
/// `queued`) — no local run dir; the claiming worker owns one on its host
/// and uploads the predictor-written artifacts when done.
pub async fn enqueue_run(
    state: &AppState,
    dataset_id: String,
    predictor_name: String,
    hyperparams: serde_json::Value,
    from_definition: Option<String>,
) -> Result<String> {
    let db = state
        .db
        .as_ref()
        .context("queueing runs requires the metadata database (start it with `zig build db-up`)")?;
    anyhow::ensure!(
        state.registry.get(&predictor_name).is_some(),
        "unknown predictor {predictor_name}"
    );
    // The dataset must exist on the hub — the worker downloads it from here.
    anyhow::ensure!(
        state.datasets_dir().join(&dataset_id).join("manifest.json").is_file(),
        "dataset {dataset_id} not found"
    );
    let run_id = new_run_id(&predictor_name);
    let meta = RunMeta {
        run_id: run_id.clone(),
        dataset_id,
        predictor: predictor_name,
        hyperparams,
        status: RunStatus::Queued,
        started_at: Utc::now().to_rfc3339(),
        finished_at: None,
        exit_code: None,
        stderr_tail: None,
        metrics: None,
        contract_version: pg_core::CONTRACT_VERSION,
        has_checkpoint: false,
        from_definition,
    };
    pg_db::queries::upsert_run(db, &meta).await?;
    Ok(run_id)
}

/// Materialize a remote run's directory from the database (meta + hp from
/// the runs row, predictor outputs from run_artifacts) so file-based paths
/// (promotion, predictions) work on the hub. Returns false when the run is
/// unknown to the database too.
pub async fn materialize_run(state: &AppState, id: &str) -> Result<bool> {
    let Some(db) = &state.db else { return Ok(false) };
    let Some(meta) = pg_db::queries::load_run(db, id).await? else {
        return Ok(false);
    };
    let files = pg_db::queries::load_run_artifacts(db, id).await?;
    let run_dir = state.runs_dir().join(id);
    std::fs::create_dir_all(&run_dir)?;
    std::fs::write(run_dir.join("meta.json"), serde_json::to_vec_pretty(&meta)?)?;
    std::fs::write(run_dir.join("hp.json"), serde_json::to_vec_pretty(&meta.hyperparams)?)?;
    for (rel, bytes) in &files {
        anyhow::ensure!(
            !rel.contains("..") && !rel.starts_with('/'),
            "run {id}: unsafe artifact path {rel:?}"
        );
        let path = run_dir.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, bytes)?;
    }
    Ok(true)
}

/// Create the run directory + meta, then spawn the orchestration task.
pub fn start_run(
    state: Arc<AppState>,
    dataset_id: String,
    predictor_name: String,
    hyperparams: serde_json::Value,
    from_definition: Option<String>,
) -> Result<String> {
    let predictor = state
        .registry
        .get(&predictor_name)
        .with_context(|| format!("unknown predictor {predictor_name}"))?
        .clone();
    let dataset_dir = state.datasets_dir().join(&dataset_id);
    anyhow::ensure!(
        dataset_dir.join("manifest.json").is_file(),
        "dataset {dataset_id} not found"
    );

    let run_id = new_run_id(&predictor_name);
    let run_dir = state.runs_dir().join(&run_id);
    std::fs::create_dir_all(&run_dir)?;
    std::fs::write(run_dir.join("hp.json"), serde_json::to_vec_pretty(&hyperparams)?)?;

    let meta = RunMeta {
        run_id: run_id.clone(),
        dataset_id,
        predictor: predictor_name,
        hyperparams,
        status: RunStatus::Running,
        started_at: Utc::now().to_rfc3339(),
        finished_at: None,
        exit_code: None,
        stderr_tail: None,
        metrics: None,
        contract_version: pg_core::CONTRACT_VERSION,
        has_checkpoint: false,
        from_definition,
    };
    write_meta(&run_dir, &meta)?;
    if let Some(sink) = &state.db_sink {
        sink.run_upsert(&meta);
    }

    let handle = Arc::new(RunHandle::new());
    state
        .live_runs
        .lock()
        .unwrap()
        .insert(run_id.clone(), handle.clone());

    tokio::spawn(orchestrate(state.clone(), predictor, meta, run_dir, dataset_dir, handle));
    Ok(run_id)
}

async fn orchestrate(
    state: Arc<AppState>,
    predictor: crate::registry::Predictor,
    mut meta: RunMeta,
    run_dir: PathBuf,
    dataset_dir: PathBuf,
    handle: Arc<RunHandle>,
) {
    let run_id = meta.run_id.clone();
    let push = |line: serde_json::Value| {
        let line = line.to_string();
        let _ = append_progress(&run_dir, &line);
        if let Some(sink) = &state.db_sink {
            sink.run_event(&run_id, &line);
        }
        handle.push(line);
    };

    let permit = match state.run_slots.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            push(json!({"event":"log","msg":"queued: waiting for a free training slot"}));
            state.run_slots.clone().acquire_owned().await.unwrap()
        }
    };

    let result = run_predictor(&state, &predictor, &run_dir, &dataset_dir, &handle, &push).await;
    drop(permit);

    meta.finished_at = Some(Utc::now().to_rfc3339());
    // Periodic checkpoint events may have set this mid-run (and persisted it
    // to disk); the final write below must not clobber it back to false.
    meta.has_checkpoint = handle.checkpoint_seen();
    match result {
        Ok((exit_code, stderr_tail)) => {
            meta.exit_code = Some(exit_code);
            meta.stderr_tail = if stderr_tail.is_empty() { None } else { Some(stderr_tail) };
            let metrics: Option<Metrics> = std::fs::read_to_string(run_dir.join("metrics.json"))
                .ok()
                .and_then(|t| serde_json::from_str(&t).ok());
            if exit_code == 0 && metrics.is_some() {
                meta.metrics = metrics;
                // A clean exit after a stop request is the graceful-stop
                // path: evaluated and checkpointed, just fewer epochs.
                meta.status = if handle.is_stop_requested() {
                    RunStatus::Stopped
                } else {
                    RunStatus::Succeeded
                };
                // The run dir now holds whatever checkpoint the predictor
                // wrote; it is promotable iff the predictor can reload it.
                meta.has_checkpoint = predictor.supports_predict();
            } else {
                meta.status = RunStatus::Failed;
                if exit_code == 0 {
                    push(json!({"event":"log","msg":"predictor exited 0 but metrics.json is missing or invalid"}));
                }
            }
        }
        Err(e) => {
            meta.status = RunStatus::Failed;
            meta.stderr_tail = Some(format!("failed to launch predictor: {e:#}"));
        }
    }
    let _ = std::fs::remove_file(run_dir.join(STOP_FILE));
    let _ = write_meta(&run_dir, &meta);
    if let Some(sink) = &state.db_sink {
        sink.run_upsert(&meta);
    }

    // Terminal line: tells SSE clients to refetch meta and close.
    push(json!({"event":"status","status":meta.status}));
    state.live_runs.lock().unwrap().remove(&meta.run_id);
}

async fn run_predictor(
    state: &AppState,
    predictor: &crate::registry::Predictor,
    run_dir: &PathBuf,
    dataset_dir: &PathBuf,
    handle: &RunHandle,
    push: &(dyn Fn(serde_json::Value) + Sync),
) -> Result<(i32, String)> {
    let args = crate::registry::substitute(
        &predictor.args,
        &[
            ("dataset", dataset_dir.to_string_lossy().into_owned()),
            ("run_dir", run_dir.to_string_lossy().into_owned()),
            ("hyperparams", run_dir.join("hp.json").to_string_lossy().into_owned()),
        ],
    );

    spawn_and_capture_cancellable(&predictor.command, &args, &state.root, &|line| {
        // Only well-formed contract events are recorded and fanned out.
        match serde_json::from_str::<ProgressEvent>(line) {
            Ok(ev) => {
                if let ProgressEvent::Checkpoint { .. } = ev {
                    // Persist promotability once, so a server crash after
                    // this point still leaves the run promotable.
                    if handle.set_checkpoint() {
                        if let Ok(mut meta) = read_meta(run_dir) {
                            meta.has_checkpoint = true;
                            let _ = write_meta(run_dir, &meta);
                            if let Some(sink) = &state.db_sink {
                                sink.run_upsert(&meta);
                            }
                        }
                    }
                }
                push(serde_json::to_value(&ev).unwrap())
            }
            Err(_) if !line.trim().is_empty() => {
                push(json!({"event":"log","msg":format!("[stdout] {line}")}))
            }
            Err(_) => {}
        }
    }, Some(handle))
    .await
}

/// Spawn a predictor process, stream its stdout lines to `on_line`, and
/// return (exit code, stderr tail). Shared by training runs and predict
/// invocations.
pub async fn spawn_and_capture(
    command: &str,
    args: &[String],
    cwd: &std::path::Path,
    on_line: &(dyn Fn(&str) + Sync),
) -> Result<(i32, String)> {
    spawn_and_capture_cancellable(command, args, cwd, on_line, None).await
}

/// Like `spawn_and_capture`, but when `kill` is set the stdout loop races
/// against its force-kill signal and SIGKILLs the child on fire. The exit
/// code then reads -1 (signal death), which the caller maps to `failed`.
async fn spawn_and_capture_cancellable(
    command: &str,
    args: &[String],
    cwd: &std::path::Path,
    on_line: &(dyn Fn(&str) + Sync),
    kill: Option<&RunHandle>,
) -> Result<(i32, String)> {
    let mut child = Command::new(command)
        .args(args)
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .spawn()
        .with_context(|| format!("spawn {command}"))?;

    let stdout = child.stdout.take().expect("piped stdout");
    let mut stderr = child.stderr.take().expect("piped stderr");

    // stderr: keep a rolling tail for the failed-run state.
    let stderr_task = tokio::spawn(async move {
        let mut buf = Vec::new();
        let _ = stderr.read_to_end(&mut buf).await;
        let start = buf.len().saturating_sub(STDERR_TAIL_BYTES);
        String::from_utf8_lossy(&buf[start..]).into_owned()
    });

    let killed = async {
        match kill {
            Some(handle) => handle.killed().await,
            None => std::future::pending().await,
        }
    };
    tokio::pin!(killed);

    let mut lines = BufReader::new(stdout).lines();
    loop {
        tokio::select! {
            line = lines.next_line() => match line {
                Ok(Some(line)) => on_line(&line),
                _ => break,
            },
            _ = &mut killed => {
                let _ = child.start_kill();
                break;
            }
        }
    }

    let status = child.wait().await.context("wait for predictor")?;
    let stderr_tail = stderr_task.await.unwrap_or_default();
    Ok((status.code().unwrap_or(-1), stderr_tail))
}

fn append_progress(run_dir: &PathBuf, line: &str) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(run_dir.join("progress.jsonl"))?;
    writeln!(f, "{line}")?;
    Ok(())
}

/// All progress lines recorded so far for a finished (or stale) run.
pub fn read_progress(run_dir: &PathBuf) -> Vec<String> {
    std::fs::read_to_string(run_dir.join("progress.jsonl"))
        .map(|t| t.lines().map(str::to_string).collect())
        .unwrap_or_default()
}
