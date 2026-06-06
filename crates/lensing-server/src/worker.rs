//! Training worker role: `lensing-server worker --hub-url <hub>`.
//!
//! Claims queued runs from Postgres (FOR UPDATE SKIP LOCKED — concurrent
//! workers never double-claim), materializes the dataset (local dir if
//! present, else downloaded from the hub's archive endpoint), spawns the
//! predictor's train process, streams progress + final metrics into the
//! database, and uploads the predictor-written artifacts so promotion and
//! inspection work from anywhere.
//!
//! Requirements on the worker host: this binary + `registry.toml` + the
//! predictor toolchains it will run (target/release predictor binaries,
//! predictors/.venv, julia envs — i.e. a checkout/image of the repo), plus
//! DATABASE_URL and the hub URL. `--once` processes a single run and exits
//! (batch/lambda-style invocation); without it the worker polls forever.
//! Graceful STOP is not supported on remote runs yet — stop before
//! enqueueing or let the run finish.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use chrono::Utc;
use lensing_core::{Metrics, ProgressEvent, RunMeta, RunStatus};
use serde_json::json;

use crate::registry::Registry;
use crate::runs;

pub async fn run(
    root: PathBuf,
    database_url: String,
    hub_url: String,
    once: bool,
    poll_secs: u64,
) -> Result<()> {
    let db = lensing_db::connect(&database_url).await?;
    let registry = Registry::load(&root.join("registry.toml"))?;
    let worker_id = format!(
        "{}:{}",
        std::env::var("HOSTNAME").unwrap_or_else(|_| "worker".into()),
        std::process::id()
    );
    std::fs::create_dir_all(root.join("data/datasets"))?;
    std::fs::create_dir_all(root.join("data/runs"))?;
    eprintln!("[worker {worker_id}] connected ({database_url}); hub {hub_url}");

    loop {
        let claimed =
            lensing_db::queries::claim_queued_run(&db, &worker_id, &Utc::now().to_rfc3339()).await?;
        match claimed {
            Some(meta) => {
                eprintln!("[worker {worker_id}] claimed {}", meta.run_id);
                if let Err(e) = process(&db, &registry, &root, &hub_url, meta.clone()).await {
                    eprintln!("[worker {worker_id}] {} failed: {e:#}", meta.run_id);
                    let mut failed = meta;
                    failed.status = RunStatus::Failed;
                    failed.finished_at = Some(Utc::now().to_rfc3339());
                    failed.stderr_tail = Some(format!("worker error: {e:#}"));
                    let _ = lensing_db::queries::upsert_run(&db, &failed).await;
                }
                if once {
                    return Ok(());
                }
            }
            None => {
                if once {
                    eprintln!("[worker {worker_id}] queue empty");
                    return Ok(());
                }
                tokio::time::sleep(std::time::Duration::from_secs(poll_secs)).await;
            }
        }
    }
}

async fn process(
    db: &lensing_db::Db,
    registry: &Registry,
    root: &Path,
    hub_url: &str,
    mut meta: RunMeta,
) -> Result<()> {
    let predictor = registry
        .get(&meta.predictor)
        .with_context(|| format!("predictor {} is not in this worker's registry", meta.predictor))?
        .clone();

    // Dataset: local dir wins; otherwise download the hub's archive.
    let dataset_dir = root.join("data/datasets").join(&meta.dataset_id);
    if !dataset_dir.join("manifest.json").is_file() {
        fetch_dataset(hub_url, &meta.dataset_id, &dataset_dir).await?;
    }

    let run_dir = root.join("data/runs").join(&meta.run_id);
    std::fs::create_dir_all(&run_dir)?;
    std::fs::write(run_dir.join("hp.json"), serde_json::to_vec_pretty(&meta.hyperparams)?)?;

    // Progress lines stream to the database through the ordered sink.
    let sink = lensing_db::sink::DbSink::spawn(db.clone());
    let run_id = meta.run_id.clone();
    let sink2 = sink.clone();
    let on_line = move |line: &str| {
        let recorded = match serde_json::from_str::<ProgressEvent>(line) {
            Ok(ev) => serde_json::to_string(&ev).unwrap(),
            Err(_) if !line.trim().is_empty() => {
                json!({"event":"log","msg":format!("[stdout] {line}")}).to_string()
            }
            Err(_) => return,
        };
        sink2.run_event(&run_id, &recorded);
    };

    let args = crate::registry::substitute(
        &predictor.args,
        &[
            ("dataset", dataset_dir.to_string_lossy().into_owned()),
            ("run_dir", run_dir.to_string_lossy().into_owned()),
            ("hyperparams", run_dir.join("hp.json").to_string_lossy().into_owned()),
        ],
    );
    let (exit_code, stderr_tail) =
        runs::spawn_and_capture(&predictor.command, &args, root, &on_line).await?;

    // Terminal meta, mirroring the hub's orchestrate logic (sans stop).
    meta.finished_at = Some(Utc::now().to_rfc3339());
    meta.exit_code = Some(exit_code);
    meta.stderr_tail = if stderr_tail.is_empty() { None } else { Some(stderr_tail) };
    let metrics: Option<Metrics> = std::fs::read_to_string(run_dir.join("metrics.json"))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok());
    if exit_code == 0 && metrics.is_some() {
        meta.metrics = metrics;
        meta.status = RunStatus::Succeeded;
        meta.has_checkpoint = predictor.supports_predict();
    } else {
        meta.status = RunStatus::Failed;
    }
    std::fs::write(run_dir.join("meta.json"), serde_json::to_vec_pretty(&meta)?)?;

    // Upload everything the predictor wrote (checkpoints, metrics,
    // predictions, viz) so the hub can inspect and promote without this
    // host's disk. meta/hp/progress live in the runs tables already.
    let files = collect_artifacts(&run_dir)?;
    let n = files.len();
    lensing_db::queries::put_run_artifacts(db, &meta.run_id, &files).await?;
    lensing_db::queries::upsert_run(db, &meta).await?;
    sink.run_event(
        &meta.run_id,
        &json!({"event":"status","status":meta.status}).to_string(),
    );
    eprintln!(
        "[worker] {} -> {:?} ({n} artifacts uploaded)",
        meta.run_id, meta.status
    );
    Ok(())
}

/// Predictor-written files in a run dir: everything except the server-owned
/// set (which lives in the database) and atomic-write temps.
fn collect_artifacts(run_dir: &Path) -> Result<Vec<(String, Vec<u8>)>> {
    const DB_OWNED: &[&str] = &["meta.json", "hp.json", "progress.jsonl", runs::STOP_FILE];
    let mut out = Vec::new();
    collect_rec(run_dir, run_dir, DB_OWNED, &mut out)?;
    Ok(out)
}

fn collect_rec(
    base: &Path,
    dir: &Path,
    top_excluded: &[&str],
    out: &mut Vec<(String, Vec<u8>)>,
) -> Result<()> {
    for e in std::fs::read_dir(dir)?.flatten() {
        let path = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        if dir == base && top_excluded.contains(&name.as_str()) {
            continue;
        }
        if name.contains(".tmp") || name.contains("-tmp") {
            continue;
        }
        let ft = e.file_type()?;
        if ft.is_symlink() {
            continue;
        }
        if ft.is_dir() {
            collect_rec(base, &path, top_excluded, out)?;
        } else {
            let rel = path.strip_prefix(base).unwrap().to_string_lossy().into_owned();
            out.push((rel, std::fs::read(&path)?));
        }
    }
    Ok(())
}

/// Download + unpack `GET <hub>/api/datasets/<id>/archive` (tar.gz).
async fn fetch_dataset(hub_url: &str, dataset_id: &str, dataset_dir: &Path) -> Result<()> {
    let url = format!("{}/api/datasets/{}/archive", hub_url.trim_end_matches('/'), dataset_id);
    eprintln!("[worker] fetching dataset {dataset_id} from {url}");
    let dir = dataset_dir.to_path_buf();
    let parent = dataset_dir
        .parent()
        .context("dataset dir has no parent")?
        .to_path_buf();
    tokio::task::spawn_blocking(move || -> Result<()> {
        let resp = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()?
            .get(&url)
            .send()
            .context("dataset archive request")?;
        if !resp.status().is_success() {
            bail!("dataset archive request failed: {}", resp.status());
        }
        let gz = flate2::read::GzDecoder::new(resp);
        let mut archive = tar::Archive::new(gz);
        // The archive's root entry is the dataset id, so unpacking into the
        // datasets dir yields exactly `dataset_dir`.
        archive.unpack(&parent).context("unpack dataset archive")?;
        anyhow::ensure!(
            dir.join("manifest.json").is_file(),
            "archive did not contain a dataset manifest"
        );
        Ok(())
    })
    .await
    .context("dataset fetch task panicked")?
}
