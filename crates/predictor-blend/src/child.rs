//! Member predictor processes: train with progress forwarding + graceful-stop
//! propagation, and synchronous predict invocations.

use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use lensing_core::registry::{substitute, Predictor};

use crate::emit;

pub const STOP_FILE: &str = "STOP";

/// Train a member in `member_dir` on `dataset_dir`, forwarding its progress
/// as prefixed log events. When `blend_stop` (the blend run dir's STOP file)
/// appears, it is propagated into the member dir so STOP-honoring members
/// finish gracefully; returns `stopped = true` in that case.
pub fn train_member(
    label: &str,
    predictor: &Predictor,
    dataset_dir: &Path,
    member_dir: &Path,
    hp_path: &Path,
    blend_stop: &Path,
) -> Result<bool> {
    let args = substitute(
        &predictor.args,
        &[
            ("dataset", dataset_dir.to_string_lossy().into_owned()),
            ("run_dir", member_dir.to_string_lossy().into_owned()),
            ("hyperparams", hp_path.to_string_lossy().into_owned()),
        ],
    );
    let mut child = Command::new(&predictor.command)
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawn {} for member {label}", predictor.command))?;

    // Forward stdout lines as logs from a reader thread; epoch events keep
    // their numbers but become logs (the blend's own epoch axis is members).
    let stdout = child.stdout.take().context("member stdout")?;
    let label_owned = label.to_string();
    let stdout_thread = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(|l| l.ok()) {
            let msg = match serde_json::from_str::<serde_json::Value>(&line) {
                Ok(v) if v["event"] == "epoch" => format!(
                    "[{label_owned}] epoch {}/{} train {:.4} val {:.4}",
                    v["epoch"], v["total_epochs"],
                    v["train_loss"].as_f64().unwrap_or(0.0),
                    v["val_loss"].as_f64().unwrap_or(0.0)
                ),
                Ok(v) if v["event"] == "log" => {
                    format!("[{label_owned}] {}", v["msg"].as_str().unwrap_or(""))
                }
                Ok(v) if v["event"] == "checkpoint" || v["event"] == "done" => continue,
                Ok(v) if v["event"] == "stopping" => format!("[{label_owned}] stopping"),
                _ => format!("[{label_owned}] {line}"),
            };
            emit(serde_json::json!({"event": "log", "msg": msg}));
        }
    });
    let mut stderr = child.stderr.take().context("member stderr")?;
    let stderr_thread = std::thread::spawn(move || {
        let mut buf = String::new();
        let _ = stderr.read_to_string(&mut buf);
        buf
    });

    // Wait loop: poll the child and the blend STOP marker.
    let stop_seen = Arc::new(AtomicBool::new(false));
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if blend_stop.exists() && !stop_seen.load(Ordering::Relaxed) {
            stop_seen.store(true, Ordering::Relaxed);
            emit(serde_json::json!({"event": "stopping"}));
            // Propagate; non-honoring members just finish their run.
            let _ = std::fs::write(member_dir.join(STOP_FILE), b"");
        }
        std::thread::sleep(std::time::Duration::from_millis(300));
    };
    let _ = stdout_thread.join();
    let stderr_tail: String = stderr_thread.join().unwrap_or_default();

    if !status.success() {
        let tail: String = stderr_tail.lines().rev().take(8).collect::<Vec<_>>().join(" | ");
        bail!("member {label} ({}) exited {status}: {tail}", predictor.name);
    }
    Ok(stop_seen.load(Ordering::Relaxed) || blend_stop.exists())
}

/// Run a member's export subcommand synchronously, writing its `model.onnx`
/// into `output_dir`. Errors if the predictor has no export support or exits
/// non-zero.
pub fn export_member(
    label: &str,
    predictor: &Predictor,
    model_dir: &Path,
    output_dir: &Path,
) -> Result<()> {
    let (command, template) = predictor.export_invocation().with_context(|| {
        format!("member {label}: predictor {} cannot be exported to ONNX", predictor.name)
    })?;
    let args = substitute(
        template,
        &[
            ("model", model_dir.to_string_lossy().into_owned()),
            ("output", output_dir.to_string_lossy().into_owned()),
        ],
    );
    std::fs::create_dir_all(output_dir)?;
    let out = Command::new(command)
        .args(&args)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("spawn {command} export for member {label}"))?;
    if !out.status.success() {
        let tail: String = String::from_utf8_lossy(&out.stderr)
            .lines()
            .rev()
            .take(8)
            .collect::<Vec<_>>()
            .join(" | ");
        bail!("member {label} export exited {}: {tail}", out.status);
    }
    if !output_dir.join("model.onnx").is_file() {
        bail!("member {label} export exited 0 but wrote no model.onnx");
    }
    Ok(())
}

/// Run a member's predict subcommand synchronously; returns the parsed
/// target-space predictions.
pub fn predict_member(
    label: &str,
    predictor: &Predictor,
    model_dir: &Path,
    input_dir: &Path,
    output_file: &Path,
) -> Result<Vec<lensing_core::InferencePrediction>> {
    let (command, template) = predictor
        .predict_invocation()
        .with_context(|| format!("member {label}: predictor {} is train-only", predictor.name))?;
    let args = substitute(
        template,
        &[
            ("model", model_dir.to_string_lossy().into_owned()),
            ("input", input_dir.to_string_lossy().into_owned()),
            ("output", output_file.to_string_lossy().into_owned()),
        ],
    );
    let out = Command::new(command)
        .args(&args)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("spawn {command} predict for member {label}"))?;
    if !out.status.success() {
        let tail: String = String::from_utf8_lossy(&out.stderr)
            .lines()
            .rev()
            .take(8)
            .collect::<Vec<_>>()
            .join(" | ");
        bail!("member {label} predict exited {}: {tail}", out.status);
    }
    let text = std::fs::read_to_string(output_file)
        .with_context(|| format!("member {label} predict wrote no output"))?;
    Ok(serde_json::from_str(&text).context("parse member predictions")?)
}
