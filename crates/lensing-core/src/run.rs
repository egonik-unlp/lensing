use serde::{Deserialize, Serialize};

/// `meta.json` in a run directory. Owned by lensing-server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMeta {
    pub run_id: String,
    pub dataset_id: String,
    pub predictor: String,
    pub hyperparams: serde_json::Value,
    pub status: RunStatus,
    /// RFC 3339, UTC.
    pub started_at: String,
    pub finished_at: Option<String>,
    pub exit_code: Option<i32>,
    /// Tail of stderr, kept for the UI's failed-run state.
    pub stderr_tail: Option<String>,
    /// Present once the run succeeds.
    pub metrics: Option<Metrics>,
    /// Predictor contract version the run was produced under. Runs written
    /// before the predict subcommand existed deserialize as 1.
    #[serde(default = "default_contract_version")]
    pub contract_version: u32,
    /// True when the run directory holds a loadable checkpoint: set on
    /// finish for predict-capable predictors, or mid-run on the first
    /// periodic `checkpoint` event. Gates promotion for non-succeeded runs.
    #[serde(default)]
    pub has_checkpoint: bool,
    /// Model definition (`models.toml`) this run was launched from, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_definition: Option<String>,
}

fn default_contract_version() -> u32 {
    1
}

/// The current predictor contract version written into new run metas.
pub const CONTRACT_VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    /// Enqueued for a remote training worker; not yet claimed.
    Queued,
    Running,
    Succeeded,
    Failed,
    Interrupted,
    /// Stopped early on user request; the predictor still evaluated and wrote
    /// its checkpoint, so the run is promotable like a succeeded one.
    Stopped,
}

/// `metrics.json` written by a predictor. All values in target space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    pub mae: f64,
    pub rmse: f64,
    pub r2: f64,
    /// Mean absolute percentage error, as a fraction (0.25 = 25%).
    pub mape: f64,
    /// Median absolute percentage error, as a fraction.
    pub medape: f64,
    pub n_test: usize,
}

/// One element of `predictions.json` written by a predictor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub row_id: u64,
    /// Target space.
    pub actual: f64,
    /// Target space.
    pub predicted: f64,
}

/// One element of a predict subcommand's output file. Target space; there is
/// no ground truth at inference time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferencePrediction {
    pub row_id: u64,
    pub predicted: f64,
}

/// One JSON line on a predictor's stdout.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum ProgressEvent {
    Epoch { epoch: usize, total_epochs: usize, train_loss: f64, val_loss: f64 },
    Log { msg: String },
    /// Periodic checkpoint saved (`checkpoint_every` hyperparam); the run dir
    /// now holds loadable params even if training dies later.
    Checkpoint { epoch: usize },
    /// The predictor saw the STOP file and is finishing up (eval + final
    /// checkpoint) before exiting.
    Stopping,
    Done,
}

/// Compute price-space metrics from (actual, predicted) pairs.
pub fn compute_metrics(pairs: &[(f64, f64)]) -> Metrics {
    let n = pairs.len();
    assert!(n > 0, "no predictions");
    let mut abs_err = 0.0;
    let mut sq_err = 0.0;
    let mut apes: Vec<f64> = Vec::with_capacity(n);
    let mean_actual = pairs.iter().map(|(a, _)| a).sum::<f64>() / n as f64;
    let mut ss_tot = 0.0;
    for &(actual, predicted) in pairs {
        let e = predicted - actual;
        abs_err += e.abs();
        sq_err += e * e;
        ss_tot += (actual - mean_actual).powi(2);
        // actual is never 0 here (dataset filter excludes target==0)
        apes.push((e / actual).abs());
    }
    apes.sort_by(|a, b| a.total_cmp(b));
    let medape = if n % 2 == 1 {
        apes[n / 2]
    } else {
        (apes[n / 2 - 1] + apes[n / 2]) / 2.0
    };
    Metrics {
        mae: abs_err / n as f64,
        rmse: (sq_err / n as f64).sqrt(),
        r2: 1.0 - sq_err / ss_tot,
        mape: apes.iter().sum::<f64>() / n as f64,
        medape,
        n_test: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// meta.json written before contract v2 (no contract_version /
    /// has_checkpoint fields) must keep deserializing.
    #[test]
    fn old_run_meta_deserializes() {
        let old = r#"{
            "run_id": "run-20260101-000000-1-burn-mlp",
            "dataset_id": "ds-x",
            "predictor": "burn-mlp",
            "hyperparams": {"epochs": 50},
            "status": "succeeded",
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:05:00Z",
            "exit_code": 0,
            "stderr_tail": null,
            "metrics": {"mae":1.0,"rmse":2.0,"r2":0.5,"mape":0.2,"medape":0.1,"n_test":10}
        }"#;
        let meta: RunMeta = serde_json::from_str(old).unwrap();
        assert_eq!(meta.contract_version, 1);
        assert!(!meta.has_checkpoint);
        assert!(meta.from_definition.is_none());
        assert_eq!(meta.status, RunStatus::Succeeded);
    }

    #[test]
    fn stopped_status_round_trips() {
        let s = serde_json::to_string(&RunStatus::Stopped).unwrap();
        assert_eq!(s, "\"stopped\"");
        let back: RunStatus = serde_json::from_str(&s).unwrap();
        assert_eq!(back, RunStatus::Stopped);
    }

    #[test]
    fn new_progress_events_round_trip() {
        let ck: ProgressEvent = serde_json::from_str(r#"{"event":"checkpoint","epoch":50}"#).unwrap();
        assert!(matches!(ck, ProgressEvent::Checkpoint { epoch: 50 }));
        let st: ProgressEvent = serde_json::from_str(r#"{"event":"stopping"}"#).unwrap();
        assert!(matches!(st, ProgressEvent::Stopping));
    }
}
