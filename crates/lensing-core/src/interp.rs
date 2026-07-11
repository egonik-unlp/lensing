use std::path::Path;

use serde::{Deserialize, Serialize};

/// A persisted interpretability analysis: one row of `interp_analyses`, the
/// database mirror of an interp job's in-memory result and its on-disk output
/// under `data/interp/<id>/`.
///
/// `tool` names the analysis kind (`model-sae`, `sae`, `layer-probe`,
/// `layer-probe-compare`, `embedding-probe`). `model`/`predictor` are set for
/// model-centric tools and `None` for dataset-centric ones. `config` captures
/// the request knobs so a list row can describe the run and the UI can re-launch
/// it. `result` is the full analysis document (identical to what
/// `GET /api/jobs/{id}` returns on completion); it is `None` while running or
/// failed and is omitted from list payloads to keep them small.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpAnalysis {
    /// The interp job id, e.g. `interp-run-20260709-010410-547ef-msae`.
    pub id: String,
    pub tool: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub dataset_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predictor: Option<String>,
    /// Request knobs (layers, n_atoms, l1, epochs, compare_embedding, …).
    pub config: serde_json::Value,
    /// `running` | `done` | `failed`.
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// `manual` (a UI/API request) | `auto` (queued on model promotion).
    pub source: String,
    /// RFC 3339, UTC — when the analysis was created (job start).
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<String>,
    /// The full analysis document; `None` until done, and omitted from list
    /// payloads (only `GET /api/interp/analyses/{id}` carries it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
}

impl InterpAnalysis {
    /// A fresh `running` analysis with no result yet.
    #[allow(clippy::too_many_arguments)]
    pub fn running(
        id: String,
        tool: &str,
        model: Option<String>,
        dataset_id: String,
        predictor: Option<String>,
        config: serde_json::Value,
        source: &str,
        created_at: String,
    ) -> Self {
        Self {
            id,
            tool: tool.to_string(),
            model,
            dataset_id,
            predictor,
            config,
            status: "running".to_string(),
            error: None,
            source: source.to_string(),
            created_at,
            finished_at: None,
            result: None,
        }
    }

    /// Reconstruct a completed analysis from its on-disk job dir
    /// (`<interp_root>/<id>/`) — for the startup backfill and the DB-absent
    /// read-back path. `models_root` (`data/models`) is consulted to recover the
    /// predictor family for model-centric tools. Returns `None` when `id` isn't a
    /// recognizable interp job, or its single result file is absent (e.g.
    /// `layer-probe-compare`, which writes per-model files, not one document).
    pub fn from_disk(
        interp_root: &Path,
        models_root: &Path,
        id: &str,
        with_result: bool,
    ) -> Option<Self> {
        let (tool, created_at) = parse_interp_id(id)?;
        let filename = match tool {
            "model-sae" => "model-sae.json",
            "sae" => "sae.json",
            "layer-probe" => "layer-probe.json",
            "embedding-probe" => "embedding-probe.json",
            _ => return None,
        };
        let text = std::fs::read_to_string(interp_root.join(id).join(filename)).ok()?;
        let doc: serde_json::Value = serde_json::from_str(&text).ok()?;

        let model = doc.get("model_dir").and_then(|v| v.as_str()).map(str::to_string);
        let dataset_id = doc
            .get("dataset_id")
            .or_else(|| doc.get("dataset"))
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| "unknown".to_string());
        let predictor = model.as_deref().and_then(|m| {
            let text = std::fs::read_to_string(models_root.join(m).join("record.json")).ok()?;
            let rec: serde_json::Value = serde_json::from_str(&text).ok()?;
            rec.get("predictor").and_then(|v| v.as_str()).map(str::to_string)
        });
        let mut config = doc.get("config").cloned().unwrap_or_else(|| serde_json::json!({}));
        if let (Some(obj), Some(ran)) =
            (config.as_object_mut(), doc.pointer("/embedding_diff/ran").and_then(|v| v.as_bool()))
        {
            obj.insert("compare_embedding".into(), serde_json::Value::Bool(ran));
        }

        Some(Self {
            id: id.to_string(),
            tool: tool.to_string(),
            model,
            dataset_id,
            predictor,
            config,
            status: "done".to_string(),
            error: None,
            source: "manual".to_string(),
            created_at,
            finished_at: None,
            result: if with_result { Some(doc) } else { None },
        })
    }
}

/// `interp-run-YYYYMMDD-HHMMSS-<hex>-<suffix>` → (tool, RFC-3339 created_at).
/// The suffix names the tool; the embedded stamp is the creation time. A
/// malformed-but-suffixed id still yields the tool with an epoch fallback time.
fn parse_interp_id(id: &str) -> Option<(&'static str, String)> {
    let rest = id.strip_prefix("interp-run-")?;
    let parts: Vec<&str> = rest.split('-').collect();
    let date = *parts.first()?;
    let time = *parts.get(1)?;
    let suffix = *parts.last()?;
    let tool = match suffix {
        "msae" => "model-sae",
        "sae" => "sae",
        "lp" => "layer-probe",
        "cmp" => "layer-probe-compare",
        "ep" => "embedding-probe",
        _ => return None,
    };
    let digits = date.len() == 8
        && time.len() == 6
        && date.bytes().chain(time.bytes()).all(|b| b.is_ascii_digit());
    let created_at = if digits {
        format!(
            "{}-{}-{}T{}:{}:{}Z",
            &date[0..4], &date[4..6], &date[6..8], &time[0..2], &time[2..4], &time[4..6]
        )
    } else {
        "1970-01-01T00:00:00Z".to_string()
    };
    Some((tool, created_at))
}
