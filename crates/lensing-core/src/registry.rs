//! `registry.toml` parsing — the predictor registry. Lives in pg-core so
//! both lensing-server (orchestration) and meta-predictors that spawn member
//! predictors (predictor-blend) share one schema and one args-template
//! substitution.

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// `registry.toml` at the repository root.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Registry {
    pub predictors: Vec<Predictor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Predictor {
    pub name: String,
    pub display_name: String,
    #[serde(default)]
    pub description: String,
    /// Implementation language (e.g. "Rust", "Python", "Julia"); shown as a
    /// badge wherever the predictor appears in the UI.
    #[serde(default)]
    pub language: Option<String>,
    /// Library/framework the predictor is built on (e.g. "burn", "PyTorch").
    #[serde(default)]
    pub framework: Option<String>,
    pub command: String,
    pub args: Vec<String>,
    /// Predict-subcommand executable; defaults to `command` when absent.
    #[serde(default)]
    pub predict_command: Option<String>,
    /// Predict-subcommand args template ({model}/{input}/{output} tokens).
    /// Absent means the predictor is train-only: its runs cannot be promoted.
    #[serde(default)]
    pub predict_args: Option<Vec<String>>,
    /// Honors the graceful-stop protocol: polls the run dir's STOP file
    /// between epochs and finishes early (eval + checkpoint, exit 0).
    /// Without it, stopping a run means killing the process.
    #[serde(default)]
    pub supports_stop: bool,
    /// Writes `viz.svg` (architecture diagram) into the run dir at training
    /// start; served on the run and any model promoted from it.
    #[serde(default)]
    pub visualization: bool,
    #[serde(default)]
    pub params: Vec<Param>,
}

/// One hyperparameter schema entry; drives the auto-generated UI form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(rename = "type")]
    pub kind: ParamKind,
    pub default: serde_json::Value,
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
    #[serde(default)]
    pub options: Option<Vec<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParamKind {
    Int,
    Float,
    Bool,
    Ints,
    Enum,
    /// Arbitrary JSON value (array/object) — passes validation untouched;
    /// the UI renders a JSON textarea. For structured params no scalar kind
    /// can express (e.g. a blend's member list).
    Json,
}

impl Registry {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?;
        let reg: Registry = toml::from_str(&text).context("parse registry.toml")?;
        Ok(reg)
    }

    pub fn get(&self, name: &str) -> Option<&Predictor> {
        self.predictors.iter().find(|p| p.name == name)
    }
}

impl Predictor {
    /// Default hyperparams from the schema.
    pub fn default_hyperparams(&self) -> serde_json::Value {
        let map: serde_json::Map<String, serde_json::Value> = self
            .params
            .iter()
            .map(|p| (p.name.clone(), p.default.clone()))
            .collect();
        serde_json::Value::Object(map)
    }

    /// Merge submitted hyperparams over the schema defaults, rejecting
    /// unknown keys. The single validation point for runs and model
    /// definitions; the result always carries every schema key.
    pub fn merge_hyperparams(
        &self,
        submitted: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let mut hp = self.default_hyperparams();
        if let (Some(serde_json::Value::Object(submitted)), serde_json::Value::Object(base)) =
            (submitted, &mut hp)
        {
            for (k, v) in submitted {
                anyhow::ensure!(base.contains_key(&k), "unknown hyperparameter {k}");
                base.insert(k, v);
            }
        }
        Ok(hp)
    }

    /// Whether this predictor implements the contract v2 predict subcommand.
    pub fn supports_predict(&self) -> bool {
        self.predict_args.is_some()
    }

    /// Effective (command, args template) for a predict invocation.
    pub fn predict_invocation(&self) -> Option<(&str, &[String])> {
        self.predict_args
            .as_deref()
            .map(|args| (self.predict_command.as_deref().unwrap_or(&self.command), args))
    }
}

/// Substitute `{token}` placeholders in an args template.
pub fn substitute(template: &[String], pairs: &[(&str, String)]) -> Vec<String> {
    template
        .iter()
        .map(|a| {
            let mut s = a.clone();
            for (token, value) in pairs {
                s = s.replace(&format!("{{{token}}}"), value);
            }
            s
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn predictor_with_json_param() -> Predictor {
        Predictor {
            name: "blend".into(),
            display_name: "Blend".into(),
            description: String::new(),
            language: None,
            framework: None,
            command: "x".into(),
            args: vec![],
            predict_command: None,
            predict_args: None,
            supports_stop: false,
            visualization: false,
            params: vec![Param {
                name: "members".into(),
                label: None,
                kind: ParamKind::Json,
                default: serde_json::json!([]),
                min: None,
                max: None,
                options: None,
            }],
        }
    }

    #[test]
    fn json_param_kind_round_trips() {
        let s = serde_json::to_string(&ParamKind::Json).unwrap();
        assert_eq!(s, "\"json\"");
        let back: ParamKind = serde_json::from_str(&s).unwrap();
        assert_eq!(back, ParamKind::Json);
    }

    #[test]
    fn merge_passes_json_value_through_and_rejects_unknown_keys() {
        let p = predictor_with_json_param();
        let members = serde_json::json!([{ "model": "m", "weight": 0.5 }]);
        let merged = p
            .merge_hyperparams(Some(serde_json::json!({ "members": members.clone() })))
            .unwrap();
        assert_eq!(merged["members"], members);
        assert!(p
            .merge_hyperparams(Some(serde_json::json!({ "nope": 1 })))
            .is_err());
    }
}
