//! `registry.toml` parsing — the predictor registry. Lives in lensing-core so
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
    /// Export-subcommand executable; defaults to `command` when absent.
    #[serde(default)]
    pub export_command: Option<String>,
    /// Export-subcommand args template ({model}/{output} tokens). The
    /// subcommand reads a promoted-model dir and writes a portable `model.onnx`
    /// (input: the assembled feature vector, output: transformed-space target)
    /// into {output}. Absent means the predictor family cannot be exported.
    #[serde(default)]
    pub export_args: Option<Vec<String>>,
    /// Probe-subcommand executable; defaults to `command` when absent.
    #[serde(default)]
    pub probe_command: Option<String>,
    /// Layer-probe args template ({model}/{dataset}/{output} tokens). The
    /// subcommand reads a promoted-model dir + a dataset dir, fits a linear
    /// probe on each stage of the network's forward pass (Alain & Bengio 2016),
    /// and writes a `layer-probe.json` report into {output}. Absent means the
    /// predictor family has no interpretability probe (only the native burn
    /// nets, whose activations are reachable from Rust, implement it).
    #[serde(default)]
    pub probe_args: Option<Vec<String>>,
    /// Model-SAE subcommand executable; defaults to `command` when absent.
    #[serde(default)]
    pub model_sae_command: Option<String>,
    /// Per-model SAE args template ({model}/{dataset}/{output} tokens). The
    /// subcommand trains a sparse autoencoder on the promoted model's own hidden
    /// activations and writes a `model-sae.json` report into {output}. Absent
    /// means the family has no per-model SAE (only the native burn nets, whose
    /// activations are reachable from Rust, implement it).
    #[serde(default)]
    pub model_sae_args: Option<Vec<String>>,
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

    /// Whether this predictor implements the optional `export` subcommand
    /// (native trained model → portable `model.onnx`).
    pub fn supports_export(&self) -> bool {
        self.export_args.is_some()
    }

    /// Effective (command, args template) for an export invocation.
    pub fn export_invocation(&self) -> Option<(&str, &[String])> {
        self.export_args
            .as_deref()
            .map(|args| (self.export_command.as_deref().unwrap_or(&self.command), args))
    }

    /// Whether this predictor implements the optional `layer-probe` subcommand
    /// (per-stage linear-probe interpretability analysis).
    pub fn supports_probe(&self) -> bool {
        self.probe_args.is_some()
    }

    /// Effective (command, args template) for a layer-probe invocation.
    pub fn probe_invocation(&self) -> Option<(&str, &[String])> {
        self.probe_args
            .as_deref()
            .map(|args| (self.probe_command.as_deref().unwrap_or(&self.command), args))
    }

    /// Whether this predictor implements the optional per-model `model-sae`
    /// subcommand (dictionary learning on its own hidden activations).
    pub fn supports_model_sae(&self) -> bool {
        self.model_sae_args.is_some()
    }

    /// Effective (command, args template) for a model-SAE invocation.
    pub fn model_sae_invocation(&self) -> Option<(&str, &[String])> {
        self.model_sae_args
            .as_deref()
            .map(|args| (self.model_sae_command.as_deref().unwrap_or(&self.command), args))
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
            export_command: None,
            export_args: None,
            probe_command: None,
            probe_args: None,
            model_sae_command: None,
            model_sae_args: None,
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
