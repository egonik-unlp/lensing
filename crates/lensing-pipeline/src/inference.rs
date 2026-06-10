//! Inference-time featurization: rebuild the frozen featurization contract
//! of a promoted model and turn raw items into the exact feature matrix the
//! model was trained on.

use std::fs;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde::Deserialize;

use lensing_core::artifact::{read_f32, write_f32, write_u64};
use lensing_core::{ColumnDesc, Contract, InputFields, InputManifest, TargetInfo};

use crate::build::write_items;
use crate::features::{self, Encoder};
use crate::pca::PcaModel;
use crate::qdrant::{Payload, RawPoint};

/// A raw inference item as posted by API callers: the payload fields the
/// model consumes (flat, by field name) plus the embedding vector. Unknown
/// fields are carried along, missing fields take the same defaults training
/// used.
#[derive(Debug, Clone, Deserialize)]
pub struct RawItem {
    /// Optional caller-supplied id, echoed back as `row_id`. Items without
    /// one get their zero-based position in the request.
    #[serde(default)]
    pub id: Option<u64>,
    pub embedding: Vec<f32>,
    /// Everything else: the domain's metadata fields, the document text and
    /// any payload-root keys (cluster, …), flat by name.
    #[serde(flatten)]
    pub fields: serde_json::Map<String, serde_json::Value>,
}

impl RawItem {
    /// Convert to the pipeline's point type, using `index` as the fallback id.
    pub fn into_point(self, index: usize) -> RawPoint {
        RawPoint {
            id: self.id.unwrap_or(index as u64),
            vector: self.embedding,
            payload: Payload { fields: self.fields },
        }
    }
}

/// The reconstructed featurization contract of one promoted model. Built
/// from `contract.json` + `pca_components.f32` in a model directory;
/// produces feature matrices bit-identical to the training dataset's.
pub struct Featurizer {
    contract: Contract,
    encoder: Encoder,
    pca: PcaModel,
}

impl Featurizer {
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let contract: Contract = serde_json::from_str(
            &fs::read_to_string(dir.join("contract.json")).context("read contract.json")?,
        )
        .context("parse contract.json")?;
        let components = read_f32(&dir.join("pca_components.f32"))?;
        Self::from_contract(contract, &components)
    }

    pub fn from_contract(contract: Contract, components: &[f32]) -> Result<Self> {
        let [k, d] = contract.pca.components_shape;
        ensure!(k == contract.pca.dims, "contract pca dims/shape mismatch");
        let pca = PcaModel::from_parts(
            &contract.pca.mean,
            components,
            k,
            d,
            contract.pca.explained_variance_ratio.clone(),
        )?;
        // New contracts freeze their coordinate bounds; pre-domain contracts
        // reproduce the legacy bounds they were trained under.
        let bounds = contract
            .feature_config
            .coordinate_bounds
            .unwrap_or_else(lensing_core::manifest::legacy_coordinate_bounds);
        let encoder = Encoder::from_columns_with_bounds(&contract.columns, bounds)?;
        ensure!(
            k + encoder.width() == contract.n_cols,
            "contract n_cols {} does not match pca {k} + metadata {}",
            contract.n_cols,
            encoder.width()
        );
        Ok(Self { contract, encoder, pca })
    }

    pub fn n_cols(&self) -> usize {
        self.contract.n_cols
    }

    /// Expected embedding dimension (hard requirement on inputs).
    pub fn embedding_dim(&self) -> usize {
        self.contract.pca.components_shape[1]
    }

    pub fn target(&self) -> &TargetInfo {
        &self.contract.target
    }

    pub fn columns(&self) -> &[ColumnDesc] {
        &self.contract.columns
    }

    /// Whether the contract was trained with the raw-numerics block (and
    /// therefore expects reconcile/repair/backfill before encoding).
    pub fn raw_numerics(&self) -> bool {
        self.contract.feature_config.raw_numerics
    }

    /// Whether training backfilled missing areas from the listing text.
    pub fn area_content_backfill(&self) -> bool {
        self.contract.feature_config.area_content_backfill
    }

    /// Whether the contract was trained with the coordinates block (lat/lon
    /// reconciled from the companion collection like the numerics).
    pub fn coordinates(&self) -> bool {
        self.contract.feature_config.coordinates
    }

    /// Frozen train-split medians, present when training imputed missing
    /// numerics; predict must repeat the fill after reconcile/normalize.
    pub fn imputation(&self) -> Option<&lensing_core::NumericImputation> {
        self.contract.imputation.as_ref()
    }

    /// Companion collection for the predict-time numerics join. Contracts
    /// frozen before the field existed fall back to `default` (the domain's
    /// companion collection — those contracts were built with it).
    pub fn numerics_collection(&self, default: Option<&str>) -> Option<String> {
        if !self.raw_numerics() && !self.coordinates() {
            return None;
        }
        self.contract
            .numerics_collection
            .clone()
            .or_else(|| default.map(str::to_string))
    }

    /// Featurize a batch into the trained column order (n × n_cols,
    /// row-major). Embedding dimension mismatches are hard errors.
    pub fn featurize(&self, points: &[RawPoint]) -> Result<Vec<f32>> {
        ensure!(!points.is_empty(), "no items to featurize");
        let d = self.embedding_dim();
        for (i, p) in points.iter().enumerate() {
            ensure!(
                p.vector.len() == d,
                "item {i} (row_id {}): embedding has {} dims, model expects {d}",
                p.id,
                p.vector.len()
            );
        }
        let mut vectors = Vec::with_capacity(points.len() * d);
        for p in points {
            vectors.extend_from_slice(&p.vector);
        }
        let projected = self.pca.project_all(&vectors, d);
        Ok(features::assemble(&projected, self.pca.k(), &self.encoder, points))
    }

    /// Per-item warnings (out-of-vocabulary or empty categoricals;
    /// unspecified reconciled numerics; implausible coordinates). Mirrors
    /// training semantics; informational only.
    pub fn warnings(&self, points: &[RawPoint]) -> Vec<String> {
        let mut out: Vec<String> = points
            .iter()
            .enumerate()
            .flat_map(|(i, p)| {
                self.encoder.category_warnings(p, &format!("item {i} (row_id {})", p.id))
            })
            .collect();
        // The value+indicator (or imputed) numeric fields the contract reads.
        let indicator_fields = self.encoder.indicator_fields();
        if !indicator_fields.is_empty() {
            for (i, p) in points.iter().enumerate() {
                let missing: Vec<&str> = indicator_fields
                    .iter()
                    .filter(|f| !p.payload.num_of(f).is_some_and(|x| x > 0.0))
                    .map(|f| f.as_str())
                    .collect();
                if !missing.is_empty() {
                    let handling = if self.contract.imputation.is_some() {
                        "imputed with frozen train medians"
                    } else {
                        "encoded as missing"
                    };
                    out.push(format!(
                        "item {i} (row_id {}): {} unspecified — {handling}, like training",
                        p.id,
                        missing.join(", ")
                    ));
                }
            }
        }
        if let Some(field) = self.encoder.coordinates_field() {
            let bounds = self
                .contract
                .feature_config
                .coordinate_bounds
                .unwrap_or_else(lensing_core::manifest::legacy_coordinate_bounds);
            for (i, p) in points.iter().enumerate() {
                if p.payload.coords_of(field, &bounds).is_none() {
                    out.push(format!(
                        "item {i} (row_id {}): coordinates unspecified or implausible — \
                         encoded as missing, like training",
                        p.id
                    ));
                }
            }
        }
        out
    }

    /// Write a predict input mini-artifact: `features.f32`, `row_ids.u64`,
    /// a trimmed `manifest.json`, and `items.json` (payload echo for
    /// payload-based predictors).
    pub fn write_input_dir(&self, dir: &Path, points: &[RawPoint]) -> Result<()> {
        let features = self.featurize(points)?;
        let row_ids: Vec<u64> = points.iter().map(|p| p.id).collect();
        fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        write_f32(&dir.join("features.f32"), &features)?;
        write_u64(&dir.join("row_ids.u64"), &row_ids)?;
        // "content" is the legacy/default document-text key; predict input
        // items feed payload predictors that read categoricals by name, so a
        // domain with a different content field just skips truncation here.
        write_items(dir, points, "content")?;
        let manifest = InputManifest {
            n_rows: points.len(),
            n_cols: self.contract.n_cols,
            columns: self.contract.columns.clone(),
            target: self.contract.target.clone(),
        };
        fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&manifest)?)?;
        Ok(())
    }

    /// The explicit per-column feature plan in trained order (PCA components
    /// first, then the metadata encoder's columns) — the spec the model export
    /// ships in `featurize.json` so a downstream consumer reproduces the
    /// feature vector without the lensing pipeline.
    pub fn feature_plan(&self) -> Vec<crate::features::ColumnPlan> {
        let mut plan: Vec<crate::features::ColumnPlan> = (0..self.pca.k())
            .map(|component| crate::features::ColumnPlan::Pca { component })
            .collect();
        plan.extend(self.encoder.plan());
        plan
    }

    /// The PCA basis: mean (length = embedding dim) and the row-major
    /// components matrix `[dims, embedding_dim]`, for the export's binary
    /// sidecar + `featurize.json` reference.
    pub fn pca_mean(&self) -> &[f32] {
        &self.contract.pca.mean
    }

    pub fn pca_dims(&self) -> usize {
        self.contract.pca.dims
    }

    /// Caller-facing input requirements, derived from the trained columns.
    pub fn input_fields(&self) -> InputFields {
        InputFields {
            required_numeric: self.encoder.numeric_fields(),
            required_categorical: self.encoder.categorical_groups(),
            embedding_dim: self.embedding_dim(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lensing_core::{
        ColumnKind, FeatureConfig, PcaInfo, TargetInfo, TargetTransform, CONTRACT_VERSION,
    };

    fn point(id: u64, vector: Vec<f32>, property_type: &str, bedrooms: f64) -> RawPoint {
        RawPoint {
            id,
            vector,
            payload: Payload::from_raw(
                serde_json::json!({
                    "content": "",
                    "metadata": {
                        "propertyType": property_type,
                        "operation": "sale",
                        "price": 100_000.0,
                        "bedrooms": bedrooms,
                    }
                }),
                "metadata",
            ),
        }
    }

    /// A Featurizer reconstructed from contract artifacts must reproduce the
    /// training-path features exactly (same PCA params, same column order).
    #[test]
    fn featurizer_matches_training_path() {
        let d = 8;
        let k = 3;
        let mut rng = crate::shuffle::SplitMix64(11);
        let mut points = Vec::new();
        for i in 0..40u64 {
            let vector: Vec<f32> =
                (0..d).map(|_| (rng.next_u64() % 1000) as f32 / 500.0 - 1.0).collect();
            let pt = if i % 3 == 0 { "house" } else { "apartment" };
            points.push(point(i, vector, pt, (i % 4) as f64));
        }

        // Training path: fit, quantize through f32 (as build_dataset does),
        // project, assemble.
        let mut vectors = Vec::new();
        for p in &points {
            vectors.extend_from_slice(&p.vector);
        }
        let rows: Vec<u32> = (0..points.len() as u32).collect();
        let fitted = crate::pca::fit(&vectors, d, &rows, k).unwrap();
        let mean_f32: Vec<f32> = fitted.mean.iter().map(|v| *v as f32).collect();
        let comp_flat: Vec<f32> = fitted
            .components
            .row_iter()
            .flat_map(|r| r.iter().map(|v| *v as f32).collect::<Vec<_>>())
            .collect();
        let model = PcaModel::from_parts(
            &mean_f32,
            &comp_flat,
            k,
            d,
            fitted.explained_variance_ratio.clone(),
        )
        .unwrap();
        let projected = model.project_all(&vectors, d);
        let cfg = FeatureConfig {
            pca_dims: k,
            neighborhood_top_n: 0,
            ..Default::default()
        };
        let domain = lensing_core::domain::Domain::example();
        let encoder = Encoder::build(&points, &domain, &cfg);
        let trained = features::assemble(&projected, k, &encoder, &points);

        // Contract as promotion would freeze it.
        let mut columns: Vec<ColumnDesc> = (0..k)
            .map(|c| ColumnDesc { name: format!("pca_{c}"), kind: ColumnKind::Pca { component: c } })
            .collect();
        columns.extend(encoder.columns());
        let contract = Contract {
            contract_version: CONTRACT_VERSION,
            n_cols: k + encoder.width(),
            columns,
            pca: PcaInfo {
                dims: k,
                mean: mean_f32,
                components_shape: [k, d],
                explained_variance_ratio: fitted.explained_variance_ratio.clone(),
            },
            target: TargetInfo {
                field: "metadata.price".into(),
                transform: TargetTransform::Log1p,
                task: Default::default(),
                classes: None,
            },
            feature_config: cfg,
            input_fields: InputFields {
                required_numeric: vec!["bedrooms".into()],
                required_categorical: vec!["propertyType".into()],
                embedding_dim: d,
            },
            numerics_collection: None,
            imputation: None,
        };

        // Inference path: reconstruct from the frozen contract.
        let fz = Featurizer::from_contract(contract, &comp_flat).unwrap();
        let inferred = fz.featurize(&points).unwrap();
        assert_eq!(trained, inferred, "inference features must equal training features");

        // Unknown category warns and lands in __other__, never errors.
        let odd = vec![point(99, points[0].vector.clone(), "castle", 2.0)];
        assert!(fz.featurize(&odd).is_ok());
        let warnings = fz.warnings(&odd);
        assert!(warnings.iter().any(|w| w.contains("castle")), "warnings: {warnings:?}");

        // Wrong embedding dim is a hard error.
        let bad = vec![point(100, vec![0.0; d - 1], "house", 1.0)];
        assert!(fz.featurize(&bad).is_err());
    }
}
