use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use chrono::Utc;
use serde_json::json;

use lensing_core::artifact::{write_f32, write_u32, write_u64};
use lensing_core::{
    ColumnDesc, ColumnKind, FeatureConfig, Manifest, PcaInfo, Source, SplitInfo, TargetInfo,
    TargetTransform,
};

use crate::features::Encoder;
use crate::qdrant::{self, RawPoint};
use crate::{pca, shuffle};

pub struct BuildConfig {
    /// The domain configuration (corpus schema, target, field descriptors).
    pub domain: lensing_core::domain::Domain,
    pub qdrant_url: String,
    pub collection: String,
    pub out_root: PathBuf,
    pub test_ratio: f64,
    pub seed: u64,
    pub log_target: bool,
    pub features: FeatureConfig,
    pub quality: lensing_core::QualityFilterConfig,
    pub currency: lensing_core::CurrencyConfig,
    /// Companion collection for the reconciled-numerics join (only used when
    /// reconcile fields are enabled); `None` relies on fields already inline.
    pub numerics_collection: Option<String>,
}

impl Default for BuildConfig {
    fn default() -> Self {
        let domain = lensing_core::domain::Domain::default();
        Self {
            qdrant_url: domain.corpus.qdrant_url.clone(),
            collection: domain.corpus.collection.clone(),
            domain,
            out_root: PathBuf::from("data/datasets"),
            test_ratio: 0.2,
            seed: 42,
            log_target: true,
            features: FeatureConfig::default(),
            quality: lensing_core::QualityFilterConfig::default(),
            currency: lensing_core::CurrencyConfig::default(),
            numerics_collection: Some("properties".into()),
        }
    }
}

const ITEM_CONTENT_MAX: usize = 300;

/// Build a dataset directory; returns the manifest.
/// `progress` receives coarse stage descriptions for UI display.
pub fn build_dataset(
    cfg: &BuildConfig,
    progress: &(dyn Fn(&str) + Sync),
) -> Result<Manifest> {
    let domain = &cfg.domain;
    // Hard shape checks (unnamed vector, filter matches > 0) fail here with a
    // clear message instead of mid-scroll or after fetching the whole corpus.
    progress("checking collection");
    qdrant::ensure_buildable(&cfg.qdrant_url, &cfg.collection, domain)?;

    progress("counting points");
    let expected = qdrant::count(&cfg.qdrant_url, &cfg.collection, domain)?;
    ensure!(expected > 0, "filter matches no points");

    progress("fetching points");
    let mut points = qdrant::scroll_all(&cfg.qdrant_url, &cfg.collection, domain, &|n| {
        progress(&format!("fetching points {n}/{expected}"));
    })?;
    ensure!(
        points.len() == expected,
        "scroll returned {} points, count API said {expected}",
        points.len()
    );

    // Currency reconciliation (and Convert-mode value rewriting) runs before
    // the quality rules so target filters see values in the kept currency.
    let currency_report =
        crate::currency::apply(&mut points, &cfg.currency, &cfg.qdrant_url, domain, progress)?;

    // Quality filters run before everything else: the split, the PCA fit and
    // the vocabularies must only ever see the rows the model will train on.
    progress("applying quality filters");
    let analysis = crate::quality::evaluate(&points, &cfg.quality, &cfg.currency, domain);
    let quality_report = analysis.report(&cfg.quality);
    let excluded = analysis.excluded(&cfg.quality);
    if !excluded.is_empty() {
        let mut i = 0usize;
        points.retain(|_| {
            let keep = !excluded.contains(&i);
            i += 1;
            keep
        });
        progress(&format!("quality filters excluded {} rows", excluded.len()));
    }
    ensure!(points.len() > 1, "quality filters excluded the whole corpus");

    // Resolve the feature config to its fully-explicit form (generic fields
    // map + frozen coordinate bounds) before anything consumes it.
    let mut feature_config = cfg.features.clone();
    domain.normalize_config(&mut feature_config);
    let has_imputable = domain
        .numeric_fields()
        .any(|f| f.indicator && domain.field_enabled(&feature_config, f));
    ensure!(
        !feature_config.impute_numerics || has_imputable,
        "impute_numerics requires raw_numerics"
    );

    // Reconciled-numeric handling (areas, baths, rooms, coordinates in the
    // original corpus) joins the companion collection by id; after quality so
    // only retained rows fetch.
    let numerics_report = crate::numerics::apply(
        &mut points,
        &feature_config,
        &cfg.numerics_collection,
        &cfg.qdrant_url,
        domain,
        progress,
    )?;

    let d = points[0].vector.len();
    ensure!(
        points.iter().all(|p| p.vector.len() == d),
        "inconsistent vector dimensions"
    );

    let n = points.len();
    let k = feature_config.pca_dims;

    progress("splitting train/test");
    let (train_idx, test_idx) = shuffle::train_test_split(n, cfg.test_ratio, cfg.seed);

    // Numeric imputation: medians fit on the train split only (the fill
    // values must not leak test information), then applied to all rows so
    // the encoder sees no missing numerics. The frozen medians ship in the
    // manifest → contract so predict fills identically.
    let imputation = if feature_config.impute_numerics {
        progress("imputing missing numerics (train-split group medians)");
        let imp = crate::numerics::fit_imputation(&points, &train_idx, domain);
        crate::numerics::impute(&mut points, &imp);
        Some(imp)
    } else {
        None
    };

    // Flat row-major copy of all vectors for PCA.
    let mut vectors = Vec::with_capacity(n * d);
    for p in &points {
        vectors.extend_from_slice(&p.vector);
    }

    progress("fitting PCA (train rows only)");
    let (fitted, spectrum) = pca::fit_with_spectrum(&vectors, d, &train_idx, k)?;

    // Quantize mean + components through f32 (the artifact precision) BEFORE
    // projecting, so inference (which reconstructs the model from the stored
    // f32 artifacts) featurizes bit-identically to training.
    let mean_f32: Vec<f32> = fitted.mean.iter().map(|v| *v as f32).collect();
    let comp_flat: Vec<f32> = fitted
        .components
        .row_iter()
        .flat_map(|r| r.iter().map(|v| *v as f32).collect::<Vec<_>>())
        .collect();
    let model = pca::PcaModel::from_parts(
        &mean_f32,
        &comp_flat,
        k,
        d,
        fitted.explained_variance_ratio.clone(),
    )?;

    progress("projecting embeddings");
    let projected = model.project_all(&vectors, d); // n × k
    drop(vectors);

    progress("encoding metadata features");
    let encoder = Encoder::build(&points, domain, &feature_config);
    let n_cols = k + encoder.width();
    let features = crate::features::assemble(&projected, k, &encoder, &points);

    progress("analyzing feature redundancy");
    let redundancy = crate::redundancy::analyze(&points, &encoder);
    // Persist the cumulative-variance curve so the detail view can show how
    // much variance more components would have captured (truncated for size).
    let cumulative_evr: Vec<f32> = spectrum.cumulative_evr().into_iter().take(256).collect();

    let transform = if cfg.log_target { TargetTransform::Log1p } else { TargetTransform::None };
    let target_field = domain.target.field.as_str();
    let target: Vec<f32> = points
        .iter()
        .map(|p| transform.apply(p.payload.num_of(target_field).unwrap_or(0.0)) as f32)
        .collect();
    let row_ids: Vec<u64> = points.iter().map(|p| p.id).collect();

    let mut columns: Vec<ColumnDesc> = (0..k)
        .map(|c| ColumnDesc {
            name: format!("pca_{c}"),
            kind: ColumnKind::Pca { component: c },
        })
        .collect();
    columns.extend(encoder.columns());
    debug_assert_eq!(columns.len(), n_cols);

    let dataset_id = format!("ds-{}-p{}-s{}", Utc::now().format("%Y%m%d-%H%M%S"), k, cfg.seed);
    let dir = cfg.out_root.join(&dataset_id);
    fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;

    progress("writing artifacts");
    write_f32(&dir.join("features.f32"), &features)?;
    write_f32(&dir.join("target.f32"), &target)?;
    write_u64(&dir.join("row_ids.u64"), &row_ids)?;
    write_u32(&dir.join("train_idx.u32"), &train_idx)?;
    write_u32(&dir.join("test_idx.u32"), &test_idx)?;

    write_f32(&dir.join("pca_components.f32"), &comp_flat)?;

    write_items(&dir, &points, &domain.corpus.content_field)?;

    let target_qualified = domain
        .field(target_field)
        .map(|f| domain.qualified_key(f))
        .unwrap_or_else(|| target_field.to_string());
    let manifest = Manifest {
        dataset_id: dataset_id.clone(),
        name: None,
        created_at: Utc::now().to_rfc3339(),
        source: Source {
            qdrant_url: cfg.qdrant_url.clone(),
            collection: cfg.collection.clone(),
            filter: format!(
                "{}{}",
                domain.corpus.filter.desc,
                crate::currency::filter_desc(&cfg.currency)
            ),
        },
        n_rows: n,
        n_cols,
        columns,
        pca: PcaInfo {
            dims: k,
            mean: mean_f32,
            components_shape: [k, d],
            explained_variance_ratio: model.explained_variance_ratio.clone(),
        },
        target: TargetInfo { field: target_qualified, transform },
        split: SplitInfo {
            test_ratio: cfg.test_ratio,
            seed: cfg.seed,
            n_train: train_idx.len(),
            n_test: test_idx.len(),
        },
        feature_config,
        quality: Some(quality_report),
        cumulative_evr,
        redundancy: Some(redundancy),
        currency: Some(currency_report),
        numerics: numerics_report,
        imputation,
    };
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&manifest)?)?;
    progress("done");
    Ok(manifest)
}

/// `items.json`: row_id → display payload for the prediction inspector.
/// Also written into inference input mini-artifacts, where payload-based
/// predictors (e.g. baseline-median) read their categoricals from it.
/// Keys are the flattened payload field names — exactly the domain's field
/// names (predictors read these), plus the truncated document text.
pub fn write_items(dir: &Path, points: &[RawPoint], content_field: &str) -> Result<()> {
    let mut map = serde_json::Map::with_capacity(points.len());
    for p in points {
        let mut obj = p.payload.fields.clone();
        obj.insert(
            content_field.to_string(),
            json!(truncate_chars(p.payload.content_of(content_field), ITEM_CONTENT_MAX)),
        );
        map.insert(p.id.to_string(), serde_json::Value::Object(obj));
    }
    fs::write(dir.join("items.json"), serde_json::to_vec(&map)?)?;
    Ok(())
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{}…", cut.trim_end())
    }
}
