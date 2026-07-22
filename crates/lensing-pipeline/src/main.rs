use anyhow::Result;
use clap::{Parser, Subcommand};

use lensing_pipeline::{build_dataset, BuildConfig};

#[derive(Parser)]
#[command(name = "lensing-pipeline", about = "Build dataset artifacts from a Qdrant corpus")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Fetch points, fit PCA, encode features and write a dataset directory.
    Build {
        #[arg(long, default_value = "http://localhost:6333")]
        qdrant_url: String,
        /// Qdrant collection (default: domain.toml `corpus.collection`).
        #[arg(long)]
        collection: Option<String>,
        #[arg(long, default_value = "data/datasets")]
        out: std::path::PathBuf,
        #[arg(long, default_value_t = 32)]
        pca_dims: usize,
        #[arg(long, default_value_t = 0.2)]
        test_ratio: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Disable the log1p target transform.
        #[arg(long)]
        no_log_target: bool,
        /// Per-field feature toggle, repeatable: NAME[=on|off] (bare NAME
        /// enables). NAME is a domain field or toggle-group name; unset
        /// fields keep their domain defaults.
        #[arg(long = "field", value_name = "NAME[=on|off]")]
        field: Vec<String>,
        /// Per-categorical vocabulary cap, repeatable: NAME=N
        /// (top-N one-hots + an __other__ bucket).
        #[arg(long = "vocab-top-n", value_name = "NAME=N")]
        vocab_top_n: Vec<String>,
        /// Enable the reconciled numeric features (the domain's
        /// reconcile-flagged fields), joined from the companion collection
        /// by point id.
        #[arg(long)]
        raw_numerics: bool,
        /// With --raw-numerics: backfill missing area-like fields from unit
        /// mentions (e.g. "… m²") in the entry text.
        #[arg(long)]
        area_content_backfill: bool,
        /// Companion collection for the raw-numerics join ("" disables;
        /// default: domain.toml `currency.reconcile_collection`).
        #[arg(long)]
        numerics_collection: Option<String>,
        /// Disable the nonpositive-target quality filter.
        #[arg(long, alias = "no-filter-nonpositive-price")]
        no_filter_nonpositive_target: bool,
        /// Enable the robust target-outlier quality filter.
        #[arg(long, alias = "filter-price-outliers")]
        filter_target_outliers: bool,
        /// MAD z-score threshold for the target-outlier filter.
        #[arg(long, alias = "price-outlier-mad-z", default_value_t = 3.5)]
        target_outlier_mad_z: f64,
        /// Enable the target-range quality filter.
        #[arg(long, alias = "filter-price-range")]
        filter_target_range: bool,
        /// target-range filter floor (in the target's unit). Default: no floor.
        #[arg(long, alias = "price-min", default_value_t = f64::MIN)]
        target_min: f64,
        /// target-range filter ceiling (in the target's unit). Default: no ceiling.
        #[arg(long, alias = "price-max", default_value_t = f64::MAX)]
        target_max: f64,
        /// Enable the missing-critical-fields quality filter.
        #[arg(long)]
        filter_missing_fields: bool,
        /// Currency handling: filter (drop foreign-currency rows, default),
        /// convert (rewrite foreign values into the kept currency @ per-date
        /// rate), or off. No-op for single-currency domains.
        #[arg(long, default_value = "filter")]
        currency_mode: String,
        /// Companion collection for the currency reconcile join ("" disables;
        /// default: domain.toml `currency.reconcile_collection`).
        #[arg(long)]
        currency_reconcile: Option<String>,
        /// Exchange-rate series for convert mode ("" = domain.toml
        /// `currency.rate_source`).
        #[arg(long, default_value = "")]
        currency_rate_source: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Build {
            qdrant_url,
            collection,
            out,
            pca_dims,
            test_ratio,
            seed,
            no_log_target,
            field,
            vocab_top_n,
            raw_numerics,
            area_content_backfill,
            numerics_collection,
            no_filter_nonpositive_target,
            filter_target_outliers,
            target_outlier_mad_z,
            filter_target_range,
            target_min,
            target_max,
            filter_missing_fields,
            currency_mode,
            currency_reconcile,
            currency_rate_source,
        } => {
            let mode = match currency_mode.as_str() {
                "off" => lensing_core::CurrencyMode::Off,
                "filter" => lensing_core::CurrencyMode::Filter,
                "convert" => lensing_core::CurrencyMode::Convert,
                other => anyhow::bail!("unknown currency mode {other:?} (off|filter|convert)"),
            };
            // domain.toml at the CWD (repo root), else the embedded default.
            // Collection flags fall back to the domain's corpus config.
            let domain = lensing_core::domain::Domain::load_or_default(std::path::Path::new("."))?;
            let collection =
                collection.unwrap_or_else(|| domain.corpus.collection.clone());
            let companion_default =
                || domain.currency.as_ref().and_then(|c| c.reconcile_collection.clone());
            let numerics_collection = match numerics_collection {
                Some(c) => (!c.is_empty()).then_some(c),
                None => companion_default(),
            };
            let currency_reconcile = match currency_reconcile {
                Some(c) => (!c.is_empty()).then_some(c),
                None => companion_default(),
            };
            // Seed the generic field map from the domain defaults, apply the
            // --field / --vocab-top-n overrides, then normalize again so the
            // maps (and the legacy mirror flags) are fully explicit.
            let mut features = lensing_core::FeatureConfig {
                pca_dims,
                raw_numerics,
                area_content_backfill,
                // API-driven options; the CLI keeps defaults.
                ..Default::default()
            };
            domain.normalize_config(&mut features);
            for spec in &field {
                let (name, on) = match spec.split_once('=') {
                    None => (spec.as_str(), true),
                    Some((n, "on")) | Some((n, "true")) | Some((n, "1")) => (n, true),
                    Some((n, "off")) | Some((n, "false")) | Some((n, "0")) => (n, false),
                    Some((_, v)) => anyhow::bail!("--field {spec:?}: unknown value {v:?} (on|off)"),
                };
                features.fields.insert(name.to_string(), on);
            }
            for spec in &vocab_top_n {
                let (name, n) = spec
                    .split_once('=')
                    .and_then(|(n, v)| Some((n, v.parse::<usize>().ok()?)))
                    .ok_or_else(|| anyhow::anyhow!("--vocab-top-n {spec:?}: expected NAME=N"))?;
                features.vocab_top_n.insert(name.to_string(), n);
            }
            domain.normalize_config(&mut features);
            let cfg = BuildConfig {
                domain,
                qdrant_url,
                collection,
                out_root: out,
                test_ratio,
                seed,
                log_target: !no_log_target,
                features,
                quality: lensing_core::QualityFilterConfig {
                    nonpositive_target: !no_filter_nonpositive_target,
                    target_outlier: filter_target_outliers,
                    target_outlier_mad_z,
                    target_range: filter_target_range,
                    target_min,
                    target_max,
                    missing_fields: filter_missing_fields,
                    // The extended rules (target-range, capped-numeric,
                    // duplicates, short content) are API-driven; CLI keeps
                    // defaults.
                    ..Default::default()
                },
                currency: lensing_core::CurrencyConfig {
                    mode,
                    reconcile_collection: currency_reconcile,
                    rate_source: currency_rate_source,
                    ..Default::default()
                },
                numerics_collection,
            };
            let manifest = build_dataset(&cfg, &|stage| eprintln!("[build] {stage}"))?;
            println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                "dataset_id": manifest.dataset_id,
                "n_rows": manifest.n_rows,
                "n_cols": manifest.n_cols,
                "n_train": manifest.split.n_train,
                "n_test": manifest.split.n_test,
                "explained_variance_ratio_sum":
                    manifest.pca.explained_variance_ratio.iter().sum::<f32>(),
            }))?);
            Ok(())
        }
    }
}
