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
        /// Disable the bedrooms numeric feature.
        #[arg(long)]
        no_bedrooms: bool,
        /// Disable propertyType one-hots.
        #[arg(long)]
        no_property_type: bool,
        /// Neighborhood one-hot vocabulary size (0 disables).
        #[arg(long, default_value_t = 40)]
        neighborhood_top_n: usize,
        /// Enable city one-hots (messy field, off by default).
        #[arg(long)]
        city: bool,
        /// Enable province one-hots (messy field, off by default).
        #[arg(long)]
        province: bool,
        /// Enable cluster one-hots (potentially leaky, off by default).
        #[arg(long)]
        cluster: bool,
        /// Enable raw-metadata numeric features (areas, baths, rooms),
        /// reconciled from the companion collection by point id.
        #[arg(long)]
        raw_numerics: bool,
        /// With --raw-numerics: backfill missing areas from "… m²" mentions
        /// in the listing text.
        #[arg(long)]
        area_content_backfill: bool,
        /// Companion collection for the raw-numerics join ("" disables;
        /// default: domain.toml `currency.reconcile_collection`).
        #[arg(long)]
        numerics_collection: Option<String>,
        /// Disable the nonpositive-price quality filter.
        #[arg(long)]
        no_filter_nonpositive_price: bool,
        /// Enable the robust price-outlier quality filter.
        #[arg(long)]
        filter_price_outliers: bool,
        /// MAD z-score threshold for the price-outlier filter.
        #[arg(long, default_value_t = 3.5)]
        price_outlier_mad_z: f64,
        /// Enable the price-range quality filter.
        #[arg(long)]
        filter_price_range: bool,
        /// Price-range filter floor (USD).
        #[arg(long, default_value_t = 1000.0)]
        price_min: f64,
        /// Price-range filter ceiling (USD).
        #[arg(long, default_value_t = 50_000_000.0)]
        price_max: f64,
        /// Enable the missing-critical-fields quality filter.
        #[arg(long)]
        filter_missing_fields: bool,
        /// Currency handling: filter (drop non-USD, default), convert
        /// (ARS→USD @ per-date rate), or off.
        #[arg(long, default_value = "filter")]
        currency_mode: String,
        /// Companion collection for the currency reconcile join ("" disables;
        /// default: domain.toml `currency.reconcile_collection`).
        #[arg(long)]
        currency_reconcile: Option<String>,
        /// Exchange-rate series for convert mode: blue or oficial.
        #[arg(long, default_value = "blue")]
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
            no_bedrooms,
            no_property_type,
            neighborhood_top_n,
            city,
            province,
            cluster,
            raw_numerics,
            area_content_backfill,
            numerics_collection,
            no_filter_nonpositive_price,
            filter_price_outliers,
            price_outlier_mad_z,
            filter_price_range,
            price_min,
            price_max,
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
            let cfg = BuildConfig {
                domain,
                qdrant_url,
                collection,
                out_root: out,
                test_ratio,
                seed,
                log_target: !no_log_target,
                features: lensing_core::FeatureConfig {
                    pca_dims,
                    bedrooms: !no_bedrooms,
                    property_type: !no_property_type,
                    neighborhood_top_n,
                    city,
                    province,
                    cluster,
                    raw_numerics,
                    area_content_backfill,
                    // API-driven options; the CLI keeps defaults.
                    ..Default::default()
                },
                quality: lensing_core::QualityFilterConfig {
                    nonpositive_price: !no_filter_nonpositive_price,
                    price_outlier: filter_price_outliers,
                    price_outlier_mad_z,
                    price_range: filter_price_range,
                    price_min,
                    price_max,
                    missing_fields: filter_missing_fields,
                    // The extended rules (price-range, bedrooms, duplicates,
                    // short content) are API-driven; the CLI keeps defaults.
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
