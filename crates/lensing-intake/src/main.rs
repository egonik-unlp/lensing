//! `lensing-intake` — run a config-driven intake pipeline.
//!
//! ```text
//! lensing-intake --config pipeline.toml
//! ```

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "lensing-intake",
    about = "Run an intake pipeline: data origin -> embeddings -> Qdrant + lensing Postgres"
)]
struct Cli {
    /// Path to the pipeline config (TOML).
    #[arg(short, long, default_value = "pipeline.toml")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let text = std::fs::read_to_string(&cli.config)
        .map_err(|e| anyhow::anyhow!("reading config {}: {e}", cli.config.display()))?;
    let config: lensing_intake::PipelineConfig =
        toml::from_str(&text).map_err(|e| anyhow::anyhow!("parsing {}: {e}", cli.config.display()))?;
    lensing_intake::run(config).await
}
