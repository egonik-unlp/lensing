//! Config-driven intake pipeline for lensing instances.
//!
//! Reads a [`PipelineConfig`] (`pipeline.toml`), instantiates the corresponding
//! `lvv` [`Source`] and [`Sink`]s, and runs a [`JobQueue`]: data origin →
//! embeddings → Qdrant (+ the internal lensing Postgres). Destinations are
//! declared per-instance, so nothing is shared across instances by default.

use std::sync::Arc;

use anyhow::Context;
use lvv::db::vector_database::{DatabaseParams, Location};
use lvv::db::{Distance, PostgresSink, QdrantSink, Sink};
use lvv::intake::{
    FileFormat, FileSource, HttpSource, Pagination, PostgresSource, SqlSource, Source,
};
use lvv::jobs::job_queue::JobQueue;
use lvv::jobs::{JobBuilder, Provider};

mod config;
pub use config::{EmbeddingConfig, PipelineConfig, SinkConfig, SourceConfig};

/// Run a pipeline end to end: fetch from the source, embed each batch, and
/// write to every configured sink.
pub async fn run(config: PipelineConfig) -> anyhow::Result<()> {
    let provider = parse_provider(&config.embedding)?;
    let distance = parse_distance(&config.embedding.distance)?;
    let dims = config.embedding.dims;
    let extends = config.embedding.extends;

    if config.sink.is_empty() {
        anyhow::bail!("pipeline has no [[sink]]; nothing to write to");
    }

    let source = build_source(&config.source)?;
    let datasets = source
        .fetch()
        .await
        .context("fetching data from the configured source")?;

    let mut jobs = Vec::with_capacity(datasets.len());
    for dataset in datasets {
        let job = JobBuilder::default()
            .dataset(dataset)
            .provider(provider.clone())
            .dims(dims)
            .extends(extends)
            .distance(distance)
            .collection_name() // derives the name from provider/distance/dataset; must be last
            .build()
            .map_err(|e| anyhow::anyhow!("building job: {e}"))?;
        jobs.push(job);
    }

    let sinks = build_sinks(&config.sink, distance, dims)?;

    let mut queue = JobQueue::from_vec(jobs);
    for sink in sinks {
        queue.with_sink(sink);
    }
    queue.run().await.context("running the intake pipeline")
}

fn parse_provider(embedding: &EmbeddingConfig) -> anyhow::Result<Provider> {
    match embedding.provider.as_str() {
        "ollama" => Ok(Provider::Ollama(embedding.model.clone())),
        "openai" => Ok(Provider::OpenAI(embedding.model.clone())),
        other => anyhow::bail!("unknown embedding provider {other:?} (expected ollama|openai)"),
    }
}

fn parse_distance(distance: &str) -> anyhow::Result<Distance> {
    match distance.to_ascii_lowercase().as_str() {
        "cosine" => Ok(Distance::Cosine),
        "euclid" | "euclidean" => Ok(Distance::Euclid),
        "dot" => Ok(Distance::Dot),
        "manhattan" => Ok(Distance::Manhattan),
        other => anyhow::bail!("unknown distance {other:?} (expected cosine|euclid|dot|manhattan)"),
    }
}

fn build_source(source: &SourceConfig) -> anyhow::Result<Box<dyn Source>> {
    Ok(match source {
        SourceConfig::Postgres {
            url,
            query,
            identifier,
            batch_size,
        } => Box::new(PostgresSource::new(url, query, identifier).with_batch_size(*batch_size)),
        SourceConfig::SqlSqlite {
            path,
            query,
            identifier,
            batch_size,
        } => Box::new(SqlSource::sqlite(path, query, identifier).with_batch_size(*batch_size)),
        SourceConfig::SqlMysql {
            url,
            query,
            identifier,
            batch_size,
        } => Box::new(SqlSource::mysql(url, query, identifier).with_batch_size(*batch_size)),
        SourceConfig::File {
            path,
            identifier,
            format,
            batch_size,
        } => {
            let mut fs = FileSource::new(path.clone(), identifier)
                .with_context(|| format!("opening file source {path}"))?;
            if let Some(fmt) = format {
                fs = fs.with_format(parse_format(fmt)?);
            }
            Box::new(fs.with_batch_size(*batch_size))
        }
        SourceConfig::Http {
            url,
            identifier,
            pointer,
            page_param,
            batch_size,
        } => {
            let mut hs = HttpSource::new(url, identifier);
            if let Some(p) = pointer {
                hs = hs.with_pointer(p);
            }
            if let Some(param) = page_param {
                hs = hs.with_pagination(Pagination::PageParam {
                    param: param.clone(),
                    start: 0,
                });
            }
            Box::new(hs.with_batch_size(*batch_size))
        }
    })
}

fn parse_format(format: &str) -> anyhow::Result<FileFormat> {
    match format.to_ascii_lowercase().as_str() {
        "csv" => Ok(FileFormat::Csv),
        "json" => Ok(FileFormat::Json),
        "jsonl" | "ndjson" => Ok(FileFormat::Jsonl),
        other => anyhow::bail!("unknown file format {other:?} (expected csv|json|jsonl)"),
    }
}

fn build_sinks(
    sinks: &[SinkConfig],
    distance: Distance,
    dims: u64,
) -> anyhow::Result<Vec<Arc<dyn Sink>>> {
    let mut out: Vec<Arc<dyn Sink>> = Vec::with_capacity(sinks.len());
    for sink in sinks {
        match sink {
            SinkConfig::Qdrant { url, api_key } => {
                let location = match api_key {
                    Some(key) => Location::Remote {
                        url: url.clone(),
                        api_key: key.clone(),
                    },
                    None => Location::Local { url: url.clone() },
                };
                // collection/dims here are placeholders: the sink uses the
                // job-derived collection name and the job's dims at write time.
                let params = DatabaseParams::new(location, String::new(), distance, dims as u16);
                out.push(Arc::new(QdrantSink::new(params)));
            }
            SinkConfig::Postgres { url, table } => {
                out.push(Arc::new(
                    PostgresSink::new(url, table).context("configuring Postgres sink")?,
                ));
            }
        }
    }
    Ok(out)
}
