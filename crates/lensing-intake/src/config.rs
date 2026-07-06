//! Declarative pipeline configuration (`pipeline.toml`).
//!
//! A pipeline declares one `[embedding]` model, one `[source]` (the data
//! origin) and one or more `[[sink]]`s (the destinations). Everything is
//! per-instance: URLs/ports come from the config, so instances stay
//! port-partitioned and never share a destination by default.

use serde::Deserialize;

/// A full intake pipeline: where data comes from, how it's embedded, and where
/// it goes.
#[derive(Debug, Clone, Deserialize)]
pub struct PipelineConfig {
    pub embedding: EmbeddingConfig,
    pub source: SourceConfig,
    /// Destinations. Written in listed order; put the authoritative store
    /// first (e.g. Postgres before Qdrant) — see the partial-failure semantics
    /// in `lvv`'s `JobQueue::run`.
    #[serde(default)]
    pub sink: Vec<SinkConfig>,
}

/// Embedding backend + vector-space parameters shared by every job.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingConfig {
    /// `ollama` or `openai`.
    pub provider: String,
    /// Embedding model name (e.g. `nomic-embed-text`, `text-embedding-3-small`).
    pub model: String,
    /// Vector dimensionality.
    pub dims: u64,
    /// `cosine` (default), `euclid`, `dot`, or `manhattan`.
    #[serde(default = "default_distance")]
    pub distance: String,
    /// Append to an already-populated target instead of skipping it.
    #[serde(default)]
    pub extends: bool,
}

fn default_distance() -> String {
    "cosine".to_string()
}

/// The data origin. `kind` selects the connector.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum SourceConfig {
    /// PostgreSQL via a `SELECT`.
    Postgres {
        url: String,
        query: String,
        identifier: String,
        #[serde(default)]
        batch_size: usize,
    },
    /// SQLite file (or `:memory:`) via a query.
    SqlSqlite {
        path: String,
        query: String,
        identifier: String,
        #[serde(default)]
        batch_size: usize,
    },
    /// MySQL via a `mysql://…` URL.
    SqlMysql {
        url: String,
        query: String,
        identifier: String,
        #[serde(default)]
        batch_size: usize,
    },
    /// A CSV / JSON / JSONL flat file.
    File {
        path: String,
        identifier: String,
        /// Override auto-detection: `csv`, `json`, or `jsonl`.
        #[serde(default)]
        format: Option<String>,
        #[serde(default)]
        batch_size: usize,
    },
    /// An HTTP/JSON endpoint.
    Http {
        url: String,
        identifier: String,
        /// JSON pointer to the items array (e.g. `/data/items`).
        #[serde(default)]
        pointer: Option<String>,
        /// If set, page via `?<page_param>=N` until a page is empty.
        #[serde(default)]
        page_param: Option<String>,
        #[serde(default)]
        batch_size: usize,
    },
}

/// A destination. `kind` selects the sink.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum SinkConfig {
    /// Qdrant vector store. `url` is the instance's Qdrant endpoint.
    Qdrant {
        url: String,
        /// Set for a remote (cloud) Qdrant; omit for a local one.
        #[serde(default)]
        api_key: Option<String>,
    },
    /// The internal lensing project database (PostgreSQL). Rows land as jsonb
    /// in `table`, upserted idempotently.
    Postgres { url: String, table: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_dual_sink_pipeline() {
        let toml_src = r#"
            [embedding]
            provider = "ollama"
            model = "nomic-embed-text"
            dims = 768

            [source]
            kind = "postgres"
            url = "postgres://user:pw@localhost:5436/app"
            query = "SELECT id, title, body FROM tracks"
            identifier = "tracks"

            [[sink]]
            kind = "postgres"
            url = "postgres://user:pw@localhost:5436/app"
            table = "corpus_items"

            [[sink]]
            kind = "qdrant"
            url = "http://localhost:6335"
        "#;
        let cfg: PipelineConfig = toml::from_str(toml_src).unwrap();
        assert_eq!(cfg.embedding.provider, "ollama");
        assert_eq!(cfg.embedding.distance, "cosine"); // default applied
        assert_eq!(cfg.sink.len(), 2);
        match &cfg.source {
            SourceConfig::Postgres { identifier, .. } => assert_eq!(identifier, "tracks"),
            _ => panic!("expected postgres source"),
        }
        // Postgres sink first (authoritative), Qdrant second.
        assert!(matches!(cfg.sink[0], SinkConfig::Postgres { .. }));
        assert!(matches!(cfg.sink[1], SinkConfig::Qdrant { .. }));
    }

    #[test]
    fn parses_file_source_and_http_source() {
        let f: PipelineConfig = toml::from_str(
            "[embedding]\nprovider=\"openai\"\nmodel=\"text-embedding-3-small\"\ndims=1536\n\
             [source]\nkind=\"file\"\npath=\"corpus.jsonl\"\nidentifier=\"c\"\n",
        )
        .unwrap();
        assert!(matches!(f.source, SourceConfig::File { .. }));

        let h: PipelineConfig = toml::from_str(
            "[embedding]\nprovider=\"ollama\"\nmodel=\"m\"\ndims=8\n\
             [source]\nkind=\"http\"\nurl=\"https://api.example.com/items\"\nidentifier=\"i\"\npage_param=\"page\"\n",
        )
        .unwrap();
        match h.source {
            SourceConfig::Http { page_param, .. } => assert_eq!(page_param.as_deref(), Some("page")),
            _ => panic!("expected http source"),
        }
    }
}
