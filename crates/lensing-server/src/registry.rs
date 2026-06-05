//! Re-export of the shared registry types; the parsing moved to pg-core so
//! meta-predictors (predictor-blend) can spawn member predictors with the
//! same schema and args-template substitution the server uses.

pub use pg_core::registry::*;
