//! Shared types and artifact I/O for the price-guesser workspace.
//!
//! A dataset directory contains raw little-endian binaries plus a
//! `manifest.json` describing them. The format is language-neutral and
//! documented in the repository README; this crate is just the Rust reader
//! and writer for it.

pub mod artifact;
pub mod currency;
pub mod domain;
pub mod definition;
pub mod manifest;
pub mod model;
pub mod numerics;
pub mod quality;
pub mod redundancy;
pub mod registry;
pub mod run;
pub mod shuffle;

pub use artifact::{Dataset, InferenceInput};
pub use currency::*;
pub use definition::*;
pub use manifest::*;
pub use model::*;
pub use numerics::*;
pub use quality::*;
pub use redundancy::*;
pub use run::*;
