//! The sparse-autoencoder engine, exposed as a library so it can be trained on
//! any standardized `[n_rows, d]` block — not just a dataset's embedding block.
//!
//! The `lensing-sae analyze` binary (see `main.rs`) trains it on a DATASET's PCA
//! columns. A predictor crate (e.g. `predictor-burn-mlp`'s `model-sae`
//! subcommand) trains the same engine on a promoted model's captured hidden-layer
//! activations, to interpret what THAT trained model represents internally. Both
//! share this identical training/encode/cache code; only the input block and the
//! surrounding analysis differ.

pub mod autointerp;
pub mod llm;
pub mod sae;
