//! Export a trained MLP to a portable `model.onnx`.
//!
//! The graph's input is the *assembled feature vector* (the same matrix the
//! server's featurizer produces — see `lensing-pipeline`), so the export's
//! input contract is identical across every predictor family. Everything the
//! native model does to that vector is baked into the graph:
//!
//!   input[N, n_cols]
//!     → Sub(mean) → Div(std)            (the train-split standardizer)
//!     → Gemm·activation · …             (the hidden stack)
//!     → Gemm                            (the output head → [N, 1])
//!     → Min(max_t)                      (optional clamp_output)
//!     → output[N, 1]                    (transformed-space target)
//!
//! The inverse target transform (expm1) and the non-negative clamp live in the
//! export's `featurize.json`, not here — keeping `model.onnx` a uniform
//! "raw predictor in transformed space" for all families.

use std::path::Path;

use anyhow::{Context, Result};
use lensing_onnx::{Activation as OnnxAct, Dim, GraphBuilder};

use crate::model::{Activation, Mlp, Inner};
use crate::scaler::Scaler;

fn map_activation(a: Activation) -> OnnxAct {
    match a {
        Activation::Relu => OnnxAct::Relu,
        Activation::Gelu => OnnxAct::Gelu,
        Activation::Silu => OnnxAct::Silu,
        Activation::Mish => OnnxAct::Mish,
        Activation::Tanh => OnnxAct::Tanh,
        Activation::LeakyRelu => OnnxAct::LeakyRelu,
        Activation::Selu => OnnxAct::Selu,
        Activation::Elu => OnnxAct::Elu,
    }
}

/// Build and write `<output_dir>/model.onnx` for a loaded MLP.
pub fn write_onnx(
    model: &Mlp<Inner>,
    scaler: &Scaler,
    clamp_max_t: Option<f32>,
    output_dir: &Path,
) -> Result<()> {
    let n_cols = scaler.mean.len();
    let mut g = GraphBuilder::new();

    // Standardizer: (x - mean) / std, per column (broadcast over the batch).
    g.add_weight("scaler_mean", vec![n_cols as i64], scaler.mean.clone());
    g.add_weight("scaler_std", vec![n_cols as i64], scaler.std.clone());
    let centered = g.op("Sub", &["input", "scaler_mean"], vec![]);
    let mut cur = g.op("Div", &[&centered, "scaler_std"], vec![]);

    // Gemm·activation chain. The activation follows every layer except the
    // output head (the last DenseLayer).
    let layers = model.dense_layers();
    let act = map_activation(model.activation_kind());
    let last = layers.len() - 1;
    for (i, layer) in layers.into_iter().enumerate() {
        cur = g.gemm(
            &cur,
            &format!("w{i}"),
            layer.weight,
            layer.w_in,
            layer.w_out,
            &format!("b{i}"),
            layer.bias,
        );
        if i != last {
            cur = g.activation(&cur, act);
        }
    }

    // Optional transformed-space output clamp (clamp_output hyperparam).
    if let Some(max_t) = clamp_max_t {
        let cap = g.scalar(max_t);
        cur = g.op("Min", &[&cur, &cap], vec![]);
    }

    let bytes = g.build(
        "input",
        &[Dim::Param("N".into()), Dim::Value(n_cols as i64)],
        &cur,
        &[Dim::Param("N".into()), Dim::Value(1)],
        concat!("lensing/predictor-burn-mlp ", env!("CARGO_PKG_VERSION")),
    );
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("create export dir {}", output_dir.display()))?;
    std::fs::write(output_dir.join("model.onnx"), &bytes)
        .with_context(|| format!("write {}/model.onnx", output_dir.display()))?;
    Ok(())
}
