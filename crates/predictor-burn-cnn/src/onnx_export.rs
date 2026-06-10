//! Export a trained 1D-CNN to a portable `model.onnx`.
//!
//! The graph mirrors `model::Cnn::forward` exactly, on the *assembled feature
//! vector* (so the input contract matches every other family's export):
//!
//!   input[N, n_cols]
//!     → Sub(mean) → Div(std)                 (standardizer, baked in)
//!     → Slice → pca[N, n_pca], meta[N, n_meta]
//!     → Reshape pca → [N, 1, n_pca]
//!     → (Conv → Relu → MaxPool?) ×channels   (MaxPool only while length ≥ 2)
//!     → GlobalAveragePool → Reshape → [N, ch]
//!     → Concat([pooled, meta], axis=1)
//!     → Gemm → Relu (dense head)
//!     → Gemm (output head) → [N, 1]          (transformed-space target)
//!
//! The conv uses symmetric `k/2` padding and stride 1 (same as burn's
//! `PaddingConfig1d::Explicit(k/2, k/2)`); MaxPool is kernel 2 / stride 2
//! (burn's `MaxPool1dConfig::new(2)`). The pooling length schedule is computed
//! statically here from `n_pca` and the kernel so the emitted graph contains
//! exactly the MaxPool nodes the forward pass would run.

use std::path::Path;

use anyhow::{Context, Result};
use lensing_onnx::{Activation, Attr, Dim, GraphBuilder};

use crate::model::{Cnn, Inner};
use crate::scaler::Scaler;

/// Conv1d output length with symmetric `k/2` padding, stride 1, dilation 1.
fn conv_out_len(len: usize, k: usize) -> usize {
    len + 2 * (k / 2) - k + 1
}

/// MaxPool1d output length, kernel 2 / stride 2, no padding.
fn pool_out_len(len: usize) -> usize {
    (len - 2) / 2 + 1
}

/// Build and write `<output_dir>/model.onnx` for a loaded CNN. `n_cols` is the
/// assembled feature width (= `scaler.mean.len()`).
pub fn write_onnx(model: &Cnn<Inner>, scaler: &Scaler, output_dir: &Path) -> Result<()> {
    let n_cols = scaler.mean.len();
    let (convs, dense, output, n_pca) = model.export_parts();
    anyhow::ensure!(n_pca > 0 && n_pca <= n_cols, "invalid pca split {n_pca}/{n_cols}");

    let mut g = GraphBuilder::new();

    // Standardizer.
    g.add_weight("scaler_mean", vec![n_cols as i64], scaler.mean.clone());
    g.add_weight("scaler_std", vec![n_cols as i64], scaler.std.clone());
    let centered = g.op("Sub", &["input", "scaler_mean"], vec![]);
    let scaled = g.op("Div", &[&centered, "scaler_std"], vec![]);

    // Split into the PCA signal and the metadata tail (axis 1).
    let pca_starts = g.add_int_tensor(vec![1], vec![0]);
    let pca_ends = g.add_int_tensor(vec![1], vec![n_pca as i64]);
    let axis1 = g.add_int_tensor(vec![1], vec![1]);
    let pca = g.op("Slice", &[&scaled, &pca_starts, &pca_ends, &axis1], vec![]);
    let meta_starts = g.add_int_tensor(vec![1], vec![n_pca as i64]);
    let meta_ends = g.add_int_tensor(vec![1], vec![n_cols as i64]);
    let meta = g.op("Slice", &[&scaled, &meta_starts, &meta_ends, &axis1], vec![]);

    // Reshape the PCA signal to NCL: [N, 1, n_pca].
    let to_ncl = g.add_int_tensor(vec![3], vec![0, 1, n_pca as i64]);
    let mut signal = g.op("Reshape", &[&pca, &to_ncl], vec![]);

    // Conv → Relu → (MaxPool while length ≥ 2), per channel layer.
    let mut len = n_pca;
    let mut last_ch = 1usize;
    for (i, c) in convs.into_iter().enumerate() {
        let k = c.kernel;
        g.add_weight(&format!("conv{i}_w"), vec![c.out_ch as i64, c.in_ch as i64, k as i64], c.weight);
        let mut inputs = vec![signal.clone(), format!("conv{i}_w")];
        if let Some(b) = c.bias {
            g.add_weight(&format!("conv{i}_b"), vec![c.out_ch as i64], b);
            inputs.push(format!("conv{i}_b"));
        }
        let refs: Vec<&str> = inputs.iter().map(|s| s.as_str()).collect();
        let conv = g.op(
            "Conv",
            &refs,
            vec![
                ("kernel_shape".into(), Attr::Ints(vec![k as i64])),
                ("pads".into(), Attr::Ints(vec![(k / 2) as i64, (k / 2) as i64])),
                ("strides".into(), Attr::Ints(vec![1])),
                ("dilations".into(), Attr::Ints(vec![1])),
                ("group".into(), Attr::Int(1)),
            ],
        );
        let relu = g.activation(&conv, Activation::Relu);
        len = conv_out_len(len, k);
        if len >= 2 {
            signal = g.op(
                "MaxPool",
                &[&relu],
                vec![
                    ("kernel_shape".into(), Attr::Ints(vec![2])),
                    ("strides".into(), Attr::Ints(vec![2])),
                ],
            );
            len = pool_out_len(len);
        } else {
            signal = relu;
        }
        last_ch = c.out_ch;
    }

    // Global average pool over the length axis → [N, ch, 1] → [N, ch].
    let gap = g.op("GlobalAveragePool", &[&signal], vec![]);
    let to_2d = g.add_int_tensor(vec![2], vec![0, last_ch as i64]);
    let pooled = g.op("Reshape", &[&gap, &to_2d], vec![]);

    // Concat pooled conv features with the metadata tail, then the dense head.
    let h = g.op("Concat", &[&pooled, &meta], vec![("axis".into(), Attr::Int(1))]);
    let h = g.gemm(&h, "dense_w", dense.weight, dense.w_in, dense.w_out, "dense_b", dense.bias);
    let h = g.activation(&h, Activation::Relu);
    let out = g.gemm(&h, "out_w", output.weight, output.w_in, output.w_out, "out_b", output.bias);

    let bytes = g.build(
        "input",
        &[Dim::Param("N".into()), Dim::Value(n_cols as i64)],
        &out,
        &[Dim::Param("N".into()), Dim::Value(1)],
        concat!("lensing/predictor-burn-cnn ", env!("CARGO_PKG_VERSION")),
    );
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("create export dir {}", output_dir.display()))?;
    std::fs::write(output_dir.join("model.onnx"), &bytes)
        .with_context(|| format!("write {}/model.onnx", output_dir.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Arch, Cnn};
    use burn::backend::ndarray::NdArrayDevice;

    /// Emit an (untrained) CNN export to a temp dir so the bytes can be fed to
    /// an external `onnx.checker`. Validates that the conv/pool/slice graph
    /// assembles without panicking and is non-trivial; semantic parity is
    /// covered by the end-to-end verification with a real trained model.
    #[test]
    fn cnn_export_assembles() {
        let device = NdArrayDevice::default();
        let n_pca = 16;
        let n_meta = 5;
        let arch = Arch {
            n_pca,
            n_meta,
            channels: vec![4, 8],
            kernel_size: 3,
            dense_hidden: 6,
            dropout: 0.0,
        };
        let cnn = Cnn::<Inner>::new(&arch, &device);
        let n_cols = n_pca + n_meta;
        let scaler = Scaler { mean: vec![0.0; n_cols], std: vec![1.0; n_cols] };
        let dir = std::env::temp_dir().join(format!("lensing-cnn-onnx-{}", std::process::id()));
        write_onnx(&cnn, &scaler, &dir).unwrap();
        let bytes = std::fs::read(dir.join("model.onnx")).unwrap();
        assert_eq!(bytes[0], 0x08, "starts with ir_version tag");
        assert!(bytes.len() > 200);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
