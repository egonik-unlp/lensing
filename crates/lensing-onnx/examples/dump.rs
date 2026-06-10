//! Emit a small MLP-shaped model to the path in argv[1], for validating the
//! writer against a real ONNX parser:
//!   cargo run -p lensing-onnx --example dump -- /tmp/t.onnx
//!   python -c "import onnx; onnx.checker.check_model(onnx.load('/tmp/t.onnx'))"

use lensing_onnx::{Activation, Dim, GraphBuilder};

fn main() {
    let path = std::env::args().nth(1).expect("usage: dump <out.onnx>");
    let mut g = GraphBuilder::new();
    // 4 -> 3 (gelu) -> 1
    let h = g.gemm("input", "w0", vec![0.1; 4 * 3], 4, 3, "b0", Some(vec![0.05; 3]));
    let a = g.activation(&h, Activation::Gelu);
    let out = g.gemm(&a, "w1", vec![0.2; 3], 3, 1, "b1", Some(vec![0.0; 1]));
    let bytes = g.build(
        "input",
        &[Dim::Param("N".into()), Dim::Value(4)],
        &out,
        &[Dim::Param("N".into()), Dim::Value(1)],
        "lensing-dump",
    );
    std::fs::write(&path, &bytes).unwrap();
    eprintln!("wrote {} bytes to {path}", bytes.len());
}
