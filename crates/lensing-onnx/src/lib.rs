//! A minimal, dependency-free ONNX model writer.
//!
//! ONNX models are protobuf messages. Rather than pull in `prost` + a vendored
//! `onnx.proto` + `protoc` (network + build-tooling the rest of this repo
//! avoids), this module hand-encodes the small subset of the ONNX schema the
//! burn predictors need: a single-input, single-output `GraphProto` of `Gemm`,
//! convolution and elementwise activation nodes, with weights carried as
//! float `initializer` tensors.
//!
//! The output is a standard `model.onnx` (IR version 7, opset 13) that loads
//! in onnxruntime / onnxruntime-web unchanged. Inputs and outputs are
//! `float32` tensors; the batch dimension is symbolic (`N`).
//!
//! Op coverage is intentionally narrow — `Gemm`, `Conv`, `MaxPool`,
//! `AveragePool`, `Flatten`, `Concat`, `Reshape`, and the activation set burn
//! exposes (relu/gelu/silu/mish/tanh/leaky_relu/selu/elu, with the ones ONNX
//! lacks a single op for decomposed into primitives). Everything is opset ≤13
//! so it runs on conservative runtimes.

// ---------- protobuf wire encoding ----------

/// Append a base-128 varint (protobuf wire type 0).
fn varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            out.push(byte | 0x80);
        } else {
            out.push(byte);
            break;
        }
    }
}

/// Field tag = (field_number << 3) | wire_type.
fn tag(out: &mut Vec<u8>, field: u32, wire: u32) {
    varint(out, ((field << 3) | wire) as u64);
}

/// A length-delimited field (wire type 2): string, bytes, or embedded message.
fn len_delimited(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    tag(out, field, 2);
    varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

/// An int64/int32/enum/bool field (wire type 0).
fn varint_field(out: &mut Vec<u8>, field: u32, v: i64) {
    tag(out, field, 0);
    varint(out, v as u64);
}

/// A 32-bit float field (wire type 5).
fn float_field(out: &mut Vec<u8>, field: u32, v: f32) {
    tag(out, field, 5);
    out.extend_from_slice(&v.to_le_bytes());
}

fn string_field(out: &mut Vec<u8>, field: u32, s: &str) {
    len_delimited(out, field, s.as_bytes());
}

// ---------- ONNX model definitions ----------

/// ONNX `TensorProto.DataType`: FLOAT=1, INT64=7.
const DT_FLOAT: i64 = 1;
const DT_INT64: i64 = 7;

/// One attribute on a node.
#[derive(Clone)]
pub enum Attr {
    Int(i64),
    Float(f32),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
}

impl Attr {
    /// `AttributeProto.AttributeType`: FLOAT=1, INT=2, FLOATS=6, INTS=7.
    fn type_code(&self) -> i64 {
        match self {
            Attr::Float(_) => 1,
            Attr::Int(_) => 2,
            Attr::Floats(_) => 6,
            Attr::Ints(_) => 7,
        }
    }

    fn encode(&self, name: &str) -> Vec<u8> {
        let mut m = Vec::new();
        string_field(&mut m, 1, name); // AttributeProto.name
        match self {
            Attr::Float(f) => float_field(&mut m, 2, *f), // .f
            Attr::Int(i) => varint_field(&mut m, 3, *i),  // .i
            Attr::Floats(fs) => {
                for f in fs {
                    float_field(&mut m, 7, *f); // .floats
                }
            }
            Attr::Ints(is) => {
                for i in is {
                    varint_field(&mut m, 8, *i); // .ints
                }
            }
        }
        varint_field(&mut m, 20, self.type_code()); // .type
        m
    }
}

struct Node {
    op_type: String,
    name: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attrs: Vec<(String, Attr)>,
}

impl Node {
    fn encode(&self) -> Vec<u8> {
        let mut m = Vec::new();
        for i in &self.inputs {
            string_field(&mut m, 1, i); // NodeProto.input
        }
        for o in &self.outputs {
            string_field(&mut m, 2, o); // .output
        }
        string_field(&mut m, 3, &self.name); // .name
        string_field(&mut m, 4, &self.op_type); // .op_type
        for (n, a) in &self.attrs {
            len_delimited(&mut m, 5, &a.encode(n)); // .attribute
        }
        m
    }
}

enum InitData {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

struct Initializer {
    name: String,
    dims: Vec<i64>,
    data: InitData,
}

impl Initializer {
    fn encode(&self) -> Vec<u8> {
        let mut m = Vec::new();
        for d in &self.dims {
            varint_field(&mut m, 1, *d); // TensorProto.dims
        }
        // raw_data: little-endian, row-major. Simpler than packed typed
        // fields and equally standard.
        let (data_type, raw) = match &self.data {
            InitData::F32(v) => {
                let mut raw = Vec::with_capacity(v.len() * 4);
                for x in v {
                    raw.extend_from_slice(&x.to_le_bytes());
                }
                (DT_FLOAT, raw)
            }
            InitData::I64(v) => {
                let mut raw = Vec::with_capacity(v.len() * 8);
                for x in v {
                    raw.extend_from_slice(&x.to_le_bytes());
                }
                (DT_INT64, raw)
            }
        };
        varint_field(&mut m, 2, data_type); // .data_type
        string_field(&mut m, 8, &self.name); // .name
        len_delimited(&mut m, 9, &raw); // .raw_data
        m
    }
}

/// One dimension of a tensor's declared shape: a fixed size or a symbolic name.
#[derive(Clone)]
pub enum Dim {
    Value(i64),
    Param(String),
}

fn encode_value_info(name: &str, dims: &[Dim]) -> Vec<u8> {
    // ValueInfoProto { name, type: TypeProto { tensor_type: { elem_type, shape } } }
    let mut shape = Vec::new();
    for d in dims {
        let mut dim = Vec::new();
        match d {
            Dim::Value(v) => varint_field(&mut dim, 1, *v), // dim_value
            Dim::Param(s) => string_field(&mut dim, 2, s),  // dim_param
        }
        len_delimited(&mut shape, 1, &dim); // TensorShapeProto.dim
    }
    let mut tensor = Vec::new();
    varint_field(&mut tensor, 1, DT_FLOAT); // Tensor.elem_type
    len_delimited(&mut tensor, 2, &shape); // Tensor.shape

    let mut type_proto = Vec::new();
    len_delimited(&mut type_proto, 1, &tensor); // TypeProto.tensor_type

    let mut vi = Vec::new();
    string_field(&mut vi, 1, name); // ValueInfoProto.name
    len_delimited(&mut vi, 2, &type_proto); // .type
    vi
}

/// Builder for a single-input / single-output ONNX graph. Nodes are appended
/// in execution order; each helper returns the name of its (single) output so
/// callers can thread a chain.
pub struct GraphBuilder {
    nodes: Vec<Node>,
    initializers: Vec<Initializer>,
    counter: usize,
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphBuilder {
    pub fn new() -> Self {
        GraphBuilder { nodes: Vec::new(), initializers: Vec::new(), counter: 0 }
    }

    fn fresh(&mut self, hint: &str) -> String {
        self.counter += 1;
        format!("{hint}_{}", self.counter)
    }

    /// Register a float32 weight tensor (an `initializer`) under `name`.
    /// `dims` is its shape; `data` is row-major float32.
    pub fn add_weight(&mut self, name: &str, dims: Vec<i64>, data: Vec<f32>) {
        self.initializers.push(Initializer {
            name: name.to_string(),
            dims,
            data: InitData::F32(data),
        });
    }

    /// Register an int64 tensor initializer (shape inputs for Reshape, index
    /// inputs for Slice). Returns the name for convenience.
    pub fn add_int_tensor(&mut self, dims: Vec<i64>, data: Vec<i64>) -> String {
        let name = self.fresh("idx");
        self.initializers.push(Initializer {
            name: name.clone(),
            dims,
            data: InitData::I64(data),
        });
        name
    }

    /// A fresh scalar float constant, returned by initializer name. Used for
    /// activation decompositions (e.g. the 0.5 in Gelu).
    pub fn scalar(&mut self, value: f32) -> String {
        let name = self.fresh("const");
        self.add_weight(&name, vec![], vec![value]);
        name
    }

    /// Append a node with one output; returns the output's fresh name.
    pub fn op(&mut self, op_type: &str, inputs: &[&str], attrs: Vec<(String, Attr)>) -> String {
        let out = self.fresh(&op_type.to_lowercase());
        let node = Node {
            op_type: op_type.to_string(),
            name: out.clone(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: vec![out.clone()],
            attrs,
        };
        self.nodes.push(node);
        out
    }

    /// `Y = X·W + b`, with `W` stored as `[in, out]` (burn's `Linear` layout,
    /// so `transB = 0`). `bias` may be empty (no bias term).
    pub fn gemm(
        &mut self,
        x: &str,
        weight_name: &str,
        weight: Vec<f32>,
        w_in: usize,
        w_out: usize,
        bias_name: &str,
        bias: Option<Vec<f32>>,
    ) -> String {
        self.add_weight(weight_name, vec![w_in as i64, w_out as i64], weight);
        let mut inputs = vec![x.to_string(), weight_name.to_string()];
        if let Some(b) = bias {
            self.add_weight(bias_name, vec![w_out as i64], b);
            inputs.push(bias_name.to_string());
        }
        let refs: Vec<&str> = inputs.iter().map(|s| s.as_str()).collect();
        self.op("Gemm", &refs, vec![])
    }

    /// Append the activation `kind` applied to `x`; returns the output name.
    /// ONNX-native ops are emitted directly; the rest are decomposed into
    /// opset-13 primitives (Erf/Sigmoid/Softplus/Mul/Add).
    pub fn activation(&mut self, x: &str, kind: Activation) -> String {
        match kind {
            Activation::Relu => self.op("Relu", &[x], vec![]),
            Activation::Tanh => self.op("Tanh", &[x], vec![]),
            Activation::LeakyRelu => {
                self.op("LeakyRelu", &[x], vec![("alpha".into(), Attr::Float(0.01))])
            }
            Activation::Elu => self.op("Elu", &[x], vec![("alpha".into(), Attr::Float(1.0))]),
            Activation::Selu => self.op(
                "Selu",
                &[x],
                // burn's SELU constants (the standard Klambauer et al. values).
                vec![
                    ("alpha".into(), Attr::Float(1.673_263_2)),
                    ("gamma".into(), Attr::Float(1.050_701)),
                ],
            ),
            // Exact (erf-based) GELU: 0.5·x·(1 + erf(x/√2)).
            Activation::Gelu => {
                let inv_sqrt2 = self.scalar(std::f32::consts::FRAC_1_SQRT_2);
                let one = self.scalar(1.0);
                let half = self.scalar(0.5);
                let scaled = self.op("Mul", &[x, &inv_sqrt2], vec![]);
                let erf = self.op("Erf", &[&scaled], vec![]);
                let plus1 = self.op("Add", &[&erf, &one], vec![]);
                let xprod = self.op("Mul", &[x, &plus1], vec![]);
                self.op("Mul", &[&xprod, &half], vec![])
            }
            // SiLU / swish: x·sigmoid(x).
            Activation::Silu => {
                let s = self.op("Sigmoid", &[x], vec![]);
                self.op("Mul", &[x, &s], vec![])
            }
            // Mish: x·tanh(softplus(x)).
            Activation::Mish => {
                let sp = self.op("Softplus", &[x], vec![]);
                let t = self.op("Tanh", &[&sp], vec![]);
                self.op("Mul", &[x, &t], vec![])
            }
        }
    }

    /// Serialize the full `ModelProto`. `input`/`output` name the graph
    /// boundary tensors; their shapes are declared with a symbolic batch dim.
    pub fn build(
        self,
        input: &str,
        input_shape: &[Dim],
        output: &str,
        output_shape: &[Dim],
        producer: &str,
    ) -> Vec<u8> {
        // GraphProto
        let mut graph = Vec::new();
        for n in &self.nodes {
            len_delimited(&mut graph, 1, &n.encode()); // node
        }
        string_field(&mut graph, 2, "lensing-export"); // name
        for init in &self.initializers {
            len_delimited(&mut graph, 5, &init.encode()); // initializer
        }
        len_delimited(&mut graph, 11, &encode_value_info(input, input_shape)); // input
        len_delimited(&mut graph, 12, &encode_value_info(output, output_shape)); // output

        // OperatorSetIdProto { domain: "", version: 13 }
        let mut opset = Vec::new();
        varint_field(&mut opset, 2, 13);

        // ModelProto
        let mut model = Vec::new();
        varint_field(&mut model, 1, 7); // ir_version (opset 13 → IR 7)
        string_field(&mut model, 2, producer); // producer_name
        len_delimited(&mut model, 7, &graph); // graph
        len_delimited(&mut model, 8, &opset); // opset_import
        model
    }
}

/// The activation kinds burn's MLP/CNN expose, named so the predictor crates
/// can map their own enum without this crate depending on burn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Relu,
    Gelu,
    Silu,
    Mish,
    Tanh,
    LeakyRelu,
    Selu,
    Elu,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2-layer MLP serializes to non-trivial bytes with the right structure.
    #[test]
    fn mlp_graph_serializes() {
        let mut g = GraphBuilder::new();
        // 4 -> 3 -> 1
        let h = g.gemm(
            "input",
            "w0",
            vec![0.1; 4 * 3],
            4,
            3,
            "b0",
            Some(vec![0.0; 3]),
        );
        let a = g.activation(&h, Activation::Relu);
        let out = g.gemm(&a, "w1", vec![0.2; 3 * 1], 3, 1, "b1", Some(vec![0.0; 1]));
        let bytes = g.build(
            "input",
            &[Dim::Param("N".into()), Dim::Value(4)],
            &out,
            &[Dim::Param("N".into()), Dim::Value(1)],
            "lensing-test",
        );
        // ONNX models start with the ir_version field tag (field 1, varint).
        assert_eq!(bytes[0], 0x08, "first byte is ir_version tag");
        assert!(bytes.len() > 100, "serialized model has substance");
    }

    /// Varint encoding matches the protobuf spec on boundary values.
    #[test]
    fn varint_boundaries() {
        let mut b = Vec::new();
        varint(&mut b, 0);
        assert_eq!(b, [0x00]);
        b.clear();
        varint(&mut b, 127);
        assert_eq!(b, [0x7f]);
        b.clear();
        varint(&mut b, 128);
        assert_eq!(b, [0x80, 0x01]);
        b.clear();
        varint(&mut b, 300);
        assert_eq!(b, [0xac, 0x02]);
    }

    /// Decomposed activations (gelu) add several nodes; native ones add one.
    #[test]
    fn activation_node_counts() {
        let mut g = GraphBuilder::new();
        let _ = g.activation("x", Activation::Relu);
        assert_eq!(g.nodes.len(), 1);

        let mut g2 = GraphBuilder::new();
        let _ = g2.activation("x", Activation::Gelu);
        assert_eq!(g2.nodes.len(), 5, "Mul, Erf, Add, Mul, Mul");
    }
}
