//! Model definition and hand-written training loop (burn 0.21, ndarray CPU).
//! Loss values are MSE in transformed (log) target space.

use std::path::Path;

use anyhow::Result;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::Autodiff;
use burn::module::{AutodiffModule, Module};
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::CompactRecorder;
use burn::tensor::backend::Backend;
use burn::tensor::{activation, Tensor, TensorData};

use lensing_core::shuffle::SplitMix64;

pub type Inner = NdArray<f32>;
pub type Auto = Autodiff<Inner>;

/// Hidden-layer activation, selectable via the `activation` hyperparam.
/// Applied functionally in `forward` (stateless, nothing to checkpoint);
/// stored as a skipped module field so the record stays parameter-only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    #[default]
    Relu,
    Gelu,
    Silu,
    Mish,
    Tanh,
    LeakyRelu,
    Selu,
    Elu,
}

impl Activation {
    pub fn name(self) -> &'static str {
        match self {
            Activation::Relu => "ReLU",
            Activation::Gelu => "GELU",
            Activation::Silu => "SiLU",
            Activation::Mish => "Mish",
            Activation::Tanh => "Tanh",
            Activation::LeakyRelu => "LeakyReLU",
            Activation::Selu => "SELU",
            Activation::Elu => "ELU",
        }
    }

    fn apply<B: Backend>(self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            Activation::Relu => activation::relu(x),
            Activation::Gelu => activation::gelu(x),
            Activation::Silu => activation::silu(x),
            Activation::Mish => activation::mish(x),
            Activation::Tanh => activation::tanh(x),
            Activation::LeakyRelu => activation::leaky_relu(x, 0.01),
            Activation::Selu => activation::selu(x),
            Activation::Elu => activation::elu(x, 1.0),
        }
    }
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    layers: Vec<Linear<B>>,
    output: Linear<B>,
    #[module(skip)]
    activation: Activation,
    dropout: Dropout,
}

impl<B: Backend> Mlp<B> {
    pub fn new(
        n_in: usize,
        hidden: &[usize],
        dropout: f64,
        activation: Activation,
        device: &B::Device,
    ) -> Self {
        let mut layers = Vec::with_capacity(hidden.len());
        let mut prev = n_in;
        for &h in hidden {
            layers.push(LinearConfig::new(prev, h).init(device));
            prev = h;
        }
        Mlp {
            layers,
            output: LinearConfig::new(prev, 1).init(device),
            activation,
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = x;
        for layer in &self.layers {
            x = self.dropout.forward(self.activation.apply(layer.forward(x)));
        }
        self.output.forward(x)
    }
}

pub struct TrainInputs<'a> {
    pub train_x: &'a [f32],
    pub train_y: &'a [f32],
    /// Monitored split for the per-epoch val_loss: the early-stop validation
    /// carve-out when `patience > 0`, else the dataset's test split.
    pub val_x: &'a [f32],
    pub val_y: &'a [f32],
    pub n_cols: usize,
}

pub struct TrainConfig {
    pub epochs: usize,
    pub lr: f64,
    pub batch_size: usize,
    pub hidden: Vec<usize>,
    pub dropout: f64,
    pub activation: Activation,
    pub seed: u64,
    /// Save a loadable checkpoint every N epochs (0 = only at the end), so a
    /// killed run can still be promoted from its last saved params.
    pub checkpoint_every: usize,
    /// Early stopping: break once val_loss hasn't improved in this many
    /// epochs and return the best-epoch weights instead of the last ones
    /// (0 = off, train the full budget and return the final weights).
    pub patience: usize,
}

fn batch_tensor<B: Backend>(
    x: &[f32],
    rows: &[usize],
    n_cols: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut flat = Vec::with_capacity(rows.len() * n_cols);
    for &r in rows {
        flat.extend_from_slice(&x[r * n_cols..(r + 1) * n_cols]);
    }
    Tensor::from_data(TensorData::new(flat, [rows.len(), n_cols]), device)
}

fn target_tensor<B: Backend>(y: &[f32], rows: &[usize], device: &B::Device) -> Tensor<B, 2> {
    let flat: Vec<f32> = rows.iter().map(|&r| y[r]).collect();
    Tensor::from_data(TensorData::new(flat, [rows.len(), 1]), device)
}

pub fn run_training(
    inputs: TrainInputs<'_>,
    cfg: TrainConfig,
    run_dir: &Path,
    emit: &dyn Fn(serde_json::Value),
) -> Result<Mlp<Auto>> {
    let device = NdArrayDevice::default();
    Auto::seed(&device, cfg.seed);

    let n_train = inputs.train_y.len();
    let mut model: Mlp<Auto> =
        Mlp::new(inputs.n_cols, &cfg.hidden, cfg.dropout, cfg.activation, &device);
    let mut optimizer = AdamConfig::new().init();
    let loss_fn = MseLoss::new();

    let mut order: Vec<usize> = (0..n_train).collect();
    let mut rng = SplitMix64(cfg.seed ^ 0xA5A5_A5A5);

    // Best-checkpoint tracking (patience > 0): (val_loss, epoch, weights) of
    // the best epoch seen so far; restored on every exit path — patience
    // exhausted, STOP file, or the full budget — so the run never keeps
    // weights from past the validation optimum.
    let mut best: Option<(f64, usize, Mlp<Auto>)> = None;

    for epoch in 1..=cfg.epochs {
        // Graceful stop (contract v2): the server drops a STOP file into the
        // run dir; finish early and let the caller run the normal
        // end-of-training path (eval + metrics + final checkpoint).
        if run_dir.join("STOP").exists() {
            emit(serde_json::json!({"event":"stopping"}));
            if cfg.patience == 0 {
                emit(serde_json::json!({"event":"log","msg":format!(
                    "stop requested; evaluating with params as of epoch {}", epoch - 1)}));
            } else {
                // The best-epoch restore below logs which weights are used.
                emit(serde_json::json!({"event":"log","msg":"stop requested"}));
            }
            break;
        }

        // Fisher-Yates reshuffle each epoch.
        for i in (1..n_train).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }

        let mut loss_sum = 0.0f64;
        for batch in order.chunks(cfg.batch_size) {
            let x = batch_tensor::<Auto>(inputs.train_x, batch, inputs.n_cols, &device);
            let y = target_tensor::<Auto>(inputs.train_y, batch, &device);
            let pred = model.forward(x);
            let loss = loss_fn.forward(pred, y, Reduction::Mean);
            let loss_value = loss.clone().into_data().to_vec::<f32>().unwrap()[0] as f64;
            loss_sum += loss_value * batch.len() as f64;

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(cfg.lr, model, grads);
        }
        let train_loss = loss_sum / n_train as f64;

        // Validation on the inner backend: no autodiff, dropout inactive.
        let val_model = model.valid();
        let val_pred = predict_inner(&val_model, inputs.val_x, inputs.n_cols, cfg.batch_size);
        let val_loss = val_pred
            .iter()
            .zip(inputs.val_y)
            .map(|(p, a)| (*p as f64 - *a as f64).powi(2))
            .sum::<f64>()
            / inputs.val_y.len() as f64;

        emit(serde_json::json!({
            "event": "epoch",
            "epoch": epoch,
            "total_epochs": cfg.epochs,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }));

        if cfg.patience > 0 && best.as_ref().is_none_or(|(b, _, _)| val_loss < *b) {
            best = Some((val_loss, epoch, model.clone()));
        }

        // Periodic checkpoint: after this event the run is promotable even
        // if the process is later killed. The final epoch is skipped — the
        // caller saves right after training anyway. With patience tracking
        // on, checkpoint the best-so-far weights, not the current ones.
        if cfg.checkpoint_every > 0 && epoch % cfg.checkpoint_every == 0 && epoch < cfg.epochs {
            save_atomic(best.as_ref().map_or(&model, |(_, _, m)| m), run_dir)?;
            emit(serde_json::json!({"event":"checkpoint","epoch":epoch}));
        }

        // Patience exhausted: the val curve hasn't improved in `patience`
        // epochs — further training only fits noise.
        if let Some((_, best_epoch, _)) = &best {
            if epoch - best_epoch >= cfg.patience {
                emit(serde_json::json!({"event":"log","msg":format!(
                    "early stop at epoch {epoch}: no val improvement in {} epochs",
                    cfg.patience)}));
                break;
            }
        }
    }

    // Best-checkpoint restore: hand the caller the lowest-val-loss weights
    // (only set when patience > 0; None also covers a STOP before epoch 1).
    if let Some((best_val, best_epoch, best_model)) = best {
        emit(serde_json::json!({"event":"log","msg":format!(
            "restoring best epoch {best_epoch} (val_loss {best_val:.6})")}));
        return Ok(best_model);
    }
    Ok(model)
}

fn predict_inner(
    model: &Mlp<Inner>,
    x: &[f32],
    n_cols: usize,
    batch_size: usize,
) -> Vec<f32> {
    let device = NdArrayDevice::default();
    let n = x.len() / n_cols;
    let rows: Vec<usize> = (0..n).collect();
    let mut out = Vec::with_capacity(n);
    for batch in rows.chunks(batch_size.max(1)) {
        let t = batch_tensor::<Inner>(x, batch, n_cols, &device);
        let pred = model.forward(t);
        out.extend(pred.into_data().to_vec::<f32>().unwrap());
    }
    out
}

/// Predict transformed-space targets for all rows of `x` (eval mode).
pub fn predict(model: &Mlp<Auto>, x: &[f32], n_cols: usize, batch_size: usize) -> Vec<f32> {
    predict_inner(&model.valid(), x, n_cols, batch_size)
}

/// Atomically write the checkpoint as `<dir>/model.mpk`: record to a temp
/// stem first, then rename — a kill mid-write can never corrupt the loadable
/// file. Saves the eval-mode (inner) module; records are backend-agnostic,
/// so `load` reads it unchanged.
///
/// The temp stem is dot-free ("model-tmp") because the recorder
/// `set_extension`s the path: "model.tmp" would collapse onto "model.mpk".
pub fn save_atomic(model: &Mlp<Auto>, dir: &Path) -> Result<()> {
    let tmp = dir.join("model-tmp");
    model
        .valid()
        .save_file(tmp.to_string_lossy().as_ref(), &CompactRecorder::new())
        .map_err(|e| anyhow::anyhow!("save checkpoint: {e}"))?;
    std::fs::rename(dir.join("model-tmp.mpk"), dir.join("model.mpk"))?;
    Ok(())
}

/// Load a checkpoint for inference (inner backend, no autodiff). The module
/// must be rebuilt with the trained architecture BEFORE loading weights;
/// `hidden`/`dropout`/`activation` come from the model dir's hyperparams snapshot.
pub fn load(
    path: &Path,
    n_in: usize,
    hidden: &[usize],
    dropout: f64,
    activation: Activation,
) -> Result<Mlp<Inner>> {
    let device = NdArrayDevice::default();
    Mlp::<Inner>::new(n_in, hidden, dropout, activation, &device)
        .load_file(path.to_string_lossy().as_ref(), &CompactRecorder::new(), &device)
        .map_err(|e| anyhow::anyhow!("load checkpoint {}: {e}", path.display()))
}

/// Predict transformed-space targets with a loaded inference model.
pub fn predict_loaded(model: &Mlp<Inner>, x: &[f32], n_cols: usize, batch_size: usize) -> Vec<f32> {
    predict_inner(model, x, n_cols, batch_size)
}

/// One dense layer's weights, extracted for ONNX export. `weight` is row-major
/// `[w_in, w_out]` (burn's `Linear` layout, so an ONNX `Gemm` with `transB=0`).
pub struct DenseLayer {
    pub weight: Vec<f32>,
    pub w_in: usize,
    pub w_out: usize,
    pub bias: Option<Vec<f32>>,
}

impl Mlp<Inner> {
    /// The hidden layers followed by the output layer, in forward order — the
    /// `Gemm` chain an ONNX export emits between activations. The activation
    /// (`self.activation`) is applied after every layer except the last; the
    /// dropout is inference-inactive and contributes nothing.
    pub fn dense_layers(&self) -> Vec<DenseLayer> {
        let extract = |layer: &Linear<Inner>| {
            let [w_in, w_out] = layer.weight.dims();
            DenseLayer {
                weight: layer.weight.val().into_data().to_vec::<f32>().unwrap(),
                w_in,
                w_out,
                bias: layer
                    .bias
                    .as_ref()
                    .map(|b| b.val().into_data().to_vec::<f32>().unwrap()),
            }
        };
        self.layers.iter().chain(std::iter::once(&self.output)).map(extract).collect()
    }

    pub fn activation_kind(&self) -> Activation {
        self.activation
    }
}
