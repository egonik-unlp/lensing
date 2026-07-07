//! Nonlinear probes for the embedding probe (P1), in burn: a small MLP
//! regressor (the "nonlinear ceiling") and a multinomial logistic classifier
//! (use-class decode). Kept low-capacity + regularized + early-stopped per the
//! probing literature — the linear ridge probe remains the primary read.

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::Autodiff;
use burn::module::{AutodiffModule, Module};
use burn::nn::loss::{CrossEntropyLossConfig, MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{activation, Int, Tensor, TensorData};

use lensing_core::shuffle::SplitMix64;

type Inner = NdArray<f32>;
type Auto = Autodiff<Inner>;

/// Column standardizer fit on the train matrix, applied to both splits → f32.
fn standardize(xtr: &[f64], xte: &[f64], dim: usize) -> (Vec<f32>, Vec<f32>) {
    let d = dim.max(1);
    let ntr = xtr.len() / d;
    let mut mean = vec![0f64; dim];
    for r in 0..ntr {
        for j in 0..dim {
            mean[j] += xtr[r * dim + j];
        }
    }
    for m in &mut mean {
        *m /= ntr.max(1) as f64;
    }
    let mut std = vec![0f64; dim];
    for r in 0..ntr {
        for j in 0..dim {
            let dv = xtr[r * dim + j] - mean[j];
            std[j] += dv * dv;
        }
    }
    for s in &mut std {
        *s = (*s / ntr.max(1) as f64).sqrt().max(1e-8);
    }
    let app = |x: &[f64]| -> Vec<f32> {
        let n = x.len() / d;
        let mut o = vec![0f32; x.len()];
        for r in 0..n {
            for j in 0..dim {
                o[r * dim + j] = ((x[r * dim + j] - mean[j]) / std[j]) as f32;
            }
        }
        o
    };
    (app(xtr), app(xte))
}

fn rows_tensor<B: Backend>(x: &[f32], rows: &[usize], dim: usize, dev: &B::Device) -> Tensor<B, 2> {
    let mut flat = Vec::with_capacity(rows.len() * dim);
    for &r in rows {
        flat.extend_from_slice(&x[r * dim..(r + 1) * dim]);
    }
    Tensor::from_data(TensorData::new(flat, [rows.len(), dim]), dev)
}

fn all_tensor<B: Backend>(x: &[f32], n: usize, dim: usize, dev: &B::Device) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(x.to_vec(), [n, dim]), dev)
}

// --------------------------------------------------------------------------
// MLP regressor (nonlinear ceiling probe)
// --------------------------------------------------------------------------

#[derive(Module, Debug)]
struct Reg<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
}

impl<B: Backend> Reg<B> {
    fn new(din: usize, h: usize, dev: &B::Device) -> Self {
        Reg { l1: LinearConfig::new(din, h).init(dev), l2: LinearConfig::new(h, 1).init(dev) }
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.l2.forward(activation::relu(self.l1.forward(x)))
    }
}

/// One-hidden-layer (128) ReLU MLP, L2-regularized + early-stopped on a 10%
/// validation carve. Targets are centred (mean added back). Returns log-space
/// test predictions.
pub fn mlp_regress(xtr: &[f64], ytr: &[f64], xte: &[f64], dim: usize, seed: u64) -> Vec<f64> {
    let dev = NdArrayDevice::default();
    Auto::seed(&dev, seed);
    let (xtr_s, xte_s) = standardize(xtr, xte, dim);
    let ntr = ytr.len();
    let ymean = ytr.iter().sum::<f64>() / ntr.max(1) as f64;
    let yc: Vec<f32> = ytr.iter().map(|v| (v - ymean) as f32).collect();

    // Deterministic 90/10 fit/val carve for early stopping.
    let mut rng = SplitMix64(seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut order: Vec<usize> = (0..ntr).collect();
    for i in (1..ntr).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    let n_val = (ntr / 10).clamp(1, ntr.saturating_sub(1).max(1));
    let val: Vec<usize> = order[..n_val].to_vec();
    let mut fit: Vec<usize> = order[n_val..].to_vec();

    let mut model: Reg<Auto> = Reg::new(dim, 128, &dev);
    let mut opt = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-3)))
        .init();
    let loss_fn = MseLoss::new();
    let batch = 256usize;

    let target = |rows: &[usize], dev: &NdArrayDevice| -> Tensor<Auto, 2> {
        let flat: Vec<f32> = rows.iter().map(|&r| yc[r]).collect();
        Tensor::from_data(TensorData::new(flat, [rows.len(), 1]), dev)
    };

    let mut best: Option<(f64, usize, Reg<Auto>)> = None;
    for epoch in 0..200usize {
        for i in (1..fit.len()).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            fit.swap(i, j);
        }
        for b in fit.chunks(batch) {
            let x = rows_tensor::<Auto>(&xtr_s, b, dim, &dev);
            let y = target(b, &dev);
            let loss = loss_fn.forward(model.forward(x), y, Reduction::Mean);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = opt.step(1e-3, model, grads);
        }
        // validation MSE on the inner (eval) module
        let vpred = model
            .valid()
            .forward(rows_tensor::<Inner>(&xtr_s, &val, dim, &dev))
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let vmse = val
            .iter()
            .zip(&vpred)
            .map(|(&r, &p)| (p as f64 - yc[r] as f64).powi(2))
            .sum::<f64>()
            / val.len().max(1) as f64;
        if best.as_ref().is_none_or(|(b, _, _)| vmse < *b) {
            best = Some((vmse, epoch, model.clone()));
        }
        // early stop: no val improvement in 15 epochs
        if let Some((_, be, _)) = &best {
            if epoch - be >= 15 {
                break;
            }
        }
    }

    let final_model = best.map(|(_, _, m)| m).unwrap_or(model);
    let n_te = xte.len() / dim.max(1);
    let pred = final_model
        .valid()
        .forward(all_tensor::<Inner>(&xte_s, n_te, dim, &dev))
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    pred.into_iter().map(|p| p as f64 + ymean).collect()
}

// --------------------------------------------------------------------------
// Multinomial logistic classifier (use-class decode)
// --------------------------------------------------------------------------

#[derive(Module, Debug)]
struct Logit<B: Backend> {
    lin: Linear<B>,
}

impl<B: Backend> Logit<B> {
    fn new(din: usize, k: usize, dev: &B::Device) -> Self {
        Logit { lin: LinearConfig::new(din, k).init(dev) }
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.lin.forward(x)
    }
}

pub struct LogisticOut {
    /// predicted class index per test row
    pub pred: Vec<usize>,
    /// softmax probabilities, row-major `[n_test, k]`
    pub proba: Vec<f64>,
}

/// L2-regularized multinomial logistic regression (softmax) trained with Adam +
/// cross-entropy. Returns argmax predictions and softmax probabilities on test.
pub fn logistic_decode(
    xtr: &[f64],
    ytr_class: &[usize],
    xte: &[f64],
    dim: usize,
    k: usize,
    seed: u64,
) -> LogisticOut {
    let dev = NdArrayDevice::default();
    Auto::seed(&dev, seed);
    let (xtr_s, xte_s) = standardize(xtr, xte, dim);
    let ntr = ytr_class.len();

    let mut model: Logit<Auto> = Logit::new(dim, k, &dev);
    let mut opt = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-3)))
        .init();
    let ce = CrossEntropyLossConfig::new().init(&dev);
    let batch = 512usize;

    let mut rng = SplitMix64(seed ^ 0x2545_F491_4F6C_DD1D);
    let mut order: Vec<usize> = (0..ntr).collect();
    for _epoch in 0..150usize {
        for i in (1..ntr).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        for b in order.chunks(batch) {
            let x = rows_tensor::<Auto>(&xtr_s, b, dim, &dev);
            let tgt: Vec<i64> = b.iter().map(|&r| ytr_class[r] as i64).collect();
            let targets = Tensor::<Auto, 1, Int>::from_data(TensorData::new(tgt, [b.len()]), &dev);
            let loss = ce.forward(model.forward(x), targets);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = opt.step(5e-3, model, grads);
        }
    }

    let n_te = xte.len() / dim.max(1);
    let logits = model.valid().forward(all_tensor::<Inner>(&xte_s, n_te, dim, &dev));
    let proba: Vec<f64> = activation::softmax(logits.clone(), 1)
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(|p| p as f64)
        .collect();
    let pred: Vec<usize> = logits
        .argmax(1)
        .into_data()
        .to_vec::<i64>()
        .unwrap()
        .into_iter()
        .map(|c| c as usize)
        .collect();
    LogisticOut { pred, proba }
}
