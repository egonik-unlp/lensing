//! Closed-form ridge regression for a single probe stage.
//!
//! The probe objective is least squares with an L2 penalty, whose exact optimum
//! is the normal-equations solution `(HᵀH + λI) w = Hᵀy`. We solve it with an
//! in-crate Cholesky (no BLAS/LAPACK dependency); the Gram matrix is
//! `dim×dim` with `dim` the stage width (≤ a few hundred), so this is instant.
//! λ is chosen by a small internal hold-out CV unless the caller forces one.

/// Ridge penalties tried by the internal hold-out CV when no λ is forced.
pub(crate) const LAMBDAS: [f64; 4] = [1.0, 10.0, 100.0, 1000.0];

pub(crate) struct ProbeFit {
    pub lambda: f64,
    pub train_pred_log: Vec<f64>,
    pub test_pred_log: Vec<f64>,
}

/// Fit a ridge probe `H → y` (log space). Columns are standardized on train; λ
/// is picked by a 90/10 hold-out over `LAMBDAS` (or forced if `lambda_override`
/// > 0), then refit on all train rows. Returns log-space predictions for both
/// splits.
pub(crate) fn fit_layer_probe(
    htr: &[f64],
    ytr: &[f64],
    hte: &[f64],
    dim: usize,
    lambda_override: f64,
) -> ProbeFit {
    let n_tr = ytr.len();
    let (mean, std) = col_stats(htr, n_tr, dim);
    let htr_s = standardize(htr, n_tr, dim, &mean, &std);
    let hte_s = standardize(hte, hte.len() / dim.max(1), dim, &mean, &std);
    let ymean = ytr.iter().sum::<f64>() / n_tr as f64;
    let yc: Vec<f64> = ytr.iter().map(|v| v - ymean).collect();

    let lambda = if lambda_override > 0.0 {
        lambda_override
    } else {
        pick_lambda(&htr_s, &yc, n_tr, dim)
    };

    // Refit on all train rows with the chosen λ.
    let (mut gram, rhs) = gram_rhs(&htr_s, &yc, n_tr, dim);
    for d in 0..dim {
        gram[d * dim + d] += lambda;
    }
    let w = chol_solve(&mut gram, &rhs, dim).unwrap_or_else(|| vec![0.0; dim]);

    let train_pred_log = predict(&htr_s, &w, ymean, n_tr, dim);
    let test_pred_log = predict(&hte_s, &w, ymean, hte.len() / dim.max(1), dim);
    ProbeFit { lambda, train_pred_log, test_pred_log }
}

/// 90/10 hold-out CV over LAMBDAS; returns the λ with the lowest val MSE.
fn pick_lambda(h_s: &[f64], yc: &[f64], n: usize, dim: usize) -> f64 {
    let n_fit = ((n as f64 * 0.9) as usize).max(1).min(n.saturating_sub(1).max(1));
    let (hfit, hval) = h_s.split_at(n_fit * dim);
    let (yfit, yval) = yc.split_at(n_fit);
    let (gram0, rhs) = gram_rhs(hfit, yfit, n_fit, dim);
    let mut best = (f64::INFINITY, LAMBDAS[0]);
    for &lam in &LAMBDAS {
        let mut a = gram0.clone();
        for d in 0..dim {
            a[d * dim + d] += lam;
        }
        if let Some(w) = chol_solve(&mut a, &rhs, dim) {
            let pred = predict(hval, &w, 0.0, yval.len(), dim);
            let mse = pred.iter().zip(yval).map(|(p, y)| (p - y).powi(2)).sum::<f64>()
                / yval.len().max(1) as f64;
            if mse < best.0 {
                best = (mse, lam);
            }
        }
    }
    best.1
}

fn col_stats(h: &[f64], n: usize, dim: usize) -> (Vec<f64>, Vec<f64>) {
    let mut mean = vec![0.0; dim];
    for r in 0..n {
        for j in 0..dim {
            mean[j] += h[r * dim + j];
        }
    }
    for m in mean.iter_mut() {
        *m /= n.max(1) as f64;
    }
    let mut var = vec![0.0; dim];
    for r in 0..n {
        for j in 0..dim {
            let d = h[r * dim + j] - mean[j];
            var[j] += d * d;
        }
    }
    let std: Vec<f64> = var.iter().map(|v| (v / n.max(1) as f64).sqrt().max(1e-8)).collect();
    (mean, std)
}

fn standardize(h: &[f64], n: usize, dim: usize, mean: &[f64], std: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; n * dim];
    for r in 0..n {
        for j in 0..dim {
            out[r * dim + j] = (h[r * dim + j] - mean[j]) / std[j];
        }
    }
    out
}

/// Symmetric Gram `HᵀH` (full `dim×dim`) and `Hᵀy`, accumulated row-streaming.
fn gram_rhs(h: &[f64], y: &[f64], n: usize, dim: usize) -> (Vec<f64>, Vec<f64>) {
    let mut gram = vec![0.0; dim * dim];
    let mut rhs = vec![0.0; dim];
    for r in 0..n {
        let row = &h[r * dim..(r + 1) * dim];
        let yr = y[r];
        for j in 0..dim {
            let hj = row[j];
            rhs[j] += hj * yr;
            let base = j * dim;
            for k in j..dim {
                gram[base + k] += hj * row[k];
            }
        }
    }
    // mirror upper → lower
    for j in 0..dim {
        for k in (j + 1)..dim {
            gram[k * dim + j] = gram[j * dim + k];
        }
    }
    (gram, rhs)
}

/// Solve a symmetric-positive-definite `A w = b` in place via Cholesky
/// (`A = L Lᵀ`). Returns `None` if `A` is not PD (shouldn't happen with λ > 0).
fn chol_solve(a: &mut [f64], b: &[f64], dim: usize) -> Option<Vec<f64>> {
    // Cholesky → write L into the lower triangle of `a`.
    for j in 0..dim {
        let mut s = a[j * dim + j];
        for k in 0..j {
            s -= a[j * dim + k] * a[j * dim + k];
        }
        if s <= 0.0 {
            return None;
        }
        let ljj = s.sqrt();
        a[j * dim + j] = ljj;
        for i in (j + 1)..dim {
            let mut s2 = a[i * dim + j];
            for k in 0..j {
                s2 -= a[i * dim + k] * a[j * dim + k];
            }
            a[i * dim + j] = s2 / ljj;
        }
    }
    // Forward solve L z = b
    let mut z = vec![0.0; dim];
    for i in 0..dim {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * dim + k] * z[k];
        }
        z[i] = s / a[i * dim + i];
    }
    // Back solve Lᵀ w = z
    let mut w = vec![0.0; dim];
    for i in (0..dim).rev() {
        let mut s = z[i];
        for k in (i + 1)..dim {
            s -= a[k * dim + i] * w[k];
        }
        w[i] = s / a[i * dim + i];
    }
    Some(w)
}

fn predict(h_s: &[f64], w: &[f64], intercept: f64, n: usize, dim: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for r in 0..n {
        let mut acc = intercept;
        let row = &h_s[r * dim..(r + 1) * dim];
        for j in 0..dim {
            acc += row[j] * w[j];
        }
        out[r] = acc;
    }
    out
}
