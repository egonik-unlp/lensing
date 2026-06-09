//! Prediction combination (target space) and the validation-split weight grid.

use anyhow::{ensure, Result};

use crate::spec::Rule;

/// Combine per-member predictions row-wise. `preds[m][r]` = member m's
/// target-space prediction for row r; `weights` are pre-normalized and only
/// used for `mean`.
pub fn combine(rule: Rule, preds: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    let n_rows = preds[0].len();
    match rule {
        Rule::Mean => (0..n_rows)
            .map(|r| preds.iter().zip(weights).map(|(p, w)| w * p[r]).sum())
            .collect(),
        Rule::Median => (0..n_rows)
            .map(|r| {
                let mut v: Vec<f64> = preds.iter().map(|p| p[r]).collect();
                v.sort_by(|a, b| a.total_cmp(b));
                let mid = v.len() / 2;
                if v.len() % 2 == 1 { v[mid] } else { (v[mid - 1] + v[mid]) / 2.0 }
            })
            .collect(),
    }
}

/// Normalize weights to sum to 1.
pub fn normalize(weights: &[f64]) -> Result<Vec<f64>> {
    let sum: f64 = weights.iter().sum();
    ensure!(sum > 0.0, "member weights must have a positive sum");
    Ok(weights.iter().map(|w| w / sum).collect())
}

/// Mean absolute error (regression grid objective). Lower is better.
pub fn mae_obj(blended: &[f64], actual: &[f64]) -> f64 {
    blended.iter().zip(actual).map(|(p, a)| (p - a).abs()).sum::<f64>() / actual.len() as f64
}

/// Binary cross-entropy (classification grid objective): `blended` is the
/// combined P(class==1), `actual` ∈ {0,1}. Lower is better.
pub fn logloss_obj(blended: &[f64], actual: &[f64]) -> f64 {
    const EPS: f64 = 1e-15;
    blended
        .iter()
        .zip(actual)
        .map(|(p, y)| {
            let p = p.clamp(EPS, 1.0 - EPS);
            -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
        })
        .sum::<f64>()
        / actual.len() as f64
}

/// MAE-optimal grid (regression default; kept for callers/tests).
pub fn grid_fit(preds: &[Vec<f64>], actual: &[f64]) -> Result<Vec<f64>> {
    grid_fit_with(preds, actual, mae_obj)
}

/// Grid-search mean weights on validation predictions (step 0.05 over the
/// simplex; 2 or 3 members) minimizing `objective` (lower-is-better). Returns
/// the optimal normalized weights.
pub fn grid_fit_with(
    preds: &[Vec<f64>],
    actual: &[f64],
    objective: impl Fn(&[f64], &[f64]) -> f64,
) -> Result<Vec<f64>> {
    ensure!(
        (2..=3).contains(&preds.len()),
        "weight_fit grid supports 2 or 3 members, got {}",
        preds.len()
    );
    let mae = |w: &[f64]| -> f64 {
        let blended = combine(Rule::Mean, preds, w);
        objective(&blended, actual)
    };
    let steps = 20usize; // 0.05 grid
    let mut best = (f64::INFINITY, vec![]);
    if preds.len() == 2 {
        for i in 0..=steps {
            let w0 = i as f64 / steps as f64;
            let w = vec![w0, 1.0 - w0];
            let m = mae(&w);
            if m < best.0 {
                best = (m, w);
            }
        }
    } else {
        for i in 0..=steps {
            for j in 0..=(steps - i) {
                let (w0, w1) = (i as f64 / steps as f64, j as f64 / steps as f64);
                let w = vec![w0, w1, 1.0 - w0 - w1];
                let m = mae(&w);
                if m < best.0 {
                    best = (m, w);
                }
            }
        }
    }
    Ok(best.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_and_median_combine_in_target_space() {
        let preds = vec![vec![100.0, 10.0], vec![200.0, 20.0], vec![600.0, 90.0]];
        let mean = combine(Rule::Mean, &preds, &[0.5, 0.25, 0.25]);
        assert_eq!(mean, vec![250.0, 32.5]);
        let med = combine(Rule::Median, &preds, &[]);
        assert_eq!(med, vec![200.0, 20.0]);
        // Even count → midpoint.
        let med2 = combine(Rule::Median, &preds[..2].to_vec(), &[]);
        assert_eq!(med2, vec![150.0, 15.0]);
    }

    #[test]
    fn grid_recovers_known_optimum() {
        // Member 0 is exactly right, member 1 is constant noise: w=[1,0] wins.
        let actual = vec![100.0, 200.0, 300.0, 400.0];
        let preds = vec![actual.clone(), vec![250.0; 4]];
        let w = grid_fit(&preds, &actual).unwrap();
        assert_eq!(w, vec![1.0, 0.0]);
        // Symmetric opposite errors: 50/50 is optimal.
        let preds = vec![
            actual.iter().map(|a| a + 50.0).collect::<Vec<_>>(),
            actual.iter().map(|a| a - 50.0).collect::<Vec<_>>(),
        ];
        let w = grid_fit(&preds, &actual).unwrap();
        assert_eq!(w, vec![0.5, 0.5]);
    }
}
