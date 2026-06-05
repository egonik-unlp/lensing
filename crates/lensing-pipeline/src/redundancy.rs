//! Feature-redundancy analysis over the encoded metadata columns (numeric +
//! one-hot). PCA components are orthogonal by construction and span 1536 raw
//! dims, so they are deliberately excluded — both for correctness (no spurious
//! correlations) and to keep the pairwise scan O(p²n) over a small p.

use pg_core::redundancy::{CorrelatedPair, NearZeroColumn, OnehotGroupStat, RedundancyReport};
use pg_core::ColumnKind;

use crate::features::Encoder;
use crate::qdrant::RawPoint;

/// Columns with variance at or below this are flagged near-zero (degenerate).
const NEAR_ZERO_VAR: f64 = 1e-6;
/// `|corr|` above this flags a column pair as redundant.
const CORR_THRESHOLD: f64 = 0.95;

/// Analyze the metadata feature matrix the `encoder` produces over `points`.
/// The encoder must be built the same way the dataset build builds it (on the
/// full quality-filtered point set) for the findings to describe the build.
pub fn analyze(points: &[RawPoint], encoder: &Encoder) -> RedundancyReport {
    let cols = encoder.columns();
    let width = encoder.width();
    let n = points.len();
    if width == 0 || n == 0 {
        return RedundancyReport::default();
    }

    // Encode the full metadata matrix, row-major n × width.
    let mut mat = vec![0.0f32; n * width];
    for (i, p) in points.iter().enumerate() {
        encoder.encode(p, &mut mat[i * width..(i + 1) * width]);
    }

    // Column means.
    let mut mean = vec![0.0f64; width];
    for i in 0..n {
        let row = &mat[i * width..(i + 1) * width];
        for j in 0..width {
            mean[j] += row[j] as f64;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }

    // Pairwise (and diagonal) centered cross-products → covariance. width is
    // small (numeric + one-hot only), so the p²n scan is cheap.
    let mut cov = vec![0.0f64; width * width];
    let mut centered = vec![0.0f64; width];
    for i in 0..n {
        let row = &mat[i * width..(i + 1) * width];
        for j in 0..width {
            centered[j] = row[j] as f64 - mean[j];
        }
        for a in 0..width {
            let xa = centered[a];
            let base = a * width;
            for b in a..width {
                cov[base + b] += xa * centered[b];
            }
        }
    }
    for c in &mut cov {
        *c /= n as f64;
    }
    let var = |j: usize| cov[j * width + j];

    // Near-zero variance columns.
    let near_zero_variance = cols
        .iter()
        .enumerate()
        .filter(|(j, _)| var(*j) <= NEAR_ZERO_VAR)
        .map(|(j, c)| NearZeroColumn { column: c.name.clone(), variance: var(j) })
        .collect();

    // Highly-correlated pairs (skip degenerate columns: corr is undefined when
    // either variance is ~0, and those are already reported as near-zero).
    let mut correlated_pairs = Vec::new();
    for a in 0..width {
        let va = var(a);
        if va <= NEAR_ZERO_VAR {
            continue;
        }
        for b in (a + 1)..width {
            let vb = var(b);
            if vb <= NEAR_ZERO_VAR {
                continue;
            }
            let corr = cov[a * width + b] / (va.sqrt() * vb.sqrt());
            if corr.abs() > CORR_THRESHOLD {
                correlated_pairs.push(CorrelatedPair {
                    a: cols[a].name.clone(),
                    b: cols[b].name.clone(),
                    corr,
                });
            }
        }
    }
    correlated_pairs.sort_by(|x, y| y.corr.abs().total_cmp(&x.corr.abs()));

    // One-hot group occupancy, from column sums (sum_j = mean_j · n).
    // Categories occupied by fewer than this many rows count as "rare".
    let rare_threshold = (n / 1000).max(5);
    let mut onehot_groups: Vec<OnehotGroupStat> = Vec::new();
    for (j, c) in cols.iter().enumerate() {
        let ColumnKind::Onehot { group, value } = &c.kind else { continue };
        let count = (mean[j] * n as f64).round() as usize;
        let is_other = value == "__other__";
        match onehot_groups.last_mut() {
            Some(g) if g.group == *group => {
                if is_other {
                    g.other_fraction = count as f64 / n as f64;
                } else {
                    g.n_values += 1;
                    if count < rare_threshold {
                        g.rare_buckets += 1;
                    }
                }
            }
            _ => {
                let mut g = OnehotGroupStat {
                    group: group.clone(),
                    n_values: 0,
                    other_fraction: 0.0,
                    rare_buckets: 0,
                };
                if is_other {
                    g.other_fraction = count as f64 / n as f64;
                } else {
                    g.n_values = 1;
                    if count < rare_threshold {
                        g.rare_buckets = 1;
                    }
                }
                onehot_groups.push(g);
            }
        }
    }

    RedundancyReport { near_zero_variance, onehot_groups, correlated_pairs }
}
