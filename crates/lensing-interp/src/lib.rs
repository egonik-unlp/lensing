//! Interpretability engine for the a lensing instance.
//!
//! Tool #1 — **layer-wise linear probes** (Alain & Bengio 2016,
//! arXiv:1610.01644): fit a cheap closed-form ridge probe on the activations at
//! each *stage* of a trained network and report how linearly decodable the
//! target is at that depth. The rising curve is a window on how the architecture
//! builds its prediction; the gap to the model's own output is what the final
//! nonlinear path still adds.
//!
//! This crate is the family-agnostic compute core: it takes named stage
//! activation matrices (captured by a predictor's own forward pass) plus the
//! split and target, and returns per-stage [`StageResult`]s. It has no `burn`
//! dependency and no architecture or domain knowledge — keeping it reusable
//! across predictor families and portable to the upstream framework.

mod ridge;
mod stage;

pub use stage::{
    feature_group_stages, fit_stages, gather_targets, reference_stage, ridge, RidgeFit, Stage,
    StageKind, StageResult,
};

#[cfg(test)]
mod tests {
    use super::*;
    use lensing_core::shuffle::SplitMix64;
    use lensing_core::TargetTransform;

    /// Synthetic check: the target is an exact linear function of a "deep" stage
    /// and unrelated to a noisy "input" stage. A linear probe must decode the
    /// target near-perfectly at the deep stage and poorly at the input — the
    /// core Alain & Bengio result that decodability rises with depth.
    #[test]
    fn decodability_rises_from_input_to_deep_stage() {
        let n = 240usize;
        let (din, ddeep) = (5usize, 2usize);
        let mut rng = SplitMix64(42);
        let mut uni = || ((rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;

        let mut input = Vec::with_capacity(n * din);
        let mut deep = Vec::with_capacity(n * ddeep);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let (a, b) = (uni(), uni());
            deep.push(a as f32);
            deep.push(b as f32);
            for _ in 0..din {
                input.push(uni() as f32);
            }
            y.push((3.0 * a + 2.0 * b) as f32);
        }

        let train_idx: Vec<u32> = (0..200).collect();
        let test_idx: Vec<u32> = (200..n as u32).collect();
        let ytr = gather_targets(&y, &train_idx);
        let yte = gather_targets(&y, &test_idx);

        let stages = vec![
            Stage { name: "input".into(), kind: StageKind::Input, dim: din, values: input },
            Stage { name: "deep".into(), kind: StageKind::Hidden, dim: ddeep, values: deep },
        ];
        let res = fit_stages(
            &stages,
            &train_idx,
            &test_idx,
            &ytr,
            &yte,
            TargetTransform::None,
            f64::INFINITY,
            0.0,
            &|_| {},
        );

        assert_eq!(res.len(), 2);
        assert!(res[1].test_r2_log > 0.9, "deep stage R² should be ~1, got {}", res[1].test_r2_log);
        assert!(
            res[1].test_r2_log > res[0].test_r2_log + 0.2,
            "deep ({}) should decode far better than input ({})",
            res[1].test_r2_log,
            res[0].test_r2_log
        );
    }
}
