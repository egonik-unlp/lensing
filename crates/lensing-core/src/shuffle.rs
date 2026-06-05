//! Deterministic seeded shuffle, dependency-free so any language can
//! reproduce the train/test split from `(n_rows, seed, test_ratio)` alone.
//! Algorithm: SplitMix64 stream driving a Fisher-Yates shuffle; documented
//! in the README next to the artifact format.

pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Unbiased integer in [0, bound) via Lemire-style rejection.
    fn below(&mut self, bound: u64) -> u64 {
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let r = self.next_u64();
            if r >= threshold {
                return r % bound;
            }
        }
    }
}

/// Split `0..n` into sorted (train, test) index lists.
pub fn train_test_split(n: usize, test_ratio: f64, seed: u64) -> (Vec<u32>, Vec<u32>) {
    let mut idx: Vec<u32> = (0..n as u32).collect();
    let mut rng = SplitMix64(seed);
    // Fisher-Yates
    for i in (1..n).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        idx.swap(i, j);
    }
    let n_test = ((n as f64) * test_ratio).round() as usize;
    let mut test: Vec<u32> = idx[..n_test].to_vec();
    let mut train: Vec<u32> = idx[n_test..].to_vec();
    train.sort_unstable();
    test.sort_unstable();
    (train, test)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_is_deterministic_and_disjoint() {
        let (tr1, te1) = train_test_split(1000, 0.2, 42);
        let (tr2, te2) = train_test_split(1000, 0.2, 42);
        assert_eq!(tr1, tr2);
        assert_eq!(te1, te2);
        assert_eq!(te1.len(), 200);
        assert_eq!(tr1.len(), 800);
        let mut all: Vec<u32> = tr1.iter().chain(te1.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..1000).collect::<Vec<u32>>());
    }

    #[test]
    fn different_seeds_differ() {
        let (_, te1) = train_test_split(1000, 0.2, 42);
        let (_, te2) = train_test_split(1000, 0.2, 43);
        assert_ne!(te1, te2);
    }
}
