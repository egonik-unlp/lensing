//! Multimodal block layout + balanced reconstruction loss + per-block R².
//!
//! When the input concatenates heterogeneous blocks (text / numeric / acoustic /
//! categorical, as in the SongAE), reconstructing them under one global MSE lets
//! a wide block (e.g. 384 text dims) drown a narrow one (e.g. 11 acoustics). The
//! fix, lifted from `build_song_ae.py`'s `block_loss`, is to sum the *per-block*
//! mean error so each modality contributes on equal footing regardless of width.

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

/// One contiguous span `[start, end)` of the concatenated feature vector.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Block {
    pub name: String,
    pub start: usize,
    pub end: usize,
}

impl Block {
    pub fn width(&self) -> usize {
        self.end - self.start
    }
}

/// The ordered set of blocks covering a width-`d` feature vector.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct BlockLayout {
    pub blocks: Vec<Block>,
}

impl BlockLayout {
    /// A single block spanning the whole vector — the "no multimodal structure"
    /// case, where block-loss degenerates to plain global MSE.
    pub fn single(d: usize) -> Self {
        BlockLayout {
            blocks: vec![Block { name: "all".into(), start: 0, end: d }],
        }
    }

    /// Build contiguous blocks from `(name, width)` pairs, laid out in order.
    pub fn from_widths(widths: &[(&str, usize)]) -> Self {
        let mut blocks = Vec::with_capacity(widths.len());
        let mut cursor = 0;
        for (name, w) in widths {
            blocks.push(Block { name: (*name).into(), start: cursor, end: cursor + w });
            cursor += w;
        }
        BlockLayout { blocks }
    }

    /// Total width covered (the last block's end).
    pub fn width(&self) -> usize {
        self.blocks.last().map_or(0, |b| b.end)
    }

    /// Validate that blocks are contiguous, non-empty and cover exactly `d`.
    pub fn validate(&self, d: usize) -> Result<()> {
        ensure!(!self.blocks.is_empty(), "block layout is empty");
        let mut expected = 0;
        for b in &self.blocks {
            ensure!(b.start == expected, "block {:?} starts at {} not {}", b.name, b.start, expected);
            ensure!(b.end > b.start, "block {:?} is empty", b.name);
            expected = b.end;
        }
        ensure!(expected == d, "blocks cover {expected} dims, expected {d}");
        Ok(())
    }

    /// Per-block reconstruction R² (`1 - MSE/Var`) over row-major matrices
    /// `recon` and `target` (both n × d). Returns `(block_name, r2)` per block;
    /// the metric the AE reports and the UI renders as bars.
    pub fn reconstruction_r2(&self, recon: &[f32], target: &[f32], d: usize) -> Vec<(String, f32)> {
        let n = target.len() / d;
        let mut out = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            let (mut sse, mut sum, mut sumsq, mut count) = (0.0f64, 0.0f64, 0.0f64, 0usize);
            for i in 0..n {
                for j in b.start..b.end {
                    let t = target[i * d + j] as f64;
                    let r = recon[i * d + j] as f64;
                    let e = r - t;
                    sse += e * e;
                    sum += t;
                    sumsq += t * t;
                    count += 1;
                }
            }
            let mse = sse / count.max(1) as f64;
            let mean = sum / count.max(1) as f64;
            let var = (sumsq / count.max(1) as f64) - mean * mean + 1e-9;
            out.push((b.name.clone(), (1.0 - mse / var) as f32));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_widths_lays_out_contiguously() {
        let l = BlockLayout::from_widths(&[("text", 384), ("numeric", 10), ("acoustic", 22)]);
        assert_eq!(l.width(), 416);
        l.validate(416).unwrap();
        assert_eq!(l.blocks[1], Block { name: "numeric".into(), start: 384, end: 394 });
    }

    #[test]
    fn perfect_reconstruction_is_r2_one() {
        let d = 4;
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let l = BlockLayout::from_widths(&[("a", 2), ("b", 2)]);
        let r2 = l.reconstruction_r2(&x, &x, d);
        for (_, v) in &r2 {
            assert!((*v - 1.0).abs() < 1e-4, "r2 = {v}");
        }
    }
}
