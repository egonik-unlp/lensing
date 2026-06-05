//! Per-column standardization fit on the train split only.

use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Scaler {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl Scaler {
    pub fn fit(x: &[f32], n_cols: usize) -> Self {
        let n = x.len() / n_cols;
        let mut mean = vec![0.0f64; n_cols];
        for row in x.chunks_exact(n_cols) {
            for (m, v) in mean.iter_mut().zip(row) {
                *m += *v as f64;
            }
        }
        for m in &mut mean {
            *m /= n as f64;
        }
        let mut var = vec![0.0f64; n_cols];
        for row in x.chunks_exact(n_cols) {
            for (j, v) in row.iter().enumerate() {
                let d = *v as f64 - mean[j];
                var[j] += d * d;
            }
        }
        let std: Vec<f32> = var
            .iter()
            .map(|v| {
                let s = (v / n as f64).sqrt();
                if s < 1e-12 { 1.0 } else { s as f32 }
            })
            .collect();
        Scaler { mean: mean.into_iter().map(|m| m as f32).collect(), std }
    }

    pub fn transform(&self, x: &[f32]) -> Vec<f32> {
        let n_cols = self.mean.len();
        let mut out = Vec::with_capacity(x.len());
        for row in x.chunks_exact(n_cols) {
            for (j, v) in row.iter().enumerate() {
                out.push((v - self.mean[j]) / self.std[j]);
            }
        }
        out
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::write(path, serde_json::to_vec(self)?)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }
}
