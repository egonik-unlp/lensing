//! PCA via covariance eigendecomposition, pure Rust (nalgebra, no LAPACK).
//! Fit on train rows only; f64 internally, f32 in artifacts.
//!
//! This module was hoisted out of `lensing-pipeline` into `lensing-compression`
//! so PCA is one implementation of the shared [`crate::Compressor`] abstraction
//! alongside the autoencoder. `lensing-pipeline` re-exports it, so the artifact
//! format (`mean` + row-major `components` f32) and every consumer stay unchanged.

use anyhow::{ensure, Result};
use nalgebra::{DMatrix, DVector, SymmetricEigen};

pub struct PcaModel {
    /// Length d (original dim).
    pub mean: Vec<f64>,
    /// k × d, row-major: each row is one principal component.
    pub components: DMatrix<f64>,
    pub explained_variance_ratio: Vec<f32>,
}

/// The full eigen-spectrum of the (train-split) covariance. Used to preview
/// how much variance any number of PCA dims would capture, without building
/// the components matrix.
pub struct Spectrum {
    /// All d eigenvalues, clamped at 0, in descending order of raw value.
    pub eigenvalues_desc: Vec<f64>,
    /// Sum of clamped eigenvalues (the normalizer for EVR).
    pub total_var: f64,
}

impl Spectrum {
    /// Per-component explained-variance ratio. Computed with the exact same
    /// arithmetic as [`PcaModel::explained_variance_ratio`] in [`fit`], so
    /// `spectrum.evr()[..k]` equals a `k`-dim fit's EVR element-for-element.
    pub fn evr(&self) -> Vec<f32> {
        self.eigenvalues_desc
            .iter()
            .map(|&v| (v / self.total_var) as f32)
            .collect()
    }

    /// Running sum of [`Spectrum::evr`].
    pub fn cumulative_evr(&self) -> Vec<f32> {
        let mut acc = 0.0f32;
        self.evr()
            .into_iter()
            .map(|v| {
                acc += v;
                acc
            })
            .collect()
    }
}

const CHUNK: usize = 2048;

/// Mean and d×d covariance over the selected `rows` of a row-major n×d matrix.
fn mean_and_covariance(vectors: &[f32], d: usize, rows: &[u32]) -> Result<(Vec<f64>, DMatrix<f64>)> {
    let n = rows.len();
    ensure!(n > 1, "need at least 2 rows to fit PCA");

    // Mean over selected rows.
    let mut mean = vec![0.0f64; d];
    for &r in rows {
        let row = &vectors[r as usize * d..(r as usize + 1) * d];
        for (m, v) in mean.iter_mut().zip(row) {
            *m += *v as f64;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }

    // Covariance d×d, accumulated chunk by chunk: C += Xcᵀ·Xc.
    let mut cov = DMatrix::<f64>::zeros(d, d);
    for chunk in rows.chunks(CHUNK) {
        let mut xc = DMatrix::<f64>::zeros(chunk.len(), d);
        for (ci, &r) in chunk.iter().enumerate() {
            let row = &vectors[r as usize * d..(r as usize + 1) * d];
            for j in 0..d {
                xc[(ci, j)] = row[j] as f64 - mean[j];
            }
        }
        cov += xc.tr_mul(&xc);
    }
    cov /= (n - 1) as f64;
    Ok((mean, cov))
}

/// The full explained-variance curve without building components: covariance
/// eigendecomposition, eigenvalues clamped and sorted descending. `rows` must
/// be the same train split a later `fit` uses for the curves to match.
pub fn eigen_spectrum(vectors: &[f32], d: usize, rows: &[u32]) -> Result<Spectrum> {
    let (_mean, cov) = mean_and_covariance(vectors, d, rows)?;
    let eig = SymmetricEigen::new(cov);
    let total_var: f64 = eig.eigenvalues.iter().map(|v| v.max(0.0)).sum();
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
    let eigenvalues_desc = order.iter().map(|&i| eig.eigenvalues[i].max(0.0)).collect();
    Ok(Spectrum { eigenvalues_desc, total_var })
}

/// `vectors` is row-major n × d. `rows` selects the rows to fit on.
pub fn fit(vectors: &[f32], d: usize, rows: &[u32], k: usize) -> Result<PcaModel> {
    Ok(fit_with_spectrum(vectors, d, rows, k)?.0)
}

/// Like [`fit`], but also returns the full eigen-spectrum from the same single
/// eigendecomposition — so a build can persist the cumulative-variance curve
/// without paying for a second decomposition.
pub fn fit_with_spectrum(
    vectors: &[f32],
    d: usize,
    rows: &[u32],
    k: usize,
) -> Result<(PcaModel, Spectrum)> {
    ensure!(k >= 1 && k <= d, "pca dims {k} out of range 1..={d}");
    let (mean, cov) = mean_and_covariance(vectors, d, rows)?;

    let eig = SymmetricEigen::new(cov);
    let total_var: f64 = eig.eigenvalues.iter().map(|v| v.max(0.0)).sum();

    // Sort eigenpairs by eigenvalue descending, take top k.
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));

    let mut components = DMatrix::<f64>::zeros(k, d);
    let mut evr = Vec::with_capacity(k);
    for (out_row, &src) in order[..k].iter().enumerate() {
        let col = eig.eigenvectors.column(src);
        // Sign convention: make the largest-magnitude entry positive, so
        // re-fits are reproducible across eigensolvers.
        let (mut max_abs, mut sign) = (0.0f64, 1.0f64);
        for v in col.iter() {
            if v.abs() > max_abs {
                max_abs = v.abs();
                sign = if *v < 0.0 { -1.0 } else { 1.0 };
            }
        }
        for j in 0..d {
            components[(out_row, j)] = col[j] * sign;
        }
        evr.push((eig.eigenvalues[src].max(0.0) / total_var) as f32);
    }

    let spectrum = Spectrum {
        eigenvalues_desc: order.iter().map(|&i| eig.eigenvalues[i].max(0.0)).collect(),
        total_var,
    };
    Ok((PcaModel { mean, components, explained_variance_ratio: evr }, spectrum))
}

impl PcaModel {
    /// Reconstruct a model from stored f32 artifacts (`manifest.pca.mean` +
    /// `pca_components.f32`). The builder also routes its freshly-fit model
    /// through this so training and inference project with identical
    /// (f32-quantized) parameters.
    pub fn from_parts(
        mean: &[f32],
        components_row_major: &[f32],
        k: usize,
        d: usize,
        explained_variance_ratio: Vec<f32>,
    ) -> Result<Self> {
        ensure!(mean.len() == d, "pca mean has {} values, expected {d}", mean.len());
        ensure!(
            components_row_major.len() == k * d,
            "pca components have {} values, expected {k}×{d}",
            components_row_major.len()
        );
        let comp_f64: Vec<f64> = components_row_major.iter().map(|v| *v as f64).collect();
        Ok(PcaModel {
            mean: mean.iter().map(|v| *v as f64).collect(),
            components: DMatrix::from_row_slice(k, d, &comp_f64),
            explained_variance_ratio,
        })
    }

    pub fn k(&self) -> usize {
        self.components.nrows()
    }

    /// Project all n rows of `vectors` (row-major n × d) to n × k (row-major f32).
    pub fn project_all(&self, vectors: &[f32], d: usize) -> Vec<f32> {
        let n = vectors.len() / d;
        let k = self.k();
        let mean = DVector::from_column_slice(&self.mean);
        let comp_t = self.components.transpose(); // d × k
        let mut out = Vec::with_capacity(n * k);
        for start in (0..n).step_by(CHUNK) {
            let end = (start + CHUNK).min(n);
            let mut xc = DMatrix::<f64>::zeros(end - start, d);
            for ci in 0..(end - start) {
                let row = &vectors[(start + ci) * d..(start + ci + 1) * d];
                for j in 0..d {
                    xc[(ci, j)] = row[j] as f64 - mean[j];
                }
            }
            let proj = xc * &comp_t; // chunk × k
            for ci in 0..proj.nrows() {
                for j in 0..k {
                    out.push(proj[(ci, j)] as f32);
                }
            }
        }
        out
    }

    /// Reconstruct the original space from projected latents (row-major n × k
    /// → n × d): `x̂ = z · components + mean`. Used by the [`crate::Compressor`]
    /// impl; PCA reconstruction is exact up to the truncated components.
    pub fn reconstruct_all(&self, latents: &[f32], k: usize) -> Vec<f32> {
        let n = latents.len() / k;
        let d = self.mean.len();
        let mut out = Vec::with_capacity(n * d);
        for i in 0..n {
            let z = &latents[i * k..(i + 1) * k];
            for j in 0..d {
                let mut acc = self.mean[j];
                for (c, &zc) in z.iter().enumerate() {
                    acc += zc as f64 * self.components[(c, j)];
                }
                out.push(acc as f32);
            }
        }
        out
    }
}

// ----------------------------------------------------------------------------
// PCA as a `Compressor`.
// ----------------------------------------------------------------------------

use std::path::Path;

use crate::Compressor;

/// PCA wrapped as a [`Compressor`]: the linear compressor. Configured with a
/// target latent dim `k`; `fit` populates `model`.
pub struct Pca {
    k: usize,
    d: usize,
    model: Option<PcaModel>,
    spectrum: Option<Spectrum>,
}

impl Pca {
    pub fn new(k: usize) -> Self {
        Pca { k, d: 0, model: None, spectrum: None }
    }

    /// The fitted model, once `fit` has run.
    pub fn model(&self) -> Option<&PcaModel> {
        self.model.as_ref()
    }

    fn fitted(&self) -> Result<&PcaModel> {
        self.model.as_ref().ok_or_else(|| anyhow::anyhow!("PCA not fitted"))
    }
}

impl Compressor for Pca {
    fn fit(&mut self, x: &[f32], d: usize, rows: &[u32]) -> Result<()> {
        self.d = d;
        let (model, spectrum) = fit_with_spectrum(x, d, rows, self.k)?;
        self.model = Some(model);
        self.spectrum = Some(spectrum);
        Ok(())
    }

    fn encode(&self, x: &[f32], d: usize) -> Result<Vec<f32>> {
        Ok(self.fitted()?.project_all(x, d))
    }

    fn reconstruct(&self, z: &[f32]) -> Result<Vec<f32>> {
        Ok(self.fitted()?.reconstruct_all(z, self.k))
    }

    fn latent_dim(&self) -> usize {
        self.k
    }

    fn quality(&self) -> serde_json::Value {
        match (&self.model, &self.spectrum) {
            (Some(m), Some(s)) => serde_json::json!({
                "kind": "evr",
                "evr": m.explained_variance_ratio,
                "cumulative_evr": s.cumulative_evr(),
                "captured": s.cumulative_evr().get(self.k.saturating_sub(1)).copied().unwrap_or(0.0),
            }),
            _ => serde_json::Value::Null,
        }
    }

    /// Persist in the existing artifact format: `pca.mean` (f32) + row-major
    /// `pca_components.f32` + the EVR curve — so `featurize.json` consumers and
    /// `PcaModel::from_parts` keep working unchanged.
    fn save(&self, dir: &Path) -> Result<()> {
        let m = self.fitted()?;
        std::fs::create_dir_all(dir).ok();
        let mean: Vec<f32> = m.mean.iter().map(|v| *v as f32).collect();
        // components row-major k×d: iterate rows then cols.
        let comp_row_major: Vec<f32> = (0..m.k())
            .flat_map(|r| (0..m.mean.len()).map(move |c| (r, c)))
            .map(|(r, c)| m.components[(r, c)] as f32)
            .collect();
        write_f32(&dir.join("pca_mean.f32"), &mean)?;
        write_f32(&dir.join("pca_components.f32"), &comp_row_major)?;
        let evr = serde_json::to_string(&m.explained_variance_ratio)?;
        std::fs::write(dir.join("pca_evr.json"), evr)?;
        Ok(())
    }

    fn export_encoder_onnx(&self, dir: &Path) -> Result<()> {
        let m = self.fitted()?;
        let d = m.mean.len();
        let k = m.k();
        let mean: Vec<f32> = m.mean.iter().map(|v| *v as f32).collect();
        // components_t: row-major [d, k], element (i,j) = components[j,i].
        let mut comp_t = Vec::with_capacity(d * k);
        for i in 0..d {
            for j in 0..k {
                comp_t.push(m.components[(j, i)] as f32);
            }
        }
        crate::onnx::export_pca_encoder(dir, &mean, comp_t, d, k)
    }
}

fn write_f32(path: &Path, data: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lensing_core::shuffle::SplitMix64;

    /// PCA of points along a known direction recovers that direction.
    #[test]
    fn recovers_dominant_direction() {
        let d = 4;
        let dir = [0.5f32, 0.5, 0.5, 0.5];
        let mut vectors = Vec::new();
        let mut rng = SplitMix64(7);
        let n = 200;
        for _ in 0..n {
            let t = (rng.next_u64() % 1000) as f32 / 100.0 - 5.0;
            let noise = (rng.next_u64() % 100) as f32 / 1000.0;
            for (j, &dj) in dir.iter().enumerate() {
                vectors.push(t * dj + if j == 0 { noise } else { 0.0 });
            }
        }
        let rows: Vec<u32> = (0..n as u32).collect();
        let model = fit(&vectors, d, &rows, 2).unwrap();
        // First component ≈ ±dir (normalized): dot with dir should be ≈ ±1.
        let dot: f64 = (0..d).map(|j| model.components[(0, j)] * dir[j] as f64).sum();
        assert!(dot.abs() > 0.99, "dot = {dot}");
        assert!(model.explained_variance_ratio[0] > 0.95);
        // Projection has the right shape.
        let proj = model.project_all(&vectors, d);
        assert_eq!(proj.len(), n * 2);
    }

    /// The analyze preview (`eigen_spectrum`) must agree with what a build of
    /// the same dims would record, element-for-element.
    #[test]
    fn spectrum_evr_matches_fit() {
        let d = 6;
        let mut vectors = Vec::new();
        let mut rng = SplitMix64(11);
        let n = 300;
        for _ in 0..n {
            for _ in 0..d {
                vectors.push((rng.next_u64() % 1000) as f32 / 100.0);
            }
        }
        let rows: Vec<u32> = (0..n as u32).collect();
        let k = 4;
        let model = fit(&vectors, d, &rows, k).unwrap();
        let spectrum = eigen_spectrum(&vectors, d, &rows).unwrap();
        let evr = spectrum.evr();
        assert_eq!(&evr[..k], &model.explained_variance_ratio[..]);
        // Cumulative curve is non-decreasing and approaches 1.0.
        let cum = spectrum.cumulative_evr();
        assert!((cum[d - 1] - 1.0).abs() < 1e-5, "cum tail = {}", cum[d - 1]);
    }
}
