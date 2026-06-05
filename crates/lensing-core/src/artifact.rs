use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use anyhow::{ensure, Context, Result};

use crate::manifest::Manifest;
use crate::model::InputManifest;

/// A dataset directory loaded into memory.
pub struct Dataset {
    pub manifest: Manifest,
    /// Row-major `n_rows × n_cols`, all rows (train and test).
    pub features: Vec<f32>,
    /// Transformed target, length `n_rows`.
    pub target: Vec<f32>,
    /// Source point ids, length `n_rows`.
    pub row_ids: Vec<u64>,
    pub train_idx: Vec<u32>,
    pub test_idx: Vec<u32>,
}

impl Dataset {
    pub fn load(dir: &Path) -> Result<Self> {
        let manifest: Manifest = serde_json::from_reader(BufReader::new(
            File::open(dir.join("manifest.json")).context("open manifest.json")?,
        ))
        .context("parse manifest.json")?;

        let features = read_f32(&dir.join("features.f32"))?;
        let target = read_f32(&dir.join("target.f32"))?;
        let row_ids = read_u64(&dir.join("row_ids.u64"))?;
        let train_idx = read_u32(&dir.join("train_idx.u32"))?;
        let test_idx = read_u32(&dir.join("test_idx.u32"))?;

        ensure!(
            features.len() == manifest.n_rows * manifest.n_cols,
            "features.f32 has {} values, manifest says {}×{}",
            features.len(),
            manifest.n_rows,
            manifest.n_cols
        );
        ensure!(target.len() == manifest.n_rows, "target length mismatch");
        ensure!(row_ids.len() == manifest.n_rows, "row_ids length mismatch");
        ensure!(
            train_idx.len() == manifest.split.n_train && test_idx.len() == manifest.split.n_test,
            "split index length mismatch"
        );

        Ok(Self { manifest, features, target, row_ids, train_idx, test_idx })
    }

    pub fn row(&self, i: usize) -> &[f32] {
        let c = self.manifest.n_cols;
        &self.features[i * c..(i + 1) * c]
    }

    /// Gather `(features, target)` for a list of row indices.
    pub fn gather(&self, idx: &[u32]) -> (Vec<f32>, Vec<f32>) {
        let c = self.manifest.n_cols;
        let mut x = Vec::with_capacity(idx.len() * c);
        let mut y = Vec::with_capacity(idx.len());
        for &i in idx {
            x.extend_from_slice(self.row(i as usize));
            y.push(self.target[i as usize]);
        }
        (x, y)
    }
}

/// An inference input mini-artifact loaded into memory: `features.f32` +
/// `row_ids.u64` + a trimmed `manifest.json`. Written by lensing-server per
/// predict invocation; loaded by predictors. No target, no split.
pub struct InferenceInput {
    pub manifest: InputManifest,
    /// Row-major `n_rows × n_cols`.
    pub features: Vec<f32>,
    pub row_ids: Vec<u64>,
}

impl InferenceInput {
    pub fn load(dir: &Path) -> Result<Self> {
        let manifest: InputManifest = serde_json::from_reader(BufReader::new(
            File::open(dir.join("manifest.json")).context("open manifest.json")?,
        ))
        .context("parse manifest.json")?;

        let features = read_f32(&dir.join("features.f32"))?;
        let row_ids = read_u64(&dir.join("row_ids.u64"))?;

        ensure!(
            features.len() == manifest.n_rows * manifest.n_cols,
            "features.f32 has {} values, manifest says {}×{}",
            features.len(),
            manifest.n_rows,
            manifest.n_cols
        );
        ensure!(row_ids.len() == manifest.n_rows, "row_ids length mismatch");

        Ok(Self { manifest, features, row_ids })
    }

    pub fn row(&self, i: usize) -> &[f32] {
        let c = self.manifest.n_cols;
        &self.features[i * c..(i + 1) * c]
    }
}

macro_rules! rw_impl {
    ($read:ident, $write:ident, $ty:ty) => {
        pub fn $read(path: &Path) -> Result<Vec<$ty>> {
            const W: usize = std::mem::size_of::<$ty>();
            let mut buf = Vec::new();
            BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?)
                .read_to_end(&mut buf)?;
            ensure!(buf.len() % W == 0, "{}: size not a multiple of {W}", path.display());
            Ok(buf
                .chunks_exact(W)
                .map(|c| <$ty>::from_le_bytes(c.try_into().unwrap()))
                .collect())
        }

        pub fn $write(path: &Path, data: &[$ty]) -> Result<()> {
            let mut w = BufWriter::new(
                File::create(path).with_context(|| format!("create {}", path.display()))?,
            );
            for v in data {
                w.write_all(&v.to_le_bytes())?;
            }
            w.flush()?;
            Ok(())
        }
    };
}

rw_impl!(read_f32, write_f32, f32);
rw_impl!(read_u32, write_u32, u32);
rw_impl!(read_u64, write_u64, u64);
