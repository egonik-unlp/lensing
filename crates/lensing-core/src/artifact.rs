use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use anyhow::{ensure, Context, Result};

use crate::manifest::{Manifest, SequenceManifest};
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

/// A SEQUENCE dataset directory (ranking / next-item), loaded into memory.
/// Parallel to [`Dataset`]; keeps the pointwise flat-matrix contract untouched.
/// Files:
/// - `sequence-manifest.json` — the [`SequenceManifest`] descriptor.
/// - `sessions.u32` — concatenated ordered item-index sequences (all sessions).
/// - `offsets.u32` — `n_sessions + 1` prefix offsets; session `s` is
///   `sessions[offsets[s]..offsets[s+1]]`, its LAST item the next-item label.
/// - `item_latents.f32` — `[n_items × latent_dim]` frozen per-item latents.
/// - `train_sessions.u32` / `test_sessions.u32` — session-index membership.
/// The item vocabulary + display metadata live in `items.json` (loaded
/// separately by consumers that need identity fields).
pub struct SequenceDataset {
    pub manifest: SequenceManifest,
    pub sessions: Vec<u32>,
    pub offsets: Vec<u32>,
    pub item_latents: Vec<f32>,
    pub train_sessions: Vec<u32>,
    pub test_sessions: Vec<u32>,
}

impl SequenceDataset {
    pub fn load(dir: &Path) -> Result<Self> {
        let manifest: SequenceManifest = serde_json::from_reader(BufReader::new(
            File::open(dir.join("sequence-manifest.json")).context("open sequence-manifest.json")?,
        ))
        .context("parse sequence-manifest.json")?;

        let sessions = read_u32(&dir.join("sessions.u32"))?;
        let offsets = read_u32(&dir.join("offsets.u32"))?;
        let item_latents = read_f32(&dir.join("item_latents.f32"))?;
        let train_sessions = read_u32(&dir.join("train_sessions.u32"))?;
        let test_sessions = read_u32(&dir.join("test_sessions.u32"))?;

        ensure!(
            offsets.len() == manifest.n_sessions + 1,
            "offsets.u32 has {} entries, expected n_sessions+1 = {}",
            offsets.len(),
            manifest.n_sessions + 1
        );
        ensure!(
            offsets.last().copied().unwrap_or(0) as usize == sessions.len(),
            "offsets tail must equal sessions.u32 length"
        );
        ensure!(
            item_latents.len() == manifest.n_items * manifest.latent_dim,
            "item_latents.f32 has {} values, manifest says {}×{}",
            item_latents.len(),
            manifest.n_items,
            manifest.latent_dim
        );
        ensure!(
            train_sessions.len() == manifest.split.n_train_sessions
                && test_sessions.len() == manifest.split.n_test_sessions,
            "session split length mismatch"
        );
        ensure!(
            sessions.iter().all(|&i| (i as usize) < manifest.n_items),
            "sessions.u32 contains an item id outside the vocabulary"
        );
        for s in 0..manifest.n_sessions {
            let (a, b) = (offsets[s] as usize, offsets[s + 1] as usize);
            ensure!(b >= a + 2, "session {s} has fewer than 2 items (need prefix + label)");
        }

        Ok(Self { manifest, sessions, offsets, item_latents, train_sessions, test_sessions })
    }

    /// Ordered item-index slice for session `s`; the last element is the label.
    pub fn session(&self, s: usize) -> &[u32] {
        &self.sessions[self.offsets[s] as usize..self.offsets[s + 1] as usize]
    }

    /// The `latent_dim` latent vector for item index `item`.
    pub fn item_latent(&self, item: usize) -> &[f32] {
        let d = self.manifest.latent_dim;
        &self.item_latents[item * d..(item + 1) * d]
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

#[cfg(test)]
mod seq_tests {
    use super::*;
    use crate::manifest::{SequenceManifest, SequenceSplit};

    #[test]
    fn sequence_dataset_round_trips() {
        let dir = std::env::temp_dir().join("lensing_seqds_fixture");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // 3 sessions over a 5-item vocab, latent_dim 2.
        let sessions: Vec<u32> = vec![0, 1, 2, /*|*/ 3, 4, /*|*/ 1, 0, 3, 4];
        let offsets: Vec<u32> = vec![0, 3, 5, 9];
        let item_latents: Vec<f32> = (0..10).map(|i| i as f32).collect(); // 5×2
        let train_sessions: Vec<u32> = vec![0, 1];
        let test_sessions: Vec<u32> = vec![2];
        write_u32(&dir.join("sessions.u32"), &sessions).unwrap();
        write_u32(&dir.join("offsets.u32"), &offsets).unwrap();
        write_f32(&dir.join("item_latents.f32"), &item_latents).unwrap();
        write_u32(&dir.join("train_sessions.u32"), &train_sessions).unwrap();
        write_u32(&dir.join("test_sessions.u32"), &test_sessions).unwrap();
        let manifest = SequenceManifest {
            dataset_id: "seq-test".into(),
            created_at: "2026-01-01T00:00:00Z".into(),
            kind: "sequence".into(),
            n_sessions: 3,
            n_items: 5,
            latent_dim: 2,
            latent_source: "item_latents".into(),
            split: SequenceSplit {
                strategy: "leave_last_out".into(),
                cut: None,
                n_train_sessions: 2,
                n_test_sessions: 1,
            },
        };
        std::fs::write(dir.join("sequence-manifest.json"), serde_json::to_vec(&manifest).unwrap())
            .unwrap();

        let ds = SequenceDataset::load(&dir).unwrap();
        assert_eq!(ds.session(2), &[1, 0, 3, 4]); // prefix [1,0,3] → label 4
        assert_eq!(ds.session(0), &[0, 1, 2]);
        assert_eq!(ds.item_latent(3), &[6.0, 7.0]);
        assert_eq!(ds.train_sessions, vec![0, 1]);
        assert_eq!(ds.test_sessions, vec![2]);

        // a corrupt manifest (wrong n_sessions) must be rejected
        let mut bad = manifest.clone();
        bad.n_sessions = 4;
        std::fs::write(dir.join("sequence-manifest.json"), serde_json::to_vec(&bad).unwrap())
            .unwrap();
        assert!(SequenceDataset::load(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
