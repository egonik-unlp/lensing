//! Preprocessing spec — the artifact that guarantees train/serve parity.
//!
//! The scattered SongAE re-derived z-score / log1p / multi-hot at fit time and
//! then again, by hand, to encode a brand-new song (`song_ae_preprocess.json`).
//! That duplication is the classic source of train/serve skew. Here the stats
//! (means, stds, vocabularies) and the block layout are computed once at `fit`,
//! persisted, and replayed identically by [`PreprocessSpec::encode_one`] for any
//! new item.

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::blocks::BlockLayout;

/// z-score stats for one numeric field (optionally log1p'd first).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NumericField {
    pub key: String,
    pub mean: f64,
    pub std: f64,
    #[serde(default)]
    pub log1p: bool,
}

/// z-score stats for one acoustic field; each also emits a missing-flag column.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AcousticField {
    pub key: String,
    pub mean: f64,
    pub std: f64,
}

/// A categorical field encoded as multi-hot (list-valued) or one-hot (scalar).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CategoricalField {
    pub key: String,
    pub vocab: Vec<String>,
    /// If true the payload value is a list (multi-hot); else a scalar (one-hot).
    pub multi: bool,
}

/// Everything needed to encode an item identically at fit and at serve time.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PreprocessSpec {
    /// Passthrough block width — a vector already stored with the item (e.g. the
    /// text/sentence embedding the SongAE reuses). 0 = no passthrough block.
    #[serde(default)]
    pub text_dim: usize,
    #[serde(default)]
    pub numerics: Vec<NumericField>,
    #[serde(default)]
    pub acoustics: Vec<AcousticField>,
    #[serde(default)]
    pub categoricals: Vec<CategoricalField>,
    /// Payload key holding the metadata object; SongAE nests fields under
    /// `metadata`. `None` reads fields off the payload root.
    #[serde(default)]
    pub metadata_key: Option<String>,
    /// Block spans over the assembled vector (for block-wise loss + R²).
    pub layout: BlockLayout,
    /// Provenance: the text embedding model, when there is a passthrough block.
    #[serde(default)]
    pub text_model: Option<String>,
}

fn zscore(x: f64, mean: f64, std: f64) -> f32 {
    let s = if std > 1e-9 { std } else { 1.0 };
    ((x - mean) / s) as f32
}

impl PreprocessSpec {
    /// Total assembled feature width: text + numeric + acoustic (value+flag) + categorical.
    pub fn feature_dim(&self) -> usize {
        self.text_dim
            + self.numerics.len()
            + self.acoustics.len() * 2
            + self.categoricals.iter().map(|c| c.vocab.len()).sum::<usize>()
    }

    /// Read the metadata object a field lives under (or the payload root).
    fn meta<'a>(&self, payload: &'a Value) -> &'a Value {
        match &self.metadata_key {
            Some(k) => payload.get(k).unwrap_or(payload),
            None => payload,
        }
    }

    /// Encode a single item to one assembled feature row, byte-for-byte the same
    /// arithmetic used at fit — this is what kills train/serve skew (REQ-7).
    ///
    /// `text` is the passthrough vector (empty if `text_dim == 0`); `payload` is
    /// the item's JSON payload.
    pub fn encode_one(&self, text: &[f32], payload: &Value) -> Result<Vec<f32>> {
        anyhow::ensure!(
            text.len() == self.text_dim,
            "passthrough vector has {} dims, spec expects {}",
            text.len(),
            self.text_dim
        );
        let m = self.meta(payload);
        let mut row = Vec::with_capacity(self.feature_dim());

        // text passthrough block
        row.extend_from_slice(text);

        // numeric block: (optional log1p) then z-score, NaN imputed to mean.
        for f in &self.numerics {
            let raw = m.get(&f.key).and_then(Value::as_f64);
            let v = match raw {
                Some(x) => {
                    let x = if f.log1p { if x >= 0.0 { (1.0 + x).ln() } else { f64::NAN } } else { x };
                    if x.is_finite() { x } else { f.mean }
                }
                None => f.mean,
            };
            row.push(zscore(v, f.mean, f.std));
        }

        // acoustic block: z-scored values, then a parallel missing-flag column
        // per field (matches build_song_ae.py's [values | flags] concat).
        let mut flags = Vec::with_capacity(self.acoustics.len());
        for f in &self.acoustics {
            let raw = m.get(&f.key).and_then(Value::as_f64);
            match raw {
                Some(x) if x.is_finite() => {
                    row.push(zscore(x, f.mean, f.std));
                    flags.push(0.0f32);
                }
                _ => {
                    row.push(zscore(f.mean, f.mean, f.std)); // imputed → 0 after z-score
                    flags.push(1.0f32);
                }
            }
        }
        row.extend_from_slice(&flags);

        // categorical block: multi-hot (list) or one-hot (scalar).
        for c in &self.categoricals {
            let mut hot = vec![0.0f32; c.vocab.len()];
            let idx = |val: &str| c.vocab.iter().position(|v| v == val);
            if c.multi {
                if let Some(arr) = m.get(&c.key).and_then(Value::as_array) {
                    for v in arr {
                        if let Some(s) = v.as_str() {
                            if let Some(i) = idx(s) {
                                hot[i] = 1.0;
                            }
                        }
                    }
                }
            } else if let Some(s) = m.get(&c.key).and_then(Value::as_str) {
                if let Some(i) = idx(s) {
                    hot[i] = 1.0;
                }
            }
            row.extend_from_slice(&hot);
        }

        Ok(row)
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir).ok();
        let path = dir.join("preprocess.json");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("preprocess.json");
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        Ok(serde_json::from_str(&text)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn spec() -> PreprocessSpec {
        PreprocessSpec {
            text_dim: 2,
            numerics: vec![NumericField { key: "pop".into(), mean: 10.0, std: 5.0, log1p: false }],
            acoustics: vec![AcousticField { key: "energy".into(), mean: 0.5, std: 0.25 }],
            categoricals: vec![CategoricalField {
                key: "genres".into(),
                vocab: vec!["rock".into(), "jazz".into()],
                multi: true,
            }],
            metadata_key: Some("metadata".into()),
            layout: BlockLayout::from_widths(&[("text", 2), ("numeric", 1), ("acoustic", 2), ("categorical", 2)]),
            text_model: None,
        }
    }

    #[test]
    fn feature_dim_matches_encode() {
        let s = spec();
        let payload = json!({"metadata": {"pop": 20.0, "energy": 0.75, "genres": ["rock"]}});
        let row = s.encode_one(&[0.1, 0.2], &payload).unwrap();
        assert_eq!(row.len(), s.feature_dim());
        assert_eq!(s.feature_dim(), 2 + 1 + 2 + 2);
        // numeric z-score: (20-10)/5 = 2.0
        assert!((row[2] - 2.0).abs() < 1e-6);
        // acoustic value present → flag 0; genres multi-hot rock=1 jazz=0
        assert_eq!(row[4], 0.0); // energy missing-flag
        assert_eq!(row[5], 1.0); // rock
        assert_eq!(row[6], 0.0); // jazz
    }

    #[test]
    fn missing_fields_impute_and_flag() {
        let s = spec();
        let payload = json!({"metadata": {}});
        let row = s.encode_one(&[0.0, 0.0], &payload).unwrap();
        assert_eq!(row[2], 0.0); // numeric imputed to mean → z-score 0
        assert_eq!(row[3], 0.0); // acoustic imputed to mean → z-score 0
        assert_eq!(row[4], 1.0); // acoustic missing-flag set
    }

    #[test]
    fn round_trip_json() {
        let dir = std::env::temp_dir().join("lensing_compression_preprocess_test");
        let s = spec();
        s.save(&dir).unwrap();
        let back = PreprocessSpec::load(&dir).unwrap();
        assert_eq!(s, back);
        std::fs::remove_dir_all(&dir).ok();
    }
}
