//! Metadata feature encoding: numeric columns + frozen one-hot vocabularies,
//! driven by the domain's field descriptors at build time and rebuilt purely
//! from frozen column descriptors at inference time (no domain needed — the
//! contract is self-describing).

use std::collections::HashMap;

use anyhow::{ensure, Result};
use lensing_core::domain::{Domain, NumericEncode};
use lensing_core::{ColumnDesc, ColumnKind, FeatureConfig};

use crate::qdrant::RawPoint;

/// A frozen categorical vocabulary: values in fixed order, plus an
/// optional trailing "other" bucket for values outside the list.
pub struct Vocab {
    pub group: String,
    pub values: Vec<String>,
    pub has_other: bool,
}

impl Vocab {
    /// All distinct values, sorted alphabetically (small cardinality fields).
    fn all(group: &str, iter: impl Iterator<Item = String>) -> Self {
        let mut values: Vec<String> = iter
            .filter(|v| !v.trim().is_empty())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        values.sort();
        Vocab { group: group.to_string(), values, has_other: true }
    }

    /// Top-N by frequency (ties alphabetical) + "other" bucket.
    fn top_n(group: &str, iter: impl Iterator<Item = String>, n: usize) -> Self {
        let mut freq: HashMap<String, usize> = HashMap::new();
        for v in iter {
            if !v.trim().is_empty() {
                *freq.entry(v).or_default() += 1;
            }
        }
        let mut pairs: Vec<(String, usize)> = freq.into_iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        pairs.truncate(n);
        Vocab {
            group: group.to_string(),
            values: pairs.into_iter().map(|(v, _)| v).collect(),
            has_other: true,
        }
    }

    pub fn width(&self) -> usize {
        self.values.len() + self.has_other as usize
    }

    /// One-hot encode `value` into `out` (must be `width()` long, zeroed).
    fn encode(&self, value: &str, out: &mut [f32]) {
        match self.values.iter().position(|v| v == value) {
            Some(i) => out[i] = 1.0,
            None => {
                if self.has_other {
                    *out.last_mut().unwrap() = 1.0;
                }
            }
        }
    }

    fn columns(&self) -> impl Iterator<Item = ColumnDesc> + '_ {
        self.values
            .iter()
            .cloned()
            .chain(self.has_other.then(|| "__other__".to_string()))
            .map(|value| ColumnDesc {
                name: format!("{}={}", self.group, value),
                kind: ColumnKind::Onehot { group: self.group.clone(), value },
            })
    }
}

/// One numeric feature column: how to read a payload field into a number.
struct NumericCol {
    /// Column (and manifest) name.
    name: String,
    /// Source payload field.
    field: String,
    /// Feature block (the domain field's toggle group, or
    /// [`COORDINATES_BLOCK`]); recorded in the emitted [`ColumnDesc`].
    group: Option<String>,
    read: NumericRead,
}

enum NumericRead {
    /// The field value verbatim (absent → 0). Original `bedrooms` semantics:
    /// a value column with no `_missing` partner.
    Verbatim,
    /// `ln(1+x)` of present (> 0) values, else 0.
    Log1p,
    /// Present (> 0) values verbatim, else 0 (the `_missing` partner carries
    /// absence).
    PresentRaw,
    /// 1.0 when the field is absent/zero.
    MissingFlag,
    /// Latitude / longitude of a bounded geo pair, else 0.
    Lat([[f64; 2]; 2]),
    Lon([[f64; 2]; 2]),
    /// 1.0 when the geo pair is absent or out of bounds.
    CoordsMissing([[f64; 2]; 2]),
}

/// The metadata feature encoder: numeric fields + one-hot vocabularies,
/// in frozen column order. PCA columns are prepended by the builder.
pub struct Encoder {
    numeric: Vec<NumericCol>,
    vocabs: Vec<Vocab>,
}

/// Block name recorded on the lat/lon/missing column triple (the
/// `FeatureConfig.coordinates` toggle's block, independent of the geo
/// field's name).
const COORDINATES_BLOCK: &str = "coordinates";

/// Legacy coordinate column names (the original domain's coordinates field).
/// Fields named "coordinates" keep emitting these so old and new artifacts
/// agree; differently-named geo fields get `<name>_lat` / `<name>_lon` /
/// `<name>_coords_missing`.
fn coord_col_names(field: &str) -> (String, String, String) {
    if field == "coordinates" {
        ("lat".into(), "lon".into(), "coords_missing".into())
    } else {
        (format!("{field}_lat"), format!("{field}_lon"), format!("{field}_coords_missing"))
    }
}

impl Encoder {
    /// Build the encoder from the domain's field descriptors and the build's
    /// feature config: numerics first, then categoricals, each in domain
    /// declaration order.
    pub fn build(points: &[RawPoint], domain: &Domain, cfg: &FeatureConfig) -> Self {
        let mut numeric: Vec<NumericCol> = Vec::new();
        let bounds = domain.coordinate_bounds_arr();
        for f in domain.numeric_fields() {
            if !domain.field_enabled(cfg, f) {
                continue;
            }
            // The domain field's toggle group, recorded on every emitted
            // column so block-level masks resolve from the artifact alone.
            let group = f.group.clone();
            match f.encode {
                NumericEncode::Log1p => numeric.push(NumericCol {
                    name: format!("{}_log", f.name),
                    field: f.name.clone(),
                    group: group.clone(),
                    read: NumericRead::Log1p,
                }),
                NumericEncode::Raw if f.indicator => numeric.push(NumericCol {
                    name: f.name.clone(),
                    field: f.name.clone(),
                    group: group.clone(),
                    read: NumericRead::PresentRaw,
                }),
                NumericEncode::Raw => numeric.push(NumericCol {
                    name: f.name.clone(),
                    field: f.name.clone(),
                    group: group.clone(),
                    read: NumericRead::Verbatim,
                }),
            }
            // Imputation fills the values at build/predict time, so the
            // indicator columns are dropped ("impute instead of
            // zero+indicator" — kernels can't branch on indicators).
            if f.indicator && !cfg.impute_numerics {
                numeric.push(NumericCol {
                    name: format!("{}_missing", f.name),
                    field: f.name.clone(),
                    group,
                    read: NumericRead::MissingFlag,
                });
            }
        }
        if let Some(f) = domain.coordinates_field() {
            if domain.field_enabled(cfg, f) {
                let (lat, lon, miss) = coord_col_names(&f.name);
                let group = Some(COORDINATES_BLOCK.to_string());
                numeric.push(NumericCol {
                    name: lat,
                    field: f.name.clone(),
                    group: group.clone(),
                    read: NumericRead::Lat(bounds),
                });
                numeric.push(NumericCol {
                    name: lon,
                    field: f.name.clone(),
                    group: group.clone(),
                    read: NumericRead::Lon(bounds),
                });
                numeric.push(NumericCol {
                    name: miss,
                    field: f.name.clone(),
                    group,
                    read: NumericRead::CoordsMissing(bounds),
                });
            }
        }

        let mut vocabs = Vec::new();
        for f in domain.categorical_fields() {
            if !domain.field_enabled(cfg, f) {
                continue;
            }
            let values = points.iter().map(|p| p.payload.str_of(&f.name));
            vocabs.push(match domain.effective_top_n(cfg, f) {
                Some(n) => Vocab::top_n(&f.name, values, n),
                None => Vocab::all(&f.name, values),
            });
        }
        Encoder { numeric, vocabs }
    }

    /// Rebuild the encoder from a frozen manifest column list (the inference
    /// path — no domain needed; the read semantics derive from the column
    /// structure exactly as `build` emitted it). PCA columns are skipped;
    /// consecutive one-hot columns regroup into vocabularies, a trailing
    /// `__other__` value marking the catch-all bucket.
    pub fn from_columns(columns: &[ColumnDesc]) -> Result<Self> {
        Self::from_columns_with_bounds(columns, lensing_core::manifest::legacy_coordinate_bounds())
    }

    /// Like [`Encoder::from_columns`], with explicit coordinate bounds (new
    /// contracts record them in `feature_config.coordinate_bounds`).
    pub fn from_columns_with_bounds(
        columns: &[ColumnDesc],
        bounds: [[f64; 2]; 2],
    ) -> Result<Self> {
        // First pass: collect numeric column names to detect _missing partners.
        let numeric_names: Vec<&str> = columns
            .iter()
            .filter(|c| matches!(c.kind, ColumnKind::Numeric { .. }))
            .map(|c| c.name.as_str())
            .collect();
        let has = |name: &str| numeric_names.contains(&name);

        let mut numeric: Vec<NumericCol> = Vec::new();
        let mut vocabs: Vec<Vocab> = Vec::new();
        for c in columns {
            match &c.kind {
                ColumnKind::Pca { .. } => {}
                ColumnKind::Numeric { field, group } => {
                    let name = c.name.clone();
                    // `group` passes through verbatim (None on pre-group
                    // artifacts) so a rebuilt encoder re-emits the columns
                    // exactly as frozen.
                    let group = group.clone();
                    let read = if let Some(src) = coord_source(&name, &numeric_names) {
                        match coord_kind(&name) {
                            CoordPart::Lat => NumericRead::Lat(bounds),
                            CoordPart::Lon => NumericRead::Lon(bounds),
                            CoordPart::Missing => NumericRead::CoordsMissing(bounds),
                        }
                        .with_field(&mut numeric, name.clone(), src, group);
                        continue;
                    } else if let Some(src) = name.strip_suffix("_log") {
                        NumericCol {
                            name: name.clone(),
                            field: src.to_string(),
                            group,
                            read: NumericRead::Log1p,
                        }
                    } else if let Some(src) = name.strip_suffix("_missing") {
                        NumericCol {
                            name: name.clone(),
                            field: src.to_string(),
                            group,
                            read: NumericRead::MissingFlag,
                        }
                    } else if has(&format!("{name}_missing")) {
                        NumericCol {
                            name: name.clone(),
                            field: field.clone(),
                            group,
                            read: NumericRead::PresentRaw,
                        }
                    } else {
                        NumericCol {
                            name: name.clone(),
                            field: field.clone(),
                            group,
                            read: NumericRead::Verbatim,
                        }
                    };
                    numeric.push(read);
                }
                ColumnKind::Onehot { group, value } => {
                    if vocabs.last().map(|v| v.group.as_str()) != Some(group.as_str()) {
                        vocabs.push(Vocab {
                            group: group.clone(),
                            values: Vec::new(),
                            has_other: false,
                        });
                    }
                    let v = vocabs.last_mut().unwrap();
                    if value == "__other__" {
                        v.has_other = true;
                    } else {
                        v.values.push(value.clone());
                    }
                }
            }
        }
        let enc = Encoder { numeric, vocabs };

        // The encoder must reproduce the trained column order exactly; the
        // scaler and the model weights are positional. Verify round-trip.
        let rebuilt: Vec<String> = enc.columns().into_iter().map(|c| c.name).collect();
        let expected: Vec<String> = columns
            .iter()
            .filter(|c| !matches!(c.kind, ColumnKind::Pca { .. }))
            .map(|c| c.name.clone())
            .collect();
        ensure!(
            rebuilt == expected,
            "encoder rebuilt from columns does not reproduce the trained column order"
        );
        Ok(enc)
    }

    pub fn width(&self) -> usize {
        self.numeric.len() + self.vocabs.iter().map(Vocab::width).sum::<usize>()
    }

    pub fn columns(&self) -> Vec<ColumnDesc> {
        let mut cols: Vec<ColumnDesc> = self
            .numeric
            .iter()
            .map(|c| ColumnDesc {
                name: c.name.clone(),
                kind: ColumnKind::Numeric { field: c.field.clone(), group: c.group.clone() },
            })
            .collect();
        for v in &self.vocabs {
            cols.extend(v.columns());
        }
        cols
    }

    /// Encode one point's metadata features into `out` (length `width()`, zeroed).
    pub fn encode(&self, p: &RawPoint, out: &mut [f32]) {
        // Zero means unspecified (scraper convention).
        let val = |field: &str| p.payload.num_of(field).filter(|x| *x > 0.0);
        let mut off = 0;
        for c in &self.numeric {
            out[off] = match &c.read {
                NumericRead::Verbatim => p.payload.num_of(&c.field).unwrap_or(0.0) as f32,
                NumericRead::Log1p => val(&c.field).map_or(0.0, |x| x.ln_1p() as f32),
                NumericRead::PresentRaw => val(&c.field).map_or(0.0, |x| x as f32),
                NumericRead::MissingFlag => val(&c.field).is_none() as u8 as f32,
                // Raw degrees; missing pairs encode as (0, 0) + indicator.
                NumericRead::Lat(b) => {
                    p.payload.coords_of(&c.field, b).map_or(0.0, |(lat, _)| lat as f32)
                }
                NumericRead::Lon(b) => {
                    p.payload.coords_of(&c.field, b).map_or(0.0, |(_, lon)| lon as f32)
                }
                NumericRead::CoordsMissing(b) => {
                    p.payload.coords_of(&c.field, b).is_none() as u8 as f32
                }
            };
            off += 1;
        }
        for v in &self.vocabs {
            let value = p.payload.str_of(&v.group);
            v.encode(&value, &mut out[off..off + v.width()]);
            off += v.width();
        }
        debug_assert_eq!(off, self.width());
    }

    /// Per-point inference warnings: empty or out-of-vocabulary categoricals
    /// that will land in the trained `__other__` bucket. Mirrors what
    /// training did with such values; informational, never an error.
    pub fn category_warnings(&self, p: &RawPoint, label: &str) -> Vec<String> {
        let mut out = Vec::new();
        for v in &self.vocabs {
            let value = p.payload.str_of(&v.group);
            if value.trim().is_empty() {
                out.push(format!("{label}: {} is empty, using __other__ bucket", v.group));
            } else if !v.values.iter().any(|x| *x == value) {
                out.push(format!(
                    "{label}: {} {value:?} not seen at training, using __other__ bucket",
                    v.group
                ));
            }
        }
        out
    }

    /// The vocabulary groups this encoder consumes (e.g. propertyType).
    pub fn categorical_groups(&self) -> Vec<String> {
        self.vocabs.iter().map(|v| v.group.clone()).collect()
    }

    /// The numeric payload fields this encoder consumes, by their *source*
    /// field name (a `_log`/`_missing` column pair both read one field),
    /// deduplicated in column order.
    pub fn numeric_fields(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for c in &self.numeric {
            if !out.contains(&c.field) {
                out.push(c.field.clone());
            }
        }
        out
    }

    /// Source fields of the value+indicator numeric pairs (the imputable /
    /// reconciled set), deduplicated in column order.
    pub fn indicator_fields(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for c in &self.numeric {
            if matches!(c.read, NumericRead::Log1p | NumericRead::PresentRaw | NumericRead::MissingFlag)
                && !out.contains(&c.field)
            {
                out.push(c.field.clone());
            }
        }
        out
    }

    /// The geo source field, if the encoder has coordinate columns.
    pub fn coordinates_field(&self) -> Option<&str> {
        self.numeric.iter().find_map(|c| match c.read {
            NumericRead::Lat(_) => Some(c.field.as_str()),
            _ => None,
        })
    }
}

enum CoordPart {
    Lat,
    Lon,
    Missing,
}

impl NumericRead {
    fn with_field(
        self,
        numeric: &mut Vec<NumericCol>,
        name: String,
        field: String,
        group: Option<String>,
    ) {
        numeric.push(NumericCol { name, field, group, read: self });
    }
}

/// Detect a coordinate column and return its source field. Legacy names
/// ("lat"/"lon"/"coords_missing") read the original "coordinates" field;
/// generic names are `<field>_lat` / `<field>_lon` / `<field>_coords_missing`.
fn coord_source(name: &str, _all: &[&str]) -> Option<String> {
    match name {
        "lat" | "lon" | "coords_missing" => Some("coordinates".to_string()),
        _ => name
            .strip_suffix("_lat")
            .or_else(|| name.strip_suffix("_lon"))
            .or_else(|| name.strip_suffix("_coords_missing"))
            .map(str::to_string),
    }
}

fn coord_kind(name: &str) -> CoordPart {
    if name == "lat" || name.ends_with("_lat") {
        CoordPart::Lat
    } else if name == "lon" || name.ends_with("_lon") {
        CoordPart::Lon
    } else {
        CoordPart::Missing
    }
}

/// Assemble the full feature matrix in trained column order: PCA columns
/// first (`projected`, n × k), then the metadata encoding. Shared by the
/// dataset builder and the inference featurizer so the two paths cannot
/// drift apart.
pub fn assemble(projected: &[f32], k: usize, encoder: &Encoder, points: &[RawPoint]) -> Vec<f32> {
    let n = points.len();
    let n_cols = k + encoder.width();
    let mut features = vec![0.0f32; n * n_cols];
    for (i, p) in points.iter().enumerate() {
        let row = &mut features[i * n_cols..(i + 1) * n_cols];
        row[..k].copy_from_slice(&projected[i * k..(i + 1) * k]);
        encoder.encode(p, &mut row[k..]);
    }
    features
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qdrant::Payload;
    use lensing_core::domain::Domain;
    use serde_json::json;

    fn pt(total_area: Option<f64>, bathrooms: Option<f64>) -> RawPoint {
        RawPoint {
            id: 0,
            vector: vec![],
            payload: Payload::from_raw(
                json!({
                    "content": "",
                    "metadata": {
                        "propertyType": "house",
                        "bedrooms": 2.0,
                        "totalArea": total_area,
                        "bathrooms": bathrooms,
                    }
                }),
                "metadata",
            ),
        }
    }

    /// Feature config enabling exactly the legacy default + raw numerics.
    fn raw_numerics_cfg(impute: bool) -> FeatureConfig {
        FeatureConfig {
            neighborhood_top_n: 0,
            raw_numerics: true,
            impute_numerics: impute,
            ..Default::default()
        }
    }

    #[test]
    fn raw_numerics_encoding_and_contract_roundtrip() {
        let domain = Domain::example();
        let cfg = raw_numerics_cfg(false);
        let points = vec![pt(Some(120.0), Some(2.0)), pt(None, Some(0.0))];
        let enc = Encoder::build(&points, &domain, &cfg);

        // Legacy column layout must be reproduced exactly.
        let names: Vec<String> = enc.columns().iter().map(|c| c.name.clone()).collect();
        assert_eq!(
            &names[..11],
            &[
                "bedrooms",
                "totalArea_log",
                "totalArea_missing",
                "coveredArea_log",
                "coveredArea_missing",
                "bathrooms",
                "bathrooms_missing",
                "garages",
                "garages_missing",
                "rooms",
                "rooms_missing",
            ]
        );

        let mut row = vec![0.0f32; enc.width()];
        enc.encode(&points[0], &mut row);
        // bedrooms, then the raw-numerics block, then propertyType one-hots.
        assert_eq!(row[0], 2.0); // bedrooms
        assert!((row[1] - 121.0f32.ln()).abs() < 1e-6); // totalArea_log = ln1p(120)
        assert_eq!(row[2], 0.0); // totalArea_missing
        assert_eq!((row[3], row[4]), (0.0, 1.0)); // coveredArea: missing
        assert_eq!((row[5], row[6]), (2.0, 0.0)); // bathrooms present

        // Zero counts as unspecified, like bedrooms.
        let mut row2 = vec![0.0f32; enc.width()];
        enc.encode(&points[1], &mut row2);
        assert_eq!((row2[1], row2[2]), (0.0, 1.0)); // totalArea missing
        assert_eq!((row2[5], row2[6]), (0.0, 1.0)); // bathrooms 0 -> missing

        // The frozen-contract path must reproduce the same encoder.
        let rebuilt = Encoder::from_columns(&enc.columns()).unwrap();
        let mut row3 = vec![0.0f32; rebuilt.width()];
        rebuilt.encode(&points[0], &mut row3);
        assert_eq!(row, row3);

        // Input requirements name the source payload fields, deduplicated.
        assert_eq!(
            enc.numeric_fields(),
            ["bedrooms", "totalArea", "coveredArea", "bathrooms", "garages", "rooms"]
        );
    }

    #[test]
    fn imputed_numerics_drop_indicator_columns() {
        let domain = Domain::example();
        let cfg = raw_numerics_cfg(true);
        // Build-time imputation fills the metadata before encoding.
        let points = vec![pt(Some(120.0), Some(2.0))];
        let enc = Encoder::build(&points, &domain, &cfg);
        let names: Vec<String> = enc.columns().iter().map(|c| c.name.clone()).collect();
        assert!(!names.iter().any(|n| n.ends_with("_missing")), "columns: {names:?}");
        assert!(names.contains(&"totalArea_log".to_string()));

        let mut row = vec![0.0f32; enc.width()];
        enc.encode(&points[0], &mut row);
        assert!((row[1] - 121.0f32.ln()).abs() < 1e-6); // totalArea_log right after bedrooms

        // Frozen-contract round-trip reproduces the imputed encoder.
        let rebuilt = Encoder::from_columns(&enc.columns()).unwrap();
        let mut row2 = vec![0.0f32; rebuilt.width()];
        rebuilt.encode(&points[0], &mut row2);
        assert_eq!(row, row2);
    }

    #[test]
    fn coordinates_encoding() {
        let domain = Domain::example();
        let cfg = FeatureConfig {
            neighborhood_top_n: 0,
            coordinates: true,
            ..Default::default()
        };
        let mut with_coords = pt(None, None);
        with_coords
            .payload
            .set("coordinates", json!({"lat": -34.92, "lon": -57.95}));
        let mut bad_geocode = pt(None, None);
        bad_geocode
            .payload
            .set("coordinates", json!({"lat": 37.4, "lon": 13.5})); // not in bounds
        let missing = pt(None, None);
        let points = vec![with_coords, bad_geocode, missing];
        let enc = Encoder::build(&points, &domain, &cfg);

        // bedrooms, lat, lon, coords_missing, then propertyType one-hots.
        let names: Vec<String> = enc.columns().iter().map(|c| c.name.clone()).collect();
        assert_eq!(&names[..4], &["bedrooms", "lat", "lon", "coords_missing"]);
        let mut row = vec![0.0f32; enc.width()];
        enc.encode(&points[0], &mut row);
        assert!((row[1] - -34.92).abs() < 1e-4);
        assert!((row[2] - -57.95).abs() < 1e-4);
        assert_eq!(row[3], 0.0);

        // Out-of-bounds geocodes and absent coordinates both count missing.
        for p in &points[1..] {
            let mut r = vec![0.0f32; enc.width()];
            enc.encode(p, &mut r);
            assert_eq!((r[1], r[2], r[3]), (0.0, 0.0, 1.0));
        }

        assert_eq!(enc.numeric_fields(), ["bedrooms", "coordinates"]);
        let rebuilt = Encoder::from_columns(&enc.columns()).unwrap();
        assert_eq!(rebuilt.width(), enc.width());
    }

    /// The generic `fields` map drives the encoder for alien domains.
    #[test]
    fn generic_fields_map_controls_encoder() {
        let domain = Domain::example();
        let mut cfg = FeatureConfig::default();
        cfg.fields.insert("bedrooms".into(), false);
        cfg.fields.insert("propertyType".into(), true);
        cfg.fields.insert("city".into(), true);
        cfg.vocab_top_n.insert("city".into(), 2);
        let mk = |city: &str| RawPoint {
            id: 0,
            vector: vec![],
            payload: Payload::from_raw(
                json!({"metadata": {"propertyType": "house", "city": city}}),
                "metadata",
            ),
        };
        let points = vec![mk("city-a"), mk("city-a"), mk("city-b"), mk("city-c")];
        let enc = Encoder::build(&points, &domain, &cfg);
        let names: Vec<String> = enc.columns().iter().map(|c| c.name.clone()).collect();
        assert!(!names.contains(&"bedrooms".to_string()));
        assert!(names.contains(&"propertyType=house".to_string()));
        // top-2 cities by frequency + other.
        assert!(names.contains(&"city=city-a".to_string()));
        assert!(names.contains(&"city=__other__".to_string()));
        assert_eq!(names.iter().filter(|n| n.starts_with("city=")).count(), 3);
    }
}
