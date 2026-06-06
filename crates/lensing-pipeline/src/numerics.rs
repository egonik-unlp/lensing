//! Reconciled-numeric handling, applied to retained points before feature
//! encoding. The build collections dropped some payload fields (areas,
//! bathrooms, garages, rooms, coordinates in the original corpus); the raw
//! scrape collection still has them under the same point ids (the same join
//! the currency pipeline uses). Optionally, rows lacking both areas get an
//! area extracted from "… m²" mentions in the document text.
//!
//! Which fields participate comes from the domain: `reconcile = true` fields
//! join from the companion collection; `encode = "log1p"` fields are treated
//! as areas (decimal repair + text backfill); `indicator = true` numerics
//! are the imputable set.
//!
//! Motivation (experiments/2026-06-04-svm-moe-cluster-scan.md): the worst
//! errors are a systematic compression of the target extremes, shown to be
//! information-limited — and size, the missing information, is sitting in
//! the raw collection.

use anyhow::{Context, Result};
use lensing_core::domain::{Domain, NumericEncode};
use lensing_core::{FeatureConfig, NumericImputation, NumericMedians, NumericsReport};
use serde_json::json;

use crate::qdrant::{self, RawPoint};

/// Reconcile numeric fields onto `points` (in place) and report coverage.
/// Returns `None` (and fetches nothing) unless reconciled numerics or
/// coordinates are enabled under `cfg`.
pub fn apply(
    points: &mut [RawPoint],
    cfg: &FeatureConfig,
    reconcile_collection: &Option<String>,
    qdrant_url: &str,
    domain: &Domain,
    progress: &(dyn Fn(&str) + Sync),
) -> Result<Option<NumericsReport>> {
    let enabled_numerics = reconciled_numeric_fields(domain, cfg);
    let coords_on = domain
        .coordinates_field()
        .map(|f| domain.field_enabled(cfg, f))
        .unwrap_or(false);
    if enabled_numerics.is_empty() && !coords_on {
        return Ok(None);
    }

    if let Some(companion) = reconcile_collection {
        progress("reconciling numeric fields");
        reconcile(points, companion, qdrant_url, domain)?;
    }

    let (n_repaired, n_backfilled) = if !enabled_numerics.is_empty() {
        if cfg.area_content_backfill {
            progress("repairing + backfilling areas");
        }
        normalize(points, cfg.area_content_backfill, domain)
    } else {
        (0, 0)
    };

    let mut coverage = std::collections::BTreeMap::new();
    for f in &enabled_numerics {
        let n = points.iter().filter(|p| present(p.payload.num_of(f))).count();
        coverage.insert(f.clone(), n);
    }
    Ok(Some(NumericsReport {
        reconcile_collection: reconcile_collection.clone(),
        content_backfill: cfg.area_content_backfill,
        coverage,
        n_area_backfilled: n_backfilled,
        n_area_repaired: n_repaired,
        n_coordinates: if coords_on {
            let field = domain.coordinates_field().map(|f| f.name.clone()).unwrap_or_default();
            let bounds = domain.coordinate_bounds_arr();
            points.iter().filter(|p| p.payload.coords_of(&field, &bounds).is_some()).count()
        } else {
            0
        },
        legacy: Default::default(),
    }))
}

/// Names of the domain's reconciled numeric fields enabled under `cfg`.
fn reconciled_numeric_fields(domain: &Domain, cfg: &FeatureConfig) -> Vec<String> {
    domain
        .numeric_fields()
        .filter(|f| f.reconcile && domain.field_enabled(cfg, f))
        .map(|f| f.name.clone())
        .collect()
}

/// Domain numeric fields treated as areas (log1p-encoded): decimal repair +
/// text backfill apply to these.
fn area_fields(domain: &Domain) -> Vec<String> {
    domain
        .numeric_fields()
        .filter(|f| f.encode == NumericEncode::Log1p)
        .map(|f| f.name.clone())
        .collect()
}

/// Join the domain's `reconcile` fields from a companion collection by point
/// id, filling only fields that are absent. Shared by the dataset build and
/// the model predict path (point_ids inputs) so the two cannot drift.
pub fn reconcile(
    points: &mut [RawPoint],
    collection: &str,
    qdrant_url: &str,
    domain: &Domain,
) -> Result<()> {
    let field_names: Vec<String> = domain.reconcile_fields().map(|f| f.name.clone()).collect();
    if field_names.is_empty() {
        return Ok(());
    }
    let ids: Vec<u64> = points
        .iter()
        .filter(|p| field_names.iter().any(|f| p.payload.get(f).is_null()))
        .map(|p| p.id)
        .collect();
    if ids.is_empty() {
        return Ok(());
    }
    let map = qdrant::fetch_fields(qdrant_url, collection, &ids, domain, &field_names)
        .with_context(|| format!("reconcile numeric fields from {collection}"))?;
    for p in points.iter_mut() {
        let Some(fetched) = map.get(&p.id) else { continue };
        for f in &field_names {
            if p.payload.get(f).is_null() {
                if let Some(v) = fetched.get(f) {
                    p.payload.set(f, v.clone());
                }
            }
        }
    }
    Ok(())
}

/// Per-group medians of the present (> 0) numeric values over the train rows
/// only — test rows must not leak into the fill values. Groups with no
/// present value for a field fall back to the global median. The grouping
/// field is the domain's outlier group; the imputable set is the
/// indicator-carrying numerics.
pub fn fit_imputation(points: &[RawPoint], train_idx: &[u32], domain: &Domain) -> NumericImputation {
    use std::collections::BTreeMap;
    let fields: Vec<String> = domain
        .numeric_fields()
        .filter(|f| f.indicator)
        .map(|f| f.name.clone())
        .collect();
    let group_field = domain.outlier_group().unwrap_or_default().to_string();

    let mut by_group: BTreeMap<String, Vec<Vec<f64>>> = BTreeMap::new();
    let mut global: Vec<Vec<f64>> = vec![Vec::new(); fields.len()];
    for &row in train_idx {
        let p = &points[row as usize];
        let per_group = by_group
            .entry(p.payload.str_of(&group_field))
            .or_insert_with(|| vec![Vec::new(); fields.len()]);
        for (i, f) in fields.iter().enumerate() {
            if let Some(v) = p.payload.num_of(f).filter(|x| *x > 0.0) {
                per_group[i].push(v);
                global[i].push(v);
            }
        }
    }

    let medians = |values: &[Vec<f64>], fallback: Option<&NumericMedians>| {
        let mut out = std::collections::BTreeMap::new();
        for (i, f) in fields.iter().enumerate() {
            let mut v = values[i].clone();
            let med = if v.is_empty() {
                fallback.and_then(|fb| fb.get(f)).unwrap_or(0.0)
            } else {
                v.sort_by(|a, b| a.total_cmp(b));
                let mid = v.len() / 2;
                if v.len() % 2 == 0 { (v[mid - 1] + v[mid]) / 2.0 } else { v[mid] }
            };
            out.insert(f.clone(), med);
        }
        NumericMedians(out)
    };

    let global_medians = medians(&global, None);
    NumericImputation {
        group_field,
        by_property_type: by_group
            .iter()
            .map(|(t, vals)| (t.clone(), medians(vals, Some(&global_medians))))
            .collect(),
        global: global_medians,
    }
}

/// Fill missing (unspecified) numeric fields in place with the frozen
/// medians. Applied identically at build and predict time, after
/// reconcile/repair/backfill.
pub fn impute(points: &mut [RawPoint], imp: &NumericImputation) {
    for p in points.iter_mut() {
        let group = p.payload.str_of(&imp.group_field);
        let med = imp.by_property_type.get(&group).unwrap_or(&imp.global);
        let fields: Vec<(String, f64)> =
            med.0.iter().map(|(f, v)| (f.clone(), *v)).collect();
        for (f, value) in fields {
            if !present(p.payload.num_of(&f)) && value > 0.0 {
                p.payload.set(&f, json!(value));
            }
        }
    }
}

/// Per-point cleanup applied identically at build and predict time:
///
/// 1. Repair scraper-mangled areas: "3.186" (European thousands dot for
///    3,186 m²) parses as 3.186. A fractional area under 15 m² whose ×1000
///    lands in the plausible range is such a value; other sub-15 m² areas
///    become unspecified. Runs before the text backfill so repaired-to-
///    missing rows can still pick an area up from the document.
/// 2. With `backfill`: rows lacking every area field get the first area
///    field filled from "… m²" mentions in the document text.
///
/// Returns (areas repaired, areas backfilled).
pub fn normalize(points: &mut [RawPoint], backfill: bool, domain: &Domain) -> (usize, usize) {
    let areas = area_fields(domain);
    let content_field = domain.corpus.content_field.clone();
    let mut n_repaired = 0usize;
    for p in points.iter_mut() {
        for field in &areas {
            let Some(v) = p.payload.num_of(field) else { continue };
            if v <= 0.0 || v >= 15.0 {
                continue;
            }
            if v.fract() > 1e-9 && (15.0..=50_000.0).contains(&(v * 1000.0)) {
                p.payload.set(field, json!(v * 1000.0));
                n_repaired += 1;
            } else {
                p.payload.set(field, serde_json::Value::Null);
            }
        }
    }

    let mut n_backfilled = 0usize;
    if backfill {
        if let Some(first_area) = areas.first() {
            for p in points.iter_mut() {
                if areas.iter().all(|f| !present(p.payload.num_of(f))) {
                    if let Some(area) = area_from_content(p.payload.content_of(&content_field)) {
                        p.payload.set(first_area, json!(area));
                        n_backfilled += 1;
                    }
                }
            }
        }
    }
    (n_repaired, n_backfilled)
}

/// Zero means unspecified (scraper convention).
fn present(v: Option<f64>) -> bool {
    v.is_some_and(|x| x > 0.0)
}

/// Largest plausible area mentioned in the document text ("250 m2",
/// "690 metros construidos", "1.200 mts de terreno"). Conservative:
/// the unit must immediately follow the number, values outside
/// 15..=50,000 m² are ignored, and "a 100 metros …" (the distance idiom)
/// is skipped via the preceding word. Taking the max lets room-sized
/// mentions ("sala de 50 metros") lose to the real total elsewhere.
pub fn area_from_content(text: &str) -> Option<f64> {
    let lower = text.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    let mut best: Option<f64> = None;
    let mut i = 0usize;
    while i < chars.len() {
        if !chars[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        // Start of a digit run; bail if it continues a word (e.g. "p128").
        if i > 0 && chars[i - 1].is_alphanumeric() {
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            continue;
        }
        let start = i;
        let mut num = String::new();
        while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.' || chars[i] == ',') {
            num.push(chars[i]);
            i += 1;
        }
        let Some(value) = parse_es_number(num.trim_end_matches(['.', ','])) else { continue };

        // The unit must be the next token.
        let mut j = i;
        while j < chars.len() && chars[j] == ' ' {
            j += 1;
        }
        let unit: String = chars[j..]
            .iter()
            .take_while(|c| c.is_alphanumeric() || **c == '²')
            .collect();
        let is_area = matches!(unit.as_str(), "m2" | "m²" | "mts" | "mts2" | "mt2" | "metros");
        if !is_area || !(15.0..=50_000.0).contains(&value) {
            continue;
        }
        // "a 100 metros del centro" is a distance, not an area.
        if preceding_word(&chars, start) == "a" {
            continue;
        }
        best = Some(best.map_or(value, |b: f64| b.max(value)));
    }
    best
}

/// Spanish-format number: "1.200" / "1,200" are thousands, "8,5" / "8.5"
/// are decimals (separator + exactly 3 digits = thousands).
fn parse_es_number(s: &str) -> Option<f64> {
    let mut normalized = String::with_capacity(s.len());
    let parts: Vec<&str> = s.split(['.', ',']).collect();
    for (i, part) in parts.iter().enumerate() {
        if i == 0 {
            normalized.push_str(part);
        } else if part.len() == 3 {
            normalized.push_str(part); // thousands group
        } else {
            normalized.push('.');
            normalized.push_str(part); // decimal tail
        }
    }
    normalized.parse().ok()
}

/// The word immediately before `pos` (used to skip distance idioms).
fn preceding_word(chars: &[char], pos: usize) -> String {
    let mut end = pos;
    while end > 0 && chars[end - 1] == ' ' {
        end -= 1;
    }
    let mut start = end;
    while start > 0 && chars[start - 1].is_alphabetic() {
        start -= 1;
    }
    chars[start..end].iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_extraction() {
        // Unit immediately after the number, max wins over room mentions.
        assert_eq!(area_from_content("Lote de 8x30, 690 metros construidos"), Some(690.0));
        assert_eq!(area_from_content("sala de 50 metros y 250 m2 cubiertos"), Some(250.0));
        assert_eq!(area_from_content("casa de 120m²"), Some(120.0));
        // Spanish thousands and decimals.
        assert_eq!(area_from_content("1.200 mts de terreno"), Some(1200.0));
        assert_eq!(area_from_content("87,5 m2 totales"), Some(87.5));
        // Distance idiom and out-of-range values are skipped.
        assert_eq!(area_from_content("a 100 metros del centro"), None);
        assert_eq!(area_from_content("a 5 m2"), None);
        assert_eq!(area_from_content("3 metros de frente"), None); // < 15
        assert_eq!(area_from_content("100000 m2"), None); // > 50k
        // No unit, no match.
        assert_eq!(area_from_content("USD 250.000, 3 dormitorios"), None);
        assert_eq!(area_from_content(""), None);
    }
}
