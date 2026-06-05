//! Column-mask computation and feature subsetting. A member sees a column
//! subset of the blend dataset, selected by name (frozen members: their
//! contract's columns) or by block exclusion (trained members).

use anyhow::{bail, Result};
use pg_core::{ColumnDesc, ColumnKind};

/// The lat/lon/indicator triple `FeatureConfig.coordinates` produces.
const COORDINATE_FIELDS: [&str; 3] = ["lat", "lon", "coords_missing"];
/// Numeric fields `FeatureConfig.raw_numerics` produces (with and without
/// `impute_numerics`, which drops the `_missing` indicators).
const RAW_NUMERIC_FIELDS: [&str; 10] = [
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
];

/// Does `col` belong to the named feature block?
fn in_block(col: &ColumnDesc, block: &str) -> bool {
    match (&col.kind, block) {
        (ColumnKind::Pca { .. }, "pca") => true,
        (ColumnKind::Numeric { field }, "coordinates") => {
            COORDINATE_FIELDS.contains(&field.as_str())
        }
        (ColumnKind::Numeric { field }, "raw_numerics") => {
            RAW_NUMERIC_FIELDS.contains(&field.as_str())
        }
        // A bare numeric field name ("bedrooms") or a one-hot group name
        // ("propertyType", "neighborhood", …).
        (ColumnKind::Numeric { field }, b) => field == b,
        (ColumnKind::Onehot { group, .. }, b) => group == b,
        _ => false,
    }
}

/// Trained-member mask: all dataset columns minus the excluded blocks.
/// Returns indices into the dataset's column order. Errors if a block name
/// matches nothing (typo guard) or the mask drops every column.
pub fn mask_by_blocks(columns: &[ColumnDesc], exclude: &[String]) -> Result<Vec<usize>> {
    for block in exclude {
        if !columns.iter().any(|c| in_block(c, block)) {
            let known: Vec<&str> = ["pca", "coordinates", "raw_numerics"]
                .into_iter()
                .chain(columns.iter().filter_map(|c| match &c.kind {
                    ColumnKind::Onehot { group, .. } => Some(group.as_str()),
                    ColumnKind::Numeric { field } => Some(field.as_str()),
                    _ => None,
                }))
                .collect();
            bail!("exclude_blocks {block:?} matches no column; known blocks: {known:?}");
        }
    }
    let kept: Vec<usize> = columns
        .iter()
        .enumerate()
        .filter(|(_, c)| !exclude.iter().any(|b| in_block(c, b)))
        .map(|(i, _)| i)
        .collect();
    if kept.is_empty() {
        bail!("exclude_blocks {exclude:?} leaves no columns");
    }
    Ok(kept)
}

/// Frozen-member mask: match the member's expected columns by name against
/// the available columns, preserving the member's order. Errors naming every
/// missing column.
pub fn mask_by_names(available: &[ColumnDesc], expected: &[String]) -> Result<Vec<usize>> {
    let mut idx = Vec::with_capacity(expected.len());
    let mut missing = Vec::new();
    for name in expected {
        match available.iter().position(|c| &c.name == name) {
            Some(i) => idx.push(i),
            None => missing.push(name.as_str()),
        }
    }
    if !missing.is_empty() {
        bail!(
            "{} member column(s) missing from the available features: {missing:?}",
            missing.len()
        );
    }
    Ok(idx)
}

/// Gather `cols` (indices into a `n_cols`-wide row-major matrix) for the
/// given row indices into a new row-major matrix.
pub fn subset(features: &[f32], n_cols: usize, rows: &[u32], cols: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows.len() * cols.len());
    for &r in rows {
        let row = &features[r as usize * n_cols..(r as usize + 1) * n_cols];
        out.extend(cols.iter().map(|&c| row[c]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cols() -> Vec<ColumnDesc> {
        vec![
            ColumnDesc { name: "pca_0".into(), kind: ColumnKind::Pca { component: 0 } },
            ColumnDesc { name: "bedrooms".into(), kind: ColumnKind::Numeric { field: "bedrooms".into() } },
            ColumnDesc { name: "lat".into(), kind: ColumnKind::Numeric { field: "lat".into() } },
            ColumnDesc { name: "lon".into(), kind: ColumnKind::Numeric { field: "lon".into() } },
            ColumnDesc {
                name: "coords_missing".into(),
                kind: ColumnKind::Numeric { field: "coords_missing".into() },
            },
            ColumnDesc {
                name: "propertyType=house".into(),
                kind: ColumnKind::Onehot { group: "propertyType".into(), value: "house".into() },
            },
        ]
    }

    #[test]
    fn block_mask_drops_coordinates_and_keeps_order() {
        let kept = mask_by_blocks(&cols(), &["coordinates".into()]).unwrap();
        assert_eq!(kept, vec![0, 1, 5]);
        let kept = mask_by_blocks(&cols(), &["propertyType".into(), "pca".into()]).unwrap();
        assert_eq!(kept, vec![1, 2, 3, 4]);
        assert!(mask_by_blocks(&cols(), &["nope".into()]).is_err());
        assert!(mask_by_blocks(
            &cols(),
            &["pca".into(), "coordinates".into(), "bedrooms".into(), "propertyType".into()]
        )
        .is_err());
    }

    #[test]
    fn name_mask_preserves_member_order_and_names_missing() {
        let idx = mask_by_names(&cols(), &["lat".into(), "pca_0".into()]).unwrap();
        assert_eq!(idx, vec![2, 0]);
        let err = mask_by_names(&cols(), &["pca_0".into(), "ghost".into()]).unwrap_err();
        assert!(err.to_string().contains("ghost"), "{err}");
    }

    #[test]
    fn subset_gathers_rows_and_columns() {
        // 3 rows × 4 cols
        let f: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let out = subset(&f, 4, &[2, 0], &[3, 1]);
        assert_eq!(out, vec![11.0, 9.0, 3.0, 1.0]);
    }
}
