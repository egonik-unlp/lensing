//! Column-mask computation and feature subsetting. A member sees a column
//! subset of the blend dataset, selected by name (frozen members: their
//! contract's columns) or by block exclusion (trained members).

use anyhow::{bail, Result};
use lensing_core::{ColumnDesc, ColumnKind};

/// Does `col` belong to the named feature block? Entirely artifact-driven:
/// a numeric column matches its source field name or the feature block the
/// encoder recorded on it (the domain field's toggle group, or
/// "coordinates" for the lat/lon/missing triple); a one-hot matches its
/// group. Pre-group artifacts carry no block tag — their numeric columns
/// match by field name only (rebuild the dataset to exclude by block).
fn in_block(col: &ColumnDesc, block: &str) -> bool {
    match (&col.kind, block) {
        (ColumnKind::Pca { .. }, "pca") => true,
        (ColumnKind::Numeric { field, group }, b) => field == b || group.as_deref() == Some(b),
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
            let known: std::collections::BTreeSet<&str> = std::iter::once("pca")
                .chain(columns.iter().flat_map(|c| match &c.kind {
                    ColumnKind::Onehot { group, .. } => vec![group.as_str()],
                    ColumnKind::Numeric { field, group } => std::iter::once(field.as_str())
                        .chain(group.as_deref())
                        .collect(),
                    _ => vec![],
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

    fn num(name: &str, field: &str, group: Option<&str>) -> ColumnDesc {
        ColumnDesc {
            name: name.into(),
            kind: ColumnKind::Numeric {
                field: field.into(),
                group: group.map(str::to_string),
            },
        }
    }

    /// Columns as the encoder emits them: numeric `field` is the SOURCE
    /// payload field (a suffixed column keeps its source field), `group` the
    /// feature block.
    fn cols() -> Vec<ColumnDesc> {
        vec![
            ColumnDesc { name: "pca_0".into(), kind: ColumnKind::Pca { component: 0 } },
            num("size", "size", None),
            num("area_log", "area", Some("raw_numerics")),
            num("area_missing", "area", Some("raw_numerics")),
            num("lat", "geo", Some("coordinates")),
            num("lon", "geo", Some("coordinates")),
            num("coords_missing", "geo", Some("coordinates")),
            ColumnDesc {
                name: "category=a".into(),
                kind: ColumnKind::Onehot { group: "category".into(), value: "a".into() },
            },
        ]
    }

    #[test]
    fn block_mask_drops_blocks_and_keeps_order() {
        let kept = mask_by_blocks(&cols(), &["coordinates".into()]).unwrap();
        assert_eq!(kept, vec![0, 1, 2, 3, 7]);
        let kept = mask_by_blocks(&cols(), &["raw_numerics".into()]).unwrap();
        assert_eq!(kept, vec![0, 1, 4, 5, 6, 7]);
        // A source-field name also works as a block ("area" → its columns).
        let kept = mask_by_blocks(&cols(), &["area".into()]).unwrap();
        assert_eq!(kept, vec![0, 1, 4, 5, 6, 7]);
        let kept = mask_by_blocks(&cols(), &["category".into(), "pca".into()]).unwrap();
        assert_eq!(kept, vec![1, 2, 3, 4, 5, 6]);
        assert!(mask_by_blocks(&cols(), &["nope".into()]).is_err());
        assert!(mask_by_blocks(
            &cols(),
            &[
                "pca".into(),
                "coordinates".into(),
                "raw_numerics".into(),
                "size".into(),
                "category".into()
            ]
        )
        .is_err());
    }

    #[test]
    fn pre_group_artifacts_match_by_field_name_only() {
        // Columns frozen before block tags existed carry no group; block
        // names match nothing (typo-guard error), field names still work.
        let legacy = vec![num("area_log", "area", None), num("size", "size", None)];
        assert!(mask_by_blocks(&legacy, &["raw_numerics".into()]).is_err());
        let kept = mask_by_blocks(&legacy, &["area".into()]).unwrap();
        assert_eq!(kept, vec![1]);
    }

    #[test]
    fn name_mask_preserves_member_order_and_names_missing() {
        let idx = mask_by_names(&cols(), &["lat".into(), "pca_0".into()]).unwrap();
        assert_eq!(idx, vec![4, 0]);
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
