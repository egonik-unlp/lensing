//! Built-in data-quality heuristics, evaluated on raw points before
//! featurization. Every rule is always *evaluated* (so previews can show
//! counts); the config decides which rules *exclude* rows.
//!
//! Rule KEYS are stable identifiers shared by manifests, the API and the UI
//! ("price-range", "bedrooms-outlier", …). The domain binds them to ITS
//! fields and labels them for display — a new domain rebinds, it does not
//! rename.

use std::collections::{HashMap, HashSet};

use pg_core::domain::Domain;
use pg_core::quality::{QualityFilterConfig, QualityReport, RuleStats};
use pg_core::{CurrencyConfig, CurrencyMode};
use serde_json::json;

use crate::qdrant::RawPoint;

/// Groups thinner than this use global target statistics for outlier scoring.
const MIN_GROUP: usize = 30;

pub const RULES: &[&str] = &[
    "nonpositive-price",
    "price-outlier",
    "missing-fields",
    "price-range",
    "bedrooms-outlier",
    "duplicate-content",
    "short-content",
    "foreign-currency",
];

/// Row indices flagged per rule.
pub struct QualityAnalysis {
    pub flagged: HashMap<&'static str, Vec<usize>>,
    /// Whether `foreign-currency` excludes (currency mode == Filter); the
    /// rule's toggle lives in `CurrencyConfig`, not `QualityFilterConfig`.
    foreign_currency_excludes: bool,
}

/// Evaluate all rules over a point set. In Convert mode foreign rows were
/// already rewritten into the kept currency before this runs, so the
/// `foreign-currency` rule only flags what conversion could not bridge.
pub fn evaluate(
    points: &[RawPoint],
    cfg: &QualityFilterConfig,
    currency: &CurrencyConfig,
    domain: &Domain,
) -> QualityAnalysis {
    let mut flagged: HashMap<&'static str, Vec<usize>> = HashMap::new();
    let target = domain.target.field.as_str();
    let target_of = |p: &RawPoint| p.payload.num_of(target).unwrap_or(0.0);
    let content_field = domain.corpus.content_field.as_str();

    // foreign-currency: currency known and not the kept one. Rows with no
    // currency after reconciliation are kept (reported, never dropped).
    // Single-currency domains (no [currency] section) never flag.
    flagged.insert(
        "foreign-currency",
        match &domain.currency {
            Some(dc) => points
                .iter()
                .enumerate()
                .filter(|(_, p)| {
                    p.payload
                        .get(&dc.currency_field)
                        .as_str()
                        .is_some_and(|c| c != currency.keep)
                })
                .map(|(i, _)| i)
                .collect(),
            None => Vec::new(),
        },
    );

    // nonpositive-price: target ≤ 0 or missing.
    flagged.insert(
        "nonpositive-price",
        points
            .iter()
            .enumerate()
            .filter(|(_, p)| !(target_of(p) > 0.0))
            .map(|(i, _)| i)
            .collect(),
    );

    // missing-fields: empty critical categoricals.
    let critical: Vec<&str> = domain.critical_fields().map(|f| f.name.as_str()).collect();
    flagged.insert(
        "missing-fields",
        points
            .iter()
            .enumerate()
            .filter(|(_, p)| critical.iter().any(|f| p.payload.str_of(f).trim().is_empty()))
            .map(|(i, _)| i)
            .collect(),
    );

    // price-outlier: MAD z-score on log1p(target), per outlier group with a
    // global fallback for thin groups. Rows with target ≤ 0 are the
    // nonpositive rule's business and are skipped here.
    let group_field = domain.outlier_group();
    let valid: Vec<(usize, String, f64)> = points
        .iter()
        .enumerate()
        .filter(|(_, p)| target_of(p) > 0.0)
        .map(|(i, p)| {
            let group = group_field.map(|g| p.payload.str_of(g)).unwrap_or_default();
            (i, group, target_of(p).ln_1p())
        })
        .collect();

    let global = mad_stats(valid.iter().map(|(_, _, lp)| *lp).collect());
    let mut by_group: HashMap<&str, Vec<f64>> = HashMap::new();
    for (_, group, lp) in &valid {
        by_group.entry(group).or_default().push(*lp);
    }
    let group_stats: HashMap<&str, (f64, f64)> = by_group
        .into_iter()
        .filter(|(_, v)| v.len() >= MIN_GROUP)
        .map(|(g, v)| (g, mad_stats(v)))
        .collect();

    let threshold = cfg.price_outlier_mad_z;
    flagged.insert(
        "price-outlier",
        valid
            .iter()
            .filter(|(_, group, lp)| {
                let (median, mad) = group_stats.get(group.as_str()).copied().unwrap_or(global);
                mad > 0.0 && (0.6745 * (lp - median) / mad).abs() > threshold
            })
            .map(|(i, _, _)| *i)
            .collect(),
    );

    // price-range: manual hard caps on the target. Rows with target ≤ 0 stay
    // the nonpositive rule's business.
    flagged.insert(
        "price-range",
        points
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                let t = target_of(p);
                t > 0.0 && (t < cfg.price_min || t > cfg.price_max)
            })
            .map(|(i, _)| i)
            .collect(),
    );

    // bedrooms-outlier: the domain's capped numeric, negative or above the
    // cap (data-entry noise). Zero is NOT flagged: it means "unspecified".
    flagged.insert(
        "bedrooms-outlier",
        match domain.quality.capped_numeric.as_deref() {
            Some(field) => points
                .iter()
                .enumerate()
                .filter(|(_, p)| {
                    let b = p.payload.num_of(field).unwrap_or(0.0);
                    !(b >= 0.0) || b > cfg.bedrooms_max
                })
                .map(|(i, _)| i)
                .collect(),
            None => Vec::new(),
        },
    );

    // duplicate-content: exact duplicate document text; the first occurrence
    // is kept, later ones are flagged. Empty content never counts as a
    // duplicate of other empty content (short-content's business).
    let mut seen: HashSet<&str> = HashSet::new();
    flagged.insert(
        "duplicate-content",
        points
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                let content = p.payload.content_of(content_field).trim();
                !content.is_empty() && !seen.insert(content)
            })
            .map(|(i, _)| i)
            .collect(),
    );

    // short-content: document text below the character floor.
    flagged.insert(
        "short-content",
        points
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                p.payload.content_of(content_field).trim().chars().count()
                    < cfg.short_content_min_chars
            })
            .map(|(i, _)| i)
            .collect(),
    );

    QualityAnalysis {
        flagged,
        foreign_currency_excludes: currency.mode == CurrencyMode::Filter
            && domain.currency.is_some(),
    }
}

/// (median, MAD) of a value set.
fn mad_stats(mut values: Vec<f64>) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let median = values[values.len() / 2];
    let mut devs: Vec<f64> = values.iter().map(|v| (v - median).abs()).collect();
    devs.sort_by(|a, b| a.total_cmp(b));
    (median, devs[devs.len() / 2])
}

impl QualityAnalysis {
    /// Whether `rule` excludes its flagged rows. `foreign-currency`'s toggle
    /// is the currency mode captured at evaluation; the rest come from `cfg`.
    fn rule_enabled(&self, cfg: &QualityFilterConfig, rule: &str) -> bool {
        match rule {
            "foreign-currency" => self.foreign_currency_excludes,
            _ => cfg.rule_enabled(rule),
        }
    }

    /// Distinct row indices excluded under `cfg`.
    pub fn excluded(&self, cfg: &QualityFilterConfig) -> HashSet<usize> {
        let mut out = HashSet::new();
        for (rule, rows) in &self.flagged {
            if self.rule_enabled(cfg, rule) {
                out.extend(rows.iter().copied());
            }
        }
        out
    }

    /// The manifest-recorded report under `cfg`.
    pub fn report(&self, cfg: &QualityFilterConfig) -> QualityReport {
        let excluded = self.excluded(cfg);
        let rules = RULES
            .iter()
            .map(|rule| {
                let rows = self.flagged.get(rule).map(Vec::as_slice).unwrap_or(&[]);
                RuleStats {
                    rule: rule.to_string(),
                    n_flagged: rows.len(),
                    n_excluded: if self.rule_enabled(cfg, rule) { rows.len() } else { 0 },
                }
            })
            .collect();
        QualityReport { config: cfg.clone(), rules, n_excluded_total: excluded.len() }
    }

    /// Up to `k` sample flagged items per rule, for preview UIs: row_id, the
    /// target, the critical/grouping categoricals and the currency.
    pub fn samples(&self, points: &[RawPoint], k: usize, domain: &Domain) -> serde_json::Value {
        let mut sample_fields: Vec<&str> =
            domain.critical_fields().map(|f| f.name.as_str()).collect();
        if let Some(g) = domain.outlier_group() {
            if !sample_fields.contains(&g) {
                sample_fields.push(g);
            }
        }
        if let Some(n) = domain.quality.capped_numeric.as_deref() {
            if !sample_fields.contains(&n) {
                sample_fields.push(n);
            }
        }
        let mut out = serde_json::Map::new();
        for rule in RULES {
            let rows = self.flagged.get(rule).map(Vec::as_slice).unwrap_or(&[]);
            let sample: Vec<serde_json::Value> = rows
                .iter()
                .take(k)
                .map(|&i| {
                    let p = &points[i];
                    let mut obj = serde_json::Map::new();
                    obj.insert("row_id".into(), json!(p.id));
                    obj.insert(
                        domain.target.field.clone(),
                        p.payload.get(&domain.target.field).clone(),
                    );
                    for f in &sample_fields {
                        obj.insert((*f).to_string(), p.payload.get(f).clone());
                    }
                    if let Some(dc) = &domain.currency {
                        obj.insert(
                            dc.currency_field.clone(),
                            p.payload.get(&dc.currency_field).clone(),
                        );
                    }
                    serde_json::Value::Object(obj)
                })
                .collect();
            out.insert(rule.to_string(), serde_json::Value::Array(sample));
        }
        serde_json::Value::Object(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qdrant::Payload;
    use pg_core::domain::Domain;
    use serde_json::json;

    fn currency_off() -> CurrencyConfig {
        CurrencyConfig { mode: CurrencyMode::Off, ..Default::default() }
    }

    fn pt(id: u64, property_type: &str, neighborhood: &str, price: f64) -> RawPoint {
        RawPoint {
            id,
            vector: vec![],
            payload: Payload::from_raw(
                json!({
                    "content": "",
                    "metadata": {
                        "propertyType": property_type,
                        "operation": "sale",
                        "price": price,
                        "neighborhood": neighborhood,
                        "bedrooms": 2.0,
                    }
                }),
                "metadata",
            ),
        }
    }

    #[test]
    fn rules_flag_and_exclude() {
        let domain = Domain::default();
        // A tight cluster of normal prices + one absurd outlier + one free
        // + one missing propertyType.
        let mut points: Vec<RawPoint> =
            (0..50).map(|i| pt(i, "house", "centro", 100_000.0 + (i as f64) * 500.0)).collect();
        points.push(pt(100, "house", "centro", 90_000_000.0)); // outlier
        points.push(pt(101, "house", "centro", 0.0)); // nonpositive
        points.push(pt(102, "", "centro", 120_000.0)); // missing field

        let cfg = QualityFilterConfig {
            nonpositive_price: true,
            price_outlier: true,
            price_outlier_mad_z: 3.5,
            missing_fields: false,
            ..Default::default()
        };
        let analysis = evaluate(&points, &cfg, &currency_off(), &domain);

        assert_eq!(analysis.flagged["nonpositive-price"], vec![51]);
        assert!(analysis.flagged["price-outlier"].contains(&50));
        assert!(!analysis.flagged["price-outlier"].contains(&10));
        assert_eq!(analysis.flagged["missing-fields"], vec![52]);

        // missing-fields is flag-only here: counted but not excluded.
        let excluded = analysis.excluded(&cfg);
        assert!(excluded.contains(&50) && excluded.contains(&51));
        assert!(!excluded.contains(&52));

        let report = analysis.report(&cfg);
        assert_eq!(report.n_excluded_total, 2);
        let mf = report.rules.iter().find(|r| r.rule == "missing-fields").unwrap();
        assert_eq!((mf.n_flagged, mf.n_excluded), (1, 0));
    }

    fn pt_full(id: u64, price: f64, bedrooms: f64, content: &str) -> RawPoint {
        let mut p = pt(id, "house", "centro", price);
        p.payload.set("bedrooms", json!(bedrooms));
        p.payload.set("content", json!(content));
        p
    }

    #[test]
    fn extended_rules_flag() {
        let domain = Domain::default();
        let long_a = "a".repeat(100);
        let long_b = "b".repeat(100);
        let points = vec![
            pt_full(0, 100_000.0, 2.0, &long_a),  // clean
            pt_full(1, 500.0, 2.0, &long_b),      // below price_min
            pt_full(2, 90_000_000.0, 2.0, &long_b), // above price_max; dup of 1
            pt_full(3, 120_000.0, -1.0, &long_a), // negative bedrooms; dup of 0
            pt_full(4, 130_000.0, 20.0, "short"), // bedrooms over cap; short content
            pt_full(5, 0.0, 2.0, ""),             // nonpositive; empty content is short, not dup
            pt_full(6, 140_000.0, 3.0, ""),       // empty content: short, NOT duplicate of 5
        ];

        let cfg = QualityFilterConfig {
            nonpositive_price: false,
            price_range: true,
            price_min: 1_000.0,
            price_max: 50_000_000.0,
            bedrooms_outlier: true,
            bedrooms_max: 15.0,
            duplicate_content: true,
            short_content: true,
            short_content_min_chars: 80,
            ..Default::default()
        };
        let analysis = evaluate(&points, &cfg, &currency_off(), &domain);

        assert_eq!(analysis.flagged["price-range"], vec![1, 2]);
        assert_eq!(analysis.flagged["bedrooms-outlier"], vec![3, 4]);
        // Keep-first: rows 0 and 1 survive, their later copies are flagged.
        assert_eq!(analysis.flagged["duplicate-content"], vec![2, 3]);
        // Empty content is short-content's business, never a duplicate.
        assert_eq!(analysis.flagged["short-content"], vec![4, 5, 6]);

        // price ≤ 0 stays the nonpositive rule's business (here disabled).
        assert!(!analysis.flagged["price-range"].contains(&5));
        let excluded = analysis.excluded(&cfg);
        assert!(excluded.contains(&1) && excluded.contains(&4));
        assert!(!excluded.contains(&0));

        let report = analysis.report(&cfg);
        assert_eq!(report.rules.len(), RULES.len());
    }

    #[test]
    fn foreign_currency_rule() {
        let domain = Domain::default();
        let mut points: Vec<RawPoint> =
            (0..3).map(|i| pt(i, "house", "centro", 100_000.0)).collect();
        points[0].payload.set("currency", json!("USD"));
        points[1].payload.set("currency", json!("ARS"));
        // points[2] has no currency: reported as missing elsewhere, never flagged.

        let cfg = QualityFilterConfig { nonpositive_price: false, ..Default::default() };

        // Filter mode: ARS row flagged AND excluded.
        let currency = CurrencyConfig { mode: CurrencyMode::Filter, ..Default::default() };
        let analysis = evaluate(&points, &cfg, &currency, &domain);
        assert_eq!(analysis.flagged["foreign-currency"], vec![1]);
        assert!(analysis.excluded(&cfg).contains(&1));
        let fc = analysis
            .report(&cfg)
            .rules
            .into_iter()
            .find(|r| r.rule == "foreign-currency")
            .unwrap();
        assert_eq!((fc.n_flagged, fc.n_excluded), (1, 1));

        // Off mode: still counted, never excluded.
        let analysis = evaluate(&points, &cfg, &currency_off(), &domain);
        assert_eq!(analysis.flagged["foreign-currency"], vec![1]);
        assert!(analysis.excluded(&cfg).is_empty());
    }
}
