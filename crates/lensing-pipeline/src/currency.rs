//! Currency reconciliation and conversion, applied to raw points before
//! quality evaluation (so target-value rules see values in the kept
//! currency).
//!
//! The build collections dropped the currency field; the raw scrape
//! collection still has it under the same point ids. `apply` reconciles it
//! back, then — depending on the mode — leaves filtering to the
//! `foreign-currency` quality rule or converts foreign values using per-date
//! exchange rates from the domain's rate endpoint. Single-currency domains
//! (no `[currency]` section) skip all of this.

use anyhow::{bail, Context, Result};
use chrono::NaiveDate;
use pg_core::domain::{CurrencyDomain, Domain};
use pg_core::{CurrencyConfig, CurrencyMode, CurrencyReport};
use serde_json::{json, Value};

use crate::qdrant::{self, RawPoint};

/// Reconcile currency onto `points` and, in Convert mode, rewrite foreign
/// target values into `cfg.keep`. Never drops rows — exclusion is the
/// `foreign-currency` quality rule's job (Filter mode).
pub fn apply(
    points: &mut [RawPoint],
    cfg: &CurrencyConfig,
    qdrant_url: &str,
    domain: &Domain,
    progress: &(dyn Fn(&str) + Sync),
) -> Result<CurrencyReport> {
    let mut report = CurrencyReport {
        config: cfg.clone(),
        n_foreign: 0,
        n_missing: 0,
        n_converted: 0,
        rate_min: None,
        rate_max: None,
    };
    let Some(dc) = &domain.currency else { return Ok(report) };
    if cfg.mode == CurrencyMode::Off {
        return Ok(report);
    }
    let currency_field = dc.currency_field.as_str();
    let date_field = dc.date_field.as_str();

    if let Some(companion) = &cfg.reconcile_collection {
        progress("reconciling currency");
        let ids: Vec<u64> = points
            .iter()
            .filter(|p| p.payload.get(currency_field).is_null())
            .map(|p| p.id)
            .collect();
        if !ids.is_empty() {
            let field_names = vec![currency_field.to_string(), date_field.to_string()];
            let map = qdrant::fetch_fields(qdrant_url, companion, &ids, domain, &field_names)
                .with_context(|| format!("reconcile currency from {companion}"))?;
            for p in points.iter_mut() {
                if p.payload.get(currency_field).is_null() {
                    if let Some(fetched) = map.get(&p.id) {
                        if let Some(c) = fetched.get(currency_field) {
                            p.payload.set(currency_field, c.clone());
                        }
                        if p.payload.get(date_field).is_null() {
                            if let Some(d) = fetched.get(date_field) {
                                p.payload.set(date_field, d.clone());
                            }
                        }
                    }
                }
            }
        }
    }

    report.n_missing =
        points.iter().filter(|p| !p.payload.get(currency_field).is_string()).count();
    report.n_foreign = points
        .iter()
        .filter(|p| p.payload.get(currency_field).as_str().is_some_and(|c| c != cfg.keep))
        .count();

    if cfg.mode == CurrencyMode::Convert && report.n_foreign > 0 {
        progress("fetching exchange rates");
        let table = RateTable::fetch(&cfg.rate_source, dc)?;
        let stats = convert(points, &table, &cfg.keep, dc, &domain.target.field);
        report.n_converted = stats.n_converted;
        report.rate_min = stats.rate_min;
        report.rate_max = stats.rate_max;
    }
    Ok(report)
}

/// Human-readable currency clause for the manifest's filter description.
pub fn filter_desc(cfg: &CurrencyConfig) -> String {
    match cfg.mode {
        CurrencyMode::Off => String::new(),
        CurrencyMode::Filter => format!(" && currency==\"{}\"", cfg.keep),
        CurrencyMode::Convert => {
            format!(" (foreign prices converted to {} @ {} rate)", cfg.keep, cfg.rate_source)
        }
    }
}

pub struct ConvertStats {
    pub n_converted: usize,
    pub rate_min: Option<f64>,
    pub rate_max: Option<f64>,
}

/// Rewrite convertible-pair values into `keep` using the per-date rate keyed
/// on the row's date field (most recent rate when absent). Currencies the
/// table cannot bridge are left untouched (they stay foreign and reported).
/// The rate is foreign-per-kept (e.g. ARS per USD).
pub fn convert(
    points: &mut [RawPoint],
    table: &RateTable,
    keep: &str,
    dc: &CurrencyDomain,
    target_field: &str,
) -> ConvertStats {
    let mut stats = ConvertStats { n_converted: 0, rate_min: None, rate_max: None };
    let [pair_a, pair_b] = &dc.pair;
    for p in points.iter_mut() {
        let Some(currency) = p.payload.get(&dc.currency_field).as_str().map(str::to_string)
        else {
            continue;
        };
        if currency == keep {
            continue;
        }
        // Only the configured pair converts, in either direction.
        let convertible = (currency == *pair_a && keep == *pair_b)
            || (currency == *pair_b && keep == *pair_a);
        if !convertible {
            continue;
        }
        let date = p.payload.get(&dc.date_field).as_str().map(str::to_string);
        let rate = table.rate_on(date.as_deref());
        stats.rate_min = Some(stats.rate_min.map_or(rate, |r: f64| r.min(rate)));
        stats.rate_max = Some(stats.rate_max.map_or(rate, |r: f64| r.max(rate)));
        // pair = [foreign, kept]: rate is foreign-per-kept.
        let factor = if currency == *pair_a { 1.0 / rate } else { rate };
        let value = p.payload.num_of(target_field).unwrap_or(0.0) * factor;
        p.payload.set(target_field, json!(value));
        p.payload.set(&dc.currency_field, json!(keep));
        stats.n_converted += 1;
    }
    stats
}

/// Daily foreign-per-kept sell rates, sorted by date.
pub struct RateTable {
    /// (date, venta), ascending by date; never empty.
    rates: Vec<(NaiveDate, f64)>,
}

impl RateTable {
    /// Fetch the full daily history for one series from the domain's rate
    /// endpoint (`{source}` substituted into the URL template). The endpoint
    /// must return `[{"fecha": "YYYY-MM-DD", "venta": <rate>}, …]`.
    pub fn fetch(source: &str, dc: &CurrencyDomain) -> Result<Self> {
        if source.is_empty() || !source.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
            bail!("invalid rate source {source:?}");
        }
        let url = dc.rate_url_template.replace("{source}", source);
        let resp: Value = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?
            .get(&url)
            .send()
            .with_context(|| format!("fetch {source} exchange rates ({url})"))?
            .error_for_status()?
            .json()?;
        let mut rates = Vec::new();
        for e in resp.as_array().context("malformed exchange-rate response")? {
            let (Some(fecha), Some(venta)) = (e["fecha"].as_str(), e["venta"].as_f64()) else {
                continue;
            };
            let Ok(date) = NaiveDate::parse_from_str(fecha, "%Y-%m-%d") else { continue };
            if venta > 0.0 {
                rates.push((date, venta));
            }
        }
        Self::from_rates(rates)
    }

    pub fn from_rates(mut rates: Vec<(NaiveDate, f64)>) -> Result<Self> {
        rates.sort_by_key(|(d, _)| *d);
        anyhow::ensure!(!rates.is_empty(), "exchange-rate series is empty");
        Ok(Self { rates })
    }

    /// Rate for an RFC 3339 / ISO date string: the most recent quote on or
    /// before that date. Missing/unparseable dates and dates before the
    /// series fall back to the latest rate (the corpus is recent).
    pub fn rate_on(&self, created_at: Option<&str>) -> f64 {
        let latest = self.rates.last().unwrap().1;
        let Some(s) = created_at else { return latest };
        let Ok(date) = NaiveDate::parse_from_str(&s[..s.len().min(10)], "%Y-%m-%d") else {
            return latest;
        };
        match self.rates.partition_point(|(d, _)| *d <= date) {
            0 => latest,
            i => self.rates[i - 1].1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qdrant::Payload;

    fn d(s: &str) -> NaiveDate {
        NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap()
    }

    fn table() -> RateTable {
        RateTable::from_rates(vec![
            (d("2026-01-02"), 1200.0),
            (d("2026-01-10"), 1300.0),
            (d("2026-02-01"), 1400.0),
        ])
        .unwrap()
    }

    fn ars_point(id: u64, price: f64, created_at: Option<&str>) -> RawPoint {
        RawPoint {
            id,
            vector: vec![],
            payload: Payload::from_raw(
                serde_json::json!({
                    "metadata": {
                        "propertyType": "house",
                        "operation": "sale",
                        "price": price,
                        "bedrooms": 2.0,
                        "currency": "ARS",
                        "createdAt": created_at,
                    }
                }),
                "metadata",
            ),
        }
    }

    #[test]
    fn rate_lookup() {
        let t = table();
        // Exact date.
        assert_eq!(t.rate_on(Some("2026-01-10")), 1300.0);
        // Gap: most recent quote on or before.
        assert_eq!(t.rate_on(Some("2026-01-20T12:00:00.000Z")), 1300.0);
        // After the series: latest.
        assert_eq!(t.rate_on(Some("2026-06-01")), 1400.0);
        // Before the series / missing / garbage: latest.
        assert_eq!(t.rate_on(Some("2025-01-01")), 1400.0);
        assert_eq!(t.rate_on(None), 1400.0);
        assert_eq!(t.rate_on(Some("not-a-date")), 1400.0);
    }

    #[test]
    fn convert_ars_to_usd() {
        let domain = Domain::default();
        let dc = domain.currency.as_ref().unwrap();
        let mut points = vec![
            ars_point(0, 26_650_000.0, Some("2026-01-05T00:00:00Z")), // @1200
            ars_point(1, 1_400_000.0, None),                          // @1400 (latest)
        ];
        // A USD row must pass through untouched.
        points.push({
            let mut p = ars_point(2, 90_000.0, None);
            p.payload.set("currency", serde_json::json!("USD"));
            p
        });

        let stats = convert(&mut points, &table(), "USD", dc, "price");
        assert_eq!(stats.n_converted, 2);
        assert_eq!((stats.rate_min, stats.rate_max), (Some(1200.0), Some(1400.0)));
        let price = |i: usize| points[i].payload.num_of("price").unwrap();
        assert!((price(0) - 26_650_000.0 / 1200.0).abs() < 1e-6);
        assert!((price(1) - 1_000.0).abs() < 1e-6);
        assert_eq!(price(2), 90_000.0);
        assert!(points
            .iter()
            .all(|p| p.payload.get("currency").as_str() == Some("USD")));
    }
}
