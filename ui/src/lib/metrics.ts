// Data-driven metric handling — the UI mirror of lensing_core's open metric
// registry. Metric *names* are not hardcoded anywhere in the UI: views read
// the columns to show from `domain.metrics.columns`, resolve each run's value
// by name with `metricValue` (case/separator-insensitive, like the Rust
// `Metrics::get`), and format it with `formatMetric`. This is what lets one
// build serve a regression, classification, or time-series instance and show
// exactly that instance's metrics.

import type { Metrics } from '../api/types'
import type { Domain } from './domain'
import { fmtLoss, fmtPct, fmtR2, fmtTargetCell } from './format'

/** Canonical key: lowercased, `_`/`-`/space stripped, `²`→`2`. So `"ROC-AUC"`,
 *  `"roc_auc"`, `"R²"`, `"r2"` collapse to one token. Mirrors Rust
 *  `normalize_metric_name`. */
export function normalizeMetricName(name: string): string {
  let out = ''
  for (const ch of name.toLowerCase()) {
    if (ch === '_' || ch === '-' || ch === ' ') continue
    out += ch === '²' ? '2' : ch
  }
  return out
}

// Built-in metric names → the typed field they live on in the Metrics blob.
const TYPED_ALIASES: Record<string, keyof Metrics> = {
  mae: 'mae',
  rmse: 'rmse',
  mape: 'mape',
  medape: 'medape',
  r2: 'r2',
  accuracy: 'accuracy',
  acc: 'accuracy',
  logloss: 'logloss',
  auc: 'auc',
  rocauc: 'auc',
  macroauc: 'auc',
  brier: 'brier',
  macrof1: 'macro_f1',
  f1: 'macro_f1',
}

/** Read a metric off a run by name. Resolves built-ins from their typed field
 *  and instance-defined metrics from the flattened extra keys (exact, then
 *  normalized). `null` when absent. Mirrors Rust `Metrics::get`. */
export function metricValue(m: Metrics | null | undefined, name: string): number | null {
  if (!m) return null
  const norm = normalizeMetricName(name)
  const typedKey = TYPED_ALIASES[norm]
  if (typedKey) {
    const v = m[typedKey]
    if (typeof v === 'number' && Number.isFinite(v)) return v
  }
  const exact = m[name]
  if (typeof exact === 'number' && Number.isFinite(exact)) return exact
  for (const [k, v] of Object.entries(m)) {
    if (normalizeMetricName(k) === norm && typeof v === 'number' && Number.isFinite(v)) return v
  }
  return null
}

/** The metric column names a domain wants shown, in order. */
export function metricColumns(domain: Domain): string[] {
  return domain.metrics.columns
}

/** Whether `name` renders as a percentage for this domain. */
export function isPercentMetric(domain: Domain, name: string): boolean {
  const norm = normalizeMetricName(name)
  if (domain.metrics.percent.some((p) => normalizeMetricName(p) === norm)) return true
  // Percentage-error metrics are always fractions, regardless of the list.
  return norm === 'mape' || norm === 'medape' || norm === 'smape'
}

const LOWER_IS_BETTER = new Set(['mae', 'rmse', 'rmsle', 'mape', 'medape', 'logloss', 'brier', 'mse'])

/** Whether a smaller value ranks better. Honors `[metrics].directions`
 *  (name-insensitive), else the error/loss-vs-score heuristic. Mirrors Rust
 *  `MetricsSpec::lower_is_better`. */
export function metricLowerIsBetter(domain: Domain, name: string): boolean {
  const norm = normalizeMetricName(name)
  const dirs = domain.metrics.directions
  if (dirs) {
    for (const [k, v] of Object.entries(dirs)) {
      if (normalizeMetricName(k) === norm) return v
    }
  }
  return LOWER_IS_BETTER.has(norm) || norm.includes('loss') || norm.includes('error')
}

/** Format a metric value for display, choosing units by metric kind: percent
 *  metrics as "35.5%", R² with its own precision, target-space errors
 *  (MAE/RMSE) in the domain's target unit, losses compactly, everything else
 *  as a plain 3-figure number. */
export function formatMetric(domain: Domain, name: string, value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '—'
  const norm = normalizeMetricName(name)
  if (isPercentMetric(domain, name)) return fmtPct(value)
  if (norm === 'r2') return fmtR2(value)
  if (norm === 'mae' || norm === 'rmse') return fmtTargetCell(value, domain)
  if (norm === 'logloss' || norm === 'brier') return fmtLoss(value)
  if (Math.abs(value) < 1000) return value.toFixed(3)
  return value.toExponential(1)
}

/** Compare two runs' values of `name` for a leaderboard sort, respecting the
 *  metric's direction. Missing values sink. Returns <0 if `a` ranks better. */
export function compareByMetric(
  domain: Domain,
  name: string,
  a: Metrics | null | undefined,
  b: Metrics | null | undefined,
): number {
  const av = metricValue(a, name)
  const bv = metricValue(b, name)
  if (av === null && bv === null) return 0
  if (av === null) return 1
  if (bv === null) return -1
  const lower = metricLowerIsBetter(domain, name)
  if (av === bv) return 0
  return (av < bv) === lower ? -1 : 1
}
