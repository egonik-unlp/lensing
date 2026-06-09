// Number formatting: real units, no rounding away inconvenient digits.

import type { Domain, TargetFormat } from './domain'

/* ---------------- domain-driven target formatting ----------------
 * Every predicted/actual/error value flows through these. They render the
 * target field per domain.target.format: the symbol prefix for "money" style
 * (none for "number"), and the locale's grouping. The target's unit comes
 * from the domain, never hardcoded — a money domain shows "$1,234", a count
 * or score domain shows "1,234". There are deliberately NO currency-baked
 * formatters; reintroducing one would re-price every non-money instance. */

const fmtCache = new Map<string, Intl.NumberFormat>()
function grouped(locale: string): Intl.NumberFormat {
  let f = fmtCache.get(locale)
  if (!f) {
    f = new Intl.NumberFormat(locale, { maximumFractionDigits: 0 })
    fmtCache.set(locale, f)
  }
  return f
}
const compactCache = new Map<string, Intl.NumberFormat>()
function groupedCompact(locale: string): Intl.NumberFormat {
  let f = compactCache.get(locale)
  if (!f) {
    f = new Intl.NumberFormat(locale, { notation: 'compact', maximumSignificantDigits: 3 })
    compactCache.set(locale, f)
  }
  return f
}

function symbolFor(fmt: TargetFormat): string {
  return fmt.style === 'money' ? fmt.symbol : ''
}

/** Target value: "$1,234,567" (money style) or "1,234,567" (number style).
 *  Sub-thousand values keep their digits. */
export function fmtTarget(v: number, domain: Domain): string {
  const fmt = domain.target.format
  return `${symbolFor(fmt)}${grouped(fmt.locale).format(Math.round(v))}`
}

/** List-column target: full digits up to 10M, then compact ("$1.85B") so a
 *  diverged run can't blow the table out of the viewport. */
export function fmtTargetCell(v: number, domain: Domain): string {
  const a = Math.abs(v)
  if (!isFinite(v)) return '—'
  if (a < 10_000_000) return fmtTarget(v, domain)
  const fmt = domain.target.format
  return `${v < 0 ? '−' : ''}${symbolFor(fmt)}${groupedCompact(fmt.locale).format(a)}`
}

/** Signed target delta: "+$12,430" / "−$12,430". */
export function fmtTargetDelta(v: number, domain: Domain): string {
  const sign = v < 0 ? '−' : '+'
  return `${sign}${fmtTarget(Math.abs(v), domain)}`
}

/** Decade-tick label for log axes: "10k" / "$1M", symbol per the domain's
 *  target format. The domain-aware counterpart of a fixed "$10k" tick. */
export function fmtTargetTick(v: number, domain: Domain): string {
  const sym = symbolFor(domain.target.format)
  if (v >= 1e6) return `${sym}${v / 1e6}M`
  if (v >= 1e3) return `${sym}${v / 1e3}k`
  return `${sym}${v}`
}

/** Fractions to percent: 0.355 → "35.5%". Absurd magnitudes (diverged runs)
 *  switch to exponential so they stay one column wide. */
export function fmtPct(frac: number, digits = 1): string {
  const pct = frac * 100
  if (!isFinite(pct)) return '—'
  if (Math.abs(pct) >= 100_000) return `${pct.toExponential(1)}%`
  return `${pct.toFixed(digits)}%`
}

export function fmtSignedPct(frac: number, digits = 1): string {
  const sign = frac < 0 ? '−' : '+'
  return `${sign}${(Math.abs(frac) * 100).toFixed(digits)}%`
}

export function fmtR2(v: number): string {
  if (!isFinite(v)) return '—'
  // A diverged run's R² of −2,597,472,592.563 is still bad news at "−2.6e9".
  if (Math.abs(v) >= 1000) return v.toExponential(1)
  return v.toFixed(3)
}

export function fmtLoss(v: number): string {
  if (v === 0) return '0'
  if (v >= 100 || v < 0.001) return v.toExponential(2)
  return v.toPrecision(3)
}

/** Table timestamp: "06-04 11:42" — fixed width, sorts visually. */
export function fmtStamp(iso: string): string {
  const d = new Date(iso)
  const p = (n: number) => String(n).padStart(2, '0')
  return `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`
}

export function fmtDateTime(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function fmtDuration(startIso: string, endIso: string | null): string {
  const end = endIso ? new Date(endIso).getTime() : Date.now()
  const s = Math.max(0, Math.round((end - new Date(startIso).getTime()) / 1000))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ${s % 60}s`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}

/** Short run ids for display: drop the date prefix, keep the tail. */
export function shortRunId(runId: string): string {
  return runId.replace(/^run-\d{8}-/, '')
}

/** Tiny run ids for tables that already show the predictor in its own
 *  column: time + hash only ("140002-35065"). */
export function tinyRunId(runId: string): string {
  const short = shortRunId(runId)
  return /^\d+-[0-9a-z]+/.exec(short)?.[0] ?? short
}

/** Short dataset ids for display: drop the date prefix, keep the tail. */
export function shortDatasetId(datasetId: string): string {
  return datasetId.replace(/^ds-\d{8}-/, '')
}

/** Capitalize the first letter — for domain nouns used at the start of copy. */
export function cap(s: string): string {
  return s ? s[0].toUpperCase() + s.slice(1) : s
}
