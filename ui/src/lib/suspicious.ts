// Suspicious-prediction flagging: robust outliers on the percentage error.
// A flagged row usually means a mislabeled listing (wrong price, wrong
// currency, land priced as a house) rather than an honest model miss.
// Flags inform; they never silently change the official metrics.

import type { Metrics, Prediction } from '../api/types'

export const SUSPICIOUS_MAD_Z = 3.5

/** MAD z-score outliers over the signed % error. */
export function suspiciousRowIds(preds: Prediction[], madZ = SUSPICIOUS_MAD_Z): Set<number> {
  if (preds.length < 20) return new Set() // too few rows for robust stats
  const pcts = preds.map((p) => (p.predicted - p.actual) / p.actual)
  const sorted = [...pcts].sort((a, b) => a - b)
  const median = sorted[sorted.length >> 1]
  const devs = pcts.map((v) => Math.abs(v - median)).sort((a, b) => a - b)
  const mad = devs[devs.length >> 1]
  if (mad === 0) return new Set()
  const out = new Set<number>()
  preds.forEach((p, i) => {
    if (Math.abs((0.6745 * (pcts[i] - median)) / mad) > madZ) out.add(p.row_id)
  })
  return out
}

/** Client-side mirror of the server's price-space metric computation. */
export function computeMetrics(preds: Prediction[]): Metrics | null {
  const n = preds.length
  if (n === 0) return null
  let absErr = 0
  let sqErr = 0
  const apes: number[] = []
  const meanActual = preds.reduce((s, p) => s + p.actual, 0) / n
  let ssTot = 0
  for (const p of preds) {
    const e = p.predicted - p.actual
    absErr += Math.abs(e)
    sqErr += e * e
    ssTot += (p.actual - meanActual) ** 2
    apes.push(Math.abs(e / p.actual))
  }
  apes.sort((a, b) => a - b)
  const medape = n % 2 === 1 ? apes[(n - 1) / 2] : (apes[n / 2 - 1] + apes[n / 2]) / 2
  return {
    mae: absErr / n,
    rmse: Math.sqrt(sqErr / n),
    r2: ssTot === 0 ? 0 : 1 - sqErr / ssTot,
    mape: apes.reduce((s, v) => s + v, 0) / n,
    medape,
    n_test: n,
  }
}
