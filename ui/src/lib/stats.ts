// Pure EDA helpers: binning, quantiles, category tallies, discrete
// derivatives. No formatting, no React.

export interface HistogramBins {
  /** bin edges, length counts.length + 1; log-space when log binning */
  edges: number[]
  counts: number[]
}

/** Equal-width bins over [min, max]; `log` bins in log10 space (values ≤ 0
 *  are dropped — callers route those to a dedicated rule/label). */
export function histogram(values: number[], opts: { bins: number; log?: boolean }): HistogramBins {
  const xs = opts.log ? values.filter((v) => v > 0).map(Math.log10) : values.slice()
  const edges: number[] = []
  const counts = new Array<number>(opts.bins).fill(0)
  if (xs.length === 0) return { edges, counts }
  let lo = Math.min(...xs)
  let hi = Math.max(...xs)
  if (lo === hi) {
    // Degenerate: one distinct value; widen so the single bar renders.
    lo -= 0.5
    hi += 0.5
  }
  const w = (hi - lo) / opts.bins
  for (let i = 0; i <= opts.bins; i++) edges.push(lo + i * w)
  for (const x of xs) {
    const b = Math.min(opts.bins - 1, Math.floor((x - lo) / w))
    counts[b]++
  }
  if (opts.log) {
    for (let i = 0; i < edges.length; i++) edges[i] = Math.pow(10, edges[i])
  }
  return { edges, counts }
}

/** Count values into a precomputed edge set (for overlay series sharing the
 *  base histogram's bins). Values outside [first, last] edge are dropped. */
export function countIntoBins(values: number[], edges: number[], log?: boolean): number[] {
  const n = edges.length - 1
  const counts = new Array<number>(Math.max(0, n)).fill(0)
  if (n < 1) return counts
  const e = log ? edges.map(Math.log10) : edges
  const lo = e[0]
  const w = (e[n] - lo) / n
  for (const raw of values) {
    if (log && raw <= 0) continue
    const x = log ? Math.log10(raw) : raw
    if (x < e[0] || x > e[n]) continue
    counts[Math.min(n - 1, Math.floor((x - lo) / w))]++
  }
  return counts
}

/** Linear-interpolated quantiles, ps in [0, 1]. */
export function quantiles(values: number[], ps: number[]): number[] {
  if (values.length === 0) return ps.map(() => NaN)
  const xs = [...values].sort((a, b) => a - b)
  return ps.map((p) => {
    const t = p * (xs.length - 1)
    const i = Math.floor(t)
    const f = t - i
    return i + 1 < xs.length ? xs[i] * (1 - f) + xs[i + 1] * f : xs[i]
  })
}

export interface CategoryStat {
  value: string
  count: number
  /** fraction of the tallied total */
  share: number
  medianTarget: number
}

/** Tally a categorical field with per-category median target value; top `n`
 *  by count, the rest folded into an "(other)" bucket when it exists. */
export function topCategories(
  rows: { key: string; value: number }[],
  n: number,
): CategoryStat[] {
  const groups = new Map<string, number[]>()
  for (const r of rows) {
    const k = r.key.trim() || '(empty)'
    const g = groups.get(k)
    if (g) g.push(r.value)
    else groups.set(k, [r.value])
  }
  const stats: CategoryStat[] = [...groups.entries()].map(([value, values]) => ({
    value,
    count: values.length,
    share: values.length / rows.length,
    medianTarget: quantiles(values, [0.5])[0],
  }))
  stats.sort((a, b) => b.count - a.count)
  if (stats.length <= n) return stats
  const head = stats.slice(0, n)
  const tail = stats.slice(n)
  const tailValues = tail.flatMap((s) => {
    const g = groups.get(s.value)
    return g ?? []
  })
  head.push({
    value: `(other · ${tail.length})`,
    count: tailValues.length,
    share: tailValues.length / rows.length,
    medianTarget: quantiles(tailValues, [0.5])[0],
  })
  return head
}

/* ---------------- discrete derivatives (training dynamics) ---------------- */

export interface DerivPoint {
  epoch: number
  value: number
}

/** First derivative by central differences; forward/backward at the ends.
 *  Assumes a contiguous epoch series (epoch step = 1). */
export function diffSeries(epochs: number[], values: number[]): DerivPoint[] {
  const n = values.length
  if (n < 2) return []
  const out: DerivPoint[] = []
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - 1)
    const hi = Math.min(n - 1, i + 1)
    const de = epochs[hi] - epochs[lo]
    if (de === 0) continue
    out.push({ epoch: epochs[i], value: (values[hi] - values[lo]) / de })
  }
  return out
}

/** Second derivative by second differences (interior points only). */
export function diff2Series(epochs: number[], values: number[]): DerivPoint[] {
  const n = values.length
  if (n < 3) return []
  const out: DerivPoint[] = []
  for (let i = 1; i < n - 1; i++) {
    const h1 = epochs[i] - epochs[i - 1]
    const h2 = epochs[i + 1] - epochs[i]
    if (h1 === 0 || h2 === 0) continue
    // Uneven-grid second difference; reduces to f[i+1]−2f[i]+f[i−1] when h1=h2=1.
    const d2 =
      (2 * (h1 * values[i + 1] - (h1 + h2) * values[i] + h2 * values[i - 1])) /
      (h1 * h2 * (h1 + h2))
    out.push({ epoch: epochs[i], value: d2 })
  }
  return out
}

/** Centered moving average over a window of `k` points (k odd). */
export function smoothSeries(points: DerivPoint[], k: number): DerivPoint[] {
  if (k <= 1 || points.length < k) return points
  const half = Math.floor(k / 2)
  return points.map((p, i) => {
    const lo = Math.max(0, i - half)
    const hi = Math.min(points.length - 1, i + half)
    let sum = 0
    for (let j = lo; j <= hi; j++) sum += points[j].value
    return { epoch: p.epoch, value: sum / (hi - lo + 1) }
  })
}
