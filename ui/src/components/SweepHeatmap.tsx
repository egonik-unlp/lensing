import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { Metrics, ModelDefinition, Predictor, RunMeta } from '../api/types'
import { heatCell } from '../lib/heat'
import { useAsync } from '../hooks/useAsync'
import { fmtMoneyCell, fmtPct, fmtR2, shortRunId } from '../lib/format'
import RampLegend from './charts/RampLegend'
import './sweepheatmap.css'

interface MetricDef {
  key: keyof Metrics
  label: string
  lowerWins: boolean
  fmt: (v: number) => string
}

const METRICS: MetricDef[] = [
  { key: 'mae', label: 'MAE', lowerWins: true, fmt: fmtMoneyCell },
  { key: 'rmse', label: 'RMSE', lowerWins: true, fmt: fmtMoneyCell },
  { key: 'r2', label: 'R²', lowerWins: false, fmt: fmtR2 },
  { key: 'mape', label: 'MAPE', lowerWins: true, fmt: (v) => fmtPct(v, 0) },
  { key: 'medape', label: 'medAPE', lowerWins: true, fmt: (v) => fmtPct(v) },
]

/** Hyperparam value as a short axis label: numbers verbatim, arrays compact. */
function fmtHp(v: unknown): string {
  if (v == null) return '—'
  if (typeof v === 'number' || typeof v === 'boolean') return String(v)
  if (typeof v === 'string') return v
  return JSON.stringify(v)
}

/** Stable identity for grouping runs by a hyperparam's value. */
const hpKey = (v: unknown) => JSON.stringify(v ?? null)

interface Cell {
  x: string
  y: string
  /** Best run at this param pair (by the active metric). */
  run: RunMeta
  value: number
  n: number
}

/** Past runs of this definition as a hyperparameter-sweep heatmap.
 *  Renders nothing when there is no sweep to show: fewer than two finished
 *  runs, or no hyperparameter ever varied. Darkest cell = best, always. */
export default function SweepHeatmap({
  def,
  predictor,
}: {
  def: ModelDefinition
  predictor: Predictor | null
}) {
  const runs = useAsync(() => api.listRuns(), [])
  const [metricKey, setMetricKey] = useState<keyof Metrics>('mae')
  const [xPick, setXPick] = useState<string | null>(null)
  const [yPick, setYPick] = useState<string | null>(null)
  const [hover, setHover] = useState<{ cell: Cell; left: number; top: number } | null>(null)

  const metric = METRICS.find((m) => m.key === metricKey)!

  // Finished runs launched from this definition, with metrics. Stopped runs
  // carry full test metrics too — comparable like succeeded (same rule as
  // the compare view).
  const eligible = useMemo(
    () =>
      (runs.data ?? []).filter(
        (r) =>
          r.from_definition === def.name &&
          (r.status === 'succeeded' || r.status === 'stopped') &&
          r.metrics != null,
      ),
    [runs.data, def.name],
  )

  // Hyperparams that actually varied across those runs, most-varying first.
  const varying = useMemo(() => {
    const distinct = new Map<string, Set<string>>()
    for (const r of eligible) {
      for (const [k, v] of Object.entries(r.hyperparams)) {
        let set = distinct.get(k)
        if (!set) distinct.set(k, (set = new Set()))
        set.add(hpKey(v))
      }
    }
    return [...distinct.entries()]
      .filter(([, set]) => set.size >= 2)
      .sort((a, b) => b[1].size - a[1].size)
      .map(([k]) => k)
  }, [eligible])

  const xParam = xPick && varying.includes(xPick) ? xPick : varying[0]
  const yParam =
    yPick && varying.includes(yPick) && yPick !== xParam
      ? yPick
      : (varying.find((k) => k !== xParam) ?? null)

  const grid = useMemo(() => {
    if (!xParam) return null
    const axisValues = (param: string) => {
      const seen = new Map<string, unknown>()
      for (const r of eligible) seen.set(hpKey(r.hyperparams[param]), r.hyperparams[param])
      const vals = [...seen.values()]
      const numeric = vals.every((v) => typeof v === 'number')
      vals.sort((a, b) =>
        numeric ? (a as number) - (b as number) : fmtHp(a).localeCompare(fmtHp(b)),
      )
      return vals
    }
    const xVals = axisValues(xParam)
    const yVals = yParam ? axisValues(yParam) : [null]

    const byPair = new Map<string, RunMeta[]>()
    for (const r of eligible) {
      const k = `${hpKey(r.hyperparams[xParam])}|${yParam ? hpKey(r.hyperparams[yParam]) : ''}`
      const bucket = byPair.get(k)
      if (bucket) bucket.push(r)
      else byPair.set(k, [r])
    }

    const cells = new Map<string, Cell>()
    let lo = Infinity
    let hi = -Infinity
    for (const [k, group] of byPair) {
      const best = group.reduce((a, b) => {
        const va = a.metrics![metric.key]
        const vb = b.metrics![metric.key]
        return (metric.lowerWins ? vb < va : vb > va) ? b : a
      })
      const value = best.metrics![metric.key]
      if (isFinite(value)) {
        lo = Math.min(lo, value)
        hi = Math.max(hi, value)
      }
      cells.set(k, { x: k.split('|')[0], y: k.split('|')[1], run: best, value, n: group.length })
    }
    // t=1 (darkest) is always the best end, whichever direction the metric runs.
    const span = hi - lo
    const tOf = (v: number) => {
      if (!isFinite(v)) return 0
      if (span <= 0) return 0.5
      return metric.lowerWins ? (hi - v) / span : (v - lo) / span
    }
    const bestValue = metric.lowerWins ? lo : hi
    return { xVals, yVals, cells, tOf, lo, hi, bestValue }
  }, [eligible, xParam, yParam, metric])

  const paramLabel = (name: string) =>
    predictor?.params.find((p) => p.name === name)?.label ?? name

  if (runs.loading) {
    return (
      <div className="panel">
        <h2>Hyperparameter sweep</h2>
        <div className="skeleton" style={{ height: '8rem' }} />
      </div>
    )
  }
  if (runs.error) {
    return (
      <div className="panel">
        <h2>Hyperparameter sweep</h2>
        <div className="error-block" role="alert">
          Could not load runs: {runs.error}{' '}
          <button className="btn" onClick={runs.reload}>
            Retry
          </button>
        </div>
      </div>
    )
  }
  // No sweep to show: stay out of the way entirely.
  if (eligible.length < 2 || !grid || varying.length === 0) return null

  const worstLabel = metric.fmt(metric.lowerWins ? grid.hi : grid.lo)
  const bestLabel = metric.fmt(grid.bestValue)

  return (
    <div className="panel" aria-label="Hyperparameter sweep">
      <h2>Hyperparameter sweep</h2>
      <p className="muted contract-line">
        <span className="num">{eligible.length}</span> finished runs · colored by{' '}
        <span className="num">{metric.label}</span> · darkest is best
      </p>

      <div className="sweep-controls">
        <div className="scale-toggle" role="group" aria-label="Sweep metric">
          <span className="muted">metric</span>
          {METRICS.map((m) => (
            <button
              key={m.key}
              className={`scale-btn${metricKey === m.key ? ' is-active' : ''}`}
              aria-pressed={metricKey === m.key}
              onClick={() => {
                setMetricKey(m.key)
                setHover(null)
              }}
            >
              {m.label}
            </button>
          ))}
        </div>
        {varying.length > 2 && (
          <div className="sweep-axis-picks">
            <label>
              x{' '}
              <select value={xParam} onChange={(e) => setXPick(e.target.value)}>
                {varying.map((k) => (
                  <option key={k} value={k}>
                    {paramLabel(k)}
                  </option>
                ))}
              </select>
            </label>
            <label>
              y{' '}
              <select value={yParam ?? ''} onChange={(e) => setYPick(e.target.value)}>
                {varying
                  .filter((k) => k !== xParam)
                  .map((k) => (
                    <option key={k} value={k}>
                      {paramLabel(k)}
                    </option>
                  ))}
              </select>
            </label>
          </div>
        )}
      </div>

      <div
        className="sweep-grid"
        style={{ gridTemplateColumns: `max-content repeat(${grid.xVals.length}, minmax(4.5rem, max-content))` }}
        onMouseLeave={() => setHover(null)}
      >
        <div className="sweep-corner">
          {yParam && <span>{paramLabel(yParam)} ↓</span>}
          <span>{paramLabel(xParam)} →</span>
        </div>
        {grid.xVals.map((xv) => (
          <div key={hpKey(xv)} className="sweep-col-head">
            {fmtHp(xv)}
          </div>
        ))}
        {grid.yVals.map((yv) => {
          const yk = yParam ? hpKey(yv) : ''
          return (
            <div key={`row-${yk}`} className="sweep-row" role="presentation">
              <div className="sweep-row-head">{yParam ? fmtHp(yv) : ''}</div>
              {grid.xVals.map((xv) => {
                const cell = grid.cells.get(`${hpKey(xv)}|${yk}`)
                if (!cell) {
                  return (
                    <div
                      key={hpKey(xv)}
                      className="sweep-cell sweep-cell-empty"
                      aria-label={`${paramLabel(xParam)} ${fmtHp(xv)}${yParam ? `, ${paramLabel(yParam)} ${fmtHp(yv)}` : ''}: not measured`}
                    />
                  )
                }
                const { bg, text } = heatCell(grid.tOf(cell.value))
                const isBest = cell.value === grid.bestValue
                const pairLabel = `${paramLabel(xParam)} ${fmtHp(xv)}${yParam ? ` × ${paramLabel(yParam)} ${fmtHp(yv)}` : ''}`
                return (
                  <Link
                    key={hpKey(xv)}
                    to={`/runs/${cell.run.run_id}`}
                    className={`sweep-cell${isBest ? ' is-best' : ''}`}
                    style={{ background: bg, color: text }}
                    aria-label={`${pairLabel}: ${metric.label} ${metric.fmt(cell.value)}, ${
                      cell.n > 1 ? `best of ${cell.n} runs, ` : ''
                    }open run ${shortRunId(cell.run.run_id)}`}
                    onMouseEnter={(e) => {
                      const el = e.currentTarget
                      setHover({
                        cell,
                        left: el.offsetLeft + el.offsetWidth / 2,
                        top: el.offsetTop,
                      })
                    }}
                    onFocus={() => setHover(null)}
                  >
                    {isBest && <span aria-hidden="true">✓ </span>}
                    {metric.fmt(cell.value)}
                  </Link>
                )
              })}
            </div>
          )
        })}
        {hover && (
          <div className="chart-tooltip" style={{ left: hover.left, top: hover.top }}>
            {`${paramLabel(xParam)} ${fmtHp(JSON.parse(hover.cell.x))}${
              yParam ? ` × ${paramLabel(yParam)} ${fmtHp(JSON.parse(hover.cell.y))}` : ''
            }\n${metric.label} ${metric.fmt(hover.cell.value)} · ${shortRunId(hover.cell.run.run_id)}${
              hover.cell.n > 1 ? ` (n=${hover.cell.n})` : ''
            }`}
          </div>
        )}
      </div>

      <RampLegend
        minLabel={worstLabel}
        maxLabel={`${bestLabel} ✓`}
        label={metric.label}
      />
    </div>
  )
}
