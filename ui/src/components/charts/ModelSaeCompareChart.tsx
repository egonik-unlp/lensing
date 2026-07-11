import { useMemo, useState } from 'react'
import type { ModelSaeReport } from '../../api/types'
import { linearScale, niceTicks } from '../../lib/scale'
import { CHART_PALETTE as PALETTE } from './palette'
import './charts.css'

const W = 680
const H = 300
const M = { top: 16, right: 20, bottom: 56, left: 62 }

type Metric = 'r2_log' | 'concepts' | 'utilization'

const METRICS: { key: Metric; label: string; fmt: (v: number) => string }[] = [
  { key: 'r2_log', label: 'decodability (R² log)', fmt: (v) => v.toFixed(2) },
  { key: 'concepts', label: 'interpretable concepts', fmt: (v) => v.toFixed(0) },
  { key: 'utilization', label: 'capacity used', fmt: (v) => `${(v * 100).toFixed(0)}%` },
]

export interface ModelSaeSeries {
  model: string
  report: ModelSaeReport
}

/** One per-layer point of the chosen metric. */
interface Pt {
  layer: number
  value: number | null
}

function points(report: ModelSaeReport, metric: Metric): Pt[] {
  return report.layers.map((l) => {
    const cvd = report.concept_vs_decodability.find((c) => c.layer === l.layer)
    const value =
      metric === 'r2_log'
        ? cvd?.linear_r2_log ?? null
        : metric === 'concepts'
          ? l.n_interpretable_concepts
          : l.capacity.utilization
    return { layer: l.layer, value }
  })
}

/** Overlay of a per-hidden-layer model-SAE metric across relative depth for
 *  several models — the read the leaderboard can't give: where each model forms
 *  concepts, how much of each layer it uses, and where the target becomes decodable.
 *  X is relative depth (first hidden layer = 0 … deepest = 1) so nets of
 *  different depths align. Vanilla SVG, matching the lab's other charts. */
export default function ModelSaeCompareChart({ series }: { series: ModelSaeSeries[] }) {
  const [metric, setMetric] = useState<Metric>('r2_log')
  const [hover, setHover] = useState<{ s: number; i: number } | null>(null)

  const perModel = useMemo(
    () => series.map((s) => ({ model: s.model, pts: points(s.report, metric) })),
    [series, metric],
  )
  const fmt = METRICS.find((m) => m.key === metric)!.fmt

  const { xScale, yScale, yTicks } = useMemo(() => {
    const vals = perModel.flatMap((s) => s.pts.map((p) => p.value).filter((v): v is number => v != null))
    let lo = vals.length ? Math.min(...vals) : 0
    let hi = vals.length ? Math.max(...vals) : 1
    // Frame the data, not an arbitrary zero baseline — the A-vs-B gap is the
    // point, and for R²≈0.7 anchoring at 0 buries it in empty space.
    const pad = (hi - lo) * 0.15 || Math.max(Math.abs(hi), 1) * 0.15
    lo -= pad
    hi += pad
    if (metric === 'utilization') {
      lo = Math.max(0, lo)
      hi = Math.min(1.02, hi)
    }
    return {
      xScale: linearScale([0, 1], [M.left, W - M.right]),
      yScale: linearScale([lo, hi], [H - M.bottom, M.top]),
      yTicks: niceTicks(lo, hi, 5),
    }
  }, [perModel, metric])

  const nx = (i: number, n: number) => (n <= 1 ? 0.5 : i / (n - 1))
  const label = METRICS.find((m) => m.key === metric)!.label

  return (
    <div className="interp-result">
      <div className="scale-toggle interp-metric-toggle">
        {METRICS.map((mk) => (
          <button
            key={mk.key}
            className={`scale-btn${metric === mk.key ? ' is-active' : ''}`}
            onClick={() => setMetric(mk.key)}
          >
            {mk.label}
          </button>
        ))}
      </div>
      <div
        className="chart"
        role="img"
        aria-label={`${label} across depth for ${series.length} models`}
      >
        <svg viewBox={`0 0 ${W} ${H}`} onMouseLeave={() => setHover(null)}>
          {yTicks.map((t) => (
            <g key={`y${t}`}>
              <line className="grid-line" x1={M.left} x2={W - M.right} y1={yScale.map(t)} y2={yScale.map(t)} />
              <text className="tick-label" x={M.left - 8} y={yScale.map(t) + 3} textAnchor="end">
                {fmt(t)}
              </text>
            </g>
          ))}
          <line className="axis-line" x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom} />
          <text className="tick-label" x={M.left} y={H - M.bottom + 16} textAnchor="start">
            first hidden
          </text>
          <text className="tick-label" x={W - M.right} y={H - M.bottom + 16} textAnchor="end">
            output head
          </text>
          <text className="axis-title" x={(M.left + W - M.right) / 2} y={H - 26} textAnchor="middle">
            relative hidden depth
          </text>

          {perModel.map((s, si) => {
            const color = PALETTE[si % PALETTE.length]
            const pts = s.pts.filter((p) => p.value != null)
            const n = s.pts.length
            const d = pts
              .map(
                (p, k) =>
                  `${k ? 'L' : 'M'}${xScale.map(nx(s.pts.indexOf(p), n)).toFixed(1)},${yScale
                    .map(p.value as number)
                    .toFixed(1)}`,
              )
              .join('')
            return (
              <g key={s.model}>
                <path d={d} fill="none" stroke={color} strokeWidth={1.75} />
                {s.pts.map((p, i) =>
                  p.value == null ? null : (
                    <circle
                      key={i}
                      cx={xScale.map(nx(i, n))}
                      cy={yScale.map(p.value)}
                      r={hover && hover.s === si && hover.i === i ? 5 : 3}
                      fill={color}
                      onMouseEnter={() => setHover({ s: si, i })}
                    />
                  ),
                )}
              </g>
            )
          })}
        </svg>

        <div className="probe-legend">
          {perModel.map((s, si) => (
            <span key={s.model} className="probe-legend-item">
              <span className="probe-legend-swatch" style={{ background: PALETTE[si % PALETTE.length] }} />
              {s.model}
            </span>
          ))}
        </div>

        {hover && perModel[hover.s].pts[hover.i].value != null && (
          <div
            className="chart-tooltip"
            style={{
              left: `${(xScale.map(nx(hover.i, perModel[hover.s].pts.length)) / W) * 100}%`,
              top: `${(yScale.map(perModel[hover.s].pts[hover.i].value as number) / H) * 100}%`,
            }}
          >
            <strong>{perModel[hover.s].model}</strong>
            <br />
            hidden {perModel[hover.s].pts[hover.i].layer} · {label}:{' '}
            {fmt(perModel[hover.s].pts[hover.i].value as number)}
          </div>
        )}
      </div>
    </div>
  )
}
