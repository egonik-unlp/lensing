import { useMemo, useState } from 'react'
import type { LayerProbeStage } from '../../api/types'
import { linearScale, niceTicks } from '../../lib/scale'
import { fmtTargetTick } from '../../lib/format'
import { useDomain } from '../../lib/DomainContext'
import type { ProbeMetric } from './LayerProbeChart'
import './charts.css'

const W = 680
const H = 320
const M = { top: 16, right: 20, bottom: 56, left: 62 }

// Distinguishable line colours for the overlaid models (cycled past 8).
const PALETTE = ['#2563eb', '#dc2626', '#16a34a', '#d97706', '#7c3aed', '#0891b2', '#db2777', '#65a30d']

const FMT: Record<ProbeMetric, (v: number) => string> = {
  test_r2_log: (v) => v.toFixed(2),
  test_mae: (v) => `${Math.round(v / 1000)}k`,
  test_medape: (v) => `${v.toFixed(0)}%`,
}
const LABEL: Record<ProbeMetric, string> = {
  test_r2_log: 'test R² (log space)',
  test_mae: 'test MAE',
  test_medape: 'test medAPE (%)',
}

export interface ProbeSeries {
  model: string
  stages: LayerProbeStage[]
}

interface Props {
  series: ProbeSeries[]
  metric: ProbeMetric
}

/** Overlay of the per-stage probe metric across model depth for several models.
 *  X is RELATIVE depth (input = 0 … output = 1) so models of different depths
 *  align; the hollow final marker on each line is that model's own output (a
 *  reference, not a fitted probe). Vanilla SVG, matching the lab's charts. */
export default function LayerProbeCompareChart({ series, metric }: Props) {
  const [hover, setHover] = useState<{ s: number; i: number } | null>(null)
  const domain = useDomain()
  const fmtVal = (v: number) => (metric === 'test_mae' ? fmtTargetTick(v, domain) : FMT[metric](v))

  const { xScale, yScale, yTicks } = useMemo(() => {
    const vals = series.flatMap((s) => s.stages.map((st) => st[metric]))
    let lo = vals.length ? Math.min(...vals) : 0
    let hi = vals.length ? Math.max(...vals) : 1
    if (metric === 'test_r2_log') {
      lo = Math.min(lo, 0)
      hi = Math.max(hi, 0)
    }
    const pad = (hi - lo) * 0.1 || 1
    lo -= pad
    hi += pad
    return {
      xScale: linearScale([0, 1], [M.left, W - M.right]),
      yScale: linearScale([lo, hi], [H - M.bottom, M.top]),
      yTicks: niceTicks(lo, hi, 5),
    }
  }, [series, metric])

  // Relative depth of stage i within an n-stage series.
  const nx = (i: number, n: number) => (n <= 1 ? 0.5 : i / (n - 1))

  return (
    <div className="chart" role="img" aria-label={`${LABEL[metric]} across depth for ${series.length} models`}>
      <svg viewBox={`0 0 ${W} ${H}`} onMouseLeave={() => setHover(null)}>
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line className="grid-line" x1={M.left} x2={W - M.right} y1={yScale.map(t)} y2={yScale.map(t)} />
            <text className="tick-label" x={M.left - 8} y={yScale.map(t) + 3} textAnchor="end">
              {fmtVal(t)}
            </text>
          </g>
        ))}
        <line className="axis-line" x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom} />
        <text className="tick-label" x={M.left} y={H - M.bottom + 16} textAnchor="middle">
          input
        </text>
        <text className="tick-label" x={W - M.right} y={H - M.bottom + 16} textAnchor="middle">
          output
        </text>
        <text className="axis-title" x={(M.left + W - M.right) / 2} y={H - 26} textAnchor="middle">
          relative depth (input → output)
        </text>

        {series.map((s, si) => {
          const color = PALETTE[si % PALETTE.length]
          const n = s.stages.length
          const d = s.stages
            .map(
              (st, i) =>
                `${i ? 'L' : 'M'}${xScale.map(nx(i, n)).toFixed(1)},${yScale.map(st[metric]).toFixed(1)}`,
            )
            .join('')
          return (
            <g key={s.model}>
              <path d={d} fill="none" stroke={color} strokeWidth={1.75} />
              {s.stages.map((st, i) => {
                const isOutput = st.lambda == null
                const on = hover && hover.s === si && hover.i === i
                return (
                  <circle
                    key={i}
                    cx={xScale.map(nx(i, n))}
                    cy={yScale.map(st[metric])}
                    r={on ? 5 : 3}
                    fill={isOutput ? 'var(--bg, #fff)' : color}
                    stroke={color}
                    strokeWidth={isOutput ? 1.75 : 0}
                    onMouseEnter={() => setHover({ s: si, i })}
                  />
                )
              })}
            </g>
          )
        })}
      </svg>

      <div className="probe-legend">
        {series.map((s, si) => (
          <span key={s.model} className="probe-legend-item">
            <span className="probe-legend-swatch" style={{ background: PALETTE[si % PALETTE.length] }} />
            {s.model}
          </span>
        ))}
      </div>

      {hover && (
        <div
          className="chart-tooltip"
          style={{
            left: `${(xScale.map(nx(hover.i, series[hover.s].stages.length)) / W) * 100}%`,
            top: `${(yScale.map(series[hover.s].stages[hover.i][metric]) / H) * 100}%`,
          }}
        >
          <strong>{series[hover.s].model}</strong>
          <br />
          {series[hover.s].stages[hover.i].name} · {LABEL[metric]}:{' '}
          {fmtVal(series[hover.s].stages[hover.i][metric])}
        </div>
      )}
    </div>
  )
}
