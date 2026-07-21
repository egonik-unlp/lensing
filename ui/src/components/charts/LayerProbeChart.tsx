import { useMemo, useState } from 'react'
import type { LayerProbeStage } from '../../api/types'
import { linearScale, niceTicks } from '../../lib/scale'
import { fmtTargetTick } from '../../lib/format'
import { useDomain } from '../../lib/DomainContext'
import './charts.css'

export type ProbeMetric = 'test_r2_log' | 'test_mae' | 'test_medape'

const PROBE_METRICS: Record<ProbeMetric, { label: string; fmt: (v: number) => string }> = {
  test_r2_log: { label: 'test R² (log space)', fmt: (v) => v.toFixed(3) },
  test_mae: { label: 'test MAE', fmt: (v) => `${Math.round(v / 1000)}k` },
  test_medape: { label: 'test medAPE (%)', fmt: (v) => `${v.toFixed(1)}%` },
}

const W = 680
const H = 300
const M = { top: 16, right: 20, bottom: 50, left: 62 }

/** Short axis label for a stage name (input / hidden_3 / output (model)). */
function shortName(name: string): string {
  if (name.startsWith('hidden_')) return `h${name.slice(7)}`
  if (name.startsWith('output')) return 'out'
  return name
}

interface Props {
  stages: LayerProbeStage[]
  metric: ProbeMetric
}

/** Per-stage probe metric across model depth (Alain & Bengio). Vanilla SVG,
 *  matching the lab's other charts. The dashed final point is the model's own
 *  output (a reference, not a fitted probe). */
export default function LayerProbeChart({ stages, metric }: Props) {
  const [hover, setHover] = useState<number | null>(null)
  const m = PROBE_METRICS[metric]
  const domain = useDomain()
  const fmtVal = (v: number) => (metric === 'test_mae' ? fmtTargetTick(v, domain) : m.fmt(v))

  const { xs, ys, path, yTicks } = useMemo(() => {
    const vals = stages.map((s) => s[metric])
    let lo = Math.min(...vals)
    let hi = Math.max(...vals)
    if (metric === 'test_r2_log') {
      lo = Math.min(lo, 0)
      hi = Math.max(hi, 0)
    }
    const pad = (hi - lo) * 0.1 || 1
    lo -= pad
    hi += pad
    const xScale = linearScale([0, Math.max(stages.length - 1, 1)], [M.left, W - M.right])
    const yScale = linearScale([lo, hi], [H - M.bottom, M.top])
    const d = stages
      .map((s, i) => `${i ? 'L' : 'M'}${xScale.map(i).toFixed(1)},${yScale.map(s[metric]).toFixed(1)}`)
      .join('')
    return { xs: xScale, ys: yScale, path: d, yTicks: niceTicks(lo, hi, 5) }
  }, [stages, metric])

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const px = ((e.clientX - rect.left) / rect.width) * W
    let best = 0
    let bestD = Infinity
    stages.forEach((_, i) => {
      const d = Math.abs(xs.map(i) - px)
      if (d < bestD) {
        bestD = d
        best = i
      }
    })
    setHover(bestD < 40 ? best : null)
  }

  const hv = hover != null ? stages[hover] : null

  return (
    <div className="chart" role="img" aria-label={`${m.label} per stage across ${stages.length} stages`}>
      <svg viewBox={`0 0 ${W} ${H}`} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line className="grid-line" x1={M.left} x2={W - M.right} y1={ys.map(t)} y2={ys.map(t)} />
            <text className="tick-label" x={M.left - 8} y={ys.map(t) + 3} textAnchor="end">
              {fmtVal(t)}
            </text>
          </g>
        ))}
        {stages.map((s, i) => (
          <text key={`x${i}`} className="tick-label" x={xs.map(i)} y={H - M.bottom + 16} textAnchor="middle">
            {shortName(s.name)}
          </text>
        ))}
        <line className="axis-line" x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom} />
        <text className="axis-title" x={(M.left + W - M.right) / 2} y={H - 6} textAnchor="middle">
          stage (input → hidden layers → output)
        </text>

        <path d={path} fill="none" stroke="var(--series-a, #2563eb)" strokeWidth={1.75} />
        {stages.map((s, i) => {
          const isOutput = s.lambda == null
          return (
            <circle
              key={i}
              cx={xs.map(i)}
              cy={ys.map(s[metric])}
              r={hover === i ? 4.5 : 3}
              fill={isOutput ? 'var(--bg, #fff)' : 'var(--series-a, #2563eb)'}
              stroke="var(--series-a, #2563eb)"
              strokeWidth={isOutput ? 1.75 : 0}
            />
          )
        })}

        {hv && (
          <line
            className="grid-line"
            x1={xs.map(hover!)}
            x2={xs.map(hover!)}
            y1={M.top}
            y2={H - M.bottom}
          />
        )}
      </svg>
      {hv && (
        <div
          className="chart-tooltip"
          style={{
            left: `${(xs.map(hover!) / W) * 100}%`,
            top: `${(ys.map(hv[metric]) / H) * 100}%`,
          }}
        >
          <strong>{hv.name}</strong> · dim {hv.dim}
          <br />
          {m.label}: {fmtVal(hv[metric])}
          <br />
          logR² {hv.test_r2_log.toFixed(3)} · MAE {fmtTargetTick(hv.test_mae, domain)} · medAPE{' '}
          {hv.test_medape.toFixed(1)}%
        </div>
      )}
    </div>
  )
}
