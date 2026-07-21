import { useMemo } from 'react'
import type { EmbSegment, EmbViewResult } from '../../api/types'
import { linearScale, niceTicks } from '../../lib/scale'
import { fmtTargetTick } from '../../lib/format'
import { useDomain } from '../../lib/DomainContext'
import './charts.css'

const W = 720
const H = 320
// Generous bottom margin: segment labels (esp. long ones) are drawn vertical.
const M = { top: 16, right: 16, bottom: 112, left: 64 }

/** One bar series = a (probe, view) pair drawn for every segment. */
interface Series {
  key: string
  label: string
  color: string
  get: (v: EmbViewResult) => number | undefined
  view: string // which view's EmbViewResult to read ('' = first/any, for the floor)
}

/** Tidy a segment name for the (vertical) axis label; truncate very long ones. */
const segLabel = (name: string) => {
  const n = name.replace(' (pooled)', '').replace(' (reference)', '')
  return n.length > 20 ? `${n.slice(0, 19)}…` : n
}

interface Props {
  segments: EmbSegment[]
  views: string[]
}

/** Per-segment test MAE: the segment-median floor vs linear/MLP probes on each
 *  embedding view (PCA-128 vs raw). Mirrors the P1 results figure — the probe
 *  beating the floor means signal is present; raw ≪ PCA-128 on a segment means
 *  PCA-128's variance ranking discarded it. Vanilla SVG. */
export default function EmbeddingProbeChart({ segments, views }: Props) {
  const domain = useDomain()
  // Build the bar series present in the data, in a readable order.
  const series: Series[] = useMemo(() => {
    const out: Series[] = [
      { key: 'floor', label: 'segment-median floor', color: '#9ca3af', view: '', get: (v) => v.type_median?.mae },
    ]
    const lin = ['#93c5fd', '#2563eb']
    const mlp = ['#86efac', '#16a34a']
    views.forEach((vw, i) => {
      out.push({ key: `lin:${vw}`, label: `linear · ${vw}`, color: lin[i] ?? '#2563eb', view: vw, get: (v) => v.seg_linear?.mae })
    })
    const anyMlp = segments.some((s) => s.views.some((v) => v.seg_mlp))
    if (anyMlp) {
      views.forEach((vw, i) => {
        out.push({ key: `mlp:${vw}`, label: `MLP · ${vw}`, color: mlp[i] ?? '#16a34a', view: vw, get: (v) => v.seg_mlp?.mae })
      })
    }
    return out
  }, [segments, views])

  const viewOf = (s: EmbSegment, view: string): EmbViewResult | undefined =>
    view ? s.views.find((v) => v.view === view) : s.views[0]

  const maxMae = useMemo(() => {
    let m = 0
    for (const s of segments) {
      for (const ser of series) {
        const vr = viewOf(s, ser.view)
        const val = vr ? ser.get(vr) : undefined
        if (val != null && isFinite(val)) m = Math.max(m, val)
      }
    }
    return m || 1
  }, [segments, series])

  const segW = (W - M.left - M.right) / Math.max(segments.length, 1)
  const inner = segW * 0.84
  const barW = inner / Math.max(series.length, 1)
  const ys = linearScale([0, maxMae * 1.08], [H - M.bottom, M.top])
  const yTicks = niceTicks(0, maxMae * 1.08, 5)

  return (
    <div className="chart" role="img" aria-label="Per-segment test MAE by probe and view">
      <svg viewBox={`0 0 ${W} ${H}`}>
        {yTicks.map((t) => (
          <g key={t}>
            <line className="grid-line" x1={M.left} x2={W - M.right} y1={ys.map(t)} y2={ys.map(t)} />
            <text className="tick-label" x={M.left - 8} y={ys.map(t) + 3} textAnchor="end">
              {fmtTargetTick(t, domain)}
            </text>
          </g>
        ))}
        <line className="axis-line" x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom} />
        <text className="axis-title" x={M.left - 48} y={M.top + 4} textAnchor="start">
          test MAE
        </text>

        {segments.map((s, si) => {
          const x0 = M.left + si * segW + (segW - inner) / 2
          return (
            <g key={s.name}>
              {series.map((ser, bi) => {
                const vr = viewOf(s, ser.view)
                const val = vr ? ser.get(vr) : undefined
                if (val == null || !isFinite(val)) return null
                const x = x0 + bi * barW
                const y = ys.map(val)
                return (
                  <rect key={ser.key} x={x + 1} y={y} width={Math.max(barW - 2, 1)} height={H - M.bottom - y} fill={ser.color}>
                    <title>
                      {s.name} · {ser.label}: {Math.round(val).toLocaleString()}
                    </title>
                  </rect>
                )
              })}
              <text
                className="tick-label"
                transform={`translate(${x0 + inner / 2} ${H - M.bottom + 6}) rotate(-90)`}
                textAnchor="end"
                dominantBaseline="middle"
              >
                {segLabel(s.name)}
              </text>
            </g>
          )
        })}
      </svg>
      <div className="probe-legend">
        {series.map((ser) => (
          <span key={ser.key} className="probe-legend-item">
            <span className="probe-legend-swatch" style={{ background: ser.color }} />
            {ser.label}
          </span>
        ))}
      </div>
    </div>
  )
}
