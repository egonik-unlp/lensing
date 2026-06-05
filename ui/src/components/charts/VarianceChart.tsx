import { useMemo } from 'react'
import { fmtPct } from '../../lib/format'
import { linearScale } from '../../lib/scale'
import './charts.css'

const W = 640
const H = 200
const M = { top: 18, right: 52, bottom: 30, left: 44 }

/** PCA explained variance: per-component bars (graphite) on the left axis,
 *  cumulative line (ink) on an implicit 0–100% scale with a direct end label.
 *  `markK` draws a vertical guide at component k (1-based) labelled with the
 *  cumulative variance captured up to it. */
export default function VarianceChart({ evr, markK }: { evr: number[]; markK?: number }) {
  const { xs, yBar, cumPath, cumulative, maxEvr } = useMemo(() => {
    const maxEvr = Math.max(...evr, 1e-9)
    const xs = linearScale([0, evr.length], [M.left, W - M.right])
    const yBar = linearScale([0, maxEvr], [H - M.bottom, M.top])
    const yCum = linearScale([0, 1], [H - M.bottom, M.top])
    const cumulative: number[] = []
    let acc = 0
    for (const v of evr) {
      acc += v
      cumulative.push(acc)
    }
    const cumPath = cumulative
      .map((c, i) => `${i === 0 ? 'M' : 'L'}${(xs.map(i) + xs.map(i + 1)) / 2},${yCum.map(c).toFixed(1)}`)
      .join('')
    return { xs, yBar, cumPath, cumulative, maxEvr }
  }, [evr])

  if (evr.length === 0) return <div className="chart-empty">No PCA components</div>

  const total = cumulative[cumulative.length - 1]
  const bw = Math.max(1, xs.map(1) - xs.map(0) - 1)
  const yCum = linearScale([0, 1], [H - M.bottom, M.top])
  const mark =
    markK && markK >= 1 && markK <= evr.length
      ? { k: markK, x: (xs.map(markK - 1) + xs.map(markK)) / 2, cum: cumulative[markK - 1] }
      : null
  const compTicks = [1, Math.round(evr.length / 2), evr.length].filter(
    (t, i, a) => a.indexOf(t) === i,
  )
  const yTicks = [0, maxEvr / 2, maxEvr]

  return (
    <div
      className="chart"
      role="img"
      aria-label={`PCA explained variance per component. ${evr.length} components capture ${fmtPct(total)} of total variance.`}
    >
      <svg viewBox={`0 0 ${W} ${H}`}>
        {yTicks.map((t) => (
          <g key={t}>
            <line className="grid-line" x1={M.left} x2={W - M.right} y1={yBar.map(t)} y2={yBar.map(t)} />
            <text className="tick-label" x={M.left - 5} y={yBar.map(t) + 3} textAnchor="end">
              {fmtPct(t)}
            </text>
          </g>
        ))}
        {evr.map((v, i) => (
          <rect
            key={i}
            x={xs.map(i) + 0.5}
            width={bw}
            y={yBar.map(v)}
            height={H - M.bottom - yBar.map(v)}
            fill="var(--heat-3)"
            opacity={0.75}
          />
        ))}
        <path d={cumPath} fill="none" stroke="var(--heat-5)" strokeWidth={1.5} />
        {mark && (
          <g>
            <line
              className="mark-line"
              x1={mark.x}
              x2={mark.x}
              y1={M.top}
              y2={H - M.bottom}
              stroke="var(--series-a)"
              strokeDasharray="3 3"
            />
            <circle cx={mark.x} cy={yCum.map(mark.cum)} r={3} fill="var(--series-a)" />
            <text
              className="series-label"
              x={mark.x + 4}
              y={M.top + 10}
              fill="var(--series-a)"
            >
              k={mark.k} · {fmtPct(mark.cum, 1)}
            </text>
          </g>
        )}
        <text
          className="series-label"
          x={W - M.right + 6}
          y={linearScale([0, 1], [H - M.bottom, M.top]).map(total) + 3}
          fill="var(--heat-5)"
        >
          Σ {fmtPct(total, 0)}
        </text>
        <line className="axis-line" x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom} />
        {compTicks.map((t) => (
          <text key={t} className="tick-label" x={(xs.map(t - 1) + xs.map(t)) / 2} y={H - M.bottom + 14} textAnchor="middle">
            {t}
          </text>
        ))}
        <text className="axis-title" x={W - M.right} y={H - 4} textAnchor="end">
          component
        </text>
        <text className="axis-title" x={M.left} y={M.top - 6} textAnchor="start">
          variance explained (bars) · cumulative (line)
        </text>
      </svg>
    </div>
  )
}
