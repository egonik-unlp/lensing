import { useMemo, useState } from 'react'
import { countIntoBins, histogram } from '../../lib/stats'
import { decadeTicks, linearScale, niceTicks } from '../../lib/scale'
import './charts.css'

const W = 640
const H = 200
const M = { top: 18, right: 16, bottom: 32, left: 44 }

export interface HistogramSeries {
  label: string
  values: number[]
  /** 'fill' (series A, graphite) or 'outline' (series B, comparison overlay). */
  style: 'fill' | 'outline'
}

export interface HistogramMarker {
  value: number
  label: string
}

interface Props {
  /** First series defines the bins; overlays count into the same edges. */
  series: HistogramSeries[]
  bins?: number
  /** Log-10 x binning (money-style values). Values ≤ 0 are dropped from the chart. */
  log?: boolean
  xLabel: string
  tickFormat: (v: number) => string
  /** Vertical annotations (quantiles). */
  markers?: HistogramMarker[]
  /** Drill-down: a bin of the first series was clicked. */
  onBinClick?: (lo: number, hi: number) => void
  /** Highlighted bin range (matches onBinClick args). */
  selected?: [number, number] | null
}

/** Generic value histogram with optional overlay series sharing the base
 *  series' bins. Fill = graphite; outline = series B, direct-labeled. */
export default function Histogram({
  series,
  bins = 40,
  log,
  xLabel,
  tickFormat,
  markers,
  onBinClick,
  selected,
}: Props) {
  const [hover, setHover] = useState<number | null>(null)

  const { edges, counts } = useMemo(() => {
    const base = histogram(series[0]?.values ?? [], { bins, log })
    const counts = series.map((s, i) =>
      i === 0 ? base.counts : countIntoBins(s.values, base.edges, log),
    )
    return { edges: base.edges, counts }
  }, [series, bins, log])

  if (edges.length < 2) return <div className="chart-empty">No values to bin</div>

  const n = edges.length - 1
  const maxCount = Math.max(1, ...counts.flat())
  // X positions interpolate bin index; tick values map through edge space.
  const xIdx = linearScale([0, n], [M.left, W - M.right])
  const e0 = log ? Math.log10(edges[0]) : edges[0]
  const e1 = log ? Math.log10(edges[n]) : edges[n]
  const xVal = (v: number) => {
    const x = log ? Math.log10(Math.max(v, Number.MIN_VALUE)) : v
    return M.left + ((x - e0) / (e1 - e0 || 1)) * (W - M.left - M.right)
  }
  const yMap = (c: number) => H - M.bottom - (c / maxCount) * (H - M.top - M.bottom)
  const bw = xIdx.map(1) - xIdx.map(0)
  const ticks = (log ? decadeTicks(edges[0], edges[n]) : niceTicks(edges[0], edges[n], 6)).filter(
    (t) => t >= edges[0] && t <= edges[n],
  )

  const isSelected = (i: number) =>
    selected != null && edges[i] >= selected[0] && edges[i + 1] <= selected[1]

  return (
    <div
      className="chart"
      role="img"
      aria-label={`Distribution of ${xLabel}${series.length > 1 ? `, ${series.map((s) => s.label).join(' vs ')}` : ''}.${onBinClick ? ' Bars are clickable to inspect the underlying items.' : ''}`}
    >
      <svg viewBox={`0 0 ${W} ${H}`} onMouseLeave={() => setHover(null)}>
        {[0.5, 1].map((f) => (
          <g key={f}>
            <line className="grid-line" x1={M.left} x2={W - M.right} y1={yMap(f * maxCount)} y2={yMap(f * maxCount)} />
            <text className="tick-label" x={M.left - 5} y={yMap(f * maxCount) + 3} textAnchor="end">
              {Math.round(f * maxCount).toLocaleString()}
            </text>
          </g>
        ))}
        {counts.map((cs, si) =>
          cs.map((c, i) => {
            if (c === 0) return null
            const x = xIdx.map(i)
            const style = series[si].style
            return style === 'fill' ? (
              <rect
                key={`${si}-${i}`}
                x={x + 0.5}
                width={Math.max(0.5, bw - 1)}
                y={yMap(c)}
                height={H - M.bottom - yMap(c)}
                fill="var(--series-a)"
                opacity={isSelected(i) ? 1 : hover === i ? 0.9 : 0.75}
                cursor={onBinClick ? 'pointer' : undefined}
                onClick={onBinClick ? () => onBinClick(edges[i], edges[i + 1]) : undefined}
                onMouseEnter={() => setHover(i)}
              />
            ) : (
              <rect
                key={`${si}-${i}`}
                x={x + 0.5}
                width={Math.max(0.5, bw - 1)}
                y={yMap(c)}
                height={H - M.bottom - yMap(c)}
                fill="none"
                stroke="var(--series-b)"
                strokeWidth={1.4}
                pointerEvents="none"
              />
            )
          }),
        )}
        {markers?.map((mk) => (
          <g key={mk.label}>
            <line x1={xVal(mk.value)} x2={xVal(mk.value)} y1={M.top} y2={H - M.bottom} stroke="var(--hairline-strong)" strokeDasharray="4 3" />
            <text className="tick-label" x={xVal(mk.value)} y={M.top - 4} textAnchor="middle">
              {mk.label}
            </text>
          </g>
        ))}
        <line className="axis-line" x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom} />
        {ticks.map((t) => (
          <text key={t} className="tick-label" x={xVal(t)} y={H - M.bottom + 14} textAnchor="middle">
            {tickFormat(t)}
          </text>
        ))}
        <text className="axis-title" x={W - M.right} y={H - 4} textAnchor="end">
          {xLabel}
        </text>
        {series.length > 1 && (
          <g>
            <rect x={M.left + 4} y={M.top - 12} width={9} height={9} fill="var(--series-a)" opacity={0.75} />
            <text className="series-label" x={M.left + 17} y={M.top - 4} fill="var(--ink-secondary)">
              {series[0].label}
            </text>
            <rect x={M.left + 17 + series[0].label.length * 6 + 12} y={M.top - 12} width={9} height={9} fill="none" stroke="var(--series-b)" strokeWidth={1.4} />
            <text className="series-label" x={M.left + 17 + series[0].label.length * 6 + 30} y={M.top - 4} fill="var(--ink-secondary)">
              {series[1].label}
            </text>
          </g>
        )}
      </svg>
      {hover != null && counts[0][hover] > 0 && (
        <div
          className="chart-tooltip"
          style={{ left: `${((xIdx.map(hover) + bw / 2) / W) * 100}%`, top: `${(yMap(counts[0][hover]) / H) * 100}%` }}
        >
          {`${tickFormat(edges[hover])} – ${tickFormat(edges[hover + 1])}\n${series
            .map((s, si) => `${series.length > 1 ? `${s.label}  ` : ''}${counts[si][hover].toLocaleString()}`)
            .join('\n')}`}
        </div>
      )}
    </div>
  )
}
