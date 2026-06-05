import { useMemo } from 'react'
import type { Prediction } from '../../api/types'
import { divergingColor } from '../../lib/heat'
import './charts.css'

// Direction words, tinted to match their side of the diverging ramp.
// Darker than the bars so the 10px labels hold ≥4.5:1 on paper.
const UNDER_TEXT = 'oklch(0.42 0.12 250)'
const OVER_TEXT = 'oklch(0.42 0.14 27)'

const W = 640
const H = 200
const M = { top: 18, right: 16, bottom: 32, left: 44 }

// Signed percent error bins: 5%-wide from -100% to +100%, plus an
// overflow bin for everything beyond +100% (underprediction floors at -100%).
const BIN_W = 0.05
const LO = -1.0
const HI = 1.0
const N_BINS = Math.round((HI - LO) / BIN_W) + 1 // last = overflow

export interface HistSeries {
  label: string
  predictions: Prediction[]
  /** 'fill' (series A, graphite) or 'outline' (series B, blue dashed). */
  style: 'fill' | 'outline'
}

function binCounts(preds: Prediction[]): number[] {
  const counts = new Array<number>(N_BINS).fill(0)
  for (const p of preds) {
    const err = (p.predicted - p.actual) / p.actual
    if (err > HI) counts[N_BINS - 1]++
    else {
      const i = Math.min(N_BINS - 2, Math.max(0, Math.floor((err - LO) / BIN_W)))
      counts[i]++
    }
  }
  return counts
}

/** Distribution of signed percent error. Overflow beyond +100% is shown, labeled. */
export default function ErrorHistogram({ series }: { series: HistSeries[] }) {
  const data = useMemo(() => series.map((s) => ({ ...s, counts: binCounts(s.predictions) })), [series])
  const maxCount = Math.max(1, ...data.flatMap((d) => d.counts))
  const total = series[0]?.predictions.length ?? 0

  if (total === 0) return <div className="chart-empty">No predictions</div>

  const bw = (W - M.left - M.right) / N_BINS
  const yMap = (c: number) => H - M.bottom - (c / maxCount) * (H - M.top - M.bottom)
  // +100% is omitted: its label would collide with the overflow bin's.
  const xTicks = [-1, -0.5, 0, 0.5]

  return (
    <div
      className="chart"
      role="img"
      aria-label={`Distribution of signed percent error${series.length > 1 ? ` for ${series.map((s) => s.label).join(' and ')}` : ''}. Bars left of zero are underpredictions, right are overpredictions; the rightmost bar collects errors above +100%.`}
    >
      <svg viewBox={`0 0 ${W} ${H}`}>
        {[0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <line className="grid-line" x1={M.left} x2={W - M.right} y1={yMap(f * maxCount)} y2={yMap(f * maxCount)} />
            <text className="tick-label" x={M.left - 5} y={yMap(f * maxCount) + 3} textAnchor="end">
              {Math.round(f * maxCount)}
            </text>
          </g>
        ))}
        {data.map((d) =>
          d.counts.map((c, i) => {
            if (c === 0) return null
            const x = M.left + i * bw
            // Bin center as signed error; the overflow bin saturates at +1.
            const center = i === N_BINS - 1 ? 1 : LO + (i + 0.5) * BIN_W
            return d.style === 'fill' ? (
              <rect key={`${d.label}${i}`} x={x + 0.5} width={bw - 1} y={yMap(c)} height={H - M.bottom - yMap(c)} fill={divergingColor(center)} />
            ) : (
              <rect key={`${d.label}${i}`} x={x + 0.5} width={bw - 1} y={yMap(c)} height={H - M.bottom - yMap(c)} fill="none" stroke="var(--series-b)" strokeWidth={1.4} />
            )
          }),
        )}
        {/* zero line */}
        <line x1={M.left + ((0 - LO) / BIN_W) * bw} x2={M.left + ((0 - LO) / BIN_W) * bw} y1={M.top} y2={H - M.bottom} stroke="var(--hairline-strong)" strokeWidth={1} />
        <line className="axis-line" x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom} />
        {xTicks.map((t) => (
          <text key={t} className="tick-label" x={M.left + ((t - LO) / BIN_W) * bw} y={H - M.bottom + 14} textAnchor="middle">
            {t > 0 ? `+${t * 100}%` : `${t * 100}%`}
          </text>
        ))}
        <text className="tick-label" x={M.left + N_BINS * bw - bw / 2} y={H - M.bottom + 14} textAnchor="middle">
          &gt;+100%
        </text>
        <text className="axis-title" x={M.left} y={H - 4} textAnchor="start" style={{ fill: UNDER_TEXT }}>
          ← under-predicts
        </text>
        <text className="axis-title" x={(M.left + W - M.right) / 2} y={H - 4} textAnchor="middle">
          signed % error
        </text>
        <text className="axis-title" x={W - M.right} y={H - 4} textAnchor="end" style={{ fill: OVER_TEXT }}>
          over-predicts →
        </text>
        {/* Direct series labels */}
        {data.length > 1 && (
          <g>
            <rect x={M.left + 4} y={M.top - 12} width={4.5} height={9} fill={divergingColor(-0.7)} />
            <rect x={M.left + 8.5} y={M.top - 12} width={4.5} height={9} fill={divergingColor(0.7)} />
            <text className="series-label" x={M.left + 17} y={M.top - 4} fill="var(--ink-secondary)">
              {data[0].label}
            </text>
            <rect x={M.left + 17 + data[0].label.length * 6 + 12} y={M.top - 12} width={9} height={9} fill="none" stroke="var(--series-b)" strokeWidth={1.4} />
            <text className="series-label" x={M.left + 17 + data[0].label.length * 6 + 30} y={M.top - 4} fill="var(--ink-secondary)">
              {data[1].label}
            </text>
          </g>
        )}
      </svg>
    </div>
  )
}
