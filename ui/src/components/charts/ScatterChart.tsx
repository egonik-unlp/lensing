import { useMemo, useState } from 'react'
import type { Prediction } from '../../api/types'
import { heatColor } from '../../lib/heat'
import { fmtMoney } from '../../lib/format'
import { decadeTicks, logScale, moneyTick, scatterDomain } from '../../lib/scale'
import RampLegend from './RampLegend'
import { DENSITY_FLOOR, SCATTER_H, SCATTER_M, SCATTER_W, scatterDensities } from './scatterDensity'
import './charts.css'

const W = SCATTER_W
const H = SCATTER_H
const M = SCATTER_M

/** Legend for the density encoding; shared by single charts and CompareView. */
export function ScatterDensityLegend({ max }: { max: number }) {
  return (
    <RampLegend
      from={DENSITY_FLOOR}
      to={1}
      minLabel="1"
      maxLabel={max.toLocaleString()}
      label="items nearby"
    />
  )
}

interface Props {
  predictions: Prediction[]
  /** Shared axis domain for honest side-by-side comparison. */
  domain?: [number, number]
  /** Shared density max for honest side-by-side color (CompareView). */
  densityMax?: number
  /** Hide the per-chart legend when a shared one is rendered outside. */
  legend?: boolean
  title?: string
  onSelect?: (rowId: number) => void
  selectedRowId?: number | null
}

/** Predicted vs actual, log-log, identity hairline. Each point is colored by
 *  local density (darker = denser), drawn sparse-first so the core layers on
 *  top. Lightness carries the signal, so the encoding survives grayscale. */
export default function ScatterChart({
  predictions,
  domain,
  densityMax,
  legend = true,
  title,
  onSelect,
  selectedRowId,
}: Props) {
  const [hover, setHover] = useState<Prediction | null>(null)

  const dom = useMemo(() => domain ?? scatterDomain(predictions), [domain, predictions])
  const xs = useMemo(() => logScale(dom, [M.left, W - M.right]), [dom])
  const ys = useMemo(() => logScale(dom, [H - M.bottom, M.top]), [dom])
  const ticks = useMemo(() => decadeTicks(dom[0], dom[1]), [dom])

  const { pts, dMax } = useMemo(() => {
    const { densities, max } = scatterDensities(predictions, dom)
    const dMax = Math.max(densityMax ?? 0, max, 1)
    const norm = (d: number) => (dMax > 1 ? Math.log(Math.max(d, 1)) / Math.log(dMax) : 1)
    const pts = predictions
      .map((p, i) => ({
        p,
        x: xs.map(Math.max(p.actual, 1)),
        y: ys.map(Math.max(p.predicted, 1)),
        d: densities[i],
        fill: heatColor(DENSITY_FLOOR + (1 - DENSITY_FLOOR) * norm(densities[i])),
      }))
      // Sparse first: the dense (dark) core draws over stragglers, not under.
      .sort((a, b) => a.d - b.d)
    return { pts, dMax }
  }, [predictions, dom, xs, ys, densityMax])

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const px = ((e.clientX - rect.left) / rect.width) * W
    const py = ((e.clientY - rect.top) / rect.height) * H
    let best: (typeof pts)[number] | null = null
    let bestD = Infinity
    for (const pt of pts) {
      const d = (pt.x - px) ** 2 + (pt.y - py) ** 2
      if (d < bestD) {
        bestD = d
        best = pt
      }
    }
    setHover(best && bestD < 12 ** 2 ? best.p : null)
  }

  if (predictions.length === 0) {
    return (
      <div className="chart-empty">
        No predictions
        <span className="chart-empty-why">they appear once the run evaluates its test set</span>
      </div>
    )
  }

  const sel = selectedRowId != null ? pts.find((pt) => pt.p.row_id === selectedRowId) : null

  return (
    <div
      className="chart"
      role="img"
      aria-label={`${title ?? 'Predicted versus actual price'}, ${predictions.length} test items on log-log axes. Points near the diagonal are accurate; darker points sit in denser regions.`}
    >
      <div className="chart-plot">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        onClick={() => hover && onSelect?.(hover.row_id)}
        style={onSelect && hover ? { cursor: 'pointer' } : undefined}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line className="grid-line" x1={xs.map(t)} x2={xs.map(t)} y1={M.top} y2={H - M.bottom} />
            <line className="grid-line" x1={M.left} x2={W - M.right} y1={ys.map(t)} y2={ys.map(t)} />
            <text className="tick-label" x={xs.map(t)} y={H - M.bottom + 14} textAnchor="middle">
              {moneyTick(t)}
            </text>
            <text className="tick-label" x={M.left - 6} y={ys.map(t) + 3} textAnchor="end">
              {moneyTick(t)}
            </text>
          </g>
        ))}
        <line
          className="identity-line"
          x1={xs.map(dom[0])}
          y1={ys.map(dom[0])}
          x2={xs.map(dom[1])}
          y2={ys.map(dom[1])}
        />
        <text className="axis-title" x={W - M.right} y={H - 8} textAnchor="end">
          actual
        </text>
        <text className="axis-title" x={M.left} y={M.top - 4} textAnchor="start">
          predicted
        </text>
        <g opacity={0.85}>
          {pts.map((pt) => (
            <circle key={pt.p.row_id} cx={pt.x} cy={pt.y} r={2.2} fill={pt.fill} />
          ))}
        </g>
        {sel && (
          <circle cx={sel.x} cy={sel.y} r={5} fill="none" stroke="var(--ink)" strokeWidth={1.5} />
        )}
        {hover && (
          <circle cx={xs.map(Math.max(hover.actual, 1))} cy={ys.map(Math.max(hover.predicted, 1))} r={4} fill="var(--ink)" stroke="var(--bg)" strokeWidth={1} />
        )}
      </svg>
      {hover && (
        <div
          className="chart-tooltip"
          style={{
            left: `${(xs.map(Math.max(hover.actual, 1)) / W) * 100}%`,
            top: `${(ys.map(Math.max(hover.predicted, 1)) / H) * 100}%`,
          }}
        >
          {`item ${hover.row_id}\nactual    ${fmtMoney(hover.actual)}\npredicted ${fmtMoney(hover.predicted)}`}
        </div>
      )}
      </div>
      {legend && <ScatterDensityLegend max={dMax} />}
    </div>
  )
}
