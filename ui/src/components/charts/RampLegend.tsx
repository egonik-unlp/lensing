import { heatColor } from '../../lib/heat'
import './charts.css'

interface Props {
  /** Sub-range of the ramp actually used (points floor at 0.22 so sparse
   *  outliers stay visible; full-range heatmaps use 0..1). */
  from?: number
  to?: number
  minLabel: string
  maxLabel: string
  /** What the ramp encodes, e.g. "items nearby" or "MAE". */
  label?: string
}

/** Compact horizontal ramp legend. The gradient mirrors heatColor() so the
 *  legend can never drift from the chart. Color is annotated, not alone:
 *  min/max labels carry the actual values in mono. */
export default function RampLegend({ from = 0, to = 1, minLabel, maxLabel, label }: Props) {
  const stops = Array.from({ length: 9 }, (_, i) => heatColor(from + ((to - from) * i) / 8))
  return (
    <div className="ramp-legend">
      {label && <span className="ramp-legend-label">{label}</span>}
      <span className="ramp-legend-end">{minLabel}</span>
      <span
        className="ramp-legend-bar"
        style={{ background: `linear-gradient(90deg, ${stops.join(', ')})` }}
        aria-hidden="true"
      />
      <span className="ramp-legend-end">{maxLabel}</span>
    </div>
  )
}
