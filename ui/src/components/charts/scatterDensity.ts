// Geometry and density math for the predicted-vs-actual scatter, separate
// from the component so views can share one density scale across charts
// (and so the chart file stays fast-refreshable).

import type { Prediction } from '../../api/types'
import { pointDensities } from '../../lib/heat'
import { logScale } from '../../lib/scale'

export const SCATTER_W = 420
export const SCATTER_H = 420
export const SCATTER_M = { top: 14, right: 16, bottom: 40, left: 64 }

/** Neighborhood radius (chart px) for local density. */
export const DENSITY_RADIUS = 9
/** Ramp floor for points: a lone outlier renders at t=0.22 (teal-green),
 *  never at the near-paper light end — outliers must stay visible. */
export const DENSITY_FLOOR = 0.22

/** Density per point and its max for a prediction set on a given domain.
 *  Used by CompareView to share one density scale across two charts. */
export function scatterDensities(
  predictions: Prediction[],
  domain: [number, number],
): { densities: number[]; max: number } {
  const xs = logScale(domain, [SCATTER_M.left, SCATTER_W - SCATTER_M.right])
  const ys = logScale(domain, [SCATTER_H - SCATTER_M.bottom, SCATTER_M.top])
  const pts = predictions.map((p) => ({
    x: xs.map(Math.max(p.actual, 1)),
    y: ys.map(Math.max(p.predicted, 1)),
  }))
  const densities = pointDensities(pts, DENSITY_RADIUS)
  let max = 1
  for (const d of densities) if (d > max) max = d
  return { densities, max }
}
