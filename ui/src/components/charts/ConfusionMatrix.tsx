import { useMemo } from 'react'
import type { Prediction } from '../../api/types'
import { heatColor } from '../../lib/heat'
import './charts.css'

/** Confusion matrix for a classification run. Rows are the true class, columns
 *  the predicted class; each cell is the count, shaded by how much of the true
 *  row it holds (the diagonal should dominate). Works for binary and
 *  multiclass: for a binary run `predicted` is P(class==1), thresholded at 0.5;
 *  for multiclass it is already the argmax class id. `actual` is the true class
 *  id in both cases. Class labels come from the domain (`target.classes`),
 *  defaulting to the observed ids. */
export default function ConfusionMatrix({
  predictions,
  classes,
}: {
  predictions: Prediction[]
  /** Ordered class labels (index = class id). */
  classes: string[]
}) {
  const { matrix, rowTotals, k } = useMemo(() => {
    const k = classes.length
    const matrix = Array.from({ length: k }, () => new Array<number>(k).fill(0))
    const rowTotals = new Array<number>(k).fill(0)
    const clamp = (c: number) => Math.min(k - 1, Math.max(0, c))
    for (const p of predictions) {
      // Binary: predicted is P(class==1); threshold. Multiclass: it is the id.
      const pred = k === 2 ? (p.predicted >= 0.5 ? 1 : 0) : clamp(Math.round(p.predicted))
      const actual = clamp(Math.round(p.actual))
      matrix[actual][pred] += 1
      rowTotals[actual] += 1
    }
    return { matrix, rowTotals, k }
  }, [predictions, classes])

  if (predictions.length === 0 || k < 2) {
    return <p className="muted">No predictions to plot.</p>
  }

  return (
    <div className="confusion-wrap" style={{ overflowX: 'auto' }}>
      <table className="confusion-matrix" aria-label="Confusion matrix">
        <thead>
          <tr>
            <th scope="col">
              <span className="sr-only">true \ predicted</span>
              <span aria-hidden="true" className="muted">
                t\p
              </span>
            </th>
            {classes.map((c) => (
              <th key={c} scope="col" className="num-col">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {classes.map((rowLabel, i) => (
            <tr key={rowLabel}>
              <th scope="row">{rowLabel}</th>
              {classes.map((colLabel, j) => {
                const n = matrix[i][j]
                const frac = rowTotals[i] > 0 ? n / rowTotals[i] : 0
                const bg = n > 0 ? heatColor(frac) : 'transparent'
                return (
                  <td
                    key={colLabel}
                    className={`num-col num confusion-cell${i === j ? ' is-diagonal' : ''}`}
                    style={{ background: bg }}
                    title={`true ${rowLabel} → predicted ${colLabel}: ${n} (${(frac * 100).toFixed(1)}%)`}
                  >
                    {n}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
