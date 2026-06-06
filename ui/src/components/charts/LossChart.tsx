import { useMemo, useState } from 'react'
import type { EpochPoint } from '../../hooks/useRunEvents'
import { fmtLoss } from '../../lib/format'
import { linearScale, logScale, niceTicks } from '../../lib/scale'
import './charts.css'

const W = 640
const H = 220
const M = { top: 12, right: 64, bottom: 30, left: 56 }

interface Props {
  epochs: EpochPoint[]
  /** When known (live run), fixes the x domain so the path extends rightward. */
  totalEpochs?: number
  /** Run still training: the val trace glows signal orange and its head
   *  pulses. Finished runs read in graphite — orange means live. */
  live?: boolean
}

/** Train/val loss per epoch. Log y; val leads, train is dimmed ink. */
export default function LossChart({ epochs, totalEpochs, live = false }: Props) {
  const [hover, setHover] = useState<EpochPoint | null>(null)

  const { xs, ys, trainPath, valPath, xTicks, yTicks } = useMemo(() => {
    const maxEpoch = Math.max(totalEpochs ?? 0, epochs.length ? epochs[epochs.length - 1].epoch : 1)
    const values = epochs.flatMap((e) => [e.train, e.val]).filter((v) => v > 0 && isFinite(v))
    const lo = values.length ? Math.min(...values) : 0.1
    const hi = values.length ? Math.max(...values) : 10
    const xs = linearScale([1, Math.max(maxEpoch, 2)], [M.left, W - M.right])
    const ys = logScale([lo, hi * 1.05], [H - M.bottom, M.top])
    const path = (key: 'train' | 'val') =>
      epochs
        .map((e, i) => `${i === 0 ? 'M' : 'L'}${xs.map(e.epoch).toFixed(1)},${ys.map(e[key]).toFixed(1)}`)
        .join('')
    // Log-decade ticks when the spread is wide, otherwise linear ticks.
    const yTicks =
      hi / Math.max(lo, 1e-12) > 30
        ? [...new Set([lo, hi].concat(decades(lo, hi)))].sort((a, b) => a - b)
        : niceTicks(lo, hi, 4)
    return {
      xs,
      ys,
      trainPath: path('train'),
      valPath: path('val'),
      xTicks: niceTicks(1, maxEpoch, 6).filter((t) => t >= 1 && Number.isInteger(t)),
      yTicks,
    }
  }, [epochs, totalEpochs])

  if (epochs.length === 0) {
    return (
      <div className="chart-empty">
        No epoch events for this run
        <span className="chart-empty-why">
          some predictors fit in one shot and never report epochs
        </span>
      </div>
    )
  }

  const last = epochs[epochs.length - 1]

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const px = ((e.clientX - rect.left) / rect.width) * W
    let best: EpochPoint | null = null
    let bestD = Infinity
    for (const p of epochs) {
      const d = Math.abs(xs.map(p.epoch) - px)
      if (d < bestD) {
        bestD = d
        best = p
      }
    }
    setHover(bestD < 24 ? best : null)
  }

  return (
    <div className="chart" role="img" aria-label={`Loss per epoch. Latest: epoch ${last.epoch}, train ${fmtLoss(last.train)}, validation ${fmtLoss(last.val)}.`}>
      <svg viewBox={`0 0 ${W} ${H}`} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line className="grid-line" x1={M.left} x2={W - M.right} y1={ys.map(t)} y2={ys.map(t)} />
            <text className="tick-label" x={M.left - 6} y={ys.map(t) + 3} textAnchor="end">
              {fmtLoss(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={`x${t}`} className="tick-label" x={xs.map(t)} y={H - M.bottom + 14} textAnchor="middle">
            {t}
          </text>
        ))}
        <line className="axis-line" x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom} />
        <text className="axis-title" x={W - M.right} y={H - 6} textAnchor="end">
          epoch
        </text>
        <text className="axis-title" x={M.left} y={M.top - 2} textAnchor="start">
          MSE, log-target space
        </text>

        {/* Train rides in the ramp's teal; val keeps the lead in graphite
            (or orange while live — orange still means live, nothing else). */}
        <path d={trainPath} fill="none" stroke="var(--heat-3)" strokeWidth={1.25} opacity={0.9} />
        <path d={valPath} fill="none" stroke={live ? 'var(--accent-strong)' : 'var(--series-a)'} strokeWidth={1.75} />
        {live && (
          <circle
            className="live-head"
            cx={xs.map(last.epoch)}
            cy={ys.map(last.val)}
            r={3}
            fill="var(--accent-strong)"
          />
        )}

        {/* Direct series labels, pushed apart when the lines converge. */}
        {(() => {
          let yVal = ys.map(last.val) + 3
          let yTrain = ys.map(last.train) + 3
          if (Math.abs(yVal - yTrain) < 12) {
            const valBelow = yVal >= yTrain
            const mid = (yVal + yTrain) / 2
            yVal = mid + (valBelow ? 6 : -6)
            yTrain = mid + (valBelow ? -6 : 6)
          }
          return (
            <>
              <text className="series-label" x={xs.map(last.epoch) + 6} y={yVal} fill={live ? 'var(--accent-text)' : 'var(--series-a)'}>
                val
              </text>
              <text className="series-label" x={xs.map(last.epoch) + 6} y={yTrain} fill="var(--heat-4)">
                train
              </text>
            </>
          )
        })()}

        {hover && (
          <g>
            <line className="grid-line" x1={xs.map(hover.epoch)} x2={xs.map(hover.epoch)} y1={M.top} y2={H - M.bottom} stroke="var(--hairline-strong)" />
            <circle cx={xs.map(hover.epoch)} cy={ys.map(hover.val)} r={3} fill={live ? 'var(--accent-strong)' : 'var(--series-a)'} />
            <circle cx={xs.map(hover.epoch)} cy={ys.map(hover.train)} r={3} fill="var(--heat-3)" />
          </g>
        )}
      </svg>
      {hover && (
        <div
          className="chart-tooltip"
          style={{ left: `${(xs.map(hover.epoch) / W) * 100}%`, top: `${(ys.map(hover.val) / H) * 100}%` }}
        >
          {`epoch ${hover.epoch}\nval   ${fmtLoss(hover.val)}\ntrain ${fmtLoss(hover.train)}`}
        </div>
      )}
    </div>
  )
}

function decades(lo: number, hi: number): number[] {
  const out: number[] = []
  for (let e = Math.ceil(Math.log10(lo)); e <= Math.floor(Math.log10(hi)); e++) {
    out.push(Math.pow(10, e))
  }
  return out
}
