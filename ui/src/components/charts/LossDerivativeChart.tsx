import { useId, useMemo, useState } from 'react'
import type { EpochPoint } from '../../hooks/useRunEvents'
import { fmtLoss } from '../../lib/format'
import { decadeTicks, linearScale, logScale, niceTicks, type Scale } from '../../lib/scale'
import { diff2Series, diffSeries, quantiles, smoothSeries, type DerivPoint } from '../../lib/stats'
import './charts.css'

const W = 640
const H = 130
const M = { top: 14, right: 64, bottom: 24, left: 56 }

type ScaleMode = 'linear' | 'log'

/** fmtLoss for signed values: derivatives go negative. */
const fmtD = (v: number) => (v < 0 ? `−${fmtLoss(-v)}` : fmtLoss(v))

/** "Is it still learning?": first and second discrete derivatives of the
 *  loss curves, as two stacked small multiples. Differentiation happens in
 *  log10(loss) space — the slope and curvature of the loss chart as drawn
 *  (its y is log) — so early-epoch cliffs don't flatten the rest of the run
 *  into the axis. Same series vocabulary as LossChart (val = series A,
 *  train = ramp teal). Two y scales: linear (signed, emphasized zero baseline —
 *  negative Δ means the loss is still falling) and log (magnitudes |Δ|, to
 *  read how fast the per-epoch change itself shrinks across decades). */
export default function LossDerivativeChart({ epochs }: { epochs: EpochPoint[] }) {
  const [scaleMode, setScaleMode] = useState<ScaleMode>('linear')

  const series = useMemo(() => {
    const es = epochs.map((e) => e.epoch)
    const logOf = (v: number) => Math.log10(Math.max(v, Number.MIN_VALUE))
    const val = epochs.map((e) => logOf(e.val))
    const train = epochs.map((e) => logOf(e.train))
    // Per-epoch differences of SGD losses oscillate; the trend is the
    // signal. Smoothing window scales with run length (≈2.5% of the run),
    // and the raw series stays visible as a faint band behind it.
    const k = Math.min(51, Math.max(1, 2 * Math.round(epochs.length / 80) + 1))
    const prep = (points: DerivPoint[]) => smoothSeries(points, k)
    // Second differences amplify noise; a double pass keeps the curvature
    // trend legible while the raw band shows the envelope.
    const prep2 = (points: DerivPoint[]) => smoothSeries(smoothSeries(points, k), k)
    const d1val = diffSeries(es, val)
    const d2val = diff2Series(es, val)
    return {
      d1: {
        val: prep(d1val),
        valRaw: k > 1 ? d1val : [],
        train: prep(diffSeries(es, train)),
      },
      d2: {
        val: prep2(d2val),
        valRaw: k > 1 ? d2val : [],
        train: prep2(diff2Series(es, train)),
      },
    }
  }, [epochs])

  if (epochs.length < 3) {
    return <div className="chart-empty">Needs at least 3 epochs to differentiate</div>
  }

  return (
    <div className="deriv-charts">
      <div className="scale-toggle" role="group" aria-label="Derivative chart y scale">
        <span className="muted">scale</span>
        {(['linear', 'log'] as const).map((m) => (
          <button
            key={m}
            className={`scale-btn${scaleMode === m ? ' is-active' : ''}`}
            aria-pressed={scaleMode === m}
            onClick={() => setScaleMode(m)}
          >
            {m}
          </button>
        ))}
      </div>
      <SubChart
        title="Improvement per epoch"
        hint={
          scaleMode === 'linear'
            ? 'below zero = still improving · Δ log MSE'
            : 'size of each epoch’s change · |Δ log MSE|'
        }
        scaleMode={scaleMode}
        val={series.d1.val}
        valRaw={series.d1.valRaw}
        train={series.d1.train}
      />
      <SubChart
        title="Change in improvement"
        hint={
          scaleMode === 'linear'
            ? 'near 0 = improvement has flattened · Δ² log MSE'
            : 'size of the curvature · |Δ² log MSE|'
        }
        scaleMode={scaleMode}
        val={series.d2.val}
        valRaw={series.d2.valRaw}
        train={series.d2.train}
      />
    </div>
  )
}

/** Robust signed domain + linear scale (zero always included). */
function linearAxis(all: number[]): { ys: Scale; yTicks: number[]; clipped: boolean } {
  const [p2, p98] = quantiles(all, [0.02, 0.98])
  let lo = Math.min(0, p2)
  let hi = Math.max(0, p98)
  if (lo === hi) {
    lo -= 1
    hi += 1
  }
  const clipped = Math.min(...all) < lo || Math.max(...all) > hi
  const pad = (hi - lo) * 0.08
  return {
    ys: linearScale([lo - pad, hi + pad], [H - M.bottom, M.top]),
    yTicks: niceTicks(lo, hi, 3),
    clipped,
  }
}

/** Robust magnitude domain + log scale (positive magnitudes only). */
function logAxis(all: number[]): { ys: Scale; yTicks: number[]; clipped: boolean } {
  const mags = all.filter((v) => v > 0)
  if (mags.length === 0) {
    return { ys: logScale([1e-6, 1], [H - M.bottom, M.top]), yTicks: [], clipped: false }
  }
  const [p2, p98] = quantiles(mags, [0.02, 0.98])
  let lo = Math.max(p2, Number.MIN_VALUE)
  let hi = Math.max(p98, lo * 10)
  const clipped = Math.min(...mags) < lo || Math.max(...mags) > hi
  // Multiplicative padding, the log-space analogue of the linear 8%.
  lo /= 1.5
  hi *= 1.5
  let yTicks = decadeTicks(lo, hi)
  if (yTicks.length < 2) yTicks = [lo, hi]
  return { ys: logScale([lo, hi], [H - M.bottom, M.top]), yTicks, clipped }
}

function SubChart({
  title,
  hint,
  scaleMode,
  val,
  valRaw,
  train,
}: {
  title: string
  hint: string
  scaleMode: ScaleMode
  val: DerivPoint[]
  valRaw: DerivPoint[]
  train: DerivPoint[]
}) {
  const [hover, setHover] = useState<number | null>(null)
  const clipId = useId()
  const isLog = scaleMode === 'log'

  // In log mode the chart reads magnitudes; zero deltas have no log position
  // and are dropped from the drawn series (the tooltip still shows the
  // signed value via the untransformed arrays).
  const tVal = useMemo(
    () => (isLog ? val.map((p) => ({ ...p, value: Math.abs(p.value) })).filter((p) => p.value > 0) : val),
    [isLog, val],
  )
  const tValRaw = useMemo(
    () =>
      isLog ? valRaw.map((p) => ({ ...p, value: Math.abs(p.value) })).filter((p) => p.value > 0) : valRaw,
    [isLog, valRaw],
  )
  const tTrain = useMemo(
    () => (isLog ? train.map((p) => ({ ...p, value: Math.abs(p.value) })).filter((p) => p.value > 0) : train),
    [isLog, train],
  )

  const { xs, ys, valPath, valRawPath, trainPath, xTicks, yTicks, clipped } = useMemo(() => {
    const all = [...tVal, ...tTrain].map((p) => p.value).filter(isFinite)
    // Robust domain: the reading lives in the bulk of the run, not in the
    // first-epochs cliff. Off-scale points clip visibly (truth, framed).
    const axis = isLog ? logAxis(all) : linearAxis(all)
    const maxEpoch = Math.max(val[val.length - 1]?.epoch ?? 2, 2)
    const minEpoch = val[0]?.epoch ?? 1
    const xs = linearScale([minEpoch, maxEpoch], [M.left, W - M.right])
    const path = (pts: DerivPoint[]) =>
      pts
        .map((p, i) => `${i === 0 ? 'M' : 'L'}${xs.map(p.epoch).toFixed(1)},${axis.ys.map(p.value).toFixed(1)}`)
        .join('')
    return {
      xs,
      ys: axis.ys,
      valPath: path(tVal),
      valRawPath: path(tValRaw),
      trainPath: path(tTrain),
      xTicks: niceTicks(minEpoch, maxEpoch, 6).filter((t) => t >= 1 && Number.isInteger(t)),
      yTicks: axis.yTicks,
      clipped: axis.clipped,
    }
  }, [tVal, tValRaw, tTrain, val, isLog])

  const lastVal = tVal[tVal.length - 1]
  const lastTrain = tTrain[tTrain.length - 1]

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const px = ((e.clientX - rect.left) / rect.width) * W
    let best: number | null = null
    let bestD = Infinity
    for (let i = 0; i < tVal.length; i++) {
      const d = Math.abs(xs.map(tVal[i].epoch) - px)
      if (d < bestD) {
        bestD = d
        best = i
      }
    }
    setHover(bestD < 24 ? best : null)
  }

  const hv = hover != null ? tVal[hover] : null
  // Tooltip shows the signed reading even when the axis shows magnitudes.
  const hvSigned = hv != null ? val.find((p) => p.epoch === hv.epoch) ?? hv : null
  const ht = hover != null ? tTrain.find((p) => p.epoch === hv?.epoch) ?? null : null
  const htSigned = ht != null ? train.find((p) => p.epoch === ht.epoch) ?? ht : null

  return (
    <div
      className="chart"
      role="img"
      aria-label={`${title}; ${hint}. Latest: validation ${fmtD(lastVal?.value ?? 0)}, train ${fmtD(lastTrain?.value ?? 0)}.`}
    >
      <svg viewBox={`0 0 ${W} ${H}`} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line className="grid-line" x1={M.left} x2={W - M.right} y1={ys.map(t)} y2={ys.map(t)} />
            <text className="tick-label" x={M.left - 6} y={ys.map(t) + 3} textAnchor="end">
              {fmtD(t)}
            </text>
          </g>
        ))}
        {/* Emphasized zero baseline: the reading anchor for signed derivatives
            (zero has no position on a log axis). */}
        {!isLog && (
          <line x1={M.left} x2={W - M.right} y1={ys.map(0)} y2={ys.map(0)} stroke="var(--hairline-strong)" strokeWidth={1.25} />
        )}
        {xTicks.map((t) => (
          <text key={`x${t}`} className="tick-label" x={xs.map(t)} y={H - M.bottom + 13} textAnchor="middle">
            {t}
          </text>
        ))}
        <text className="axis-title" x={M.left} y={M.top - 2} textAnchor="start">
          {title} <tspan fill="var(--ink-faint)">· {hint}</tspan>
          {clipped && <tspan fill="var(--ink-faint)"> · y clipped to p2–p98</tspan>}
        </text>
        <clipPath id={clipId}>
          <rect x={M.left} y={M.top} width={W - M.left - M.right} height={H - M.top - M.bottom} />
        </clipPath>
        <g clipPath={`url(#${clipId})`}>
          {valRawPath && (
            <path d={valRawPath} fill="none" stroke="var(--series-a)" strokeWidth={0.75} opacity={0.12} />
          )}
          {/* Same series vocabulary as LossChart: train teal, val graphite. */}
          <path d={trainPath} fill="none" stroke="var(--heat-3)" strokeWidth={1.25} opacity={0.9} />
          <path d={valPath} fill="none" stroke="var(--series-a)" strokeWidth={1.75} />
        </g>
        {lastVal && (
          <text className="series-label" x={xs.map(lastVal.epoch) + 6} y={ys.map(lastVal.value) + 3} fill="var(--series-a)">
            val
          </text>
        )}
        {lastTrain && (
          <text
            className="series-label"
            x={xs.map(lastTrain.epoch) + 6}
            y={
              lastVal && Math.abs(ys.map(lastTrain.value) - ys.map(lastVal.value)) < 12
                ? ys.map(lastVal.value) + (ys.map(lastTrain.value) >= ys.map(lastVal.value) ? 14 : -10)
                : ys.map(lastTrain.value) + 3
            }
            fill="var(--heat-4)"
          >
            train
          </text>
        )}
        {hv && (
          <g>
            <line className="grid-line" x1={xs.map(hv.epoch)} x2={xs.map(hv.epoch)} y1={M.top} y2={H - M.bottom} stroke="var(--hairline-strong)" />
            <circle cx={xs.map(hv.epoch)} cy={ys.map(hv.value)} r={3} fill="var(--series-a)" />
            {ht && <circle cx={xs.map(ht.epoch)} cy={ys.map(ht.value)} r={3} fill="var(--heat-3)" />}
          </g>
        )}
      </svg>
      {hv && hvSigned && (
        <div
          className="chart-tooltip"
          style={{ left: `${(xs.map(hv.epoch) / W) * 100}%`, top: `${(ys.map(hv.value) / H) * 100}%` }}
        >
          {`epoch ${hv.epoch}\nval   ${fmtD(hvSigned.value)}${htSigned ? `\ntrain ${fmtD(htSigned.value)}` : ''}`}
        </div>
      )}
    </div>
  )
}
