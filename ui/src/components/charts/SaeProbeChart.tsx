import { useMemo } from 'react'
import type { SaeProbe } from '../../api/types'
import { linearScale, niceTicks } from '../../lib/scale'
import './charts.css'

const W = 560
const H = 280
const M = { top: 16, right: 16, bottom: 40, left: 56 }

/** Grouped bars: the raw embedding block vs the SAE code, per segment, scored by
 *  test R² (log). A code probe that diverged (over-fit a tiny slice) is drawn
 *  off-scale with a marker rather than squashing the axis. */
export default function SaeProbeChart({ probes }: { probes: SaeProbe[] }) {
  const { segs, raw, code } = useMemo(() => {
    const order: string[] = []
    for (const p of probes) if (!order.includes(p.segment)) order.push(p.segment)
    const pick = (seg: string, stage: string) =>
      probes.find((p) => p.segment === seg && p.stage === stage)?.test_r2_log ?? null
    return {
      segs: order,
      raw: order.map((s) => pick(s, 'input:pca')),
      code: order.map((s) => pick(s, 'sae:code')),
    }
  }, [probes])

  const diverged = code.map((v) => v != null && v <= -1)
  const finite = [...raw, ...code.map((v, i) => (diverged[i] ? null : v))].filter(
    (v): v is number => v != null,
  )
  const lo = Math.min(0, ...finite) - 0.05
  const hi = Math.max(...finite, 0.1) + 0.12
  const xs = linearScale([0, Math.max(segs.length, 1)], [M.left, W - M.right])
  const ys = linearScale([lo, hi], [H - M.bottom, M.top])
  const yTicks = niceTicks(lo, hi, 5)
  const bw = (xs.map(1) - xs.map(0)) * 0.3

  const bar = (cx: number, v: number | null, fill: string) => {
    if (v == null) return null
    const y0 = ys.map(0)
    const y1 = ys.map(v)
    return <rect x={cx - bw / 2} y={Math.min(y0, y1)} width={bw} height={Math.abs(y1 - y0)} fill={fill} />
  }

  return (
    <div className="chart" role="img" aria-label="SAE code vs raw embedding decodability by segment">
      <svg viewBox={`0 0 ${W} ${H}`}>
        {yTicks.map((t) => (
          <g key={t}>
            <line className="grid-line" x1={M.left} x2={W - M.right} y1={ys.map(t)} y2={ys.map(t)} />
            <text className="tick-label" x={M.left - 6} y={ys.map(t) + 3} textAnchor="end">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        <line className="axis-line" x1={M.left} x2={W - M.right} y1={ys.map(0)} y2={ys.map(0)} />
        <text className="axis-title" x={M.left} y={M.top - 4} textAnchor="start">
          test R² (log)
        </text>
        {segs.map((seg, i) => {
          const center = xs.map(i + 0.5)
          return (
            <g key={seg}>
              {bar(center - bw * 0.6, raw[i], 'var(--series-a, #2563eb)')}
              {diverged[i] ? (
                <text x={center + bw * 0.6} y={ys.map(0) - 6} textAnchor="middle" className="tick-label" fill="var(--bad, #dc2626)">
                  diverges
                </text>
              ) : (
                bar(center + bw * 0.6, code[i], 'var(--good, #16a34a)')
              )}
              <text className="tick-label" x={center} y={H - M.bottom + 16} textAnchor="middle">
                {seg}
              </text>
            </g>
          )
        })}
      </svg>
      <div className="probe-legend">
        <span className="probe-legend-item">
          <span className="probe-legend-swatch" style={{ background: 'var(--series-a, #2563eb)' }} />
          raw embedding block
        </span>
        <span className="probe-legend-item">
          <span className="probe-legend-swatch" style={{ background: 'var(--good, #16a34a)' }} />
          SAE sparse code
        </span>
      </div>
    </div>
  )
}
