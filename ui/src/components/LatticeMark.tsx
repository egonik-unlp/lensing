import { useMemo } from 'react'
import './lattice.css'

/* The empty-state mark: a regular lattice curving around a mass at its center,
 * the way the lab bends to fit your data. Companion to the Velocity Map brand
 * mark; here it stays monochrome (graphite ring, ink mass, hairline grid) —
 * the velocity ramp is brand-surface-only, and the one orange on an empty
 * screen belongs to its primary action. The ring draws in on mount; reduced
 * motion shows it resting and visible. */

const C = 60
const RE = 30

function deflect(x: number, y: number, strength = 1, core = 0.55, minr = 0.46): [number, number] {
  const dx = x - C
  const dy = y - C
  const r = Math.hypot(dx, dy)
  if (r < 1e-6) return [x, y]
  const ru = r / RE
  const p = strength / Math.sqrt(ru * ru + core * core)
  const nru = Math.max(ru - p, minr)
  return [C + (dx / r) * (nru * RE), C + (dy / r) * (nru * RE)]
}

export default function LatticeMark({ className }: { className?: string }) {
  const lines = useMemo(() => {
    const step = 12
    const n = 5
    const samp = 40
    const lo = C - n * step
    const hi = C + n * step
    const coords = Array.from({ length: 2 * n + 1 }, (_, i) => C + (i - n) * step)
    const sample = (fixed: number, axis: 'x' | 'y') => {
      const pts: string[] = []
      for (let i = 0; i <= samp; i++) {
        const t = lo + (hi - lo) * (i / samp)
        const [px, py] = axis === 'x' ? deflect(fixed, t) : deflect(t, fixed)
        pts.push(`${px.toFixed(1)},${py.toFixed(1)}`)
      }
      return pts.join(' ')
    }
    return [
      ...coords.map((gx) => sample(gx, 'x')),
      ...coords.map((gy) => sample(gy, 'y')),
    ]
  }, [])

  return (
    <svg
      className={className ? `lattice-mark ${className}` : 'lattice-mark'}
      viewBox="0 0 120 120"
      role="img"
      aria-label="A regular grid curving around a mass at its center"
    >
      <g className="lattice-grid">
        {lines.map((p, i) => (
          <polyline key={i} points={p} />
        ))}
      </g>
      <circle className="lattice-ring" cx={C} cy={C} r={RE} />
      <circle className="lattice-mass" cx={C} cy={C} r={RE * 0.3} />
    </svg>
  )
}
