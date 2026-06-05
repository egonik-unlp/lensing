interface Props {
  values: number[]
  width?: number
  height?: number
}

/** Tiny inline val-loss trend for live run rows. */
export default function Sparkline({ values, width = 56, height = 18 }: Props) {
  if (values.length < 2) return null
  const finite = values.filter((v) => isFinite(v) && v > 0)
  if (finite.length < 2) return null
  const logs = values.map((v) => Math.log10(Math.max(v, Number.MIN_VALUE)))
  const lo = Math.min(...logs)
  const hi = Math.max(...logs)
  const span = hi - lo || 1
  const coords = logs.map((v, i) => {
    const x = (i / (logs.length - 1)) * (width - 2) + 1
    const y = height - 2 - ((v - lo) / span) * (height - 4)
    return [x, y] as const
  })
  const pts = coords.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ')
  const [hx, hy] = coords[coords.length - 1]
  return (
    <svg width={width} height={height} aria-hidden style={{ display: 'inline-block', verticalAlign: 'middle' }}>
      <polyline points={pts} fill="none" stroke="var(--accent-strong)" strokeWidth={1.25} />
      {/* head dot: where the live trend is right now */}
      <circle cx={hx.toFixed(1)} cy={hy.toFixed(1)} r={1.75} fill="var(--accent-strong)" />
    </svg>
  )
}
