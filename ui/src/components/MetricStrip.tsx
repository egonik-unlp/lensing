/* One metric vocabulary for headline numbers. A strip of cells: mono value
 * over a small label. Clickable cells drill into the predictions behind the
 * aggregate (PRODUCT.md: no dead-end summaries). */

export interface MetricCellSpec {
  label: string
  value: string
  hint?: string
  /** Present = the cell drills down (renders as a button). */
  onClick?: () => void
}

export function MetricCell({ label, value, hint, onClick }: MetricCellSpec) {
  if (!onClick) {
    return (
      <div className="metric-cell">
        <span className="metric-label">{label}</span>
        <span className="num metric-value">{value}</span>
        {hint && <span className="metric-hint">{hint}</span>}
      </div>
    )
  }
  return (
    <button
      className="metric-cell metric-cell-drill"
      onClick={onClick}
      title={`${hint ?? label} — view the predictions behind it`}
    >
      <span className="metric-label">{label}</span>
      <span className="num metric-value">{value}</span>
      {hint && <span className="metric-hint">{hint}</span>}
    </button>
  )
}

export default function MetricStrip({
  metrics,
  ariaLabel,
}: {
  metrics: MetricCellSpec[]
  ariaLabel?: string
}) {
  return (
    <div className="metrics-strip" role="group" aria-label={ariaLabel}>
      {metrics.map((m) => (
        <MetricCell key={m.label} {...m} />
      ))}
    </div>
  )
}
