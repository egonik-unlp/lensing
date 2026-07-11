import './charts.css'

/** A compact profile of an MLP's hidden-layer widths: one bar per hidden layer,
 *  left→right = depth (input side → output head), bar height ∝ √width so a
 *  pyramid reads as a pyramid and a rectangle as a rectangle. This is what makes
 *  the compare chart's "relative depth" axis legible — you can see each model's
 *  shape, not just its curve. `color` ties it to the model's overlay line. */
export default function MlpTopologyDiagram({
  hidden,
  activation,
  color = 'var(--ink-secondary, #6b7280)',
  height = 60,
}: {
  hidden: number[]
  activation?: string
  color?: string
  height?: number
}) {
  const layers = hidden ?? []
  if (layers.length === 0) return null

  const barW = 16
  const gap = 10
  const padX = 6
  const labelH = 14
  const topPad = 4
  const plotH = height - labelH - topPad
  const W = padX * 2 + layers.length * barW + (layers.length - 1) * gap
  const maxW = Math.max(...layers)
  const minBar = 6
  // √ scale: perceptual across a wide 512→32 span without the tail vanishing.
  const barH = (w: number) => minBar + (plotH - minBar) * Math.sqrt(w / maxW)
  const cx = (i: number) => padX + i * (barW + gap) + barW / 2
  // Centre the bars on a midline so the widths read as a network funnel
  // (a pyramid narrows left→right), not a bottom-aligned bar chart.
  const midY = topPad + plotH / 2

  const aria = `MLP hidden layers ${layers.join(', ')}${activation ? `, ${activation}` : ''}`

  return (
    <div className="mlp-topo" role="img" aria-label={aria}>
      <svg viewBox={`0 0 ${W} ${height}`} width={W} height={height}>
        {/* faint backbone linking the layer centres */}
        <line className="mlp-topo-base" x1={cx(0)} x2={cx(layers.length - 1)} y1={midY} y2={midY} />
        {layers.map((w, i) => {
          const h = barH(w)
          return (
            <g key={i}>
              <rect
                x={cx(i) - barW / 2}
                y={midY - h / 2}
                width={barW}
                height={h}
                rx={2}
                fill={color}
                opacity={0.9}
              />
              <text className="mlp-topo-w" x={cx(i)} y={height - 3} textAnchor="middle">
                {w}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

export interface TopoItem {
  label: string
  hidden?: number[]
  activation?: string
  color: string
}

/** A titled row of topology diagrams for the compared models — colour-matched to
 *  the overlay chart, so the strip doubles as the chart's architecture legend. */
export function MlpTopologyStrip({ items }: { items: TopoItem[] }) {
  const withHidden = items.filter((it) => it.hidden && it.hidden.length > 0)
  if (withHidden.length === 0) return null
  return (
    <div className="mlp-topo-strip">
      <h4>Architectures</h4>
      <p className="interp-tools-note">
        Each model&apos;s hidden-layer widths (bar height ∝ √width), left → right by depth — the same
        depth the overlay chart plots. Colours match the lines above.
      </p>
      <div className="mlp-topo-row">
        {withHidden.map((it) => (
          <div className="mlp-topo-item" key={it.label}>
            <div className="mlp-topo-head">
              <span className="mlp-topo-swatch" style={{ background: it.color }} />
              <span className="mlp-topo-label">{it.label}</span>
            </div>
            <MlpTopologyDiagram hidden={it.hidden!} activation={it.activation} color={it.color} />
            {it.activation && <span className="mlp-topo-act">{it.activation}</span>}
          </div>
        ))}
      </div>
    </div>
  )
}
