import { useMemo, useState } from 'react'
import { deriveArch, type ArchFeatures, type ArchLayer, type ArchSpec } from '../lib/arch'
import './archviz.css'

/** Architecture diagram. Known predictor families (MLP/CNN) render natively
 *  from hyperparams + feature counts: a width-profile ribbon where each
 *  layer's height encodes its activation count, so deep models stay legible
 *  (the canvas scrolls instead of shrinking). Unknown families fall back to
 *  the predictor-exported viz.svg image; with neither, renders nothing. */
export default function ArchViz({
  predictor,
  hyperparams,
  features,
  fallbackUrl,
  title = 'Architecture',
}: {
  predictor: string
  hyperparams: Record<string, unknown> | null
  features: ArchFeatures | null
  /** viz.svg endpoint (registry capability `visualization`). */
  fallbackUrl?: string
  title?: string
}) {
  const spec = useMemo(
    () => deriveArch(predictor, hyperparams, features),
    [predictor, hyperparams, features],
  )
  if (spec) return <RibbonArch spec={spec} title={title} />
  if (fallbackUrl) return <ImageArch url={fallbackUrl} title={title} />
  return null
}

/* ---------------- native ribbon diagram ---------------- */

const SLOT_W = 72
const NODE_W = 10
const PAD_X = 24
const MAX_HALF = 50 // ribbon half-height of the widest layer
const MIN_HALF = 3
const NAME_Y = 12
const RIBBON_TOP = 24

function RibbonArch({ spec, title }: { spec: ArchSpec; title: string }) {
  const [active, setActive] = useState<number | null>(null)

  const n = spec.layers.length
  const w = PAD_X * 2 + n * SLOT_W
  const yc = RIBBON_TOP + MAX_HALF
  const ribbonBottom = yc + MAX_HALF
  const laneY = ribbonBottom + 16 // meta-bypass lane (CNN only)
  const unitsY = (spec.metaJoin ? laneY + 26 : ribbonBottom + 16) + 4
  const h = unitsY + 8

  const maxM = Math.max(...spec.layers.map((l) => l.magnitude))
  const xs = spec.layers.map((_, i) => PAD_X + SLOT_W / 2 + i * SLOT_W)
  const hs = spec.layers.map((l) =>
    Math.max(MIN_HALF, MAX_HALF * Math.sqrt(l.magnitude / maxM)),
  )

  const layer = active === null ? null : spec.layers[active]
  const readout = layer
    ? `${layer.name} · ${layer.detail}${layer.params > 0 ? ` · ${layer.params.toLocaleString()} params` : ''}`
    : `${spec.family} · ${spec.shape} · ${spec.totalParams.toLocaleString()} params`

  return (
    <figure className="arch-viz" role="group" aria-label={`${spec.family} architecture`}>
      <h2 className="panel-title">{title}</h2>
      <div className="arch-canvas">
        <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
          <path className="arch-band" d={bandPath(xs, hs, yc)} />
          {spec.metaJoin && (
            <MetaBypass xs={xs} hs={hs} yc={yc} laneY={laneY} join={spec.metaJoin} />
          )}
          {spec.layers.map((l, i) => (
            <LayerNode
              key={l.key}
              layer={l}
              x={xs[i]}
              half={hs[i]}
              yc={yc}
              unitsY={unitsY}
              ribbonHeight={2 * MAX_HALF}
              active={active === i}
              onActive={(on) => setActive(on ? i : null)}
            />
          ))}
        </svg>
      </div>
      <figcaption className="arch-readout" aria-live="polite">
        {readout}
      </figcaption>
    </figure>
  )
}

/** Smooth band through every layer: top edge left→right, bottom edge back. */
function bandPath(xs: number[], hs: number[], yc: number): string {
  const n = xs.length
  let d = `M${xs[0]},${(yc - hs[0]).toFixed(1)}`
  for (let i = 1; i < n; i++) {
    const mx = (xs[i - 1] + xs[i]) / 2
    d += `C${mx},${(yc - hs[i - 1]).toFixed(1)} ${mx},${(yc - hs[i]).toFixed(1)} ${xs[i]},${(yc - hs[i]).toFixed(1)}`
  }
  d += `L${xs[n - 1]},${(yc + hs[n - 1]).toFixed(1)}`
  for (let i = n - 2; i >= 0; i--) {
    const mx = (xs[i] + xs[i + 1]) / 2
    d += `C${mx},${(yc + hs[i + 1]).toFixed(1)} ${mx},${(yc + hs[i]).toFixed(1)} ${xs[i]},${(yc + hs[i]).toFixed(1)}`
  }
  return d + 'Z'
}

/** The meta features skip the conv stack: a dashed slate line from the input
 *  down a lower lane, rejoining the ribbon at the concat layer. */
function MetaBypass({
  xs,
  hs,
  yc,
  laneY,
  join,
}: {
  xs: number[]
  hs: number[]
  yc: number
  laneY: number
  join: { index: number; nMeta: number }
}) {
  const x0 = xs[0]
  const xj = xs[join.index]
  const bend = SLOT_W * 0.6
  const d =
    `M${x0},${(yc + hs[0]).toFixed(1)}` +
    `Q${x0},${laneY} ${x0 + bend},${laneY}` +
    `L${xj - bend},${laneY}` +
    `Q${xj},${laneY} ${xj},${(yc + hs[join.index]).toFixed(1)}`
  return (
    <g aria-hidden>
      <path className="arch-meta-path" d={d} />
      <text className="arch-meta-label" x={(x0 + xj) / 2} y={laneY + 12} textAnchor="middle">
        {join.nMeta} meta · skips conv
      </text>
    </g>
  )
}

function LayerNode({
  layer,
  x,
  half,
  yc,
  unitsY,
  ribbonHeight,
  active,
  onActive,
}: {
  layer: ArchLayer
  x: number
  half: number
  yc: number
  unitsY: number
  ribbonHeight: number
  active: boolean
  onActive: (on: boolean) => void
}) {
  return (
    <g
      className={`arch-hit${active ? ' is-active' : ''}`}
      tabIndex={0}
      role="img"
      aria-label={`${layer.name}: ${layer.units} — ${layer.detail}`}
      onMouseEnter={() => onActive(true)}
      onMouseLeave={() => onActive(false)}
      onFocus={() => onActive(true)}
      onBlur={() => onActive(false)}
    >
      <title>{`${layer.name} · ${layer.detail}`}</title>
      {/* full-height invisible hit target so hovering anywhere in the slot works */}
      <rect
        x={x - SLOT_W / 2}
        y={yc - ribbonHeight / 2 - 12}
        width={SLOT_W}
        height={ribbonHeight + 24}
        fill="transparent"
      />
      <rect
        className="arch-node"
        x={x - NODE_W / 2}
        y={yc - half - 2}
        width={NODE_W}
        height={2 * half + 4}
        rx={2}
      />
      <text className="arch-name" x={x} y={NAME_Y} textAnchor="middle">
        {layer.name}
      </text>
      <text className="arch-units" x={x} y={unitsY} textAnchor="middle">
        {layer.units}
      </text>
    </g>
  )
}

/* ---------------- fallback: predictor-exported viz.svg ---------------- */

function ImageArch({ url, title }: { url: string; title: string }) {
  const [failed, setFailed] = useState(false)

  // A new url (different run/model) gets a fresh chance: reset during
  // render instead of in an effect (react-hooks/set-state-in-effect).
  const [lastUrl, setLastUrl] = useState(url)
  if (url !== lastUrl) {
    setLastUrl(url)
    setFailed(false)
  }

  if (failed) return null
  return (
    <div className="arch-viz">
      <h2 className="panel-title">{title}</h2>
      <img
        src={url}
        alt="Model architecture diagram exported by the predictor"
        onError={() => setFailed(true)}
      />
    </div>
  )
}
