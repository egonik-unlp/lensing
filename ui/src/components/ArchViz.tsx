import { useId, useMemo, useState, type ReactNode } from 'react'
import {
  deriveArch,
  type ArchFeatures,
  type ArchLayer,
  type ArchSpec,
  type BlendSpec,
  type ForestSpec,
  type KernelSpec,
  type LinearSpec,
  type MedianSpec,
  type MoeSpec,
  type RibbonSpec,
} from '../lib/arch'
import { fmtMoney } from '../lib/format'
import './archviz.css'

/** Architecture diagram. Every shipped predictor family renders natively
 *  from hyperparams + feature counts in one visual grammar — the signal
 *  path: features enter left, the family's transformation sits in the
 *  middle, the prediction exits right. Hovering any element drives the mono
 *  readout line. Unknown predictors fall back to the predictor-exported
 *  viz.svg image; with neither, renders nothing. */
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
  if (spec) return <ArchDiagram spec={spec} title={title} />
  if (fallbackUrl) return <ImageArch url={fallbackUrl} title={title} />
  return null
}

/** The renderer behind ArchViz, for callers that already hold a spec
 *  (BlendPanel builds one from blend.json). */
export function ArchDiagram({ spec, title }: { spec: ArchSpec; title: string }) {
  switch (spec.kind) {
    case 'ribbon':
      return <RibbonArch spec={spec} title={title} />
    case 'forest':
      return <ForestArch spec={spec} title={title} />
    case 'kernel':
      return <KernelArch spec={spec} title={title} />
    case 'linear':
      return <LinearArch spec={spec} title={title} />
    case 'moe':
      return <MoeArch spec={spec} title={title} />
    case 'median':
      return <MedianArch spec={spec} title={title} />
    case 'blend':
      return <BlendArch spec={spec} title={title} />
  }
}

/* ---------------- shared frame, fit and hover plumbing ---------------- */

/** Scales down with the panel; below 70% of natural size it scrolls instead
 *  of shrinking the type into illegibility. */
function fitStyle(w: number): React.CSSProperties {
  return { width: '100%', maxWidth: w, minWidth: Math.round(w * 0.7), height: 'auto' }
}

function ArchFigure({
  family,
  title,
  readout,
  w,
  h,
  children,
}: {
  family: string
  title: string
  readout: string
  w: number
  h: number
  children: ReactNode
}) {
  return (
    <figure className="arch-viz" role="group" aria-label={`${family} architecture`}>
      <h2 className="panel-title">{title}</h2>
      <div className="arch-canvas">
        <svg viewBox={`0 0 ${w} ${h}`} style={fitStyle(w)}>
          {children}
        </svg>
      </div>
      <figcaption className="arch-readout" aria-live="polite">
        {readout}
      </figcaption>
    </figure>
  )
}

/** Hover/focus target: pushes its detail line into the figure readout.
 *  Highlight styling is pure CSS (:hover / :focus-visible). */
function Hit({
  detail,
  onDetail,
  children,
}: {
  detail: string
  onDetail: (d: string | null) => void
  children: ReactNode
}) {
  return (
    <g
      className="arch-hit"
      tabIndex={0}
      role="img"
      aria-label={detail}
      onMouseEnter={() => onDetail(detail)}
      onMouseLeave={() => onDetail(null)}
      onFocus={() => onDetail(detail)}
      onBlur={() => onDetail(null)}
    >
      <title>{detail}</title>
      {children}
    </g>
  )
}

/** Right-pointing arrowhead marker; id is per-instance to survive multiple
 *  diagrams on one page. */
function ArrowDefs({ id }: { id: string }) {
  return (
    <defs>
      <marker
        id={id}
        viewBox="0 0 8 8"
        refX="7"
        refY="4"
        markerWidth="6"
        markerHeight="6"
        orient="auto-start-reverse"
      >
        <path d="M0 0.5 L7.5 4 L0 7.5" className="arch-arrowhead" />
      </marker>
    </defs>
  )
}

function Flow({ d, marker }: { d: string; marker?: string }) {
  return <path className="arch-flow" d={d} markerEnd={marker ? `url(#${marker})` : undefined} />
}

/** Input glyph: a feature column — stacked ticks inside a soft slot. */
function InputGlyph({ x, cy, count }: { x: number; cy: number; count: number | string }) {
  const ticks = [-18, -12, -6, 0, 6, 12, 18]
  return (
    <g>
      <rect className="arch-slot" x={x - 11} y={cy - 26} width={22} height={52} rx={4} />
      {ticks.map((dy) => (
        <line key={dy} className="arch-tick" x1={x - 5.5} x2={x + 5.5} y1={cy + dy} y2={cy + dy} />
      ))}
      <text className="arch-name" x={x} y={cy - 33} textAnchor="middle">
        features
      </text>
      <text className="arch-units" x={x} y={cy + 41} textAnchor="middle">
        {count}
      </text>
    </g>
  )
}

/** Output glyph: the prediction leaving the path. */
function OutputGlyph({ x, cy }: { x: number; cy: number }) {
  return (
    <g>
      <circle className="arch-node-dot" cx={x} cy={cy} r={4.5} />
      <text className="arch-name" x={x} y={cy - 33} textAnchor="middle">
        output
      </text>
      <text className="arch-units" x={x} y={cy + 41} textAnchor="middle">
        ŷ
      </text>
    </g>
  )
}

/* ---------------- ribbon (MLP / CNN) ---------------- */

const SLOT_W = 72
const NODE_W = 10
const PAD_X = 24
const MAX_HALF = 50 // ribbon half-height of the widest layer
const MIN_HALF = 3
const NAME_Y = 12
const RIBBON_TOP = 24

function RibbonArch({ spec, title }: { spec: RibbonSpec; title: string }) {
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
    <ArchFigure family={spec.family} title={title} readout={readout} w={w} h={h}>
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
    </ArchFigure>
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

/* ---------------- tree ensembles ---------------- */

/** Stylized tree: solid root fork, then a canopy outline whose height
 *  encodes depth. */
function TreeGlyph({
  x,
  baseY,
  depth,
  width,
  fade = 1,
}: {
  x: number
  baseY: number
  depth: number
  width: number
  fade?: number
}) {
  const d = Math.max(2, Math.min(depth, 6))
  const h = 16 + d * 6
  const top = baseY - h
  const half = width / 2
  const forkY = top + h * 0.34
  return (
    <g className="arch-tree" opacity={fade}>
      <circle className="arch-node-dot" cx={x} cy={top} r={2.6} />
      <line className="arch-tree-edge" x1={x} y1={top} x2={x - half * 0.52} y2={forkY} />
      <line className="arch-tree-edge" x1={x} y1={top} x2={x + half * 0.52} y2={forkY} />
      <circle className="arch-node-dot" cx={x - half * 0.52} cy={forkY} r={2.2} />
      <circle className="arch-node-dot" cx={x + half * 0.52} cy={forkY} r={2.2} />
      {/* canopy: the rest of the depth, drawn as an outline */}
      <path
        className="arch-tree-canopy"
        d={`M${x - half * 0.52},${forkY} L${x - half},${baseY} L${x + half},${baseY} L${x + half * 0.52},${forkY}`}
      />
      <line className="arch-tree-base" x1={x - half} y1={baseY} x2={x + half} y2={baseY} />
    </g>
  )
}

function ForestArch({ spec, title }: { spec: ForestSpec; title: string }) {
  const [detail, setDetail] = useState<string | null>(null)
  const markerId = useId()

  const boosted = spec.mode === 'boosted'
  const shown = boosted ? 4 : 3
  const treeW = 40
  const gap = boosted ? 26 : 18
  const cy = 64
  const baseY = cy + 26
  const inputX = 42
  const firstTree = inputX + 64

  const depthBits = [
    spec.depth !== null ? `depth ≤${spec.depth}` : null,
    spec.leaves !== null ? `${spec.leaves} leaves` : null,
  ].filter(Boolean)
  const treeDetail = boosted
    ? `each tree fits the residual of the ensemble so far${depthBits.length ? ` · ${depthBits.join(' · ')}` : ''}${spec.subsample !== null ? ` · row subsample ${Math.round(spec.subsample * 100)}%` : ''}`
    : `each tree trains on a bootstrap sample${depthBits.length ? ` · ${depthBits.join(' · ')}` : ''}`
  const inputDetail = `${spec.inputs} features${
    spec.colsample !== null ? ` · each tree samples ${Math.round(spec.colsample * 100)}% of columns` : ''
  }`
  const countLabel = spec.nTrees !== null ? `×${spec.nTrees.toLocaleString()}` : '×?'
  // label starts at treesEnd + 34 (left-anchored); reserve its full width
  const countW = 34 + countLabel.length * 7.2 + 10
  const countDetail =
    spec.nTrees !== null
      ? boosted
        ? `${spec.nTrees.toLocaleString()} rounds${spec.lr !== null ? `, each scaled by η ${spec.lr}` : ''}`
        : `${spec.nTrees.toLocaleString()} independent trees`
      : 'tree count not recorded'
  const combineGlyph = boosted ? 'Σ' : 'x̄'
  const combineDetail = boosted
    ? `prediction = Σ ${spec.lr !== null ? `${spec.lr}·` : 'η·'}tree(x), in transformed target space`
    : 'prediction = mean of all trees'

  if (boosted) {
    const treesEnd = firstTree + shown * (treeW + gap) - gap
    const sumX = treesEnd + countW + 46
    const outX = sumX + 64
    const w = outX + 40
    const h = 132
    return (
      <ArchFigure family={spec.family} title={title} readout={detail ?? `${spec.family} · ${spec.shape}`} w={w} h={h}>
        <ArrowDefs id={markerId} />
        <Hit detail={inputDetail} onDetail={setDetail}>
          <InputGlyph x={inputX} cy={cy} count={spec.inputs || '—'} />
        </Hit>
        <Flow d={`M${inputX + 14},${cy} H${firstTree - treeW / 2 - 8}`} marker={markerId} />
        <Hit detail={treeDetail} onDetail={setDetail}>
          <rect
            x={firstTree - treeW / 2 - 6}
            y={cy - 38}
            width={shown * (treeW + gap) - gap + 12}
            height={84}
            fill="transparent"
          />
          {Array.from({ length: shown }, (_, i) => {
            const x = firstTree + i * (treeW + gap)
            return (
              <g key={i}>
                <TreeGlyph
                  x={x}
                  baseY={baseY}
                  depth={spec.depth ?? 4}
                  width={treeW}
                  fade={1 - i * 0.17}
                />
                {i > 0 && (
                  <text className="arch-op" x={x - treeW / 2 - gap / 2} y={cy + 3} textAnchor="middle">
                    +
                  </text>
                )}
              </g>
            )
          })}
          <text className="arch-name" x={firstTree + (shown * (treeW + gap) - gap) / 2 - treeW / 2} y={NAME_Y} textAnchor="middle">
            {boosted ? 'residual fits' : 'trees'}
          </text>
        </Hit>
        <Hit detail={countDetail} onDetail={setDetail}>
          <text className="arch-count" x={treesEnd + treeW / 2 + 14} y={cy + 4}>
            {countLabel}
          </text>
        </Hit>
        <Flow d={`M${treesEnd + countW + 4},${cy} H${sumX - 22}`} marker={markerId} />
        <Hit detail={combineDetail} onDetail={setDetail}>
          <circle className="arch-node-ring" cx={sumX} cy={cy} r={15} />
          <text className="arch-op" x={sumX} y={cy + 4} textAnchor="middle">
            {combineGlyph}
          </text>
          {spec.lr !== null && (
            <text className="arch-units" x={sumX} y={cy + 41} textAnchor="middle">
              η {spec.lr}
            </text>
          )}
        </Hit>
        <Flow d={`M${sumX + 18},${cy} H${outX - 12}`} marker={markerId} />
        <Hit detail="prediction, inverted back to target space" onDetail={setDetail}>
          <OutputGlyph x={outX} cy={cy} />
        </Hit>
      </ArchFigure>
    )
  }

  // Bagged: parallel trees fanning from the input into a mean vote.
  const rows = shown
  const rowGap = 56
  const h = rows * rowGap + 44
  const midY = h / 2 - 4
  const treeX = inputX + 110
  const meanX = treeX + treeW / 2 + countW + 56
  const outX = meanX + 62
  const w = outX + 40
  return (
    <ArchFigure family={spec.family} title={title} readout={detail ?? `${spec.family} · ${spec.shape}`} w={w} h={h}>
      <ArrowDefs id={markerId} />
      <Hit detail={inputDetail} onDetail={setDetail}>
        <InputGlyph x={inputX} cy={midY} count={spec.inputs || '—'} />
      </Hit>
      {Array.from({ length: rows }, (_, i) => {
        const ty = 30 + i * rowGap + 26
        return (
          <g key={i}>
            <Flow d={`M${inputX + 14},${midY} C${inputX + 64},${midY} ${inputX + 56},${ty - 14} ${treeX - treeW / 2 - 8},${ty - 14}`} />
            <Hit detail={treeDetail} onDetail={setDetail}>
              <rect x={treeX - treeW / 2 - 4} y={ty - 44} width={treeW + 8} height={52} fill="transparent" />
              <TreeGlyph x={treeX} baseY={ty} depth={spec.depth ?? 5} width={treeW} />
            </Hit>
            <Flow d={`M${treeX + treeW / 2 + 6},${ty - 14} C${meanX - 40},${ty - 14} ${meanX - 44},${midY} ${meanX - 20},${midY}`} />
          </g>
        )
      })}
      <Hit detail={countDetail} onDetail={setDetail}>
        <text className="arch-count" x={treeX + treeW / 2 + 18} y={midY + 4}>
          {countLabel}
        </text>
      </Hit>
      <Hit detail={combineDetail} onDetail={setDetail}>
        <circle className="arch-node-ring" cx={meanX} cy={midY} r={15} />
        <text className="arch-op" x={meanX} y={midY + 4} textAnchor="middle">
          {combineGlyph}
        </text>
      </Hit>
      <Flow d={`M${meanX + 18},${midY} H${outX - 12}`} marker={markerId} />
      <Hit detail="prediction, inverted back to target space" onDetail={setDetail}>
        <OutputGlyph x={outX} cy={midY} />
      </Hit>
    </ArchFigure>
  )
}

/* ---------------- kernel machines ---------------- */

function KernelArch({ spec, title }: { spec: KernelSpec; title: string }) {
  const [detail, setDetail] = useState<string | null>(null)
  const markerId = useId()

  const cy = 64
  const inputX = 42
  const phiX = inputX + 92
  const fitX = phiX + 64
  const fitW = 170
  const outX = fitX + fitW + 64
  const w = outX + 40
  const h = 132

  const svr = spec.variant === 'svr'
  const gammaBit = spec.kernel === 'rbf' ? ` · γ ${spec.gamma !== null ? spec.gamma : 'scale'}` : ''
  const phiDetail = `φ maps features into ${spec.kernel} kernel space${gammaBit}${
    spec.degree !== null ? ` · degree ${spec.degree}` : ''
  }`
  const fitDetail = svr
    ? `ε-tube${spec.epsilon !== null ? ` ${spec.epsilon}` : ''}: errors inside the tube are free; C${
        spec.c !== null ? ` ${spec.c}` : ''
      } prices the rest · support vectors sit on the tube`
    : `L2-shrunk fit in kernel space${spec.alpha !== null ? ` · α ${spec.alpha}` : ''} · every train row contributes`

  // The fit glyph: a gentle curve through kernel space.
  const x0 = fitX
  const x1 = fitX + fitW
  const curve = (dy: number) =>
    `M${x0},${cy + 14 + dy} C${x0 + fitW * 0.3},${cy - 26 + dy} ${x0 + fitW * 0.55},${cy + 30 + dy} ${x1},${cy - 12 + dy}`
  const curveYAt = (t: number, dy: number) => {
    // cubic bezier y at parameter t for the curve above
    const y0 = cy + 14 + dy
    const y1 = cy - 26 + dy
    const y2 = cy + 30 + dy
    const y3 = cy - 12 + dy
    const u = 1 - t
    return u * u * u * y0 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t * t * t * y3
  }
  const xAt = (t: number) => {
    const cx1 = x0 + fitW * 0.3
    const cx2 = x0 + fitW * 0.55
    const u = 1 - t
    return u * u * u * x0 + 3 * u * u * t * cx1 + 3 * u * t * t * cx2 + t * t * t * x1
  }
  const tube = 11
  const svDots = svr
    ? [0.12, 0.34, 0.58, 0.82].map((t, i) => ({
        x: xAt(t),
        y: curveYAt(t, i % 2 === 0 ? -tube : tube),
      }))
    : [0.1, 0.3, 0.5, 0.7, 0.9].map((t, i) => ({
        x: xAt(t),
        y: curveYAt(t, (i % 2 === 0 ? -1 : 1) * 4),
      }))

  return (
    <ArchFigure family={spec.family} title={title} readout={detail ?? `${spec.family} · ${spec.shape}`} w={w} h={h}>
      <ArrowDefs id={markerId} />
      <Hit detail={`${spec.inputs} features, standardized on the train split`} onDetail={setDetail}>
        <InputGlyph x={inputX} cy={cy} count={spec.inputs || '—'} />
      </Hit>
      <Flow d={`M${inputX + 14},${cy} H${phiX - 22}`} marker={markerId} />
      <Hit detail={phiDetail} onDetail={setDetail}>
        <circle className="arch-node-ring" cx={phiX} cy={cy} r={16} />
        <text className="arch-op" x={phiX} y={cy + 4} textAnchor="middle">
          φ
        </text>
        <text className="arch-name" x={phiX} y={cy - 33} textAnchor="middle">
          kernel map
        </text>
        <text className="arch-units" x={phiX} y={cy + 41} textAnchor="middle">
          {spec.kernel}
          {spec.degree !== null ? ` d${spec.degree}` : ''}
        </text>
      </Hit>
      <Flow d={`M${phiX + 19},${cy} H${fitX - 10}`} marker={markerId} />
      <Hit detail={fitDetail} onDetail={setDetail}>
        <rect x={fitX - 6} y={cy - 40} width={fitW + 12} height={84} fill="transparent" />
        {svr && (
          <>
            <path className="arch-kernel-tube" d={curve(-tube)} />
            <path className="arch-kernel-tube" d={curve(tube)} />
          </>
        )}
        <path className="arch-kernel-fit" d={curve(0)} />
        {svDots.map((p, i) => (
          <circle key={i} className="arch-sv-dot" cx={p.x} cy={p.y} r={2.6} />
        ))}
        <text className="arch-name" x={fitX + fitW / 2} y={NAME_Y} textAnchor="middle">
          {svr ? 'ε-insensitive fit' : 'ridge fit in kernel space'}
        </text>
        <text className="arch-units" x={fitX + fitW / 2} y={cy + 41} textAnchor="middle">
          {svr
            ? `${spec.c !== null ? `C ${spec.c}` : ''}${spec.c !== null && spec.epsilon !== null ? ' · ' : ''}${spec.epsilon !== null ? `ε ${spec.epsilon}` : ''}` || '—'
            : spec.alpha !== null
              ? `α ${spec.alpha}`
              : '—'}
        </text>
      </Hit>
      <Flow d={`M${fitX + fitW + 8},${cy} H${outX - 12}`} marker={markerId} />
      <Hit detail="prediction, inverted back to target space" onDetail={setDetail}>
        <OutputGlyph x={outX} cy={cy} />
      </Hit>
    </ArchFigure>
  )
}

/* ---------------- linear (ridge) ---------------- */

function LinearArch({ spec, title }: { spec: LinearSpec; title: string }) {
  const [detail, setDetail] = useState<string | null>(null)
  const markerId = useId()

  const cy = 64
  const inputX = 42
  const wRackX = inputX + 78
  const sumX = wRackX + 132
  const outX = sumX + 70
  const w = outX + 40
  const h = 132

  // Symbolic coefficient rack: a shrinking weight profile (L2's signature).
  const bars = [30, 24, 19, 15, 11, 8, 6, 4]
  const barGap = 7

  const alphaLabel = spec.autoAlpha ? 'α by CV' : spec.alpha !== null ? `α ${spec.alpha}` : null
  const sumDetail = `ŷ = w·x + b · L2 shrinkage ${
    spec.autoAlpha ? 'α selected by cross-validation' : spec.alpha !== null ? `α ${spec.alpha}` : ''
  }${spec.loss === 'huber' ? ` · huber loss${spec.huberEpsilon !== null ? ` (ε ${spec.huberEpsilon})` : ''}` : ''}`
  const rackDetail = `one coefficient per feature; the L2 penalty shrinks the rack toward zero${
    spec.weightGamma !== null ? ` · rows weighted by target^${spec.weightGamma}` : ''
  }`

  return (
    <ArchFigure family={spec.family} title={title} readout={detail ?? `${spec.family} · ${spec.shape}`} w={w} h={h}>
      <ArrowDefs id={markerId} />
      <Hit detail={`${spec.inputs} features, standardized on the train split`} onDetail={setDetail}>
        <InputGlyph x={inputX} cy={cy} count={spec.inputs || '—'} />
      </Hit>
      <Flow d={`M${inputX + 14},${cy} H${wRackX - 12}`} marker={markerId} />
      <Hit detail={rackDetail} onDetail={setDetail}>
        <rect x={wRackX - 8} y={cy - 36} width={56} height={76} fill="transparent" />
        {bars.map((b, i) => (
          <rect
            key={i}
            className="arch-weight-bar"
            x={wRackX}
            y={cy - 28 + i * barGap}
            width={b}
            height={3.5}
            rx={1.5}
          />
        ))}
        <text className="arch-name" x={wRackX + 18} y={cy - 33} textAnchor="middle">
          weights
        </text>
        <text className="arch-units" x={wRackX + 18} y={cy + 41} textAnchor="middle">
          w₁…w{subscript(spec.inputs)}
        </text>
      </Hit>
      <Flow d={`M${wRackX + 44},${cy} H${sumX - 22}`} marker={markerId} />
      <Hit detail={sumDetail} onDetail={setDetail}>
        <circle className="arch-node-ring" cx={sumX} cy={cy} r={15} />
        <text className="arch-op" x={sumX} y={cy + 4} textAnchor="middle">
          Σ
        </text>
        {alphaLabel && (
          <text className="arch-units" x={sumX} y={cy + 41} textAnchor="middle">
            {alphaLabel}
          </text>
        )}
        <text className="arch-name" x={sumX} y={cy - 33} textAnchor="middle">
          w·x + b
        </text>
      </Hit>
      <Flow d={`M${sumX + 18},${cy} H${outX - 12}`} marker={markerId} />
      <Hit detail="prediction, inverted back to target space" onDetail={setDetail}>
        <OutputGlyph x={outX} cy={cy} />
      </Hit>
    </ArchFigure>
  )
}

function subscript(n: number): string {
  if (n <= 0) return 'ₙ'
  const map: Record<string, string> = { 0: '₀', 1: '₁', 2: '₂', 3: '₃', 4: '₄', 5: '₅', 6: '₆', 7: '₇', 8: '₈', 9: '₉' }
  return String(n).replace(/\d/g, (d) => map[d])
}

/* ---------------- mixture of experts ---------------- */

function MoeArch({ spec, title }: { spec: MoeSpec; title: string }) {
  const [detail, setDetail] = useState<string | null>(null)
  const markerId = useId()

  const shown = Math.min(spec.nExperts, 5)
  const rowGap = 44
  const h = Math.max(shown * rowGap + 52, 150)
  const midY = h / 2 - 4
  const inputX = 42
  const gateX = inputX + 104
  const expertX = gateX + 116
  const mergeX = expertX + 104
  const outX = mergeX + 62
  const w = outX + 40

  const expertYs = Array.from({ length: shown }, (_, i) => 34 + i * rowGap + 8)
  const overflow = spec.nExperts > shown

  const mergeDetail =
    spec.gate === 'band'
      ? 'prediction = Σ gate-probability · expert(x)'
      : 'prediction = Σ regime-responsibility · expert(x)'

  return (
    <ArchFigure family={spec.family} title={title} readout={detail ?? `${spec.family} · ${spec.shape}`} w={w} h={h}>
      <ArrowDefs id={markerId} />
      <Hit detail={`${spec.inputs} features, standardized on the train split`} onDetail={setDetail}>
        <InputGlyph x={inputX} cy={midY} count={spec.inputs || '—'} />
      </Hit>
      <Flow d={`M${inputX + 14},${midY} H${gateX - 26}`} marker={markerId} />
      <Hit detail={spec.gateDetail} onDetail={setDetail}>
        {spec.gate === 'band' ? (
          <g>
            <rect className="arch-slot" x={gateX - 20} y={midY - 26} width={40} height={52} rx={4} />
            {Array.from({ length: Math.min(spec.nExperts, 5) }, (_, i) => (
              <rect
                key={i}
                className="arch-gate-band"
                x={gateX - 15}
                y={midY - 21 + i * (44 / Math.min(spec.nExperts, 5))}
                width={30}
                height={44 / Math.min(spec.nExperts, 5) - 2.5}
                rx={1.5}
                opacity={0.25 + (0.75 * i) / Math.max(Math.min(spec.nExperts, 5) - 1, 1)}
              />
            ))}
          </g>
        ) : (
          <g>
            <rect className="arch-slot" x={gateX - 20} y={midY - 26} width={40} height={52} rx={4} />
            {[
              [-8, -12, 3.4], [7, -7, 4.4], [-6, 6, 4], [8, 12, 3.2], [-1, 16, 2.6],
            ].slice(0, Math.max(3, Math.min(spec.nExperts, 5))).map(([dx, dy, r], i) => (
              <circle key={i} className="arch-gate-blob" cx={gateX + dx} cy={midY + dy} r={r} />
            ))}
          </g>
        )}
        <text className="arch-name" x={gateX} y={midY - 33} textAnchor="middle">
          gate
        </text>
        <text className="arch-units" x={gateX} y={midY + 41} textAnchor="middle">
          {spec.gate === 'band' ? `${spec.nExperts} bands` : `GMM k=${spec.nExperts}`}
        </text>
      </Hit>
      {expertYs.map((ey, i) => (
        <g key={i}>
          <Flow d={`M${gateX + 22},${midY} C${gateX + 62},${midY} ${gateX + 54},${ey} ${expertX - 20},${ey}`} />
          <Hit detail={spec.expertDetail} onDetail={setDetail}>
            <circle className="arch-node-ring" cx={expertX} cy={ey} r={12} />
            <text className="arch-op arch-op-sm" x={expertX} y={ey + 3.5} textAnchor="middle">
              φ
            </text>
          </Hit>
          <Flow d={`M${expertX + 15},${ey} C${expertX + 56},${ey} ${expertX + 48},${midY} ${mergeX - 19},${midY}`} />
        </g>
      ))}
      <text className="arch-name" x={expertX} y={NAME_Y} textAnchor="middle">
        {overflow ? `experts ×${spec.nExperts}` : 'experts'}
      </text>
      <Hit detail={mergeDetail} onDetail={setDetail}>
        <circle className="arch-node-ring" cx={mergeX} cy={midY} r={15} />
        <text className="arch-op" x={mergeX} y={midY + 4} textAnchor="middle">
          Σ
        </text>
      </Hit>
      <Flow d={`M${mergeX + 18},${midY} H${outX - 12}`} marker={markerId} />
      <Hit detail="prediction, inverted back to target space" onDetail={setDetail}>
        <OutputGlyph x={outX} cy={midY} />
      </Hit>
    </ArchFigure>
  )
}

/* ---------------- median baseline ---------------- */

function MedianArch({ spec, title }: { spec: MedianSpec; title: string }) {
  const [detail, setDetail] = useState<string | null>(null)
  const markerId = useId()

  const cy = 64
  const inputX = 42
  const groupX = inputX + 104
  const tableX = groupX + 110
  const outX = tableX + 104
  const w = outX + 40
  const h = 132
  const group = spec.group ?? 'category'

  return (
    <ArchFigure family={spec.family} title={title} readout={detail ?? `${spec.family} · ${spec.shape}`} w={w} h={h}>
      <ArrowDefs id={markerId} />
      <Hit detail="reads the payload only — the feature matrix is irrelevant to it" onDetail={setDetail}>
        <InputGlyph x={inputX} cy={cy} count="payload" />
      </Hit>
      <Flow d={`M${inputX + 14},${cy} H${groupX - 30}`} marker={markerId} />
      <Hit detail={`rows split by ${group}`} onDetail={setDetail}>
        {[-16, 0, 16].map((dy, i) => (
          <g key={i}>
            <path className="arch-flow" d={`M${groupX - 26},${cy} C${groupX - 10},${cy} ${groupX - 12},${cy + dy} ${groupX + 2},${cy + dy}`} />
            <circle className="arch-node-dot" cx={groupX + 6} cy={cy + dy} r={2.6} />
          </g>
        ))}
        <text className="arch-name" x={groupX - 6} y={cy - 33} textAnchor="middle">
          group by
        </text>
        <text className="arch-units" x={groupX - 6} y={cy + 41} textAnchor="middle">
          {group}
        </text>
      </Hit>
      <Flow d={`M${groupX + 16},${cy} H${tableX - 28}`} marker={markerId} />
      <Hit detail="train-split median per group; unseen groups fall back to the global median" onDetail={setDetail}>
        <rect className="arch-slot" x={tableX - 24} y={cy - 22} width={48} height={44} rx={4} />
        {[-13, -4.5, 4, 12.5].map((dy, i) => (
          <line
            key={i}
            className={i === 1 ? 'arch-median-row' : 'arch-tick'}
            x1={tableX - 16}
            x2={tableX + 16}
            y1={cy + dy + 1.5}
            y2={cy + dy + 1.5}
          />
        ))}
        <text className="arch-name" x={tableX} y={cy - 33} textAnchor="middle">
          lookup
        </text>
        <text className="arch-units" x={tableX} y={cy + 41} textAnchor="middle">
          median
        </text>
      </Hit>
      <Flow d={`M${tableX + 30},${cy} H${outX - 12}`} marker={markerId} />
      <Hit detail="the floor every model must beat" onDetail={setDetail}>
        <OutputGlyph x={outX} cy={cy} />
      </Hit>
    </ArchFigure>
  )
}

/* ---------------- blend (member fan-in) ---------------- */

export function BlendArch({ spec, title }: { spec: BlendSpec; title: string }) {
  const [detail, setDetail] = useState<string | null>(null)
  const markerId = useId()

  const n = spec.members.length
  const rowH = 40
  const padTop = 26
  const h = Math.max(padTop + n * rowH + 22, 130)
  const midY = padTop + (n * rowH) / 2 - rowH / 2 + 14
  const boxW = 196
  const boxX = 16
  const combX = boxX + boxW + 108
  const outX = combX + 66
  const w = outX + 44

  const maxW = Math.max(...spec.members.map((m) => m.weight), 0.0001)
  const combGlyph = spec.rule === 'median' ? 'x̃' : 'Σ'
  const combDetail =
    spec.rule === 'median'
      ? 'median vote across members, in target space'
      : spec.weightFit === 'grid'
        ? 'weighted mean; weights grid-searched on a validation split'
        : 'weighted mean in target space'

  return (
    <ArchFigure family={spec.family} title={title} readout={detail ?? `${spec.family} · ${spec.shape}`} w={w} h={h}>
      <ArrowDefs id={markerId} />
      {spec.members.map((m, i) => {
        const y = padTop + i * rowH
        const cy = y + 14
        const stroke = m.excluded ? 1 : 1 + 2.5 * (m.weight / maxW)
        return (
          <g key={m.key} className={m.excluded ? 'arch-blend-excluded' : undefined}>
            <Hit detail={`${m.detail}${m.soloMae !== null ? ` · solo MAE ${fmtMoney(m.soloMae)}` : ''}`} onDetail={setDetail}>
              <rect className="arch-member-box" x={boxX} y={y} width={boxW} height={28} rx={4} />
              <text className="arch-member-name" x={boxX + 9} y={y + 18}>
                {truncate(m.label, 22)}
              </text>
              <text className="arch-member-sub" x={boxX + boxW - 8} y={y + 18} textAnchor="end">
                {m.predictor || (m.frozen ? 'frozen' : m.kind)}
              </text>
            </Hit>
            <path
              className="arch-blend-edge"
              style={{ strokeWidth: stroke }}
              d={`M${boxX + boxW + 2},${cy} C${boxX + boxW + 56},${cy} ${combX - 60},${midY} ${combX - 19},${midY}`}
            />
            {!m.excluded && spec.rule === 'mean' && (
              <Hit detail={`weight ${(m.weight * 100).toFixed(0)}%${spec.weightFit === 'grid' ? ' (grid-fitted)' : ''}`} onDetail={setDetail}>
                <text className="arch-weight-label" x={boxX + boxW + 12} y={cy - 5}>
                  {(m.weight * 100).toFixed(0)}%
                </text>
              </Hit>
            )}
          </g>
        )
      })}
      <Hit detail={combDetail} onDetail={setDetail}>
        <circle className="arch-node-ring" cx={combX} cy={midY} r={16} />
        <text className="arch-op" x={combX} y={midY + 4} textAnchor="middle">
          {combGlyph}
        </text>
        <text className="arch-units" x={combX} y={midY + 41} textAnchor="middle">
          {spec.rule === 'median' ? 'median vote' : 'weighted mean'}
        </text>
      </Hit>
      <Flow d={`M${combX + 19},${midY} H${outX - 12}`} marker={markerId} />
      <Hit detail="consensus prediction in target space" onDetail={setDetail}>
        <OutputGlyph x={outX} cy={midY} />
      </Hit>
    </ArchFigure>
  )
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + '…' : s
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
