// Heat ramp: quantitative color for the chart layer. Viridis-family adapted
// for the paper readout — lightness descends monotonically (the signal
// survives grayscale, honoring "never hue alone"), hue runs yellow-green →
// teal → blue → indigo and never enters orange (orange means live).
// Darkest = most / best, always. Stops mirror --heat-0..6 in tokens.css;
// change them together.

const STOPS: [number, number, number][] = [
  // [L, C, H] — oklch
  [0.92, 0.13, 110],
  [0.82, 0.13, 150],
  [0.7, 0.11, 185],
  [0.58, 0.1, 220],
  [0.46, 0.12, 250],
  [0.37, 0.12, 275],
  [0.28, 0.1, 290],
]

const clamp01 = (t: number) => (Number.isFinite(t) ? Math.min(1, Math.max(0, t)) : 0)

function lerpStops(t: number): [number, number, number] {
  const x = clamp01(t) * (STOPS.length - 1)
  const i = Math.min(STOPS.length - 2, Math.floor(x))
  const f = x - i
  const a = STOPS[i]
  const b = STOPS[i + 1]
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}

/** Ramp color at t ∈ [0,1]. t=0 light yellow-green, t=1 deep indigo. */
export function heatColor(t: number): string {
  const [l, c, h] = lerpStops(t)
  return `oklch(${l.toFixed(3)} ${c.toFixed(3)} ${h.toFixed(1)})`
}

/** Lightness of the ramp at t — for contrast decisions. */
export function heatLightness(t: number): number {
  return lerpStops(t)[0]
}

/** Text color legible on a heat cell: ink while the cell is light,
 *  paper once the ramp passes mid-lightness. */
export function heatTextColor(t: number): string {
  return heatLightness(t) >= 0.62 ? 'var(--ink)' : 'var(--on-heat-dark)'
}

// Cells that print their value need ≥4.5:1 for the printed number, but the
// ramp's midband (L ≈ 0.48–0.66) is a dead zone where neither ink nor paper
// gets there. Cell backgrounds skip the band — lightness steps over it like
// a quantized choropleth while hue keeps moving, so every cell stays
// readable and the dark-is-better order is preserved.
const CELL_L_LIGHT_MIN = 0.67 // ink text holds 4.5:1 down to here
const CELL_L_DARK_MAX = 0.47 // paper text holds 4.5:1 up to here

/** Background + text pair for a value-printing heat cell at t ∈ [0,1]. */
export function heatCell(t: number): { bg: string; text: string } {
  const [rawL, c, h] = lerpStops(t)
  let l = rawL
  if (l < CELL_L_LIGHT_MIN && l > CELL_L_DARK_MAX) {
    l = l >= (CELL_L_LIGHT_MIN + CELL_L_DARK_MAX) / 2 ? CELL_L_LIGHT_MIN : CELL_L_DARK_MAX
  }
  return {
    bg: `oklch(${l.toFixed(3)} ${c.toFixed(3)} ${h.toFixed(1)})`,
    text: l >= CELL_L_LIGHT_MIN ? 'var(--ink)' : 'var(--on-heat-dark)',
  }
}

// Diverging variant for signed error: blue = under-prediction, red =
// over-prediction (the --bad hue), intensity by magnitude. The near-zero
// middle stays light but never invisible on paper.
const DIV_NEUTRAL_L = 0.88
const DIV_DEEP_L = 0.48

/** Diverging color at t ∈ [-1,1]. Negative = blue (under), positive = red (over). */
export function divergingColor(t: number): string {
  const m = Math.min(1, Math.abs(Number.isFinite(t) ? t : 0))
  const l = DIV_NEUTRAL_L + (DIV_DEEP_L - DIV_NEUTRAL_L) * m
  const c = 0.03 + 0.13 * m
  const h = t < 0 ? 250 : 27
  return `oklch(${l.toFixed(3)} ${c.toFixed(3)} ${h})`
}

/** Local point density via grid-binned neighbor counting in normalized
 *  coordinate space (O(n·k)). Returns each point's neighbor count
 *  (including itself) within `radius`. */
export function pointDensities(pts: { x: number; y: number }[], radius: number): number[] {
  const cell = radius
  const key = (cx: number, cy: number) => cx * 100_003 + cy
  const grid = new Map<number, number[]>()
  for (let i = 0; i < pts.length; i++) {
    const k = key(Math.floor(pts[i].x / cell), Math.floor(pts[i].y / cell))
    const bucket = grid.get(k)
    if (bucket) bucket.push(i)
    else grid.set(k, [i])
  }
  const r2 = radius * radius
  const out = new Array<number>(pts.length).fill(0)
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i]
    const cx = Math.floor(p.x / cell)
    const cy = Math.floor(p.y / cell)
    let n = 0
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const bucket = grid.get(key(cx + dx, cy + dy))
        if (!bucket) continue
        for (const j of bucket) {
          const q = pts[j]
          const ddx = q.x - p.x
          const ddy = q.y - p.y
          if (ddx * ddx + ddy * ddy <= r2) n++
        }
      }
    }
    out[i] = n
  }
  return out
}
