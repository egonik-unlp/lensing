// Client-side architecture spec, derived from what the API already returns:
// predictor name + hyperparams + the dataset's feature counts. Known families
// render natively in ArchViz (responsive, on-token, any depth); unknown
// predictors fall back to the predictor-exported viz.svg image.

export interface ArchLayer {
  key: string
  /** Short name: input, conv 1, hidden 2, pool, output. */
  name: string
  /** Display size, mono: "256", "64×25". */
  units: string
  /** Activation count — drives the ribbon height. */
  magnitude: number
  /** Hover/focus readout: operation · activation · regularization. */
  detail: string
  /** Learnable parameter count (0 for input / pooling). */
  params: number
}

export interface ArchSpec {
  family: 'MLP' | 'CNN'
  layers: ArchLayer[]
  /** CNN: the meta features skip the conv stack and rejoin at this layer. */
  metaJoin?: { index: number; nMeta: number }
  /** One-line shape summary for the resting readout. */
  shape: string
  totalParams: number
}

/** Feature counts of the dataset (run view: manifest; model view: contract). */
export interface ArchFeatures {
  nCols: number
  nPca: number
}

const MLP_PREDICTORS = new Set(['burn-mlp', 'flux-mlp'])
const CNN_PREDICTORS = new Set(['burn-cnn', 'torch-cnn', 'flux-cnn'])

const posInts = (v: unknown): number[] | null =>
  Array.isArray(v) && v.length > 0 && v.every((n) => typeof n === 'number' && Number.isInteger(n) && n > 0)
    ? (v as number[])
    : null

const posInt = (v: unknown): number | null =>
  typeof v === 'number' && Number.isInteger(v) && v > 0 ? v : null

const fraction = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) && v >= 0 && v < 1 ? v : null

/** Null when the predictor family is unknown or the inputs don't add up;
 *  the caller then falls back to the predictor-exported image. */
export function deriveArch(
  predictor: string,
  hyperparams: Record<string, unknown> | null | undefined,
  features: ArchFeatures | null | undefined,
): ArchSpec | null {
  if (!hyperparams || !features || features.nCols <= 0) return null
  if (MLP_PREDICTORS.has(predictor)) return mlpSpec(hyperparams, features)
  if (CNN_PREDICTORS.has(predictor)) return cnnSpec(hyperparams, features)
  return null
}

function mlpSpec(hp: Record<string, unknown>, f: ArchFeatures): ArchSpec | null {
  const hidden = posInts(hp.hidden)
  const dropout = fraction(hp.dropout)
  if (!hidden || dropout === null) return null

  const layers: ArchLayer[] = [
    {
      key: 'input',
      name: 'input',
      units: String(f.nCols),
      magnitude: f.nCols,
      detail: `${f.nCols} features`,
      params: 0,
    },
  ]
  let prev = f.nCols
  hidden.forEach((h, i) => {
    layers.push({
      key: `hidden-${i}`,
      name: `hidden ${i + 1}`,
      units: String(h),
      magnitude: h,
      detail: `Dense ${prev}→${h} · ReLU · dropout ${dropout}`,
      params: prev * h + h,
    })
    prev = h
  })
  layers.push({
    key: 'output',
    name: 'output',
    units: '1',
    magnitude: 1,
    detail: `Dense ${prev}→1 · linear`,
    params: prev + 1,
  })

  return {
    family: 'MLP',
    layers,
    shape: `${f.nCols} → ${hidden.join(' → ')} → 1`,
    totalParams: layers.reduce((s, l) => s + l.params, 0),
  }
}

function cnnSpec(hp: Record<string, unknown>, f: ArchFeatures): ArchSpec | null {
  const channels = posInts(hp.channels)
  const k = posInt(hp.kernel_size)
  const denseHidden = posInt(hp.dense_hidden)
  const dropout = fraction(hp.dropout)
  const nPca = f.nPca
  const nMeta = f.nCols - f.nPca
  if (!channels || k === null || denseHidden === null || dropout === null) return null
  if (nPca <= 0 || nMeta < 0) return null

  const layers: ArchLayer[] = [
    {
      key: 'input',
      name: 'input',
      units: String(f.nCols),
      magnitude: f.nCols,
      detail: `${nPca} pca + ${nMeta} meta · conv sees pca only`,
      params: 0,
    },
  ]
  // Mirrors the forward pass: pool halves the signal, skipped at length < 2.
  let prevCh = 1
  let len = nPca
  channels.forEach((ch, i) => {
    if (len >= 2) len = Math.floor(len / 2)
    layers.push({
      key: `conv-${i}`,
      name: `conv ${i + 1}`,
      units: `${ch}×${len}`,
      magnitude: ch * len,
      detail: `Conv1d ${prevCh}→${ch} k${k} · ReLU · pool/2 · dropout ${dropout}`,
      params: prevCh * ch * k + ch,
    })
    prevCh = ch
  })
  const joinIndex = layers.length
  layers.push({
    key: 'pool',
    name: 'pool ⊕ meta',
    units: String(prevCh + nMeta),
    magnitude: prevCh + nMeta,
    detail: `global avg → ${prevCh} · ⊕ ${nMeta} meta`,
    params: 0,
  })
  layers.push({
    key: 'dense',
    name: 'dense',
    units: String(denseHidden),
    magnitude: denseHidden,
    detail: `Dense ${prevCh + nMeta}→${denseHidden} · ReLU · dropout ${dropout}`,
    params: (prevCh + nMeta) * denseHidden + denseHidden,
  })
  layers.push({
    key: 'output',
    name: 'output',
    units: '1',
    magnitude: 1,
    detail: `Dense ${denseHidden}→1 · linear`,
    params: denseHidden + 1,
  })

  return {
    family: 'CNN',
    layers,
    metaJoin: { index: joinIndex, nMeta },
    shape: `${nPca} pca → [${channels.join(' → ')}] k${k} → ⊕${nMeta} meta → ${denseHidden} → 1`,
    totalParams: layers.reduce((s, l) => s + l.params, 0),
  }
}
