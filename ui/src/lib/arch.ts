// Client-side architecture spec, derived from what the API already returns:
// predictor name + hyperparams + the dataset's feature counts. Every shipped
// predictor family renders natively in ArchViz (responsive, on-token, any
// depth); unknown predictors fall back to the predictor-exported viz.svg.

/* ---------------- ribbon (MLP / CNN) ---------------- */

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

export interface RibbonSpec {
  kind: 'ribbon'
  family: 'MLP' | 'CNN'
  layers: ArchLayer[]
  /** CNN: the meta features skip the conv stack and rejoin at this layer. */
  metaJoin?: { index: number; nMeta: number }
  shape: string
  totalParams: number
}

/* ---------------- tree ensembles ---------------- */

export interface ForestSpec {
  kind: 'forest'
  family: string
  /** boosted: sequential residual chain; bagged: parallel vote. */
  mode: 'boosted' | 'bagged'
  nTrees: number | null
  /** Drawn + reported depth; null = unbounded (lightgbm leaf-wise). */
  depth: number | null
  /** lightgbm: leaf cap instead of depth. */
  leaves: number | null
  lr: number | null
  subsample: number | null
  colsample: number | null
  inputs: number
  shape: string
}

/* ---------------- kernel machines (SVR / kernel ridge) ---------------- */

export interface KernelSpec {
  kind: 'kernel'
  family: string
  variant: 'svr' | 'krr'
  kernel: string
  /** SVR regularization. */
  c: number | null
  /** SVR tube half-width. */
  epsilon: number | null
  /** KRR regularization. */
  alpha: number | null
  gamma: number | null
  degree: number | null
  inputs: number
  shape: string
}

/* ---------------- linear (ridge) ---------------- */

export interface LinearSpec {
  kind: 'linear'
  family: string
  alpha: number | null
  autoAlpha: boolean
  loss: 'squared' | 'huber'
  huberEpsilon: number | null
  weightGamma: number | null
  inputs: number
  shape: string
}

/* ---------------- mixture of experts ---------------- */

export interface MoeSpec {
  kind: 'moe'
  family: string
  /** cluster: unsupervised GMM regimes; band: supervised target quantiles. */
  gate: 'cluster' | 'band'
  nExperts: number
  gateDetail: string
  expertDetail: string
  inputs: number
  shape: string
}

/* ---------------- median baseline ---------------- */

export interface MedianSpec {
  kind: 'median'
  family: string
  /** Grouping field (the dataset's first one-hot group), when known. */
  group: string | null
  shape: string
}

/* ---------------- blend (member fan-in) ---------------- */

export interface BlendVizMember {
  key: string
  /** Model / definition name, or the inline predictor name. */
  label: string
  predictor: string
  /** "model" | "definition" | "inline". */
  kind: string
  /** Normalized weight (mean rule); equal shares under median vote. */
  weight: number
  frozen: boolean
  /** Trained but excluded from the combine (graceful stop). */
  excluded: boolean
  excludeBlocks: string[]
  /** Solo test MAE when blend.json is available. */
  soloMae: number | null
  detail: string
}

export interface BlendSpec {
  kind: 'blend'
  family: string
  rule: 'mean' | 'median'
  weightFit: 'none' | 'grid'
  members: BlendVizMember[]
  shape: string
}

export type ArchSpec =
  | RibbonSpec
  | ForestSpec
  | KernelSpec
  | LinearSpec
  | MoeSpec
  | MedianSpec
  | BlendSpec

/** Feature counts of the dataset (run view: manifest; model view: contract). */
export interface ArchFeatures {
  nCols: number
  nPca: number
  /** One-hot group names, declaration order (median baseline group label). */
  onehotGroups?: string[]
}

const MLP_PREDICTORS = new Set(['burn-mlp', 'flux-mlp'])
const CNN_PREDICTORS = new Set(['burn-cnn', 'torch-cnn', 'flux-cnn'])

const posInts = (v: unknown): number[] | null =>
  Array.isArray(v) && v.length > 0 && v.every((n) => typeof n === 'number' && Number.isInteger(n) && n > 0)
    ? (v as number[])
    : null

const posInt = (v: unknown): number | null =>
  typeof v === 'number' && Number.isInteger(v) && v > 0 ? v : null

const posNum = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) && v > 0 ? v : null

const fraction = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) && v >= 0 && v < 1 ? v : null

const str = (v: unknown): string | null => (typeof v === 'string' && v ? v : null)

/** Null when the predictor family is unknown or the inputs don't add up;
 *  the caller then falls back to the predictor-exported image. */
export function deriveArch(
  predictor: string,
  hyperparams: Record<string, unknown> | null | undefined,
  features: ArchFeatures | null | undefined,
): ArchSpec | null {
  const hp = hyperparams ?? {}
  const f = features ?? { nCols: 0, nPca: 0 }
  if (MLP_PREDICTORS.has(predictor)) return mlpSpec(hp, f)
  if (CNN_PREDICTORS.has(predictor)) return cnnSpec(hp, f)
  switch (predictor) {
    case 'xgboost':
      return forestSpec('XGBoost', 'boosted', hp, f)
    case 'lightgbm':
      return forestSpec('LightGBM', 'boosted', hp, f)
    case 'random-forest':
      return forestSpec('Random forest', 'bagged', hp, f)
    case 'svm':
      return kernelSpec('SVR', 'svr', hp, f)
    case 'kernel-ridge':
      return kernelSpec('Kernel ridge', 'krr', hp, f)
    case 'ridge':
      return linearSpec(hp, f)
    case 'svm-moe':
      return moeSpec('SVM mixture', 'cluster', hp, f)
    case 'svm-quantile-moe':
    case 'svm-price-moe': // pre-rename instances keep the old registry name
      return moeSpec('SVM quantile mixture', 'band', hp, f)
    case 'baseline-median':
      return medianSpec(f)
    case 'blend':
      return blendSpec(hp)
    default:
      return null
  }
}

function mlpSpec(hp: Record<string, unknown>, f: ArchFeatures): RibbonSpec | null {
  const hidden = posInts(hp.hidden)
  const dropout = fraction(hp.dropout)
  if (!hidden || dropout === null || f.nCols <= 0) return null

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
    kind: 'ribbon',
    family: 'MLP',
    layers,
    shape: `${f.nCols} → ${hidden.join(' → ')} → 1`,
    totalParams: layers.reduce((s, l) => s + l.params, 0),
  }
}

function cnnSpec(hp: Record<string, unknown>, f: ArchFeatures): RibbonSpec | null {
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
    kind: 'ribbon',
    family: 'CNN',
    layers,
    metaJoin: { index: joinIndex, nMeta },
    shape: `${nPca} pca → [${channels.join(' → ')}] k${k} → ⊕${nMeta} meta → ${denseHidden} → 1`,
    totalParams: layers.reduce((s, l) => s + l.params, 0),
  }
}

function forestSpec(
  family: string,
  mode: 'boosted' | 'bagged',
  hp: Record<string, unknown>,
  f: ArchFeatures,
): ForestSpec {
  const nTrees = posInt(hp.n_estimators)
  const rawDepth = posInt(hp.max_depth) // 0 = unlimited in both libs
  const leaves = posInt(hp.num_leaves)
  // LightGBM grows leaf-wise: with no depth cap, the leaf budget implies one.
  const depth = rawDepth ?? (leaves ? Math.ceil(Math.log2(leaves)) : null)
  const lr = posNum(hp.learning_rate)
  const subsample = fraction(hp.subsample) ?? (posNum(hp.subsample) === 1 ? 1 : null)
  const colsample = fraction(hp.colsample_bytree) ?? (posNum(hp.colsample_bytree) === 1 ? 1 : null)

  const bits = [
    nTrees !== null ? `${nTrees} trees` : 'trees',
    depth !== null ? `depth ≤${depth}` : leaves !== null ? `${leaves} leaves` : null,
    lr !== null && mode === 'boosted' ? `η ${lr}` : null,
  ].filter(Boolean)
  return {
    kind: 'forest',
    family,
    mode,
    nTrees,
    depth,
    leaves,
    lr,
    subsample,
    colsample,
    inputs: f.nCols,
    shape: `${f.nCols} features → ${bits.join(' · ')} → ${mode === 'boosted' ? 'Σ residual fits' : 'mean vote'}`,
  }
}

function kernelSpec(
  family: string,
  variant: 'svr' | 'krr',
  hp: Record<string, unknown>,
  f: ArchFeatures,
): KernelSpec {
  const kernel = str(hp.kernel) ?? 'rbf'
  const c = posNum(hp.C)
  const epsilon = posNum(hp.epsilon)
  const alpha = posNum(hp.alpha)
  const gamma = posNum(hp.gamma) // 0 = scale heuristic
  const degree = kernel === 'poly' ? posInt(hp.degree) : null
  const bits = [
    `${kernel} kernel`,
    variant === 'svr' ? (c !== null ? `C ${c}` : null) : alpha !== null ? `α ${alpha}` : null,
    variant === 'svr' && epsilon !== null ? `ε ${epsilon}` : null,
    degree !== null ? `d${degree}` : null,
  ].filter(Boolean)
  return {
    kind: 'kernel',
    family,
    variant,
    kernel,
    c,
    epsilon,
    alpha,
    gamma,
    degree,
    inputs: f.nCols,
    shape: `${f.nCols} features → φ ${bits.join(' · ')} → fit`,
  }
}

function linearSpec(hp: Record<string, unknown>, f: ArchFeatures): LinearSpec {
  const autoAlpha = hp.auto_alpha === true
  const alpha = posNum(hp.alpha)
  const loss = hp.loss === 'huber' ? 'huber' : 'squared'
  const huberEpsilon = loss === 'huber' ? posNum(hp.huber_epsilon) : null
  const weightGamma = posNum(hp.weight_gamma)
  const bits = [
    autoAlpha ? 'α by CV' : alpha !== null ? `α ${alpha}` : null,
    loss === 'huber' ? `huber${huberEpsilon !== null ? ` ε ${huberEpsilon}` : ''}` : null,
    weightGamma !== null ? `target-weighted γ ${weightGamma}` : null,
  ].filter(Boolean)
  return {
    kind: 'linear',
    family: 'Ridge',
    alpha,
    autoAlpha,
    loss,
    huberEpsilon,
    weightGamma,
    inputs: f.nCols,
    shape: `${f.nCols} features → w·x + b${bits.length ? ` · ${bits.join(' · ')}` : ''}`,
  }
}

function moeSpec(
  family: string,
  gate: 'cluster' | 'band',
  hp: Record<string, unknown>,
  f: ArchFeatures,
): MoeSpec {
  const n = posInt(gate === 'cluster' ? hp.n_clusters : hp.n_bands) ?? 4
  const c = posNum(hp.C)
  const epsilon = posNum(hp.epsilon)
  const covariance = str(hp.covariance)
  const gateDetail =
    gate === 'cluster'
      ? `GMM soft-assigns rows to ${n} feature-space regimes${covariance ? ` (${covariance} covariance)` : ''}`
      : `logistic gate scores ${n} target-quantile bands`
  const expertDetail = `rbf SVR per ${gate === 'cluster' ? 'regime' : 'band'}${
    c !== null ? ` · C ${c}` : ''
  }${epsilon !== null ? ` · ε ${epsilon}` : ''}`
  return {
    kind: 'moe',
    family,
    gate,
    nExperts: n,
    gateDetail,
    expertDetail,
    inputs: f.nCols,
    shape: `${f.nCols} features → ${gate === 'cluster' ? `GMM gate` : `quantile gate`} → ${n} SVR experts → weighted blend`,
  }
}

function medianSpec(f: ArchFeatures): MedianSpec {
  const group = f.onehotGroups?.[0] ?? null
  return {
    kind: 'median',
    family: 'Median baseline',
    group,
    shape: group
      ? `group by ${group} → train-split median`
      : 'group by leading categorical → train-split median',
  }
}

/** Members straight from hyperparams (run view, or models whose blend.json
 *  predates the endpoint). BlendPanel upgrades these with blend.json. */
function blendSpec(hp: Record<string, unknown>): BlendSpec | null {
  const raw = Array.isArray(hp.members) ? hp.members : null
  if (!raw || raw.length === 0) return null
  const rule = hp.rule === 'median' ? 'median' : 'mean'
  const weightFit = hp.weight_fit === 'grid' ? 'grid' : 'none'
  const weights = raw.map((m) =>
    m && typeof m === 'object' && posNum((m as Record<string, unknown>).weight) !== null
      ? (m as Record<string, unknown>).weight as number
      : 1,
  )
  const sum = weights.reduce((a, b) => a + b, 0) || 1
  const members: BlendVizMember[] = raw.map((m, i) => {
    const o = (m && typeof m === 'object' ? m : {}) as Record<string, unknown>
    const model = str(o.model)
    const definition = str(o.definition)
    const predictor = str(o.predictor) ?? 'member'
    const kind = model ? 'model' : definition ? 'definition' : 'inline'
    const label = model ?? definition ?? predictor
    const excludeBlocks = Array.isArray(o.exclude_blocks)
      ? (o.exclude_blocks as unknown[]).filter((b): b is string => typeof b === 'string')
      : []
    return {
      key: `m-${i}`,
      label,
      predictor: model || definition ? '' : predictor,
      kind,
      weight: rule === 'median' ? 1 / raw.length : weights[i] / sum,
      frozen: kind === 'model',
      excluded: false,
      excludeBlocks,
      soloMae: null,
      detail: `${label} · ${kind === 'model' ? 'frozen model' : kind === 'definition' ? 'trained from definition' : 'trained inline'}${
        excludeBlocks.length ? ` · blocks −${excludeBlocks.join(', −')}` : ''
      }`,
    }
  })
  return {
    kind: 'blend',
    family: 'Blend',
    rule,
    weightFit,
    members,
    shape: `${members.length} members → ${rule === 'median' ? 'median vote' : weightFit === 'grid' ? 'grid-fitted weighted mean' : 'weighted mean'}`,
  }
}

/** Build a BlendSpec from a fetched blend.json (richer than hyperparams). */
export function blendSpecFromFile(file: {
  rule: 'mean' | 'median'
  weight_fit: 'none' | 'grid'
  weights: number[]
  members: {
    index: number
    kind: string
    predictor: string
    source?: string | null
    n_cols: number
    exclude_blocks?: string[]
    frozen: boolean
    included: boolean
    solo_metrics?: { mae: number } | null
  }[]
}): BlendSpec {
  const included = file.members.filter((m) => m.included)
  const wsum =
    included.reduce((s, m) => s + (file.weights[m.index] ?? 1), 0) || included.length || 1
  const members: BlendVizMember[] = file.members.map((m) => {
    const label = m.source ?? m.predictor
    const excludeBlocks = m.exclude_blocks ?? []
    return {
      key: `m-${m.index}`,
      label,
      predictor: m.source ? m.predictor : '',
      kind: m.kind,
      weight:
        !m.included ? 0
        : file.rule === 'median' ? 1 / (included.length || 1)
        : (file.weights[m.index] ?? 1) / wsum,
      frozen: m.frozen,
      excluded: !m.included,
      excludeBlocks,
      soloMae: m.solo_metrics?.mae ?? null,
      detail: `${label} · ${m.frozen ? 'frozen' : 'trained'} · ${m.n_cols} cols${
        excludeBlocks.length ? ` · blocks −${excludeBlocks.join(', −')}` : ''
      }${!m.included ? ' · excluded (stopped before training)' : ''}`,
    }
  })
  return {
    kind: 'blend',
    family: 'Blend',
    rule: file.rule,
    weightFit: file.weight_fit,
    members,
    shape: `${included.length} of ${file.members.length} members → ${
      file.rule === 'median' ? 'median vote' : file.weight_fit === 'grid' ? 'grid-fitted weighted mean' : 'weighted mean'
    }`,
  }
}
