import type { ModelSaeReport } from '../api/types'

/** The headline numbers of a per-model SAE — the answer, before the detail.
 *  Shared by the single report header and the comparison digest so they agree. */
export interface ModelSaeDigest {
  peakR2: number
  maxUtil: number
  concepts: number
  segRepr: string
  dropped: number | null
}

export function modelSaeDigest(r: ModelSaeReport): ModelSaeDigest {
  const layers = r.layers ?? []
  const deepest = layers[layers.length - 1]
  const cvd = r.concept_vs_decodability ?? []
  const depth = r.depth_linear_probe ?? []
  const peakR2 = Math.max(
    ...cvd.map((c) => c.linear_r2_log ?? -Infinity),
    ...depth.map((s) => s.test_r2_log),
    -Infinity,
  )
  const segs = deepest?.segments ?? []
  return {
    peakR2: Number.isFinite(peakR2) ? peakR2 : 0,
    maxUtil: layers.length ? Math.max(...layers.map((l) => l.capacity.utilization)) : 0,
    concepts: layers.reduce((s, l) => s + l.n_interpretable_concepts, 0),
    segRepr: segs.length ? `${segs.filter((s) => s.represented).length}/${segs.length}` : '—',
    dropped: r.embedding_diff?.ran ? (deepest?.dropped_vs_embedding?.n_dropped ?? null) : null,
  }
}
