// Predicting manual listings: shared between the listings index (auto
// median across the best-model group) and the listing detail predict panel.

import { api } from '../api/client'
import type { Listing, ModelRecord } from '../api/types'
import type { Domain } from './domain'

/** Display-only extras stored on a listing that are never a model input,
 *  independent of the domain's field roles. */
const DISPLAY_EXTRAS = ['images', 'sourceUrl']

/**
 * The inline-items payload for one listing. The captured target value is the
 * ground truth we compare predictions against — it must NEVER reach a
 * predictor, so it is stripped here along with every role:"display" field and
 * the display-only extras. This is the only place predict items are built;
 * keep it that way.
 */
export function toPredictItem(listing: Listing, domain: Domain): Record<string, unknown> {
  const metadata: Record<string, unknown> = { ...listing.metadata }
  delete metadata[domain.target.field]
  for (const f of domain.fields) if (f.role === 'display') delete metadata[f.name]
  for (const k of DISPLAY_EXTRAS) delete metadata[k]
  return { id: listing.id, content: listing.content, embedding: listing.embedding, ...metadata }
}

/** Legacy heuristic group, kept as the fallback when the server-side
 *  best-models group is empty (nothing recomputed yet): notes marked
 *  `Group: best-models`, then name matching, then everything. */
export function bestModels(models: ModelRecord[]): ModelRecord[] {
  const curated = models.filter((m) => m.notes?.includes('Group: best-models'))
  if (curated.length > 0) return curated
  const named = models.filter((m) => m.name.includes('best'))
  return named.length > 0 ? named : models
}

/** The model set consensus predictions run against. `viaServer` selects the
 *  single-call `/api/best-models/predict` fan-out (server-maintained group)
 *  over the legacy client-side per-model fan-out (heuristic fallback). */
export interface PredictGroup {
  names: string[]
  viaServer: boolean
}

/** Resolve the prediction group: the server-maintained best-models group
 *  when populated, else the legacy heuristic over all promoted models. */
export async function resolvePredictGroup(models: ModelRecord[]): Promise<PredictGroup> {
  try {
    const group = await api.bestModels()
    if (group.entries.length > 0) {
      return { names: group.entries.map((e) => e.name), viaServer: true }
    }
  } catch {
    /* endpoint unavailable: fall through to the heuristic */
  }
  return { names: bestModels(models).map((m) => m.name), viaServer: false }
}

export function median(values: number[]): number | null {
  if (values.length === 0) return null
  const v = [...values].sort((a, b) => a - b)
  const mid = v.length >> 1
  return v.length % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2
}

/* ---------------- index-view consensus cache ---------------- */

export interface ListingConsensus {
  /** Median predicted value across the models that answered. */
  median: number
  /** How many models answered / were asked. */
  n_ok: number
  n_models: number
}

const CACHE_KEY = 'lensing:listing-consensus:v1'

interface CacheShape {
  /** Sorted model names the cached numbers were computed with. */
  models: string[]
  byId: Record<string, ListingConsensus>
}

function readCache(modelNames: string[]): CacheShape {
  const fresh: CacheShape = { models: modelNames, byId: {} }
  try {
    const raw = sessionStorage.getItem(CACHE_KEY)
    if (!raw) return fresh
    const parsed = JSON.parse(raw) as CacheShape
    // A different promoted-model set invalidates every cached consensus.
    if (JSON.stringify(parsed.models) !== JSON.stringify(modelNames)) return fresh
    return parsed
  } catch {
    return fresh
  }
}

function writeCache(cache: CacheShape) {
  try {
    sessionStorage.setItem(CACHE_KEY, JSON.stringify(cache))
  } catch {
    /* storage full/unavailable: predictions simply recompute next visit */
  }
}

/**
 * Consensus (median of the best-model group) for one listing, cached in
 * sessionStorage so revisiting the index doesn't respawn predictors.
 * Fetches the stored listing (the embedding rides along), then either makes
 * one `/api/best-models/predict` call (server-side fan-out + median) or, on
 * the heuristic fallback, fans out one inline-items predict call per model;
 * either way the server's run-slot semaphore paces the predictor processes.
 */
export async function listingConsensus(
  id: number,
  group: PredictGroup,
  domain: Domain,
): Promise<ListingConsensus> {
  const names = [...group.names].sort()
  const cache = readCache(names)
  const hit = cache.byId[String(id)]
  if (hit) return hit

  const listing = await api.getListing(id)
  const item = toPredictItem(listing, domain)
  let consensus: ListingConsensus
  if (group.viaServer) {
    const r = await api.predictBestModels({ items: [item] })
    const point = r.consensus[0]
    if (!point) throw new Error('no model returned a prediction')
    consensus = { median: point.predicted, n_ok: point.n_models, n_models: group.names.length }
  } else {
    const settled = await Promise.allSettled(
      group.names.map((name) => api.predictModel(name, { items: [item] })),
    )
    const values = settled.flatMap((r) =>
      r.status === 'fulfilled' && r.value.predictions[0]
        ? [r.value.predictions[0].predicted]
        : [],
    )
    const med = median(values)
    if (med === null) throw new Error('no model returned a prediction')
    consensus = { median: med, n_ok: values.length, n_models: group.names.length }
  }

  // Re-read before writing: parallel rows may have written in the meantime.
  const latest = readCache(names)
  latest.byId[String(id)] = consensus
  writeCache(latest)
  return consensus
}

/** Drop one listing's cached consensus (after edit/delete). */
export function invalidateConsensus(id: number) {
  try {
    const raw = sessionStorage.getItem(CACHE_KEY)
    if (!raw) return
    const parsed = JSON.parse(raw) as CacheShape
    delete parsed.byId[String(id)]
    sessionStorage.setItem(CACHE_KEY, JSON.stringify(parsed))
  } catch {
    /* ignore */
  }
}
