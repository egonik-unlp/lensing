import type { CompressionMethod, RepresentationQuality } from '../api/types'
import { fmtPct } from './format'

export const methodLabel: Record<CompressionMethod, string> = {
  pca: 'PCA',
  autoencoder: 'autoencoder',
  sparse_ae: 'sparse-AE',
}

/** One-line quality summary: EVR captured for PCA, mean block-R² for the AE. */
export function qualitySummary(q: RepresentationQuality): string {
  if (!q) return '—'
  if (q.kind === 'evr') return `EVR ${fmtPct(q.captured, 0)}`
  if (q.blocks.length === 0) return '—'
  const mean = q.blocks.reduce((a, [, v]) => a + v, 0) / q.blocks.length
  return `block-R² ${mean.toFixed(2)}`
}
