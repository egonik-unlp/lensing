// TypeScript mirror of the serde-serialized pg_core::domain::Domain, served at
// GET /api/domain. The whole UI reads its nouns, target formatting, field set
// and quality labels from this once-at-boot config (see DomainContext).

export type FieldRole =
  | 'target'
  | 'categorical'
  | 'numeric'
  | 'coordinates'
  | 'filter_only'
  | 'timestamp'
  | 'display'

export interface DomainProject {
  name: string
  title: string
  entity_noun: string
  entity_noun_plural: string
  target_noun: string
}

export interface CorpusFilterClause {
  field: string
  equals?: unknown
}

export interface CorpusFilter {
  desc?: string | null
  must?: CorpusFilterClause[]
  must_not?: CorpusFilterClause[]
}

export interface DomainCorpus {
  qdrant_url: string
  collection: string
  manual_collection: string
  embedding_dim: number
  metadata_root: string
  content_field: string
  filter: CorpusFilter
}

export interface TargetFormat {
  /** "money" draws the symbol; "number" is plain grouped digits. */
  style: 'money' | 'number'
  symbol: string
  locale: string
}

export interface DomainTarget {
  field: string
  transform: 'log1p' | 'none'
  format: TargetFormat
}

/** `"all"` keeps every value; `{top_n}` keeps the N most frequent. */
export type VocabSpec = 'all' | { top_n: number }

export interface DomainField {
  name: string
  role: FieldRole
  encode?: string | null
  zero_is_missing?: boolean
  default_on?: boolean
  label?: string | null
  path?: 'metadata' | 'payload' | string
  vocab?: VocabSpec | null
  indicator?: boolean
  critical?: boolean
  reconcile?: boolean
  group?: string | null
  suggestions?: string[] | null
  required?: boolean
}

export interface DomainCoordinates {
  field: string
  lat_range: [number, number]
  lon_range: [number, number]
}

export interface DomainCurrency {
  keep: string
  currency_field: string
  date_field: string
  reconcile_collection: string
  rate_source: string
  rate_url_template: string
  pair: [string, string]
}

export interface DomainQuality {
  outlier_group: string
  capped_numeric: string
  rule_labels: Record<string, string>
}

export interface DomainMetrics {
  primary: string
  columns: string[]
  percent: string[]
  value_unit: string
}

export interface Domain {
  schema_version: number
  project: DomainProject
  corpus: DomainCorpus
  target: DomainTarget
  fields: DomainField[]
  coordinates: DomainCoordinates | null
  currency: DomainCurrency | null
  quality: DomainQuality
  metrics: DomainMetrics
}

export async function fetchDomain(): Promise<Domain> {
  const res = await fetch('/api/domain')
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = (await res.json()) as { error?: string }
      if (body.error) detail = body.error
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`GET /api/domain failed: ${detail}`)
  }
  return res.json() as Promise<Domain>
}

/* ---------------- field helpers ---------------- */

/** The value an `operation`-style filter `must` clause pins, if any. Lets new
 *  listings be created matching the corpus filter (e.g. operation=="sale"). */
export function pinnedFilterValues(domain: Domain): Record<string, string> {
  const out: Record<string, string> = {}
  for (const c of domain.corpus.filter.must ?? []) {
    if (c.equals != null && (typeof c.equals === 'string' || typeof c.equals === 'number')) {
      out[c.field] = String(c.equals)
    }
  }
  return out
}

export function fieldLabel(f: DomainField): string {
  return f.label ?? f.name
}

export function vocabTopN(f: DomainField): number | null {
  if (f.vocab && typeof f.vocab === 'object' && 'top_n' in f.vocab) return f.vocab.top_n
  return null
}
