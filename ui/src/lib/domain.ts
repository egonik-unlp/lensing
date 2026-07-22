// TypeScript mirror of the serde-serialized lensing_core::domain::Domain, served at
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

/** The learning task, mirroring lensing_core::domain::Task. Drives which
 *  metrics and result charts the UI renders. Absent ⇒ regression (back-compat
 *  for domains served before the field existed). */
export type Task = 'regression' | 'binary' | 'multiclass' | 'time_series'

export interface DomainTarget {
  field: string
  transform: 'log1p' | 'none'
  format: TargetFormat
  /** Absent ⇒ 'regression'. */
  task?: Task
  /** Ordered class labels for a multiclass target (index = class id). */
  classes?: string[] | null
}

/** The domain's task, defaulting to regression when the server omits it. */
export function domainTask(domain: Domain): Task {
  return domain.target.task ?? 'regression'
}

/** Whether the task predicts a discrete class (binary or multiclass). */
export function isClassification(domain: Domain): boolean {
  const t = domainTask(domain)
  return t === 'binary' || t === 'multiclass'
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
  /** Grouping field for the target-outlier MAD rule; may be absent. */
  outlier_group?: string | null
  /** Numeric field checked by the cap rule; absent disables it. */
  capped_numeric?: string | null
  rule_labels: Record<string, string>
}

export interface DomainMetrics {
  primary: string
  columns: string[]
  percent: string[]
  value_unit: string
  /** Per-metric ranking direction override (name → lower-is-better). */
  directions?: Record<string, boolean>
  best_models_size?: number | null
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

/** Categorical fields ordered for display: the outlier-group field leads,
 *  then critical fields, then declaration order (stable sort). */
export function categoricalFields(domain: Domain): DomainField[] {
  const cats = domain.fields.filter((f) => f.role === 'categorical')
  const score = (f: DomainField) =>
    f.name === domain.quality.outlier_group ? 0 : f.critical ? 1 : 2
  return [...cats].sort((a, b) => score(a) - score(b))
}

/** The capped-numeric field bound to the cap quality rule, if the domain has one. */
export function cappedNumericField(domain: Domain): DomainField | null {
  const name = domain.quality.capped_numeric
  if (!name) return null
  return domain.fields.find((f) => f.name === name) ?? null
}

/** The domain's timestamp field name, if any. */
export function timestampFieldName(domain: Domain): string | null {
  return domain.fields.find((f) => f.role === 'timestamp')?.name ?? null
}

/** A record's target value (e.g. a stored entry's captured ground truth), null
 *  when absent or non-numeric. Any finite value is valid — including 0 and
 *  negatives (a class id of 0, a signed regression target); the old
 *  positive-only guard was a price assumption. */
export function targetValue(domain: Domain, record: Record<string, unknown>): number | null {
  const v = record[domain.target.field]
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

/** A record's currency code, read via the domain's currency field. */
export function currencyValue(domain: Domain, record: Record<string, unknown>): string | null {
  const f = domain.currency?.currency_field
  const v = f ? record[f] : null
  return typeof v === 'string' && v ? v : null
}
