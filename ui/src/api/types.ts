// Mirrors of the lensing-server API types.

export type RunStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'interrupted' | 'stopped'

export interface Metrics {
  mae: number
  rmse: number
  r2: number
  /** fraction: 0.25 = 25% */
  mape: number
  /** fraction */
  medape: number
  n_test: number
}

export interface RunMeta {
  run_id: string
  dataset_id: string
  predictor: string
  hyperparams: Record<string, unknown>
  status: RunStatus
  started_at: string
  finished_at: string | null
  exit_code: number | null
  stderr_tail: string | null
  metrics: Metrics | null
  contract_version: number
  has_checkpoint: boolean
  /** Model definition this run was launched from, if any. */
  from_definition?: string | null
}

export interface Prediction {
  row_id: number
  actual: number
  predicted: number
}

export type ProgressEvent =
  | { event: 'epoch'; epoch: number; total_epochs: number; train_loss: number; val_loss: number }
  | { event: 'log'; msg: string }
  /** Periodic checkpoint saved; the run is now promotable even if killed. */
  | { event: 'checkpoint'; epoch: number }
  /** The predictor saw the stop request and is finishing up. */
  | { event: 'stopping' }
  | { event: 'done' }
  | { event: 'status'; status: RunStatus }

export type ParamKind = 'int' | 'float' | 'bool' | 'ints' | 'enum' | 'json'

export interface Param {
  name: string
  label: string | null
  type: ParamKind
  default: unknown
  min: number | null
  max: number | null
  options: string[] | null
}

export interface Predictor {
  name: string
  display_name: string
  description: string
  /** Implementation language (e.g. "Rust", "Python", "Julia"). */
  language: string | null
  /** Library/framework the predictor is built on (e.g. "burn", "PyTorch"). */
  framework: string | null
  params: Param[]
  /** Present iff the predictor implements the predict subcommand. */
  predict_args: string[] | null
  /** Honors the graceful-stop protocol (otherwise stop = kill). */
  supports_stop: boolean
  /** Writes an architecture viz.svg at training start. */
  visualization: boolean
}

export interface ColumnDesc {
  name: string
  kind:
    | { type: 'pca'; component: number }
    | { type: 'numeric'; field: string }
    | { type: 'onehot'; group: string; value: string }
}

export interface Manifest {
  dataset_id: string
  /** Optional display name; the slug `dataset_id` stays the stable identity. */
  name?: string | null
  created_at: string
  source: { qdrant_url: string; collection: string; filter: string }
  n_rows: number
  n_cols: number
  columns: ColumnDesc[]
  pca: {
    dims: number
    mean: number[]
    components_shape: [number, number]
    explained_variance_ratio: number[]
  }
  target: { field: string; transform: 'log1p' | 'none' }
  split: { test_ratio: number; seed: number; n_train: number; n_test: number }
  feature_config: FeatureConfig
  quality?: QualityReport | null
  /** Cumulative explained-variance curve over the full spectrum (train split). */
  cumulative_evr?: number[]
  redundancy?: RedundancyReport | null
  /** Currency handling applied at build time. Absent on older datasets. */
  currency?: CurrencyReport | null
}

export interface FeatureConfig {
  pca_dims: number
  bedrooms: boolean
  property_type: boolean
  neighborhood_top_n: number
  city: boolean
  province: boolean
  cluster: boolean
  /** Generic domain-driven toggles: field/group name → enabled. */
  fields?: Record<string, boolean>
  /** Per-categorical vocabulary cap: field name → top-N. */
  vocab_top_n?: Record<string, number>
  /** Lat/lon bounds used at build time, when coordinates are on. */
  coordinate_bounds?: { lat_range: [number, number]; lon_range: [number, number] } | null
}

/** One corpus row, mirroring the collection's metadata schema. A record so
 *  views can read domain fields by name (GET /api/domain drives which); only
 *  the framework-owned keys are typed. */
export interface Item extends Record<string, unknown> {
  content: string
  cluster_label: string | null
}

export type Items = Record<string, Item>

/** Train/test membership as row ids, joinable with Items keys. */
export interface DatasetSplit {
  train: number[]
  test: number[]
}

export type BuildStatus =
  | { state: 'building'; stage: string }
  | { state: 'done'; dataset_id: string }
  | { state: 'failed'; error: string }

export interface BuildRequest {
  pca_dims: number
  test_ratio: number
  seed: number
  log_target: boolean
  bedrooms: boolean
  property_type: boolean
  neighborhood_top_n: number
  city: boolean
  province: boolean
  cluster: boolean
  /** Raw-collection numerics (areas, baths, rooms) reconciled by point id. */
  raw_numerics: boolean
  /** With raw_numerics: backfill missing areas from "… m²" in the text. */
  area_content_backfill: boolean
  /** Lat/lon from the raw collection (raw degrees + missing indicator). */
  coordinates: boolean
  /** With raw_numerics: impute missing numerics with train-split
   *  outlier-group medians and drop the missing-indicator columns. */
  impute_numerics: boolean
  quality: QualityFilterConfig
  currency: CurrencyConfig
  /** Source collection; omit to use the server's configured one. */
  collection?: string | null
  /** Generic domain-driven feature toggles: field/group name → enabled. When
   *  non-empty this is authoritative over the legacy flags above. */
  fields?: Record<string, boolean>
  /** Per-categorical vocabulary cap: field name → top-N. */
  vocab_top_n?: Record<string, number>
}

/* ---------------- latent representations ---------------- */

export type CompressionMethod = 'pca' | 'autoencoder' | 'sparse_ae'

/** A compressor's quality metric: PCA reports EVR, the AE per-block R². */
export type RepresentationQuality =
  | { kind: 'evr'; evr: number[]; cumulative_evr: number[]; captured: number }
  | { kind: 'block_r2'; blocks: [string, number][] }
  | null

export interface RepresentationMeta {
  id: string
  name: string
  method: CompressionMethod
  latent_dim: number
  source_collection: string
  sink_collection: string
  created_at: string
  n_points: number
  quality: RepresentationQuality
}

export interface BuildRepresentationRequest {
  name: string
  method: CompressionMethod
  latent: number
  sink_collection: string
  source_collection?: string | null
  epochs?: number | null
  hidden?: number[] | null
  sparse_weight?: number | null
  distance?: string
}

/** Status of a heavy async job (analyze, export). */
export type JobStatus =
  | { state: 'running'; stage: string }
  | { state: 'done'; result: unknown }
  | { state: 'failed'; error: string }

/* ---------------- feature redundancy ---------------- */

export interface RedundancyReport {
  near_zero_variance: { column: string; variance: number }[]
  onehot_groups: {
    group: string
    n_values: number
    other_fraction: number
    rare_buckets: number
  }[]
  correlated_pairs: { a: string; b: string; corr: number }[]
}

/** Result payload of a successful analyze job. */
export interface AnalyzeResult {
  /** Per-component explained variance ratio (full spectrum, train split). */
  evr: number[]
  cumulative_evr: number[]
  redundancy: RedundancyReport
  n_rows: number
  n_excluded: number
}

/** Result payload of a successful export job. */
export interface ExportResult {
  collection: string
  n_source: number
  n_excluded: number
  n_written: number
}

/* ---------------- quality filters ---------------- */

export interface QualityFilterConfig {
  nonpositive_price: boolean
  price_outlier: boolean
  price_outlier_mad_z: number
  missing_fields: boolean
  price_range: boolean
  price_min: number
  price_max: number
  bedrooms_outlier: boolean
  bedrooms_max: number
  duplicate_content: boolean
  short_content: boolean
  short_content_min_chars: number
}

export const DEFAULT_QUALITY: QualityFilterConfig = {
  nonpositive_price: true,
  price_outlier: false,
  price_outlier_mad_z: 3.5,
  missing_fields: false,
  price_range: false,
  price_min: 1000,
  price_max: 50_000_000,
  bedrooms_outlier: false,
  bedrooms_max: 15,
  duplicate_content: false,
  short_content: false,
  short_content_min_chars: 80,
}

export interface RuleStats {
  rule: string
  n_flagged: number
  n_excluded: number
}

export interface QualityReport {
  config: QualityFilterConfig
  rules: RuleStats[]
  n_excluded_total: number
}

/** Sample flagged row: row_id plus the domain's target, critical/grouping
 *  and currency fields, keyed by their domain field names. */
export interface PreflightSample extends Record<string, unknown> {
  row_id: number
}

export interface PreflightResponse {
  n_total: number
  n_excluded_total: number
  rules: RuleStats[]
  samples: Record<string, PreflightSample[]>
  currency: CurrencyReport
}

/* ---------------- currency handling ---------------- */

export type CurrencyMode = 'off' | 'filter' | 'convert'

export interface CurrencyConfig {
  mode: CurrencyMode
  /** The currency the target is expressed in. */
  keep: string
  /** Companion collection joined by point id for currency; null = inline only. */
  reconcile_collection: string | null
  /** Exchange-rate series for convert mode. */
  rate_source: 'blue' | 'oficial'
}

export const DEFAULT_CURRENCY: CurrencyConfig = {
  mode: 'filter',
  keep: 'USD',
  reconcile_collection: 'properties',
  rate_source: 'blue',
}

export interface CurrencyReport {
  config: CurrencyConfig
  n_foreign: number
  n_missing: number
  n_converted: number
  rate_min?: number | null
  rate_max?: number | null
}

/* ---------------- collection shape validation ---------------- */

/** POST /api/collections/validate — shape problems are data, always 200. */
export interface CollectionValidation {
  collection: string
  exists: boolean
  /** errors is empty — safe to build from this collection. */
  ok_to_build: boolean
  vector: { size: number; distance: string } | null
  count_filtered: number | null
  sample_size: number
  /** Domain field name (plus pseudo-keys like "content", "vector",
   *  `target>0`, a pinned filter value) → fraction (0..1) present in the sample. */
  coverage: Record<string, number> | null
  vector_dims_in_sample: number[]
  errors: string[]
  warnings: string[]
}

/* ---------------- model definitions ---------------- */

/** One `[[definitions]]` entry of models.toml: a named, git-versionable
 *  preset of predictor + concrete hyperparams. */
export interface ModelDefinition {
  name: string
  predictor: string
  /** Always merged over the predictor's schema defaults. */
  hyperparams: Record<string, unknown>
  /** Datasets this definition has been used with / is intended for. */
  dataset_tags: string[]
  notes?: string | null
  created_at: string
  updated_at?: string | null
}

/* ---------------- models ---------------- */

export interface ModelRecord {
  name: string
  run_id: string
  predictor: string
  dataset_id: string
  created_at: string
  notes?: string | null
}

export interface InputFields {
  required_numeric: string[]
  required_categorical: string[]
  embedding_dim: number
}

export interface ContractSummary {
  contract_version: number
  n_cols: number
  input_fields: InputFields
  target: { field: string; transform: 'log1p' | 'none' }
  feature_config: FeatureConfig
  pca_dims: number
}

export interface InferencePrediction {
  row_id: number
  predicted: number
}

export interface PredictResponse {
  predictions: InferencePrediction[]
  warnings: string[]
}

/* ---------------- blend record (blend.json) ---------------- */

/** One member of a blend, as recorded by the blend predictor. */
export interface BlendMemberRecord {
  index: number
  /** "model" | "definition" | "inline". */
  kind: string
  predictor: string
  /** Model or definition name; absent for inline members. */
  source?: string | null
  columns: string[]
  n_cols: number
  exclude_blocks?: string[]
  frozen: boolean
  /** False when a graceful stop landed before this member trained. */
  included: boolean
  /** Solo test metrics in target space (absent if excluded). */
  solo_metrics?: Metrics | null
}

/** `blend.json`: how the blend was assembled, served for blend runs/models. */
export interface BlendFile {
  contract_version: number
  rule: 'mean' | 'median'
  weight_fit: 'none' | 'grid'
  /** Final (post-grid, renormalized) weights, parallel to `members`. */
  weights: number[]
  members: BlendMemberRecord[]
}

/* ---------------- best-models group ---------------- */

export interface BestModelEntry {
  name: string
  rank: number
  metric: string
  metric_value: number
  run_id: string
  predictor: string
  dataset_id: string
  source: 'auto' | 'pinned'
  selected_at: string
}

export interface BestModelGroup {
  primary_metric: string
  size: number
  entries: BestModelEntry[]
  pinned: string[]
  excluded: string[]
  updated_at: string
}

export interface MemberPredictions {
  name: string
  rank: number
  predictions: InferencePrediction[]
}

export interface ConsensusPoint {
  row_id: number
  predicted: number
  n_models: number
}

export interface GroupPredictResponse {
  members: MemberPredictions[]
  consensus: ConsensusPoint[]
  warnings: string[]
}

/* ---------------- manual listings ---------------- */

export interface ListingCoordinates {
  lat?: number | null
  lon?: number | null
}

/** Corpus payload `metadata` keys, verbatim (camelCase as stored in Qdrant).
 *  Mirrors the domain field set; loosened to a record so generic views can
 *  read arbitrary domain fields while the well-known keys stay typed. */
/** Stored listing metadata: domain fields keyed by their domain names (read
 *  them via GET /api/domain), plus the framework-owned display extras. */
export interface ListingMetadata extends Record<string, unknown> {
  coordinates?: ListingCoordinates | null
  /** Photo URLs from the source page; display-only, never a model input. */
  images?: string[] | null
  /** Original page the listing was captured from; display-only. */
  sourceUrl?: string | null
}

export interface ListingSummary {
  id: number
  content: string
  metadata: ListingMetadata
}

/** Single-listing GET carries the embedding so the UI can predict inline. */
export interface Listing extends ListingSummary {
  embedding: number[]
}

/** Create/update body: content is required; metadata fields are FLAT by
 *  domain field name and all optional (set via the index signature). Only
 *  the framework-owned keys stay typed. */
export interface CreateListingRequest extends Record<string, unknown> {
  content: string
  coordinates?: ListingCoordinates
  images?: string[]
  sourceUrl?: string
}
