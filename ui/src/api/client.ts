import type {
  BestModelGroup,
  BlendFile,
  BuildRequest,
  BuildStatus,
  CollectionValidation,
  ContractSummary,
  CreateListingRequest,
  CurrencyConfig,
  DatasetSplit,
  GroupPredictResponse,
  Items,
  JobStatus,
  Listing,
  ListingSummary,
  Manifest,
  ModelDefinition,
  ModelRecord,
  PredictResponse,
  Prediction,
  Predictor,
  PreflightResponse,
  QualityFilterConfig,
  RunMeta,
} from './types'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = (await res.json()) as { error?: string }
      if (body.error) detail = body.error
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<T>
}

const post = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify(body),
})

const patch = (body: unknown): RequestInit => ({
  method: 'PATCH',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify(body),
})

const put = (body: unknown): RequestInit => ({
  method: 'PUT',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify(body),
})

export const api = {
  listPredictors: () =>
    request<{ predictors: Predictor[] }>('/api/predictors').then((r) => r.predictors),
  listDatasets: () =>
    request<{ datasets: Manifest[] }>('/api/datasets').then((r) => r.datasets),
  getDataset: (id: string) => request<Manifest>(`/api/datasets/${id}`),
  getItems: (id: string) => request<Items>(`/api/datasets/${id}/items`),
  getDatasetSplit: (id: string) => request<DatasetSplit>(`/api/datasets/${id}/split`),
  buildDataset: (req: BuildRequest) =>
    request<{ build_id: string }>('/api/datasets', post(req)).then((r) => r.build_id),
  getBuild: (id: string) => request<BuildStatus>(`/api/builds/${id}`),
  getJob: (id: string) => request<JobStatus>(`/api/jobs/${id}`),
  renameDataset: (id: string, name: string) =>
    request<Manifest>(`/api/datasets/${id}/rename`, post({ name })),
  analyzeDataset: (req: BuildRequest) =>
    request<{ job_id: string }>('/api/datasets/analyze', post(req)).then((r) => r.job_id),
  listCollections: () =>
    request<{ collections: string[]; source: string }>('/api/collections'),
  validateCollection: (collection?: string) =>
    request<CollectionValidation>('/api/collections/validate', post({ collection })),
  exportCollection: (req: {
    name_suffix: string
    quality: QualityFilterConfig
    currency?: CurrencyConfig
    source?: string
  }) => request<{ job_id: string }>('/api/collections/export', post(req)).then((r) => r.job_id),
  listRuns: () => request<{ runs: RunMeta[] }>('/api/runs').then((r) => r.runs),
  getRun: (id: string) => request<RunMeta>(`/api/runs/${id}`),
  getPredictions: (id: string) => request<Prediction[]>(`/api/runs/${id}/predictions`),
  startRun: (req: {
    dataset_id: string
    predictor?: string
    definition?: string
    hyperparams?: Record<string, unknown>
  }) => request<{ run_id: string }>('/api/runs', post(req)).then((r) => r.run_id),
  eventsUrl: (runId: string) => `/api/runs/${runId}/events`,
  stopRun: (id: string, force = false) =>
    request<{ ok: boolean; mode: 'graceful' | 'force' }>(`/api/runs/${id}/stop`, post({ force })),
  deleteRun: (id: string) =>
    request<{ ok: boolean }>(`/api/runs/${id}`, { method: 'DELETE' }),
  runVizUrl: (runId: string) => `/api/runs/${runId}/viz`,
  modelVizUrl: (name: string) => `/api/models/${name}/viz`,
  runBlend: (runId: string) => request<BlendFile>(`/api/runs/${runId}/blend`),
  modelBlend: (name: string) => request<BlendFile>(`/api/models/${name}/blend`),
  preflight: (quality: QualityFilterConfig, currency: CurrencyConfig, sample = 8, collection?: string) =>
    request<PreflightResponse>('/api/datasets/preflight', post({ quality, currency, sample, collection })),
  listModels: () =>
    request<{ models: ModelRecord[] }>('/api/models').then((r) => r.models),
  promoteModel: (name: string, run_id: string, notes?: string) =>
    request<ModelRecord>('/api/models', post({ name, run_id, notes: notes || undefined })),
  getModel: (name: string) =>
    request<{
      record: ModelRecord
      contract: ContractSummary
      hyperparams: Record<string, unknown> | null
    }>(`/api/models/${name}`),
  deleteModel: (name: string) =>
    request<{ ok: boolean }>(`/api/models/${name}`, { method: 'DELETE' }),
  renameModel: (name: string, new_name: string) =>
    request<ModelRecord>(`/api/models/${name}/rename`, post({ new_name })),
  predictModel: (name: string, body: { items?: unknown[]; point_ids?: number[] }) =>
    request<PredictResponse>(`/api/models/${name}/predict`, post(body)),
  bestModels: () => request<BestModelGroup>('/api/best-models'),
  recomputeBestModels: () =>
    request<BestModelGroup>('/api/best-models/recompute', { method: 'POST' }),
  curateBestModels: (req: {
    pin?: string[]
    exclude?: string[]
    unpin?: string[]
    unexclude?: string[]
  }) => request<BestModelGroup>('/api/best-models', put(req)),
  predictBestModels: (body: { items?: unknown[]; point_ids?: number[] }) =>
    request<GroupPredictResponse>('/api/best-models/predict', post(body)),
  listListings: () =>
    request<{ listings: ListingSummary[] }>('/api/listings').then((r) => r.listings),
  getListing: (id: number) => request<Listing>(`/api/listings/${id}`),
  createListing: (req: CreateListingRequest) =>
    request<ListingSummary>('/api/listings', post(req)),
  updateListing: (id: number, req: CreateListingRequest) =>
    request<ListingSummary>(`/api/listings/${id}`, put(req)),
  deleteListing: (id: number) =>
    request<{ ok: boolean }>(`/api/listings/${id}`, { method: 'DELETE' }),
  listDefinitions: () =>
    request<{ definitions: ModelDefinition[] }>('/api/definitions').then((r) => r.definitions),
  getDefinition: (name: string) => request<ModelDefinition>(`/api/definitions/${name}`),
  createDefinition: (req: {
    name: string
    predictor: string
    hyperparams?: Record<string, unknown>
    dataset_tags?: string[]
    notes?: string
  }) => request<ModelDefinition>('/api/definitions', post(req)),
  updateDefinition: (
    name: string,
    req: { hyperparams?: Record<string, unknown>; dataset_tags?: string[]; notes?: string },
  ) => request<ModelDefinition>(`/api/definitions/${name}`, patch(req)),
  renameDefinition: (name: string, new_name: string) =>
    request<ModelDefinition>(`/api/definitions/${name}/rename`, post({ new_name })),
  cloneDefinition: (name: string, new_name: string) =>
    request<ModelDefinition>(`/api/definitions/${name}/clone`, post({ new_name })),
  deleteDefinition: (name: string) =>
    request<{ ok: boolean }>(`/api/definitions/${name}`, { method: 'DELETE' }),
}

/**
 * Poll a heavy async job (analyze, export) until terminal. Mirrors the build
 * polling cadence (700 ms). Returns a cancel function.
 */
export function pollJob(
  id: string,
  cbs: {
    onStage?: (stage: string) => void
    onDone: (result: unknown) => void
    onError: (message: string) => void
  },
): () => void {
  let timer: number | undefined
  let cancelled = false
  const tick = () => {
    api.getJob(id).then(
      (st) => {
        if (cancelled) return
        if (st.state === 'running') {
          cbs.onStage?.(st.stage)
          timer = window.setTimeout(tick, 700)
        } else if (st.state === 'done') {
          cbs.onDone(st.result)
        } else {
          cbs.onError(st.error)
        }
      },
      (e: unknown) => {
        if (cancelled) return
        cbs.onError(e instanceof Error ? e.message : String(e))
      },
    )
  }
  tick()
  return () => {
    cancelled = true
    if (timer) window.clearTimeout(timer)
  }
}
