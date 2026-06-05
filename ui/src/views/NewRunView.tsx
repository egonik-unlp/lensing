import { useEffect, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import type { Manifest } from '../api/types'
import { DefinitionRef } from '../components/EntityRef'
import HyperparamForm from '../components/HyperparamForm'
import SubmitRow from '../components/SubmitRow'
import { scrollToFirstError } from '../lib/scrollToError'
import ViewHeader from '../components/ViewHeader'
import {
  hpStateFromValues,
  hpToJson,
  initialHpState,
  validateHp,
  type HpState,
} from '../lib/hyperparams'
import { useAsync } from '../hooks/useAsync'
import { fmtDateTime, fmtPct, shortDatasetId } from '../lib/format'
import './newrun.css'

export default function NewRunView() {
  useEffect(() => {
    document.title = 'New run · Price Guesser Models'
  }, [])
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const defName = searchParams.get('definition')
  const datasetParam = searchParams.get('dataset')
  const datasets = useAsync(() => api.listDatasets(), [])
  const predictors = useAsync(() => api.listPredictors(), [])
  const definition = useAsync(
    () => (defName ? api.getDefinition(defName) : Promise.resolve(null)),
    [defName],
  )

  const [datasetId, setDatasetId] = useState<string | null>(null)
  const [predictorChoice, setPredictorChoice] = useState<string | null>(null)
  // Per-predictor form edits; the visible state derives from these + the
  // seed (definition or schema defaults), so switching predictors never
  // needs an effect to reset the form.
  const [hpEdits, setHpEdits] = useState<Record<string, HpState>>({})
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)

  // Defaults are derived, not effect-synced: explicit choices win, then a
  // ?dataset= deep link (when it exists), else the first item once data arrives.
  const effectiveDatasetId =
    datasetId ??
    (datasetParam && datasets.data?.some((d) => d.dataset_id === datasetParam)
      ? datasetParam
      : null) ??
    datasets.data?.[0]?.dataset_id ??
    null
  // Predictor default: a definition in the URL wins over "first in list".
  const defaultPredictor = defName
    ? (definition.data?.predictor ?? (definition.error ? predictors.data?.[0]?.name : undefined))
    : predictors.data?.[0]?.name
  const predictorName = predictorChoice ?? defaultPredictor ?? null
  const predictor = predictors.data?.find((p) => p.name === predictorName) ?? null
  // The definition still applies while the user stays on its predictor.
  const fromDefinition = !!definition.data && predictor?.name === definition.data.predictor
  const hp: HpState = predictor
    ? (hpEdits[predictor.name] ??
      (fromDefinition
        ? hpStateFromValues(predictor.params, definition.data!.hyperparams)
        : initialHpState(predictor.params)))
    : { values: {}, errors: {} }

  const hpValid = predictor ? Object.keys(validateHp(predictor.params, hp.values).errors).length === 0 : false
  const canStart = !!effectiveDatasetId && !!predictor && hpValid && !submitting

  const blockReason = !effectiveDatasetId
    ? 'Select or build a dataset first'
    : !predictor
      ? 'Select a predictor'
      : !hpValid
        ? 'Fix the highlighted hyperparameters'
        : null

  const start = async () => {
    if (!canStart || !predictor || !effectiveDatasetId) return
    setSubmitting(true)
    setSubmitError(null)
    try {
      const runId = await api.startRun({
        dataset_id: effectiveDatasetId,
        predictor: predictor.name,
        // Sent so the server records provenance + tags the dataset on the
        // definition; only meaningful while its predictor is selected.
        definition: fromDefinition ? definition.data!.name : undefined,
        hyperparams: hpToJson(predictor.params, hp.values),
      })
      navigate(`/runs/${runId}`)
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e))
      setSubmitting(false)
    }
  }

  return (
    <section aria-label="Set up a training run">
      <ViewHeader
        title="New run"
        meta={<span>Pick a dataset, pick a predictor, tune its hyperparameters, train.</span>}
      />
      <div className="view-body newrun">
      {defName && definition.data && (
        <p className="muted" role="status">
          Starting from definition <DefinitionRef name={definition.data.name} />
          {fromDefinition
            ? ' — its hyperparameters are pre-filled below.'
            : ' — switched predictor, so its hyperparameters no longer apply.'}
        </p>
      )}
      {defName && definition.error && (
        <p className="hp-error" role="alert">
          Definition “{defName}” could not be loaded: {definition.error}
        </p>
      )}

      <DatasetPanel
        datasets={datasets.data}
        loading={datasets.loading}
        error={datasets.error}
        retry={datasets.reload}
        selected={effectiveDatasetId}
        onSelect={setDatasetId}
      />

      <div className="panel">
        <h2>2 · Predictor</h2>
        {predictors.error ? (
          <div className="error-block" role="alert">
            Could not load predictors: {predictors.error}{' '}
            <button className="btn" onClick={predictors.reload}>
              Retry
            </button>
          </div>
        ) : (
          <>
            <div className="predictor-row">
              <label htmlFor="predictor-select">Predictor</label>
              <select
                id="predictor-select"
                value={predictorName ?? ''}
                onChange={(e) => setPredictorChoice(e.target.value)}
                disabled={!predictors.data}
              >
                {/* display_name already names the framework; the language
                    is the missing bit. */}
                {(predictors.data ?? []).map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.display_name}
                    {p.language ? ` — ${p.language}` : ''}
                  </option>
                ))}
              </select>
              {predictor && <span className="muted predictor-desc">{predictor.description}</span>}
            </div>
            {predictor && (
              <HyperparamForm
                params={predictor.params}
                state={{ values: hp.values, errors: validateHp(predictor.params, hp.values).errors }}
                onChange={(name, value) =>
                  setHpEdits((m) => ({
                    ...m,
                    [predictor.name]: { ...hp, values: { ...hp.values, [name]: value } },
                  }))
                }
                disabled={submitting}
              />
            )}
          </>
        )}
      </div>

      {/* Stays clickable on hp errors: the click scrolls to the first one. */}
      <SubmitRow
        label="Start training"
        busyLabel="Starting…"
        busy={submitting}
        disabled={!effectiveDatasetId || !predictor}
        blockReason={blockReason}
        error={submitError}
        onSubmit={() => {
          if (!hpValid) {
            scrollToFirstError()
            return
          }
          void start()
        }}
      />
      </div>
    </section>
  )
}

/* ---------------- dataset panel ---------------- */

function DatasetPanel({
  datasets,
  loading,
  error,
  retry,
  selected,
  onSelect,
}: {
  datasets: Manifest[] | null
  loading: boolean
  error: string | null
  retry: () => void
  selected: string | null
  onSelect: (id: string) => void
}) {
  const empty = !loading && !error && (datasets?.length ?? 0) === 0

  return (
    <div className="panel">
      <h2>1 · Dataset</h2>
      {error && (
        <div className="error-block" role="alert">
          Could not load datasets: {error}{' '}
          <button className="btn" onClick={retry}>
            Retry
          </button>
        </div>
      )}
      {loading && <div className="skeleton" style={{ height: '3.5rem' }} />}
      {datasets && datasets.length > 0 && (
        <div className="dataset-list" role="radiogroup" aria-label="Choose a dataset">
          {datasets.map((d) => (
            <label key={d.dataset_id} className={`dataset-option${selected === d.dataset_id ? ' is-selected' : ''}`}>
              <input
                type="radio"
                name="dataset"
                checked={selected === d.dataset_id}
                onChange={() => onSelect(d.dataset_id)}
              />
              <span className="num dataset-name">
                {d.name ? (
                  <>
                    {d.name} <span className="muted">{shortDatasetId(d.dataset_id)}</span>
                  </>
                ) : (
                  d.dataset_id
                )}
              </span>
              <span className="num dataset-meta">
                {d.n_rows.toLocaleString()} rows · {d.n_cols} features · PCA {d.pca.dims} (
                {fmtPct(d.pca.explained_variance_ratio.reduce((a, b) => a + b, 0), 0)} var) · test{' '}
                {fmtPct(d.split.test_ratio, 0)} · seed {d.split.seed} ·{' '}
                {d.target.transform === 'log1p' ? 'log target' : 'raw target'} ·{' '}
                {fmtDateTime(d.created_at)} ·{' '}
                <Link to={`/datasets/${d.dataset_id}`} onClick={(e) => e.stopPropagation()}>
                  explore
                </Link>
              </span>
            </label>
          ))}
        </div>
      )}
      {empty && (
        <p className="muted">
          No datasets yet. <Link to="/datasets/new">Build the first one</Link>; it fetches all
          sale listings from Qdrant and freezes a reproducible train/test split.
        </p>
      )}
      {!empty && (
        <p className="dataset-build-link">
          <Link to="/datasets/new">Build new dataset ↗</Link>
        </p>
      )}
    </div>
  )
}
