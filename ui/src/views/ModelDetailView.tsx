import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api/client'
import type { ContractSummary, PredictResponse } from '../api/types'
import ArchViz from '../components/ArchViz'
import { DatasetRef, DefinitionRef, ModelRef, PredictorRef, RunRef } from '../components/EntityRef'
import SubmitRow from '../components/SubmitRow'
import ViewHeader from '../components/ViewHeader'
import { useAsync } from '../hooks/useAsync'
import { fmtDateTime, fmtMoney } from '../lib/format'
import './models.css'
import './definitions.css'

const NAME_RE = /^[a-z0-9][a-z0-9-]{0,63}$/

export default function ModelDetailView() {
  const { name } = useParams<{ name: string }>()
  const navigate = useNavigate()
  const model = useAsync(() => api.getModel(name!), [name])
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState<string | null>(null)

  useEffect(() => {
    if (name) document.title = `${name} · Price Guesser Models`
  }, [name])

  if (model.error) {
    return (
      <>
        <ViewHeader glyph="ml" title="Model" crumbs={[{ label: 'Models', to: '/models' }]} />
        <div className="view-body">
          <div className="error-block" role="alert">
            Could not load model: {model.error}{' '}
            <button className="btn" onClick={model.reload}>
              Retry
            </button>{' '}
            <Link to="/models">Back to models</Link>
          </div>
        </div>
      </>
    )
  }
  if (model.loading || !model.data) {
    return (
      <>
        <ViewHeader glyph="ml" title="Model" crumbs={[{ label: 'Models', to: '/models' }]} />
        <div className="view-body">
          <div className="skeleton skeleton-md" />
        </div>
      </>
    )
  }

  const { record, contract, hyperparams } = model.data

  const remove = async () => {
    if (!window.confirm(`Delete model ${record.name}? Its snapshot directory is removed; the original run is untouched.`)) return
    setDeleting(true)
    setDeleteError(null)
    try {
      await api.deleteModel(record.name)
      navigate('/models')
    } catch (e) {
      setDeleteError(e instanceof Error ? e.message : String(e))
      setDeleting(false)
    }
  }

  return (
    <section aria-label={`Model ${record.name}`}>
      <ViewHeader
        glyph="ml"
        crumbs={[{ label: 'Models', to: '/models' }, { label: record.name }]}
        title={
          <>
            <ModelRef name={record.name} self />{' '}
            <span className="model-kind">
              <PredictorRef name={record.predictor} impl />
            </span>
          </>
        }
        meta={
          <span>
            promoted <span className="num">{fmtDateTime(record.created_at)}</span> from{' '}
            <RunRef id={record.run_id} />
          </span>
        }
        actions={
          <button className="btn-on-chrome" onClick={remove} disabled={deleting}>
            {deleting ? 'Deleting…' : 'Delete model'}
          </button>
        }
      />
      <div className="view-body">
      <header className="run-header">
        {deleteError && (
          <p className="hp-error" role="alert">
            {deleteError}
          </p>
        )}
        <dl className="run-meta">
          <div>
            <dt>From run</dt>
            <dd>
              <RunRef id={record.run_id} />
            </dd>
          </div>
          <div>
            <dt>Dataset</dt>
            <dd>
              <DatasetRef id={record.dataset_id} short={false} />
            </dd>
          </div>
          <div>
            <dt>Promoted</dt>
            <dd className="num">{fmtDateTime(record.created_at)}</dd>
          </div>
          <div>
            <dt>Target</dt>
            <dd className="num">
              {contract.target.field} ({contract.target.transform})
            </dd>
          </div>
          <div>
            <dt>Features</dt>
            <dd className="num">
              {contract.n_cols} cols · PCA {contract.pca_dims}
            </dd>
          </div>
          {record.notes && (
            <div className="run-hp">
              <dt>Notes</dt>
              <dd>{record.notes}</dd>
            </div>
          )}
        </dl>
        <ModelActions
          name={record.name}
          predictor={record.predictor}
          datasetId={record.dataset_id}
          hyperparams={hyperparams}
          onRenamed={(n) => navigate(`/models/${n}`)}
        />
      </header>

      <ArchViz
        predictor={record.predictor}
        hyperparams={hyperparams}
        features={{ nCols: contract.n_cols, nPca: contract.pca_dims }}
        fallbackUrl={api.modelVizUrl(record.name)}
      />
      <ContractPanel contract={contract} />
      <Playground name={record.name} contract={contract} />
      </div>
    </section>
  )
}

/* ---------------- rename / clone params ---------------- */

function ModelActions({
  name,
  predictor,
  datasetId,
  hyperparams,
  onRenamed,
}: {
  name: string
  predictor: string
  datasetId: string
  hyperparams: Record<string, unknown> | null
  onRenamed: (name: string) => void
}) {
  const [mode, setMode] = useState<'idle' | 'rename' | 'clone'>('idle')
  const [newName, setNewName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [cloned, setCloned] = useState<string | null>(null)

  const nameOk = NAME_RE.test(newName)

  const submit = async () => {
    setBusy(true)
    setError(null)
    try {
      if (mode === 'rename') {
        const record = await api.renameModel(name, newName)
        onRenamed(record.name)
      } else {
        // Clone the training params into a fresh, launchable definition.
        const def = await api.createDefinition({
          name: newName,
          predictor,
          hyperparams: hyperparams ?? undefined,
          dataset_tags: [datasetId],
          notes: `params cloned from model ${name}`,
        })
        setCloned(def.name)
      }
      setMode('idle')
      setNewName('')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="def-actions" style={{ marginTop: 'var(--sp-3)' }}>
      <button className="btn" onClick={() => setMode(mode === 'rename' ? 'idle' : 'rename')} disabled={busy}>
        Rename
      </button>
      {hyperparams && (
        <button
          className="btn"
          onClick={() => setMode(mode === 'clone' ? 'idle' : 'clone')}
          disabled={busy}
          title="Save this model's training hyperparameters as a reusable definition"
        >
          Clone params to definition
        </button>
      )}
      {cloned && (
        <span className="muted" role="status">
          cloned to <DefinitionRef name={cloned} />
        </span>
      )}
      {mode !== 'idle' && (
        <div className="def-inline-form">
          <div className="hp-field">
            <label htmlFor="model-new-name">
              {mode === 'rename' ? 'New model name' : 'Definition name'}
            </label>
            <input
              id="model-new-name"
              type="text"
              className="mono-input"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              disabled={busy}
              autoFocus
            />
            {newName && !nameOk && (
              <span className="hp-hint">lowercase letters, digits and dashes only</span>
            )}
          </div>
          <button className="btn" onClick={submit} disabled={!nameOk || busy}>
            {busy ? 'Working…' : mode === 'rename' ? 'Rename' : 'Create definition'}
          </button>
        </div>
      )}
      {error && (
        <span className="hp-error" role="alert">
          {error}
        </span>
      )}
    </div>
  )
}

function ContractPanel({ contract }: { contract: ContractSummary }) {
  const f = contract.input_fields
  return (
    <div className="panel">
      <h2>Inference contract</h2>
      <p className="muted contract-line">
        Frozen at training time; every prediction conforms to it. An input item needs{' '}
        {f.required_numeric.length > 0 && (
          <>
            numeric <span className="num">{f.required_numeric.join(', ')}</span>,{' '}
          </>
        )}
        {f.required_categorical.length > 0 && (
          <>
            categorical <span className="num">{f.required_categorical.join(', ')}</span>{' '}
            (unseen values fall into the trained <span className="num">__other__</span> bucket),{' '}
          </>
        )}
        and a <span className="num">{f.embedding_dim}</span>-dim <span className="num">embedding</span>.
        Qdrant point ids skip all of that: the server fetches payload + embedding itself.
      </p>
    </div>
  )
}

/* ---------------- predict playground ---------------- */

type Tab = 'ids' | 'json'

function Playground({ name, contract }: { name: string; contract: ContractSummary }) {
  const [tab, setTab] = useState<Tab>('ids')
  const [ids, setIds] = useState('')
  const [rawJson, setRawJson] = useState('')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const run = async () => {
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      let body: { items?: unknown[]; point_ids?: number[] }
      if (tab === 'ids') {
        const point_ids = ids
          .split(/[\s,]+/)
          .filter(Boolean)
          .map((s) => {
            const n = Number(s)
            if (!Number.isInteger(n) || n < 0) throw new Error(`"${s}" is not a point id`)
            return n
          })
        if (point_ids.length === 0) throw new Error('Enter at least one point id')
        body = { point_ids }
      } else {
        const parsed: unknown = JSON.parse(rawJson)
        body = { items: Array.isArray(parsed) ? parsed : [parsed] }
      }
      setResult(await api.predictModel(name, body))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="panel">
      <h2>Predict</h2>
      <div className="tab-row" role="tablist" aria-label="Input mode">
        <button
          role="tab"
          aria-selected={tab === 'ids'}
          className={`tab${tab === 'ids' ? ' is-active' : ''}`}
          onClick={() => setTab('ids')}
        >
          By Qdrant point id
        </button>
        <button
          role="tab"
          aria-selected={tab === 'json'}
          className={`tab${tab === 'json' ? ' is-active' : ''}`}
          onClick={() => setTab('json')}
        >
          Raw item JSON
        </button>
      </div>

      {tab === 'ids' ? (
        <div className="hp-field playground-field">
          <label htmlFor="pg-ids">Point ids (comma or space separated)</label>
          <input
            id="pg-ids"
            type="text"
            className="mono-input"
            placeholder="e.g. 184220, 191502"
            value={ids}
            onChange={(e) => setIds(e.target.value)}
            disabled={busy}
          />
        </div>
      ) : (
        <div className="hp-field playground-field">
          <label htmlFor="pg-json">Item JSON (one object or an array)</label>
          <textarea
            id="pg-json"
            className="mono-input json-input"
            rows={8}
            spellCheck={false}
            placeholder={`{\n  "propertyType": "apartment",\n  "bedrooms": 2,\n  "neighborhood": "…",\n  "embedding": [${contract.input_fields.embedding_dim} floats]\n}`}
            value={rawJson}
            onChange={(e) => setRawJson(e.target.value)}
            disabled={busy}
          />
        </div>
      )}

      <SubmitRow
        label="Predict prices"
        busyLabel="Predicting…"
        busy={busy}
        error={error}
        onSubmit={() => void run()}
      />

      {result && (
        <div className="playground-result">
          <table className="models-table">
            <thead>
              <tr>
                <th>Item</th>
                <th className="num-col">Predicted price</th>
              </tr>
            </thead>
            <tbody>
              {result.predictions.map((p) => (
                <tr key={p.row_id}>
                  <td className="num">{p.row_id}</td>
                  <td className="num-col num">{fmtMoney(p.predicted)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {result.warnings.length > 0 && (
            <ul className="warning-list" aria-label="Input warnings">
              {result.warnings.map((w, i) => (
                <li key={i}>
                  <span className="warn-glyph" aria-hidden>
                    ⚠
                  </span>{' '}
                  {w}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
