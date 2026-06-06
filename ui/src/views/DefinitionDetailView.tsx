import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api/client'
import type { ModelDefinition, Predictor } from '../api/types'
import { DatasetRef, DefinitionRef, PredictorRef } from '../components/EntityRef'
import HyperparamForm from '../components/HyperparamForm'
import SweepHeatmap from '../components/SweepHeatmap'
import ViewHeader from '../components/ViewHeader'
import { hpStateFromValues, hpToJson, validateHp, type HpState } from '../lib/hyperparams'
import { useAsync } from '../hooks/useAsync'
import { fmtDateTime } from '../lib/format'
import './models.css'
import './definitions.css'
import { useDocTitle } from '../lib/DomainContext'

const NAME_RE = /^[a-z0-9][a-z0-9-]{0,63}$/

export default function DefinitionDetailView() {
  const { name } = useParams<{ name: string }>()
  const navigate = useNavigate()
  const def = useAsync(() => api.getDefinition(name!), [name])
  const predictors = useAsync(() => api.listPredictors(), [])

  useDocTitle(name ?? null)

  if (def.error) {
    return (
      <>
        <ViewHeader glyph="def" title="Definition" crumbs={[{ label: 'Definitions', to: '/definitions' }]} />
        <div className="view-body">
          <div className="error-block" role="alert">
            Could not load definition: {def.error}{' '}
            <button className="btn" onClick={def.reload}>
              Retry
            </button>{' '}
            <Link to="/definitions">Back to definitions</Link>
          </div>
        </div>
      </>
    )
  }
  if (def.loading || !def.data) {
    return (
      <>
        <ViewHeader glyph="def" title="Definition" crumbs={[{ label: 'Definitions', to: '/definitions' }]} />
        <div className="view-body">
          <div className="skeleton skeleton-md" />
        </div>
      </>
    )
  }

  const d = def.data
  const predictor = predictors.data?.find((p) => p.name === d.predictor) ?? null

  return (
    <section aria-label={`Definition ${d.name}`}>
      <ViewHeader
        glyph="def"
        crumbs={[{ label: 'Definitions', to: '/definitions' }, { label: d.name }]}
        title={
          <>
            <DefinitionRef name={d.name} self />{' '}
            <span className="model-kind">
              <PredictorRef name={d.predictor} impl />
            </span>
          </>
        }
        meta={
          <span>
            created <span className="num">{fmtDateTime(d.created_at)}</span>
            {d.updated_at && (
              <>
                {' '}
                · updated <span className="num">{fmtDateTime(d.updated_at)}</span>
              </>
            )}{' '}
            · stored in <span className="num">models.toml</span> (git-versioned)
          </span>
        }
        actions={
          <Link to={`/new?definition=${encodeURIComponent(d.name)}`} className="btn btn-primary">
            Launch run
          </Link>
        }
      />
      <div className="view-body">
      <header className="run-header">
        <Actions def={d} onChanged={(newName) => navigate(`/definitions/${newName}`)} />
      </header>

      {predictor ? (
        <ConfigPanel key={`${d.name}-${d.updated_at ?? ''}`} def={d} predictor={predictor} onSaved={def.reload} />
      ) : predictors.loading ? (
        <div className="skeleton" style={{ height: '6rem' }} />
      ) : (
        <div className="error-block" role="alert">
          Predictor <span className="num">{d.predictor}</span> is not in the registry; this
          definition cannot be edited or launched until it is restored in registry.toml.
        </div>
      )}

      <SweepHeatmap def={d} predictor={predictor} />

      <TagsPanel def={d} onSaved={def.reload} />
      </div>
    </section>
  )
}

/* ---------------- rename / clone / delete ---------------- */

function Actions({ def, onChanged }: { def: ModelDefinition; onChanged: (name: string) => void }) {
  const navigate = useNavigate()
  const [mode, setMode] = useState<'idle' | 'rename' | 'clone'>('idle')
  const [newName, setNewName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const nameOk = NAME_RE.test(newName) && newName !== def.name

  const submit = async () => {
    setBusy(true)
    setError(null)
    try {
      if (mode === 'rename') {
        await api.renameDefinition(def.name, newName)
      } else {
        await api.cloneDefinition(def.name, newName)
      }
      onChanged(newName)
      setMode('idle')
      setNewName('')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const remove = async () => {
    if (!window.confirm(`Delete definition ${def.name}? Past runs launched from it are untouched.`)) return
    setBusy(true)
    setError(null)
    try {
      await api.deleteDefinition(def.name)
      navigate('/definitions')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setBusy(false)
    }
  }

  return (
    <div className="def-actions" style={{ marginTop: 'var(--sp-3)' }}>
      <button className="btn" onClick={() => setMode(mode === 'rename' ? 'idle' : 'rename')} disabled={busy}>
        Rename
      </button>
      <button className="btn" onClick={() => setMode(mode === 'clone' ? 'idle' : 'clone')} disabled={busy}>
        Clone
      </button>
      <button className="btn" onClick={remove} disabled={busy}>
        Delete
      </button>
      {mode !== 'idle' && (
        <div className="def-inline-form">
          <div className="hp-field">
            <label htmlFor="def-new-name">
              {mode === 'rename' ? 'New name' : 'Clone as'}
            </label>
            <input
              id="def-new-name"
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
            {busy ? 'Working…' : mode === 'rename' ? 'Rename' : 'Clone'}
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

/* ---------------- hyperparams + notes ---------------- */

function ConfigPanel({
  def,
  predictor,
  onSaved,
}: {
  def: ModelDefinition
  predictor: Predictor
  onSaved: () => void
}) {
  const [hp, setHp] = useState<HpState>(() => hpStateFromValues(predictor.params, def.hyperparams))
  const [notes, setNotes] = useState(def.notes ?? '')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)

  const hpValid = Object.keys(validateHp(predictor.params, hp.values).errors).length === 0
  const dirty =
    notes !== (def.notes ?? '') ||
    JSON.stringify(hpToJson(predictor.params, hp.values)) !== JSON.stringify(def.hyperparams)

  const save = async () => {
    setBusy(true)
    setError(null)
    try {
      await api.updateDefinition(def.name, {
        hyperparams: hpToJson(predictor.params, hp.values),
        notes,
      })
      setSaved(true)
      window.setTimeout(() => setSaved(false), 1500)
      onSaved()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="panel">
      <h2>Configuration</h2>
      <HyperparamForm
        params={predictor.params}
        state={{ values: hp.values, errors: validateHp(predictor.params, hp.values).errors }}
        onChange={(n, v) => setHp((s) => ({ ...s, values: { ...s.values, [n]: v } }))}
        disabled={busy}
      />
      <div className="hp-field" style={{ marginTop: 'var(--sp-3)', maxWidth: '48ch' }}>
        <label htmlFor="def-notes">Notes</label>
        <input
          id="def-notes"
          type="text"
          className="mono-input"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          disabled={busy}
        />
      </div>
      <div className="start-row">
        <button className="btn" onClick={save} disabled={!dirty || !hpValid || busy}>
          {busy ? 'Saving…' : 'Save changes'}
        </button>
        {saved && (
          <span className="muted" role="status">
            saved
          </span>
        )}
        {error && (
          <span className="hp-error" role="alert">
            {error}
          </span>
        )}
      </div>
    </div>
  )
}

/* ---------------- dataset tags ---------------- */

function TagsPanel({ def, onSaved }: { def: ModelDefinition; onSaved: () => void }) {
  const datasets = useAsync(() => api.listDatasets(), [])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [adding, setAdding] = useState('')

  const setTags = async (tags: string[]) => {
    setBusy(true)
    setError(null)
    try {
      await api.updateDefinition(def.name, { dataset_tags: tags })
      onSaved()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const untagged = (datasets.data ?? [])
    .map((m) => m.dataset_id)
    .filter((id) => !def.dataset_tags.includes(id))

  return (
    <div className="panel">
      <h2>Datasets used</h2>
      <p className="muted contract-line">
        Tagged automatically when a run is launched from this definition; tag by hand to mark a
        dataset as intended for it. Any existing dataset can be picked at launch time.
      </p>
      {def.dataset_tags.length > 0 ? (
        <ul className="tag-list" aria-label="Dataset tags">
          {def.dataset_tags.map((t) => (
            <li key={t} className="tag-chip">
              <DatasetRef id={t} short={false} />
              <button
                className="tag-remove"
                aria-label={`Remove tag ${t}`}
                title="Remove tag"
                disabled={busy}
                onClick={() => void setTags(def.dataset_tags.filter((x) => x !== t))}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted">No datasets tagged yet.</p>
      )}
      {untagged.length > 0 && (
        <div className="tag-add-row">
          <select
            aria-label="Dataset to tag"
            value={adding}
            onChange={(e) => setAdding(e.target.value)}
            disabled={busy}
          >
            <option value="">— pick a dataset —</option>
            {untagged.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
          <button
            className="btn"
            disabled={!adding || busy}
            onClick={() => {
              void setTags([...def.dataset_tags, adding])
              setAdding('')
            }}
          >
            Tag dataset
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
