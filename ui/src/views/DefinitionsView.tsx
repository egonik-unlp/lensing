import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { ModelDefinition } from '../api/types'
import { DefinitionRef, PredictorRef } from '../components/EntityRef'
import LatticeMark from '../components/LatticeMark'
import HyperparamForm from '../components/HyperparamForm'
import SubmitRow from '../components/SubmitRow'
import ViewHeader from '../components/ViewHeader'
import { hpToJson, initialHpState, validateHp, type HpState } from '../lib/hyperparams'
import { useAsync } from '../hooks/useAsync'
import { fmtDateTime } from '../lib/format'
import './models.css'
import './definitions.css'
import { useDocTitle } from '../lib/DomainContext'

const NAME_RE = /^[a-z0-9][a-z0-9-]{0,63}$/

export default function DefinitionsView() {
  useDocTitle('Definitions')
  const defs = useAsync(() => api.listDefinitions(), [])
  const [showNew, setShowNew] = useState(false)

  const list = defs.data ?? []

  return (
    <section aria-label="Model definitions">
      <ViewHeader
        glyph="def"
        title="Definitions"
        count={!defs.loading && !defs.error ? list.length : undefined}
        meta={
          <span>
            Named predictor + hyperparameter presets, versioned in{' '}
            <span className="num">models.toml</span>.
          </span>
        }
        actions={
          !showNew && !defs.loading && !defs.error && list.length > 0 ? (
            <button className="btn btn-primary" onClick={() => setShowNew(true)}>
              New definition
            </button>
          ) : undefined
        }
      />
      <div className="view-body">
      {defs.error ? (
        <div className="error-block" role="alert">
          Could not load definitions: {defs.error}{' '}
          <button className="btn" onClick={defs.reload}>
            Retry
          </button>
        </div>
      ) : defs.loading ? (
        <div className="skeleton skeleton-sm" />
      ) : list.length === 0 && !showNew ? (
        <section className="empty-state" aria-label="No definitions yet">
          <LatticeMark />
          <h1>No definitions yet</h1>
          <p>
            A definition is a named, reusable configuration — a predictor plus the exact
            hyperparameters — saved in <span className="num">models.toml</span> so it can be
            versioned and shared via git. Launch it against any dataset, clone it to iterate,
            and track which datasets it has been run on.
          </p>
          <p style={{ marginTop: 'var(--sp-3)' }}>
            <button className="btn btn-primary" onClick={() => setShowNew(true)}>
              New definition
            </button>
          </p>
        </section>
      ) : (
        <>
          <table className="models-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Predictor</th>
                <th>Hyperparameters</th>
                <th>Datasets used</th>
                <th>Updated</th>
                <th>Notes</th>
              </tr>
            </thead>
            <tbody>
              {list.map((d) => (
                <tr key={d.name}>
                  <td>
                    <DefinitionRef name={d.name} />
                  </td>
                  <td>
                    <PredictorRef name={d.predictor} impl />
                  </td>
                  <td className="num def-hp-summary" title={hpSummary(d)}>
                    {hpSummary(d)}
                  </td>
                  <td className="num" title={d.dataset_tags.join('\n') || undefined}>
                    {d.dataset_tags.length || '—'}
                  </td>
                  <td className="num">{fmtDateTime(d.updated_at ?? d.created_at)}</td>
                  <td className="muted notes-cell">{d.notes || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {showNew && (
        <NewDefinitionForm
          onCancel={() => setShowNew(false)}
          existing={list.map((d) => d.name)}
        />
      )}
      </div>
    </section>
  )
}

function hpSummary(d: ModelDefinition): string {
  return Object.entries(d.hyperparams)
    .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
    .join(' · ')
}

function NewDefinitionForm({ onCancel, existing }: { onCancel: () => void; existing: string[] }) {
  const navigate = useNavigate()
  const predictors = useAsync(() => api.listPredictors(), [])
  const [name, setName] = useState('')
  const [notes, setNotes] = useState('')
  const [predictorChoice, setPredictorChoice] = useState<string | null>(null)
  // Per-predictor edits over the schema defaults; derived, never effect-reset.
  const [hpEdits, setHpEdits] = useState<Record<string, HpState>>({})
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const predictorName = predictorChoice ?? predictors.data?.[0]?.name ?? null
  const predictor = predictors.data?.find((p) => p.name === predictorName) ?? null
  const hp: HpState = predictor
    ? (hpEdits[predictor.name] ?? initialHpState(predictor.params))
    : { values: {}, errors: {} }

  const nameOk = NAME_RE.test(name)
  const taken = existing.includes(name)
  const hpValid = predictor
    ? Object.keys(validateHp(predictor.params, hp.values).errors).length === 0
    : false
  const canCreate = nameOk && !taken && !!predictor && hpValid && !busy

  const create = async () => {
    if (!canCreate || !predictor) return
    setBusy(true)
    setError(null)
    try {
      const def = await api.createDefinition({
        name,
        predictor: predictor.name,
        hyperparams: hpToJson(predictor.params, hp.values),
        notes: notes.trim() || undefined,
      })
      navigate(`/definitions/${def.name}`)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setBusy(false)
    }
  }

  return (
    <div className="panel" style={{ marginTop: 'var(--sp-4)' }}>
      <h2>New definition</h2>
      <div className="def-form">
        <div className="hp-field">
          <label htmlFor="def-name">Name</label>
          <input
            id="def-name"
            type="text"
            className="mono-input"
            placeholder="e.g. mlp-deep-dropout"
            value={name}
            onChange={(e) => setName(e.target.value)}
            disabled={busy}
            autoFocus
          />
          {name && !nameOk && (
            <span className="hp-hint">lowercase letters, digits and dashes only</span>
          )}
          {taken && (
            <span className="hp-error" role="alert">
              A definition with this name already exists
            </span>
          )}
        </div>
        <div className="predictor-row">
          <label htmlFor="def-predictor">Predictor</label>
          <select
            id="def-predictor"
            value={predictorName ?? ''}
            onChange={(e) => setPredictorChoice(e.target.value)}
            disabled={!predictors.data || busy}
          >
            {(predictors.data ?? []).map((p) => (
              <option key={p.name} value={p.name}>
                {p.display_name}
              </option>
            ))}
          </select>
          {predictor && <span className="muted predictor-desc">{predictor.description}</span>}
        </div>
        {predictor && (
          <HyperparamForm
            params={predictor.params}
            state={{ values: hp.values, errors: validateHp(predictor.params, hp.values).errors }}
            onChange={(n, v) =>
              setHpEdits((m) => ({
                ...m,
                [predictor.name]: { ...hp, values: { ...hp.values, [n]: v } },
              }))
            }
            disabled={busy}
          />
        )}
        <div className="hp-field">
          <label htmlFor="def-notes">Notes (optional)</label>
          <input
            id="def-notes"
            type="text"
            className="mono-input"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            disabled={busy}
          />
        </div>
        <SubmitRow
          label="Create definition"
          busyLabel="Creating…"
          busy={busy}
          disabled={!canCreate}
          blockReason={
            !nameOk
              ? 'name the definition first'
              : taken
                ? 'that name is taken'
                : !hpValid
                  ? 'fix the hyperparameter errors first'
                  : undefined
          }
          error={error}
          onSubmit={() => void create()}
        >
          <button className="btn" onClick={onCancel} disabled={busy}>
            Cancel
          </button>
        </SubmitRow>
      </div>
    </div>
  )
}
