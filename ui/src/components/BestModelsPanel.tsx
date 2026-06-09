import { useState } from 'react'
import { api } from '../api/client'
import type { BestModelGroup } from '../api/types'
import { useAsync } from '../hooks/useAsync'
import { fmtDateTime, fmtTarget } from '../lib/format'
import { useDomain } from '../lib/DomainContext'
import { DatasetRef, ModelRef, PredictorRef, RunRef } from './EntityRef'
import './bestmodels.css'

/** The server-maintained best-models group: top-N promoted models by the
 *  primary metric, recomputed on every run completion and promotion, and the
 *  set behind consensus predictions. Read-only here — curation judgment
 *  (pinning, excluding suspicious metrics) belongs to the best-model-selector
 *  agent; the panel offers Recompute only. */
export default function BestModelsPanel({
  onGroup,
}: {
  /** Lifts the loaded group so the caller can badge its own rows. */
  onGroup?: (group: BestModelGroup) => void
}) {
  const domain = useDomain()
  const group = useAsync(async () => {
    const g = await api.bestModels()
    onGroup?.(g)
    return g
  }, [])
  const [busy, setBusy] = useState(false)
  const [recomputeError, setRecomputeError] = useState<string | null>(null)

  const recompute = async () => {
    setBusy(true)
    setRecomputeError(null)
    try {
      await api.recomputeBestModels()
      group.reload()
    } catch (e) {
      setRecomputeError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  if (group.error) {
    return (
      <div className="best-models" aria-label="Best models">
        <PanelHead onRecompute={recompute} busy={busy} />
        <div className="error-block" role="alert">
          Could not load the best-models group: {group.error}{' '}
          <button className="btn" onClick={group.reload}>
            Retry
          </button>
        </div>
      </div>
    )
  }
  if (group.loading || !group.data) {
    return (
      <div className="best-models" aria-label="Best models">
        <PanelHead busy={false} />
        <div className="skeleton" style={{ height: '4rem' }} />
      </div>
    )
  }

  const g = group.data
  return (
    <div className="best-models" aria-label="Best models">
      <PanelHead
        group={g}
        onRecompute={recompute}
        busy={busy}
      />
      {recomputeError && (
        <p className="hp-error" role="alert">
          {recomputeError}
        </p>
      )}
      {g.entries.length === 0 ? (
        <p className="muted best-models-empty">
          The group fills itself: when a run finishes, the top {g.size || 12} promoted models by{' '}
          {g.primary_metric} are selected automatically. Promote a run, or recompute to seed it
          from what exists.
        </p>
      ) : (
        <table className="models-table best-models-table">
          <thead>
            <tr>
              <th className="num-col best-rank-col">#</th>
              <th>Model</th>
              <th>Predictor</th>
              <th className="num-col">{g.primary_metric}</th>
              <th>Dataset</th>
              <th>From run</th>
              <th>Selected</th>
            </tr>
          </thead>
          <tbody>
            {g.entries.map((e) => (
              <tr key={e.name} className="run-row">
                <td className="num-col num best-rank">{e.rank}</td>
                <td>
                  <ModelRef name={e.name} />
                  {e.source === 'pinned' && <span className="best-pin">pinned</span>}
                </td>
                <td>
                  <PredictorRef name={e.predictor} impl />
                </td>
                <td className="num-col num">{fmtTarget(e.metric_value, domain)}</td>
                <td>
                  <DatasetRef id={e.dataset_id} />
                </td>
                <td>
                  <RunRef id={e.run_id} />
                </td>
                <td className="num">{fmtDateTime(e.selected_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {g.excluded.length > 0 && (
        <details className="best-excluded">
          <summary>
            {g.excluded.length} excluded from the group
          </summary>
          <p className="num">{g.excluded.join(' · ')}</p>
        </details>
      )}
    </div>
  )
}

function PanelHead({
  group,
  onRecompute,
  busy,
}: {
  group?: BestModelGroup
  onRecompute?: () => void
  busy: boolean
}) {
  return (
    <div className="best-models-head">
      <h2 className="section-title">
        Best models{' '}
        {group && (
          <span className="muted num best-models-meta">
            top {group.size} by {group.primary_metric} · feeds consensus predictions · updated{' '}
            {fmtDateTime(group.updated_at)}
          </span>
        )}
      </h2>
      {onRecompute && (
        <button className="btn" onClick={onRecompute} disabled={busy}>
          {busy ? 'Recomputing…' : 'Recompute'}
        </button>
      )}
    </div>
  )
}
