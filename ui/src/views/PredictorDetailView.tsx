import { useEffect, useMemo } from 'react'
import { Link, useParams } from 'react-router-dom'

import { api } from '../api/client'
import type { Param, RunMeta } from '../api/types'
import { DatasetRef, DefinitionRef, PredictorRef, RunRef } from '../components/EntityRef'
import StatusBadge from '../components/StatusBadge'
import ViewHeader from '../components/ViewHeader'
import { useAsync } from '../hooks/useAsync'
import { fmtDateTime, fmtMoney, fmtR2 } from '../lib/format'
import './predictordetail.css'

/* A predictor is a fixed registry entry, not user CRUD: this page reads the
 * registry (display name, description, param schema, capabilities) and the
 * runs that exercised it. Reached via PredictorRef, not the nav. */

function paramRange(p: Param): string {
  if (p.options) return p.options.join(' · ')
  if (p.min != null && p.max != null) return `${p.min} – ${p.max}`
  if (p.min != null) return `≥ ${p.min}`
  if (p.max != null) return `≤ ${p.max}`
  return '—'
}

export default function PredictorDetailView() {
  const { name } = useParams<{ name: string }>()
  const predictors = useAsync(() => api.listPredictors(), [])
  const runs = useAsync(() => api.listRuns(), [])

  const p = predictors.data?.find((x) => x.name === name)

  useEffect(() => {
    document.title = p ? `${p.display_name} · predictors` : 'predictor'
  }, [p])

  const myRuns = useMemo(
    () => (runs.data ?? []).filter((r: RunMeta) => r.predictor === name),
    [runs.data, name],
  )

  const crumbs = [{ label: 'Runs', to: '/' }, { label: 'predictors' }, { label: name ?? '' }]

  if (predictors.loading)
    return (
      <>
        <ViewHeader glyph="pr" title="Predictor" crumbs={crumbs} />
        <div className="view-body">
          <div className="skeleton skeleton-md" />
        </div>
      </>
    )
  if (predictors.error)
    return (
      <>
        <ViewHeader glyph="pr" title="Predictor" crumbs={crumbs} />
        <div className="view-body">
          <div className="error-block" role="alert">
            Could not load the predictor registry: {predictors.error}
          </div>
        </div>
      </>
    )
  if (!p)
    return (
      <>
        <ViewHeader glyph="pr" title="Predictor" crumbs={crumbs} />
        <div className="view-body">
          <div className="error-block" role="alert">
            No predictor named <span className="num">{name}</span> in the registry.{' '}
            <Link to="/">Back to runs</Link>
          </div>
        </div>
      </>
    )

  return (
    <>
      <ViewHeader
        glyph="pr"
        crumbs={crumbs}
        title={
          <>
            {p.display_name} <PredictorRef name={p.name} self />
          </>
        }
        meta={p.description ? <span>{p.description}</span> : undefined}
      />
      <div className="view-body">
      <header className="run-header">
        <dl className="run-meta">
          <div>
            <dt>Name</dt>
            <dd className="num">{p.name}</dd>
          </div>
          <div>
            <dt>Language</dt>
            <dd>{p.language ?? <span className="muted">—</span>}</dd>
          </div>
          <div>
            <dt>Framework</dt>
            <dd>{p.framework ?? <span className="muted">—</span>}</dd>
          </div>
          <div>
            <dt>Graceful stop</dt>
            <dd>{p.supports_stop ? 'supported' : 'not supported (stop = kill)'}</dd>
          </div>
          <div>
            <dt>Architecture viz</dt>
            <dd>{p.visualization ? 'exports viz.svg' : 'none'}</dd>
          </div>
          <div>
            <dt>Predict</dt>
            <dd>
              {p.predict_args ? (
                p.predict_args.length > 0 ? (
                  p.predict_args.map((a) => (
                    <span key={a} className="chip">
                      {a}
                    </span>
                  ))
                ) : (
                  'supported'
                )
              ) : (
                'not supported'
              )}
            </dd>
          </div>
        </dl>
      </header>

      <section className="panel">
        <h2 className="section-title panel-title">Hyperparameters</h2>
        {p.params.length === 0 ? (
          <p className="muted">No tunable hyperparameters.</p>
        ) : (
          <table className="params-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Label</th>
                <th>Type</th>
                <th className="num-col">Default</th>
                <th>Range / options</th>
              </tr>
            </thead>
            <tbody>
              {p.params.map((param) => (
                <tr key={param.name}>
                  <td className="num">{param.name}</td>
                  <td>{param.label ?? <span className="muted">—</span>}</td>
                  <td className="num">{param.type}</td>
                  <td className="num-col num">{JSON.stringify(param.default)}</td>
                  <td className="num">{paramRange(param)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="panel">
        <h2 className="section-title panel-title">
          Runs <span className="view-count num">{myRuns.length}</span>
        </h2>
        {runs.loading ? (
          <div className="skeleton skeleton-row" />
        ) : myRuns.length === 0 ? (
          <p className="muted">
            No runs have used this predictor yet. <Link to="/new">Start one</Link>.
          </p>
        ) : (
          <table className="lineage-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Dataset</th>
                <th>Definition</th>
                <th>Status</th>
                <th className="num-col">MAE</th>
                <th className="num-col">R²</th>
                <th>Started</th>
              </tr>
            </thead>
            <tbody>
              {myRuns.map((r) => (
                <tr key={r.run_id}>
                  <td>
                    <RunRef id={r.run_id} />
                  </td>
                  <td>
                    <DatasetRef id={r.dataset_id} />
                  </td>
                  <td>
                    {r.from_definition ? (
                      <DefinitionRef name={r.from_definition} />
                    ) : (
                      <span className="muted">—</span>
                    )}
                  </td>
                  <td>
                    <StatusBadge status={r.status} />
                  </td>
                  <td className="num-col num">{r.metrics ? fmtMoney(r.metrics.mae) : '—'}</td>
                  <td className="num-col num">{r.metrics ? fmtR2(r.metrics.r2) : '—'}</td>
                  <td className="num">{fmtDateTime(r.started_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
      </div>
    </>
  )
}
