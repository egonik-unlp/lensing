import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { DatasetRef, ModelRef, PredictorRef, RunRef } from '../components/EntityRef'
import ViewHeader from '../components/ViewHeader'
import DensityToggle from '../components/DensityToggle'
import { useDensity } from '../hooks/useDensity'
import { useNavigate } from 'react-router-dom'
import { useAsync } from '../hooks/useAsync'
import { fmtDateTime } from '../lib/format'
import './models.css'

export default function ModelsView() {
  useEffect(() => {
    document.title = 'Models · Price Guesser Models'
  }, [])
  const navigate = useNavigate()
  const models = useAsync(() => api.listModels(), [])
  const [density, toggleDensity] = useDensity()

  const list = models.data ?? []

  return (
    <section aria-label="Promoted models">
      <ViewHeader
        glyph="ml"
        title="Models"
        count={!models.loading && !models.error ? list.length : undefined}
        meta={<span>Promoted runs: weights and featurization contract frozen under a name.</span>}
        actions={<DensityToggle density={density} onToggle={toggleDensity} />}
      />
      <div className="view-body">
        {models.error ? (
          <div className="error-block" role="alert">
            Could not load models: {models.error}{' '}
            <button className="btn" onClick={models.reload}>
              Retry
            </button>
          </div>
        ) : models.loading ? (
          <div className="skeleton skeleton-sm" />
        ) : list.length === 0 ? (
          <section className="empty-state" aria-label="No models yet">
            <h1>No models yet</h1>
            <p>
              A model is a promoted training run: its weights and featurization contract, frozen
              under a name you can invoke any time. Open a succeeded (or stopped) run and promote
              it.
            </p>
            <p style={{ marginTop: 'var(--sp-3)' }}>
              <Link to="/" className="btn btn-primary">
                View runs
              </Link>
            </p>
          </section>
        ) : (
          <table className="models-table" data-density={density}>
            <thead>
              <tr>
                <th>Name</th>
                <th>Predictor</th>
                <th>From run</th>
                <th>Dataset</th>
                <th>Promoted</th>
                <th>Notes</th>
              </tr>
            </thead>
            <tbody>
              {list.map((m) => (
                <tr
                  key={m.name}
                  className="run-row"
                  onClick={(e) => {
                    if ((e.target as HTMLElement).closest('a,button')) return
                    navigate(`/models/${m.name}`)
                  }}
                >
                  <td>
                    <ModelRef name={m.name} />
                  </td>
                  <td>
                    <PredictorRef name={m.predictor} impl />
                  </td>
                  <td>
                    <RunRef id={m.run_id} />
                  </td>
                  <td>
                    <DatasetRef id={m.dataset_id} />
                  </td>
                  <td className="num">{fmtDateTime(m.created_at)}</td>
                  <td className="muted notes-cell">{m.notes || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  )
}
