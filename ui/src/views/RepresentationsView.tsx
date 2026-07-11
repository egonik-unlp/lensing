import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { RepresentationMeta } from '../api/types'
import ViewHeader from '../components/ViewHeader'
import { RepresentationRef } from '../components/EntityRef'
import { useAsync } from '../hooks/useAsync'
import { fmtDateTime } from '../lib/format'
import { useDocTitle } from '../lib/DomainContext'
import { methodLabel, qualitySummary } from '../lib/representation'

export default function RepresentationsView() {
  useDocTitle('Representations')
  const navigate = useNavigate()
  const reps = useAsync(() => api.listRepresentations(), [])
  const list = reps.data ?? []

  return (
    <section aria-label="Representations">
      <ViewHeader
        glyph="rp"
        title="Representations"
        count={!reps.loading && !reps.error ? list.length : undefined}
        meta={
          <span>
            Latent encodings of the corpus: a compressor (PCA / autoencoder) fit on a source collection,
            producing a dense latent collection that datasets can build on.
          </span>
        }
        actions={
          <Link to="/representations/new" className="btn btn-primary">
            New representation
          </Link>
        }
      />
      <div className="view-body">
        {reps.error ? (
          <div className="error-block" role="alert">
            Could not load representations: {reps.error}{' '}
            <button className="btn" onClick={reps.reload}>
              Retry
            </button>
          </div>
        ) : reps.loading ? (
          <p className="muted">Loading…</p>
        ) : list.length === 0 ? (
          <div className="empty-state">
            <p>No representations yet.</p>
            <p className="muted">
              Build one to compress a collection into a dense latent — PCA, an autoencoder, or a sparse
              autoencoder. Datasets and predictors can then ride the latent collection.
            </p>
            <Link to="/representations/new" className="btn btn-primary">
              New representation
            </Link>
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>name</th>
                <th>method</th>
                <th>source → latent</th>
                <th className="num-col">dim</th>
                <th className="num-col">points</th>
                <th>quality</th>
                <th>built</th>
              </tr>
            </thead>
            <tbody>
              {list.map((r: RepresentationMeta) => (
                <tr
                  key={r.id}
                  className="row-clickable"
                  onClick={() => navigate(`/representations/${r.id}`)}
                >
                  <td>
                    <RepresentationRef id={r.id} name={r.name} />
                  </td>
                  <td>{methodLabel[r.method]}</td>
                  <td className="num">
                    {r.source_collection} → {r.sink_collection}
                  </td>
                  <td className="num-col num">{r.latent_dim}</td>
                  <td className="num-col num">{r.n_points.toLocaleString()}</td>
                  <td className="num">{qualitySummary(r.quality)}</td>
                  <td className="num">{fmtDateTime(r.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  )
}
