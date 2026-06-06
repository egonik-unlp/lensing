import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api/client'
import type { Listing, PredictResponse } from '../api/types'
import { ListingRef, ModelRef } from '../components/EntityRef'
import ListingImage from '../components/ListingImage'
import SubmitRow from '../components/SubmitRow'
import ViewHeader from '../components/ViewHeader'
import { useAsync } from '../hooks/useAsync'
import { useDomain, useDocTitle } from '../lib/DomainContext'
import {
  categoricalFields,
  currencyValue,
  fieldLabel,
  targetValue,
  timestampFieldName,
} from '../lib/domain'
import { cap, fmtDateTime, fmtMoney, fmtSignedPct } from '../lib/format'
import { invalidateConsensus, median, resolvePredictGroup, toPredictItem } from '../lib/listingPredict'
import './models.css'
import './newrun.css'
import './listings.css'

export default function ListingDetailView() {
  const domain = useDomain()
  const { id } = useParams<{ id: string }>()
  const listingId = Number(id)
  const navigate = useNavigate()
  const listing = useAsync(() => api.getListing(listingId), [listingId])
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState<string | null>(null)

  useDocTitle(id ? `${cap(domain.project.entity_noun)} ${id}` : null)

  if (listing.error) {
    return (
      <>
        <ViewHeader glyph="ls" title="Listing" crumbs={[{ label: 'Listings', to: '/listings' }]} />
        <div className="view-body">
          <div className="error-block" role="alert">
            Could not load listing: {listing.error}{' '}
            <button className="btn" onClick={listing.reload}>
              Retry
            </button>{' '}
            <Link to="/listings">Back to listings</Link>
          </div>
        </div>
      </>
    )
  }
  if (listing.loading || !listing.data) {
    return (
      <>
        <ViewHeader glyph="ls" title="Listing" crumbs={[{ label: 'Listings', to: '/listings' }]} />
        <div className="view-body">
          <div className="skeleton skeleton-md" />
        </div>
      </>
    )
  }

  const l = listing.data
  const md = l.metadata
  const listed = targetValue(domain, md)
  const currency = currencyValue(domain, md)
  const tsField = timestampFieldName(domain)
  const created = tsField ? md[tsField] : null
  const cats = categoricalFields(domain)
  const lead = cats[0] ? md[cats[0].name] : null
  const sub = cats[1] ? md[cats[1].name] : null
  const images = md.images ?? []
  const sourceHost = (() => {
    if (!md.sourceUrl) return null
    try {
      return new URL(md.sourceUrl).hostname.replace(/^www\./, '')
    } catch {
      return md.sourceUrl
    }
  })()

  const remove = async () => {
    if (!window.confirm(`Delete ${domain.project.entity_noun} ${l.id}? Its stored embedding is removed too.`)) return
    setDeleting(true)
    setDeleteError(null)
    try {
      await api.deleteListing(l.id)
      invalidateConsensus(l.id)
      navigate('/listings')
    } catch (e) {
      setDeleteError(e instanceof Error ? e.message : String(e))
      setDeleting(false)
    }
  }

  return (
    <section aria-label={`Listing ${l.id}`}>
      <ViewHeader
        glyph="ls"
        crumbs={[{ label: 'Listings', to: '/listings' }, { label: String(l.id) }]}
        title={<ListingRef id={l.id} self />}
        meta={
          <span>
            {typeof lead === 'string' && lead ? lead : domain.project.entity_noun}
            {typeof sub === 'string' && sub ? <> · {sub}</> : null}
            {listed != null ? (
              <>
                {' '}
                · listed at{' '}
                <span className="num">
                  {fmtMoney(listed)}
                  {currency ? ` ${currency}` : ''}
                </span>
              </>
            ) : null}{' '}
            · created{' '}
            <span className="num">
              {typeof created === 'string' && created ? fmtDateTime(created) : '—'}
            </span>
          </span>
        }
        actions={
          <>
            <Link className="btn-on-chrome" to={`/listings/${l.id}/edit`}>
              Edit listing
            </Link>
            <button className="btn-on-chrome" onClick={remove} disabled={deleting}>
              {deleting ? 'Deleting…' : 'Delete listing'}
            </button>
          </>
        }
      />
      <div className="view-body">
        {deleteError && (
          <p className="hp-error" role="alert">
            {deleteError}
          </p>
        )}

        <div className={images.length > 0 ? 'listing-overview' : undefined}>
          {images.length > 0 && <Gallery images={images} listingId={l.id} />}

          <dl className="run-meta listing-meta">
            {domain.fields
              .filter((f) => f.role === 'categorical' || f.role === 'filter_only')
              .map((f) => {
                const v = md[f.name]
                return (
                  <div key={f.name}>
                    <dt>{cap(fieldLabel(f))}</dt>
                    <dd>{typeof v === 'string' && v ? v : '—'}</dd>
                  </div>
                )
              })}
            {domain.fields
              .filter((f) => f.role === 'numeric')
              .map((f) => {
                const v = md[f.name]
                return (
                  <div key={f.name}>
                    <dt>{cap(fieldLabel(f))}</dt>
                    <dd className="num">{typeof v === 'number' && v ? v : '—'}</dd>
                  </div>
                )
              })}
            <div>
              <dt>Listed {domain.project.target_noun}</dt>
              <dd className="num">
                {listed != null ? `${fmtMoney(listed)} ${currency ?? ''}` : '— (no value)'}
              </dd>
            </div>
            {md.coordinates && (md.coordinates.lat != null || md.coordinates.lon != null) && (
              <div>
                <dt>Coordinates</dt>
                <dd className="num">
                  {md.coordinates.lat ?? '—'}, {md.coordinates.lon ?? '—'}
                </dd>
              </div>
            )}
            {md.sourceUrl && (
              <div>
                <dt>Source</dt>
                <dd>
                  <a href={md.sourceUrl} target="_blank" rel="noopener noreferrer">
                    {sourceHost} ↗
                  </a>
                </dd>
              </div>
            )}
            <div>
              <dt>Embedding</dt>
              <dd className="num">{l.embedding.length} dims</dd>
            </div>
          </dl>
        </div>

        <div className="panel">
          <h2>Description</h2>
          <p className="listing-content">{l.content}</p>
        </div>

        <PredictPanel listing={l} />
      </div>
    </section>
  )
}

/* ---------------- photo gallery ---------------- */

function Gallery({ images, listingId }: { images: string[]; listingId: number }) {
  const [idx, setIdx] = useState(0)
  const current = Math.min(idx, images.length - 1)
  return (
    <figure className="listing-gallery" aria-label={`Photos of listing ${listingId}`}>
      <ListingImage
        src={images[current]}
        alt={`Listing ${listingId}, photo ${current + 1} of ${images.length}`}
        className="listing-gallery-main"
      />
      {images.length > 1 && (
        <div className="listing-gallery-thumbs" role="group" aria-label="Choose photo">
          {images.map((u, i) => (
            <button
              key={`${i}-${u}`}
              type="button"
              className="listing-gallery-thumbbtn"
              aria-label={`Photo ${i + 1}`}
              aria-current={i === current}
              onClick={() => setIdx(i)}
            >
              <ListingImage src={u} alt="" className="listing-gallery-thumb" />
            </button>
          ))}
        </div>
      )}
      <figcaption className="listing-gallery-caption">
        <span className="num">
          {current + 1}/{images.length}
        </span>{' '}
        · hotlinked from the source page
      </figcaption>
    </figure>
  )
}

/* ---------------- predict panel ---------------- */

/** One model's prediction slot: queued, settled, or failed. */
type ModelResult =
  | { state: 'pending' }
  | { state: 'done'; predicted: number; warnings: string[] }
  | { state: 'error'; error: string }

function PredictPanel({ listing }: { listing: Listing }) {
  const domain = useDomain()
  // One load: every promoted model (the picker) plus the best-models group
  // (the default selection; server-maintained, heuristic fallback).
  const models = useAsync(async () => {
    const all = await api.listModels()
    const group = await resolvePredictGroup(all)
    return { all, group }
  }, [])
  // null until the user touches the picker: the default (the best-model
  // group) is derived from the loaded list, not stored.
  const [chosen, setChosen] = useState<Set<string> | null>(null)
  const [busy, setBusy] = useState(false)
  const [results, setResults] = useState<Record<string, ModelResult> | null>(null)

  const all = models.data?.all ?? []
  const groupNames = models.data?.group.names ?? []
  const selectedNames = chosen ?? new Set(groupNames)
  const selected = all.filter((m) => selectedNames.has(m.name))

  const toggle = (name: string) =>
    setChosen(() => {
      const next = new Set(selectedNames)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })

  const run = async () => {
    if (selected.length === 0 || busy) return
    setBusy(true)
    setResults(Object.fromEntries(selected.map((m) => [m.name, { state: 'pending' } as ModelResult])))
    // The stored listing already holds its embedding, so each prediction is
    // one inline-items call. The item deliberately excludes the listed
    // target value — it is the answer key, not an input. Calls go out together;
    // the server's run-slot semaphore paces the predictors.
    const item = toPredictItem(listing, domain)
    await Promise.all(
      selected.map(async (m) => {
        let result: ModelResult
        try {
          const r: PredictResponse = await api.predictModel(m.name, { items: [item] })
          const p = r.predictions[0]
          result = p
            ? { state: 'done', predicted: p.predicted, warnings: r.warnings }
            : { state: 'error', error: 'predictor returned no prediction' }
        } catch (e) {
          result = { state: 'error', error: e instanceof Error ? e.message : String(e) }
        }
        setResults((prev) => (prev ? { ...prev, [m.name]: result } : prev))
      }),
    )
    setBusy(false)
  }

  const done = results
    ? Object.values(results).filter((r): r is Extract<ModelResult, { state: 'done' }> => r.state === 'done')
    : []
  const consensus = median(done.map((r) => r.predicted))
  const warnings = results ? [...new Set(done.flatMap((r) => r.warnings))] : []
  const listed = targetValue(domain, listing.metadata)

  return (
    <div className="panel">
      <h2>Predict {domain.project.target_noun}</h2>
      {models.error ? (
        <div className="error-block" role="alert">
          Could not load models: {models.error}{' '}
          <button className="btn" onClick={models.reload}>
            Retry
          </button>
        </div>
      ) : models.loading ? (
        <div className="skeleton skeleton-sm" />
      ) : all.length === 0 ? (
        <p className="muted">
          No promoted models yet. <Link to="/">Train a run</Link> and promote it, then come back to
          predict this {domain.project.entity_noun}&rsquo;s {domain.project.target_noun}.
        </p>
      ) : (
        <>
          <div className="listing-model-picker" role="group" aria-label="Models to predict with">
            <div className="listing-model-picker-head">
              <span className="muted">
                {selected.length} of {all.length} models selected
              </span>
              <span>
                <button
                  className="btn-link"
                  onClick={() => setChosen(new Set(groupNames))}
                  disabled={busy}
                >
                  best
                </button>
                {' · '}
                <button
                  className="btn-link"
                  onClick={() => setChosen(new Set(all.map((m) => m.name)))}
                  disabled={busy}
                >
                  all
                </button>
                {' · '}
                <button className="btn-link" onClick={() => setChosen(new Set())} disabled={busy}>
                  none
                </button>
              </span>
            </div>
            <div className="listing-model-grid">
              {all.map((m) => (
                <label key={m.name} className="listing-model-check">
                  <input
                    type="checkbox"
                    checked={selectedNames.has(m.name)}
                    onChange={() => toggle(m.name)}
                    disabled={busy}
                  />
                  <span className="num">{m.name}</span> <span className="muted">({m.predictor})</span>
                </label>
              ))}
            </div>
          </div>
          <SubmitRow
            label={selected.length > 1 ? `Predict with ${selected.length} models` : `Predict ${domain.project.target_noun}`}
            busyLabel="Predicting…"
            busy={busy}
            disabled={selected.length === 0}
            blockReason={selected.length === 0 ? 'Select at least one model' : null}
            error={null}
            onSubmit={() => void run()}
          />
          {results && (
            <div className="playground-result">
              {consensus !== null && (
                <p className="listing-predicted">
                  <span className="num listing-predicted-value">{fmtMoney(consensus)}</span>{' '}
                  <span className="muted">
                    {done.length > 1 ? `median of ${done.length} models` : 'predicted'}
                    {listed ? (
                      <>
                        {' '}
                        · listed at <span className="num">{fmtMoney(listed)}</span> (
                        <span className="num">{fmtSignedPct(consensus / listed - 1)}</span>)
                      </>
                    ) : null}
                  </span>
                </p>
              )}
              <table className="listing-models-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Predictor</th>
                    <th className="num-col">Predicted</th>
                    {listed ? <th className="num-col">vs listed</th> : null}
                  </tr>
                </thead>
                <tbody>
                  {selected.map((m) => {
                    const r = results[m.name]
                    if (!r) return null
                    return (
                      <tr key={m.name}>
                        <td>
                          <ModelRef name={m.name} />
                        </td>
                        <td className="muted">{m.predictor}</td>
                        {r.state === 'pending' ? (
                          <td className="num muted" colSpan={listed ? 2 : 1}>
                            …
                          </td>
                        ) : r.state === 'error' ? (
                          <td className="listing-predict-error" colSpan={listed ? 2 : 1}>
                            {r.error}
                          </td>
                        ) : (
                          <>
                            <td className="num">{fmtMoney(r.predicted)}</td>
                            {listed ? (
                              <td className="num">{fmtSignedPct(r.predicted / listed - 1)}</td>
                            ) : null}
                          </>
                        )}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
              {warnings.length > 0 && (
                <ul className="warning-list" aria-label="Input warnings">
                  {warnings.map((w, i) => (
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
          <p className="hp-hint listing-embed-note">
            The embedding was computed server-side at creation time; predictions assume the
            training corpus used the same embedding model. The listed {domain.project.target_noun}, photos, and source
            URL are never part of the predict payload.
          </p>
        </>
      )}
    </div>
  )
}
