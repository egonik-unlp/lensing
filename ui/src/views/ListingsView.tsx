import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { ListingMetadata } from '../api/types'
import DensityToggle from '../components/DensityToggle'
import { ListingRef } from '../components/EntityRef'
import ListingImage from '../components/ListingImage'
import ViewHeader from '../components/ViewHeader'
import { useAsync } from '../hooks/useAsync'
import { useDensity } from '../hooks/useDensity'
import { useDomain } from '../lib/DomainContext'
import { cap } from '../lib/format'
import { fmtMoney, fmtMoneyCell, fmtSignedPct, fmtStamp } from '../lib/format'
import {
  bestModels,
  invalidateConsensus,
  listingConsensus,
  type ListingConsensus,
} from '../lib/listingPredict'
import './models.css'
import './listings.css'

/** One listing's consensus slot while the view predicts in the background. */
type ConsensusSlot =
  | { state: 'pending' }
  | { state: 'done'; consensus: ListingConsensus }
  | { state: 'error'; error: string }

/** "apartment · Centro, La Plata · 2 bd · 1 ba · 85 m²" */
function attrLine(md: ListingMetadata): string {
  const parts: string[] = []
  if (md.propertyType) parts.push(md.propertyType)
  const loc = [md.neighborhood, md.city].filter(Boolean).join(', ')
  if (loc) parts.push(loc)
  if (md.bedrooms) parts.push(`${md.bedrooms} bd`)
  if (md.bathrooms) parts.push(`${md.bathrooms} ba`)
  if (md.totalArea) parts.push(`${md.totalArea} m²`)
  return parts.join(' · ')
}

export default function ListingsView() {
  const domain = useDomain()
  useEffect(() => {
    document.title = `${cap(domain.project.entity_noun_plural)} · ${domain.project.title}`
  }, [domain])
  const navigate = useNavigate()
  const listings = useAsync(() => api.listListings(), [])
  const models = useAsync(() => api.listModels(), [])
  const [density, toggleDensity] = useDensity()
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const [slots, setSlots] = useState<Record<number, ConsensusSlot>>({})

  const list = listings.data ?? []
  const group = models.data ? bestModels(models.data) : []

  // Auto-predict: one listing at a time (each fans out over the model
  // group), so a cold visit never floods the server's run slots. Cached
  // consensus (sessionStorage) resolves instantly.
  useEffect(() => {
    if (!listings.data || !models.data) return
    const group = bestModels(models.data)
    if (group.length === 0) return
    let cancelled = false
    const queue = listings.data
    // Sync fill is the point: every row shows its shimmer the moment the
    // queue starts (same convention as useAsync's sync reset).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSlots((s) => {
      const next = { ...s }
      for (const l of queue) next[l.id] ??= { state: 'pending' }
      return next
    })
    void (async () => {
      for (const l of queue) {
        if (cancelled) return
        try {
          const consensus = await listingConsensus(l.id, group, domain)
          if (!cancelled) setSlots((s) => ({ ...s, [l.id]: { state: 'done', consensus } }))
        } catch (e) {
          const error = e instanceof Error ? e.message : String(e)
          if (!cancelled) setSlots((s) => ({ ...s, [l.id]: { state: 'error', error } }))
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [listings.data, models.data, domain])

  const remove = async (id: number) => {
    if (!window.confirm(`Delete ${domain.project.entity_noun} ${id}? Its stored embedding is removed too.`)) return
    setDeleteError(null)
    try {
      await api.deleteListing(id)
      invalidateConsensus(id)
      listings.reload()
    } catch (e) {
      setDeleteError(e instanceof Error ? e.message : String(e))
    }
  }

  const noun = domain.project.entity_noun
  const nounPl = domain.project.entity_noun_plural
  return (
    <section aria-label={`Manual ${nounPl}`}>
      <ViewHeader
        glyph="ls"
        title={cap(nounPl)}
        count={!listings.loading && !listings.error ? list.length : undefined}
        meta={
          <span>
            Manually captured {nounPl}, embedded server-side.
            {group.length > 0 ? (
              <>
                {' '}
                Predicted = median of the <span className="num">{group.length}</span> best models;
                listed {domain.project.target_noun}s are never sent to them.
              </>
            ) : null}
          </span>
        }
        actions={
          <>
            <DensityToggle density={density} onToggle={toggleDensity} />
            <Link to="/listings/new" className="btn-on-chrome">
              New {noun}
            </Link>
          </>
        }
      />
      <div className="view-body">
        {deleteError && (
          <p className="hp-error" role="alert">
            {deleteError}
          </p>
        )}
        {listings.error ? (
          <div className="error-block" role="alert">
            Could not load {nounPl}: {listings.error}{' '}
            <button className="btn" onClick={listings.reload}>
              Retry
            </button>
          </div>
        ) : listings.loading ? (
          <div className="skeleton skeleton-sm" />
        ) : list.length === 0 ? (
          <section className="empty-state" aria-label={`No ${nounPl} yet`}>
            <h1>No {nounPl} yet</h1>
            <p>
              A manual {noun} is one you describe yourself: the server embeds the
              description, stores it alongside its details, and any promoted model can predict its
              {' '}{domain.project.target_noun}.
            </p>
            <p style={{ marginTop: 'var(--sp-3)' }}>
              <Link to="/listings/new" className="btn btn-primary">
                Create the first {noun}
              </Link>
            </p>
          </section>
        ) : (
          <table className="models-table listings-table" data-density={density}>
            <thead>
              <tr>
                <th aria-label="Photo" className="listing-thumb-col" />
                <th>Listing</th>
                <th>Property</th>
                <th className="num-col col-group-start">Listed</th>
                <th className="num-col">Predicted</th>
                <th className="num-col">Δ vs listed</th>
                <th className="num-col col-group-start">Created</th>
                <th aria-label="Actions" />
              </tr>
            </thead>
            <tbody>
              {list.map((l) => {
                const md = l.metadata
                const listed = md.price > 0 ? md.price : null
                const slot: ConsensusSlot | undefined =
                  group.length > 0 ? slots[l.id] : undefined
                return (
                  <tr
                    key={l.id}
                    className="run-row"
                    onClick={(e) => {
                      if ((e.target as HTMLElement).closest('a,button')) return
                      navigate(`/listings/${l.id}`)
                    }}
                  >
                    <td className="listing-thumb-cell">
                      <ListingImage src={md.images?.[0]} alt="" className="listing-thumb" />
                    </td>
                    <td>
                      <ListingRef id={l.id} />
                    </td>
                    <td className="listing-property-cell">
                      <div className="listing-snippet">{l.content}</div>
                      <div className="listing-attrs">{attrLine(md) || '—'}</div>
                    </td>
                    <td className="num-col num col-group-start">
                      {listed ? (
                        <span title={`${fmtMoney(listed)} ${md.currency ?? ''}`.trim()}>
                          {fmtMoneyCell(listed)}
                        </span>
                      ) : (
                        <span className="muted">—</span>
                      )}
                    </td>
                    <td className="num-col num">
                      {!slot ? (
                        <span className="muted">—</span>
                      ) : slot.state === 'pending' ? (
                        <span
                          className="skeleton listing-pred-skeleton"
                          aria-label="Predicting…"
                        />
                      ) : slot.state === 'error' ? (
                        <span className="muted" title={slot.error}>
                          failed
                        </span>
                      ) : (
                        <span
                          title={`${fmtMoney(slot.consensus.median)} — median of ${slot.consensus.n_ok} of ${slot.consensus.n_models} models`}
                        >
                          {fmtMoneyCell(slot.consensus.median)}
                        </span>
                      )}
                    </td>
                    <td className="num-col num">
                      {slot?.state === 'done' && listed ? (
                        <span
                          title={`Predicted ${fmtMoney(slot.consensus.median)} vs listed ${fmtMoney(listed)}`}
                        >
                          {fmtSignedPct(slot.consensus.median / listed - 1)}
                        </span>
                      ) : (
                        <span className="muted">—</span>
                      )}
                    </td>
                    <td className="num-col num col-group-start">
                      {md.createdAt ? fmtStamp(md.createdAt) : '—'}
                    </td>
                    <td>
                      <button className="btn-inline" onClick={() => void remove(l.id)}>
                        delete
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </section>
  )
}
