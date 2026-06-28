import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { ListingMetadata } from '../api/types'
import DensityToggle from '../components/DensityToggle'
import { ListingRef } from '../components/EntityRef'
import LatticeMark from '../components/LatticeMark'
import ListingImage from '../components/ListingImage'
import ViewHeader from '../components/ViewHeader'
import { useAsync } from '../hooks/useAsync'
import { useDensity } from '../hooks/useDensity'
import { useDomain } from '../lib/DomainContext'
import {
  categoricalFields,
  currencyValue,
  fieldLabel,
  targetValue,
  timestampFieldName,
  type Domain,
} from '../lib/domain'
import { cap } from '../lib/format'
import { fmtTarget, fmtTargetCell, fmtSignedPct, fmtStamp } from '../lib/format'
import {
  invalidateConsensus,
  listingConsensus,
  resolvePredictGroup,
  type ListingConsensus,
} from '../lib/listingPredict'
import './models.css'
import './listings.css'

/** One listing's consensus slot while the view predicts in the background. */
type ConsensusSlot =
  | { state: 'pending' }
  | { state: 'done'; consensus: ListingConsensus }
  | { state: 'error'; error: string }

/** One-line attribute summary, domain-driven: the prominent categorical
 *  values, then every populated numeric with its label. */
function attrLine(domain: Domain, md: ListingMetadata): string {
  const parts: string[] = []
  for (const f of categoricalFields(domain).slice(0, 3)) {
    const v = md[f.name]
    if (typeof v === 'string' && v) parts.push(v)
  }
  for (const f of domain.fields.filter((f) => f.role === 'numeric')) {
    const v = md[f.name]
    if (typeof v === 'number' && v) parts.push(`${v} ${fieldLabel(f)}`)
  }
  return parts.join(' · ')
}

export default function ListingsView() {
  const domain = useDomain()
  useEffect(() => {
    document.title = `${cap(domain.project.entity_noun_plural)} · ${domain.project.title}`
  }, [domain])
  const navigate = useNavigate()
  const listings = useAsync(() => api.listListings(), [])
  // The server-maintained best-models group, falling back to the legacy
  // heuristic over all promoted models when the group is empty.
  const predictGroup = useAsync(async () => resolvePredictGroup(await api.listModels()), [])
  const [density, toggleDensity] = useDensity()
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const [slots, setSlots] = useState<Record<number, ConsensusSlot>>({})

  const list = listings.data ?? []
  const group = predictGroup.data?.names ?? []

  // Auto-predict: one listing at a time (a server-side group consensus call,
  // or the per-model fallback fan-out), so a cold visit never floods the
  // server's run slots. Cached consensus (sessionStorage) resolves instantly.
  useEffect(() => {
    if (!listings.data || !predictGroup.data) return
    const group = predictGroup.data
    if (group.names.length === 0) return
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
  }, [listings.data, predictGroup.data, domain])

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
  const tsField = timestampFieldName(domain)
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
            <LatticeMark />
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
                <th>{cap(noun)}</th>
                <th>Description</th>
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
                const listed = targetValue(domain, md)
                const created = tsField ? md[tsField] : null
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
                      <div className="listing-attrs">{attrLine(domain, md) || '—'}</div>
                    </td>
                    <td className="num-col num col-group-start">
                      {listed ? (
                        <span title={`${fmtTarget(listed, domain)} ${currencyValue(domain, md) ?? ''}`.trim()}>
                          {fmtTargetCell(listed, domain)}
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
                          title={`${fmtTarget(slot.consensus.median, domain)} — median of ${slot.consensus.n_ok} of ${slot.consensus.n_models} models`}
                        >
                          {fmtTargetCell(slot.consensus.median, domain)}
                        </span>
                      )}
                    </td>
                    <td className="num-col num">
                      {slot?.state === 'done' && listed ? (
                        <span
                          title={`Predicted ${fmtTarget(slot.consensus.median, domain)} vs listed ${fmtTarget(listed, domain)}`}
                        >
                          {fmtSignedPct(slot.consensus.median / listed - 1)}
                        </span>
                      ) : (
                        <span className="muted">—</span>
                      )}
                    </td>
                    <td className="num-col num col-group-start">
                      {typeof created === 'string' && created ? fmtStamp(created) : '—'}
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
