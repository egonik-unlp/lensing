import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api/client'
import type { CreateListingRequest, ListingCoordinates, ListingSummary } from '../api/types'
import SubmitRow from '../components/SubmitRow'
import ViewHeader from '../components/ViewHeader'
import { useAsync } from '../hooks/useAsync'
import { useDomain } from '../lib/DomainContext'
import { cap } from '../lib/format'
import {
  fieldLabel,
  pinnedFilterValues,
  type Domain,
  type DomainField,
} from '../lib/domain'
import { invalidateConsensus } from '../lib/listingPredict'
import './newrun.css'
import './listings.css'

/* Field state is all strings; parsing happens at submit so partially typed
 * numbers never fight the input. Empty fields are simply omitted from the
 * request — the server treats them as unspecified, like the corpus does. The
 * input set is generated from the domain's editable fields, plus the
 * display-only extras (images, source URL) and the lat/lon coordinate pair. */

interface Draft {
  content: string
  /** Domain field name → raw string. */
  fields: Record<string, string>
  lat: string
  lon: string
  /** One image URL per line. */
  images: string
  sourceUrl: string
}

/** Roles that map to a single editable input in the details form. */
function editableFields(domain: Domain): DomainField[] {
  return domain.fields.filter(
    (f) => f.role === 'numeric' || f.role === 'categorical' || f.role === 'target',
  )
}

function emptyDraft(domain: Domain): Draft {
  const pinned = pinnedFilterValues(domain)
  const fields: Record<string, string> = {}
  for (const f of editableFields(domain)) {
    fields[f.name] = pinned[f.name] ?? ''
  }
  return { content: '', fields, lat: '', lon: '', images: '', sourceUrl: '' }
}

function parseNum(raw: string, label: string, allowNegative = false): number | undefined {
  const s = raw.trim()
  if (!s) return undefined
  const n = Number(s)
  if (!Number.isFinite(n)) throw new Error(`${label} is not a number`)
  if (!allowNegative && n < 0) throw new Error(`${label} must not be negative`)
  return n
}

/** Image URLs, one per line; anything non-http(s) is a typo worth stopping on. */
function parseImages(raw: string): string[] | undefined {
  const urls = raw
    .split('\n')
    .map((s) => s.trim())
    .filter(Boolean)
  if (urls.length === 0) return undefined
  for (const u of urls) {
    if (!/^https?:\/\//.test(u)) throw new Error(`Image URL "${u.slice(0, 40)}…" must start with http(s)://`)
  }
  return urls
}

/** A stored listing, restated as form field strings. */
function toDraft(l: ListingSummary, domain: Domain): Draft {
  const md = l.metadata
  const pinned = pinnedFilterValues(domain)
  const fields: Record<string, string> = {}
  for (const f of editableFields(domain)) {
    const v = md[f.name]
    if (f.role === 'numeric' || f.role === 'target') {
      fields[f.name] = typeof v === 'number' && v > 0 ? String(v) : ''
    } else {
      fields[f.name] = v != null && v !== '' ? String(v) : pinned[f.name] ?? ''
    }
  }
  const coords = (md.coordinates as ListingCoordinates | null | undefined) ?? null
  return {
    content: l.content,
    fields,
    lat: coords?.lat != null ? String(coords.lat) : '',
    lon: coords?.lon != null ? String(coords.lon) : '',
    images: (md.images ?? []).join('\n'),
    sourceUrl: md.sourceUrl ?? '',
  }
}

/** Create form at /listings/new; the same form edits at /listings/:id/edit. */
export default function NewListingView() {
  const domain = useDomain()
  const { id } = useParams<{ id: string }>()
  const editId = id !== undefined ? Number(id) : null
  const noun = domain.project.entity_noun
  useEffect(() => {
    document.title = `${editId !== null ? `Edit ${noun} ${editId}` : `New ${noun}`} · ${domain.project.title}`
  }, [editId, noun, domain.project.title])
  const navigate = useNavigate()
  const [draft, setDraft] = useState<Draft>(() => emptyDraft(domain))
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)

  const pinned = useMemo(() => pinnedFilterValues(domain), [domain])
  const fields = useMemo(() => editableFields(domain), [domain])
  const coordField = domain.coordinates
  // Detail fields are everything bar the target; the target gets the
  // "advertised value, never sent to models" treatment in its own section.
  const targetField = domain.target.field
  const detailFields = fields.filter((f) => f.name !== targetField && !pinned[f.name])
  const targetDef = fields.find((f) => f.name === targetField)
  // Pinned fields (e.g. operation=="sale" from the corpus filter) are shown
  // disabled so created listings match the corpus, even if filter_only.
  const pinnedFields = useMemo(
    () => domain.fields.filter((f) => pinned[f.name] != null && f.name !== targetField),
    [domain, pinned, targetField],
  )

  // Edit mode: prefill once the listing loads.
  const existing = useAsync(
    () => (editId !== null ? api.getListing(editId) : Promise.resolve(null)),
    [editId],
  )
  useEffect(() => {
    // Prefill from the fetched listing: external-sync, not a render-loop.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (existing.data) setDraft(toDraft(existing.data, domain))
  }, [existing.data, domain])

  const setContent = (e: React.ChangeEvent<HTMLTextAreaElement>) =>
    setDraft((d) => ({ ...d, content: e.target.value }))
  const setField = (name: string) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setDraft((d) => ({ ...d, fields: { ...d.fields, [name]: e.target.value } }))
  const setExtra = (k: 'lat' | 'lon' | 'images' | 'sourceUrl') => (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) => setDraft((d) => ({ ...d, [k]: e.target.value }))

  const contentOk = draft.content.trim().length > 0
  const blockReason = !contentOk ? 'Write a description first — it is the text that gets embedded' : null

  const submit = async () => {
    if (!contentOk || submitting) return
    setSubmitting(true)
    setSubmitError(null)
    try {
      const req: CreateListingRequest = { content: draft.content.trim() }
      for (const f of fields) {
        const raw = (draft.fields[f.name] ?? '').trim()
        if (f.required && !raw) throw new Error(`${fieldLabel(f)} is required`)
        if (!raw) continue
        if (f.role === 'numeric' || f.role === 'target') {
          req[f.name] = parseNum(raw, fieldLabel(f))
        } else {
          req[f.name] = raw
        }
      }
      // Filter-pinned values (e.g. operation=="sale") that aren't editable
      // inputs still ride along so the listing matches the corpus filter.
      for (const [name, value] of Object.entries(pinned)) {
        if (req[name] === undefined && name !== targetField) req[name] = value
      }
      const images = parseImages(draft.images)
      if (images) req.images = images
      if (draft.sourceUrl.trim()) req.sourceUrl = draft.sourceUrl.trim()
      if (coordField) {
        const lat = parseNum(draft.lat, 'Latitude', true)
        const lon = parseNum(draft.lon, 'Longitude', true)
        if (lat !== undefined || lon !== undefined) req.coordinates = { lat, lon }
      }
      const saved =
        editId !== null ? await api.updateListing(editId, req) : await api.createListing(req)
      if (editId !== null) invalidateConsensus(editId)
      navigate(`/listings/${saved.id}`)
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e))
      setSubmitting(false)
    }
  }

  const editing = editId !== null
  if (editing && existing.error) {
    return (
      <>
        <ViewHeader glyph="ls" title={`Edit ${noun}`} crumbs={[{ label: cap(domain.project.entity_noun_plural), to: '/listings' }]} />
        <div className="view-body">
          <div className="error-block" role="alert">
            Could not load {noun}: {existing.error}{' '}
            <button className="btn" onClick={existing.reload}>
              Retry
            </button>{' '}
            <Link to="/listings">Back to {domain.project.entity_noun_plural}</Link>
          </div>
        </div>
      </>
    )
  }
  if (editing && (existing.loading || !existing.data)) {
    return (
      <>
        <ViewHeader glyph="ls" title={`Edit ${noun}`} crumbs={[{ label: cap(domain.project.entity_noun_plural), to: '/listings' }]} />
        <div className="view-body">
          <div className="skeleton skeleton-md" />
        </div>
      </>
    )
  }

  return (
    <section aria-label={editing ? `Edit ${noun} ${editId}` : `Create a manual ${noun}`}>
      <ViewHeader
        glyph="ls"
        title={editing ? `Edit ${noun} ${editId}` : `New ${noun}`}
        crumbs={
          editing
            ? [
                { label: cap(domain.project.entity_noun_plural), to: '/listings' },
                { label: String(editId), to: `/listings/${editId}` },
                { label: 'Edit' },
              ]
            : [{ label: cap(domain.project.entity_noun_plural), to: '/listings' }, { label: 'New' }]
        }
        meta={
          <span>
            {editing
              ? 'Changing the description re-embeds it; metadata-only edits keep the stored embedding.'
              : `Describe the ${noun}; the server embeds the description and stores it for later ${domain.project.target_noun} prediction.`}
          </span>
        }
      />
      <div className="view-body newrun">
        <div className="panel">
          <h2>1 · Description</h2>
          <div className="hp-field">
            <label htmlFor="listing-content">
              {cap(noun)} text — written like a real {noun}; this is what gets embedded
            </label>
            <textarea
              id="listing-content"
              className="mono-input listing-content-input"
              rows={7}
              value={draft.content}
              onChange={setContent}
              placeholder="Luminoso departamento de 85 m² en Palermo, 2 dormitorios, balcón al frente…"
              disabled={submitting}
              autoFocus
            />
            <span className="hp-hint">
              Mentions like “85 m²” also let raw-numerics models backfill a missing area, as in
              training.
            </span>
          </div>
        </div>

        <div className="panel">
          <h2>2 · Details</h2>
          <div className="hp-grid">
            {pinnedFields.map((f) => (
              <FieldInput
                key={f.name}
                field={f}
                value={pinned[f.name] ?? ''}
                onChange={setField(f.name)}
                disabled
                hint="pinned to match the training corpus"
              />
            ))}
            {detailFields.map((f) => (
              <FieldInput
                key={f.name}
                field={f}
                value={draft.fields[f.name] ?? ''}
                onChange={setField(f.name)}
                disabled={submitting}
              />
            ))}
          </div>
        </div>

        <div className="panel">
          <h2>3 · Optional</h2>
          <div className="hp-grid">
            {targetDef && (
              <FieldInput
                field={targetDef}
                value={draft.fields[targetField] ?? ''}
                onChange={setField(targetField)}
                disabled={submitting}
                placeholder="as advertised"
                hint={`kept for comparison only — never sent to the models`}
              />
            )}
            {coordField && (
              <>
                <div className="hp-field">
                  <label htmlFor="listing-lat">Latitude</label>
                  <input
                    id="listing-lat"
                    type="number"
                    step="any"
                    className="mono-input"
                    value={draft.lat}
                    onChange={setExtra('lat')}
                    placeholder={String(coordField.lat_range[0])}
                    disabled={submitting}
                  />
                </div>
                <div className="hp-field">
                  <label htmlFor="listing-lon">Longitude</label>
                  <input
                    id="listing-lon"
                    type="number"
                    step="any"
                    className="mono-input"
                    value={draft.lon}
                    onChange={setExtra('lon')}
                    placeholder={String(coordField.lon_range[0])}
                    disabled={submitting}
                  />
                </div>
              </>
            )}
            <div className="hp-field">
              <label htmlFor="listing-sourceUrl">Source URL</label>
              <input
                id="listing-sourceUrl"
                type="url"
                className="mono-input"
                value={draft.sourceUrl}
                onChange={setExtra('sourceUrl')}
                placeholder="https://www.zonaprop.com.ar/…"
                disabled={submitting}
              />
            </div>
            <div className="hp-field listing-images-field">
              <label htmlFor="listing-images">Photo URLs — one per line</label>
              <textarea
                id="listing-images"
                className="mono-input listing-content-input"
                rows={3}
                value={draft.images}
                onChange={setExtra('images')}
                placeholder={'https://…/frente.jpg\nhttps://…/living.jpg'}
                disabled={submitting}
              />
              <span className="hp-hint">shown in the UI, never sent to the models</span>
            </div>
          </div>
        </div>

        <SubmitRow
          label={editing ? 'Save changes' : `Create ${noun}`}
          busyLabel="Embedding & saving…"
          busy={submitting}
          disabled={!contentOk}
          blockReason={blockReason}
          error={submitError}
          onSubmit={() => void submit()}
        />
      </div>
    </section>
  )
}

/** One domain field as a form input: number for numeric/target, text +
 *  datalist (from suggestions) for categorical. */
function FieldInput({
  field,
  value,
  onChange,
  disabled,
  placeholder,
  hint,
}: {
  field: DomainField
  value: string
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void
  disabled?: boolean
  placeholder?: string
  hint?: string
}) {
  const id = `listing-${field.name}`
  const listId = `${id}-suggestions`
  const numeric = field.role === 'numeric' || field.role === 'target'
  const suggestions = field.suggestions ?? []
  return (
    <div className="hp-field">
      <label htmlFor={id}>
        {fieldLabel(field)}
        {field.required ? ' *' : ''}
      </label>
      <input
        id={id}
        type={numeric ? 'number' : 'text'}
        min={numeric ? 0 : undefined}
        className="mono-input"
        list={suggestions.length > 0 ? listId : undefined}
        value={value}
        onChange={onChange}
        placeholder={placeholder ?? (numeric ? 'unspecified' : '')}
        disabled={disabled}
      />
      {suggestions.length > 0 && (
        <datalist id={listId}>
          {suggestions.map((s) => (
            <option key={s} value={s} />
          ))}
        </datalist>
      )}
      {hint && <span className="hp-hint">{hint}</span>}
    </div>
  )
}
