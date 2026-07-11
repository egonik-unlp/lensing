import { useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { usePredictorInfo } from '../hooks/usePredictors'
import { shortDatasetId, shortRunId } from '../lib/format'
import './entityref.css'

/* The permeation primitive: any run id, dataset id, model name, definition
 * name or predictor name rendered as text becomes one of these — a compact
 * mono link with a dimmed kind glyph. No chip background, no border; the
 * link color and the glyph are the whole affordance.
 *
 * `self` marks the entity's own page (no link, ink color, aria-current).
 * `copy` appends an inline copy action for the full id. */

function CopyButton({ full }: { full: string }) {
  const [copied, setCopied] = useState(false)
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined)
  return (
    <button
      className={copied ? 'btn-inline entity-ref-copy is-copied' : 'btn-inline entity-ref-copy'}
      onClick={() => {
        void navigator.clipboard.writeText(full)
        setCopied(true)
        clearTimeout(timer.current)
        timer.current = setTimeout(() => setCopied(false), 1200)
      }}
    >
      {copied ? 'copied' : 'copy'}
    </button>
  )
}

function Ref({
  to,
  glyph,
  label,
  full,
  kind,
  self = false,
  copy = false,
}: {
  to: string
  glyph: string
  label: string
  full: string
  kind: string
  self?: boolean
  copy?: boolean
}) {
  const inner = (
    <>
      <span className="entity-glyph" aria-hidden>
        {glyph}
      </span>
      {label}
    </>
  )
  const ref = self ? (
    <span className="entity-ref entity-ref-self num" aria-current="page" title={`${kind} ${full}`}>
      {inner}
    </span>
  ) : (
    <Link to={to} className="entity-ref num" title={`${kind} ${full}`}>
      {inner}
    </Link>
  )
  if (!copy) return ref
  return (
    <span className="entity-ref-wrap">
      {ref}
      <CopyButton full={full} />
    </span>
  )
}

export function RunRef({
  id,
  short = true,
  self = false,
  copy = false,
  label,
}: {
  id: string
  short?: boolean
  self?: boolean
  copy?: boolean
  /** Display override (e.g. tinyRunId in tables that show the predictor
   *  separately); the title attribute still carries the full id. */
  label?: string
}) {
  return (
    <Ref
      to={`/runs/${id}`}
      glyph="run"
      label={label ?? (short ? shortRunId(id) : id)}
      full={id}
      kind="run"
      self={self}
      copy={copy}
    />
  )
}

export function DatasetRef({
  id,
  name,
  short = true,
  self = false,
  copy = false,
}: {
  id: string
  name?: string | null
  short?: boolean
  self?: boolean
  copy?: boolean
}) {
  if (name) {
    return (
      <span className="dataset-ref-named">
        <Ref to={`/datasets/${id}`} glyph="ds" label={name} full={id} kind="dataset" self={self} />{' '}
        <span className="muted num dataset-ref-id">{short ? shortDatasetId(id) : id}</span>
        {copy && <CopyButton full={id} />}
      </span>
    )
  }
  return (
    <Ref
      to={`/datasets/${id}`}
      glyph="ds"
      label={short ? shortDatasetId(id) : id}
      full={id}
      kind="dataset"
      self={self}
      copy={copy}
    />
  )
}

export function ModelRef({ name, self = false }: { name: string; self?: boolean }) {
  return <Ref to={`/models/${name}`} glyph="ml" label={name} full={name} kind="model" self={self} />
}

export function RepresentationRef({
  id,
  name,
  self = false,
}: {
  id: string
  name?: string
  self?: boolean
}) {
  return (
    <Ref
      to={`/representations/${id}`}
      glyph="rp"
      label={name ?? id}
      full={id}
      kind="representation"
      self={self}
    />
  )
}

export function DefinitionRef({ name, self = false }: { name: string; self?: boolean }) {
  return (
    <Ref to={`/definitions/${name}`} glyph="def" label={name} full={name} kind="definition" self={self} />
  )
}

export function ListingRef({ id, self = false }: { id: number; self?: boolean }) {
  return (
    <Ref to={`/listings/${id}`} glyph="ls" label={String(id)} full={String(id)} kind="listing" self={self} />
  )
}

export function PredictorRef({
  name,
  self = false,
  impl = false,
}: {
  name: string
  self?: boolean
  /** Trail the link with the implementation badge (language · framework). */
  impl?: boolean
}) {
  const ref = (
    <Ref to={`/predictors/${name}`} glyph="pr" label={name} full={name} kind="predictor" self={self} />
  )
  if (!impl) return ref
  return (
    <span className="entity-ref-wrap">
      {ref}
      <ImplBadge predictor={name} />
    </span>
  )
}

/** Implementation language · framework of a predictor, from the registry.
 *  Renders nothing while the registry loads or when the entry doesn't
 *  declare a language — older registries simply show no badge. */
export function ImplBadge({ predictor }: { predictor: string }) {
  const p = usePredictorInfo(predictor)
  if (!p?.language) return null
  const label = p.framework ? `${p.language} · ${p.framework}` : p.language
  return (
    <span
      className="impl-badge"
      title={`implemented in ${p.language}${p.framework ? `, using ${p.framework}` : ''}`}
    >
      {label}
    </span>
  )
}
