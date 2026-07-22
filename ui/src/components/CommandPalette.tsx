import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import { useDomain } from '../lib/DomainContext'
import { categoricalFields } from '../lib/domain'
import { cap, shortDatasetId, shortRunId } from '../lib/format'
import { formatMetric, metricValue } from '../lib/metrics'
import './palette.css'

/* The command palette: the hundredth-visit way around. ⌘K / Ctrl+K from
 * anywhere; jump to any run, model, dataset, definition, listing or
 * predictor by name or id; fire the common actions. Entity indexes are
 * fetched when the palette opens and cached for its lifetime. */

interface Entry {
  key: string
  group: string
  label: string
  /** Dim trailing context (predictor, metric, kind). */
  sub?: string
  /** Extra haystack beyond the label (full ids, predictor names). */
  keywords?: string
  to?: string
  run?: () => void
}

const PER_GROUP = 6

function score(e: Entry, q: string): number {
  if (!q) return 1
  const hay = `${e.label} ${e.sub ?? ''} ${e.keywords ?? ''}`.toLowerCase()
  const terms = q.toLowerCase().split(/\s+/).filter(Boolean)
  let s = 0
  for (const t of terms) {
    const i = hay.indexOf(t)
    if (i === -1) return 0
    // Earlier and label-anchored matches rank higher.
    s += e.label.toLowerCase().startsWith(t) ? 3 : e.label.toLowerCase().includes(t) ? 2 : 1
  }
  return s
}

export default function CommandPalette({ open, onClose }: { open: boolean; onClose: () => void }) {
  const navigate = useNavigate()
  const domain = useDomain()
  // The prominent categorical field (domain-driven, same ordering the entity
  // list view uses) is the one-line sub for each entity result.
  const catField = categoricalFields(domain)[0]?.name
  const entityGroup = cap(domain.project.entity_noun_plural)
  const dialogRef = useRef<HTMLDialogElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLUListElement>(null)
  const [q, setQ] = useState('')
  const [active, setActive] = useState(0)
  const [index, setIndex] = useState<Entry[]>([])

  // Sync the native dialog with the `open` prop.
  useEffect(() => {
    const d = dialogRef.current
    if (!d) return
    if (open && !d.open) {
      d.showModal()
      setQ('')
      setActive(0)
      inputRef.current?.focus()
    } else if (!open && d.open) {
      d.close()
    }
  }, [open])

  // Build the entity index when the palette opens. Failures degrade to
  // whatever loaded — the static entries always work.
  useEffect(() => {
    if (!open) return
    let cancelled = false
    void Promise.allSettled([
      api.listRuns(),
      api.listModels(),
      api.listDatasets(),
      api.listDefinitions(),
      api.listListings(),
      api.listPredictors(),
    ]).then(([runs, models, datasets, defs, listings, predictors]) => {
      if (cancelled) return
      const out: Entry[] = []
      if (runs.status === 'fulfilled') {
        for (const r of runs.value) {
          out.push({
            key: `run-${r.run_id}`,
            group: 'Runs',
            label: shortRunId(r.run_id),
            sub: `${r.predictor}${
              r.metrics
                ? ` · ${domain.metrics.primary} ${formatMetric(
                    domain,
                    domain.metrics.primary,
                    metricValue(r.metrics, domain.metrics.primary),
                  )}`
                : ''
            } · ${r.status}`,
            keywords: `${r.run_id} ${r.dataset_id}`,
            to: `/runs/${r.run_id}`,
          })
        }
      }
      if (models.status === 'fulfilled') {
        for (const m of models.value) {
          out.push({
            key: `model-${m.name}`,
            group: 'Models',
            label: m.name,
            sub: m.predictor,
            keywords: m.run_id,
            to: `/models/${m.name}`,
          })
        }
      }
      if (datasets.status === 'fulfilled') {
        for (const d of datasets.value) {
          out.push({
            key: `ds-${d.dataset_id}`,
            group: 'Datasets',
            label: d.name ?? shortDatasetId(d.dataset_id),
            sub: `${d.n_rows.toLocaleString()} rows · ${d.n_cols} cols`,
            keywords: d.dataset_id,
            to: `/datasets/${d.dataset_id}`,
          })
        }
      }
      if (defs.status === 'fulfilled') {
        for (const d of defs.value) {
          out.push({
            key: `def-${d.name}`,
            group: 'Definitions',
            label: d.name,
            sub: d.predictor,
            to: `/definitions/${d.name}`,
          })
        }
      }
      if (listings.status === 'fulfilled') {
        for (const l of listings.value) {
          const cat = catField ? String(l.metadata[catField] ?? '') : ''
          out.push({
            key: `ls-${l.id}`,
            group: entityGroup,
            label: `#${l.id} ${l.content.slice(0, 60)}`,
            sub: cat || undefined,
            to: `/listings/${l.id}`,
          })
        }
      }
      if (predictors.status === 'fulfilled') {
        for (const p of predictors.value) {
          out.push({
            key: `pr-${p.name}`,
            group: 'Predictors',
            label: p.name,
            sub: p.display_name,
            to: `/predictors/${p.name}`,
          })
        }
      }
      setIndex(out)
    })
    return () => {
      cancelled = true
    }
  }, [open, domain, catField, entityGroup])

  const go = useCallback(
    (e: Entry) => {
      onClose()
      if (e.run) e.run()
      else if (e.to) navigate(e.to)
    },
    [navigate, onClose],
  )

  const statics = useMemo<Entry[]>(() => {
    const acts: Entry[] = [
      { key: 'act-new-run', group: 'Actions', label: 'New run', sub: 'train a model', to: '/new' },
      { key: 'act-new-ds', group: 'Actions', label: 'New dataset', sub: 'build a feature matrix', to: '/datasets/new' },
      {
        key: 'act-new-listing',
        group: 'Actions',
        label: `New ${domain.project.entity_noun}`,
        sub: 'manual entry',
        to: '/listings/new',
      },
    ]
    const views: Entry[] = [
      { key: 'v-runs', group: 'Views', label: 'Runs', sub: 'training runs', to: '/' },
      { key: 'v-datasets', group: 'Views', label: 'Datasets', to: '/datasets' },
      { key: 'v-definitions', group: 'Views', label: 'Definitions', to: '/definitions' },
      { key: 'v-models', group: 'Views', label: 'Models', sub: 'promoted registry', to: '/models' },
      { key: 'v-listings', group: 'Views', label: entityGroup, to: '/listings' },
      { key: 'v-compare', group: 'Views', label: 'Compare', to: '/compare' },
      { key: 'v-interp', group: 'Views', label: 'Interpretability', to: '/interpretability' },
    ]
    return [...acts, ...views]
  }, [domain, entityGroup])

  const results = useMemo(() => {
    const all = [...statics, ...index]
    const scored = all
      .map((e) => ({ e, s: score(e, q) }))
      .filter((x) => x.s > 0)
      .sort((x, y) => y.s - x.s)
    // Group, capped per group, in fixed group order.
    const byGroup = new Map<string, Entry[]>()
    for (const { e } of scored) {
      const g = byGroup.get(e.group) ?? []
      if (g.length < PER_GROUP) {
        g.push(e)
        byGroup.set(e.group, g)
      }
    }
    const order = ['Actions', 'Views', 'Runs', 'Models', 'Datasets', 'Definitions', entityGroup, 'Predictors']
    return order.filter((g) => byGroup.has(g)).map((g) => ({ group: g, entries: byGroup.get(g)! }))
  }, [statics, index, q, entityGroup])

  const flat = useMemo(() => results.flatMap((r) => r.entries), [results])

  // A new query resets the cursor: derived-state reset during render,
  // not an effect (same pattern as ListingImage).
  const [prevQ, setPrevQ] = useState(q)
  if (q !== prevQ) {
    setPrevQ(q)
    setActive(0)
  }

  // Keep the active row in view while arrowing.
  useEffect(() => {
    listRef.current
      ?.querySelector('[aria-selected="true"]')
      ?.scrollIntoView({ block: 'nearest' })
  }, [active])

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActive((i) => Math.min(i + 1, flat.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActive((i) => Math.max(i - 1, 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (flat[active]) go(flat[active])
    }
  }

  let flatIdx = -1
  return (
    <dialog
      ref={dialogRef}
      className="palette"
      aria-label="Command palette"
      onClose={onClose}
      onMouseDown={(e) => {
        // Backdrop click closes; clicks inside the panel don't.
        if (e.target === dialogRef.current) onClose()
      }}
    >
      <div className="palette-panel" onKeyDown={onKeyDown}>
        <input
          ref={inputRef}
          className="palette-input"
          type="text"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Jump to a run, model, dataset… or type an action"
          aria-label="Search everything"
          role="combobox"
          aria-expanded={flat.length > 0}
          aria-controls="palette-results"
          aria-activedescendant={flat[active] ? `palette-opt-${flat[active].key}` : undefined}
        />
        <ul className="palette-results" id="palette-results" role="listbox" ref={listRef}>
          {flat.length === 0 && (
            <li className="palette-empty" aria-live="polite">
              Nothing matches “{q}”
            </li>
          )}
          {results.map((g) => (
            <li key={g.group} className="palette-group">
              <span className="palette-group-label" aria-hidden>
                {g.group}
              </span>
              <ul role="presentation">
                {g.entries.map((e) => {
                  flatIdx++
                  const idx = flatIdx
                  return (
                    <li
                      key={e.key}
                      id={`palette-opt-${e.key}`}
                      role="option"
                      aria-selected={idx === active}
                      className={idx === active ? 'palette-opt is-active' : 'palette-opt'}
                      onMouseMove={() => setActive(idx)}
                      onClick={() => go(e)}
                    >
                      <span className="palette-opt-label">{e.label}</span>
                      {e.sub && <span className="palette-opt-sub">{e.sub}</span>}
                    </li>
                  )
                })}
              </ul>
            </li>
          ))}
        </ul>
        <div className="palette-foot" aria-hidden>
          <span>
            <kbd>↑</kbd>
            <kbd>↓</kbd> move
          </span>
          <span>
            <kbd>↵</kbd> open
          </span>
          <span>
            <kbd>esc</kbd> close
          </span>
        </div>
      </div>
    </dialog>
  )
}
