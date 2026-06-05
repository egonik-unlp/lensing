import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, pollJob } from '../api/client'
import type { ColumnDesc, DatasetSplit, ExportResult, Items, Manifest, RunMeta } from '../api/types'
import CategoryBars from '../components/charts/CategoryBars'
import Histogram from '../components/charts/Histogram'
import VarianceChart from '../components/charts/VarianceChart'
import { RedundancyFindings } from '../components/DatasetBuildForm'
import { DatasetRef, DefinitionRef, ModelRef, PredictorRef, RunRef } from '../components/EntityRef'
import { MetricCell } from '../components/MetricStrip'
import StatusBadge from '../components/StatusBadge'
import ViewHeader from '../components/ViewHeader'
import { useAsync } from '../hooks/useAsync'
import { loadItems, loadSplit } from '../lib/itemsCache'
import { fmtDateTime, fmtMoney, fmtPct, fmtR2, shortDatasetId } from '../lib/format'
import { ruleLabel } from '../lib/rules'
import { useDomain } from '../lib/DomainContext'
import { fieldLabel, type Domain } from '../lib/domain'
import { quantiles, topCategories } from '../lib/stats'
import { moneyTick } from '../lib/scale'
import './datasetdetail.css'

/** One corpus row: items.json entry + its id + split membership. */
interface Row {
  row_id: number
  price: number
  propertyType: string
  neighborhood: string
  city: string
  province: string
  bedrooms: number
  content: string
  cluster_label: string | null
  split: 'train' | 'test' | null
  /** Domain categorical field name → value, for the generic distributions. */
  cats: Record<string, string>
}

/** A selected EDA slice: a histogram bin range or a category value. */
type Slice =
  | { kind: 'price'; lo: number; hi: number }
  | { kind: 'bedrooms'; lo: number; hi: number }
  | { kind: 'category'; field: string; value: string }

export default function DatasetDetailView() {
  const domain = useDomain()
  const { id } = useParams<{ id: string }>()
  const manifest = useAsync(() => api.getDataset(id!), [id])

  const catFields = useMemo(
    () => domain.fields.filter((f) => f.role === 'categorical'),
    [domain],
  )

  useEffect(() => {
    if (id) document.title = `${shortDatasetId(id)} · ${domain.project.title}`
  }, [id, domain.project.title])

  // Items + split load independently of the manifest: the identity panels
  // paint immediately, the EDA panels hydrate when the corpus arrives. A
  // missing split degrades gracefully (no overlay), it doesn't error.
  const itemsAsync = useAsync(() => loadItems(id!), [id])
  const splitAsync = useAsync<DatasetSplit | null>(() => loadSplit(id!).catch(() => null), [id])
  const items: Items | null = itemsAsync.loading ? null : itemsAsync.data
  const itemsError = itemsAsync.error
  const split = splitAsync.loading ? null : splitAsync.data

  const rows = useMemo<Row[]>(() => {
    if (!items) return []
    const train = new Set(split?.train ?? [])
    const test = new Set(split?.test ?? [])
    return Object.entries(items).map(([rid, it]) => {
      const row_id = Number(rid)
      const cats: Record<string, string> = {}
      for (const f of catFields) {
        const v = it[f.name]
        cats[f.name] = typeof v === 'string' ? v : v != null ? String(v) : ''
      }
      return {
        row_id,
        price: it.price,
        propertyType: it.propertyType,
        neighborhood: it.neighborhood,
        city: it.city,
        province: it.province,
        bedrooms: it.bedrooms,
        content: it.content,
        cluster_label: it.cluster_label,
        split: train.has(row_id) ? 'train' : test.has(row_id) ? 'test' : null,
        cats,
      }
    })
  }, [items, split, catFields])

  if (manifest.error) {
    return (
      <>
        <ViewHeader glyph="ds" title="Dataset" crumbs={[{ label: 'Datasets', to: '/datasets' }]} />
        <div className="view-body">
          <div className="error-block" role="alert">
            Could not load dataset: {manifest.error}{' '}
            <button className="btn" onClick={manifest.reload}>
              Retry
            </button>{' '}
            <Link to="/datasets">Back to datasets</Link>
          </div>
        </div>
      </>
    )
  }
  if (manifest.loading || !manifest.data) {
    return (
      <>
        <ViewHeader glyph="ds" title="Dataset" crumbs={[{ label: 'Datasets', to: '/datasets' }]} />
        <div className="view-body">
          <div className="skeleton skeleton-lg" />
        </div>
      </>
    )
  }
  const m = manifest.data

  return (
    <section aria-label={`Dataset ${shortDatasetId(m.dataset_id)}`}>
      <ViewHeader
        glyph="ds"
        crumbs={[
          { label: 'Datasets', to: '/datasets' },
          { label: m.name ?? shortDatasetId(m.dataset_id) },
        ]}
        title={
          <>
            {m.name ?? 'Dataset'}{' '}
            <span className="run-id-frag">
              <DatasetRef id={m.dataset_id} self />
            </span>
          </>
        }
        meta={
          <span>
            <span className="num">{m.n_rows.toLocaleString()}</span> rows ·{' '}
            <span className="num">{m.n_cols}</span> features · created{' '}
            <span className="num">{fmtDateTime(m.created_at)}</span>
          </span>
        }
        actions={
          <Link to={`/new?dataset=${m.dataset_id}`} className="btn btn-primary">
            New run on it
          </Link>
        }
      />
      <div className="view-body">
        <DatasetHeader m={m} onRenamed={manifest.reload} />
        <Lineage datasetId={m.dataset_id} />
        {itemsError ? (
          <div className="error-block" role="alert">
            Could not load the corpus for exploration: {itemsError}
          </div>
        ) : (
          <Exploration rows={rows} loading={!items} hasSplit={!!split} m={m} domain={domain} catFields={catFields} />
        )}
        <FeatureInventory columns={m.columns} />
        {m.quality && <QualityReportPanel m={m} />}
        <ExportPanel m={m} />
      </div>
    </section>
  )
}

/* ---------------- A. header ---------------- */

function DatasetHeader({ m, onRenamed }: { m: Manifest; onRenamed: () => void }) {
  return (
    <header className="run-header">
      <dl className="run-meta">
        <div>
          <dt>Dataset ID</dt>
          <dd>
            <DatasetRef id={m.dataset_id} short={false} self copy />
          </dd>
        </div>
        <div>
          <dt>Source</dt>
          <dd className="num">{m.source.collection}</dd>
        </div>
        <div>
          <dt>Filter</dt>
          <dd className="num source-filter" title={m.source.filter}>
            {m.source.filter}
          </dd>
        </div>
        <div>
          <dt>Target</dt>
          <dd className="num">
            {m.target.field} · {m.target.transform === 'log1p' ? 'log1p' : 'raw'}
          </dd>
        </div>
        <div>
          <dt>Split</dt>
          <dd className="num">
            test {fmtPct(m.split.test_ratio, 0)} · seed {m.split.seed} ·{' '}
            {m.split.n_train.toLocaleString()} train / {m.split.n_test.toLocaleString()} test
          </dd>
        </div>
      </dl>
      <RenameAction m={m} onRenamed={onRenamed} />
    </header>
  )
}

/** Inline display-name rename, mirroring the definition Actions pattern. The
 *  slug `dataset_id` is unaffected, so lineage keeps working. */
function RenameAction({ m, onRenamed }: { m: Manifest; onRenamed: () => void }) {
  const [mode, setMode] = useState<'idle' | 'rename'>('idle')
  const [name, setName] = useState(m.name ?? '')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const trimmed = name.trim()
  const ok = trimmed.length > 0 && trimmed.length <= 80

  const submit = async () => {
    setBusy(true)
    setError(null)
    try {
      await api.renameDataset(m.dataset_id, trimmed)
      onRenamed()
      setMode('idle')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="def-actions">
      <button className="btn" onClick={() => setMode(mode === 'rename' ? 'idle' : 'rename')} disabled={busy}>
        Rename
      </button>
      {mode === 'rename' && (
        <div className="def-inline-form">
          <div className="hp-field">
            <label htmlFor="ds-new-name">Display name</label>
            <input
              id="ds-new-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={busy}
              autoFocus
              maxLength={80}
            />
          </div>
          <button className="btn" onClick={submit} disabled={!ok || busy}>
            {busy ? 'Saving…' : 'Save'}
          </button>
        </div>
      )}
      {error && (
        <span className="hp-error" role="alert">
          {error}
        </span>
      )}
    </div>
  )
}

/** Export the quality-filter survivors to a new Qdrant collection, source
 *  untouched. Pre-filled from the dataset's recorded quality config. */
function ExportPanel({ m }: { m: Manifest }) {
  const [suffix, setSuffix] = useState('')
  const [phase, setPhase] = useState<'idle' | 'running' | 'failed'>('idle')
  const [stage, setStage] = useState('')
  const [error, setError] = useState('')
  const [result, setResult] = useState<ExportResult | null>(null)
  const cancel = useRef<(() => void) | null>(null)

  useEffect(() => () => cancel.current?.(), [])

  const ok = /^[a-z0-9][a-z0-9-]{0,39}$/.test(suffix)

  const run = () => {
    if (!m.quality) return
    setPhase('running')
    setStage('starting')
    setError('')
    setResult(null)
    cancel.current?.()
    api
      .exportCollection({
        name_suffix: suffix,
        quality: m.quality.config,
        currency: m.currency?.config,
        source: m.source.collection,
      })
      .then(
        (jobId) => {
          cancel.current = pollJob(jobId, {
            onStage: setStage,
            onDone: (r) => {
              setResult(r as ExportResult)
              setPhase('idle')
            },
            onError: (msg) => {
              setError(msg)
              setPhase('failed')
            },
          })
        },
        (e: unknown) => {
          setError(e instanceof Error ? e.message : String(e))
          setPhase('failed')
        },
      )
  }

  return (
    <div className="panel">
      <h2 className="section-title">
        Export clean collection{' '}
        <span className="muted num">survivors of this dataset&rsquo;s quality filters → a new Qdrant collection</span>
      </h2>
      {!m.quality ? (
        <p className="muted">This dataset has no recorded quality config to replay.</p>
      ) : (
        <>
          <div className="quality-preview-row export-row">
            <div className="hp-field">
              <label htmlFor="export-suffix">collection name</label>
              <div className="export-name">
                <span className="num muted">properties-tagged-clean-</span>
                <input
                  id="export-suffix"
                  type="text"
                  className="mono-input"
                  value={suffix}
                  onChange={(e) => setSuffix(e.target.value)}
                  disabled={phase === 'running'}
                  placeholder="suffix"
                />
              </div>
              {suffix && !ok && (
                <span className="hp-hint">lowercase letters, digits and dashes; max 40</span>
              )}
            </div>
            <button className="btn" onClick={run} disabled={!ok || phase === 'running'}>
              {phase === 'running' ? 'Exporting…' : 'Export'}
            </button>
            {phase === 'running' && (
              <span className="muted num" role="status">
                {stage}
              </span>
            )}
            {phase === 'failed' && (
              <span className="hp-error" role="alert">
                {error}
              </span>
            )}
          </div>
          {result && (
            <p className="num" role="status">
              Wrote {result.n_written.toLocaleString()} of {result.n_source.toLocaleString()} points to{' '}
              <span className="mono">{result.collection}</span> ({result.n_excluded.toLocaleString()} excluded).
            </p>
          )}
        </>
      )}
    </div>
  )
}

/* ---------------- B. lineage ---------------- */

function Lineage({ datasetId }: { datasetId: string }) {
  const runs = useAsync(() => api.listRuns(), [])
  const models = useAsync(() => api.listModels(), [])
  const definitions = useAsync(() => api.listDefinitions(), [])

  const myRuns = useMemo(
    () => (runs.data ?? []).filter((r) => r.dataset_id === datasetId),
    [runs.data, datasetId],
  )
  const myModels = useMemo(
    () => (models.data ?? []).filter((r) => r.dataset_id === datasetId),
    [models.data, datasetId],
  )
  const myDefs = useMemo(
    () => (definitions.data ?? []).filter((d) => d.dataset_tags.includes(datasetId)),
    [definitions.data, datasetId],
  )

  if (runs.loading) return <div className="skeleton" style={{ height: '3rem' }} />

  return (
    <div className="panel lineage" aria-label="Dataset lineage">
      <h2 className="section-title">
        Lineage{' '}
        <span className="muted lineage-counts num">
          {myRuns.length} {myRuns.length === 1 ? 'run' : 'runs'} · {myModels.length}{' '}
          {myModels.length === 1 ? 'model' : 'models'} · {myDefs.length}{' '}
          {myDefs.length === 1 ? 'definition' : 'definitions'}
        </span>
      </h2>
      {myRuns.length === 0 ? (
        <p className="muted">
          Nothing has trained on this dataset yet.{' '}
          <Link to={`/new?dataset=${datasetId}`}>Start a run on it</Link>.
        </p>
      ) : (
        <table className="lineage-table">
          <thead>
            <tr>
              <th>Run</th>
              <th>Predictor</th>
              <th>Definition</th>
              <th>Status</th>
              <th className="num-col">MAE</th>
              <th className="num-col">R²</th>
              <th>Started</th>
            </tr>
          </thead>
          <tbody>
            {myRuns.map((r: RunMeta) => (
              <tr key={r.run_id}>
                <td>
                  <RunRef id={r.run_id} />
                </td>
                <td>
                  <PredictorRef name={r.predictor} impl />
                </td>
                <td>{r.from_definition ? <DefinitionRef name={r.from_definition} /> : <span className="muted">—</span>}</td>
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
      {(myModels.length > 0 || myDefs.length > 0) && (
        <dl className="run-meta lineage-refs">
          {myModels.length > 0 && (
            <div>
              <dt>Models promoted from it</dt>
              <dd className="lineage-list">
                {myModels.map((mr) => (
                  <ModelRef key={mr.name} name={mr.name} />
                ))}
              </dd>
            </div>
          )}
          {myDefs.length > 0 && (
            <div>
              <dt>Definitions tagging it</dt>
              <dd className="lineage-list">
                {myDefs.map((d) => (
                  <DefinitionRef key={d.name} name={d.name} />
                ))}
              </dd>
            </div>
          )}
        </dl>
      )}
    </div>
  )
}

/* ---------------- C–F. EDA: stats, distributions, categories ---------------- */

function Exploration({
  rows,
  loading,
  hasSplit,
  m,
  domain,
  catFields,
}: {
  rows: Row[]
  loading: boolean
  hasSplit: boolean
  m: Manifest
  domain: Domain
  catFields: Domain['fields']
}) {
  const [overlay, setOverlay] = useState(false)
  const [slice, setSlice] = useState<Slice | null>(null)

  const prices = useMemo(() => rows.map((r) => r.price).filter((p) => p > 0), [rows])
  const [p10, p50, p90] = useMemo(() => quantiles(prices, [0.1, 0.5, 0.9]), [prices])
  const bedrooms = useMemo(() => rows.map((r) => r.bedrooms).filter((b) => b >= 0), [rows])
  const meanBedrooms = useMemo(
    () => (bedrooms.length ? bedrooms.reduce((a, b) => a + b, 0) / bedrooms.length : 0),
    [bedrooms],
  )

  // Per categorical field: the top values by row count, with median target.
  // The first two are surfaced prominently; the rest (often messier fields,
  // off by default in features) collapse under a details disclosure.
  const byCat = useMemo(
    () =>
      catFields.map((f) => ({
        field: f,
        data: topCategories(rows.map((r) => ({ key: r.cats[f.name] ?? '', price: r.price })), 12),
      })),
    [rows, catFields],
  )
  const primaryCats = byCat.slice(0, 2)
  const extraCats = byCat.slice(2)
  const targetNoun = domain.project.target_noun

  const priceSeries = useMemo(() => {
    if (overlay && hasSplit) {
      return [
        { label: 'train', values: rows.filter((r) => r.split === 'train').map((r) => r.price), style: 'fill' as const },
        { label: 'test', values: rows.filter((r) => r.split === 'test').map((r) => r.price), style: 'outline' as const },
      ]
    }
    return [{ label: 'corpus', values: prices, style: 'fill' as const }]
  }, [rows, prices, overlay, hasSplit])

  // Chart domain caps at p99.5: a handful of 50-bedroom typos would
  // otherwise own the axis. The slice drill-down still reaches them.
  const bedCap = useMemo(
    () => Math.max(quantiles(bedrooms, [0.995])[0] || 1, 1),
    [bedrooms],
  )
  const bedroomSeries = useMemo(() => {
    const chartable = (rs: Row[]) => rs.map((r) => r.bedrooms).filter((b) => b >= 0 && b <= bedCap)
    if (overlay && hasSplit) {
      return [
        { label: 'train', values: chartable(rows.filter((r) => r.split === 'train')), style: 'fill' as const },
        { label: 'test', values: chartable(rows.filter((r) => r.split === 'test')), style: 'outline' as const },
      ]
    }
    return [{ label: 'corpus', values: bedrooms.filter((b) => b <= bedCap), style: 'fill' as const }]
  }, [rows, bedrooms, bedCap, overlay, hasSplit])

  if (loading) {
    return (
      <div className="panel">
        <h2 className="section-title">Exploring the corpus</h2>
        <div className="skeleton" style={{ height: '14rem', marginTop: 'var(--sp-3)' }} />
      </div>
    )
  }

  return (
    <>
      <div className="panel">
        <div className="eda-head">
          <h2 className="panel-title">Distributions</h2>
          {hasSplit && (
            <label className="toggle overlay-toggle">
              <input type="checkbox" checked={overlay} onChange={(e) => setOverlay(e.target.checked)} />
              train vs test overlay
            </label>
          )}
        </div>

        <div className="metrics-strip eda-strip" role="group" aria-label="Corpus summary statistics">
          <Stat label={`median ${targetNoun}`} value={fmtMoney(p50)} hint="p50 over the corpus" />
          <Stat label="p10" value={fmtMoney(p10)} hint="cheap tail" />
          <Stat label="p90" value={fmtMoney(p90)} hint="expensive tail" />
          <Stat label="bedrooms" value={meanBedrooms.toFixed(1)} hint="mean" />
          {primaryCats[0] && (
            <Stat
              label={`${fieldLabel(primaryCats[0].field)}s`}
              value={String(primaryCats[0].data.length)}
              hint="distinct values"
            />
          )}
          <Stat label="rows" value={rows.length.toLocaleString()} hint={`target: ${m.target.field}`} />
        </div>

        <div className="eda-grid">
          <div>
            <h3 className="panel-title">Price</h3>
            <Histogram
              series={priceSeries}
              log
              bins={48}
              xLabel={`${targetNoun}, log scale`}
              tickFormat={moneyTick}
              markers={[
                { value: p50, label: 'median' },
                { value: p10, label: 'p10' },
                { value: p90, label: 'p90' },
              ]}
              onBinClick={(lo, hi) => setSlice({ kind: 'price', lo, hi })}
              selected={slice?.kind === 'price' ? [slice.lo, slice.hi] : null}
            />
          </div>
          <div>
            <h3 className="panel-title">Bedrooms</h3>
            <Histogram
              series={bedroomSeries}
              bins={Math.min(16, Math.ceil(bedCap) + 1)}
              xLabel="bedrooms"
              tickFormat={(v) => String(Math.round(v))}
              onBinClick={(lo, hi) => setSlice({ kind: 'bedrooms', lo, hi })}
              selected={slice?.kind === 'bedrooms' ? [slice.lo, slice.hi] : null}
            />
          </div>
        </div>
      </div>

      <div className="panel">
        <h2 className="panel-title">Categories</h2>
        <p className="muted eda-blurb">
          Click a bar or a category to inspect the {domain.project.entity_noun_plural} behind it.
        </p>
        <div className="eda-grid">
          {primaryCats.map(({ field, data }) => (
            <div key={field.name}>
              <h3 className="panel-title">{fieldLabel(field)} (top 12)</h3>
              <CategoryBars
                data={data}
                ariaLabel={`Rows and median ${targetNoun} by ${fieldLabel(field)}`}
                onSelect={(value) =>
                  setSlice(
                    slice?.kind === 'category' && slice.field === field.name && slice.value === value
                      ? null
                      : { kind: 'category', field: field.name, value },
                  )
                }
                selected={slice?.kind === 'category' && slice.field === field.name ? slice.value : null}
              />
            </div>
          ))}
        </div>
        {extraCats.length > 0 && (
          <details className="messy-fields">
            <summary>
              {extraCats.map((c) => fieldLabel(c.field)).join(', ')} (messier fields, often off by
              default in features)
            </summary>
            <div className="eda-grid">
              {extraCats.map(({ field, data }) => (
                <div key={field.name}>
                  <h3 className="panel-title">{fieldLabel(field)} (top 12)</h3>
                  <CategoryBars
                    data={data}
                    ariaLabel={`Rows and median ${targetNoun} by ${fieldLabel(field)}`}
                    onSelect={(value) =>
                      setSlice(
                        slice?.kind === 'category' && slice.field === field.name && slice.value === value
                          ? null
                          : { kind: 'category', field: field.name, value },
                      )
                    }
                    selected={slice?.kind === 'category' && slice.field === field.name ? slice.value : null}
                  />
                </div>
              ))}
            </div>
          </details>
        )}
        {slice && <SlicePanel rows={rows} slice={slice} onClear={() => setSlice(null)} />}
      </div>

      <div className="panel">
        <h2 className="panel-title">PCA explained variance</h2>
        <p className="muted eda-blurb num">
          {m.pca.dims} components over {m.pca.components_shape[1]}-dim embeddings ·{' '}
          {fmtPct(m.pca.explained_variance_ratio.reduce((a, b) => a + b, 0))} captured
        </p>
        {m.cumulative_evr && m.cumulative_evr.length > m.pca.dims ? (
          <>
            {/* Stored as a cumulative curve; VarianceChart wants per-component
                values (it re-cumulates), so undo the running sum. */}
            <VarianceChart
              evr={m.cumulative_evr.map((v, i) => (i === 0 ? v : v - m.cumulative_evr![i - 1]))}
              markK={m.pca.dims}
            />
            <p className="muted eda-blurb num">
              full spectrum (train split); the guide marks the {m.pca.dims} kept components
            </p>
          </>
        ) : (
          <VarianceChart evr={m.pca.explained_variance_ratio} />
        )}
        {m.redundancy && <RedundancyFindings report={m.redundancy} />}
      </div>
    </>
  )
}

/** Corpus summary stat: the shared MetricCell vocabulary, never clickable. */
function Stat({ label, value, hint }: { label: string; value: string; hint: string }) {
  return <MetricCell label={label} value={value} hint={hint} />
}

/* ---------------- slice drill-down ---------------- */

const SLICE_LIMIT = 20

function sliceLabel(slice: Slice): string {
  switch (slice.kind) {
    case 'price':
      return `price ${fmtMoney(slice.lo)} – ${fmtMoney(slice.hi)}`
    case 'bedrooms':
      return `bedrooms ${Math.ceil(slice.lo)} – ${Math.floor(slice.hi)}`
    case 'category':
      return `${slice.field} = ${slice.value}`
  }
}

function matches(r: Row, slice: Slice): boolean {
  switch (slice.kind) {
    case 'price':
      return r.price >= slice.lo && r.price < slice.hi
    case 'bedrooms':
      return r.bedrooms >= slice.lo && r.bedrooms < slice.hi
    case 'category': {
      const v = (r.cats[slice.field] ?? '').trim() || '(empty)'
      return v === slice.value
    }
  }
}

/** Drill-down one gesture away: the listings inside the selected slice. */
function SlicePanel({ rows, slice, onClear }: { rows: Row[]; slice: Slice; onClear: () => void }) {
  const matched = useMemo(() => {
    const out = rows.filter((r) => matches(r, slice))
    out.sort((a, b) => b.price - a.price)
    return out
  }, [rows, slice])

  return (
    <div className="slice-panel" role="region" aria-label={`Items in slice ${sliceLabel(slice)}`}>
      <div className="slice-head">
        <h3 className="panel-title">
          <span className="num">{sliceLabel(slice)}</span>{' '}
          <span className="muted num">
            {matched.length.toLocaleString()} {matched.length === 1 ? 'listing' : 'listings'}
            {matched.length > SLICE_LIMIT ? `, showing ${SLICE_LIMIT} priciest` : ''}
          </span>
        </h3>
        <button className="btn" onClick={onClear}>
          Clear slice
        </button>
      </div>
      <table className="pred-table">
        <thead>
          <tr>
            <th>Item</th>
            <th className="num-col">Price</th>
            <th>Type</th>
            <th>Neighborhood</th>
            <th className="num-col">Bedrooms</th>
            <th>Split</th>
          </tr>
        </thead>
        <tbody>
          {matched.slice(0, SLICE_LIMIT).map((r) => (
            <SliceRow key={r.row_id} r={r} />
          ))}
        </tbody>
      </table>
    </div>
  )
}

function SliceRow({ r }: { r: Row }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <>
      <tr className="pred-row" onClick={() => setExpanded((v) => !v)} aria-expanded={expanded}>
        <td className="num">{r.row_id}</td>
        <td className="num-col num">{fmtMoney(r.price)}</td>
        <td>{r.propertyType || '—'}</td>
        <td>{r.neighborhood || '—'}</td>
        <td className="num-col num">{r.bedrooms}</td>
        <td className="num split-tag">{r.split ?? '—'}</td>
      </tr>
      {expanded && (
        <tr className="pred-detail">
          <td colSpan={6}>
            <div className="item-detail">
              <p className="item-content">{r.content || <span className="muted">No description</span>}</p>
              <dl className="item-meta">
                <div>
                  <dt>city</dt>
                  <dd>{r.city || '—'}</dd>
                </div>
                {r.cluster_label && (
                  <div>
                    <dt>cluster</dt>
                    <dd>{r.cluster_label}</dd>
                  </div>
                )}
              </dl>
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

/* ---------------- H. feature inventory ---------------- */

function FeatureInventory({ columns }: { columns: ColumnDesc[] }) {
  const groups = useMemo(() => {
    let pca = 0
    const numeric: string[] = []
    const onehot = new Map<string, string[]>()
    for (const c of columns) {
      if (c.kind.type === 'pca') pca++
      else if (c.kind.type === 'numeric') numeric.push(c.kind.field)
      else {
        const g = onehot.get(c.kind.group)
        if (g) g.push(c.kind.value)
        else onehot.set(c.kind.group, [c.kind.value])
      }
    }
    return { pca, numeric, onehot }
  }, [columns])

  return (
    <div className="panel">
      <h2 className="section-title">
        Feature inventory <span className="muted num">{columns.length} columns, exactly what the model sees</span>
      </h2>
      <dl className="run-meta inventory">
        <div>
          <dt>PCA</dt>
          <dd className="num">
            {groups.pca} components (pca_0 … pca_{groups.pca - 1})
          </dd>
        </div>
        {groups.numeric.length > 0 && (
          <div>
            <dt>Numeric ({groups.numeric.length})</dt>
            <dd className="inventory-values">
              {groups.numeric.map((f) => (
                <span key={f} className="chip num">
                  {f}
                </span>
              ))}
            </dd>
          </div>
        )}
        {[...groups.onehot.entries()].map(([group, values]) => (
          <div key={group} className="inventory-group">
            <dt>
              {group} <span className="num">×{values.length} one-hot</span>
            </dt>
            <dd className="inventory-values">
              {values.map((v) => (
                <span key={v} className="chip num">
                  {v}
                </span>
              ))}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

/* ---------------- I. quality report ---------------- */

function QualityReportPanel({ m }: { m: Manifest }) {
  const domain = useDomain()
  const q = m.quality!
  const cur = m.currency
  return (
    <div className="panel">
      <h2 className="section-title">
        Data quality{' '}
        <span className="muted num">{q.n_excluded_total.toLocaleString()} rows excluded at build time</span>
      </h2>
      <ul className="quality-counts">
        {q.rules.map((r) => (
          <li key={r.rule} className="num">
            {ruleLabel(r.rule, domain)}: {r.n_flagged.toLocaleString()} flagged
            {r.n_excluded > 0 ? ', excluded' : <span className="muted"> (rule off, kept)</span>}
          </li>
        ))}
      </ul>
      {cur && cur.config.mode !== 'off' && (
        <p className="num">
          Currency: {cur.config.mode === 'filter' ? `kept ${cur.config.keep} only` : `converted to ${cur.config.keep}`}
          {cur.config.reconcile_collection
            ? ` · reconciled from ${cur.config.reconcile_collection}`
            : ''}
          {cur.n_converted > 0 &&
            ` · ${cur.n_converted.toLocaleString()} converted @ ${Math.round(cur.rate_min ?? 0)}–${Math.round(cur.rate_max ?? 0)} ${cur.config.rate_source}`}
          {cur.n_missing > 0 && ` · ${cur.n_missing.toLocaleString()} without currency (kept)`}
        </p>
      )}
    </div>
  )
}
