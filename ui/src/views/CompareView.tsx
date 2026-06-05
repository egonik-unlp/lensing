import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import type { Items, Metrics, Prediction, RunMeta } from '../api/types'
import ErrorHistogram from '../components/charts/ErrorHistogram'
import ScatterChart, { ScatterDensityLegend } from '../components/charts/ScatterChart'
import { scatterDensities } from '../components/charts/scatterDensity'
import { DatasetRef, PredictorRef, RunRef } from '../components/EntityRef'
import ViewHeader from '../components/ViewHeader'
import { useAsync } from '../hooks/useAsync'
import { useDomain } from '../lib/DomainContext'
import { loadItems } from '../lib/itemsCache'
import { fmtMoney, fmtPct, fmtR2, shortRunId } from '../lib/format'
import { scatterDomain } from '../lib/scale'
import { suspiciousRowIds } from '../lib/suspicious'
import './compare.css'

export default function CompareView() {
  const domain = useDomain()
  useEffect(() => {
    document.title = `Compare · ${domain.project.title}`
  }, [domain.project.title])
  const [params, setParams] = useSearchParams()
  const a = params.get('a')
  const b = params.get('b')
  const runs = useAsync(() => api.listRuns(), [])
  // Stopped runs carry full test metrics too — comparable like succeeded.
  const finished = useMemo(
    () => (runs.data ?? []).filter((r) => r.status === 'succeeded' || r.status === 'stopped'),
    [runs.data],
  )

  // Default any missing side once runs load (most recent finished first).
  useEffect(() => {
    if (!runs.data || finished.length < 2) return
    if (a && b) return
    const next = new URLSearchParams(params)
    if (!a) next.set('a', finished[0].run_id)
    if (!b) next.set('b', finished.find((r) => r.run_id !== (a ?? finished[0].run_id))?.run_id ?? '')
    setParams(next, { replace: true })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runs.data])

  const metaA = finished.find((r) => r.run_id === a) ?? null
  const metaB = finished.find((r) => r.run_id === b) ?? null

  const setSide = (side: 'a' | 'b', id: string) => {
    const next = new URLSearchParams(params)
    next.set(side, id)
    setParams(next, { replace: true })
  }
  const swap = () => {
    if (!a || !b) return
    const next = new URLSearchParams(params)
    next.set('a', b)
    next.set('b', a)
    setParams(next, { replace: true })
  }

  const ready = !runs.error && !runs.loading && finished.length >= 2

  return (
    <section aria-label="Compare two runs">
      <ViewHeader
        title="Compare"
        meta={
          ready ? (
            <div className="ab-pickers">
              <SidePicker side="A" value={metaA?.run_id ?? ''} runs={finished} exclude={metaB?.run_id} onChange={(id) => setSide('a', id)} />
              <button className="btn-on-chrome" onClick={swap} aria-label="Swap A and B" disabled={!a || !b}>
                ⇄ swap
              </button>
              <SidePicker side="B" value={metaB?.run_id ?? ''} runs={finished} exclude={metaA?.run_id} onChange={(id) => setSide('b', id)} />
            </div>
          ) : undefined
        }
      />
      <div className="view-body">
        {runs.error ? (
          <div className="error-block" role="alert">
            Could not load runs: {runs.error}{' '}
            <button className="btn" onClick={runs.reload}>
              Retry
            </button>
          </div>
        ) : runs.loading ? (
          <div className="skeleton skeleton-md" />
        ) : finished.length < 2 ? (
          <section className="empty-state" aria-label="Not enough runs to compare">
            <h1>Comparison needs two finished runs</h1>
            <p>
              {finished.length === 0
                ? 'No run has finished yet.'
                : 'Only one run has finished so far.'}{' '}
              Train at least two models on the same dataset, then put them side by side here.
            </p>
            <p style={{ marginTop: 'var(--sp-3)' }}>
              <Link to="/new" className="btn btn-primary">
                Start a run
              </Link>{' '}
              <Link to="/" className="btn">
                View runs
              </Link>
            </p>
          </section>
        ) : (
          metaA && metaB && <Comparison a={metaA} b={metaB} />
        )}
      </div>
    </section>
  )
}

function SidePicker({
  side,
  value,
  runs,
  exclude,
  onChange,
}: {
  side: 'A' | 'B'
  value: string
  runs: RunMeta[]
  exclude?: string
  onChange: (id: string) => void
}) {
  return (
    <label className={`side-picker side-${side.toLowerCase()}`}>
      <span className="side-tag">{side}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)} aria-label={`Run ${side}`}>
        {runs.map((r) => (
          <option key={r.run_id} value={r.run_id} disabled={r.run_id === exclude}>
            {r.predictor} · {shortRunId(r.run_id)} · {r.dataset_id.replace(/^ds-/, '')}
          </option>
        ))}
      </select>
    </label>
  )
}

/* ---------------- comparison body ---------------- */

interface MetricRow {
  key: keyof Metrics
  label: string
  /** true when lower values win */
  lowerWins: boolean
  fmt: (v: number) => string
}

const METRIC_ROWS: MetricRow[] = [
  { key: 'mae', label: 'MAE', lowerWins: true, fmt: fmtMoney },
  { key: 'rmse', label: 'RMSE', lowerWins: true, fmt: fmtMoney },
  { key: 'r2', label: 'R²', lowerWins: false, fmt: fmtR2 },
  { key: 'mape', label: 'MAPE', lowerWins: true, fmt: (v) => fmtPct(v, 0) },
  { key: 'medape', label: 'medAPE', lowerWins: true, fmt: (v) => fmtPct(v) },
]

function Comparison({ a, b }: { a: RunMeta; b: RunMeta }) {
  const predsA = useAsync(() => api.getPredictions(a.run_id), [a.run_id])
  const predsB = useAsync(() => api.getPredictions(b.run_id), [b.run_id])
  const sameDataset = a.dataset_id === b.dataset_id

  const domain = useMemo<[number, number] | undefined>(() => {
    if (!predsA.data || !predsB.data) return undefined
    const da = scatterDomain(predsA.data)
    const db = scatterDomain(predsB.data)
    return [Math.min(da[0], db[0]), Math.max(da[1], db[1])]
  }, [predsA.data, predsB.data])

  // One density scale for both charts: identical saturation means identical
  // crowding, so A-vs-B color reads honestly.
  const densityMax = useMemo(() => {
    if (!predsA.data || !predsB.data || !domain) return undefined
    return Math.max(
      scatterDensities(predsA.data, domain).max,
      scatterDensities(predsB.data, domain).max,
    )
  }, [predsA.data, predsB.data, domain])

  // Truth over comfort: a side whose predictions fail to load is an error
  // told plainly, never an empty chart pretending the side has no data.
  const predsError = predsA.error ?? predsB.error
  if (predsError) {
    return (
      <div className="error-block" role="alert">
        Could not load predictions for run{' '}
        <RunRef id={predsA.error ? a.run_id : b.run_id} />: {predsError}{' '}
        <button
          className="btn"
          onClick={() => {
            if (predsA.error) predsA.reload()
            if (predsB.error) predsB.reload()
          }}
        >
          Retry
        </button>
      </div>
    )
  }
  const predsLoading = predsA.loading || predsB.loading

  return (
    <>
      <table className="metrics-table" aria-label="Metric comparison">
        <thead>
          <tr>
            <th>Metric</th>
            <th className="num-col col-a">
              A · <PredictorRef name={a.predictor} /> <RunRef id={a.run_id} />
            </th>
            <th className="num-col col-b">
              B · <PredictorRef name={b.predictor} /> <RunRef id={b.run_id} />
            </th>
            <th className="num-col">Δ (B − A)</th>
          </tr>
        </thead>
        <tbody>
          {METRIC_ROWS.map((row) => (
            <MetricRowEl key={row.key} row={row} ma={a.metrics!} mb={b.metrics!} />
          ))}
        </tbody>
      </table>
      <p className="muted delta-note">
        ▲ marks the metric where B improves on A, ▼ where it gets worse. n ={' '}
        <span className="num">{a.metrics!.n_test.toLocaleString()}</span> test items
        {sameDataset ? ' (same dataset, identical split)' : ''}.
      </p>

      {!sameDataset && (
        <div className="dataset-warning" role="note">
          These runs trained on different datasets (<DatasetRef id={a.dataset_id} /> vs{' '}
          <DatasetRef id={b.dataset_id} />
          ). Their test sets differ, so metrics are not directly comparable and per-item
          disagreement is unavailable.
        </div>
      )}

      {predsLoading ? (
        <div className="cmp-section">
          <div className="skeleton skeleton-md" />
        </div>
      ) : (
        <>
          <div className="cmp-section">
            <h2 className="panel-title">Error distribution</h2>
            <ErrorHistogram
              series={[
                { label: `A · ${a.predictor}`, predictions: predsA.data ?? [], style: 'fill' },
                { label: `B · ${b.predictor}`, predictions: predsB.data ?? [], style: 'outline' },
              ]}
            />
          </div>

          <div className="cmp-scatters">
            <div>
              <h2 className="panel-title">
                <span className="series-chip chip-a">A</span> <PredictorRef name={a.predictor} impl />
              </h2>
              <ScatterChart predictions={predsA.data ?? []} domain={domain} densityMax={densityMax} legend={false} title={`Run A, ${a.predictor}`} />
            </div>
            <div>
              <h2 className="panel-title">
                <span className="series-chip chip-b">B</span> <PredictorRef name={b.predictor} impl />
              </h2>
              <ScatterChart predictions={predsB.data ?? []} domain={domain} densityMax={densityMax} legend={false} title={`Run B, ${b.predictor}`} />
            </div>
            {densityMax != null && (
              <div className="cmp-scatter-legend">
                <ScatterDensityLegend max={densityMax} />
              </div>
            )}
          </div>
        </>
      )}

      {sameDataset && predsA.data && predsB.data && (
        <Disagreements a={a} b={b} predsA={predsA.data} predsB={predsB.data} />
      )}
    </>
  )
}

function MetricRowEl({ row, ma, mb }: { row: MetricRow; ma: Metrics; mb: Metrics }) {
  const va = ma[row.key]
  const vb = mb[row.key]
  const delta = vb - va
  const bWins = row.lowerWins ? vb < va : vb > va
  const tie = va === vb
  const deltaStr =
    row.key === 'mae' || row.key === 'rmse'
      ? `${delta < 0 ? '−' : '+'}${fmtMoney(Math.abs(delta))}`
      : row.key === 'r2'
        ? `${delta < 0 ? '−' : '+'}${Math.abs(delta).toFixed(3)}`
        : `${delta < 0 ? '−' : '+'}${(Math.abs(delta) * 100).toFixed(1)}pp`
  return (
    <tr>
      <td>{row.label}</td>
      <td className={`num-col num${!tie && !bWins ? ' winner' : ''}`}>{row.fmt(va)}</td>
      <td className={`num-col num${!tie && bWins ? ' winner' : ''}`}>{row.fmt(vb)}</td>
      <td className="num-col num">
        {deltaStr}{' '}
        {!tie && (
          <span className={bWins ? 'delta-good' : 'delta-bad'} title={bWins ? 'B improves on A' : 'B is worse than A'}>
            {bWins ? '▲' : '▼'}
          </span>
        )}
      </td>
    </tr>
  )
}

/* ---------------- disagreements ---------------- */

interface DisRow {
  row_id: number
  actual: number
  predA: number
  predB: number
  spread: number
}

function Disagreements({
  a,
  b,
  predsA,
  predsB,
}: {
  a: RunMeta
  b: RunMeta
  predsA: Prediction[]
  predsB: Prediction[]
}) {
  const [limit, setLimit] = useState(25)
  const [expanded, setExpanded] = useState<number | null>(null)
  const [items, setItems] = useState<Items | null>(null)

  const rows = useMemo<DisRow[]>(() => {
    const byId = new Map(predsB.map((p) => [p.row_id, p]))
    const out: DisRow[] = []
    for (const pa of predsA) {
      const pb = byId.get(pa.row_id)
      if (!pb) continue
      out.push({
        row_id: pa.row_id,
        actual: pa.actual,
        predA: pa.predicted,
        predB: pb.predicted,
        spread: Math.abs(pa.predicted - pb.predicted),
      })
    }
    out.sort((x, y) => y.spread - x.spread)
    return out
  }, [predsA, predsB])

  // Rows where either model's % error is a robust outlier: usually a
  // mislabeled listing, not a real disagreement worth studying.
  const suspicious = useMemo(() => {
    const s = suspiciousRowIds(predsA)
    for (const id of suspiciousRowIds(predsB)) s.add(id)
    return s
  }, [predsA, predsB])

  const ensureItems = () => {
    if (items) return
    void loadItems(a.dataset_id).then(setItems, () => setItems({}))
  }

  const closer = (r: DisRow): 'A' | 'B' =>
    Math.abs(r.predA - r.actual) <= Math.abs(r.predB - r.actual) ? 'A' : 'B'

  return (
    <section className="cmp-section" aria-label="Largest disagreements">
      <h2 className="panel-title">
        Disagreements{' '}
        <span className="muted">
          where <PredictorRef name={a.predictor} /> and <PredictorRef name={b.predictor} /> differ
          most, and who lands closer
        </span>
      </h2>
      <table className="dis-table">
        <thead>
          <tr>
            <th>Item</th>
            <th className="num-col">Actual</th>
            <th className="num-col col-a">A predicts</th>
            <th className="num-col col-b">B predicts</th>
            <th className="num-col">Spread</th>
            <th>Closer</th>
            <th>
              <span className="sr-only">Details</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, limit).map((r) => (
            <DisRowEl
              key={r.row_id}
              r={r}
              closer={closer(r)}
              suspicious={suspicious.has(r.row_id)}
              item={items?.[String(r.row_id)] ?? null}
              expanded={expanded === r.row_id}
              onToggle={() => {
                ensureItems()
                setExpanded(expanded === r.row_id ? null : r.row_id)
              }}
            />
          ))}
        </tbody>
      </table>
      {limit < rows.length && (
        <button className="btn show-more" onClick={() => setLimit((l) => l + 25)}>
          Show 25 more of {(rows.length - limit).toLocaleString()}
        </button>
      )}
    </section>
  )
}

function DisRowEl({
  r,
  closer,
  suspicious,
  item,
  expanded,
  onToggle,
}: {
  r: DisRow
  closer: 'A' | 'B'
  suspicious: boolean
  item: Items[string] | null
  expanded: boolean
  onToggle: () => void
}) {
  const domain = useDomain()
  return (
    <>
      <tr className="pred-row" onClick={onToggle} aria-expanded={expanded}>
        <td className="num">
          {r.row_id}
          {suspicious && (
            <span
              className="sus-glyph"
              title={`Suspicious: robust outlier on % error for at least one model, likely a mislabeled ${domain.project.entity_noun}`}
            >
              {' '}
              ⚠
            </span>
          )}
        </td>
        <td className="num-col num">{fmtMoney(r.actual)}</td>
        <td className="num-col num">{fmtMoney(r.predA)}</td>
        <td className="num-col num">{fmtMoney(r.predB)}</td>
        <td className="num-col num">{fmtMoney(r.spread)}</td>
        <td>
          <span className={`series-chip ${closer === 'A' ? 'chip-a' : 'chip-b'}`}>{closer}</span>
        </td>
        <td className="expand-cell" aria-hidden>
          {expanded ? '▾' : '▸'}
        </td>
      </tr>
      {expanded && (
        <tr className="pred-detail">
          <td colSpan={7}>
            {item === null ? (
              <div className="skeleton skeleton-row" />
            ) : (
              <div className="item-detail">
                <p className="item-content">{item.content || <span className="muted">No description</span>}</p>
                <dl className="item-meta">
                  <div>
                    <dt>type</dt>
                    <dd>{item.propertyType}</dd>
                  </div>
                  <div>
                    <dt>bedrooms</dt>
                    <dd className="num">{item.bedrooms}</dd>
                  </div>
                  <div>
                    <dt>neighborhood</dt>
                    <dd>{item.neighborhood || '—'}</dd>
                  </div>
                  {item.cluster_label && (
                    <div>
                      <dt>cluster</dt>
                      <dd>{item.cluster_label}</dd>
                    </div>
                  )}
                </dl>
              </div>
            )}
          </td>
        </tr>
      )}
    </>
  )
}
