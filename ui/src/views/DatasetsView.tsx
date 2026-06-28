import { useMemo, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { ColumnDesc, Manifest } from '../api/types'
import DensityToggle from '../components/DensityToggle'
import LatticeMark from '../components/LatticeMark'
import ViewHeader from '../components/ViewHeader'
import { useDensity } from '../hooks/useDensity'
import { DatasetRef } from '../components/EntityRef'
import { useAsync } from '../hooks/useAsync'
import { fmtDateTime, fmtPct } from '../lib/format'
import './datasets.css'
import { useDocTitle } from '../lib/DomainContext'

type SortKey = 'created_at' | 'n_rows' | 'n_cols' | 'var' | 'runs' | 'models'

interface Sort {
  key: SortKey
  dir: 1 | -1
}

/** Column counts by kind, for the features breakdown cell. */
function kindCounts(columns: ColumnDesc[]): { pca: number; numeric: number; onehot: number } {
  const c = { pca: 0, numeric: 0, onehot: 0 }
  for (const col of columns) c[col.kind.type]++
  return c
}

const evrSum = (m: Manifest) => m.pca.explained_variance_ratio.reduce((a, b) => a + b, 0)

export default function DatasetsView() {
  useDocTitle('Datasets')
  const navigate = useNavigate()
  const datasets = useAsync(() => api.listDatasets(), [])
  const runs = useAsync(() => api.listRuns(), [])
  const models = useAsync(() => api.listModels(), [])
  const [sort, setSort] = useState<Sort>({ key: 'created_at', dir: -1 })
  const [focusIdx, setFocusIdx] = useState(0)
  const tableRef = useRef<HTMLTableElement>(null)
  const [density, toggleDensity] = useDensity()

  const runCount = useMemo(() => {
    const m = new Map<string, number>()
    for (const r of runs.data ?? []) m.set(r.dataset_id, (m.get(r.dataset_id) ?? 0) + 1)
    return m
  }, [runs.data])
  const modelCount = useMemo(() => {
    const m = new Map<string, number>()
    for (const r of models.data ?? []) m.set(r.dataset_id, (m.get(r.dataset_id) ?? 0) + 1)
    return m
  }, [models.data])

  const sorted = useMemo(() => {
    const list = datasets.data ? [...datasets.data] : []
    const val = (m: Manifest): number | string => {
      switch (sort.key) {
        case 'created_at':
          return m.created_at
        case 'n_rows':
          return m.n_rows
        case 'n_cols':
          return m.n_cols
        case 'var':
          return evrSum(m)
        case 'runs':
          return runCount.get(m.dataset_id) ?? 0
        case 'models':
          return modelCount.get(m.dataset_id) ?? 0
      }
    }
    list.sort((a, b) => {
      const av = val(a)
      const bv = val(b)
      return av < bv ? -sort.dir : av > bv ? sort.dir : 0
    })
    return list
  }, [datasets.data, sort, runCount, modelCount])

  const toggleSort = (key: SortKey) =>
    setSort((s) => ({ key, dir: s.key === key ? ((-s.dir) as 1 | -1) : key === 'created_at' ? -1 : 1 }))

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (sorted.length === 0) return
    if (e.key === 'j' || e.key === 'ArrowDown') {
      e.preventDefault()
      setFocusIdx((i) => Math.min(i + 1, sorted.length - 1))
    } else if (e.key === 'k' || e.key === 'ArrowUp') {
      e.preventDefault()
      setFocusIdx((i) => Math.max(i - 1, 0))
    } else if (e.key === 'Enter') {
      navigate(`/datasets/${sorted[focusIdx].dataset_id}`)
    }
  }

  const sortBtn = (key: SortKey, label: string) => (
    <button className="sort" onClick={() => toggleSort(key)} aria-label={`Sort by ${label}`}>
      {label}
      {sort.key === key && <span className="sort-arrow">{sort.dir === 1 ? '▲' : '▼'}</span>}
    </button>
  )

  return (
    <section aria-label="Datasets">
      <ViewHeader
        glyph="ds"
        title="Datasets"
        count={!datasets.loading && !datasets.error ? sorted.length : undefined}
        meta={
          <span>
            Frozen snapshots of the corpus: source filter, PCA, features and split. Keys:{' '}
            <kbd>j</kbd>/<kbd>k</kbd> move, <kbd>Enter</kbd> open.
          </span>
        }
        actions={
          <>
            <DensityToggle density={density} onToggle={toggleDensity} />
            <Link to="/datasets/new" className="btn btn-primary">
              Build dataset
            </Link>
          </>
        }
      />
      <div className="view-body">
        {datasets.error ? (
          <div className="error-block" role="alert">
            Could not load datasets: {datasets.error}{' '}
            <button className="btn" onClick={datasets.reload}>
              Retry
            </button>
          </div>
        ) : !datasets.loading && sorted.length === 0 ? (
          <EmptyDatasets />
        ) : (
      <table
        ref={tableRef}
        className="datasets-table"
        data-density={density}
        onKeyDown={onKeyDown}
        tabIndex={0}
      >
        <thead>
          <tr>
            <th>Dataset</th>
            <th className="num-col" aria-sort={sort.key === 'n_rows' ? (sort.dir === 1 ? 'ascending' : 'descending') : undefined}>
              {sortBtn('n_rows', 'Rows')}
            </th>
            <th className="num-col">{sortBtn('n_cols', 'Features')}</th>
            <th>Breakdown</th>
            <th className="num-col">{sortBtn('var', 'PCA var')}</th>
            <th>Split</th>
            <th>Target</th>
            <th className="num-col">Filtered</th>
            <th className="num-col">{sortBtn('runs', 'Runs')}</th>
            <th className="num-col">{sortBtn('models', 'Models')}</th>
            <th>{sortBtn('created_at', 'Created')}</th>
          </tr>
        </thead>
        <tbody>
          {datasets.loading
            ? Array.from({ length: 3 }, (_, i) => (
                <tr key={i}>
                  <td colSpan={11}>
                    <div className="skeleton skeleton-xs" />
                  </td>
                </tr>
              ))
            : sorted.map((m, i) => (
                <DatasetRow
                  key={m.dataset_id}
                  m={m}
                  focused={i === focusIdx}
                  runs={runCount.get(m.dataset_id) ?? 0}
                  models={modelCount.get(m.dataset_id) ?? 0}
                  onFocusRow={() => setFocusIdx(i)}
                />
              ))}
        </tbody>
      </table>
        )}
      </div>
    </section>
  )
}

function DatasetRow({
  m,
  focused,
  runs,
  models,
  onFocusRow,
}: {
  m: Manifest
  focused: boolean
  runs: number
  models: number
  onFocusRow: () => void
}) {
  const navigate = useNavigate()
  const k = kindCounts(m.columns)
  return (
    <tr
      className={`run-row${focused ? ' is-focused' : ''}`}
      onClick={(e) => {
        if ((e.target as HTMLElement).closest('a,button')) return
        onFocusRow()
        navigate(`/datasets/${m.dataset_id}`)
      }}
    >
      <td>
        <DatasetRef id={m.dataset_id} name={m.name} />
      </td>
      <td className="num-col num">{m.n_rows.toLocaleString()}</td>
      <td className="num-col num">{m.n_cols}</td>
      <td className="num breakdown-cell">
        pca {k.pca} · num {k.numeric} · oh {k.onehot}
      </td>
      <td className="num-col num">{fmtPct(evrSum(m), 0)}</td>
      <td className="num split-cell">
        test {fmtPct(m.split.test_ratio, 0)} · {m.split.n_train.toLocaleString()}/
        {m.split.n_test.toLocaleString()}
      </td>
      <td className="num">{m.target.transform === 'log1p' ? 'log1p' : 'raw'}</td>
      <td className="num-col num">
        {m.quality && m.quality.n_excluded_total > 0
          ? `−${m.quality.n_excluded_total.toLocaleString()}`
          : '—'}
      </td>
      <td className="num-col num">{runs || '—'}</td>
      <td className="num-col num">{models || '—'}</td>
      <td className="num">{fmtDateTime(m.created_at)}</td>
    </tr>
  )
}

function EmptyDatasets() {
  return (
    <section className="empty-state" aria-label="No datasets yet">
      <LatticeMark />
      <h1>No datasets yet</h1>
      <p>
        A dataset freezes everything downstream work depends on: a Qdrant pull under a recorded
        filter, a PCA reduction fit on the train rows, the feature encoding, and a reproducible
        train/test split. Runs train on one; models inherit its contract.
      </p>
      <p style={{ marginTop: 'var(--sp-3)' }}>
        <Link to="/datasets/new" className="btn btn-primary">
          Build first dataset
        </Link>
      </p>
    </section>
  )
}
