import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api/client'
import type { Items, Metrics, Prediction, Predictor, RunMeta } from '../api/types'
import ArchViz from '../components/ArchViz'
import BlendPanel from '../components/BlendPanel'
import ConfusionMatrix from '../components/charts/ConfusionMatrix'
import ErrorHistogram from '../components/charts/ErrorHistogram'
import LossChart from '../components/charts/LossChart'
import LossDerivativeChart from '../components/charts/LossDerivativeChart'
import ScatterChart from '../components/charts/ScatterChart'
import { DatasetRef, DefinitionRef, ModelRef, PredictorRef, RunRef } from '../components/EntityRef'
import ItemMeta from '../components/ItemMeta'
import LogPane from '../components/LogPane'
import MetricStrip from '../components/MetricStrip'
import StatusBadge from '../components/StatusBadge'
import ViewHeader from '../components/ViewHeader'
import { useAsync } from '../hooks/useAsync'
import { useRunEvents } from '../hooks/useRunEvents'
import { loadItems } from '../lib/itemsCache'
import { computeMetrics, suspiciousRowIds } from '../lib/suspicious'
import {
  fmtDateTime,
  fmtDuration,
  fmtTarget,
  fmtTargetDelta,
  fmtSignedPct,
  shortRunId,
} from '../lib/format'
import { formatMetric, isPercentMetric, metricColumns, metricValue } from '../lib/metrics'
import { domainTask, isClassification } from '../lib/domain'
import './rundetail.css'
import { useDocTitle, useDomain } from '../lib/DomainContext'

export default function RunDetailView() {
  const { runId } = useParams<{ runId: string }>()
  const { data: run, error, loading, reload } = useAsync(() => api.getRun(runId!), [runId])
  const predictors = useAsync(() => api.listPredictors(), [])
  const events = useRunEvents(runId ?? null)

  // The SSE stream ends with a status event once the run finishes: refetch meta.
  useEffect(() => {
    if (events.terminal && run?.status === 'running') reload()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [events.terminal])

  useDocTitle(run ? shortRunId(run.run_id) : null)

  if (error) {
    return (
      <>
        <ViewHeader glyph="run" title="Run" crumbs={[{ label: 'Runs', to: '/' }]} />
        <div className="view-body">
          <div className="error-block" role="alert">
            Could not load run: {error}{' '}
            <button className="btn" onClick={reload}>
              Retry
            </button>{' '}
            <Link to="/">Back to runs</Link>
          </div>
        </div>
      </>
    )
  }
  if (loading || !run) {
    return (
      <>
        <ViewHeader glyph="run" title="Run" crumbs={[{ label: 'Runs', to: '/' }]} />
        <div className="view-body">
          <div className="skeleton skeleton-lg" />
        </div>
      </>
    )
  }

  const pred = predictors.data?.find((p) => p.name === run.predictor)
  const supportsPredict = !!pred?.predict_args
  // Promotable from the last periodic checkpoint even though the run died.
  const checkpointPromotable =
    (run.status === 'failed' || run.status === 'interrupted') &&
    supportsPredict &&
    run.has_checkpoint

  return (
    <section aria-label={`Run ${shortRunId(run.run_id)}`}>
      <ViewHeader
        glyph="run"
        crumbs={[{ label: 'Runs', to: '/' }, { label: shortRunId(run.run_id) }]}
        title={
          <>
            <PredictorRef name={run.predictor} impl />{' '}
            <span className="run-id-frag">
              <RunRef id={run.run_id} self />
            </span>
          </>
        }
        meta={
          <>
            <StatusBadge status={run.status} />
            <span>
              started <span className="num">{fmtDateTime(run.started_at)}</span>
            </span>
            <span>
              took <span className="num">{fmtDuration(run.started_at, run.finished_at)}</span>
            </span>
          </>
        }
        actions={run.status !== 'running' ? <DeleteRunButton run={run} /> : undefined}
      />
      <div className="view-body">
      <RunHeader run={run} />
      {run.status === 'running' && (
        <>
          <StopControls run={run} predictor={pred} stopping={events.stopping} />
          <div className="live-grid">
            <div>
              <h2 className="panel-title">
                Loss{' '}
                {events.connection === 'reconnecting' && (
                  <span className="reconnect" role="status">
                    connection lost, retrying…
                  </span>
                )}
              </h2>
              <LossChart epochs={events.epochs} totalEpochs={events.epochs[0]?.total} live />
              {events.epochs.length >= 3 && (
                <>
                  <h2 className="panel-title dynamics-title">Is it still learning?</h2>
                  <LossDerivativeChart epochs={events.epochs} />
                </>
              )}
            </div>
            <div>
              <h2 className="panel-title">Log</h2>
              <LogPane lines={events.logs} />
            </div>
          </div>
          <RunArch run={run} canFallback={!!pred?.visualization} />
        </>
      )}
      {run.status === 'failed' && (
        <div className="error-block" role="alert">
          <strong>Run failed</strong>
          {run.exit_code !== null && (
            <>
              {' '}
              — predictor exited with code <span className="num">{run.exit_code}</span>
            </>
          )}
          .{' '}
          {checkpointPromotable
            ? 'A periodic checkpoint was saved before it died, so the params as they were can still be promoted below.'
            : 'Fix the cause and start a new run; run directories are immutable.'}
          {run.stderr_tail && <pre>{run.stderr_tail}</pre>}
          {events.logs.length > 0 && (
            <>
              <h2 className="panel-title">Log</h2>
              <LogPane lines={events.logs} />
            </>
          )}
        </div>
      )}
      {run.status === 'interrupted' && (
        <div className="error-block" role="alert">
          <strong>Run interrupted</strong> — the server restarted while this run was in
          progress.{' '}
          {checkpointPromotable
            ? 'A periodic checkpoint survived, so the params as they were can still be promoted below.'
            : 'Its results are incomplete; start a new run with the same configuration.'}
        </div>
      )}
      {checkpointPromotable && <PromotePanel run={run} checkpointOnly />}
      {(run.status === 'succeeded' || run.status === 'stopped') && (
        <FinishedRun run={run} events={events} />
      )}
      </div>
    </section>
  )
}

/** Native architecture diagram from hyperparams + the dataset's feature
 *  counts; falls back to the predictor-exported viz.svg if the dataset is
 *  gone (manifest unavailable) or the family is unknown. */
function RunArch({ run, canFallback }: { run: RunMeta; canFallback: boolean }) {
  const manifest = useAsync(() => api.getDataset(run.dataset_id), [run.dataset_id])
  if (manifest.loading) return null
  const m = manifest.data
  const features = m
    ? {
        nCols: m.n_cols,
        nPca: m.columns.filter((c) => c.kind.type === 'pca').length,
        onehotGroups: [
          ...new Set(
            m.columns.flatMap((c) => (c.kind.type === 'onehot' ? [c.kind.group] : [])),
          ),
        ],
      }
    : null
  if (run.predictor === 'blend') {
    return (
      <BlendPanel
        load={() => api.runBlend(run.run_id)}
        hyperparams={run.hyperparams}
        features={features}
        blendMetrics={run.metrics}
      />
    )
  }
  return (
    <ArchViz
      predictor={run.predictor}
      hyperparams={run.hyperparams}
      features={features}
      fallbackUrl={canFallback ? api.runVizUrl(run.run_id) : undefined}
    />
  )
}

/* ---------------- stop a running run ---------------- */

function StopControls({
  run,
  predictor,
  stopping,
}: {
  run: RunMeta
  predictor: Predictor | undefined
  stopping: boolean
}) {
  const [requested, setRequested] = useState(false)
  const [forceRequested, setForceRequested] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const supportsStop = !!predictor?.supports_stop

  const stop = async (force: boolean) => {
    // First force on a STOP-honoring predictor still saves: the server kills
    // only if the process is alive after the grace window. A repeat force
    // (or any force on a predictor that ignores STOP) kills immediately.
    const kills = force && (forceRequested || !supportsStop)
    if (
      force &&
      !window.confirm(
        kills
          ? 'This kills the training process now. Unless a periodic checkpoint was saved ' +
              '(checkpoint_every), the run cannot be promoted. Continue?'
          : 'Force stop asks the run to evaluate and save now, and kills it only if it is ' +
              'still alive after 60 seconds. Continue?',
      )
    ) {
      return
    }
    setBusy(true)
    setError(null)
    try {
      await api.stopRun(run.run_id, force)
      setRequested(true)
      if (force) setForceRequested(true)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
    setBusy(false)
  }

  return (
    <div className="stop-row">
      {stopping ? (
        <span className="muted" role="status">
          stopping — evaluating and saving the model with the params as they are…
        </span>
      ) : (
        <>
          {supportsStop && (
            <button className="btn" onClick={() => void stop(false)} disabled={busy || requested}>
              {requested ? 'Stop requested…' : 'Stop training'}
            </button>
          )}
          {(requested || !supportsStop) && (
            <button className="btn btn-danger" onClick={() => void stop(true)} disabled={busy}>
              {forceRequested ? 'Kill now' : 'Force stop'}
            </button>
          )}
          <span className="muted">
            {supportsStop
              ? requested
                ? 'finishing the current epoch, then evaluating and saving — the run stays promotable'
                : 'finishes the current epoch, evaluates and saves a model from the params as they are'
              : 'this predictor cannot stop gracefully; force stop kills the process'}
          </span>
        </>
      )}
      {error && (
        <span className="hp-error" role="alert">
          {error}
        </span>
      )}
    </div>
  )
}

function DeleteRunButton({ run }: { run: RunMeta }) {
  const navigate = useNavigate()
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const remove = async () => {
    if (
      !window.confirm(
        `Delete run ${shortRunId(run.run_id)}? Its directory (metrics, predictions, ` +
          'checkpoint) is removed. Models promoted from it are untouched.',
      )
    )
      return
    setDeleting(true)
    setError(null)
    try {
      await api.deleteRun(run.run_id)
      navigate('/')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setDeleting(false)
    }
  }

  return (
    <>
      <button className="btn-on-chrome" onClick={() => void remove()} disabled={deleting}>
        {deleting ? 'Deleting…' : 'Delete run'}
      </button>
      {error && (
        <span className="hp-error" role="alert">
          {error}
        </span>
      )}
    </>
  )
}

function RunHeader({ run }: { run: RunMeta }) {
  const hp = Object.entries(run.hyperparams)
  return (
    <header className="run-header">
      <dl className="run-meta">
        <div>
          <dt>Run ID</dt>
          <dd>
            <RunRef id={run.run_id} short={false} self copy />
          </dd>
        </div>
        <div>
          <dt>Dataset</dt>
          <dd>
            <DatasetRef id={run.dataset_id} short={false} />
          </dd>
        </div>
        {run.from_definition && (
          <div>
            <dt>From definition</dt>
            <dd>
              <DefinitionRef name={run.from_definition} />
            </dd>
          </div>
        )}
        {hp.length > 0 && (
          <div className="run-hp">
            <dt>Hyperparameters</dt>
            <dd>
              {hp.map(([k, v]) => (
                <span key={k} className="chip">
                  <span className="chip-key">{k}=</span>
                  <span className="num">{JSON.stringify(v)}</span>
                </span>
              ))}
              <SaveAsDefinition run={run} />
            </dd>
          </div>
        )}
      </dl>
    </header>
  )
}

/** Freeze this run's predictor + hyperparams as a reusable definition. */
function SaveAsDefinition({ run }: { run: RunMeta }) {
  const [open, setOpen] = useState(false)
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState<string | null>(null)

  if (saved) {
    return (
      <span className="save-def" role="status">
        saved as <DefinitionRef name={saved} />
      </span>
    )
  }

  const nameOk = NAME_RE.test(name)
  const save = async () => {
    setBusy(true)
    setError(null)
    try {
      const def = await api.createDefinition({
        name,
        predictor: run.predictor,
        hyperparams: run.hyperparams,
        dataset_tags: [run.dataset_id],
      })
      setSaved(def.name)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setBusy(false)
    }
  }

  if (!open) {
    return (
      <button className="btn-inline" onClick={() => setOpen(true)} title="Save these hyperparameters as a reusable definition">
        save as definition
      </button>
    )
  }
  return (
    <span className="save-def">
      <input
        type="text"
        className="mono-input"
        placeholder="definition name"
        aria-label="Definition name"
        value={name}
        onChange={(e) => setName(e.target.value)}
        disabled={busy}
        autoFocus
      />
      <button className="btn-inline" onClick={save} disabled={!nameOk || busy}>
        {busy ? 'saving…' : 'save'}
      </button>
      <button className="btn-inline" onClick={() => setOpen(false)} disabled={busy}>
        cancel
      </button>
      {error && (
        <span className="hp-error" role="alert">
          {error}
        </span>
      )}
    </span>
  )
}

/* ---------------- finished run ---------------- */

type PredSort = 'abs_err' | 'pct_err' | 'actual' | 'predicted' | 'row_id'

interface Row extends Prediction {
  err: number
  pct: number
  suspicious: boolean
}

function FinishedRun({ run, events }: { run: RunMeta; events: ReturnType<typeof useRunEvents> }) {
  const domain = useDomain()
  const preds = useAsync(() => api.getPredictions(run.run_id), [run.run_id])
  const predictors = useAsync(() => api.listPredictors(), [])
  const m = run.metrics ?? ({} as Metrics)
  const metricCols = metricColumns(domain)
  const classification = isClassification(domain)
  const classLabels = useMemo(() => {
    if (domain.target.classes && domain.target.classes.length) return domain.target.classes
    if (domainTask(domain) === 'binary') return ['0', '1']
    const ids = new Set<number>()
    for (const p of preds.data ?? []) ids.add(Math.round(p.actual))
    return [...ids].sort((a, b) => a - b).map(String)
  }, [domain, preds.data])
  const tableRef = useRef<HTMLDivElement>(null)
  const [sort, setSort] = useState<{ key: PredSort; dir: 1 | -1 }>({ key: 'abs_err', dir: -1 })
  const [selectedRow, setSelectedRow] = useState<number | null>(null)
  const [excludeSuspicious, setExcludeSuspicious] = useState(false)

  const suspicious = useMemo(() => suspiciousRowIds(preds.data ?? []), [preds.data])
  const filteredMetrics = useMemo(
    () =>
      excludeSuspicious && suspicious.size > 0
        ? computeMetrics((preds.data ?? []).filter((p) => !suspicious.has(p.row_id)))
        : null,
    [excludeSuspicious, preds.data, suspicious],
  )

  const pred = predictors.data?.find((p) => p.name === run.predictor)
  const supportsPredict = !!pred?.predict_args

  const rows: Row[] = useMemo(
    () =>
      (preds.data ?? []).map((p) => ({
        ...p,
        err: p.predicted - p.actual,
        pct: (p.predicted - p.actual) / p.actual,
        suspicious: suspicious.has(p.row_id),
      })),
    [preds.data, suspicious],
  )

  const jumpToTable = useCallback((key: PredSort) => {
    setSort({ key, dir: key === 'row_id' ? 1 : -1 })
    tableRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [])

  const selectFromScatter = useCallback((rowId: number) => {
    setSelectedRow(rowId)
    tableRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [])

  if (!run.metrics) {
    return (
      <p className="muted" role="status">
        Metrics appear when the run finishes evaluating.
      </p>
    )
  }

  return (
    <>
      {/* Metrics strip: every aggregate links to the predictions behind it. */}
      <MetricStrip
        ariaLabel="Test-set metrics; each opens the predictions behind it"
        metrics={[
          ...metricCols.map((col) => ({
            label: col,
            value: formatMetric(domain, col, metricValue(m, col)),
            onClick: () => jumpToTable(isPercentMetric(domain, col) ? 'pct_err' : 'abs_err'),
          })),
          { label: 'test items', value: m.n_test.toLocaleString() },
        ]}
      />

      {suspicious.size > 0 && (
        <div className="suspicious-row">
          <label className="toggle suspicious-toggle">
            <input
              type="checkbox"
              checked={excludeSuspicious}
              onChange={(e) => setExcludeSuspicious(e.target.checked)}
            />
            recompute without <span className="num">{suspicious.size}</span> suspicious{' '}
            {suspicious.size === 1 ? 'prediction' : 'predictions'}
            <span className="hp-hint">
              robust outliers on % error, usually mislabeled listings; marked ⚠ in the table
            </span>
          </label>
          {filteredMetrics && (
            <p className="filtered-strip num" role="status">
              filtered:{' '}
              {metricCols
                .map((col) => `${col} ${formatMetric(domain, col, metricValue(filteredMetrics, col))}`)
                .join(' · ')}{' '}
              · n {filteredMetrics.n_test.toLocaleString()}
              <span className="muted"> (official metrics above are unchanged)</span>
            </p>
          )}
        </div>
      )}

      {supportsPredict && <PromotePanel run={run} />}

      <div className="charts-grid">
        <div>
          {classification ? (
            <>
              {/* Class labels, not a continuous target: a confusion matrix, not
                  a log-scale scatter. */}
              <h2 className="panel-title">Confusion matrix</h2>
              <ConfusionMatrix predictions={preds.data ?? []} classes={classLabels} />
            </>
          ) : (
            <>
              <h2 className="panel-title">Predicted vs actual</h2>
              <ScatterChart predictions={preds.data ?? []} onSelect={selectFromScatter} selectedRowId={selectedRow} />
            </>
          )}
        </div>
        <div className="charts-col">
          <div>
            <h2 className="panel-title">Loss</h2>
            <LossChart epochs={events.epochs} />
            {events.epochs.length >= 3 && (
              <>
                <h2 className="panel-title dynamics-title">Is it still learning?</h2>
                <LossDerivativeChart epochs={events.epochs} />
              </>
            )}
          </div>
          {/* Percent-error histogram is target-space; only meaningful for a
              continuous target (regression / forecast), not class labels. */}
          {!classification && (
            <div>
              <h2 className="panel-title">Error distribution</h2>
              <ErrorHistogram series={[{ label: run.predictor, predictions: preds.data ?? [], style: 'fill' }]} />
            </div>
          )}
        </div>
      </div>

      <RunArch run={run} canFallback={!!pred?.visualization} />

      <div ref={tableRef}>
        <PredictionsTable
          run={run}
          rows={rows}
          loading={preds.loading}
          error={preds.error}
          retry={preds.reload}
          sort={sort}
          setSort={setSort}
          selectedRow={selectedRow}
          setSelectedRow={setSelectedRow}
        />
      </div>
    </>
  )
}

/* ---------------- promote to model ---------------- */

const NAME_RE = /^[a-z0-9][a-z0-9-]{0,63}$/

function PromotePanel({ run, checkpointOnly = false }: { run: RunMeta; checkpointOnly?: boolean }) {
  const [open, setOpen] = useState(false)
  const [name, setName] = useState('')
  const [notes, setNotes] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [promoted, setPromoted] = useState<string | null>(null)

  if (promoted) {
    return (
      <p className="promote-done" role="status">
        Promoted to <ModelRef name={promoted} />. The model is self-contained: it keeps
        predicting even if this run or its dataset is
        deleted.
      </p>
    )
  }

  const nameOk = NAME_RE.test(name)

  const promote = async () => {
    setBusy(true)
    setError(null)
    try {
      const record = await api.promoteModel(name, run.run_id, notes.trim() || undefined)
      setPromoted(record.name)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setBusy(false)
    }
  }

  return (
    <div className="promote-row">
      {!open ? (
        <>
          <button className="btn" onClick={() => setOpen(true)}>
            Promote to model
          </button>
          <span className="muted">
            {checkpointOnly
              ? 'freeze the last saved checkpoint under a name — promoted without final test metrics'
              : 'freeze these weights under a name and predict with them any time'}
          </span>
        </>
      ) : (
        <div className="promote-form">
          <div className="hp-field">
            <label htmlFor="promote-name">Model name</label>
            <input
              id="promote-name"
              type="text"
              className="mono-input"
              placeholder="e.g. mlp-prod-v1"
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={busy}
              autoFocus
            />
            {name && !nameOk && (
              <span className="hp-hint">lowercase letters, digits and dashes only</span>
            )}
          </div>
          <div className="hp-field">
            <label htmlFor="promote-notes">Notes (optional)</label>
            <input
              id="promote-notes"
              type="text"
              className="mono-input"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              disabled={busy}
            />
          </div>
          <div className="promote-actions">
            <button className="btn btn-primary" onClick={promote} disabled={!nameOk || busy}>
              {busy ? 'Promoting…' : 'Promote run'}
            </button>
            <button className="btn" onClick={() => setOpen(false)} disabled={busy}>
              Cancel
            </button>
            {error && (
              <span className="hp-error" role="alert">
                {error}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

/* ---------------- predictions table ---------------- */

const PAGE = 100

function PredictionsTable({
  run,
  rows,
  loading,
  error,
  retry,
  sort,
  setSort,
  selectedRow,
  setSelectedRow,
}: {
  run: RunMeta
  rows: Row[]
  loading: boolean
  error: string | null
  retry: () => void
  sort: { key: PredSort; dir: 1 | -1 }
  setSort: (s: { key: PredSort; dir: 1 | -1 }) => void
  selectedRow: number | null
  setSelectedRow: (id: number | null) => void
}) {
  const [filter, setFilter] = useState('')
  const [limit, setLimit] = useState(PAGE)
  const [items, setItems] = useState<Items | null>(null)
  const [expanded, setExpanded] = useState<number | null>(null)

  // Item details load lazily, on the first expansion.
  const ensureItems = useCallback(() => {
    if (!items) {
      void loadItems(run.dataset_id).then(setItems, () => setItems({}))
    }
  }, [items, run.dataset_id])

  const sorted = useMemo(() => {
    const val = (r: Row): number => {
      switch (sort.key) {
        case 'abs_err':
          return Math.abs(r.err)
        case 'pct_err':
          return Math.abs(r.pct)
        case 'actual':
          return r.actual
        case 'predicted':
          return r.predicted
        case 'row_id':
          return r.row_id
      }
    }
    const f = filter.trim()
    const filtered = f === '' ? rows : rows.filter((r) => String(r.row_id).includes(f))
    return [...filtered].sort((a, b) => (val(a) - val(b)) * sort.dir)
  }, [rows, sort, filter])

  // A scatter click selects a row: surface it even if it's deep in the list.
  useEffect(() => {
    if (selectedRow == null) return
    const idx = sorted.findIndex((r) => r.row_id === selectedRow)
    if (idx >= 0 && idx >= limit) setLimit(Math.ceil((idx + 1) / PAGE) * PAGE)
    setExpanded(selectedRow)
    ensureItems()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedRow])

  const visible = sorted.slice(0, limit)

  const th = (key: PredSort, label: string, numCol = true) => (
    <th
      className={numCol ? 'num-col' : ''}
      aria-sort={sort.key === key ? (sort.dir === 1 ? 'ascending' : 'descending') : undefined}
    >
      <button
        className="sort"
        onClick={() => setSort({ key, dir: sort.key === key ? ((-sort.dir) as 1 | -1) : -1 })}
        aria-label={`Sort by ${label}`}
      >
        {label}
        {sort.key === key && <span className="sort-arrow">{sort.dir === 1 ? '▲' : '▼'}</span>}
      </button>
    </th>
  )

  if (error) {
    return (
      <div className="error-block" role="alert">
        Could not load predictions: {error}{' '}
        <button className="btn" onClick={retry}>
          Retry
        </button>
      </div>
    )
  }

  return (
    <section aria-label="Individual predictions">
      <div className="pred-head">
        <h2 className="panel-title">
          Predictions <span className="muted">({sorted.length.toLocaleString()} test items, worst first)</span>
        </h2>
        <input
          type="text"
          className="mono-input"
          placeholder="Filter by item id"
          aria-label="Filter predictions by item id"
          value={filter}
          onChange={(e) => {
            setFilter(e.target.value)
            setLimit(PAGE)
          }}
        />
      </div>
      {loading ? (
        <div className="skeleton skeleton-md" />
      ) : (
        <table className="pred-table">
          <thead>
            <tr>
              {th('row_id', 'Item', false)}
              {th('actual', 'Actual')}
              {th('predicted', 'Predicted')}
              {th('abs_err', 'Error')}
              {th('pct_err', '% error')}
              <th>
                <span className="sr-only">Details</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {visible.map((r) => (
              <PredRow
                key={r.row_id}
                row={r}
                item={items?.[String(r.row_id)] ?? null}
                expanded={expanded === r.row_id}
                highlighted={selectedRow === r.row_id}
                onToggle={() => {
                  ensureItems()
                  setExpanded(expanded === r.row_id ? null : r.row_id)
                  setSelectedRow(expanded === r.row_id ? null : r.row_id)
                }}
              />
            ))}
          </tbody>
        </table>
      )}
      {!loading && limit < sorted.length && (
        <button className="btn show-more" onClick={() => setLimit((l) => l + PAGE)}>
          Show {Math.min(PAGE, sorted.length - limit)} more of {(sorted.length - limit).toLocaleString()}
        </button>
      )}
      {!loading && sorted.length === 0 && filter && (
        <p className="muted">No items match “{filter}”.</p>
      )}
    </section>
  )
}

function PredRow({
  row,
  item,
  expanded,
  highlighted,
  onToggle,
}: {
  row: Row
  item: Items[string] | null
  expanded: boolean
  highlighted: boolean
  onToggle: () => void
}) {
  const domain = useDomain()
  return (
    <>
      <tr
        className={`pred-row${highlighted ? ' is-selected' : ''}`}
        onClick={onToggle}
        aria-expanded={expanded}
      >
        <td className="num">{row.row_id}</td>
        <td className="num-col num">{fmtTarget(row.actual, domain)}</td>
        <td className="num-col num">{fmtTarget(row.predicted, domain)}</td>
        <td className="num-col num">{fmtTargetDelta(row.err, domain)}</td>
        <td className={`num-col num pct-cell ${Math.abs(row.pct) >= 0.5 ? 'pct-large' : ''}`}>
          {fmtSignedPct(row.pct)}
          {row.suspicious && (
            <span
              className="sus-glyph"
              title="Suspicious: robust outlier on % error, likely a mislabeled listing"
            >
              {' '}
              ⚠
            </span>
          )}
        </td>
        <td className="expand-cell" aria-hidden>
          {expanded ? '▾' : '▸'}
        </td>
      </tr>
      {expanded && (
        <tr className="pred-detail">
          <td colSpan={6}>
            {item === null ? (
              <div className="skeleton skeleton-row" />
            ) : (
              <div className="item-detail">
                <p className="item-content">{item.content || <span className="muted">No description</span>}</p>
                <ItemMeta item={item} />
              </div>
            )}
          </td>
        </tr>
      )}
    </>
  )
}
