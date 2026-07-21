import { type CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, pollJob } from '../api/client'
import type {
  EmbeddingProbeReport,
  InterpAnalysis,
  LayerProbeComparison,
  LayerProbeReport,
  ModelSaeReport,
  SaeReport,
} from '../api/types'
import { type ProbeMetric } from '../components/charts/LayerProbeChart'
import LayerProbeCompareChart from '../components/charts/LayerProbeCompareChart'
import ModelSaeCompareChart from '../components/charts/ModelSaeCompareChart'
import { MlpTopologyStrip } from '../components/charts/MlpTopologyDiagram'
import { paletteColor } from '../components/charts/palette'
import EmbeddingProbeReportView from '../components/EmbeddingProbeReportView'
import LayerProbeReportView from '../components/LayerProbeReportView'
import ModelSaeReportView from '../components/ModelSaeReportView'
import SaeReportView from '../components/SaeReportView'
import { useAsync } from '../hooks/useAsync'
import { modelSaeDigest } from '../lib/modelSae'

const TOOL_LABEL: Record<string, string> = {
  'model-sae': 'per-model SAE',
  sae: 'dataset SAE',
  'layer-probe': 'layer probe',
  'layer-probe-compare': 'layer-probe compare',
  'embedding-probe': 'embedding probe',
}

const METRICS: { key: ProbeMetric; label: string }[] = [
  { key: 'test_r2_log', label: 'test R² (log)' },
  { key: 'test_mae', label: 'test MAE' },
  { key: 'test_medape', label: 'test medAPE' },
]

/** A compact description of the run's knobs, per tool. */
function configSummary(a: InterpAnalysis): string {
  const c = a.config ?? {}
  switch (a.tool) {
    case 'model-sae': {
      const parts: string[] = []
      if (c.layers) parts.push(`layers ${String(c.layers)}`)
      if (c.compare_embedding) parts.push('emb-diff')
      if (c.label_atoms) parts.push('labeled')
      return parts.join(' · ') || 'all layers'
    }
    case 'sae':
      return c.n_atoms ? `${String(c.n_atoms)} atoms` : 'default'
    case 'layer-probe':
      return c.lambda ? `λ=${String(c.lambda)}` : 'λ auto'
    case 'layer-probe-compare':
      return Array.isArray(c.models) ? `${c.models.length} models` : ''
    case 'embedding-probe':
      return c.split_by ? `split ${String(c.split_by)}` : ''
    default:
      return ''
  }
}

/** Whether an SAE result document carries any GPT atom label. */
function resultHasLabels(res: unknown): boolean {
  const r = res as
    | {
        atoms_by_target_corr?: { label?: string | null }[]
        layers?: { atoms_by_target_corr?: { label?: string | null }[] }[]
      }
    | null
    | undefined
  if (!r) return false
  if ((r.atoms_by_target_corr ?? []).some((x) => x.label)) return true
  return (r.layers ?? []).some((l) => (l.atoms_by_target_corr ?? []).some((x) => x.label))
}

/** Whether an analysis already carries GPT atom labels (config flag or a real
 *  label on any surfaced atom). */
function hasLabels(a: InterpAnalysis): boolean {
  return a.config?.label_atoms === true || resultHasLabels(a.result)
}

type SortKey = 'created_at' | 'tool' | 'model' | 'dataset_id' | 'status'

function MetricToggle({ metric, onChange }: { metric: ProbeMetric; onChange: (m: ProbeMetric) => void }) {
  return (
    <div className="scale-toggle interp-metric-toggle">
      {METRICS.map((mk) => (
        <button
          key={mk.key}
          className={`scale-btn${metric === mk.key ? ' is-active' : ''}`}
          onClick={() => onChange(mk.key)}
        >
          {mk.label}
        </button>
      ))}
    </div>
  )
}

/** GPT atom-labelling control for the SAE tools — re-runs the SAE with
 *  auto-interp and persists it as a NEW labelled analysis. Rendered on every
 *  labelable report, so it's reachable whether you're viewing one analysis or
 *  comparing several (one per column). */
function LabelAtomsBar({
  analysis,
  swap,
  reload,
}: {
  analysis: InterpAnalysis
  swap: (oldId: string, newId: string) => void
  reload: () => void
}) {
  const [labeling, setLabeling] = useState(false)
  const [stage, setStage] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const cancelRef = useRef<(() => void) | null>(null)
  useEffect(() => () => cancelRef.current?.(), [])

  const cfg = analysis.config ?? {}
  const labelable =
    (analysis.tool === 'model-sae' || analysis.tool === 'sae') && analysis.status === 'done'
  if (!labelable) return null
  if (hasLabels(analysis)) {
    return <p className="interp-tools-note saved-label-done">✓ GPT atom labels included</p>
  }

  const run = () => {
    setLabeling(true)
    setErr(null)
    setStage('starting…')
    const launch =
      analysis.tool === 'model-sae'
        ? api.startModelSae({
            model: analysis.model as string,
            dataset: analysis.dataset_id,
            layers: (cfg.layers as string) || undefined,
            n_atoms: cfg.n_atoms as number,
            l1: cfg.l1 as number,
            epochs: cfg.epochs as number,
            compare_embedding: cfg.compare_embedding as boolean,
            label_atoms: true,
          })
        : api.startSae({
            dataset: analysis.dataset_id,
            n_atoms: cfg.n_atoms as number,
            l1: cfg.l1 as number,
            epochs: cfg.epochs as number,
            segment: cfg.segment as string,
            label_atoms: true,
          })
    launch.then(
      (jobId) => {
        cancelRef.current = pollJob(jobId, {
          onStage: setStage,
          onDone: (res) => {
            setLabeling(false)
            if (resultHasLabels(res)) {
              swap(analysis.id, jobId)
              reload()
            } else {
              // No labels came back — the server has no OPENAI_API_KEY. Don't
              // leave a useless unlabelled duplicate behind.
              setErr('No labels produced — set OPENAI_API_KEY on the server, then retry.')
              api.deleteInterpAnalysis(jobId).finally(reload)
            }
          },
          onError: (m) => {
            setErr(m)
            setLabeling(false)
          },
        })
      },
      (e: unknown) => {
        setErr(e instanceof Error ? e.message : String(e))
        setLabeling(false)
      },
    )
  }

  return (
    <div className="saved-label-bar">
      <button className="btn-primary" onClick={run} disabled={labeling}>
        {labeling ? 'Labelling…' : 'Label atoms with GPT'}
      </button>
      {labeling ? (
        <span className="interp-progress">{stage}</span>
      ) : (
        <span className="interp-tools-note">
          Re-runs this SAE with auto-interp and saves it as a new labelled analysis. Needs{' '}
          <code>OPENAI_API_KEY</code> on the server.
        </span>
      )}
      {err && (
        <span className="error-block" role="alert">
          {err}
        </span>
      )}
    </div>
  )
}

/** Saved analyses — the durable, comparable record of every interpretability
 *  run. Select freely, then explicitly View/Compare (nothing scrolls until you
 *  ask). Renders any one instantly and compares same-tool analyses side-by-side
 *  with an overlay chart + colour-matched MLP topology diagrams. GPT labelling
 *  is available on every SAE report, including each compare column. */
export default function SavedAnalysesTool() {
  const analyses = useAsync(() => api.listInterpAnalyses(), [])
  const [selected, setSelected] = useState<string[]>([])
  // The set actually being viewed/compared — committed explicitly, so picking a
  // second model to compare never yanks you down to a half-formed result.
  const [committed, setCommitted] = useState<string[]>([])
  const [details, setDetails] = useState<Record<string, InterpAnalysis>>({})
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: 'created_at', dir: -1 })

  const topRef = useRef<HTMLDivElement | null>(null)
  const resultRef = useRef<HTMLDivElement | null>(null)
  const wantScroll = useRef(false)
  const scrollToResult = useCallback(() => {
    resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [])
  const scrollToList = () => topRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })

  const rows = useMemo(() => {
    const list = [...(analyses.data ?? [])]
    const { key, dir } = sort
    list.sort((a, b) => {
      const av = (a[key] ?? '') as string
      const bv = (b[key] ?? '') as string
      return av < bv ? -dir : av > bv ? dir : 0
    })
    return list
  }, [analyses.data, sort])

  // Pre-warm full results for selected ids so committing is instant.
  useEffect(() => {
    const missing = selected.filter((id) => !details[id])
    if (missing.length === 0) return
    let cancelled = false
    Promise.all(
      missing.map((id) =>
        api
          .getInterpAnalysis(id)
          .then((a) => [id, a] as const)
          .catch(() => null),
      ),
    ).then((res) => {
      if (cancelled) return
      setDetails((prev) => {
        const next = { ...prev }
        for (const r of res) if (r) next[r[0]] = r[1]
        return next
      })
    })
    return () => {
      cancelled = true
    }
  }, [selected, details])

  // Scroll ONLY on an explicit View/Compare click — never on a checkbox toggle
  // and never on a label swap-in (which shouldn't yank you around).
  useEffect(() => {
    if (committed.length > 0 && wantScroll.current) {
      wantScroll.current = false
      scrollToResult()
    }
  }, [committed, scrollToResult])

  // Selection metadata (from the list; no fetch needed) drives the action bar.
  const selMeta = useMemo(
    () => (analyses.data ?? []).filter((a) => selected.includes(a.id)),
    [analyses.data, selected],
  )
  const selTools = new Set(selMeta.map((a) => a.tool))
  const mixed = selTools.size > 1
  const canCommit = selected.length === 1 || (selected.length > 1 && !mixed)

  const toggle = (id: string) =>
    setSelected((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]))

  const commit = () => {
    if (!canCommit) return
    wantScroll.current = true
    setCommitted([...selected])
  }

  const remove = (id: string) => {
    api.deleteInterpAnalysis(id).finally(() => {
      setSelected((prev) => prev.filter((x) => x !== id))
      setCommitted((prev) => prev.filter((x) => x !== id))
      setDetails((prev) => {
        const next = { ...prev }
        delete next[id]
        return next
      })
      analyses.reload()
    })
  }

  // A labelling re-run persists as a NEW analysis. Swap it in place — in both
  // the selection and the committed set — so a comparison keeps both models and
  // both labelled results survive (labelling A then B never drops A).
  const swap = (oldId: string, newId: string) => {
    setSelected((prev) => prev.map((x) => (x === oldId ? newId : x)))
    setCommitted((prev) => prev.map((x) => (x === oldId ? newId : x)))
  }

  const sortHeader = (key: SortKey, label: string) => (
    <th
      className="saved-th-sort"
      onClick={() => setSort((s) => ({ key, dir: s.key === key && s.dir === -1 ? 1 : -1 }))}
    >
      {label}
      {sort.key === key ? (sort.dir === -1 ? ' ▾' : ' ▴') : ''}
    </th>
  )

  const cmDetails = committed.map((id) => details[id]).filter(Boolean) as InterpAnalysis[]
  const cmReady = committed.length > 0 && cmDetails.length === committed.length
  const cmTools = new Set(cmDetails.map((a) => a.tool))
  const stale = committed.length > 0 && (committed.length !== selected.length || committed.some((id) => !selected.includes(id)))

  return (
    <>
      <div ref={topRef} />
      <p className="interp-lede">
        <strong>Saved analyses</strong>: every interpretability run is persisted — SAE captures,
        layer probes, embedding probes — so a past result is one click away with no recomputation.
        Tick one to view it, or several of the same tool to compare, then hit the action bar. Model
        SAEs are queued automatically when a model is promoted.
      </p>

      {analyses.error && (
        <div className="error-block" role="alert">
          Could not load analyses: {analyses.error}
        </div>
      )}
      {analyses.loading && <div className="interp-progress">Loading analyses…</div>}
      {!analyses.loading && rows.length === 0 && (
        <div className="interp-progress">
          No saved analyses yet. Run a tool from the rail, or promote an MLP model to auto-queue one.
        </div>
      )}

      {rows.length > 0 && (
        <table className="interp-table saved-table">
          <thead>
            <tr>
              <th aria-label="select" />
              {sortHeader('tool', 'tool')}
              {sortHeader('model', 'model / dataset')}
              <th>config</th>
              <th>source</th>
              {sortHeader('status', 'status')}
              {sortHeader('created_at', 'created')}
              <th aria-label="actions" />
            </tr>
          </thead>
          <tbody>
            {rows.map((a) => (
              <tr
                key={a.id}
                className={
                  committed.includes(a.id)
                    ? 'is-committed'
                    : selected.includes(a.id)
                      ? 'is-selected'
                      : undefined
                }
              >
                <td>
                  <input
                    type="checkbox"
                    checked={selected.includes(a.id)}
                    onChange={() => toggle(a.id)}
                    aria-label={`select ${a.id}`}
                  />
                </td>
                <td>{TOOL_LABEL[a.tool] ?? a.tool}</td>
                <td>
                  {a.model ? <span className="saved-model">{a.model}</span> : <em>—</em>}
                  <span className="saved-dataset"> · {a.dataset_id}</span>
                </td>
                <td>{configSummary(a)}</td>
                <td>
                  <span className={`saved-badge saved-badge-${a.source}`}>{a.source}</span>
                </td>
                <td>
                  <span className={`saved-badge saved-status-${a.status}`}>{a.status}</span>
                </td>
                <td className="saved-when">{new Date(a.created_at).toLocaleString()}</td>
                <td className="saved-actions">
                  <button className="btn-link" onClick={() => remove(a.id)} title="Delete analysis">
                    delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {/* Sticky action bar: selecting never scrolls — you commit here, and only
          then does the result render + scroll into view. */}
      {selected.length > 0 && (
        <div className="saved-sticky-bar">
          <span className="saved-sticky-count">
            {selected.length} selected
            {mixed
              ? ' · mixed tools'
              : selected.length > 1
                ? ` · ${TOOL_LABEL[[...selTools][0]] ?? [...selTools][0]}`
                : ''}
            {stale ? ' · not shown yet' : ''}
          </span>
          <button
            type="button"
            className="btn-primary saved-sticky-go"
            onClick={commit}
            disabled={!canCommit}
            title={mixed ? 'Pick analyses of a single tool to compare' : undefined}
          >
            {mixed
              ? 'Same tool only'
              : selected.length > 1
                ? `Compare ${selected.length}`
                : 'View analysis'}
            {canCommit && (
              <span className="saved-chevron" aria-hidden="true">
                {' '}
                ↓
              </span>
            )}
          </button>
        </div>
      )}

      {committed.length > 0 && (
        <div ref={resultRef} className="interp-result saved-result">
          <button type="button" className="btn-link saved-back" onClick={scrollToList}>
            ↑ back to the list
          </button>
          {!cmReady ? (
            <div className="interp-progress">Loading selected analyses…</div>
          ) : cmTools.size > 1 ? (
            <div className="interp-progress">
              Select analyses of the same tool to compare (chosen:{' '}
              {[...cmTools].map((t) => TOOL_LABEL[t] ?? t).join(', ')}).
            </div>
          ) : cmDetails.length === 1 ? (
            <SingleAnalysisView analysis={cmDetails[0]} swap={swap} reload={analyses.reload} />
          ) : (
            <CompareView
              tool={[...cmTools][0]}
              analyses={cmDetails}
              swap={swap}
              reload={analyses.reload}
            />
          )}
        </div>
      )}
    </>
  )
}

/** One saved analysis: GPT-labelling control (for SAE tools) + its report. */
function SingleAnalysisView({
  analysis,
  swap,
  reload,
}: {
  analysis: InterpAnalysis
  swap: (oldId: string, newId: string) => void
  reload: () => void
}) {
  const [metric, setMetric] = useState<ProbeMetric>('test_r2_log')
  return (
    <>
      <LabelAtomsBar analysis={analysis} swap={swap} reload={reload} />
      {renderReport(analysis, metric, setMetric)}
    </>
  )
}

/** Render one saved analysis with its tool-specific view. */
function renderReport(a: InterpAnalysis, metric: ProbeMetric, setMetric: (m: ProbeMetric) => void) {
  if (a.status !== 'done' || a.result == null) {
    return (
      <div className="interp-progress">
        {a.status === 'failed' ? (a.error ?? 'analysis failed') : 'analysis still running…'}
      </div>
    )
  }
  switch (a.tool) {
    case 'model-sae':
      return <ModelSaeReportView report={a.result as ModelSaeReport} />
    case 'sae':
      return <SaeReportView report={a.result as SaeReport} />
    case 'embedding-probe':
      return <EmbeddingProbeReportView report={a.result as EmbeddingProbeReport} />
    case 'layer-probe':
      return (
        <>
          <MetricToggle metric={metric} onChange={setMetric} />
          <LayerProbeReportView report={a.result as LayerProbeReport} metric={metric} />
        </>
      )
    case 'layer-probe-compare': {
      const c = a.result as LayerProbeComparison
      const series = c.reports
        .filter((r) => r.report)
        .map((r) => ({ model: r.model, stages: (r.report as LayerProbeReport).stages }))
      return (
        <>
          <MetricToggle metric={metric} onChange={setMetric} />
          <LayerProbeCompareChart series={series} metric={metric} />
          <p className="interp-caption">
            {series.length} models on {c.dataset_id}
          </p>
        </>
      )
    }
    default:
      return null
  }
}

const colGrid = (n: number): CSSProperties => ({
  display: 'grid',
  gridTemplateColumns: `repeat(${n}, minmax(22rem, 1fr))`,
  gap: '1.5rem',
  alignItems: 'start',
  overflowX: 'auto',
})

/** Compare several same-tool analyses: an overlay chart + colour-matched MLP
 *  topology diagrams (where the tool has them), then each report in its own
 *  column with its own GPT-labelling control. */
function CompareView({
  tool,
  analyses,
  swap,
  reload,
}: {
  tool: string
  analyses: InterpAnalysis[]
  swap: (oldId: string, newId: string) => void
  reload: () => void
}) {
  const [metric, setMetric] = useState<ProbeMetric>('test_r2_log')
  const done = analyses.filter((a) => a.status === 'done' && a.result != null)
  if (done.length < 2) {
    return (
      <div className="interp-progress">
        Need at least two completed analyses to compare ({done.length} ready).
      </div>
    )
  }

  const topo = done.map((a, i) => {
    const r = a.result as { hidden?: number[]; activation?: string }
    return { label: a.model ?? a.id, hidden: r.hidden, activation: r.activation, color: paletteColor(i) }
  })

  const columns = (
    <div className="saved-compare-cols" style={colGrid(done.length)}>
      {done.map((a, i) => (
        <div key={a.id} className="saved-compare-col">
          <h4 className="saved-col-head">
            <span className="mlp-topo-swatch" style={{ background: paletteColor(i) }} />
            {a.model ?? a.id}
            <span className="saved-dataset"> · {a.dataset_id}</span>
          </h4>
          <LabelAtomsBar analysis={a} swap={swap} reload={reload} />
          {columnReport(a)}
        </div>
      ))}
    </div>
  )

  if (tool === 'model-sae') {
    return (
      <>
        <table className="interp-table msae-compare-digest">
          <thead>
            <tr>
              <th aria-label="colour" />
              <th>model</th>
              <th>peak R² log</th>
              <th>width used</th>
              <th>concepts</th>
              <th>segments</th>
              <th>dropped</th>
            </tr>
          </thead>
          <tbody>
            {done.map((a, i) => {
              const dg = modelSaeDigest(a.result as ModelSaeReport)
              return (
                <tr key={a.id}>
                  <td>
                    <span className="mlp-topo-swatch" style={{ background: paletteColor(i) }} />
                  </td>
                  <td className="saved-model">{a.model ?? a.id}</td>
                  <td>{dg.peakR2.toFixed(3)}</td>
                  <td>{(dg.maxUtil * 100).toFixed(0)}%</td>
                  <td>{dg.concepts}</td>
                  <td>{dg.segRepr}</td>
                  <td>{dg.dropped ?? '—'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
        <ModelSaeCompareChart
          series={done.map((a) => ({ model: a.model ?? a.id, report: a.result as ModelSaeReport }))}
        />
        <MlpTopologyStrip items={topo} />
        <MismatchNote list={done} />
        {columns}
      </>
    )
  }
  if (tool === 'layer-probe') {
    return (
      <>
        <MetricToggle metric={metric} onChange={setMetric} />
        <LayerProbeCompareChart
          series={done.map((a) => ({
            model: a.model ?? a.id,
            stages: (a.result as LayerProbeReport).stages,
          }))}
          metric={metric}
        />
        <MlpTopologyStrip items={topo} />
        <MismatchNote list={done} />
        {columns}
      </>
    )
  }
  return (
    <>
      <MismatchNote list={done} />
      {columns}
    </>
  )
}

/** One column's report (captions off — the column header names it). */
function columnReport(a: InterpAnalysis) {
  switch (a.tool) {
    case 'model-sae':
      return <ModelSaeReportView report={a.result as ModelSaeReport} caption={false} />
    case 'sae':
      return <SaeReportView report={a.result as SaeReport} caption={false} />
    case 'embedding-probe':
      return <EmbeddingProbeReportView report={a.result as EmbeddingProbeReport} caption={false} />
    case 'layer-probe':
      return <LayerProbeReportView report={a.result as LayerProbeReport} metric="test_r2_log" />
    default:
      return null
  }
}

/** Warn when the compared analyses aren't on the same dataset — their numbers
 *  aren't strictly comparable then (mirrors the layer-probe width gate). */
function MismatchNote({ list }: { list: InterpAnalysis[] }) {
  const datasets = new Set(list.map((a) => a.dataset_id))
  if (datasets.size <= 1) return null
  return (
    <p className="interp-tools-note">
      ⚠ These analyses span different datasets ({[...datasets].join(', ')}) — compare with care; the
      metrics aren&apos;t on the same footing.
    </p>
  )
}
