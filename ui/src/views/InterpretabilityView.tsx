import { useEffect, useMemo, useRef, useState } from 'react'
import { api, pollJob } from '../api/client'
import type {
  CompareModelReport,
  InterpModel,
  LayerProbeComparison,
  LayerProbeReport,
  LayerProbeStage,
} from '../api/types'
import LayerProbeChart, { type ProbeMetric } from '../components/charts/LayerProbeChart'
import LayerProbeCompareChart from '../components/charts/LayerProbeCompareChart'
import SaeTool from './SaeTool'
import EmbeddingProbeTool from './EmbeddingProbeTool'
import ModelSaeTool from './ModelSaeTool'
import SavedAnalysesTool from './SavedAnalysesTool'
import ViewHeader from '../components/ViewHeader'
import { useAsync } from '../hooks/useAsync'
import { useDocTitle } from '../lib/DomainContext'
import './interpretability.css'

const METRICS: { key: ProbeMetric; label: string }[] = [
  { key: 'test_r2_log', label: 'test R² (log)' },
  { key: 'test_mae', label: 'test MAE' },
  { key: 'test_medape', label: 'test medAPE' },
]

/** A model whose probe completed, kept for the chart + tables. */
interface Probed {
  model: string
  report: LayerProbeReport
}

const archLabel = (m: InterpModel) =>
  m.hidden.length ? `[${m.hidden.join(', ')}] · ${m.activation}` : m.predictor

const outputStage = (r: LayerProbeReport): LayerProbeStage =>
  r.stages.find((s) => s.lambda == null) ?? r.stages[r.stages.length - 1]

export default function InterpretabilityView() {
  useDocTitle('Interpretability')
  const models = useAsync(() => api.listInterpModels(), [])
  const datasets = useAsync(() => api.listDatasets(), [])

  const [tool, setTool] = useState<
    'layer-probe' | 'sae' | 'model-sae' | 'embedding' | 'saved'
  >('layer-probe')
  const [datasetId, setDatasetId] = useState('')
  const [selected, setSelected] = useState<string[]>([])
  const [metric, setMetric] = useState<ProbeMetric>('test_r2_log')
  const [running, setRunning] = useState(false)
  const [stage, setStage] = useState('')
  const [result, setResult] = useState<{ dataset_id: string; series: Probed[] } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const cancelRef = useRef<(() => void) | null>(null)

  // The selected dataset's width, and the probeable models that match it — only
  // same-width models can be probed on (and compared over) the same dataset.
  const datasetNCols = useMemo(
    () => datasets.data?.find((d) => d.dataset_id === datasetId)?.n_cols ?? null,
    [datasets.data, datasetId],
  )
  const compatible = useMemo(
    () =>
      datasetNCols == null
        ? []
        : (models.data ?? []).filter((m) => m.n_cols === datasetNCols),
    [models.data, datasetNCols],
  )
  const chosen = selected.filter((name) => compatible.some((m) => m.name === name))

  // Stop polling if the view unmounts mid-run.
  useEffect(() => () => cancelRef.current?.(), [])

  const toggle = (name: string) =>
    setSelected((prev) => (prev.includes(name) ? prev.filter((x) => x !== name) : [...prev, name]))

  const run = () => {
    if (!datasetId || chosen.length === 0) return
    cancelRef.current?.()
    setRunning(true)
    setError(null)
    setResult(null)
    setStage('starting…')
    const job =
      chosen.length === 1
        ? api.startLayerProbe({ model: chosen[0], dataset: datasetId })
        : api.compareLayerProbes({ models: chosen, dataset: datasetId })
    job.then(
      (jobId) => {
        cancelRef.current = pollJob(jobId, {
          onStage: setStage,
          onDone: (res) => {
            if (chosen.length === 1) {
              const r = res as LayerProbeReport
              setResult({ dataset_id: r.dataset_id, series: [{ model: chosen[0], report: r }] })
            } else {
              const c = res as LayerProbeComparison
              const ok = c.reports.filter(
                (x): x is CompareModelReport & { report: LayerProbeReport } => x.report != null,
              )
              setResult({
                dataset_id: c.dataset_id,
                series: ok.map((x) => ({ model: x.model, report: x.report })),
              })
              const errs = c.reports.filter((x) => x.error).map((x) => `${x.model}: ${x.error}`)
              if (errs.length) setError(errs.join(' · '))
            }
            setRunning(false)
          },
          onError: (msg) => {
            setError(msg)
            setRunning(false)
          },
        })
      },
      (e: unknown) => {
        setError(e instanceof Error ? e.message : String(e))
        setRunning(false)
      },
    )
  }

  const single = result && result.series.length === 1 ? result.series[0].report : null
  const featureGroups = single?.feature_groups ?? []

  return (
    <>
      <ViewHeader
        glyph="ip"
        title="Interpretability"
        meta="Tools that explain how a trained model reaches its predictions"
      />
      <div className="view-body interp">
        <aside className="interp-tools" aria-label="Interpretability tools">
          <button
            type="button"
            className={`interp-tool${tool === 'layer-probe' ? ' is-active' : ''}`}
            onClick={() => setTool('layer-probe')}
          >
            Layer-wise linear probes
          </button>
          <button
            type="button"
            className={`interp-tool${tool === 'sae' ? ' is-active' : ''}`}
            onClick={() => setTool('sae')}
          >
            Sparse autoencoder (P2)
          </button>
          <button
            type="button"
            className={`interp-tool${tool === 'embedding' ? ' is-active' : ''}`}
            onClick={() => setTool('embedding')}
          >
            Embedding probe (P1)
          </button>
          <button
            type="button"
            className={`interp-tool${tool === 'model-sae' ? ' is-active' : ''}`}
            onClick={() => setTool('model-sae')}
          >
            Per-model SAE (P3)
          </button>
          <button
            type="button"
            className={`interp-tool${tool === 'saved' ? ' is-active' : ''}`}
            onClick={() => setTool('saved')}
          >
            Saved analyses
          </button>
          <p className="interp-tools-note">More tools land here over time.</p>
        </aside>

        <section className="interp-main">
          {tool === 'sae' ? (
            <SaeTool />
          ) : tool === 'embedding' ? (
            <EmbeddingProbeTool />
          ) : tool === 'model-sae' ? (
            <ModelSaeTool />
          ) : tool === 'saved' ? (
            <SavedAnalysesTool />
          ) : (
          <>
          <p className="interp-lede">
            <strong>Layer-wise linear probes</strong> (Alain &amp; Bengio,{' '}
            <a href="https://arxiv.org/abs/1610.01644" target="_blank" rel="noreferrer">
              2016
            </a>
            ): a cheap ridge probe is fit on the activations at each stage of the network — the raw
            input, then every hidden layer — measuring how linearly decodable the target is at that
            depth. A rising curve shows the architecture progressively building the prediction; the
            gap to the model&apos;s own output (the hollow point) is what the trained head still adds.
            Pick one model, or <strong>several to overlay and compare</strong> how different
            architectures get there.
          </p>

          <div className="interp-controls">
            <label>
              <span>Dataset</span>
              <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
                <option value="">Select a dataset…</option>
                {(datasets.data ?? []).map((d) => (
                  <option key={d.dataset_id} value={d.dataset_id}>
                    {d.dataset_id} · {d.n_cols} cols
                  </option>
                ))}
              </select>
            </label>
            <button
              className="btn-primary"
              onClick={run}
              disabled={!datasetId || chosen.length === 0 || running}
            >
              {running
                ? 'Probing…'
                : chosen.length > 1
                  ? `Compare ${chosen.length} models`
                  : 'Run layer probe'}
            </button>
          </div>

          {datasetId && (
            <fieldset className="interp-models">
              <legend>
                Models to probe{datasetNCols != null && ` · ${datasetNCols} cols`}
              </legend>
              {compatible.length === 0 ? (
                <p className="interp-tools-note">No probeable model matches this dataset&apos;s width.</p>
              ) : (
                <div className="interp-model-grid">
                  {compatible.map((m) => (
                    <label key={m.name} className="interp-model-opt">
                      <input
                        type="checkbox"
                        checked={selected.includes(m.name)}
                        onChange={() => toggle(m.name)}
                        disabled={running}
                      />
                      <span className="interp-model-name">{m.name}</span>
                      <span className="interp-model-arch">{archLabel(m)}</span>
                    </label>
                  ))}
                </div>
              )}
              <p className="interp-tools-note">
                Pick one model, or several to overlay and compare their curves.
              </p>
            </fieldset>
          )}

          {models.error && (
            <div className="error-block" role="alert">
              Could not load models: {models.error}
            </div>
          )}
          {running && <div className="interp-progress">Running probe — {stage}</div>}
          {error && (
            <div className="error-block" role="alert">
              {error}
            </div>
          )}

          {result && result.series.length > 0 && (
            <div className="interp-result">
              <div className="scale-toggle interp-metric-toggle">
                {METRICS.map((mk) => (
                  <button
                    key={mk.key}
                    className={`scale-btn${metric === mk.key ? ' is-active' : ''}`}
                    onClick={() => setMetric(mk.key)}
                  >
                    {mk.label}
                  </button>
                ))}
              </div>

              {single ? (
                <>
                  <LayerProbeChart stages={single.stages} metric={metric} />
                  <p className="interp-caption">
                    {single.model_dir} on {single.dataset_id} · {single.n_train} train /{' '}
                    {single.n_test} test · hidden [{single.hidden.join(', ')}] · {single.activation}
                  </p>
                  <table className="interp-table">
                    <thead>
                      <tr>
                        <th>stage</th>
                        <th>dim</th>
                        <th>test logR²</th>
                        <th>test R² (target)</th>
                        <th>MAE</th>
                        <th>medAPE</th>
                        <th>λ</th>
                      </tr>
                    </thead>
                    <tbody>
                      {single.stages.map((s) => (
                        <tr key={s.index} className={s.lambda == null ? 'is-output' : undefined}>
                          <td>{s.name}</td>
                          <td>{s.dim}</td>
                          <td>{s.test_r2_log.toFixed(3)}</td>
                          <td>{s.test_r2_target.toFixed(3)}</td>
                          <td>{Math.round(s.test_mae).toLocaleString()}</td>
                          <td>{s.test_medape.toFixed(1)}%</td>
                          <td>{s.lambda == null ? '—' : s.lambda}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>

                  {featureGroups.length > 0 && (
                    <div className="interp-fgroups">
                      <h4>Feature-family decodability (input columns)</h4>
                      <p className="interp-tools-note">
                        A ridge probe on each feature family alone — where the linearly decodable
                        target signal lives before the network mixes it.
                      </p>
                      <table className="interp-table">
                        <thead>
                          <tr>
                            <th>feature group</th>
                            <th>dim</th>
                            <th>test logR²</th>
                            <th>MAE</th>
                          </tr>
                        </thead>
                        <tbody>
                          {[...featureGroups]
                            .sort((a, b) => b.test_r2_log - a.test_r2_log)
                            .map((g) => (
                              <tr key={g.name}>
                                <td>{g.name.replace(/^input:/, '')}</td>
                                <td>{g.dim}</td>
                                <td>{g.test_r2_log.toFixed(3)}</td>
                                <td>{Math.round(g.test_mae).toLocaleString()}</td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </>
              ) : (
                <>
                  <LayerProbeCompareChart
                    series={result.series.map((s) => ({ model: s.model, stages: s.report.stages }))}
                    metric={metric}
                  />
                  <p className="interp-caption">
                    {result.series.length} models on {result.dataset_id}
                  </p>
                  <table className="interp-table">
                    <thead>
                      <tr>
                        <th>model</th>
                        <th>arch</th>
                        <th>output MAE</th>
                        <th>output logR²</th>
                        <th>input→output gain</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.series.map((s) => {
                        const out = outputStage(s.report)
                        const gain = out.test_r2_log - s.report.stages[0].test_r2_log
                        return (
                          <tr key={s.model}>
                            <td>{s.model}</td>
                            <td>[{s.report.hidden.join(', ')}]</td>
                            <td>{Math.round(out.test_mae).toLocaleString()}</td>
                            <td>{out.test_r2_log.toFixed(3)}</td>
                            <td>{gain >= 0 ? '+' : ''}{gain.toFixed(3)}</td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </>
              )}
            </div>
          )}
          </>
          )}
        </section>
      </div>
    </>
  )
}
