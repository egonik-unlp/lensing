import { useEffect, useMemo, useRef, useState } from 'react'
import { api, pollJob } from '../api/client'
import type { EmbeddingProbeReport, EmbViewResult } from '../api/types'
import EmbeddingProbeChart from '../components/charts/EmbeddingProbeChart'
import { useAsync } from '../hooks/useAsync'

const pct = (m?: { medape: number }) => (m ? `${m.medape.toFixed(1)}%` : '—')

/** P1 — the embedding probe (absent-vs-unused diagnostic). Self-contained tool
 *  rendered inside the Interpretability section's tool rail. */
export default function EmbeddingProbeTool() {
  const datasets = useAsync(() => api.listDatasets(), [])
  const [datasetId, setDatasetId] = useState('')
  const [splitBy, setSplitBy] = useState('')
  const [noMlp, setNoMlp] = useState(false)
  const [running, setRunning] = useState(false)
  const [stage, setStage] = useState('')
  const [report, setReport] = useState<EmbeddingProbeReport | null>(null)
  const [error, setError] = useState<string | null>(null)
  const cancelRef = useRef<(() => void) | null>(null)

  // Default to the widest dataset — the embedding probe needs a full-rank
  // (≫128-dim) artifact to contrast raw embeddings against PCA-128.
  const widest = useMemo(() => {
    const ds = datasets.data ?? []
    return ds.reduce<string>((best, d) => {
      const bn = ds.find((x) => x.dataset_id === best)?.n_cols ?? -1
      return d.n_cols > bn ? d.dataset_id : best
    }, '')
  }, [datasets.data])
  const effective = datasetId || widest

  // Derive the categorical one-hot groups the rows can be sliced by from the
  // selected dataset's manifest (so the split axis is no longer hardcoded).
  const manifest = useAsync(
    () => (effective ? api.getDataset(effective) : Promise.resolve(null)),
    [effective],
  )
  const splitGroups = useMemo(() => {
    const cols = manifest.data?.columns ?? []
    const seen = new Set<string>()
    for (const c of cols) if (c.kind.type === 'onehot') seen.add(c.kind.group)
    const groups = [...seen]
    groups.sort((a, b) => a.localeCompare(b))
    return groups
  }, [manifest.data])
  const effectiveSplit = splitBy || (splitGroups[0] ?? '')

  useEffect(() => () => cancelRef.current?.(), [])

  const run = () => {
    if (!effective) return
    cancelRef.current?.()
    setRunning(true)
    setError(null)
    setReport(null)
    setStage('starting…')
    api.startEmbeddingProbe({ dataset: effective, split_by: effectiveSplit || undefined, no_mlp: noMlp }).then(
      (jobId) => {
        cancelRef.current = pollJob(jobId, {
          onStage: setStage,
          onDone: (res) => {
            setReport(res as EmbeddingProbeReport)
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

  const present = report?.verdict.startsWith('PRESENT')
  const decode = report?.segment_decode

  return (
    <>
      <p className="interp-lede">
        <strong>Embedding probe (P1)</strong>: on a dataset&apos;s embeddings (no model needed), a
        cheap ridge — and an MLP ceiling — probe is fit on the <strong>raw embeddings vs the first
        128 PCA columns</strong>, sliced into segments by a categorical group (any one-hot field;
        defaults to property type), against a per-segment-median floor. It decides whether the
        hardest segment&apos;s target error is <em>information-absent</em> (signal not in the
        embeddings) or merely <em>information-unused</em> (present but discarded by PCA-128 or a
        global fit). Best run on a full-rank (≫128-dim) dataset.
      </p>

      <div className="interp-controls">
        <label>
          <span>Dataset</span>
          <select value={effective} onChange={(e) => setDatasetId(e.target.value)}>
            <option value="">Select a dataset…</option>
            {(datasets.data ?? []).map((d) => (
              <option key={d.dataset_id} value={d.dataset_id}>
                {d.dataset_id} · {d.n_cols} cols
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Split by</span>
          <select
            value={effectiveSplit}
            onChange={(e) => setSplitBy(e.target.value)}
            disabled={running || splitGroups.length === 0}
          >
            {splitGroups.length === 0 && <option value="">—</option>}
            {splitGroups.map((g) => (
              <option key={g} value={g}>
                {g}
              </option>
            ))}
          </select>
        </label>
        <label style={{ flexDirection: 'row', alignItems: 'center', gap: '0.4rem' }}>
          <input type="checkbox" checked={noMlp} onChange={(e) => setNoMlp(e.target.checked)} disabled={running} />
          <span>skip MLP ceiling (faster)</span>
        </label>
        <button className="btn-primary" onClick={run} disabled={!effective || running}>
          {running ? 'Probing…' : 'Run embedding probe'}
        </button>
      </div>

      {datasets.error && (
        <div className="error-block" role="alert">
          Could not load datasets: {datasets.error}
        </div>
      )}
      {running && <div className="interp-progress">Running embedding probe — {stage}</div>}
      {error && (
        <div className="error-block" role="alert">
          {error}
        </div>
      )}

      {report && (
        <div className="interp-result">
          <div
            className="interp-progress"
            style={{
              borderLeft: `3px solid ${present ? 'var(--ok, #16a34a)' : 'var(--warn, #d97706)'}`,
              paddingLeft: '0.7rem',
            }}
          >
            {report.verdict}
          </div>

          <EmbeddingProbeChart segments={report.segments} views={report.views} />
          <p className="interp-caption">
            {report.dataset} · split by <strong>{report.split_by}</strong> · emb {report.emb_dim} ·{' '}
            {report.n_train} train / {report.n_test} test · views {report.views.join(' vs ')}
          </p>

          {decode && (
            <p className="interp-caption">
              {report.split_by} decode: accuracy {decode.accuracy.toFixed(3)} (majority{' '}
              {decode.majority_acc.toFixed(3)}) · macro-F1 {decode.macro_f1.toFixed(3)}
              {decode.hardest_vs_rest_auc != null &&
                ` · ${decode.hardest_value ?? 'hardest'}-vs-rest AUC ${decode.hardest_vs_rest_auc.toFixed(3)}`}
            </p>
          )}

          <table className="interp-table" style={{ maxWidth: '56rem' }}>
            <thead>
              <tr>
                <th>segment</th>
                <th>view</th>
                <th>n test</th>
                <th>floor</th>
                <th>global</th>
                <th>seg-linear</th>
                <th>seg-MLP</th>
              </tr>
            </thead>
            <tbody>
              {report.segments.flatMap((s) =>
                s.views.map((v: EmbViewResult) => (
                  <tr key={`${s.name}:${v.view}`}>
                    <td>{s.name.replace(' (reference)', '')}</td>
                    <td>{v.view}</td>
                    <td>{v.n_test}</td>
                    <td>{v.skip ? `skip (${v.skip})` : pct(v.type_median)}</td>
                    <td>{pct(v.global_linear)}</td>
                    <td>{pct(v.seg_linear)}</td>
                    <td>{pct(v.seg_mlp)}</td>
                  </tr>
                )),
              )}
            </tbody>
          </table>
          <p className="interp-tools-note">
            Values are test medAPE. &ldquo;floor&rdquo; is the per-segment median; &ldquo;global&rdquo;
            a ridge trained on all rows; &ldquo;seg-linear/MLP&rdquo; trained on the segment only.
          </p>
        </div>
      )}
    </>
  )
}
