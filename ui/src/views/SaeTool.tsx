import { useEffect, useRef, useState } from 'react'
import { api, pollJob } from '../api/client'
import type { Manifest, SaeReport } from '../api/types'
import SaeProbeChart from '../components/charts/SaeProbeChart'
import { useAsync } from '../hooks/useAsync'

/** P2 — sparse-autoencoder (dictionary-learning) tool. Trains an SAE on a
 *  dataset's embedding block and probes its code against the raw embeddings,
 *  pooled and on the hard segment. Self-contained so it can be swapped into the
 *  Interpretability section's tool rail without touching the layer-probe tool. */
export default function SaeTool() {
  const datasets = useAsync(() => api.listDatasets(), [])
  const [datasetId, setDatasetId] = useState('')
  const [running, setRunning] = useState(false)
  const [stage, setStage] = useState('')
  const [report, setReport] = useState<SaeReport | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [labelAtoms, setLabelAtoms] = useState(false)
  const cancelRef = useRef<(() => void) | null>(null)

  useEffect(() => () => cancelRef.current?.(), [])

  const run = () => {
    if (!datasetId) return
    cancelRef.current?.()
    setRunning(true)
    setError(null)
    setReport(null)
    setStage('starting…')
    api.startSae({ dataset: datasetId, label_atoms: labelAtoms }).then(
      (jobId) => {
        cancelRef.current = pollJob(jobId, {
          onStage: setStage,
          onDone: (res) => {
            setReport(res as SaeReport)
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

  // Prefer wide datasets — the segment signal P2 chases lives in the embedding tail.
  const widest = (a: Manifest, b: Manifest) => b.n_cols - a.n_cols
  const probe = (seg: string, st: string) =>
    report?.probes.find((p) => p.segment === seg && p.stage === st)
  const segments = report ? [...new Set(report.probes.map((p) => p.segment))] : []
  const labeled = report
    ? [...report.atoms_by_target_corr, ...report.atoms_by_segment_separation].some((a) => !!a.label)
    : false

  return (
    <>
      <p className="interp-lede">
        <strong>Sparse autoencoder</strong> (dictionary learning): an over-complete ReLU
        autoencoder is trained to reconstruct the embeddings while keeping few atoms active, so
        each learned atom is a candidate rare feature — the regime where variance-PCA buries
        structure. The sparse code is then probed (same ridge engine) against the raw embeddings,
        pooled and on the hard segment, and the most target- / segment-aligned atoms are surfaced.
        It is the program&apos;s generative &ldquo;breakthrough shot&rdquo;.
      </p>

      <div className="interp-controls">
        <label>
          <span>Dataset (embeddings)</span>
          <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
            <option value="">Select a dataset…</option>
            {[...(datasets.data ?? [])].sort(widest).map((d) => (
              <option key={d.dataset_id} value={d.dataset_id}>
                {d.dataset_id} · {d.n_cols} cols
              </option>
            ))}
          </select>
        </label>
        <label className="interp-checkbox">
          <input
            type="checkbox"
            checked={labelAtoms}
            onChange={(e) => setLabelAtoms(e.target.checked)}
            disabled={running}
          />
          <span>Label atoms with GPT (auto-interp)</span>
        </label>
        <button className="btn-primary" onClick={run} disabled={!datasetId || running}>
          {running ? 'Training SAE…' : 'Train & probe SAE'}
        </button>
      </div>
      <p className="interp-tools-note">
        Trains a fresh SAE then probes it — a heavier job than the layer probe (a few minutes).
        Atom labeling names each top atom from its most-activating items via an LLM
        (Bills et al. 2023); needs <code>OPENAI_API_KEY</code> on the server (~$0.01 with a mini model).
      </p>

      {datasets.error && (
        <div className="error-block" role="alert">
          Could not load datasets: {datasets.error}
        </div>
      )}
      {running && <div className="interp-progress">{stage}</div>}
      {error && (
        <div className="error-block" role="alert">
          {error}
        </div>
      )}

      {report && (
        <div className="interp-result">
          <p className="interp-caption">
            {report.dataset_id} · {report.n_train} train / {report.n_test} test · input{' '}
            {report.config.input_dims}-d → {report.config.n_atoms} atoms · reconstructs{' '}
            {(report.recon.var_explained * 100).toFixed(0)}% of variance · {report.recon.l0_mean.toFixed(0)}{' '}
            atoms active/row ({(report.recon.l0_frac * 100).toFixed(1)}%)
            {report.cached ? ' · SAE loaded from cache' : ' · SAE freshly trained'}
          </p>

          <SaeProbeChart probes={report.probes} />
          <p className="interp-tools-note">
            If the green (code) bars sit below the blue (raw) bars, the sparse code is a worse
            linear substrate than the embeddings it came from — the SAE found no extra usable signal.
          </p>

          <table className="interp-table">
            <thead>
              <tr>
                <th>segment</th>
                <th>raw logR²</th>
                <th>code logR²</th>
                <th>raw MAE</th>
                <th>code MAE</th>
              </tr>
            </thead>
            <tbody>
              {segments.map((seg) => {
                const r = probe(seg, 'input:pca')
                const c = probe(seg, 'sae:code')
                return (
                  <tr key={seg}>
                    <td>{seg}</td>
                    <td>{r ? r.test_r2_log.toFixed(3) : '—'}</td>
                    <td>{c ? c.test_r2_log.toFixed(3) : '—'}</td>
                    <td>{r ? Math.round(r.test_mae).toLocaleString() : '—'}</td>
                    <td>{c ? Math.round(c.test_mae).toLocaleString() : '—'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>

          <div className="interp-fgroups">
            <h4>Most interpretable atoms</h4>
            <p className="interp-tools-note">
              Atoms whose activation tracks the target, or separates the segment from the rest (σ units).
              Strong, sparse atoms would be candidate monosemantic features.
            </p>
            <div className="sae-atom-cols">
              <table className="interp-table">
                <thead>
                  <tr>
                    <th>atom</th>
                    <th>target corr</th>
                    <th>freq</th>
                    {labeled && <th>label</th>}
                  </tr>
                </thead>
                <tbody>
                  {report.atoms_by_target_corr.slice(0, 6).map((a) => (
                    <tr key={a.atom}>
                      <td>#{a.atom}</td>
                      <td>{a.target_corr?.toFixed(3) ?? '—'}</td>
                      <td>{(a.freq * 100).toFixed(1)}%</td>
                      {labeled && <td className="sae-label">{a.label ?? '—'}</td>}
                    </tr>
                  ))}
                </tbody>
              </table>
              <table className="interp-table">
                <thead>
                  <tr>
                    <th>atom</th>
                    <th>{report.segment} sep (σ)</th>
                    <th>freq</th>
                    {labeled && <th>label</th>}
                  </tr>
                </thead>
                <tbody>
                  {report.atoms_by_segment_separation.slice(0, 6).map((a) => (
                    <tr key={a.atom}>
                      <td>#{a.atom}</td>
                      <td>{a.separation?.toFixed(2) ?? '—'}</td>
                      <td>{(a.freq * 100).toFixed(1)}%</td>
                      {labeled && <td className="sae-label">{a.label ?? '—'}</td>}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
