import type { SaeReport } from '../api/types'
import SaeProbeChart from './charts/SaeProbeChart'

/** Renders a dataset-level SAE report: reconstruction summary, the raw-vs-code
 *  probe chart + table, and the most interpretable atoms. Extracted from SaeTool
 *  so the live tool and the Saved-analyses view render identically. */
export default function SaeReportView({
  report,
  caption = true,
}: {
  report: SaeReport
  caption?: boolean
}) {
  const probe = (seg: string, st: string) =>
    report.probes.find((p) => p.segment === seg && p.stage === st)
  const segments = [...new Set(report.probes.map((p) => p.segment))]
  const labeled = [
    ...report.atoms_by_target_corr,
    ...report.atoms_by_segment_separation,
  ].some((a) => !!a.label)

  return (
    <div className="interp-result">
      {caption && (
        <p className="interp-caption">
          {report.dataset_id} · {report.n_train} train / {report.n_test} test · input{' '}
          {report.config.input_dims}-d → {report.config.n_atoms} atoms · reconstructs{' '}
          {(report.recon.var_explained * 100).toFixed(0)}% of variance ·{' '}
          {report.recon.l0_mean.toFixed(0)} atoms active/row (
          {(report.recon.l0_frac * 100).toFixed(1)}%)
          {report.cached ? ' · SAE loaded from cache' : ' · SAE freshly trained'}
        </p>
      )}

      <SaeProbeChart probes={report.probes} />
      <p className="interp-tools-note">
        If the green (code) bars sit below the blue (raw) bars, the sparse code is a worse linear
        substrate than the embeddings it came from — the SAE found no extra usable signal.
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
          Atoms whose activation tracks target, or separates the segment from the rest (σ units).
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
  )
}
