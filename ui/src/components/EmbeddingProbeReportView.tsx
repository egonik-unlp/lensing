import type { EmbeddingProbeReport, EmbViewResult } from '../api/types'
import EmbeddingProbeChart from './charts/EmbeddingProbeChart'

const pct = (m?: { medape: number }) => (m ? `${m.medape.toFixed(1)}%` : '—')

/** Renders an embedding-probe (P1) report: the absent-vs-unused verdict, the
 *  per-segment chart, the decode summary, and the medAPE table. Extracted from
 *  EmbeddingProbeTool so the live tool and the Saved-analyses view render
 *  identically. */
export default function EmbeddingProbeReportView({
  report,
  caption = true,
}: {
  report: EmbeddingProbeReport
  caption?: boolean
}) {
  const present = report.verdict.startsWith('PRESENT')
  const decode = report.segment_decode

  return (
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
      {caption && (
        <p className="interp-caption">
          {report.dataset} · split by <strong>{report.split_by}</strong> · emb {report.emb_dim} ·{' '}
          {report.n_train} train / {report.n_test} test · views {report.views.join(' vs ')}
        </p>
      )}

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
        Values are test medAPE. &ldquo;floor&rdquo; is the per-segment median; &ldquo;global&rdquo; a
        ridge trained on all rows; &ldquo;seg-linear/MLP&rdquo; trained on the segment only.
      </p>
    </div>
  )
}
