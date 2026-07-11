import type { LayerProbeReport } from '../api/types'
import LayerProbeChart, { type ProbeMetric } from './charts/LayerProbeChart'

/** Renders a single layer-probe report: the decodability-by-depth chart, the
 *  per-stage table, and the feature-family attribution. `metric` is a prop so
 *  the container owns the toggle (the live tool and the Saved-analyses view each
 *  render one). Extracted from InterpretabilityView for reuse. */
export default function LayerProbeReportView({
  report,
  metric,
}: {
  report: LayerProbeReport
  metric: ProbeMetric
}) {
  const featureGroups = report.feature_groups ?? []
  return (
    <>
      <LayerProbeChart stages={report.stages} metric={metric} />
      <p className="interp-caption">
        {report.model_dir} on {report.dataset_id} · {report.n_train} train / {report.n_test} test ·
        hidden [{report.hidden.join(', ')}] · {report.activation}
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
          {report.stages.map((s) => (
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
            A ridge probe on each feature family alone — where the linearly decodable target signal
            lives before the network mixes it.
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
  )
}
