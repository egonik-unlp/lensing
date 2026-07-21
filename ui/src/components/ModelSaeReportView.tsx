import { useMemo, useState } from 'react'
import type { ModelSaeReport } from '../api/types'
import { modelSaeDigest } from '../lib/modelSae'
import MlpTopologyDiagram from './charts/MlpTopologyDiagram'

/** Per-model SAE report. Reads answer-first: a one-line digest and one scannable
 *  "by depth" table carry the story; the per-layer atom detail (the old wall of
 *  tables) is collapsed behind expanders. Extracted from ModelSaeTool so the live
 *  tool and the Saved-analyses view/compare render identically. `caption` off in
 *  compare columns (the column header + top digest name it). */
export default function ModelSaeReportView({
  report,
  caption = true,
}: {
  report: ModelSaeReport
  caption?: boolean
}) {
  const [diffLayer, setDiffLayer] = useState<number | null>(null)

  const conceptByStage = useMemo(() => {
    const m = new Map<string, number>()
    for (const c of report.concept_vs_decodability ?? [])
      m.set(`hidden_${c.layer}`, c.n_interpretable_concepts)
    return m
  }, [report])
  const capByStage = useMemo(() => {
    const m = new Map<string, ModelSaeReport['layers'][number]['capacity']>()
    for (const l of report.layers ?? []) m.set(`hidden_${l.layer}`, l.capacity)
    return m
  }, [report])

  const rawProbe = (l: ModelSaeReport['layers'][number]) =>
    l.probe.find((p) => p.name.endsWith(':raw'))
  const codeProbe = (l: ModelSaeReport['layers'][number]) =>
    l.probe.find((p) => p.name.endsWith(':sae'))
  const deepest = report.layers[report.layers.length - 1]
  const d = modelSaeDigest(report)

  const diffLayers = (report.layers ?? []).filter((l) => l.dropped_vs_embedding)
  const selectedDiffLayer =
    diffLayers.find((l) => l.layer === diffLayer) ?? diffLayers[diffLayers.length - 1]

  return (
    <div className="interp-result msae">
      {/* In a compare column (caption off) the top digest table + Architectures
          strip already carry the header, topology and digest — don't repeat. */}
      {caption && (
        <header className="msae-head">
          <p className="interp-caption">
            {report.model_dir} · {report.dataset_id} · {report.n_train} train / {report.n_test} test
          </p>
          {report.hidden.length > 0 && (
            <div className="mlp-topo-single">
              <MlpTopologyDiagram hidden={report.hidden} activation={report.activation} />
              <span className="mlp-topo-act">{report.activation}</span>
            </div>
          )}
          <p className="msae-digest">
            peak decodable <strong>R²(log) {d.peakR2.toFixed(2)}</strong> · uses up to{' '}
            <strong>{(d.maxUtil * 100).toFixed(0)}%</strong> of a layer · forms{' '}
            <strong>{d.concepts}</strong> interpretable concepts · represents{' '}
            <strong>{d.segRepr}</strong> segments
            {d.dropped != null && (
              <>
                {' '}
                · drops <strong>{d.dropped}</strong> embedding concepts
              </>
            )}
          </p>
        </header>
      )}

      {/* One scannable table for the whole depth story: where the target is decodable
          and where the model forms/uses structure. */}
      <h4>By depth</h4>
      <table className="interp-table msae-depth">
        <thead>
          <tr>
            <th>stage</th>
            <th>dim</th>
            <th>R² log</th>
            <th>R² target</th>
            <th>MAE</th>
            <th>concepts</th>
            <th>width used</th>
          </tr>
        </thead>
        <tbody>
          {report.depth_linear_probe.map((s) => {
            const cap = capByStage.get(s.name)
            const concepts = conceptByStage.get(s.name)
            return (
              <tr key={s.name} className={s.lambda == null ? 'is-output' : undefined}>
                <td>{s.name}</td>
                <td>{s.dim}</td>
                <td>{s.test_r2_log.toFixed(3)}</td>
                <td>{s.test_r2_target.toFixed(3)}</td>
                <td>{Math.round(s.test_mae).toLocaleString()}</td>
                <td>{concepts ?? '—'}</td>
                <td>{cap ? `${(cap.utilization * 100).toFixed(0)}%` : '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="interp-tools-note">
        A layer whose R² barely rises but whose concept count is high is doing generic
        transformation, not building target structure. Read R²(log) for early layers — target-space R²
        can go negative where the log→target exponentiation amplifies a few high-target errors.
      </p>

      {/* The former wall of per-layer tables, now on demand: one expander per
          hidden layer, its key line always visible in the summary. */}
      <div className="msae-layers">
        {report.layers.map((l) => {
          const raw = rawProbe(l)
          const code = codeProbe(l)
          const labeled = l.atoms_by_target_corr.some((a) => !!a.label)
          return (
            <details className="msae-layer" key={l.layer}>
              <summary>
                <span className="msae-layer-name">Hidden layer {l.layer}</span>
                <span className="msae-layer-meta">
                  {l.dim}-d · {(l.capacity.utilization * 100).toFixed(0)}% used ·{' '}
                  {l.n_interpretable_concepts} concepts
                  {raw && code && (
                    <> · sparsify {code.test_r2_target >= raw.test_r2_target ? 'holds' : 'loses'} R²</>
                  )}
                </span>
              </summary>
              <p className="interp-tools-note">
                {l.capacity.active_atoms}/{l.capacity.n_atoms} atoms active ·{' '}
                {l.capacity.dead_atoms} dead · {l.capacity.rare_atoms} rare ·{' '}
                {l.capacity.l0_mean.toFixed(1)} active/row · reconstructs{' '}
                {(l.capacity.var_explained * 100).toFixed(0)}% of variance
                {raw && code && (
                  <>
                    {' '}
                    · raw R² {raw.test_r2_target.toFixed(3)} → code {code.test_r2_target.toFixed(3)}
                  </>
                )}
              </p>
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
                  {l.atoms_by_target_corr.slice(0, 8).map((a) => (
                    <tr key={a.atom}>
                      <td>#{a.atom}</td>
                      <td>{a.target_corr?.toFixed(3) ?? '—'}</td>
                      <td>{(a.freq * 100).toFixed(1)}%</td>
                      {labeled && <td className="sae-label">{a.label ?? '—'}</td>}
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          )
        })}
      </div>

      {deepest && deepest.segments.length > 0 && (
        <section className="msae-section">
          <h4>Segment representation · hidden {deepest.layer}</h4>
          <p className="interp-tools-note">
            Whether the model built a dedicated atom for each one-hot value (best segment-vs-rest
            separation, σ). Unrepresented = smeared across shared atoms.
          </p>
          <table className="interp-table">
            <thead>
              <tr>
                <th>segment</th>
                <th>rows</th>
                <th>represented</th>
                <th>best sep (σ)</th>
              </tr>
            </thead>
            <tbody>
              {deepest.segments.map((s) => (
                <tr key={s.segment}>
                  <td>{s.segment}</td>
                  <td>{s.n_rows.toLocaleString()}</td>
                  <td>{s.represented ? '✓' : '✗'}</td>
                  <td>{s.top_atoms[0]?.separation.toFixed(2) ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {report.embedding_diff.ran && selectedDiffLayer?.dropped_vs_embedding && (
        <section className="msae-section">
          <div className="interp-diff-head">
            <h4>Dropped signal vs. the embedding</h4>
            <label>
              <span>at layer</span>
              <select
                value={selectedDiffLayer.layer}
                onChange={(e) => setDiffLayer(Number(e.target.value))}
              >
                {diffLayers.map((l) => (
                  <option key={l.layer} value={l.layer}>
                    hidden {l.layer} · {l.dim}-d
                    {l.layer === report.layers[report.layers.length - 1].layer ? ' (output head)' : ''}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <p className="interp-tools-note">
            Target-relevant embedding concepts whose best match with any atom here falls below{' '}
            {(report.embedding_diff.match_corr_threshold ?? 0.3).toFixed(2)} —{' '}
            {selectedDiffLayer.dropped_vs_embedding.n_dropped}/
            {selectedDiffLayer.dropped_vs_embedding.n_checked} unmatched. Deeper (narrower) layers
            usually drop more.
          </p>
          {selectedDiffLayer.dropped_vs_embedding.n_dropped === 0 ? (
            <p className="interp-tools-note">
              Every target-relevant embedding concept has an atom tracking it at this layer.
            </p>
          ) : (
            <table className="interp-table">
              <thead>
                <tr>
                  <th>embedding atom</th>
                  <th>target corr</th>
                  <th>best match corr</th>
                  {selectedDiffLayer.dropped_vs_embedding.dropped.some((x) => !!x.label) && (
                    <th>label</th>
                  )}
                </tr>
              </thead>
              <tbody>
                {selectedDiffLayer.dropped_vs_embedding.dropped.map((x) => (
                  <tr key={x.dataset_atom}>
                    <td>#{x.dataset_atom}</td>
                    <td>{x.target_corr.toFixed(3)}</td>
                    <td>{x.best_match_corr.toFixed(3)}</td>
                    {selectedDiffLayer.dropped_vs_embedding!.dropped.some((y) => !!y.label) && (
                      <td className="sae-label">{x.label ?? '—'}</td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      )}
    </div>
  )
}
