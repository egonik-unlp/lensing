import { useMemo } from 'react'
import type { BlendFile, Metrics } from '../api/types'
import { blendSpecFromFile, deriveArch, type ArchFeatures } from '../lib/arch'
import { useAsync } from '../hooks/useAsync'
import { fmtMoney, fmtPct, fmtR2 } from '../lib/format'
import { BlendArch } from './ArchViz'
import { DefinitionRef, ModelRef, PredictorRef } from './EntityRef'
import './blendpanel.css'

/** Blend models get their own panel: the member fan-in diagram plus a
 *  members table — who is inside, at what weight, and how each member tests
 *  solo against the blend. Reads blend.json (written when the run finishes);
 *  while it's absent (running, or a stop before the write) the diagram is
 *  derived from the hyperparams spec instead and the table waits. */
export default function BlendPanel({
  load,
  hyperparams,
  features,
  blendMetrics,
}: {
  /** blend.json fetcher (run or model endpoint). */
  load: () => Promise<BlendFile>
  hyperparams: Record<string, unknown> | null
  features: ArchFeatures | null
  /** The blend's own test metrics, for the comparison row. */
  blendMetrics: Metrics | null
}) {
  const blend = useAsync(load, [])

  const fileSpec = useMemo(
    () => (blend.data ? blendSpecFromFile(blend.data) : null),
    [blend.data],
  )
  const hpSpec = useMemo(
    () => deriveArch('blend', hyperparams, features),
    [hyperparams, features],
  )

  const spec = fileSpec ?? (hpSpec?.kind === 'blend' ? hpSpec : null)
  if (!spec) return null

  const file = blend.data ?? null
  const bestSoloMae = file
    ? Math.min(
        ...file.members
          .filter((m) => m.included && m.solo_metrics)
          .map((m) => m.solo_metrics!.mae),
      )
    : Infinity
  const edge =
    blendMetrics && Number.isFinite(bestSoloMae) ? bestSoloMae - blendMetrics.mae : null

  return (
    <div className="blend-panel">
      <BlendArch spec={spec} title="Blend" />
      {file ? (
        <>
          <table className="blend-members" aria-label="Blend members">
            <thead>
              <tr>
                <th>Member</th>
                <th>Predictor</th>
                <th className="num-col">Weight</th>
                <th className="num-col">Cols</th>
                <th className="num-col col-group-start">MAE solo</th>
                <th className="num-col">medAPE</th>
                <th className="num-col">R²</th>
              </tr>
            </thead>
            <tbody>
              {spec.members.map((m, i) => {
                const rec = file.members[i]
                const sm = rec?.solo_metrics ?? null
                const blocks = rec?.exclude_blocks ?? []
                return (
                  <tr key={m.key} className={`blend-member-row${m.excluded ? ' is-excluded' : ''}`}>
                    <td>
                      <MemberLink kind={m.kind} label={m.label} predictor={rec?.predictor ?? ''} />
                      {m.excluded && <span className="blend-flag">excluded</span>}
                      {blocks.length > 0 && (
                        <span className="blend-blocks num" title={`feature blocks withheld: ${blocks.join(', ')}`}>
                          −{blocks.join(' −')}
                        </span>
                      )}
                    </td>
                    <td className="muted">
                      {rec?.predictor}
                      {m.frozen ? ' · frozen' : ''}
                    </td>
                    <td className="num-col num">
                      <span className="blend-weight">
                        <span className="blend-weight-bar" aria-hidden>
                          <span style={{ width: `${Math.round(m.weight * 100)}%` }} />
                        </span>
                        {spec.rule === 'median' ? 'vote' : fmtPct(m.weight, 0)}
                      </span>
                    </td>
                    <td className="num-col num">{rec?.n_cols ?? '—'}</td>
                    <td className="num-col num col-group-start">{sm ? fmtMoney(sm.mae) : '—'}</td>
                    <td className="num-col num">{sm ? fmtPct(sm.medape) : '—'}</td>
                    <td className="num-col num">{sm ? fmtR2(sm.r2) : '—'}</td>
                  </tr>
                )
              })}
              {blendMetrics && (
                <tr className="blend-total-row">
                  <td>blend</td>
                  <td className="muted">
                    {spec.rule === 'median' ? 'median vote' : 'weighted mean'}
                    {spec.weightFit === 'grid' ? ' · grid-fitted' : ''}
                  </td>
                  <td className="num-col num">{spec.rule === 'median' ? '' : '100%'}</td>
                  <td className="num-col num" />
                  <td className="num-col num col-group-start">{fmtMoney(blendMetrics.mae)}</td>
                  <td className="num-col num">{fmtPct(blendMetrics.medape)}</td>
                  <td className="num-col num">{fmtR2(blendMetrics.r2)}</td>
                </tr>
              )}
            </tbody>
          </table>
          {edge !== null && (
            <p className="blend-verdict num" role="status">
              {edge >= 0
                ? `the blend beats its best member by ${fmtMoney(edge)} MAE`
                : `its best member beats the blend by ${fmtMoney(-edge)} MAE`}
            </p>
          )}
        </>
      ) : blend.loading ? (
        <div className="skeleton" style={{ height: '4rem' }} />
      ) : (
        <p className="muted blend-pending">
          No member record yet — fitted weights and solo metrics are written when training
          finishes.
        </p>
      )}
    </div>
  )
}

function MemberLink({ kind, label, predictor }: { kind: string; label: string; predictor: string }) {
  if (kind === 'model') return <ModelRef name={label} />
  if (kind === 'definition') return <DefinitionRef name={label} />
  return <PredictorRef name={predictor || label} />
}
