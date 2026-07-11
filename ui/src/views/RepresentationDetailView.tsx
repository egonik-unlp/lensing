import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { api, ApiError } from '../api/client'
import type { RepresentationQuality } from '../api/types'
import ViewHeader from '../components/ViewHeader'
import { useAsync } from '../hooks/useAsync'
import { fmtDateTime, fmtPct } from '../lib/format'
import { useDocTitle } from '../lib/DomainContext'
import { methodLabel } from '../lib/representation'

/** Per-block reconstruction R² as labelled horizontal bars (the AE metric that
 * used to only print to stdout). Numeric label + aria-label, never color alone. */
function BlockR2Bars({ blocks }: { blocks: [string, number][] }) {
  return (
    <div
      role="img"
      aria-label={
        'Per-block reconstruction R²: ' + blocks.map(([n, v]) => `${n} ${v.toFixed(2)}`).join(', ')
      }
    >
      {blocks.map(([name, v]) => (
        <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '4px 0' }}>
          <span style={{ width: 110, textAlign: 'right' }} className="muted">
            {name}
          </span>
          <div style={{ flex: 1, background: 'var(--surface-2, #eee)', borderRadius: 3, height: 14 }}>
            <div
              style={{
                width: `${Math.max(0, Math.min(1, v)) * 100}%`,
                background: 'var(--accent, #d1622b)',
                height: '100%',
                borderRadius: 3,
              }}
            />
          </div>
          <span className="num" style={{ width: 48 }}>
            {v.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  )
}

/** Cumulative EVR for PCA — the captured-variance curve as labelled steps. */
function EvrCurve({ cumulative, captured }: { cumulative: number[]; captured: number }) {
  const step = Math.max(1, Math.floor(cumulative.length / 24))
  const pts = cumulative.filter((_, i) => i % step === 0 || i === cumulative.length - 1)
  return (
    <div
      role="img"
      aria-label={`Cumulative explained variance reaches ${fmtPct(captured, 0)} at the latent dim`}
    >
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2, height: 80 }}>
        {pts.map((v, i) => (
          <div
            key={i}
            title={fmtPct(v, 0)}
            style={{ flex: 1, height: `${v * 100}%`, background: 'var(--accent, #d1622b)', borderRadius: 2 }}
          />
        ))}
      </div>
      <p className="muted">Captured at latent dim: {fmtPct(captured, 1)}</p>
    </div>
  )
}

function QualityPanel({ quality }: { quality: RepresentationQuality }) {
  if (!quality) return <p className="muted">No quality metric recorded.</p>
  if (quality.kind === 'evr') {
    return <EvrCurve cumulative={quality.cumulative_evr} captured={quality.captured} />
  }
  return <BlockR2Bars blocks={quality.blocks} />
}

function EncodePlayground({ id, method }: { id: string; method: string }) {
  const [input, setInput] = useState('')
  const [latent, setLatent] = useState<number[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const run = () => {
    const vector = input
      .split(/[\s,]+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0)
      .map(Number)
    if (vector.some((n) => Number.isNaN(n))) {
      setError('Input must be a list of numbers (comma- or space-separated).')
      return
    }
    setBusy(true)
    setError(null)
    api.encodeRepresentation(id, vector).then(
      (r) => {
        setLatent(r.latent)
        setBusy(false)
      },
      (e: unknown) => {
        setError(e instanceof ApiError ? e.message : String(e))
        setBusy(false)
      },
    )
  }

  return (
    <div>
      <h2 className="section-title">Encode an item</h2>
      {method !== 'pca' ? (
        <p className="muted">
          One-shot encode is currently available for PCA representations. Autoencoder encoders are
          exported to ONNX (see below) for portable inference.
        </p>
      ) : (
        <>
          <p className="muted">Paste a source vector to project it into the latent space.</p>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            rows={3}
            style={{ width: '100%', fontFamily: 'monospace' }}
            placeholder="0.12, -0.03, 0.44, …"
            aria-label="Source vector to encode"
          />
          <div style={{ marginTop: 8 }}>
            <button className="btn btn-primary" onClick={run} disabled={busy}>
              {busy ? 'Encoding…' : 'Encode'}
            </button>
          </div>
          {error && (
            <p className="error-block" role="alert">
              {error}
            </p>
          )}
          {latent && (
            <pre style={{ overflowX: 'auto', marginTop: 8 }}>
              [{latent.map((v) => v.toFixed(4)).join(', ')}]
            </pre>
          )}
        </>
      )}
    </div>
  )
}

export default function RepresentationDetailView() {
  const { id = '' } = useParams()
  const rep = useAsync(() => api.getRepresentation(id), [id])
  useDocTitle(rep.data ? rep.data.name : 'Representation')

  if (rep.error) {
    return (
      <div className="view-body">
        <div className="error-block" role="alert">
          Could not load representation: {rep.error}
        </div>
      </div>
    )
  }
  if (!rep.data) {
    return (
      <div className="view-body">
        <p className="muted">Loading…</p>
      </div>
    )
  }
  const r = rep.data

  return (
    <section aria-label={`Representation ${r.name}`}>
      <ViewHeader glyph="rp" title={r.name} meta={<span>{methodLabel[r.method]} · latent dim {r.latent_dim}</span>} />
      <div className="view-body">
        <div className="metrics-strip" role="group" aria-label="Representation summary">
          <div>
            <div className="muted">method</div>
            <div>{methodLabel[r.method]}</div>
          </div>
          <div>
            <div className="muted">source → latent</div>
            <div className="num">
              {r.source_collection} → {r.sink_collection}
            </div>
          </div>
          <div>
            <div className="muted">latent dim</div>
            <div className="num">{r.latent_dim}</div>
          </div>
          <div>
            <div className="muted">points</div>
            <div className="num">{r.n_points.toLocaleString()}</div>
          </div>
          <div>
            <div className="muted">built</div>
            <div className="num">{fmtDateTime(r.created_at)}</div>
          </div>
        </div>

        <h2 className="section-title">
          {r.quality?.kind === 'evr' ? 'Explained variance' : 'Reconstruction quality'}
        </h2>
        <QualityPanel quality={r.quality} />

        <h2 className="section-title">Encoder</h2>
        <p className="muted">
          The encoder is exported to ONNX (<code>encoder.onnx</code>) under this representation's
          artifact directory for portable inference outside the framework.
        </p>

        <EncodePlayground id={r.id} method={r.method} />
      </div>
    </section>
  )
}
