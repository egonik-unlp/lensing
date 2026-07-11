import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import type { BuildRepresentationRequest, CompressionMethod } from '../api/types'

const METHODS: { value: CompressionMethod; label: string; hint: string }[] = [
  { value: 'pca', label: 'PCA', hint: 'linear; fast; EVR-justified dims' },
  { value: 'autoencoder', label: 'Autoencoder', hint: 'nonlinear; block-wise reconstruction' },
  { value: 'sparse_ae', label: 'Sparse autoencoder', hint: 'nonlinear; sparse latent' },
]

/** Mirrors DatasetBuildForm: pick a source collection + method + latent dim,
 * fit the compressor, write the latent collection. Polls the build like a
 * dataset build and calls `onBuilt(id)` with the new representation id. */
export function RepresentationBuildForm({ onBuilt }: { onBuilt: (id: string) => void }) {
  const [name, setName] = useState('')
  const [method, setMethod] = useState<CompressionMethod>('pca')
  const [latent, setLatent] = useState(64)
  const [epochs, setEpochs] = useState(200)
  const [sparseWeight, setSparseWeight] = useState(0.01)
  const [sinkCollection, setSinkCollection] = useState('')
  const [sourceCollection, setSourceCollection] = useState<string | undefined>(undefined)

  const [collections, setCollections] = useState<string[]>([])
  const [phase, setPhase] = useState<'idle' | 'building' | 'failed'>('idle')
  const [stage, setStage] = useState('')
  const [error, setError] = useState('')
  const timer = useRef<number | undefined>(undefined)

  useEffect(() => {
    api.listCollections().then(
      (r) => {
        setCollections(r.collections)
        setSourceCollection((c) => c ?? r.source)
      },
      () => {
        /* leave the picker empty; the server falls back to its configured collection */
      },
    )
    return () => {
      if (timer.current) window.clearTimeout(timer.current)
    }
  }, [])

  const isAe = method !== 'pca'

  const poll = (buildId: string) => {
    api.getBuild(buildId).then(
      (st) => {
        if (st.state === 'building') {
          setStage(st.stage)
          timer.current = window.setTimeout(() => poll(buildId), 700)
        } else if (st.state === 'done') {
          onBuilt(st.dataset_id)
        } else {
          setPhase('failed')
          setError(st.error)
        }
      },
      (e: unknown) => {
        setPhase('failed')
        setError(e instanceof Error ? e.message : String(e))
      },
    )
  }

  const submit = async () => {
    if (!name.trim()) {
      setError('Name is required.')
      setPhase('failed')
      return
    }
    if (!sinkCollection.trim()) {
      setError('Latent (sink) collection name is required.')
      setPhase('failed')
      return
    }
    setPhase('building')
    setError('')
    setStage('queued')
    const req: BuildRepresentationRequest = {
      name: name.trim(),
      method,
      latent,
      sink_collection: sinkCollection.trim(),
      source_collection: sourceCollection,
      epochs: isAe ? epochs : null,
      hidden: isAe ? [256, latent] : null,
      sparse_weight: method === 'sparse_ae' ? sparseWeight : null,
    }
    try {
      const buildId = await api.buildRepresentation(req)
      poll(buildId)
    } catch (e) {
      setPhase('failed')
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div className="build-form">
      <div className="hp-field">
        <label htmlFor="rf-name">name</label>
        <input id="rf-name" value={name} onChange={(e) => setName(e.target.value)} placeholder="Song AE" />
      </div>

      <div className="hp-field">
        <label htmlFor="rf-method">method</label>
        <select id="rf-method" value={method} onChange={(e) => setMethod(e.target.value as CompressionMethod)}>
          {METHODS.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
        <span className="hp-hint">{METHODS.find((m) => m.value === method)?.hint}</span>
      </div>

      <div className="hp-field">
        <label htmlFor="rf-latent">latent dim</label>
        <input
          id="rf-latent"
          type="number"
          min={1}
          max={4096}
          value={latent}
          onChange={(e) => setLatent(Number(e.target.value))}
        />
      </div>

      {collections.length > 1 && (
        <div className="hp-field">
          <label htmlFor="rf-source">source collection</label>
          <select
            id="rf-source"
            value={sourceCollection ?? ''}
            onChange={(e) => setSourceCollection(e.target.value || undefined)}
          >
            {collections.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>
      )}

      <div className="hp-field">
        <label htmlFor="rf-sink">latent (sink) collection</label>
        <input
          id="rf-sink"
          value={sinkCollection}
          onChange={(e) => setSinkCollection(e.target.value)}
          placeholder="spotify_tracks_song_ae"
        />
        <span className="hp-hint">the new collection the latent vectors are written to</span>
      </div>

      {isAe && (
        <div className="hp-field">
          <label htmlFor="rf-epochs">epochs</label>
          <input
            id="rf-epochs"
            type="number"
            min={1}
            max={5000}
            value={epochs}
            onChange={(e) => setEpochs(Number(e.target.value))}
          />
        </div>
      )}

      {method === 'sparse_ae' && (
        <div className="hp-field">
          <label htmlFor="rf-sparse">sparsity weight</label>
          <input
            id="rf-sparse"
            type="number"
            min={0}
            step={0.001}
            value={sparseWeight}
            onChange={(e) => setSparseWeight(Number(e.target.value))}
          />
        </div>
      )}

      {error && (
        <p className="error-block" role="alert">
          {error}
        </p>
      )}

      <div style={{ marginTop: 12 }}>
        <button className="btn btn-primary" onClick={submit} disabled={phase === 'building'}>
          {phase === 'building' ? `Building… ${stage}` : 'Build representation'}
        </button>
      </div>
    </div>
  )
}
