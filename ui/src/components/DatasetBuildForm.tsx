import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, pollJob } from '../api/client'
import type {
  AnalyzeResult,
  BuildRequest,
  CollectionValidation,
  CurrencyConfig,
  PreflightResponse,
  PreflightSample,
  QualityFilterConfig,
  RedundancyReport,
} from '../api/types'
import { DEFAULT_CURRENCY, DEFAULT_QUALITY } from '../api/types'
import { fmtTarget, fmtPct } from '../lib/format'
import { ruleLabel } from '../lib/rules'
import { useDomain } from '../lib/DomainContext'
import {
  cappedNumericField,
  fieldLabel,
  vocabTopN,
  type Domain,
  type DomainField,
} from '../lib/domain'
import VarianceChart from './charts/VarianceChart'

// Styles live in views/newrun.css (hp-grid, toggles, quality-*); all view CSS
// is bundled globally, and this form renders both in New Run and /datasets/new.

const DEFAULT_BUILD: BuildRequest = {
  pca_dims: 32,
  test_ratio: 0.2,
  seed: 42,
  log_target: true,
  bedrooms: true,
  property_type: true,
  neighborhood_top_n: 40,
  city: false,
  province: false,
  cluster: false,
  raw_numerics: false,
  area_content_backfill: false,
  coordinates: false,
  impute_numerics: false,
  quality: DEFAULT_QUALITY,
  currency: DEFAULT_CURRENCY,
}

/* ---------------- domain-driven feature toggles ----------------
 * One toggle per encodable field, with grouped fields (e.g. the raw numerics)
 * collapsed under a single group toggle keyed by the group name. The build
 * request carries the generic `fields` / `vocab_top_n` maps. */

interface ToggleDesc {
  /** Key used in the request `fields` map (field name, or group name). */
  key: string
  label: string
  hint: string
  /** Member field names a group toggle stands for. */
  members: string[]
  /** Categorical with a configurable vocabulary cap. */
  vocabKey: string | null
  vocabDefault: number
}

const ROLE_HINT: Partial<Record<DomainField['role'], string>> = {
  numeric: 'numeric',
  categorical: 'one-hot encoded',
  coordinates: 'lat/lon (raw degrees + missing indicator)',
}

function buildToggles(domain: Domain): ToggleDesc[] {
  const out: ToggleDesc[] = []
  const seenGroups = new Set<string>()
  for (const f of domain.fields) {
    if (f.role !== 'categorical' && f.role !== 'numeric' && f.role !== 'coordinates') continue
    if (f.group) {
      if (seenGroups.has(f.group)) continue
      seenGroups.add(f.group)
      const members = domain.fields.filter((g) => g.group === f.group).map((g) => g.name)
      out.push({
        key: f.group,
        label: f.group.replace(/_/g, ' '),
        hint: members.join(', '),
        members,
        vocabKey: null,
        vocabDefault: 0,
      })
      continue
    }
    const top = vocabTopN(f)
    out.push({
      key: f.name,
      label: fieldLabel(f),
      hint: ROLE_HINT[f.role] ?? '',
      members: [f.name],
      vocabKey: f.role === 'categorical' && top != null ? f.name : null,
      vocabDefault: top ?? 0,
    })
  }
  return out
}

function defaultEnabled(domain: Domain, toggles: ToggleDesc[]): Record<string, boolean> {
  const out: Record<string, boolean> = {}
  for (const t of toggles) {
    // A group is on if any member defaults on; a field uses its own default_on.
    const fields = domain.fields.filter((f) => t.members.includes(f.name))
    out[t.key] = fields.some((f) => f.default_on)
  }
  return out
}

function defaultVocab(toggles: ToggleDesc[]): Record<string, number> {
  const out: Record<string, number> = {}
  for (const t of toggles) if (t.vocabKey) out[t.vocabKey] = t.vocabDefault
  return out
}

export function DatasetBuildForm({ onBuilt }: { onBuilt: (id: string) => void }) {
  const domain = useDomain()
  const toggles = useMemo(() => buildToggles(domain), [domain])
  const [enabled, setEnabled] = useState<Record<string, boolean>>(() => defaultEnabled(domain, toggles))
  const [vocabN, setVocabN] = useState<Record<string, number>>(() => defaultVocab(toggles))
  // Cross-field options that aren't a single field's toggle.
  const hasGroups = toggles.some((t) => t.members.length > 1)

  const [req, setReq] = useState<BuildRequest>(DEFAULT_BUILD)
  const [phase, setPhase] = useState<'idle' | 'building' | 'failed'>('idle')
  const [stage, setStage] = useState('')
  const [error, setError] = useState('')
  const timer = useRef<number | null>(null)

  // Source-collection picker (any Qdrant collection; shape is validated below).
  const [collections, setCollections] = useState<string[]>([])
  // Latent collection name → representation id, so the picker can flag a
  // collection that is itself a built representation (build a dataset on it).
  const [latentColls, setLatentColls] = useState<Record<string, string>>({})

  // Shape validation of the selected collection. Auto-runs on every change;
  // hard errors disable Build (the server re-checks, this is just early UX).
  const [validation, setValidation] = useState<CollectionValidation | null>(null)
  const [validating, setValidating] = useState(false)
  const [validateError, setValidateError] = useState('')

  // EVR preview + redundancy. Cached until any config that would change the
  // build's outcome is edited, so the preview always matches a later build.
  const [analysis, setAnalysis] = useState<AnalyzeResult | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [analyzeStage, setAnalyzeStage] = useState('')
  const [analyzeError, setAnalyzeError] = useState('')
  const cancelAnalyze = useRef<(() => void) | null>(null)

  useEffect(() => {
    api.listCollections().then(
      (r) => {
        setCollections(r.collections)
        setReq((q) => (q.collection ? q : { ...q, collection: r.source }))
      },
      () => {},
    )
    // Which source collections are latent representations, so a dataset can be
    // built explicitly on one (its sink collection).
    api.listRepresentations().then(
      (reps) => setLatentColls(Object.fromEntries(reps.map((r) => [r.sink_collection, r.id]))),
      () => {},
    )
  }, [])

  useEffect(
    () => () => {
      if (timer.current != null) window.clearTimeout(timer.current)
      cancelAnalyze.current?.()
    },
    [],
  )

  // Re-validate whenever the source collection changes; the cleanup flag
  // keeps a slow earlier response from clobbering a newer selection.
  useEffect(() => {
    if (!req.collection) return
    let stale = false
    // Begin validating the moment the source collection changes (external sync).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setValidating(true)
    setValidateError('')
    setValidation(null)
    api.validateCollection(req.collection).then(
      (v) => {
        if (stale) return
        setValidation(v)
        setValidating(false)
      },
      (e: unknown) => {
        if (stale) return
        setValidateError(e instanceof Error ? e.message : String(e))
        setValidating(false)
      },
    )
    return () => {
      stale = true
    }
  }, [req.collection])

  const set = <K extends keyof BuildRequest>(k: K, v: BuildRequest[K]) => {
    setReq((r) => ({ ...r, [k]: v }))
    // pca_dims only moves the preview's guide line; everything else changes
    // what the build would actually produce, so the cached preview is stale.
    if (k !== 'pca_dims') setAnalysis(null)
  }

  const setToggle = (key: string, on: boolean) => {
    setEnabled((e) => ({ ...e, [key]: on }))
    setAnalysis(null)
  }
  const setVocab = (key: string, n: number) => {
    setVocabN((v) => ({ ...v, [key]: n }))
    setAnalysis(null)
  }

  // The authoritative generic feature maps, layered onto the request the
  // moment we analyze or build (legacy flags stay for older servers).
  const withFeatures = (r: BuildRequest): BuildRequest => ({
    ...r,
    fields: { ...enabled },
    vocab_top_n: { ...vocabN },
  })

  const analyze = () => {
    setAnalyzing(true)
    setAnalyzeStage('starting')
    setAnalyzeError('')
    cancelAnalyze.current?.()
    api.analyzeDataset(withFeatures(req)).then(
      (jobId) => {
        cancelAnalyze.current = pollJob(jobId, {
          onStage: setAnalyzeStage,
          onDone: (result) => {
            setAnalysis(result as AnalyzeResult)
            setAnalyzing(false)
          },
          onError: (msg) => {
            setAnalyzeError(msg)
            setAnalyzing(false)
          },
        })
      },
      (e: unknown) => {
        setAnalyzeError(e instanceof Error ? e.message : String(e))
        setAnalyzing(false)
      },
    )
  }

  const valid = useMemo(
    () =>
      req.pca_dims >= 1 &&
      req.pca_dims <= 1536 &&
      req.test_ratio >= 0.05 &&
      req.test_ratio <= 0.5 &&
      Object.values(vocabN).every((n) => n >= 0) &&
      (!req.quality.target_range || req.quality.target_min < req.quality.target_max),
    [req, vocabN],
  )

  const poll = (buildId: string) => {
    api.getBuild(buildId).then(
      (st) => {
        if (st.state === 'building') {
          setStage(st.stage)
          timer.current = window.setTimeout(() => poll(buildId), 700)
        } else if (st.state === 'done') {
          setPhase('idle')
          setStage('')
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

  const build = async () => {
    setPhase('building')
    setStage('starting')
    setError('')
    try {
      const buildId = await api.buildDataset(withFeatures(req))
      poll(buildId)
    } catch (e) {
      setPhase('failed')
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const num = (
    label: string,
    key: 'pca_dims' | 'seed',
    hint?: string,
  ) => (
    <div className="hp-field">
      <label htmlFor={`bf-${key}`}>{label}</label>
      <input
        id={`bf-${key}`}
        type="text"
        inputMode="numeric"
        className="mono-input"
        value={String(req[key])}
        onChange={(e) => set(key, Number(e.target.value.replace(/[^\d]/g, '')) as never)}
        disabled={phase === 'building'}
      />
      {hint && <span className="hp-hint">{hint}</span>}
    </div>
  )

  return (
    <div className="build-form">
      {collections.length > 1 && (
        <div className="hp-field">
          <label htmlFor="bf-collection">source collection</label>
          <select
            id="bf-collection"
            className="mono-input"
            value={req.collection ?? ''}
            onChange={(e) => set('collection', e.target.value || undefined)}
            disabled={phase === 'building'}
          >
            {collections.map((c) => (
              <option key={c} value={c}>
                {c}
                {latentColls[c] ? ' — latent representation' : ''}
              </option>
            ))}
          </select>
          <span className="hp-hint">
            any Qdrant collection; shape is checked below. Collections tagged “latent representation”
            are the output of a{' '}
            <Link to="/representations">representation</Link> — build a dataset directly on the latent.
          </span>
        </div>
      )}
      <CollectionShapePanel validation={validation} validating={validating} error={validateError} />
      <div className="hp-grid">
        {num('PCA dims', 'pca_dims', 'embedding components kept')}
        <div className="hp-field">
          <label htmlFor="bf-ratio">test ratio</label>
          <input
            id="bf-ratio"
            type="text"
            inputMode="decimal"
            className="mono-input"
            value={String(req.test_ratio)}
            onChange={(e) => set('test_ratio', Number(e.target.value) || 0)}
            disabled={phase === 'building'}
          />
          <span className="hp-hint">0.05 – 0.5</span>
        </div>
        {num('seed', 'seed', 'split reproducibility')}
      </div>
      <fieldset className="toggles" disabled={phase === 'building'}>
        <legend>Features</legend>
        <label className="toggle">
          <input
            type="checkbox"
            checked={req.log_target}
            onChange={(e) => set('log_target', e.target.checked)}
          />
          log1p {domain.project.target_noun}
          <span className="hp-hint">train on log scale for a long-tailed target</span>
        </label>
        {toggles.map((t) => (
          <div key={t.key}>
            <label className="toggle">
              <input
                type="checkbox"
                checked={enabled[t.key] ?? false}
                onChange={(e) => setToggle(t.key, e.target.checked)}
              />
              {t.label}
              {t.hint && <span className="hp-hint">{t.hint}</span>}
            </label>
            {t.vocabKey && enabled[t.key] && (
              <div className="hp-field quality-threshold">
                <label htmlFor={`bf-vocab-${t.vocabKey}`}>{t.label} top-N</label>
                <input
                  id={`bf-vocab-${t.vocabKey}`}
                  type="text"
                  inputMode="numeric"
                  className="mono-input"
                  value={String(vocabN[t.vocabKey] ?? 0)}
                  onChange={(e) =>
                    setVocab(t.vocabKey!, Number(e.target.value.replace(/[^\d]/g, '')) || 0)
                  }
                />
                <span className="hp-hint">0 keeps all values</span>
              </div>
            )}
          </div>
        ))}
        {hasGroups && (
          <label className="toggle">
            <input
              type="checkbox"
              checked={req.area_content_backfill}
              onChange={(e) => set('area_content_backfill', e.target.checked)}
            />
            area backfill from text
            <span className="hp-hint">fill missing areas from “… m²” mentions in the listing</span>
          </label>
        )}
        {hasGroups && (
          <label className="toggle">
            <input
              type="checkbox"
              checked={req.impute_numerics}
              onChange={(e) => set('impute_numerics', e.target.checked)}
            />
            impute missing numerics
            <span className="hp-hint">
              train-split medians instead of zero + indicator (for kernel models)
            </span>
          </label>
        )}
      </fieldset>
      <CurrencyPanel
        currency={req.currency}
        collections={collections}
        onChange={(c) => set('currency', c)}
        disabled={phase === 'building'}
      />
      <QualityPanel
        quality={req.quality}
        currency={req.currency}
        collection={req.collection ?? undefined}
        onChange={(q) => set('quality', q)}
        disabled={phase === 'building'}
      />

      <div className="analyze-panel">
        <div className="quality-preview-row">
          <button
            type="button"
            className="btn"
            onClick={analyze}
            disabled={analyzing || phase === 'building' || validation?.ok_to_build === false}
          >
            {analyzing ? 'Analyzing…' : 'Analyze components'}
          </button>
          {analyzing && (
            <span className="muted num" role="status">
              {analyzeStage}
            </span>
          )}
          {analyzeError && (
            <span className="hp-error" role="alert">
              {analyzeError}
            </span>
          )}
        </div>
        {analysis && (
          <div className="analyze-result">
            <VarianceChart evr={analysis.evr} markK={req.pca_dims} />
            <div className="hp-field">
              <label htmlFor="bf-k">
                k = {req.pca_dims} captures{' '}
                {fmtPct(analysis.cumulative_evr[Math.min(req.pca_dims, analysis.cumulative_evr.length) - 1] ?? 0, 1)} of
                variance
              </label>
              <input
                id="bf-k"
                type="range"
                min={1}
                max={Math.min(1536, analysis.evr.length)}
                value={req.pca_dims}
                onChange={(e) => set('pca_dims', Number(e.target.value))}
                disabled={phase === 'building'}
              />
              <span className="hp-hint">
                analyzed {analysis.n_rows.toLocaleString()} rows
                {analysis.n_excluded > 0 ? ` (${analysis.n_excluded.toLocaleString()} excluded by quality)` : ''}
              </span>
            </div>
            <RedundancyFindings report={analysis.redundancy} />
          </div>
        )}
      </div>

      <div className="build-actions">
        <button
          className="btn btn-primary"
          onClick={build}
          disabled={!valid || phase === 'building' || validation?.ok_to_build === false}
          aria-busy={phase === 'building'}
        >
          {phase === 'building' ? 'Building…' : 'Build dataset'}
        </button>
        {phase === 'building' && (
          <span className="muted build-stage" role="status">
            <span className="status-running status">
              <span className="dot">●</span>
            </span>
            <span className="num">{stage}</span>
          </span>
        )}
        {phase === 'failed' && (
          <span className="hp-error" role="alert">
            Build failed: {error}
          </span>
        )}
      </div>
    </div>
  )
}

/* ---------------- collection shape validation ---------------- */

function CollectionShapePanel({
  validation,
  validating,
  error,
}: {
  validation: CollectionValidation | null
  validating: boolean
  error: string
}) {
  if (validating) {
    return (
      <p className="muted num" role="status">
        checking collection shape…
      </p>
    )
  }
  if (error) {
    return (
      <p className="hp-error" role="alert">
        Shape check failed: {error}
      </p>
    )
  }
  if (!validation) return null

  const v = validation
  return (
    <div className="quality-preview" role="status">
      {v.ok_to_build && (
        <p className="num">
          {(v.count_filtered ?? 0).toLocaleString()} matching points
          {v.vector ? ` · ${v.vector.size}d ${v.vector.distance} vectors` : ''}
        </p>
      )}
      {v.errors.map((e) => (
        <p key={e} className="hp-error" role="alert">
          {e}
        </p>
      ))}
      {v.warnings.map((w) => (
        <p key={w} className="hp-hint">
          ⚠ {w}
        </p>
      ))}
      {v.coverage && (
        <details className="quality-samples">
          <summary>field coverage ({v.sample_size} sampled)</summary>
          <ul className="quality-counts">
            {Object.entries(v.coverage).map(([key, frac]) => (
              <li key={key} className="num">
                {key}: {fmtPct(frac, 0)}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  )
}

/* ---------------- feature redundancy ---------------- */

export function RedundancyFindings({ report }: { report: RedundancyReport }) {
  const { near_zero_variance, correlated_pairs, onehot_groups } = report
  const notableGroups = onehot_groups.filter((g) => g.other_fraction > 0.2 || g.rare_buckets > 0)
  const nothing =
    near_zero_variance.length === 0 && correlated_pairs.length === 0 && notableGroups.length === 0

  return (
    <div className="redundancy">
      <h4>Feature redundancy</h4>
      {nothing && <p className="muted">No notable redundancy in the metadata features.</p>}
      {near_zero_variance.length > 0 && (
        <p className="num redundancy-line">
          Near-zero variance: {near_zero_variance.map((c) => c.column).join(', ')}
        </p>
      )}
      {correlated_pairs.length > 0 && (
        <ul className="num redundancy-list">
          {correlated_pairs.slice(0, 20).map((p, i) => (
            <li key={i}>
              {p.a} ↔ {p.b}: r = {p.corr.toFixed(2)}
            </li>
          ))}
        </ul>
      )}
      {notableGroups.length > 0 && (
        <ul className="num redundancy-list">
          {notableGroups.map((g) => (
            <li key={g.group}>
              {g.group}: {g.n_values} values · {fmtPct(g.other_fraction, 0)} in __other__ · {g.rare_buckets} rare
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/* ---------------- currency handling ---------------- */

/** Display host of the domain's rate endpoint, for the hint line. */
function rateHost(template: string): string {
  try {
    return new URL(template.replace('{source}', 'x')).host
  } catch {
    return template
  }
}

function CurrencyPanel({
  currency,
  collections,
  onChange,
  disabled,
}: {
  currency: CurrencyConfig
  /** Qdrant collections, for the reconcile source-of-truth picker. */
  collections: string[]
  onChange: (c: CurrencyConfig) => void
  disabled: boolean
}) {
  const domain = useDomain()
  const pair = domain.currency?.pair
  const set = <K extends keyof CurrencyConfig>(k: K, v: CurrencyConfig[K]) =>
    onChange({ ...currency, [k]: v })

  return (
    <fieldset className="toggles quality-panel" disabled={disabled}>
      <legend>Currency</legend>
      <p className="muted quality-blurb">
        The corpus mixes {pair ? `${pair[0]} and ${pair[1]}` : 'multiple'} values. Currency
        is reconciled by point id from a companion collection, then foreign-currency rows
        are dropped or converted before the quality rules run.
      </p>
      <div className="hp-field">
        <label htmlFor="cur-mode">mode</label>
        <select
          id="cur-mode"
          className="mono-input"
          value={currency.mode}
          onChange={(e) => set('mode', e.target.value as CurrencyConfig['mode'])}
        >
          <option value="filter">keep {currency.keep} only (drop others)</option>
          <option value="convert">convert to {currency.keep} @ per-date rate</option>
          <option value="off">off (ignore currency)</option>
        </select>
        <span className="hp-hint">the target is conventionally denominated in {currency.keep}</span>
      </div>
      {currency.mode !== 'off' && (
        <div className="hp-field">
          <label htmlFor="cur-reconcile">currency source-of-truth</label>
          <select
            id="cur-reconcile"
            className="mono-input"
            value={currency.reconcile_collection ?? ''}
            onChange={(e) => set('reconcile_collection', e.target.value || null)}
          >
            <option value="">(none — use inline currency)</option>
            {collections.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <span className="hp-hint">joined by point id; derived collections dropped currency</span>
        </div>
      )}
      {currency.mode === 'convert' && (
        <div className="hp-field">
          <label htmlFor="cur-rate">exchange-rate series</label>
          <select
            id="cur-rate"
            className="mono-input"
            value={currency.rate_source}
            onChange={(e) => set('rate_source', e.target.value as CurrencyConfig['rate_source'])}
          >
            <option value="blue">blue</option>
            <option value="oficial">oficial</option>
          </select>
          <span className="hp-hint">
            daily rate keyed on {domain.currency?.date_field ?? 'the timestamp field'}
            {domain.currency ? ` · ${rateHost(domain.currency.rate_url_template)}` : ''}
          </span>
        </div>
      )}
    </fieldset>
  )
}

/* ---------------- data quality ---------------- */

function QualityPanel({
  quality,
  currency,
  collection,
  onChange,
  disabled,
}: {
  quality: QualityFilterConfig
  /** Currency handling; reconciled before the rules run, so it changes counts. */
  currency: CurrencyConfig
  /** Source collection the build will read; the preview must scan the same one. */
  collection?: string
  onChange: (q: QualityFilterConfig) => void
  disabled: boolean
}) {
  const domain = useDomain()
  const targetNoun = domain.project.target_noun
  const cappedDesc = cappedNumericField(domain)
  const cappedLabel = cappedDesc ? fieldLabel(cappedDesc) : 'capped numeric'
  const outlierGroupDesc = domain.fields.find((f) => f.name === domain.quality.outlier_group)
  const outlierGroupLabel = outlierGroupDesc ? fieldLabel(outlierGroupDesc) : null
  const criticalLabels = domain.fields
    .filter((f) => f.critical)
    .map(fieldLabel)
    .join(' / ')
  // Mirrors the server's sample-field choice: critical fields, the outlier
  // group and the capped numeric, deduplicated.
  const sampleFields = useMemo(() => {
    const names = domain.fields.filter((f) => f.critical).map((f) => f.name)
    for (const extra of [domain.quality.outlier_group, domain.quality.capped_numeric]) {
      if (extra && !names.includes(extra)) names.push(extra)
    }
    return names.flatMap((n) => domain.fields.find((f) => f.name === n) ?? [])
  }, [domain])
  const sampleLine = (s: PreflightSample): string => {
    const parts: string[] = []
    for (const f of sampleFields) {
      const v = s[f.name]
      parts.push(v != null && v !== '' && v !== 0 ? String(v) : `(no ${fieldLabel(f)})`)
    }
    const t = s[domain.target.field]
    if (typeof t === 'number') parts.push(fmtTarget(t, domain))
    const cur = domain.currency ? s[domain.currency.currency_field] : null
    if (typeof cur === 'string' && cur) parts.push(cur)
    return parts.join(' · ')
  }
  const [preview, setPreview] = useState<PreflightResponse | null>(null)
  const [previewing, setPreviewing] = useState(false)
  const [previewError, setPreviewError] = useState<string | null>(null)

  // Counts are also stale when the source collection or the currency config
  // changes (reset during render, matching ArchViz).
  const currencyKey = JSON.stringify(currency)
  const [lastSource, setLastSource] = useState(`${collection}|${currencyKey}`)
  if (`${collection}|${currencyKey}` !== lastSource) {
    setLastSource(`${collection}|${currencyKey}`)
    setPreview(null)
  }

  const set = <K extends keyof QualityFilterConfig>(k: K, v: QualityFilterConfig[K]) => {
    onChange({ ...quality, [k]: v })
    setPreview(null) // counts are stale once the config changes
  }

  const runPreview = async () => {
    setPreviewing(true)
    setPreviewError(null)
    try {
      setPreview(await api.preflight(quality, currency, 8, collection))
    } catch (e) {
      setPreviewError(e instanceof Error ? e.message : String(e))
    } finally {
      setPreviewing(false)
    }
  }

  const numField = (
    id: string,
    label: string,
    key: 'target_outlier_mad_z' | 'target_min' | 'target_max' | 'capped_numeric_max' | 'short_content_min_chars',
    hint: string,
  ) => (
    <div className="hp-field quality-threshold">
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        type="text"
        inputMode="decimal"
        className="mono-input"
        value={String(quality[key])}
        onChange={(e) => set(key, (Number(e.target.value.replace(/[^\d.]/g, '')) || 0) as never)}
      />
      <span className="hp-hint">{hint}</span>
    </div>
  )

  return (
    <fieldset className="toggles quality-panel" disabled={disabled}>
      <legend>Data quality</legend>
      <p className="muted quality-blurb">
        Suspicious {domain.project.entity_noun_plural} are excluded before the split, the PCA fit
        and the vocabularies; the applied rules are recorded in the dataset manifest.
      </p>
      <label className="toggle">
        <input
          type="checkbox"
          checked={quality.nonpositive_target}
          onChange={(e) => set('nonpositive_target', e.target.checked)}
        />
        drop {targetNoun} ≤ 0
        <span className="hp-hint">rows without a {targetNoun} poison the target</span>
      </label>
      <label className="toggle">
        <input
          type="checkbox"
          checked={quality.target_outlier}
          onChange={(e) => set('target_outlier', e.target.checked)}
        />
        drop {targetNoun} outliers
        <span className="hp-hint">
          MAD z-score on log {targetNoun}
          {outlierGroupLabel ? `, per ${outlierGroupLabel}` : ''}
        </span>
      </label>
      {quality.target_outlier &&
        numField('qf-madz', 'outlier threshold (MAD z)', 'target_outlier_mad_z', '3.5 is conservative; lower drops more')}
      <label className="toggle">
        <input
          type="checkbox"
          checked={quality.target_range}
          onChange={(e) => set('target_range', e.target.checked)}
        />
        drop {targetNoun} out of range
        <span className="hp-hint">hard caps; catches noise the statistics miss</span>
      </label>
      {quality.target_range && (
        <div className="quality-range">
          {numField('qf-pmin', `min ${targetNoun}`, 'target_min', 'below this is noise')}
          {numField('qf-pmax', `max ${targetNoun}`, 'target_max', 'above this is noise or mistyped')}
        </div>
      )}
      {cappedDesc && (
        <label className="toggle">
          <input
            type="checkbox"
            checked={quality.capped_numeric_outlier}
            onChange={(e) => set('capped_numeric_outlier', e.target.checked)}
          />
          drop {cappedLabel} outliers
          <span className="hp-hint">negative or above the cap; 0 means unspecified and is kept</span>
        </label>
      )}
      {cappedDesc &&
        quality.capped_numeric_outlier &&
        numField('qf-bmax', `max ${cappedLabel}`, 'capped_numeric_max', 'above this is likely a typo')}
      <label className="toggle">
        <input
          type="checkbox"
          checked={quality.duplicate_content}
          onChange={(e) => set('duplicate_content', e.target.checked)}
        />
        drop duplicate {domain.project.entity_noun_plural}
        <span className="hp-hint">identical text; duplicates double-count in PCA and split</span>
      </label>
      <label className="toggle">
        <input
          type="checkbox"
          checked={quality.short_content}
          onChange={(e) => set('short_content', e.target.checked)}
        />
        drop thin descriptions
        <span className="hp-hint">little text means little embedding signal</span>
      </label>
      {quality.short_content &&
        numField(
          'qf-minchars',
          'min characters',
          'short_content_min_chars',
          `${domain.project.entity_noun_plural} shorter than this are dropped`,
        )}
      <label className="toggle">
        <input
          type="checkbox"
          checked={quality.missing_fields}
          onChange={(e) => set('missing_fields', e.target.checked)}
        />
        drop missing critical fields
        <span className="hp-hint">
          empty {criticalLabels || 'critical fields'}; training tolerates them
        </span>
      </label>

      <div className="quality-preview-row">
        <button type="button" className="btn" onClick={runPreview} disabled={previewing}>
          {previewing ? 'Scanning corpus…' : 'Preview impact'}
        </button>
        {previewError && (
          <span className="hp-error" role="alert">
            {previewError}
          </span>
        )}
      </div>

      {preview && (
        <div className="quality-preview" role="status">
          <p className="num">
            {preview.n_excluded_total.toLocaleString()} of {preview.n_total.toLocaleString()} rows
            would be excluded
          </p>
          {preview.currency.n_converted > 0 && (
            <p className="num">
              {preview.currency.n_converted.toLocaleString()} rows converted to{' '}
              {preview.currency.config.keep}
              {preview.currency.rate_min != null &&
                ` @ ${Math.round(preview.currency.rate_min)}–${Math.round(preview.currency.rate_max ?? 0)} ${preview.currency.config.rate_source}`}
            </p>
          )}
          {preview.currency.n_missing > 0 && (
            <p className="hp-hint">
              ⚠ {preview.currency.n_missing.toLocaleString()} rows have no currency after
              reconciliation (kept)
            </p>
          )}
          <ul className="quality-counts">
            {preview.rules.map((r) => (
              <li key={r.rule} className="num">
                {ruleLabel(r.rule, domain)}: {r.n_flagged.toLocaleString()} flagged
                {r.n_excluded > 0 ? (
                  ', excluded'
                ) : (
                  <span className="muted"> (rule off, kept)</span>
                )}
              </li>
            ))}
          </ul>
          {Object.entries(preview.samples).map(([rule, samples]) =>
            samples.length > 0 ? (
              <details key={rule} className="quality-samples">
                <summary>
                  sample flagged: {ruleLabel(rule, domain)} ({samples.length})
                </summary>
                <ul>
                  {samples.map((s) => (
                    <li key={s.row_id} className="num">
                      #{s.row_id} · {sampleLine(s)}
                    </li>
                  ))}
                </ul>
              </details>
            ) : null,
          )}
        </div>
      )}
    </fieldset>
  )
}
