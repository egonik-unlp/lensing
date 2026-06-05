import type { Param } from '../api/types'
import type { HpState } from '../lib/hyperparams'

interface Props {
  params: Param[]
  state: HpState
  onChange: (name: string, value: string) => void
  disabled?: boolean
}

/** Schema-driven hyperparameter form (registry.toml params). State helpers
 *  (initialHpState, validateHp, hpToJson, …) live in lib/hyperparams. */
export default function HyperparamForm({ params, state, onChange, disabled }: Props) {
  if (params.length === 0) {
    return <p className="muted">This predictor has no hyperparameters.</p>
  }
  return (
    <div className="hp-grid">
      {params.map((p) => {
        const id = `hp-${p.name}`
        const err = state.errors[p.name]
        return (
          <div key={p.name} className="hp-field">
            <label htmlFor={id}>{p.label ?? p.name}</label>
            {p.type === 'bool' ? (
              <input
                id={id}
                type="checkbox"
                checked={state.values[p.name] === 'true'}
                onChange={(e) => onChange(p.name, String(e.target.checked))}
                disabled={disabled}
              />
            ) : p.type === 'enum' ? (
              <select
                id={id}
                value={state.values[p.name]}
                onChange={(e) => onChange(p.name, e.target.value)}
                disabled={disabled}
              >
                {(p.options ?? []).map((o) => (
                  <option key={o} value={o}>
                    {o}
                  </option>
                ))}
              </select>
            ) : p.type === 'json' ? (
              <textarea
                id={id}
                className="mono-input hp-json"
                rows={Math.min(14, Math.max(4, state.values[p.name].split('\n').length + 1))}
                spellCheck={false}
                value={state.values[p.name]}
                onChange={(e) => onChange(p.name, e.target.value)}
                aria-invalid={!!err}
                aria-describedby={err ? `${id}-err` : undefined}
                disabled={disabled}
              />
            ) : (
              <input
                id={id}
                type="text"
                inputMode={p.type === 'ints' ? 'text' : 'decimal'}
                className="mono-input"
                value={state.values[p.name]}
                onChange={(e) => onChange(p.name, e.target.value)}
                aria-invalid={!!err}
                aria-describedby={err ? `${id}-err` : undefined}
                disabled={disabled}
              />
            )}
            {err && (
              <span id={`${id}-err`} className="hp-error" role="alert">
                {err}
              </span>
            )}
          </div>
        )
      })}
    </div>
  )
}
