// Hyperparameter form state helpers (registry.toml param schemas).
// Separate from the HyperparamForm component so that file only exports a
// component (react-refresh) and non-form code can reuse the conversions.

import type { Param } from '../api/types'

export interface HpState {
  values: Record<string, string>
  errors: Record<string, string>
}

function toRaw(p: Param, v: unknown): string {
  if (p.type === 'ints' && Array.isArray(v)) return v.join(', ')
  if (p.type === 'json') return JSON.stringify(v, null, 2)
  return String(v)
}

export function initialHpState(params: Param[]): HpState {
  const values: Record<string, string> = {}
  for (const p of params) {
    values[p.name] = toRaw(p, p.default)
  }
  return { values, errors: {} }
}

/** Seed form state from concrete values (a model definition, a past run),
 *  falling back to schema defaults for any missing key. */
export function hpStateFromValues(params: Param[], values: Record<string, unknown>): HpState {
  const out: Record<string, string> = {}
  for (const p of params) {
    out[p.name] = toRaw(p, values[p.name] ?? p.default)
  }
  return { values: out, errors: {} }
}

export function validateHp(params: Param[], values: Record<string, string>): HpState {
  const errors: Record<string, string> = {}
  for (const p of params) {
    const raw = (values[p.name] ?? '').trim()
    if (p.type === 'bool' || p.type === 'enum') continue
    if (raw === '') {
      errors[p.name] = 'Required'
      continue
    }
    if (p.type === 'int' || p.type === 'float') {
      const n = Number(raw)
      if (!isFinite(n)) errors[p.name] = 'Not a number'
      else if (p.type === 'int' && !Number.isInteger(n)) errors[p.name] = 'Must be an integer'
      else if (p.min != null && n < p.min) errors[p.name] = `Min ${p.min}`
      else if (p.max != null && n > p.max) errors[p.name] = `Max ${p.max}`
    } else if (p.type === 'ints') {
      const parts = raw.split(/[,\s]+/).filter(Boolean)
      if (parts.length === 0 || parts.some((s) => !/^\d+$/.test(s) || Number(s) < 1)) {
        errors[p.name] = 'Comma-separated positive integers, e.g. 256, 128'
      }
    } else if (p.type === 'json') {
      try {
        JSON.parse(raw)
      } catch (e) {
        errors[p.name] = `Invalid JSON: ${e instanceof Error ? e.message : String(e)}`
      }
    }
  }
  return { values, errors }
}

export function hpToJson(params: Param[], values: Record<string, string>): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const p of params) {
    const raw = (values[p.name] ?? '').trim()
    switch (p.type) {
      case 'int':
        out[p.name] = Math.round(Number(raw))
        break
      case 'float':
        out[p.name] = Number(raw)
        break
      case 'bool':
        out[p.name] = raw === 'true'
        break
      case 'ints':
        out[p.name] = raw.split(/[,\s]+/).filter(Boolean).map(Number)
        break
      case 'enum':
        out[p.name] = raw
        break
      case 'json':
        out[p.name] = JSON.parse(raw)
        break
    }
  }
  return out
}
