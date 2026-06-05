import { useEffect, useState } from 'react'

import { api } from '../api/client'
import type { Predictor } from '../api/types'

/* The predictor registry is a small, fixed list (registry.toml): fetch it
 * once per page load and share it across every component that wants to
 * decorate a predictor name — most notably the ImplBadge in EntityRef.
 * Failures stay null; callers degrade to showing nothing extra. */

let cached: Predictor[] | null = null
let pending: Promise<Predictor[] | null> | null = null

function fetchOnce(): Promise<Predictor[] | null> {
  pending ??= api.listPredictors().then(
    (p) => (cached = p),
    () => null,
  )
  return pending
}

/** The registry entry for `name`, or null while loading / if unknown. */
export function usePredictorInfo(name: string | undefined): Predictor | null {
  const [list, setList] = useState(cached)
  useEffect(() => {
    if (cached) return
    let alive = true
    void fetchOnce().then((p) => {
      if (alive && p) setList(p)
    })
    return () => {
      alive = false
    }
  }, [])
  if (!name) return null
  return list?.find((p) => p.name === name) ?? null
}
