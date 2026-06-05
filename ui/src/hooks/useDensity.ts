import { useState } from 'react'

/* Table density, persisted across visits: the hundredth-visit affordance. */

const KEY = 'pg-density'

export type Density = 'comfortable' | 'compact'

export function useDensity(): [Density, () => void] {
  const [density, setDensity] = useState<Density>(() =>
    localStorage.getItem(KEY) === 'compact' ? 'compact' : 'comfortable',
  )
  const toggle = () =>
    setDensity((d) => {
      const next: Density = d === 'compact' ? 'comfortable' : 'compact'
      localStorage.setItem(KEY, next)
      return next
    })
  return [density, toggle]
}
