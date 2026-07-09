// Compare is a gesture, not a destination: any run row can be added to the
// tray (max two — A and B is the comparison primitive), the tray follows
// across views, and expands into /compare. Selection survives navigation via
// sessionStorage but deliberately not across tabs or restarts.

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'

const KEY = 'compare-tray'

interface CompareState {
  /** Selected run ids in pick order: [A] or [A, B]. */
  sel: string[]
  /** Add/remove a run. Adding past two is a no-op (full is exposed). */
  toggle: (runId: string) => void
  clear: () => void
  has: (runId: string) => boolean
  full: boolean
}

const Ctx = createContext<CompareState | null>(null)

function load(): string[] {
  try {
    const raw = sessionStorage.getItem(KEY)
    const v: unknown = raw ? JSON.parse(raw) : []
    return Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string').slice(0, 2) : []
  } catch {
    return []
  }
}

export function CompareProvider({ children }: { children: ReactNode }) {
  const [sel, setSel] = useState<string[]>(load)

  const persist = (next: string[]) => {
    try {
      sessionStorage.setItem(KEY, JSON.stringify(next))
    } catch {
      /* storage full/denied: the tray still works for this page */
    }
    return next
  }

  const toggle = useCallback((runId: string) => {
    setSel((s) =>
      persist(s.includes(runId) ? s.filter((x) => x !== runId) : s.length >= 2 ? s : [...s, runId]),
    )
  }, [])
  const clear = useCallback(() => setSel(() => persist([])), [])

  const value = useMemo<CompareState>(
    () => ({ sel, toggle, clear, has: (id) => sel.includes(id), full: sel.length >= 2 }),
    [sel, toggle, clear],
  )
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

// eslint-disable-next-line react-refresh/only-export-components
export function useCompare(): CompareState {
  const v = useContext(Ctx)
  if (!v) throw new Error('useCompare() called outside CompareProvider')
  return v
}
