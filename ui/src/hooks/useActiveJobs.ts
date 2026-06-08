import { useEffect, useState } from 'react'
import { api } from '../api/client'

/* Topbar-wide count of in-flight training runs. A slow background poll —
 * faster while anything is live so the chip tracks finishes promptly,
 * lazy when idle so the chrome costs nothing. */
const ACTIVE_POLL_MS = 5000
const IDLE_POLL_MS = 30000

export function useActiveJobs(): number {
  const [running, setRunning] = useState(0)
  useEffect(() => {
    let cancelled = false
    let timer: number | undefined
    const tick = () => {
      api.listRuns().then(
        (runs) => {
          if (cancelled) return
          const n = runs.filter((r) => r.status === 'running').length
          setRunning(n)
          timer = window.setTimeout(tick, n > 0 ? ACTIVE_POLL_MS : IDLE_POLL_MS)
        },
        () => {
          // Chrome stays quiet on errors; the Runs view surfaces them.
          if (!cancelled) timer = window.setTimeout(tick, IDLE_POLL_MS)
        },
      )
    }
    tick()
    return () => {
      cancelled = true
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [])
  return running
}
