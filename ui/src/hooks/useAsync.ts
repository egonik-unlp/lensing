import { useCallback, useEffect, useRef, useState } from 'react'

export interface AsyncState<T> {
  data: T | null
  error: string | null
  loading: boolean
  reload: () => void
}

/** Fetch-on-mount with reload; ignores stale responses. */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): AsyncState<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [nonce, setNonce] = useState(0)
  const seq = useRef(0)

  useEffect(() => {
    const id = ++seq.current
    // Sync reset is the point: a new fetch begins the moment deps change.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLoading(true)
    setError(null)
    fn().then(
      (d) => {
        if (seq.current === id) {
          setData(d)
          setLoading(false)
        }
      },
      (e: unknown) => {
        if (seq.current === id) {
          setError(e instanceof Error ? e.message : String(e))
          setLoading(false)
        }
      },
    )
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce])

  const reload = useCallback(() => setNonce((n) => n + 1), [])
  return { data, error, loading, reload }
}
