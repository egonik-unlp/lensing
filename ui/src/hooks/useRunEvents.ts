import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import type { ProgressEvent, RunStatus } from '../api/types'

export interface EpochPoint {
  epoch: number
  total: number
  train: number
  val: number
}

export interface RunEvents {
  epochs: EpochPoint[]
  logs: string[]
  /** The predictor acknowledged a stop request and is finishing up. */
  stopping: boolean
  /** Terminal status from the stream, when the run has ended. */
  terminal: RunStatus | null
  connection: 'connecting' | 'open' | 'reconnecting' | 'closed'
}

/** Subscribe to a run's SSE stream (history replays first, then live). */
export function useRunEvents(runId: string | null): RunEvents {
  const [state, setState] = useState<RunEvents>({
    epochs: [],
    logs: [],
    stopping: false,
    terminal: null,
    connection: 'connecting',
  })
  const buf = useRef<{ epochs: EpochPoint[]; logs: string[] }>({ epochs: [], logs: [] })
  const flushTimer = useRef<number | null>(null)

  useEffect(() => {
    if (!runId) return
    buf.current = { epochs: [], logs: [] }
    // Sync reset is the point: a fresh subscription starts from scratch.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setState({ epochs: [], logs: [], stopping: false, terminal: null, connection: 'connecting' })

    const es = new EventSource(api.eventsUrl(runId))
    let closed = false

    // Batch high-frequency epoch events into ~10 renders/sec.
    const flush = (extra?: Partial<RunEvents>) => {
      flushTimer.current = null
      setState((s) => ({
        ...s,
        epochs: [...buf.current.epochs],
        logs: [...buf.current.logs],
        ...extra,
      }))
    }
    const scheduleFlush = () => {
      if (flushTimer.current == null) {
        flushTimer.current = window.setTimeout(() => flush(), 100)
      }
    }

    es.onopen = () => {
      if (!closed) setState((s) => ({ ...s, connection: 'open' }))
    }
    es.onerror = () => {
      if (!closed) setState((s) => ({ ...s, connection: 'reconnecting' }))
    }
    es.onmessage = (m) => {
      let ev: ProgressEvent
      try {
        ev = JSON.parse(m.data) as ProgressEvent
      } catch {
        return
      }
      switch (ev.event) {
        case 'epoch':
          buf.current.epochs.push({
            epoch: ev.epoch,
            total: ev.total_epochs,
            train: ev.train_loss,
            val: ev.val_loss,
          })
          scheduleFlush()
          break
        case 'log':
          buf.current.logs.push(ev.msg)
          scheduleFlush()
          break
        case 'checkpoint':
          buf.current.logs.push(`checkpoint saved @ epoch ${ev.epoch}`)
          scheduleFlush()
          break
        case 'stopping':
          flush({ stopping: true })
          break
        case 'status':
          closed = true
          es.close()
          flush({ terminal: ev.status, connection: 'closed' })
          break
        case 'done':
          // Terminal meta arrives via the trailing status event.
          break
      }
    }

    return () => {
      closed = true
      es.close()
      if (flushTimer.current != null) window.clearTimeout(flushTimer.current)
    }
  }, [runId])

  return state
}
