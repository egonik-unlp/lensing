import type { RunStatus } from '../api/types'

const GLYPH: Record<RunStatus, string> = {
  queued: '…',
  running: '●',
  succeeded: '✓',
  failed: '✕',
  interrupted: '◌',
  stopped: '■',
}

/** Status word + glyph: state is never encoded by color alone. */
export default function StatusBadge({ status }: { status: RunStatus }) {
  return (
    <span className={`status status-${status}`}>
      <span className="dot" aria-hidden>
        {GLYPH[status]}
      </span>
      {status}
    </span>
  )
}
