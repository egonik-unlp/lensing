import { Link, useLocation } from 'react-router-dom'
import { useCompare } from '../lib/CompareContext'
import { shortRunId } from '../lib/format'
import './compare-tray.css'

/* The compare tray: the floating A-vs-B workbench. Appears the moment a run
 * is added from any view, follows across navigation, and expands into
 * /compare. Hidden on /compare itself (the tray has landed). Its action is
 * the luminous button, not amber — amber stays live + page-primary. */

export default function CompareTray() {
  const { sel, toggle, clear } = useCompare()
  const location = useLocation()
  if (sel.length === 0 || location.pathname.startsWith('/compare')) return null

  const [a, b] = sel
  return (
    <div className="compare-tray" role="region" aria-label="Compare tray">
      <span className="compare-tray-label" aria-hidden>
        A·B
      </span>
      <TraySlot side="A" runId={a} onRemove={() => toggle(a)} />
      <span className="compare-tray-vs" aria-hidden>
        vs
      </span>
      <TraySlot side="B" runId={b} onRemove={b ? () => toggle(b) : undefined} />
      {sel.length === 2 ? (
        <Link className="btn btn-bright" to={`/compare?a=${a}&b=${b}`}>
          Compare runs
        </Link>
      ) : (
        <span className="compare-tray-hint">pick a second run</span>
      )}
      <button className="btn-inline compare-tray-clear" onClick={clear}>
        clear
      </button>
    </div>
  )
}

function TraySlot({
  side,
  runId,
  onRemove,
}: {
  side: 'A' | 'B'
  runId?: string
  onRemove?: () => void
}) {
  if (!runId) {
    return (
      <span className="compare-tray-slot is-empty">
        <span className="side-tag-mini" aria-hidden>
          {side}
        </span>
        —
      </span>
    )
  }
  return (
    <span className="compare-tray-slot">
      <span className="side-tag-mini" aria-hidden>
        {side}
      </span>
      <span className="num">{shortRunId(runId)}</span>
      {onRemove && (
        <button className="btn-inline" onClick={onRemove} aria-label={`Remove ${shortRunId(runId)} from comparison`}>
          ✕
        </button>
      )}
    </span>
  )
}
