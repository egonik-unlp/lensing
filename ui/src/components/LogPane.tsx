import { useEffect, useRef, useState } from 'react'

/** Mono log pane with stick-to-bottom autoscroll. */
export default function LogPane({ lines }: { lines: string[] }) {
  const ref = useRef<HTMLDivElement>(null)
  const [pinned, setPinned] = useState(true)

  useEffect(() => {
    if (pinned && ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [lines, pinned])

  const onScroll = () => {
    const el = ref.current
    if (!el) return
    setPinned(el.scrollHeight - el.scrollTop - el.clientHeight < 24)
  }

  return (
    <div className="log-pane-wrap">
      <div ref={ref} className="log-pane" onScroll={onScroll} role="log" aria-live="polite" aria-label="Run log">
        {lines.length === 0 ? (
          <span className="muted">No log output yet</span>
        ) : (
          lines.map((l, i) => <div key={i}>{l}</div>)
        )}
      </div>
      {!pinned && lines.length > 0 && (
        <button
          className="btn-on-chrome log-pin"
          onClick={() => {
            setPinned(true)
            if (ref.current) ref.current.scrollTop = ref.current.scrollHeight
          }}
        >
          ↓ Jump to latest
        </button>
      )}
    </div>
  )
}
