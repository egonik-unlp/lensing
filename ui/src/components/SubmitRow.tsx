import type { ReactNode } from 'react'

/* One submit grammar for every form: a primary button that names its action,
 * visible in-progress feedback, the reason it's blocked, and the error inline
 * where the user is looking. */

export default function SubmitRow({
  label,
  busyLabel,
  busy = false,
  disabled = false,
  blockReason,
  error,
  onSubmit,
  children,
}: {
  /** Verb + object: "Start training", "Build dataset". */
  label: string
  /** Shown while busy: "Starting…". Defaults to label. */
  busyLabel?: string
  busy?: boolean
  disabled?: boolean
  /** Why the button is disabled, told instead of guessed. */
  blockReason?: string | null
  error?: string | null
  onSubmit: () => void
  /** Extra quiet actions (cancel, secondary links). */
  children?: ReactNode
}) {
  return (
    <div className="submit-row">
      <button
        type="button"
        className="btn btn-primary"
        onClick={onSubmit}
        disabled={busy || disabled}
        aria-busy={busy}
      >
        {busy ? (busyLabel ?? label) : label}
      </button>
      {children}
      {busy && (
        <span className="muted" role="status">
          working…
        </span>
      )}
      {!busy && blockReason && <span className="muted">{blockReason}</span>}
      {error && (
        <span className="hp-error" role="alert">
          {error}
        </span>
      )}
    </div>
  )
}
