import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'

/* The slate band: every view opens with one. The bezel carries identity
 * (breadcrumb, glyph, title, count), the dim meta line, and at most one
 * orange primary action; the data below stays on paper. */

export interface Crumb {
  label: ReactNode
  to?: string
}

export default function ViewHeader({
  glyph,
  title,
  count,
  crumbs,
  meta,
  actions,
}: {
  /** EntityRef-alphabet kind glyph (run / ds / ml / def / pr). */
  glyph?: string
  title: ReactNode
  count?: number | string
  /** Lineage for detail pages: where this entity hangs in the pipeline. */
  crumbs?: Crumb[]
  /** Dim meta line under the title (status, dataset, timing…). */
  meta?: ReactNode
  /** Right-hand slot. At most one .btn-primary lives here. */
  actions?: ReactNode
}) {
  return (
    <div className="view-band">
      <div className="view-band-inner">
        {crumbs && crumbs.length > 0 && (
          <nav className="breadcrumb" aria-label="Breadcrumb">
            {crumbs.map((c, i) => (
              <span key={i} className="crumb">
                {i > 0 && (
                  <span className="crumb-sep" aria-hidden>
                    {' / '}
                  </span>
                )}
                {c.to ? <Link to={c.to}>{c.label}</Link> : c.label}
              </span>
            ))}
          </nav>
        )}
        <div className="view-band-row">
          <h1>
            {glyph && (
              <>
                <span className="view-glyph" aria-hidden>
                  {glyph}
                </span>{' '}
              </>
            )}
            {title}
            {count !== undefined && (
              <>
                {' '}
                <span className="view-count num">{count}</span>
              </>
            )}
          </h1>
          {actions && <span className="view-band-actions">{actions}</span>}
        </div>
        {meta && <div className="view-band-meta">{meta}</div>}
      </div>
    </div>
  )
}
