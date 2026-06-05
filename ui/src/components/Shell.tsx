import { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { useDomain } from '../lib/DomainContext'
import { cap } from '../lib/format'
import './shell.css'

export default function Shell() {
  const domain = useDomain()
  // Tone-step the topbar's bottom hairline once content scrolls under it
  // (flat elevation: a border step, never a resting shadow).
  const [stuck, setStuck] = useState(false)
  useEffect(() => {
    const onScroll = () => setStuck(window.scrollY > 0)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <>
      <header className={stuck ? 'topbar is-stuck' : 'topbar'}>
        <div className="topbar-inner">
          <NavLink to="/" className="wordmark">
            {/* The favicon's trend line, reused as the mark. */}
            <svg className="wordmark-glyph" viewBox="0 0 16 16" width="16" height="16" aria-hidden>
              <path d="M2 12 L6 7 L9 9.5 L14 3" fill="none" stroke="currentColor" strokeWidth="2" />
            </svg>
            {(() => {
              const parts = domain.project.title.split(' ')
              const last = parts.length > 1 ? parts.pop()! : ''
              return (
                <>
                  {parts.join(' ')} {last && <span className="wordmark-dim">{last}</span>}
                </>
              )
            })()}
          </NavLink>
          {/* The pipeline spine, in pipeline order: build → train → promote. */}
          <nav aria-label="Primary">
            <NavLink to="/datasets" className="nav-link">
              Datasets
            </NavLink>
            <NavLink to="/" end className="nav-link">
              Runs
            </NavLink>
            <NavLink to="/compare" className="nav-link">
              Compare
            </NavLink>
            <NavLink to="/definitions" className="nav-link">
              Definitions
            </NavLink>
            <NavLink to="/models" className="nav-link">
              Models
            </NavLink>
            <NavLink to="/listings" className="nav-link">
              {cap(domain.project.entity_noun_plural)}
            </NavLink>
          </nav>
          {/* Quiet on purpose: the orange budget belongs to each page's own
              primary action (Start training, Compare runs, live state). */}
          <NavLink to="/new" className="btn-on-chrome topbar-cta">
            New run
          </NavLink>
        </div>
      </header>
      <main className="content">
        <Outlet />
      </main>
    </>
  )
}
