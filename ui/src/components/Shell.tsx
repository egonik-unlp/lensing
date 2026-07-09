import { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { useDomain } from '../lib/DomainContext'
import { cap } from '../lib/format'
import CommandPalette from './CommandPalette'
import './shell.css'

const IS_MAC = /Mac|iPhone|iPad/.test(navigator.platform)

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

  // ⌘K / Ctrl+K from anywhere opens the palette.
  const [paletteOpen, setPaletteOpen] = useState(false)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setPaletteOpen((o) => !o)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return (
    <>
      <header className={stuck ? 'topbar is-stuck' : 'topbar'}>
        <div className="topbar-inner">
          <NavLink to="/" className="wordmark">
            {/* The instance's seeded velocity-map mark (branding/make_mark.py). */}
            <img className="wordmark-glyph" src="/brand/mark-dark.svg" width={18} height={18} alt="" />
            lensing <span className="wordmark-dim">· {domain.project.title}</span>
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
            <NavLink to="/interpretability" className="nav-link">
              Interpretability
            </NavLink>
          </nav>
          <button
            className="topbar-search"
            onClick={() => setPaletteOpen(true)}
            aria-label="Open command palette"
          >
            <span className="topbar-search-word">Jump to…</span>
            <kbd>{IS_MAC ? '⌘K' : 'Ctrl K'}</kbd>
          </button>
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
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
    </>
  )
}
