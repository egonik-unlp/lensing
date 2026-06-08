import { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { useDomain } from '../lib/DomainContext'
import { cap } from '../lib/format'
import './shell.css'

/* The instance's seeded velocity-map mark (branding/make_mark.py), with the
 * swarm orbiting the dark sphere. Stacked static layers rotated by page CSS
 * (shell.css): every in-image mechanism fails somewhere at wordmark size —
 * CSS inside SVG-as-image doesn't run in Firefox, Chromium freezes composited
 * transforms on inlined SVG children, and stops driving SMIL in images at
 * small raster sizes. Compositor transforms on plain HTML elements are the
 * one size-independent, everywhere-supported primitive. */
function WordmarkGlyph() {
  return (
    <span className="wordmark-glyph" aria-hidden>
      <img className="glyph-band glyph-band-0" src="/brand/mark-dark-band0.svg" alt="" />
      <img className="glyph-band glyph-band-1" src="/brand/mark-dark-band1.svg" alt="" />
      <img className="glyph-band glyph-band-2" src="/brand/mark-dark-band2.svg" alt="" />
      <img src="/brand/mark-dark-core.svg" alt="" />
    </span>
  )
}

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
            <WordmarkGlyph />
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
