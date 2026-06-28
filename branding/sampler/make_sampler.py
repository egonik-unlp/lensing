#!/usr/bin/env python3
"""Assemble the lensing brand sampler into one self-contained HTML document,
ready for Chrome's print-to-PDF. Inlines the shipped SVGs, the project's own
Sora / Inter / JetBrains Mono webfonts, and renders the empty-state mark and a
motion filmstrip from the same deflection math as make_lattice.py.

  python3 branding/sampler/make_sampler.py          # writes sampler.html
  # then: google-chrome --headless --print-to-pdf=sampler.pdf sampler.html
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "branding"))
from make_lattice import (  # noqa: E402
    oklch, deflect, warped_grid, colored_ring, mass_glyph,
    CHROME, CHROME_BORDER, GRID, GRID_DIM, ON_CHROME,
)

HERE = Path(__file__).resolve().parent
FONTS = ROOT / "ui/dist/assets"
SORA = (FONTS / "sora-latin-wght-normal-DdqRvwsR.woff2").as_uri()
INTER = (FONTS / "inter-latin-wght-normal-Dx4kXJAl.woff2").as_uri()
MONO = (FONTS / "jetbrains-mono-latin-400-normal-V6pRDFza.woff2").as_uri()
MONO6 = (FONTS / "jetbrains-mono-latin-600-normal-C8RAYTDA.woff2").as_uri()

# paper tokens (content pages)
BG = oklch(0.992, 0, 0)
SURFACE = oklch(0.972, 0.002, 250)
INK = oklch(0.24, 0.01, 250)
INK2 = oklch(0.45, 0.012, 250)
INK3 = oklch(0.56, 0.01, 250)
HAIR = oklch(0.9, 0.003, 250)
HAIRS = oklch(0.82, 0.005, 250)
SERIESA = oklch(0.35, 0.012, 250)
ACCENT = oklch(0.5, 0.155, 42)


def read_svg(rel):
    return (ROOT / rel).read_text().split("\n", 0)[0] if False else (ROOT / rel).read_text()


def empty_state_mark():
    """The in-app empty-state mark: monochrome, paper. Mirrors LatticeMark.tsx."""
    C, RE = 60.0, 30.0
    step, n, samp = 12, 5, 40
    lo, hi = C - n * step, C + n * step
    coords = [C + (i - n) * step for i in range(2 * n + 1)]
    lines = []
    for gx in coords:
        pts = " ".join(f"{deflect(gx, lo + (hi - lo) * i / samp, C, C, RE)[0]:.1f},"
                       f"{deflect(gx, lo + (hi - lo) * i / samp, C, C, RE)[1]:.1f}" for i in range(samp + 1))
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{HAIRS}" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"/>')
    for gy in coords:
        pts = " ".join(f"{deflect(lo + (hi - lo) * i / samp, gy, C, C, RE)[0]:.1f},"
                       f"{deflect(lo + (hi - lo) * i / samp, gy, C, C, RE)[1]:.1f}" for i in range(samp + 1))
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{HAIRS}" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"/>')
    return (f'<svg viewBox="0 0 120 120" style="overflow:visible">{"".join(lines)}'
            f'<circle cx="60" cy="60" r="30" fill="none" stroke="{SERIESA}" stroke-width="2.4"/>'
            f'<circle cx="60" cy="60" r="9" fill="{INK}"/></svg>')


def mini_card(strength, reveal, mscale):
    """One frame of the motion study (slate)."""
    W, Hh = 240, 150
    cx, cy, Re = W / 2, Hh / 2, 46.0
    panel = (f'<rect width="{W}" height="{Hh}" rx="12" fill="{CHROME}"/>'
             f'<rect x="1" y="1" width="{W-2}" height="{Hh-2}" rx="11" fill="none" stroke="{CHROME_BORDER}" stroke-width="1"/>')
    clip = f'<clipPath id="m{int(strength*100)}{int(reveal*100)}"><rect x="4" y="4" width="{W-8}" height="{Hh-8}" rx="9"/></clipPath>'
    grid = warped_grid(cx, cy, Re, extent=max(W, Hh) * 0.62, step=24, strength=max(strength, 0.02),
                       minr=0.46, stroke=GRID, w=1.1, op=0.9)
    ring = colored_ring(cx, cy, Re, 0, 360 * reveal, 3.2, 0.95, max(2, int(64 * reveal)), phase=180) if reveal > 0.01 else ""
    mr = Re * 0.30 * mscale
    mass = mass_glyph(cx, cy, max(0.1, mr)) if mscale > 0.05 else ""
    cid = f"m{int(strength*100)}{int(reveal*100)}"
    return (f'<svg viewBox="0 0 {W} {Hh}" style="width:100%;height:auto;display:block"><defs>{clip}</defs>'
            f'{panel}<g clip-path="url(#{cid})">{grid}{ring}{mass}</g></svg>')


def build():
    logo = read_svg("assets/logo.svg")
    lattice = read_svg("assets/lattice.svg")
    data_bent = read_svg("assets/lattice-data.svg")
    sprite = read_svg("ui/public/brand/lensing-icons.svg")
    sprite_inner = sprite.split(">", 1)[1].rsplit("</svg>", 1)[0]
    mark_dark = read_svg("ui/public/brand/mark-dark.svg")

    icons = [
        ("dataset", "the ring forms — a frozen featurization contract"),
        ("run", "a live arc lights up — training in progress"),
        ("model", "locked: full ring, solid mass — a promoted model"),
        ("predict", "one source resolved into a bright arc and its image"),
        ("compare", "two arcs, two images held side by side"),
        ("export", "a ray leaves the lens — the model, portable"),
    ]
    icon_cells = "".join(
        f'<figure class="icon"><svg viewBox="0 0 24 24"><use href="#lz-{k}"/></svg>'
        f'<figcaption><b>{k}</b><span>{d}</span></figcaption></figure>'
        for k, d in icons
    )

    css = f"""
    @font-face {{ font-family: Sora; src: url('{SORA}') format('woff2'); font-weight: 100 800; font-display: block; }}
    @font-face {{ font-family: InterV; src: url('{INTER}') format('woff2'); font-weight: 100 900; font-display: block; }}
    @font-face {{ font-family: JBMono; src: url('{MONO}') format('woff2'); font-weight: 400; font-display: block; }}
    @font-face {{ font-family: JBMono; src: url('{MONO6}') format('woff2'); font-weight: 600; font-display: block; }}
    @page {{ size: letter; margin: 0; }}
    * {{ box-sizing: border-box; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    html, body {{ margin: 0; padding: 0; }}
    body {{ font-family: InterV, system-ui, sans-serif; color: {INK}; background: {BG}; }}
    .page {{ position: relative; width: 8.5in; height: 11in; padding: 0.82in 0.8in; overflow: hidden;
      page-break-after: always; background: {BG}; }}
    .page:last-child {{ page-break-after: auto; }}
    h1, h2, .word {{ font-family: Sora, sans-serif; letter-spacing: -0.02em; }}
    .kicker {{ font-family: JBMono, monospace; font-size: 9.5px; letter-spacing: 0.16em;
      text-transform: uppercase; color: {INK3}; }}
    p {{ line-height: 1.55; max-width: 64ch; color: {INK2}; }}
    .lead {{ color: {INK}; font-size: 15px; }}

    /* cover */
    .cover {{ background: {CHROME}; color: {ON_CHROME}; padding: 0.95in 0.85in; display: flex; flex-direction: column; }}
    .cover .mark {{ width: 96px; height: 96px; }}
    .cover .word {{ font-size: 74px; font-weight: 600; line-height: 1; margin: 0.34in 0 0; color: {ON_CHROME}; }}
    .cover .sub {{ font-size: 20px; color: {oklch(0.78,0.015,250)}; margin-top: 8px; font-family: InterV; }}
    .cover .rule {{ height: 1px; background: {CHROME_BORDER}; margin: 0.42in 0; }}
    .cover .blurb {{ color: {oklch(0.78,0.015,250)}; max-width: 46ch; font-size: 13.5px; }}
    .cover .foot {{ margin-top: auto; display: flex; justify-content: space-between; align-items: baseline;
      font-family: JBMono, monospace; font-size: 11px; color: {oklch(0.7,0.02,250)}; }}
    .cover .accent {{ color: {oklch(0.78,0.16,55)}; }}

    .sechead {{ margin: 0 0 0.34in; }}
    .sechead h2 {{ font-size: 26px; font-weight: 600; margin: 6px 0 8px; color: {INK}; }}

    .asset {{ margin: 0.3in 0; }}
    .asset svg {{ width: 100%; height: auto; display: block; }}
    .frame {{ border: 1px solid {HAIR}; border-radius: 12px; padding: 22px; background: {SURFACE}; }}
    .cap {{ font-size: 12px; color: {INK2}; margin-top: 12px; max-width: 60ch; }}
    .cap b {{ color: {INK}; font-weight: 600; }}
    .mono {{ font-family: JBMono, monospace; font-size: 11.5px; color: {INK}; }}
    .strike {{ color: {INK3}; text-decoration: line-through; text-decoration-color: {ACCENT}; }}

    .iconwrap {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.26in; margin-top: 0.3in; }}
    .icon {{ margin: 0; display: flex; flex-direction: column; align-items: center; gap: 12px;
      padding: 22px 14px 18px; border: 1px solid {HAIR}; border-radius: 12px; background: {SURFACE}; text-align: center; }}
    .icon svg {{ width: 46px; height: 46px; color: {INK}; }}
    .icon b {{ font-family: JBMono, monospace; font-weight: 600; font-size: 12px; color: {INK}; }}
    .icon figcaption {{ display: flex; flex-direction: column; gap: 5px; }}
    .icon span {{ font-size: 10.5px; line-height: 1.4; color: {INK2}; }}

    .two {{ display: grid; grid-template-columns: 1.1fr 1fr; gap: 0.42in; align-items: start; margin-top: 0.34in; }}
    .demo {{ border: 1px solid {HAIR}; border-radius: 12px; background: {SURFACE}; padding: 26px; text-align: center; }}
    .demo .emptymark {{ width: 124px; margin: 0 auto; }}
    .demo h3 {{ font-family: Sora; font-size: 17px; margin: 14px 0 6px; color: {INK}; font-weight: 600; }}
    .demo p {{ font-size: 11.5px; margin: 0 auto; color: {INK2}; }}
    .strip {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
    .strip figcaption {{ font-family: JBMono, monospace; font-size: 9px; letter-spacing: 0.06em; color: {INK3}; text-align: center; margin-top: 6px; text-transform: uppercase; }}
    """

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>lensing — brand sampler</title>
<style>{css}</style></head><body>
<svg style="display:none">{sprite_inner}</svg>

<!-- COVER -->
<section class="page cover">
  <div class="mark">{mark_dark}</div>
  <div class="word">lensing</div>
  <div class="sub">brand sampler</div>
  <div class="rule"></div>
  <p class="blurb">A prediction lab that bends to fit your data. This sheet collects the
  marks, the icon family, and the in-product surfaces built on one idea: a generic
  framework taking the shape of your dataset the way mass curves spacetime.</p>
  <div class="foot"><span>lensing <span class="accent">·</span> framework</span><span>2026 · 06 · 11</span></div>
</section>

<!-- THE IDEA -->
<section class="page">
  <div class="sechead"><div class="kicker">The idea</div>
    <h2>The lab takes the shape of your data</h2></div>
  <p class="lead">Point lensing at a corpus, declare your target, and the whole lab bends to
  fit. The brand says this without explaining it: a regular framework grid curves around the
  mass of your dataset, and an Einstein ring lights where the deflection peaks. It is the
  textbook picture of mass curving spacetime and a literal picture of a framework conforming to
  your data, in one image.</p>
  <div class="asset"><div class="frame">{lattice}</div>
    <div class="cap"><b>The concept card</b> — <span class="mono">assets/lattice.svg</span>. Slate so it
    reads on light and dark alike; the red→orange→blue velocity ramp is reserved for the mark and
    brand surfaces. Generated, re-runnable: <span class="mono">make_lattice.py</span>.</div></div>
</section>

<!-- BANNER + DATA BENT -->
<section class="page">
  <div class="sechead"><div class="kicker">Wordmark &amp; tagline</div>
    <h2>The banner</h2></div>
  <div class="asset">{logo}
    <div class="cap"><b>The tagline lost its caption.</b> Was
    <span class="strike">“bends around your data, the way mass bends light”</span>; now
    <span class="mono">“a prediction lab that bends to fit your data.”</span> One restrained verb;
    the physics lives in the mark, not the sentence.</div></div>
  <div class="asset" style="margin-top:0.5in"><div class="frame">{data_bent}</div>
    <div class="cap"><b>Data, bent</b> — <span class="mono">assets/lattice-data.svg</span>. A straight,
    evenly-spaced row of data points deflected into a lensed arc as it passes the mass.</div></div>
</section>

<!-- ICON FAMILY -->
<section class="page">
  <div class="sechead"><div class="kicker">Icon family</div>
    <h2>The Einstein-ring alignment sweep</h2></div>
  <p>Full ring → arcs → single arc is a real physical progression, so each stage of the pipeline
  carries a different alignment state. Stroke inherits <span class="mono">currentColor</span>;
  <span class="mono">--brand-src</span> tints the source dot. They feed the app’s empty-state mark
  and external surfaces — they do <b>not</b> replace the text-monogram glyphs the app uses for
  runs, datasets and models.</p>
  <div class="iconwrap">{icon_cells}</div>
</section>

<!-- IN APP + MOTION -->
<section class="page">
  <div class="sechead"><div class="kicker">In product &amp; in motion</div>
    <h2>Empty states, and how it forms</h2></div>
  <div class="two">
    <div class="demo">
      <div class="emptymark">{empty_state_mark()}</div>
      <h3>No data yet</h3>
      <p>Every empty state opens with the lattice — monochrome here, because the one orange on the
      screen belongs to its primary button. The ring draws in on mount; reduced motion holds it.</p>
    </div>
    <div>
      <div class="strip">
        <figure style="margin:0">{mini_card(0.02, 0.0, 0.0)}<figcaption>flat</figcaption></figure>
        <figure style="margin:0">{mini_card(0.5, 0.4, 0.6)}<figcaption>data arrives</figcaption></figure>
        <figure style="margin:0">{mini_card(1.0, 1.0, 1.0)}<figcaption>formed</figcaption></figure>
      </div>
      <div class="cap" style="margin-top:14px"><b>Motion study</b> — the grid starts flat, the dataset
      arrives, the grid bends to fit and the ring lights, then it loops.
      <span class="mono">branding/explore/animated.html</span>.</div>
    </div>
  </div>
</section>

</body></html>"""
    out = HERE / "sampler.html"
    out.write_text(html)
    print("wrote", out)


if __name__ == "__main__":
    build()
