#!/usr/bin/env python3
"""Exploration panel: alternative visual directions for the lensing brand,
all hinging on gravitational lensing AND "the framework takes the shape of
your data". Generates a set of candidate SVGs + a gallery index.html.

This is a SHAPE artifact, not a shipped one: it exists to choose a direction.
Geometry is physically motivated (point-mass deflection ~ 1/r, Einstein ring,
the two-image alignment sweep), colors are the Control Room tokens with the
red->orange->blue velocity ramp kept brand-surface-only.

  python3 branding/explore/make_explore.py     # writes SVGs + index.html here
"""

import math
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- color: OKLCH -> sRGB (the <img> sandbox can't resolve oklch()) ----------
def lab_to_srgb_hex(L, a, b):
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    r = 4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
    g = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
    bb = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_

    def f(x):
        x = max(0.0, min(1.0, x))
        return 12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055

    return "#%02x%02x%02x" % tuple(round(f(v) * 255) for v in (r, g, bb))


def oklch(L, C, H):
    h = math.radians(H)
    return lab_to_srgb_hex(L, C * math.cos(h), C * math.sin(h))


# Control Room tokens (paper surface)
INK = oklch(0.24, 0.01, 250)         # the mass
INK_2 = oklch(0.45, 0.012, 250)
GRID = oklch(0.74, 0.012, 250)       # lattice hairline (slate, faint)
GRID_FAINT = oklch(0.86, 0.008, 250)
LOCUS = oklch(0.70, 0.02, 250)       # Einstein-radius locus (dashed)
SRC = oklch(0.68, 0.18, 47)          # the true source / data point (accent)

# velocity ramp: redshift -> rest orange -> blueshift, in OKLCH, Oklab-interp
RAMP = [(0.42, 0.20, 26), (0.64, 0.18, 47), (0.55, 0.14, 245)]


def ramp(t):
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        p, q, f = RAMP[0], RAMP[1], t * 2
    else:
        p, q, f = RAMP[1], RAMP[2], (t - 0.5) * 2
    L, C, H = (pi + (qi - pi) * f for pi, qi in zip(p, q))
    return oklch(L, C, H)


# --- geometry ----------------------------------------------------------------
CX = CY = 60.0          # tile center
RE = 30.0               # Einstein radius


def deflect(x, y, strength=1.0, core=7.0, minr=3.0):
    """Pull a point radially inward toward the mass, deflection ~ 1/r (softened)."""
    dx, dy = x - CX, y - CY
    r = math.hypot(dx, dy)
    if r < 1e-6:
        return x, y
    p = strength * RE * RE / math.sqrt(r * r + core * core)
    nr = max(r - p, minr)
    return CX + dx / r * nr, CY + dy / r * nr


def poly(points, stroke, width, op=1.0):
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (f'<polyline points="{pts}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width}" stroke-opacity="{op}" '
            f'stroke-linecap="round" stroke-linejoin="round"/>')


def warped_grid(strength, minr, stroke=GRID, width=1.0, op=1.0, step=15, samp=48):
    """A square lattice, each line displaced by the deflection field."""
    out = []
    lo, hi = 0, 120
    coords = list(range(lo, hi + 1, step))
    for gx in coords:
        line = [deflect(gx, lo + (hi - lo) * i / samp, strength, minr=minr)
                for i in range(samp + 1)]
        out.append(poly(line, stroke, width, op))
    for gy in coords:
        line = [deflect(lo + (hi - lo) * i / samp, gy, strength, minr=minr)
                for i in range(samp + 1)]
        out.append(poly(line, stroke, width, op))
    return "".join(out)


def arc_pts(cx, cy, r, a0, a1, n=40):
    return [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
            for a in (a0 + (a1 - a0) * i / n for i in range(n + 1))]


def colored_arc(cx, cy, r, a0, a1, width, op=1.0, nseg=28, t_of=None):
    """An arc drawn as ramp-colored segments. t_of(frac)->ramp parameter."""
    out = []
    for i in range(nseg):
        f0, f1 = i / nseg, (i + 1) / nseg
        a = a0 + (a1 - a0) * f0 - 0.4
        b = a0 + (a1 - a0) * f1 + 0.4
        t = (f0 + f1) / 2 if t_of is None else t_of((f0 + f1) / 2)
        x0, y0 = cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))
        x1, y1 = cx + r * math.cos(math.radians(b)), cy + r * math.sin(math.radians(b))
        out.append(f'<path d="M{x0:.2f} {y0:.2f} A{r:.2f} {r:.2f} 0 0 1 {x1:.2f} {y1:.2f}" '
                   f'fill="none" stroke="{ramp(t)}" stroke-opacity="{op}" '
                   f'stroke-width="{width}" stroke-linecap="round"/>')
    return "".join(out)


def mass(r=11.0, glow=True):
    g = (f'<circle cx="{CX}" cy="{CY}" r="{r + 3}" fill="{INK}" opacity="0.10"/>'
         if glow else "")
    return g + f'<circle cx="{CX}" cy="{CY}" r="{r}" fill="{INK}"/>'


def locus(r=RE):
    return (f'<circle cx="{CX}" cy="{CY}" r="{r}" fill="none" stroke="{LOCUS}" '
            f'stroke-width="0.8" stroke-dasharray="2 3" stroke-opacity="0.7"/>')


def svg(body, bg="transparent", vb=120):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {vb} {vb}" '
            f'width="{vb}" height="{vb}">'
            f'{"" if bg=="transparent" else f"<rect width=\"{vb}\" height=\"{vb}\" fill=\"{bg}\"/>"}'
            f'{body}</svg>')


# --- candidate builders ------------------------------------------------------
def lattice_mass():
    return svg(warped_grid(1.0, minr=12.5) + locus() + mass())


def lattice_implied():
    # no sphere: the void the lattice bends around IS the (unseen) dataset
    return svg(warped_grid(1.1, minr=4.0))


def lattice_offcenter():
    global CX, CY
    CX, CY = 74.0, 70.0
    out = warped_grid(0.95, minr=11.0) + locus() + mass(10)
    CX = CY = 60.0
    return svg(out)


def lattice_ring():
    body = warped_grid(0.85, minr=12.5, op=0.85)
    body += colored_arc(CX, CY, RE, 0, 360, 2.6, op=0.95, nseg=48)
    body += mass()
    return svg(body)


def ring_full():
    body = locus() + colored_arc(CX, CY, RE, 0, 360, 3.0, op=0.95, nseg=56) + mass()
    return svg(body)


def ring_alignment(offset_frac, src_phi=90):
    """A point source at offset breaks the ring into two arcs (the classic
    alignment sweep). offset_frac in [0,1] of RE."""
    fr = offset_frac
    body = [locus()]
    # primary arc on the source side, secondary opposite (shorter, dimmer)
    pspan = 150 * (1 - fr) + 22
    sspan = 120 * (1 - fr * fr)
    body.append(colored_arc(CX, CY, RE * 1.05, src_phi - pspan, src_phi + pspan,
                            3.0, op=0.95, nseg=44))
    if 1 - fr > 0.05:
        body.append(colored_arc(CX, CY, RE * 0.95, src_phi + 180 - sspan / 2,
                                src_phi + 180 + sspan / 2, 2.4,
                                op=round(0.85 * (1 - fr), 2), nseg=30))
    # the true source position (where the data actually is)
    sx = CX + offset_frac * RE * 1.6 * math.cos(math.radians(src_phi))
    sy = CY + offset_frac * RE * 1.6 * math.sin(math.radians(src_phi))
    body.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="2.6" fill="{SRC}"/>')
    body.append(mass())
    return svg("".join(body))


def einstein_cross():
    body = [locus()]
    for ang in (45, 135, 225, 315):
        x = CX + RE * math.cos(math.radians(ang))
        y = CY + RE * math.sin(math.radians(ang))
        t = (1 - math.cos(math.radians(ang))) / 2
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{ramp(t)}"/>')
    body.append(f'<circle cx="{CX}" cy="{CY}" r="2.4" fill="{SRC}"/>')
    body.append(mass(9))
    return svg("".join(body))


def horseshoe():
    body = locus() + colored_arc(CX, CY, RE * 1.04, 35, 35 + 305, 3.4, op=0.95, nseg=52)
    body += mass()
    return svg(body)


def data_arc():
    """A straight row of data ticks passing the lens, bent into an arc."""
    body = [locus()]
    n = 26
    for i in range(n):
        x = 6 + (108) * i / (n - 1)
        y0 = 22
        # tick bends down/around as it nears the lens column
        bx, by = deflect(x, y0, strength=1.25, minr=10.0)
        h = 5.0
        # short vertical tick oriented along local radial
        dx, dy = bx - CX, by - CY
        r = math.hypot(dx, dy) or 1
        ux, uy = dx / r, dy / r
        x1, y1 = bx - ux * h / 2, by - uy * h / 2
        x2, y2 = bx + ux * h / 2, by + uy * h / 2
        t = (1 - math.cos(math.atan2(by - CY, bx - CX))) / 2
        body.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                    f'stroke="{ramp(t)}" stroke-width="2.2" stroke-linecap="round"/>')
    body.append(mass())
    return svg("".join(body))


def data_doubled():
    """A small data glyph (3 bars) lensed into two images around the mass."""
    def bars(ox, oy, scale, op):
        out = []
        for i, hgt in enumerate((6, 10, 7)):
            x = ox + i * 3.2 * scale
            out.append(f'<rect x="{x:.1f}" y="{oy - hgt*scale:.1f}" width="{2.0*scale:.1f}" '
                       f'height="{hgt*scale:.1f}" fill="{SRC}" opacity="{op}" rx="0.6"/>')
        return "".join(out)
    body = [locus()]
    body.append(bars(78, 44, 1.0, 0.95))      # primary image (bright, larger)
    body.append(bars(30, 80, 0.62, 0.6))      # secondary image (dim, smaller)
    body.append(mass(9))
    return svg("".join(body))


# --- gallery -----------------------------------------------------------------
SECTIONS = [
    ("Lattice well — the framework taking the data's shape", [
        ("lattice-mass", "Mass present", "Square lattice drawn into a visible mass. The framework's grid; your dataset deforms it.", lattice_mass),
        ("lattice-implied", "Mass implied", "No sphere — the void the lattice bends around IS the data. Quieter, more abstract.", lattice_implied),
        ("lattice-offcenter", "Off-center mass", "The data sits where it sits; the grid bends asymmetrically around it.", lattice_offcenter),
        ("lattice-ring", "Lattice + ring", "Warped lattice with the Einstein ring lit where deflection peaks. Hero candidate.", lattice_ring),
    ]),
    ("Einstein ring — alignment sweep (different angles)", [
        ("ring-full", "Perfect alignment", "Source dead-on behind the mass: a complete ring.", ring_full),
        ("ring-a25", "Slight offset", "Ring begins to break: a long primary arc, faint counter-arc.", lambda: ring_alignment(0.25)),
        ("ring-a55", "Moderate offset", "Two distinct arcs at an angle — the canonical lensing look.", lambda: ring_alignment(0.55)),
        ("ring-a85", "Strong offset", "Source far off-axis: one bright arc, the counter-image nearly gone.", lambda: ring_alignment(0.85)),
    ]),
    ("Other lensing signatures", [
        ("cross", "Einstein cross", "Quadruple image around the lens. One source, structured into four.", einstein_cross),
        ("horseshoe", "Horseshoe arc", "Near-alignment: the arc wraps most of the way around.", horseshoe),
        ("data-arc", "Data row, bent", "A straight row of data ticks bent into a lensed arc as it passes the mass.", data_arc),
        ("data-doubled", "Doubled data", "A data glyph lensed into two images — bright primary, dim secondary.", data_doubled),
    ]),
]


def build():
    tiles = []
    for title, items in SECTIONS:
        cards = []
        for key, name, desc, fn in items:
            markup = fn()
            (HERE / f"{key}.svg").write_text(markup)
            cards.append(f'''<figure class="card">
  <div class="art">{markup}</div>
  <figcaption><b>{name}</b><span>{desc}</span></figcaption>
</figure>''')
        tiles.append(f'<section><h2>{title}</h2><div class="grid">{"".join(cards)}</div></section>')

    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>lensing — brand exploration</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: {oklch(0.972,0.002,250)}; color: {INK};
    font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, Segoe UI, Inter, sans-serif;
    padding: 40px clamp(20px, 5vw, 64px) 72px; }}
  header {{ max-width: 760px; margin: 0 0 36px; }}
  header h1 {{ font-size: 30px; font-weight: 600; letter-spacing: -0.02em; margin: 0 0 8px; }}
  header p {{ margin: 0; color: {INK_2}; max-width: 62ch; }}
  h2 {{ font-size: 14px; font-weight: 600; letter-spacing: 0.01em; color: {INK_2};
    margin: 40px 0 16px; padding-bottom: 8px; border-bottom: 1px solid {GRID_FAINT}; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 18px; }}
  .card {{ margin: 0; background: {oklch(0.992,0,0)}; border: 1px solid {GRID_FAINT};
    border-radius: 10px; overflow: hidden; }}
  .art {{ display: grid; place-items: center; padding: 18px; background:
    linear-gradient(180deg, {oklch(0.992,0,0)}, {oklch(0.975,0.002,250)}); }}
  .art svg {{ width: 150px; height: 150px; display: block; }}
  figcaption {{ padding: 12px 14px 16px; border-top: 1px solid {GRID_FAINT};
    display: flex; flex-direction: column; gap: 4px; }}
  figcaption b {{ font-weight: 600; font-size: 13.5px; }}
  figcaption span {{ color: {INK_2}; font-size: 12.5px; line-height: 1.45; }}
</style></head><body>
<header>
  <h1>lensing — brand directions</h1>
  <p>Candidates for the mark/icon family: gravitational lensing as the picture
  of a framework taking the shape of your data. Pick a direction (or mix); the
  winner becomes the icon set, the README concept image, and the new banner.</p>
</header>
{"".join(tiles)}
</body></html>'''
    (HERE / "index.html").write_text(html)
    print("wrote", HERE / "index.html", "and", len(list(HERE.glob("*.svg"))), "svgs")


if __name__ == "__main__":
    build()
