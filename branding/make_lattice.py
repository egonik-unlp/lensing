#!/usr/bin/env python3
"""Generate the "lensed lattice" brand surfaces: a regular framework grid that
takes the shape of your data the way mass curves spacetime. Companion to the
Velocity Map mark (make_mark.py), not a replacement.

Outputs (all generated, re-runnable; colors are the Control Room tokens with
the red->orange->blue velocity ramp kept brand-surface-only):

  assets/lattice.svg          hero concept card  — slate panel, warped lattice,
                              Einstein ring, the mass. Theme-robust (slate works
                              on GitHub light + dark), used in the README.
  assets/lattice-data.svg     wide figure — a straight row of data bent into a
                              lensed arc as it passes the mass.
  ui/public/brand/lensing-icons.svg
                              icon sprite (the Einstein-ring alignment sweep):
                              dataset / run / model / predict / compare / export.

  python3 branding/make_lattice.py            # write all three
  python3 branding/make_lattice.py --install  # (default) same; flag kept for symmetry
"""

import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# --- color: OKLCH -> sRGB (the <img> sandbox can't resolve oklch()) ----------
def _lab_to_hex(L, a, b):
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
    return _lab_to_hex(L, C * math.cos(h), C * math.sin(h))


# Control Room tokens
CHROME = oklch(0.27, 0.025, 250)        # the bezel
CHROME_BORDER = oklch(0.42, 0.025, 250)
ON_CHROME = oklch(0.96, 0.008, 250)
GRID = oklch(0.60, 0.02, 250)           # lattice line on chrome
GRID_DIM = oklch(0.45, 0.02, 250)
MASS = oklch(0.16, 0.02, 250)           # the data: darker than the chrome
MASS_RIM = oklch(0.52, 0.03, 250)
SRC = oklch(0.78, 0.16, 55)             # source / data point (accent-on-chrome)

# velocity ramp (dark/on-chrome anchors), Oklab-interpolated
RAMP = [(0.58, 0.20, 26), (0.76, 0.17, 52), (0.82, 0.10, 240)]


def ramp(t):
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        p, q, f = RAMP[0], RAMP[1], t * 2
    else:
        p, q, f = RAMP[1], RAMP[2], (t - 0.5) * 2
    L, C, H = (pi + (qi - pi) * f for pi, qi in zip(p, q))
    return oklch(L, C, H)


# --- geometry ----------------------------------------------------------------
def deflect(x, y, cx, cy, Re, strength=1.0, core=0.55, minr=0.42):
    """Pull a point toward the mass, deflection ~ 1/r (softened). Distances in
    units of Re so the look is scale-free."""
    dx, dy = x - cx, y - cy
    r = math.hypot(dx, dy)
    if r < 1e-6:
        return x, y
    ru = r / Re
    p = strength / math.sqrt(ru * ru + core * core)
    nru = max(ru - p, minr)
    return cx + dx / r * (nru * Re), cy + dy / r * (nru * Re)


def _poly(points, stroke, w, op=1.0):
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{w}" '
            f'stroke-opacity="{op}" stroke-linecap="round" stroke-linejoin="round"/>')


def warped_grid(cx, cy, Re, extent, step, strength, minr, stroke, w, op, samp=64):
    out = []
    n = int(extent / step)
    lo_x, hi_x = cx - n * step, cx + n * step
    lo_y, hi_y = cy - n * step, cy + n * step
    xs = [lo_x + i * step for i in range(2 * n + 1)]
    ys = [lo_y + i * step for i in range(2 * n + 1)]
    for gx in xs:
        line = [deflect(gx, lo_y + (hi_y - lo_y) * i / samp, cx, cy, Re, strength, minr=minr)
                for i in range(samp + 1)]
        out.append(_poly(line, stroke, w, op))
    for gy in ys:
        line = [deflect(lo_x + (hi_x - lo_x) * i / samp, gy, cx, cy, Re, strength, minr=minr)
                for i in range(samp + 1)]
        out.append(_poly(line, stroke, w, op))
    return "".join(out)


def colored_ring(cx, cy, r, a0, a1, w, op, nseg, phase=0.0):
    """Arc as ramp-colored segments; ramp runs by azimuth (Doppler-like)."""
    out = []
    for i in range(nseg):
        f0, f1 = i / nseg, (i + 1) / nseg
        a = a0 + (a1 - a0) * f0 - 0.5
        b = a0 + (a1 - a0) * f1 + 0.5
        am = math.radians((a + b) / 2)
        t = (1 - math.cos(am - math.radians(phase))) / 2
        large = 0
        x0, y0 = cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))
        x1, y1 = cx + r * math.cos(math.radians(b)), cy + r * math.sin(math.radians(b))
        out.append(f'<path d="M{x0:.2f} {y0:.2f} A{r:.2f} {r:.2f} 0 {large} 1 {x1:.2f} {y1:.2f}" '
                   f'fill="none" stroke="{ramp(t)}" stroke-opacity="{op}" '
                   f'stroke-width="{w}" stroke-linecap="round"/>')
    return "".join(out)


def mass_glyph(cx, cy, r):
    return (f'<circle cx="{cx}" cy="{cy}" r="{r + r*0.34:.2f}" fill="{MASS}" opacity="0.35"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{r:.2f}" fill="{MASS}"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{r:.2f}" fill="none" stroke="{MASS_RIM}" '
            f'stroke-width="{r*0.07:.2f}" stroke-opacity="0.6"/>')


# --- hero concept card -------------------------------------------------------
def hero_card():
    W, H = 720, 432
    pad = 1.5
    cx, cy = W / 2, H / 2
    Re = 96.0
    panel = (f'<rect x="0" y="0" width="{W}" height="{H}" rx="18" fill="{CHROME}"/>'
             f'<rect x="{pad}" y="{pad}" width="{W-2*pad}" height="{H-2*pad}" rx="{18-pad}" '
             f'fill="none" stroke="{CHROME_BORDER}" stroke-width="1.5"/>')
    # clip the lattice to the panel so the well fills the card edge-to-edge
    clip = (f'<clipPath id="panel"><rect x="6" y="6" width="{W-12}" height="{H-12}" rx="13"/></clipPath>')
    grid = warped_grid(cx, cy, Re, extent=max(W, H) * 0.62, step=34, strength=1.0,
                       minr=0.46, stroke=GRID, w=1.3, op=0.9)
    grid_dim = warped_grid(cx, cy, Re, extent=max(W, H) * 0.62, step=34, strength=1.0,
                           minr=0.46, stroke=GRID_DIM, w=2.6, op=0.18)  # soft underlay
    ring = colored_ring(cx, cy, Re, 0, 360, 4.2, 0.95, 64, phase=180)
    body = (f'{panel}<g clip-path="url(#panel)">{grid_dim}{grid}{ring}'
            f'{mass_glyph(cx, cy, Re*0.30)}</g>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
            f'height="{H}" role="img" aria-label="A regular lattice bending around a '
            f'mass at its center, lit by an Einstein ring: lensing takes the shape of '
            f'the data the way mass curves spacetime."><title>lensing</title>'
            f'<defs>{clip}</defs>{body}</svg>\n')


# --- wide "data, bent" figure ------------------------------------------------
def data_bent():
    """A straight, evenly-spaced row of data points enters from the left; as it
    passes the mass it is deflected into a smooth lensed arc that drapes over
    the mass, then straightens out again on the right. Deflection is purely
    vertical (toward the mass row) and falls off with horizontal distance, so
    points spread along the arc instead of collapsing into the center."""
    W, H = 720, 220
    cx, cy = W / 2, 150.0          # the mass, low-center
    base_y = 78.0                 # the undisturbed data row
    sigma = 132.0                 # how wide the bend reaches
    pull = (cy - base_y) + 30     # how far the row dips toward/over the mass
    panel = (f'<rect x="0" y="0" width="{W}" height="{H}" rx="16" fill="{CHROME}"/>'
             f'<rect x="1.5" y="1.5" width="{W-3}" height="{H-3}" rx="14.5" fill="none" '
             f'stroke="{CHROME_BORDER}" stroke-width="1.5"/>')
    n = 44
    pts, ticks = [], []
    for i in range(n):
        x = 26 + (W - 52) * i / (n - 1)
        u = (x - cx) / sigma
        dip = pull * math.exp(-u * u)                 # bell-shaped deflection
        # bend points apart horizontally near the center so they don't pile up
        spread = 1.0 + 0.55 * math.exp(-u * u)
        bx = cx + (x - cx) * spread
        by = base_y + dip
        pts.append((bx, by, u))
    for bx, by, u in pts:
        # tick oriented along the local tangent normal (so it sits on the curve)
        slope = -2 * (bx - cx) / (sigma * sigma) * pull * math.exp(-((bx - cx) / sigma) ** 2)
        ang = math.atan2(slope, 1) + math.pi / 2
        h = 11.0
        x1, y1 = bx - math.cos(ang) * h / 2, by - math.sin(ang) * h / 2
        x2, y2 = bx + math.cos(ang) * h / 2, by + math.sin(ang) * h / 2
        t = (1 - math.cos(math.atan2(by - cy, bx - cx) - math.radians(90))) / 2
        op = 0.55 + 0.45 * math.exp(-u * u)
        ticks.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="{ramp(t)}" stroke-opacity="{op:.2f}" stroke-width="3.2" '
                     f'stroke-linecap="round"/>')
    body = panel + "".join(ticks) + mass_glyph(cx, cy, 26)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
            f'height="{H}" role="img" aria-label="A straight row of data points bent into '
            f'a lensed arc as it passes a mass."><title>data, bent</title>{body}</svg>\n')


# --- icon family: the Einstein-ring alignment sweep --------------------------
# Monochrome (currentColor) so the app can tint them; one source accent.
def _icon(body):
    return body


def icons_sprite():
    """24x24 symbols. The mass is currentColor; the ring/arcs use currentColor
    with opacity; the source is the accent. Geometry is the alignment sweep:
    full ring -> arcs -> single arc, one signature per pipeline stage."""
    c = 12.0
    Re = 7.2

    def arc(r, a0, a1, w, op, n=24):
        return colored_ring_mono(c, c, r, a0, a1, w, op, n)

    def locus(op=0.5):
        return (f'<circle cx="{c}" cy="{c}" r="{Re}" fill="none" stroke="currentColor" '
                f'stroke-width="1" stroke-dasharray="1.4 2" stroke-opacity="{op}"/>')

    def mass(r):
        return f'<circle cx="{c}" cy="{c}" r="{r}" fill="currentColor"/>'

    def src(x, y, rr=1.7):
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rr}" fill="var(--brand-src,currentColor)"/>'

    syms = {}
    # dataset — the full ring forms (perfect alignment: structure crystallized)
    syms["dataset"] = locus(0) + ring_mono(c, c, Re, 0, 360, 2.0, 0.95) + mass(2.6)
    # run — the ring lighting up (an arc growing): a live alignment
    syms["run"] = locus() + ring_mono(c, c, Re, -40, 180, 2.2, 0.95) + \
        src(c + Re * math.cos(math.radians(70)), c + Re * math.sin(math.radians(70))) + mass(2.6)
    # model — locked: full ring + solid mass (a frozen contract)
    syms["model"] = ring_mono(c, c, Re, 0, 360, 1.6, 0.55) + mass(4.2)
    # predict — a single source resolved into one bright arc + its image
    syms["predict"] = locus() + ring_mono(c, c, Re * 1.04, 50, 170, 2.4, 0.95) + \
        ring_mono(c, c, Re * 0.96, 240, 290, 1.6, 0.5) + \
        src(c + Re * 1.5 * math.cos(math.radians(110)), c + Re * 1.5 * math.sin(math.radians(110))) + mass(2.4)
    # compare — two arcs, two images side by side
    syms["compare"] = locus() + ring_mono(c, c, Re, 120, 240, 2.2, 0.9) + \
        ring_mono(c, c, Re, -60, 60, 2.2, 0.9) + mass(2.4)
    # export — an arc, with a ray leaving the lens up-and-out (light escaping)
    syms["export"] = ring_mono(c, c, Re, 150, 320, 2.2, 0.9) + mass(2.4) + \
        ('<path d="M13.4 10.6 L19 5 M19 5 L14.7 5 M19 5 L19 9.3" '
         'stroke="currentColor" stroke-width="2" fill="none" '
         'stroke-linecap="round" stroke-linejoin="round"/>')

    symbols = "".join(
        f'<symbol id="lz-{k}" viewBox="0 0 24 24">{v}</symbol>' for k, v in syms.items()
    )
    return (f'<svg xmlns="http://www.w3.org/2000/svg" style="display:none">'
            f'<!-- lensing icon family: the Einstein-ring alignment sweep. '
            f'Use as <svg class="lz-icon"><use href="/brand/lensing-icons.svg#lz-run"/></svg>. '
            f'Stroke inherits currentColor; --brand-src tints the source dot. -->'
            f'{symbols}</svg>\n')


def ring_mono(cx, cy, r, a0, a1, w, op, n=28):
    out = []
    for i in range(n):
        f0, f1 = i / n, (i + 1) / n
        a = a0 + (a1 - a0) * f0 - 0.5
        b = a0 + (a1 - a0) * f1 + 0.5
        x0, y0 = cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))
        x1, y1 = cx + r * math.cos(math.radians(b)), cy + r * math.sin(math.radians(b))
        out.append(f'<path d="M{x0:.2f} {y0:.2f} A{r:.2f} {r:.2f} 0 0 1 {x1:.2f} {y1:.2f}" '
                   f'fill="none" stroke="currentColor" stroke-opacity="{op}" '
                   f'stroke-width="{w}" stroke-linecap="round"/>')
    return "".join(out)


def colored_ring_mono(*a, **k):
    return ring_mono(*a, **k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true", default=True)
    ap.parse_args()

    (ROOT / "assets/lattice.svg").write_text(hero_card())
    (ROOT / "assets/lattice-data.svg").write_text(data_bent())
    brand = ROOT / "ui/public/brand"
    brand.mkdir(parents=True, exist_ok=True)
    (brand / "lensing-icons.svg").write_text(icons_sprite())

    for p in ("assets/lattice.svg", "assets/lattice-data.svg", "ui/public/brand/lensing-icons.svg"):
        print("wrote", p)


if __name__ == "__main__":
    main()
