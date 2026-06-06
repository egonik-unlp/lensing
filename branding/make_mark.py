#!/usr/bin/env python3
"""Generate the lensing brand mark (the "Velocity Map"): a dark sphere inside
a swarm of tangentially smeared trails, each colored by line-of-sight velocity
(blueshifted approaching, redshifted receding), with a Doppler-shifted rim.

The grammar is fixed; instances inject personality through two knobs read
from domain.toml [branding] (or CLI flags):

  mark_seed       arc arrangement (string, hashed; "lensing" and the absent
                  default reproduce the canonical mark exactly)
  mark_hue_shift  degrees of hue rotation applied to the red/orange/blue ramp

Usage:
  python3 branding/make_mark.py                 # write branding/lensing-mark{,-dark}.svg
  python3 branding/make_mark.py --install       # also install ui/public/favicon.svg + ui/public/brand/
  python3 branding/make_mark.py --seed demo --hue-shift 40 --out-dir /tmp/x

Colors are anchored to the Control Room tokens (--bad red, --accent orange,
--series-b blue) and interpolated in Oklab. Paper and dark (on-chrome) ramps
differ in lightness; both rotate together under --hue-shift.
"""

import argparse
import math
import random
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The rng seed that produced the approved canonical mark.
CANONICAL_SEED = 19
CANONICAL_NAMES = {"", "lensing"}

# Ramp anchors as OKLCH (L, C, H): redshifted -> rest orange -> blueshifted.
RAMP_ANCHORS = {
    "paper": [(0.42, 0.20, 26), (0.64, 0.18, 47), (0.55, 0.14, 245)],
    "dark": [(0.58, 0.20, 26), (0.76, 0.17, 52), (0.82, 0.10, 240)],
}
SHADOW = {"paper": "#1c2024", "dark": "#0b0e13"}
INK = {"paper": "#1c2024", "dark": "#eef2f7"}


def fmt(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def oklch_to_lab(L, C, H):
    h = math.radians(H)
    return (L, C * math.cos(h), C * math.sin(h))


def lab_to_srgb_hex(L, a, b):
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_**3, m_**3, s_**3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    def f(x):
        x = max(0.0, min(1.0, x))
        return 12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055

    return "#%02x%02x%02x" % tuple(round(f(v) * 255) for v in (r, g, bb))


def make_ramp(mode: str, hue_shift: float):
    anchors = [oklch_to_lab(L, C, H + hue_shift) for L, C, H in RAMP_ANCHORS[mode]]

    def ramp(t: float) -> str:
        t = max(0.0, min(1.0, t))
        if t < 0.5:
            p, q, f = anchors[0], anchors[1], t * 2
        else:
            p, q, f = anchors[1], anchors[2], (t - 0.5) * 2
        return lab_to_srgb_hex(*(pi + (qi - pi) * f for pi, qi in zip(p, q)))

    return ramp


def doppler_ring(cx, cy, R, basew, phi0_deg, beta, ramp, nseg=64, dim=1.0):
    """Rim segments colored/weighted by the relativistic Doppler factor."""
    phi0 = math.radians(phi0_deg)
    dmin, dmax = 1 / (1 + beta), 1 / (1 - beta)
    segs = []
    step = 2 * math.pi / nseg
    for i in range(nseg):
        a = i * step - 0.004  # tiny overlap kills antialiasing gaps
        b = (i + 1) * step + 0.004
        am = (a + b) / 2
        d = 1 / (1 - beta * math.cos(am - phi0))
        t = (d - dmin) / (dmax - dmin)
        op = round(min(0.95, (0.38 + 0.57 * t**1.15)) * dim, 2)
        sw = basew * (0.6 + 0.85 * t)
        x0, y0 = cx + R * math.cos(a), cy + R * math.sin(a)
        x1, y1 = cx + R * math.cos(b), cy + R * math.sin(b)
        segs.append(
            f'<path d="M{fmt(x0)} {fmt(y0)} A{fmt(R)} {fmt(R)} 0 0 1 {fmt(x1)} {fmt(y1)}" '
            f'stroke="{ramp(t)}" stroke-opacity="{op}" stroke-width="{sw:.2f}" fill="none"/>'
        )
    return "".join(segs)


def velocity_swarm(rng, ramp):
    """Tangential trail arcs, magnification-weighted, velocity-colored."""
    arcs = []
    for _ in range(34):
        r = max(12.0, min(22.0, rng.gauss(15.5, 3.6)))
        mag = math.exp(-(((r - 15.5) / 4.6) ** 2))
        span = math.radians((16 + 70 * mag) * rng.uniform(0.7, 1.3))
        a0 = rng.uniform(0, 2 * math.pi)
        am = a0 + span / 2
        # rotation about the vertical axis: left approaches (blue), right recedes (red)
        t = (1 - math.cos(am)) / 2
        op = 0.3 + 0.55 * mag
        sw = 0.8 + 1.1 * mag
        x0, y0 = 24 + r * math.cos(a0), 24 + r * math.sin(a0)
        x1, y1 = 24 + r * math.cos(a0 + span), 24 + r * math.sin(a0 + span)
        large = 1 if span > math.pi else 0
        arcs.append(
            f'<path d="M{fmt(x0)} {fmt(y0)} A{fmt(r)} {fmt(r)} 0 {large} 1 {fmt(x1)} {fmt(y1)}" '
            f'stroke="{ramp(t)}" stroke-opacity="{op:.2f}" stroke-width="{sw:.2f}" '
            f'fill="none" stroke-linecap="round"/>'
        )
    return "".join(arcs)


def mark_svg(mode: str, seed_int: int, hue_shift: float) -> str:
    ramp = make_ramp(mode, hue_shift)
    rng = random.Random(seed_int)
    body = (
        velocity_swarm(rng, ramp)
        + doppler_ring(24, 24, 10.0, 1.9, 180, 0.5, ramp, dim=0.9)
        + f'<circle cx="24" cy="24" r="8.6" fill="{SHADOW[mode]}"/>'
    )
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">{body}</svg>\n'


def seed_to_int(seed: str) -> int:
    if seed.strip().lower() in CANONICAL_NAMES:
        return CANONICAL_SEED
    return zlib.crc32(seed.strip().lower().encode())


def read_branding_defaults():
    """domain.toml [branding].mark_seed / mark_hue_shift, falling back to [project].name."""
    path = ROOT / "domain.toml"
    seed, hue = "lensing", 0.0
    try:
        import tomllib

        with open(path, "rb") as f:
            data = tomllib.load(f)
        seed = data.get("project", {}).get("name", seed)
        branding = data.get("branding", {})
        seed = branding.get("mark_seed", seed)
        hue = float(branding.get("mark_hue_shift", hue))
    except FileNotFoundError:
        pass
    return seed, hue


def main():
    d_seed, d_hue = read_branding_defaults()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", default=d_seed, help=f"instance seed (default from domain.toml: {d_seed!r})")
    ap.add_argument("--hue-shift", type=float, default=d_hue, help=f"ramp hue rotation in degrees (default {d_hue})")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "branding", help="where to write lensing-mark{,-dark}.svg")
    ap.add_argument("--install", action="store_true", help="also write ui/public/favicon.svg and ui/public/brand/")
    args = ap.parse_args()

    seed_int = seed_to_int(args.seed)
    paper = mark_svg("paper", seed_int, args.hue_shift)
    dark = mark_svg("dark", seed_int, args.hue_shift)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "lensing-mark.svg").write_text(paper)
    (args.out_dir / "lensing-mark-dark.svg").write_text(dark)
    written = [args.out_dir / "lensing-mark.svg", args.out_dir / "lensing-mark-dark.svg"]

    if args.install:
        (ROOT / "ui/public/favicon.svg").write_text(paper)
        brand = ROOT / "ui/public/brand"
        brand.mkdir(parents=True, exist_ok=True)
        (brand / "mark.svg").write_text(paper)
        (brand / "mark-dark.svg").write_text(dark)
        written += [ROOT / "ui/public/favicon.svg", brand / "mark.svg", brand / "mark-dark.svg"]

    for p in written:
        print(f"wrote {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
    print(f"seed={args.seed!r} (rng {seed_int}) hue_shift={args.hue_shift}")


if __name__ == "__main__":
    main()
