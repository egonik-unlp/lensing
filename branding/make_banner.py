#!/usr/bin/env python3
"""Generate the README hero banner (assets/logo.svg): the Control Room topbar
rendered as a lockup — a deep slate chrome bezel framing the Velocity Map mark,
the "lensing" wordmark in Sora 550 (on-chrome white), a dim instance form, and
a one-line tagline in Inter.

Why a generator and not a hand-drawn SVG: GitHub renders README SVGs through an
<img> sandbox that does not load @font-face web fonts, so the wordmark and
tagline are outlined to vector paths here. Re-run after editing the copy below.

  python3 branding/make_banner.py            # write assets/logo.svg
  python3 branding/make_banner.py --out /tmp/logo.svg

Identity rules it honors (DESIGN.md): chrome carries identity (slate, chroma
<=0.03); the velocity ramp lives only in the mark; no decorative orange (no
live state and no primary action on a static banner); Sora is the display face,
Inter the body face. Colors are the shipped Control Room tokens, converted to
sRGB hex because the <img> sandbox will not resolve oklch().
"""

import argparse
import math
import re
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from fontTools.pens.svgPathPen import SVGPathPen

ROOT = Path(__file__).resolve().parent.parent
SORA = ROOT / "ui/dist/assets/sora-latin-wght-normal-DdqRvwsR.woff2"
INTER = ROOT / "ui/dist/assets/inter-latin-wght-normal-Dx4kXJAl.woff2"
MARK_DARK = ROOT / "ui/public/brand/mark-dark.svg"

# --- copy --------------------------------------------------------------------
WORDMARK = "lensing"
INSTANCE = "· framework"  # the topbar instance form: [mark] lensing · <instance>
TAGLINE = "a prediction-lab framework that bends around your data, the way mass bends light"

# --- Control Room tokens (ui/src/styles/tokens.css), as OKLCH ----------------
TOK = {
    "chrome":        (0.27, 0.025, 250),
    "chrome_raised": (0.32, 0.025, 250),
    "chrome_border": (0.42, 0.025, 250),
    "on_chrome":     (0.96, 0.008, 250),
    "on_chrome_dim": (0.78, 0.015, 250),
}


def oklch_hex(L, C, H):
    a, b = C * math.cos(math.radians(H)), C * math.sin(math.radians(H))
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    r = 4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
    g = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
    bl = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_

    def enc(c):
        c = max(0.0, min(1.0, c))
        c = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
        return round(max(0.0, min(1.0, c)) * 255)

    return "#%02x%02x%02x" % (enc(r), enc(g), enc(bl))


C = {k: oklch_hex(*v) for k, v in TOK.items()}


class Text:
    """An outlined run of text. Glyph paths are kept in font units; placement
    scales and y-flips them (fonts are y-up, SVG is y-down)."""

    def __init__(self, text, font_path, weight, size):
        f = TTFont(str(font_path))
        instantiateVariableFont(f, {"wght": weight}, inplace=True)
        upm = f["head"].unitsPerEm
        self.scale = size / upm
        self.cap = (getattr(f["OS/2"], "sCapHeight", None) or int(0.7 * upm)) * self.scale
        glyphs, cmap, hmtx = f.getGlyphSet(), f.getBestCmap(), f["hmtx"]
        self.parts, x = [], 0.0
        for ch in text:
            gname = cmap[ord(ch)]
            pen = SVGPathPen(glyphs)
            glyphs[gname].draw(pen)
            if pen.getCommands():
                self.parts.append((x, pen.getCommands()))
            x += hmtx[gname][0]
        self.advance = x * self.scale

    def place(self, tx, baseline, fill):
        s = self.scale
        body = "".join(
            f'<path transform="translate({tx + gx * s:.2f} {baseline:.2f}) '
            f'scale({s:.5f} {-s:.5f})" d="{d}"/>'
            for gx, d in self.parts
        )
        return f'<g fill="{fill}">{body}</g>'


def mark_inner(size, tx, ty):
    raw = MARK_DARK.read_text()
    body = re.sub(r"^.*?<svg[^>]*>", "", raw, count=1, flags=re.S)
    body = re.sub(r"</svg>\s*$", "", body, flags=re.S)
    s = size / 48.0
    return f'<g transform="translate({tx} {ty}) scale({s:.5f})">{body}</g>'


def build():
    W, H = 980, 248
    pad = 60
    # --- mark ---
    ms = 108
    mx, my = pad, (H - ms) / 2 - 14
    mcy = my + ms / 2

    # --- wordmark (Sora 550), optical-center aligned to the mark center ---
    wm = Text(WORDMARK, SORA, 550, 96)
    wm_x = mx + ms + 30
    baseline = mcy + wm.cap / 2

    # --- instance form (Sora 550, dim), baseline-aligned, smaller ---
    ins = Text(INSTANCE, SORA, 550, 40)
    ins_x = wm_x + wm.advance + 22

    # --- tagline (Inter), under the wordmark; auto-fit to the right edge ---
    tag = Text(TAGLINE, INTER, 400, 25)
    avail = W - wm_x - pad
    if tag.advance > avail:  # shrink to fit rather than overflow the bezel
        tag = Text(TAGLINE, INTER, 400, 25 * avail / tag.advance)
    tag_baseline = baseline + 54

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="lensing: a prediction-lab framework that bends around your data, the way mass bends light">
  <title>lensing</title>
  <!-- The Control Room topbar as a lockup: slate chrome bezel, the Velocity Map
       mark, the Sora wordmark, a dim instance form, an Inter tagline. Wordmark
       and tagline are outlined to paths so they render without web fonts.
       Regenerate with branding/make_banner.py. -->
  <rect x="0" y="0" width="{W}" height="{H}" rx="16" fill="{C['chrome']}"/>
  <rect x="0.75" y="0.75" width="{W - 1.5}" height="{H - 1.5}" rx="15.25" fill="none" stroke="{C['chrome_border']}" stroke-width="1.5"/>
  {mark_inner(ms, mx, my)}
  {wm.place(wm_x, baseline, C['on_chrome'])}
  {ins.place(ins_x, baseline, C['on_chrome_dim'])}
  {tag.place(wm_x, tag_baseline, C['on_chrome_dim'])}
</svg>
'''
    return svg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "assets/logo.svg"))
    args = ap.parse_args()
    Path(args.out).write_text(build())
    print("wrote", args.out)
    print("colors:", C)
