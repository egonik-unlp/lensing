# Lensing brand

**Decided:** the mark is **X "Velocity Map"** (a dark sphere inside a swarm of
tangentially smeared trails colored by line-of-sight velocity, with a
Doppler-weighted rim) and the wordmark face is **Sora 550**, used as the
display layer across the project (wordmark, page titles, section titles; body
stays Inter, numbers stay JetBrains Mono). See DESIGN.md §3 and §3b for the
rules, including the brand-surface-only scope of the red/blue ramp.

## The generator

`make_mark.py` produces the mark from `domain.toml [branding]`:

```sh
python3 branding/make_mark.py            # branding/lensing-mark{,-dark}.svg
python3 branding/make_mark.py --install  # + ui/public/favicon.svg, ui/public/brand/mark{,-dark}.svg
python3 branding/make_mark.py --seed my-lab --hue-shift 40 --out-dir /tmp/x
```

- `mark_seed` — arc arrangement. `"lensing"` is the canonical mark;
  bootstrapped instances usually use their project name.
- `mark_hue_shift` — degrees of hue rotation applied to the red→orange→blue
  ramp (anchored to the tokens' `--bad` / `--accent` / `--series-b`,
  interpolated in Oklab). The grammar (sphere, swarm, rim, Sora) is constant;
  these two knobs are an instance's personality.

## The lensed-lattice companion

`make_lattice.py` generates the brand surfaces that carry the other half of the
idea: a generic framework grid taking the shape of your data the way mass
curves spacetime. It **complements** the Velocity Map mark, it does not replace
it.

```sh
python3 branding/make_lattice.py   # assets/lattice.svg, assets/lattice-data.svg,
                                    # ui/public/brand/lensing-icons.svg
```

- `assets/lattice.svg` — the README concept card: a warped lattice drawn into a
  central mass, lit by an Einstein ring. The README hero alongside the banner.
- `assets/lattice-data.svg` — a straight row of data points bent into a lensed
  arc as it passes the mass.
- `ui/public/brand/lensing-icons.svg` — an icon family built from the Einstein-
  ring alignment sweep (dataset / run / model / predict / compare / export):
  stroke inherits `currentColor`, `--brand-src` tints the source dot. It feeds
  the app's empty-state mark; it is **not** a replacement for the text-monogram
  EntityRef glyphs (run / ds / ml / def / pr / ls), which remain the app's
  identity primitive.

`explore/` keeps the direction panel these were chosen from
(`python3 branding/explore/make_explore.py`, then open `explore/index.html`):
the lattice well (mass present / implied / off-center), the Einstein-ring
alignment sweep, and the lensed-data-row candidates.

## The animated README figures

`make_animations.py` generates the three moving figures the README opens with.
They are outlined and declarative for the same reason the banner is: GitHub
renders README SVGs through an `<img>` sandbox that loads no web fonts and runs
no scripts, so every glyph is a path and all motion is CSS keyframes plus SMIL
`animateMotion`. Each figure also carries a `prefers-reduced-motion` block that
pins it to a readable end state.

```sh
python3 branding/make_animations.py            # all three into assets/
python3 branding/make_animations.py --only loop.svg
python3 branding/make_animations.py --freeze 13 --outdir /tmp/f  # debug a frame
```

- `assets/session.svg` — a session start to finish: two plain-language
  requests, a preflight, a six-run scan with a live val-loss trace, and the
  report the campaign leaves behind. The 26 s loop.
- `assets/harnesses.svg` — Claude Code / Gemini CLI / Codex reading the same
  rendered skill layer and driving the same server. 9.6 s.
- `assets/loop.svg` — corpus → dataset → runs → model → record, and the dashed
  return edge that makes the record the next design's input. 12 s.

The generator needs the shipped font faces on disk (`zig build ui`, or
`npm install` in `ui/`) and honors the Live-or-Primary rule: orange marks the
runs while they are live and goes graphite the moment they are not.

## Design record

`index.html` is the candidate showcase from the exploration that led here
(open in a browser; `?only=X` filters by key). Twelve candidates across three
directions are kept for the record:

- L/M/N/Q — long-exposure ink: trail swarm, caustic pileup, lensed star
  trails, Paczyński light curve.
- R/S/T/U — the lensed-source inversion: orange as the source's smeared,
  doubled, displaced light; the mass dark.
- V/W/X/Y — the redshift family: Doppler photon ring, trails + rendered
  sphere, **velocity map (winner)**, gravitational-redshift ladder.

Mark bodies live in `marks.js` (generated; paper + dark variants), standalone
files as `lensing-<key>-<name>{,-dark}.svg`. The wordmark studies section at
the top of the showcase records the font decision (Sora 550 vs STIX Two
italic, Familjen Grotesk, Bricolage Grotesque, Inter).
