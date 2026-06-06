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
