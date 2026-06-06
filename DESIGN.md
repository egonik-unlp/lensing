<!-- Tokens resolved 2026-06-04 from the shipped implementation (ui/src/styles/tokens.css). -->
<!-- For a full component capture, run /impeccable document. -->
---
name: Lensing
description: Internal tool for evaluating and comparing prediction ML models
---

# Design System: Lensing

## 1. Overview

**Creative North Star: "The Control Room"**

A deep slate-ink bezel frames a bright paper readout. The chrome (topbar, view bands, selection bars) is dark and structural, like the housing of a Braun instrument; the data surfaces inside it stay paper-white and dense. One signal orange is reserved for exactly two meanings: something is *live* (a training run, its loss trace, its pulse) or something is *the* primary action on this screen. Everything else reads in ink, graphite, and hairlines.

The references are the Braun ET66 calculator (dark housing, bright readout, one orange key), McMaster-Carr's catalog density, and Swiss rail timetables (rules, tabular numerals, no decoration). This system explicitly rejects the generic admin template, flashy SaaS polish (gradients, glassmorphism, glow), and the cluttered BI-tool look — and equally the all-light "quiet tool" look it replaced ("The Measurement Bench"): the chrome now carries identity so the data doesn't have to.

**Key Characteristics:**
- Two-tone committed color: ~30% slate chrome framing near-zero-chroma paper surfaces
- Signal orange ≤10% of any screen: live state + one primary action, never decoration
- Links and selection are slate; chart series are graphite (A) and blue (B); orange means live
- Sans for UI, JetBrains Mono with tabular figures for every number, value, ID, and log line
- Denser than before: 44px topbar, 13px body, 12.5px tables, tight row padding, density toggle on every entity table
- Flat elevation: hairlines and tone steps at rest; shadow only on things that float (menus, the compare bar, the log "jump to latest" pill)

## 2. Colors

Shipped values (OKLCH, see `ui/src/styles/tokens.css`):

### Chrome (the bezel)
- `--chrome: oklch(0.27 0.025 250)` — topbar, view bands, selection bars. Chroma capped ≤0.03 so it reads ink-slate, never blue-brand.
- `--chrome-raised / -hover / -border`: 0.32 / 0.36 / 0.42 lightness steps for controls on chrome.
- `--on-chrome: oklch(0.96 0.008 250)` (~13:1), `--on-chrome-dim: oklch(0.78 0.015 250)` (~7.5:1, meta/breadcrumbs), `--on-chrome-faint: oklch(0.66 0.02 250)` (~4.9:1, large labels only).

### Paper (the readout)
- `--bg: oklch(0.992 0 0)`; `--surface: oklch(0.972 0.002 250)`; `--surface-sunken: oklch(0.95 0.003 250)`; hairlines at hue 250, chroma ≤0.005. No cream, ever.
- `--ink: oklch(0.24 0.01 250)` (~16:1); `--ink-secondary: oklch(0.45 0.012 250)` (~7:1); `--ink-faint: oklch(0.56 0.01 250)` (4.5:1, meta only).

### Signal orange (the one accent)
- `--accent: oklch(0.68 0.18 47)` fills (primary button, live indicators); `--accent-strong` hover + live traces; `--accent-text: oklch(0.5 0.155 42)` (6.3:1 on bg); `--accent-on-chrome: oklch(0.78 0.16 55)` (7:1 on chrome — nav underline, wordmark glyph); `--accent-ink` text on orange fills (5.9:1); `--accent-soft` live-row tint only.

### Navigation & selection (slate, not orange)
- `--link: oklch(0.42 0.05 250)` (8.2:1) — every EntityRef and text link.
- `--select-tint: oklch(0.945 0.006 250)` + `--select-border: oklch(0.6 0.04 250)` — selected rows, options, active toggles. Selection means *chosen*, not *alight*.

### Series (charts)
- `--series-a: oklch(0.35 0.012 250)` graphite — the run under the lens; solid fill/stroke.
- `--series-b: oklch(0.55 0.13 245)` blue — the comparator; always dashed/outline + direct label.
- A finished run's val trace is graphite; a *live* run's val trace is orange with a pulsing head. Orange in a chart is a status, not a palette choice.

### Semantic
- `--bad: oklch(0.5 0.17 27)`, `--good: oklch(0.5 0.105 155)` — always paired with a glyph or word (✕/✓/▲/▼), never hue alone.

### Named Rules
**The Live-or-Primary Rule.** Orange appears only on what is live (running status, live trace, live sparkline) and on at most one primary action per screen. If two things are orange and neither is live, one is wrong.

**The Slate-Not-Blue Rule.** Chrome chroma stays ≤0.03. If it starts reading "blue dashboard," reduce chroma, never lightness.

**The No-Cream Rule.** Paper stays at near-zero chroma. Identity lives in the chrome and the orange, never in a tinted body background.

## 3. Typography

**Display Font:** Sora Variable 550 (`--font-display`, self-hosted) — the brand face: the wordmark, page titles (h1 / view-band titles) and section titles. **UI Font:** Inter Variable (self-hosted) — everything below section scale: body, tables, forms; geometric sans at dense-table sizes hurts scanability, so Inter holds the data layer. **Mono:** JetBrains Mono 400/600 — every number, value, ID, metric, hyperparameter and log line; tabular figures always.

Fixed rem scale (denser than v1): 11.5 label / 12.5 dense-table / 13 body / 15 panel title / 18 section / 22 page title / 24 metric hero. The measurement still outranks the page title.

### Named Rules
**The Tabular Figures Rule.** A column of values that doesn't align vertically is a bug, not a style choice.

**The Display Boundary Rule.** Sora stops at section scale (18px). Panel titles, body, tables and forms are Inter; if Sora appears in a data row, it has leaked.

## 3b. Brand mark

The mark is the **Velocity Map**: a dark sphere (the unobservable mass) inside a swarm of tangentially smeared trails colored by line-of-sight velocity — blueshifted approaching, redshifted receding — with a Doppler-weighted rim. It is generated, not drawn: `branding/make_mark.py --install` writes `ui/public/favicon.svg` and `ui/public/brand/mark{,-dark}.svg` from `domain.toml [branding]`.

- **Ramp colors are brand-surface only.** The red→orange→blue shift ramp is anchored to the system's own `--bad` / `--accent` / `--series-b` hues (interpolated in Oklab), but it lives exclusively in the mark and brand surfaces. On screens, the Live-or-Primary rule still governs: orange means live or primary, red/green stay semantic, and nothing in the UI borrows the ramp.
- **Sub-instance branding.** Lensing deployments keep the grammar (dark sphere, velocity swarm, Sora wordmark, the `lensing · <instance>` topbar form) and inject personality through `mark_seed` (arc arrangement) and `mark_hue_shift` (ramp rotation) in `domain.toml [branding]`. The canonical lensing mark is seed `"lensing"`, hue 0.
- **Wordmark form.** Topbar reads `[mark] lensing · <instance title>` — "lensing" in Sora 550 on-chrome, the instance title in `--on-chrome-dim`.

## 4. Structure

**The band.** Every view opens with a full-bleed slate band (`ViewHeader`): breadcrumb line on detail pages (Runs / run_xxx), glyph + title + count, a dim meta line, and an actions slot holding at most one orange primary. The band scrolls away (density wins); only the 44px topbar and table headers pin.

**The data column.** Below the band, content sits in `.view-body` (max 1320px). Tables share one row vocabulary (`.run-row` hover/selected/focused, `data-density` on every entity table, `.col-group-start` hairlines between identity | metrics | time groups). Metrics share one cell vocabulary (`MetricStrip` / `MetricCell`; drillable cells carry a resting underline). Forms share one submit grammar (`SubmitRow`: named action, busy state, block reason, inline error, scroll-to-first-error).

## 5. Elevation

Flat at rest. Hairlines and tone steps convey structure. `--shadow-overlay` appears only on floating state: the compare selection bar, menus, the log pane's jump pill.

## 6. Do's and Don'ts

### Do:
- **Do** put every view under a slate band with breadcrumbs on detail pages; the bezel is the navigation context.
- **Do** keep orange for live + one primary action; recolor anything else to slate/graphite.
- **Do** render every number in the mono with tabular figures; right-align numeric columns.
- **Do** pass `data-density` to every entity table and keep the toggle in the band.
- **Do** hold WCAG AA everywhere, including on-chrome text (all shipped pairs ≥4.5:1).

### Don't:
- **Don't** put two oranges on one screen unless one of them is live state.
- **Don't** tint selection orange; selection is slate (`--select-tint` + `--select-border`).
- **Don't** let chrome chroma climb past 0.03 (blue-brand creep) or paper pick up warmth (cream creep).
- **Don't** use shadows on resting surfaces, side-stripe accents, gradients, or glassmorphism.
- **Don't** smooth, hide, or round away inconvenient numbers; outliers render at full magnitude.
