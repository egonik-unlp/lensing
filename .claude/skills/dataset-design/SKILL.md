---
name: dataset-design
description: Design and build datasets (feature matrices + train/test splits derived from the Qdrant corpus) via the lensing-server API — preflight quality filters, analyze feature spectra, build, inspect, and rename datasets. Use when the user wants to build a new dataset, preview what a quality/currency filter would drop, analyze PCA variance or feature redundancy before a build, inspect a dataset's manifest/items/split, or rename a dataset.
user-invocable: true
argument-hint: "[preflight|analyze|build|inspect|rename] [id]"
allowed-tools:
  - Read
  - Bash(curl *)
  - Bash(sleep *)
---
<!-- GENERATED from agents-src/skills/dataset-design/SKILL.md by agents-src/render.py — edit the template (and domain.toml), not this file; then run `zig build render-agents`. -->

Design and build **datasets**: feature matrices + train/test splits built
server-side from the Qdrant corpus, identified as `ds-…` and consumed by runs
and the model-definitions experiment workflow.

## Concepts — keep these straight

- A **dataset** is an immutable build artifact under `data/datasets/<id>/`
  (gitignored): manifest (provenance: feature config, quality filters,
  currency handling, PCA basis), feature matrix, and split.
- Builds are **async and serialized** — one at a time server-side; `POST`
  returns a `build_id` to poll, it does not block.
- Renaming sets a display `name` only; the `ds-…` id and directory never
  change, so run/definition lineage is unaffected.

## Ground rules

- The server must be running (default `http://localhost:8080`; start with
  `zig build serve` if needed). Never write under `data/` by hand.
- Pretty-print JSON responses for the user (pipe through `python3 -m json.tool`
  or summarize the relevant fields).
- For the dataset format / predictor contract, read README.md — don't guess.
- Iterate filters with **preflight** (cheap, builds nothing) before committing
  to a build; agree the config with the user first.

## Build parameters

`POST /api/datasets` body (every field optional; defaults shown):

| field | default | meaning |
|---|---|---|
| `pca_dims` | 32 | PCA dims of the embedding block (1..=1536) |
| `test_ratio` | 0.2 | test split fraction (0.05..=0.5) |
| `seed` | 42 | split shuffle seed |
| `log_target` | true | train on log1p(target) |
| `fields` | `{}` | per-field enables, keyed by field or group name (below); when non-empty, authoritative |
| `vocab_top_n` | `{}` | per-categorical vocabulary-size overrides (field name → top-N) |
| `area_content_backfill` | false | backfill missing areas from "… m²" mentions in the document text |
| `impute_numerics` | false | fill missing reconciled numerics with train-split group medians (drops indicator columns) |
| `numerics_collection` | `""` | companion collection for the reconciled-numerics join |
| `collection` | server's | source Qdrant collection |
| `quality` | see below | quality filter config |
| `currency` | see below | currency handling |

The feature fields come from the domain config (`GET /api/domain`, or read
`domain.toml`); pass them in `fields`, e.g.
`{"fields": {"raw_numerics": true, "coordinates": true, "city": false}}`.
For this domain:

| field | default | what it is |
|---|---|---|
| `numeric_example` | true | numeric_example; numeric |
| `category_example` | true | category_example; categorical; one-hot over all values + `__other__` |

(Pre-domain clients may still send the legacy named flags — `bedrooms`,
`property_type`, `neighborhood_top_n`, `city`, `province`, `cluster`,
`raw_numerics`, `coordinates` — which apply only when `fields` is empty.)

`quality` (rule toggles + thresholds; default excludes only nonpositive
target values — rule keys are stable identifiers bound to this domain's
fields by domain.toml): `nonpositive_price` (true), `price_outlier` (false,
MAD z on the log target per outlier group, threshold `price_outlier_mad_z`
3.5), `price_range` (false, hard caps `price_min` / `price_max`),
`bedrooms_outlier` (false, cap `bedrooms_max` on the domain's capped
numeric), `duplicate_content` (false), `short_content` (false, floor
`short_content_min_chars` 80), `missing_fields` (false, the domain's
critical fields).

`currency` (domains with a `[currency]` section in domain.toml; single-
currency domains ignore it): `mode` `"filter"` (default; also `"off"` |
`"convert"`), `keep` `""`, `reconcile_collection`
`""`, `rate_source` (the rate series for `convert`).


## Workflows

### Preflight quality filters (no build)

```sh
curl -s localhost:8080/api/datasets/preflight -H content-type:application/json \
  -d '{"quality":{...},"currency":{...},"sample":8}'
```
Synchronous; returns `n_total`, `n_excluded_total`, per-rule flag counts, and
sample flagged rows (`sample` ≤ 50). Counts are reported for every rule even
when disabled — show the user what each toggle *would* drop, then iterate.

### Analyze (EVR spectrum + feature redundancy, no build)

```sh
curl -s localhost:8080/api/datasets/analyze -H content-type:application/json -d '{<build params>}'
```
Takes the same body as a build, returns `{"job_id": ...}` — poll
`GET /api/jobs/<job_id>` (`sleep` a few seconds between rounds) until `state`
is `done` (full result under `.result`) or `failed`. Use it to pick
`pca_dims` and spot redundant features before building.

### Build

```sh
curl -s localhost:8080/api/datasets -H content-type:application/json -d '{<build params>}'
```
Returns `{"build_id": ...}` — poll `GET /api/builds/<build_id>`:
`{"state":"building","stage":...}` → `{"state":"done","dataset_id":"ds-..."}`
or `{"state":"failed","error":...}`. Then `GET /api/datasets/<id>` to confirm
the manifest (row counts, feature config, filter provenance) and report the
new id. Builds queue behind any in-flight one.

### Inspect

- `GET /api/datasets` — list all (id, name, counts).
- `GET /api/datasets/<id>` — manifest: provenance, feature config, sizes.
- `GET /api/datasets/<id>/items` — rows.
- `GET /api/datasets/<id>/split` — train/test row ids.

### Rename

```sh
curl -s localhost:8080/api/datasets/<id>/rename -H content-type:application/json -d '{"name":"<display name>"}'
```

## Delegate to the dataset-architect agent

The workflows above are for quick, user-driven operations (one preflight, an
inspect, a rename, a single agreed build). For bigger jobs — several dataset
variants for a scan, lineage-aware design (what already exists, what fed the
champions), or anything the user doesn't want to drive call-by-call — spawn
the **dataset-architect** agent with the brief instead:

- Default is **design mode**: it mines `experiments/PROJECT-FACTS.md`'s dataset lineage,
  preflights, runs EVR/redundancy analysis, and returns a proposal with exact
  build bodies — building nothing. Relay the proposal to the user.
- To iterate, continue the SAME agent (SendMessage) with the feedback; once
  the user approves, continue it with the approval and it builds, verifies
  the manifests, and records the new datasets in `experiments/PROJECT-FACTS.md`.

## Relation to experiments

The **model-definitions** skill's experiment workflow scans hyperparameters on
*existing* datasets. When a scan's axis is dataset-level (PCA dims, quality
filters, features like `raw_numerics`), build one dataset per axis value
first — inline via this skill, or via the dataset-architect agent — then hand
the `ds-…` ids to the scan. The experiment-designer agent consults the
architect directly for the same purpose.
