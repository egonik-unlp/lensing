---
name: dataset-design
description: Design and build datasets (feature matrices + train/test splits derived from the Qdrant corpus) via the lensing-server API — preflight quality filters, analyze feature spectra, build, inspect, and rename datasets. Use when the user wants to build a new dataset, preview what a quality/currency filter would drop, analyze PCA variance or feature redundancy before a build, inspect a dataset's manifest/items/split, or rename a dataset.
user-invocable: true
argument-hint: "[preflight|analyze|build|inspect|rename] [id]"
allowed-tools:
  - Read
  - Bash(curl *)
  - Bash(python3 tools/lensing_guard.py *)
  - Bash(sleep *)
---

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

- The server must be running (default `{{api_base_url}}`; start with
  `zig build serve` if needed). Never write under `data/` by hand.
- Pretty-print JSON responses for the user (pipe through `python3 -m json.tool`
  or summarize the relevant fields).
- For the dataset format / predictor contract, read README.md — don't guess.
- Iterate filters with **preflight** (cheap, builds nothing) before committing
  to a build; agree the config with the user first.

## Build parameters

{{> dataset-build-params}}

## Workflows

### Preflight quality filters (no build)

```sh
curl -s {{api_host}}/api/datasets/preflight -H content-type:application/json \
  -d '{"quality":{...},"currency":{...},"sample":8}'
```
Synchronous; returns `n_total`, `n_excluded_total`, per-rule flag counts, and
sample flagged rows (`sample` ≤ 50). Counts are reported for every rule even
when disabled — show the user what each toggle *would* drop, then iterate.

### Analyze (EVR spectrum + feature redundancy, no build)

```sh
curl -s {{api_host}}/api/datasets/analyze -H content-type:application/json -d '{<build params>}'
```
Takes the same body as a build, returns `{"job_id": ...}` — poll
`GET /api/jobs/<job_id>` (`sleep` a few seconds between rounds) until `state`
is `done` (full result under `.result`) or `failed`. Use it to pick
`pca_dims` and spot redundant features before building.

### Design a latent representation (compression: PCA / autoencoder)

`pca_dims` is one point on a bigger axis: **how the corpus is represented**
before a dataset ever sees it. A *representation* is a compressor (PCA, an
autoencoder, or a sparse autoencoder) fit over a source collection, producing a
dense latent collection that datasets can then build on — the "Representations"
section of the UI, backed by `lensing-compression`.

```sh
# List / inspect
curl -s {{api_host}}/api/representations
curl -s {{api_host}}/api/representations/<id>
# Build one (async, polls like a dataset build via GET /api/builds/<id>)
curl -s {{api_host}}/api/representations -H content-type:application/json -d '{
  "name": "Song AE", "method": "autoencoder", "latent": 64,
  "source_collection": "<corpus>", "sink_collection": "<corpus>_song_ae",
  "epochs": 200, "hidden": [256, 64] }'
```

Pick method + latent dim **justified by a metric**, never a round number:
- **PCA** → cumulative **EVR** (same curve as `analyze`); report EVR captured at
  the chosen `latent`.
- **autoencoder / sparse-AE** → per-block **reconstruction R²** (each modality
  weighted equally, so a wide text block doesn't drown a narrow acoustic one).

A representation proposal that picks a `latent` without an EVR or block-R²
number behind it is not done. Once built, a dataset can be built on the latent
collection (it appears in the source-collection picker tagged "latent
representation").

### Build

```sh
curl -s {{api_host}}/api/datasets -H content-type:application/json -d '{<build params>}'
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
curl -s {{api_host}}/api/datasets/<id>/rename -H content-type:application/json -d '{"name":"<display name>"}'
```

## Delegate to the dataset-architect agent

The workflows above are for quick, user-driven operations (one preflight, an
inspect, a rename, a single agreed build). For bigger jobs — several dataset
variants for a scan, lineage-aware design (what already exists, what fed the
champions), or anything the user doesn't want to drive call-by-call — spawn
the **dataset-architect** agent with the brief instead:

- Default is **design mode**: it mines `{{facts_file}}`'s dataset lineage,
  preflights, runs EVR/redundancy analysis, and returns a proposal with exact
  build bodies — building nothing. Relay the proposal to the user.
- To iterate, continue the SAME agent (SendMessage) with the feedback; once
  the user approves, continue it with the approval and it builds, verifies
  the manifests, and records the new datasets in `{{facts_file}}`.

## Relation to experiments

The **model-definitions** skill's experiment workflow scans hyperparameters on
*existing* datasets. When a scan's axis is dataset-level (PCA dims, quality
filters, features like `raw_numerics`), build one dataset per axis value
first — inline via this skill, or via the dataset-architect agent — then hand
the `ds-…` ids to the scan. The experiment-designer agent consults the
architect directly for the same purpose.
