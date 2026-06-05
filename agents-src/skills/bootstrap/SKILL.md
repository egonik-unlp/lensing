---
name: bootstrap
description: Initialize this template around a NEW dataset/prediction problem: probe a Qdrant corpus and infer its payload schema, interview the user into a domain.toml (field roles, quality levers, metrics, nouns), render the agent layer, reset the empirical record, decide how manual entries are fetched/presented (ingestion), build the first dataset with tuned quality levers, and set up the starter model definitions for a first evaluation. Use when the user wants to bootstrap, initialize, or adapt this framework to their own data/target variable, or asks "how do I point this at my dataset".
user-invocable: true
argument-hint: "[probe|configure|reset|ingestion|first-run] "
allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Bash(curl *)
  - Bash(python3 *)
  - Bash(zig build *)
  - Bash(docker compose *)
  - Bash(cargo test *)
  - Bash(sleep *)
---

Initialize this Lensing template around the user's dataset. This skill
automates BOOTSTRAP.md — read it first; it is the authoritative checklist.
Everything flows from **`domain.toml`** (the single source of domain truth)
and the corpus assumption: a Qdrant collection of embedded documents with
JSON metadata payloads.

Work phase by phase, conversationally — every phase ends with the user
confirming before you write anything. Resume at whatever phase the argument
names or the repo state implies (e.g. domain.toml already customized → skip
to ingestion/first-run).

## Phase −1 — Environment & toolchain

Check what's installed before anything else; install only with the user's
explicit go-ahead, per tool:

| tool | check | install if missing |
|---|---|---|
| Rust (cargo/rustc) | `cargo --version` | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh -s -- -y` then `. ~/.cargo/env` |
| zig (build runner) | `zig version` | their package manager, or https://ziglang.org/download |
| docker + compose | `docker compose version` | their package manager / Docker docs |
| node + npm (UI) | `node --version` | their package manager / nvm |
| python ≥ 3.11 | `python3 --version` | their package manager |

Then the Lensing code itself:

- **Working inside this template checkout** (the normal case): everything
  is already here — `cargo build --release --workspace` (the committed
  `Cargo.lock` pins the dependency tree) and `zig build` drive the rest.
- **Once the Lensing crates are published to crates.io**: the Rust binaries
  can instead be fetched with
  `cargo install lensing-server lensing-pipeline lensing-predictor-burn-mlp lensing-predictor-burn-cnn lensing-predictor-blend`
  — but the template repo remains the full distribution (UI, Python/Julia
  predictors, registry, agent layer aren't crates). Prefer the template;
  mention `cargo install` for binary-only hosts (training workers,
  inference nodes — see docs/DEPLOY.md).
- One-time predictor toolchains as needed: `zig build py-setup` (Python
  venv), `zig build julia-setup` (Flux predictors).

## Phase 0 — Probe the corpus

Qdrant first (default `{{qdrant_url}}`; ask if theirs differs):

```sh
curl -s <qdrant>/collections                                   # what exists
curl -s -X POST <qdrant>/collections/<name>/points/scroll \
  -H content-type:application/json \
  -d '{"limit": 16, "with_payload": true, "with_vector": false}'
```

From the sample, build and SHOW a schema inference table: payload key →
observed JSON type → coverage in sample → proposed role (`target` |
`categorical` | `numeric` | `coordinates` | `filter_only` | `timestamp` |
`display`) → proposed options (encode log1p for heavy-tailed sizes,
indicator for sparse numerics, vocab top-N for high-cardinality
categoricals, critical for must-have fields). Also: where the metadata
nests (`metadata_root`), which key holds the embedded text
(`content_field`), the embedding dim (scroll once with
`"with_vector": true, "limit": 1`), and a candidate corpus filter. If they
have no corpus yet, explain the requirement (vectors + JSON payloads) and
offer `docker compose --profile qdrant up -d` plus their own embedding
pipeline — this skill does not embed documents.

## Phase 1 — Interview → domain.toml

Ask only what you cannot infer; propose defaults for everything else:

1. **Nouns + title**: entity noun ("listing"/"vehicle"/"posting"), target
   noun, project name/title.
2. **Target**: field, transform (`log1p` for heavy-tailed positive targets —
   the default and well-trodden path; `none` otherwise), display format
   (symbol/locale).
3. **Fields = the dataset-pane levers.** Walk the inference table together.
   Each `[[fields]]` entry becomes a build-form toggle (`default_on`), and
   the `[quality]` bindings become the preflight/filter levers in the
   dataset pane: `outlier_group` (which categorical groups the
   target-outlier MAD rule), `capped_numeric` (which numeric gets the
   sanity cap), `critical` fields (the missing-fields rule), plus per-rule
   labels in the user's vocabulary. Field ORDER is column order — put
   numerics first in the order they should appear.
4. **Coordinates** (only with a geo field): plausible lat/lon bounds —
   out-of-bounds geocodes count as missing.
5. **Currency**: multi-currency corpus → `[currency]` (pair, rate endpoint
   template, reconcile collection); single-currency → delete the section.
6. **Metrics**: primary + columns + % metrics + value unit.
7. **[agents]**: API port, naming convention, `ingestion` on/off (decided
   properly in Phase 3).

Then: write `domain.toml`, validate via `cargo test -p pg-core` (the
embedded-domain tests re-read it) — fix anything it rejects, and run
`zig build render-agents` so the skills/agents speak the new domain.
`zig build check` must pass before moving on.

## Phase 2 — Reset the empirical record (DESTRUCTIVE — explicit confirmation)

The repo ships the original project's history. Confirm each, then:

- Move `{{report_dir}}/*.md` reports into `{{report_dir}}/archive/` (or
  delete if the user prefers); seed `{{facts_file}}` with the empty skeleton
  (leaderboard / noise bands / lineage / pitfalls tables + "No facts yet —
  the first campaign seeds this").
- Empty `models.toml`'s `[[definitions]]` entries (server must be stopped).
- Reset `docs/experiments.tex` title/abstract to the new project; trim
  `docs/figures/make_figures.py` to its style header (set PRIMARY_METRIC /
  VALUE_UNIT); rewrite PRODUCT.md's one-pager.
- Never touch `data/` or the Postgres volume of the ORIGINAL project if the
  user is bootstrapping in a copy — and if they bootstrapped in-place by
  mistake, stop and say so before deleting anything.

## Phase 3 — Ingestion: how their "{{entity_noun}} equivalent" arrives

Decide together how one-off entries are fetched and presented:

| option | what to do |
|---|---|
| **Scrape from source sites** | Rewrite the fenced "Example domain notes" section of `agents-src/agents/listing-generator.md` with THEIR source-site lore (portals, embedded-JSON tricks, field-name translations, regional quirks); keep `[agents] ingestion = true`; re-render. |
| **Manual entry only** | `[agents] ingestion = false` (drops the agent); the UI's New-entry form (generated from `[[fields]]`, with `suggestions` datalists and `required` flags) is the whole story — tune those field options now. |
| **API/CSV import later** | Same as manual for now; note it as a follow-up (the `POST /api/listings` flat-fields shape is the integration point). |

Either way, review how an entry is *presented*: field `label`s, `display`
role fields (photos, source links), and which fields the predict path must
strip (the target + display fields — automatic).

## Phase 4 — First run: levers live in the dataset pane

```sh
zig build db-up          # Postgres (compose)
zig build serve          # in the user's terminal — long-running
curl -s {{api_host}}/api/collections/validate -H content-type:application/json \
  -d '{"collection": "<theirs>"}'        # shape verdict + field coverage
curl -s {{api_host}}/api/datasets/preflight -H content-type:application/json \
  -d '{"quality": {...}, "sample": 8}'   # per-rule counts BEFORE building
```

Iterate the preflight with the user — these counts are the levers doing
their job (outlier thresholds, ranges, content floors). When the config
reads right, build the first dataset (the **dataset-design** skill owns the
details), then open `/datasets/<id>` and confirm the pane shows their
fields, their rule labels, their target histogram.

## Phase 5 — Starter models for the first evaluation

Create definitions (via the **model-definitions** skill) in this order, and
explain why:

1. `baseline-median` — the floor; every model must beat "group median".
2. `xgboost` — the low-drama default: bounded predictions (no transform
   blow-ups), native handling of one-hots/indicators/missing values.
3. Optional third seat by data shape: `burn-mlp` **with
   `clamp_output: true`** (mandatory insurance for any log-space MLP) when
   rows ≳ 10k; `svm`/`kernel-ridge` when rows ≲ 5k.

Launch one run each on the first dataset, watch them complete, then seed
`{{facts_file}}`'s leaderboard with the results. Hand the user off to the
**experiment-designer** agent for the first real campaign.

## Ground rules

- Every file write and every destructive step is announced and confirmed
  first. Phase 2 doubly so.
- Never write under `data/` by hand; never hand-edit the rendered
  `.claude/.agents/.gemini` outputs (edit `agents-src/` + `domain.toml`,
  then `zig build render-agents`).
- The Rust crates, predictors, `registry.toml` and the UI should NOT need
  changes — if the domain seems to require it (a missing quality-rule kind,
  a new field role), say so explicitly rather than improvising code.
