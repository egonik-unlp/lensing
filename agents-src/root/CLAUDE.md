# {{project_title}}

A {{target_noun}}-prediction lab: a Qdrant corpus of embedded
{{entity_noun_plural}} feeds dataset builds (PCA + metadata features),
language-neutral predictor plugins train on them, and lensing-server orchestrates
runs, promoted models and predictions behind a React UI.

## Where domain truth lives

- **`domain.toml` is the single source of domain truth** — corpus schema,
  target variable, feature fields, quality-rule bindings, metrics, UI
  vocabulary, and the agentic-layer parameters. Bootstrapping this template
  into a new problem = editing that file (see BOOTSTRAP.md).
- The files under `.claude/`, `.agents/` and `.gemini/` (skills + agents,
  except `impeccable`) are **GENERATED** from `agents-src/` — edit the
  template and run `zig build render-agents`; never hand-edit the outputs.
  `zig build check` fails on drift.
- Volatile empirical knowledge (leaderboard, noise bands, per-family
  pitfalls, dataset lineage) lives in `{{facts_file}}` — agents read it
  first and reconcile it after every campaign. The `{{report_dir}}/*.md`
  campaign reports are primary; the newest report wins.

## State & persistence

- **Postgres** (docker compose; `zig build db-up`) is the metadata store:
  model definitions (authoritative; `models.toml` is the git-diffable
  export), runs + metrics, promoted-model records, the best-models group,
  dataset index, run events. `zig build migrate-data` backfills it from the
  files and prints a consistency report — it never modifies the files.
- The **best-models group** (top-{{best_models_size}} by {{primary_metric}})
  is server-maintained: recomputed on every run completion and model
  promotion/deletion, auto-promoting top runs. Served at
  `GET /api/best-models`; consensus predictions at
  `POST /api/best-models/predict`; curation (pin/exclude) via
  `PUT /api/best-models`. Mirror: `data/best-models.json` + the
  `best_models` table.
- **`data/`** holds the binary artifacts (datasets, run dirs, model
  snapshots). NEVER write under `data/` by hand.
- The Qdrant corpus (`{{collection}}` at `{{qdrant_url}}`) is external;
  manual entries live in `{{manual_collection}}`.

## Non-negotiables

- The server (`{{api_base_url}}`) must already be running for skills/agents;
  **never restart lensing-server** without checking `GET /api/runs` for live
  training runs first — they die on restart.
- Never hand-edit `models.toml` while the server runs (it is rewritten on
  every mutation).
- Talk to the system through the API, not the filesystem.

## Build entry points

| command | what it does |
|---|---|
| `zig build serve` | build backend + UI, start Postgres, run lensing-server |
| `zig build db-up` / `db-down` | start/stop the compose services (pgdata volume survives) |
| `zig build migrate-data` | idempotent file→Postgres backfill + consistency report |
| `zig build dataset` | build a dataset with default flags |
| `zig build render-agents` | render `.claude/.agents/.gemini` from `agents-src/` + `domain.toml` |
| `zig build check` | tests + lint + py-check + agent-template drift |
| `zig build worker` / `infer` | distributed training / inference roles (docs/DEPLOY.md) |
| `zig build package` | distributable template under `dist/` |

## Skills & agents

- **dataset-design** — preflight quality filters, analyze spectra, build and
  inspect datasets inline; delegates bigger jobs to the dataset-architect
  agent.
- **dataset-architect** (agent) — designs datasets (lineage-mined,
  preflighted, EVR-justified proposals) and builds them on approval; spawned
  via the dataset-design skill or by experiment-designer for dataset-level
  experimental needs.
- **model-definitions** — define/clone/tag/launch model definitions; the
  experiment workflow (design → launch → collect → report → reconcile
  `{{facts_file}}`).
- **experiment-designer** (agent) — designs the next experiment with you:
  mines the record, proposes a self-contained design (consulting the
  dataset-architect for dataset-level axes); iterate via SendMessage.
  Design-only — it never launches.
- **experiment-runner** (agent) — executes a user-approved design verbatim:
  launches, babysits, collects, reports, reconciles the facts file. Refuses
  incomplete designs.
- **best-model-selector** (agent) — curates the best-models group with the
  judgment the deterministic recompute can't apply (family diversity,
  suspicious-metric exclusion, pinning); spawn after a campaign concludes or
  new models are registered.
- **report-curator** — maintains docs/experiments.tex (+ figures) from the
  campaign reports; destructive edits stop for approval.
- **listing-generator** (agent) — URL → manual {{entity_noun}} → consensus
  predictions via `POST /api/best-models/predict`.
- **upstream-sync** (agent) — sync upstream lensing framework changes into
  this instance, stepwise with approval gates: 3-way classification against
  the `.lensing-upstream.json` provenance manifest (diff-driven when
  absent), gated batches (build/tooling → rust → predictors → ui →
  agents-src + re-render → docs) as one git commit each; never touches
  domain.toml / models.toml / data/ / registry.toml and never restarts the
  server.
