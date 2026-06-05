<!-- GENERATED from agents-src/root/CLAUDE.md by agents-src/render.py — edit the template (and domain.toml), not this file; then run `zig build render-agents`. -->
# Price Guesser Models

A price-prediction lab: a Qdrant corpus of embedded
listings feeds dataset builds (PCA + metadata features),
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
  pitfalls, dataset lineage) lives in `experiments/PROJECT-FACTS.md` — agents read it
  first and reconcile it after every campaign. The `experiments/*.md`
  campaign reports are primary; the newest report wins.

## State & persistence

- **Postgres** (docker compose; `zig build db-up`) is the metadata store:
  model definitions (authoritative; `models.toml` is the git-diffable
  export), runs + metrics, promoted-model records, dataset index, run
  events. `zig build migrate-data` backfills it from the files and prints a
  consistency report — it never modifies the files.
- **`data/`** holds the binary artifacts (datasets, run dirs, model
  snapshots). NEVER write under `data/` by hand.
- The Qdrant corpus (`properties-tagged` at `http://localhost:6333`) is external;
  manual entries live in `manual-listings`.

## Non-negotiables

- The server (`http://localhost:8080`) must already be running for skills/agents;
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
  inspect datasets.
- **model-definitions** — define/clone/tag/launch model definitions; the
  experiment workflow (design → launch → collect → report → reconcile
  `experiments/PROJECT-FACTS.md`).
- **experiment-designer** (agent) — full experiment lifecycle behind a
  two-phase approval: it proposes, you approve, it executes.
- **report-curator** — maintains docs/experiments.tex (+ figures) from the
  campaign reports; destructive edits stop for approval.
- **listing-generator** (agent) — URL → manual listing → predictions
  from the best-models group.
