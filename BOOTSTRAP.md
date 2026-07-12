# Bootstrapping this template into a new prediction problem

This repository is a template: a complete lab for predicting one target
variable over a corpus of embedded documents (Qdrant collection: vectors +
JSON metadata payloads). It ships BLANK — `domain.toml` is a neutral
placeholder and `CLAUDE.md` is a bootstrap-pending stub; everything
domain-specific is parameterized and comes only from the user during
bootstrap — never assume any field is the target. A complete worked
example of `domain.toml` (the framework's original problem: Argentine
real-estate sale prices, exercising every lever including `[currency]`,
`[coordinates]` and reconciled fields) lives at
`crates/lensing-core/src/example-domain.toml` — read it for reference,
never as a default.

Day-1 checklist for a new project (e.g. used-car prices, salaries):

1. **Write `domain.toml`** — the single source of domain truth:
   - `[project]`: name, title, entity/target nouns (drive UI copy + agents).
   - `[corpus]`: Qdrant collection, embedding dim, `metadata_root`,
     `content_field`, and the corpus point filter.
   - `[target]`: target field, transform (`log1p`|`none`), display format.
   - `[[fields]]`: one entry per payload field — role (`target` |
     `categorical` | `numeric` | `coordinates` | `filter_only` | `timestamp`
     | `display`), encode/indicator/vocab options. **Order is load-bearing**
     (feature column order). Field names are payload keys AND the
     `items.json` keys payload-based predictors read.
   - `[coordinates]` bounds — only with a coordinates-role field.
   - `[currency]` — delete the whole section for single-currency domains.
   - `[quality]`: bind the stable rule keys to your fields + display labels.
   - `[metrics]`: primary metric, column order, % metrics, value unit.
   - `[agents]`: API URL, report dir, facts file, naming convention,
     `ingestion = true|false`.

2. **`rm CLAUDE.md && zig build render-agents`** — regenerates the
   skills/agents under `.claude/`, `.agents/`, `.gemini/` AND the repo-root
   `CLAUDE.md` from `agents-src/` with your domain words. (The template
   ships `CLAUDE.md` as a bootstrap-pending stub; the renderer preserves
   the stub until you delete it — hence the `rm`.) Skim the outputs;
   `zig build check` fails on drift or leftover `{{tokens}}`.

3. **Fill in the domain notes** — the bottom section of
   `agents-src/agents/listing-generator.md` ("Domain notes — FILLED IN BY
   BOOTSTRAP") is a placeholder for your domain's source-site scraping
   lore. Write it, or set `[agents] ingestion = false` to drop the agent.
   Also skim PRODUCT.md / DESIGN.md.

3b. **Brand the instance** — set `[branding]` in `domain.toml` (`mark_seed`,
   usually your project name, and optionally `mark_hue_shift`) and run
   `python3 branding/make_mark.py --install`. Your instance keeps the lensing
   grammar (dark sphere, velocity swarm, Sora wordmark, `lensing · <title>`
   topbar) with its own arc arrangement and ramp hue. See DESIGN.md §3b.

4. **Reset the empirical record** — this repo ships the original project's
   campaign history. For a fresh project: archive or delete
   `experiments/*.md`, and seed `experiments/PROJECT-FACTS.md` with empty
   leaderboard/pitfalls/lineage tables and the line "No facts yet — the
   first campaign seeds this." Reset `docs/experiments.tex` title/abstract
   and trim `docs/figures/make_figures.py` to the style header (the
   report-curator agent grows it back from your reports).

5. **Point at your corpus** — an existing Qdrant collection (embeddings +
   payloads matching your `[[fields]]`), or start a local one:
   `docker compose --profile qdrant up -d`. Validate the shape:
   `POST /api/collections/validate`.

6. **Start the stack** — `zig build serve` (starts Postgres via compose,
   runs schema migrations, backfills any existing file state, serves the
   UI). `models.toml` ships with the original definitions — delete its
   entries for a fresh start (while the server is stopped).

   *Containerized instead?* `cp docker/.env.example .env && python3
   scripts/render-deploy-env.py --write` (seeds the per-instance container
   names/ports from `domain.toml`), then `zig build docker-up` (or `docker
   compose --profile hub up -d --build --wait`). For splitting hub / workers /
   inference across machines, see **docs/DEPLOY.md §Deploy with Docker**.

7. **First dataset** — `/dataset-design` (preflight the quality filters,
   then build).

8. **Baselines** — `/model-definitions define`: start with
   `baseline-median` (the floor) plus one real family (xgboost is the
   low-drama default).

9. **First campaign** — `/model-definitions experiment`, or the
   experiment-designer agent (design) handing off to the experiment-runner
   agent (execution). The report the runner writes seeds the first
   PROJECT-FACTS leaderboard row.

10. **First report sync** — `/report-curator sync` folds the report into
    the PDF.

What you should NOT need to touch: the Rust crates (lensing-core/lensing-db/
lensing-pipeline/lensing-server), the predictor plugins + `registry.toml` (generic
regressors over the artifact format), the UI (renders from `GET
/api/domain`), `docker-compose.yml`, the `Dockerfile` / `docker/` deploy files,
`build.zig`. If your domain needs a
quality rule the built-in set lacks, add a rule evaluator in
`crates/lensing-pipeline/src/quality.rs` (keys are stable identifiers; bind +
label via domain.toml).
