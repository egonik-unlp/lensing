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

## Day 0 — instantiate into a folder of its own

A new project is a **full copy of the framework in its own directory**, never
the framework checkout itself and never a subdirectory of it. From the lensing
framework repo:

```sh
zig build package                              # -> dist/lensing.tar.gz
tar xzf dist/lensing.tar.gz -C ~/projects
mv ~/projects/lensing ~/projects/<your-project>
cd ~/projects/<your-project> && zig build bootstrap-check
```

The package carries `.lensing-upstream.json` (a provenance manifest listing
every framework file) and omits `.lensing-mother`; together those are how the
instance knows what it is. `zig build bootstrap-check` — or
`python3 tools/lensing_guard.py check` — prints the verdict:

| verdict | meaning |
|---|---|
| `FRAMEWORK` | you are still in the framework checkout; bootstrap refuses to run here |
| `INVALID` | not a complete packaged copy (no manifest, missing framework files, or nested inside the framework) — re-instantiate rather than patching |
| `INCOMPLETE` | a real instance, ready to bootstrap |
| `READY` | bootstrap is finished |

`git clone` of the framework is not an instantiation path: the clone carries
`.lensing-mother` and no manifest, and the guard treats it as the framework.

The same guard backs three rules that hooks in `.claude/settings.json`
enforce for agent sessions, so they hold whether or not anyone reads this
file: work outside bootstrap is blocked until the instance is configured, a
started bootstrap has to finish (or be explicitly aborted with
`python3 tools/lensing_guard.py bootstrap abort --reason "…"`), and run
launches / dataset builds need a run ticket so experiments stay attached to a
campaign report. See §Guard rails below.

## Day 1 — checklist for a new project (e.g. used-car prices, salaries)

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

5. **Point at your corpus** — two ways in, depending on what you have:

   *Already a Qdrant collection* (embeddings + payloads matching your
   `[[fields]]`)? Point at it and validate the shape:
   `POST /api/collections/validate`.

   *Data still at the source* — a database, a flat file, an HTTP API? Build
   the collection with the intake pipeline: `cp
   crates/lensing-intake/pipeline.example.toml pipeline.toml`, fill in the
   `[source]`, `[embedding]` and `[[sink]]` blocks, then `cargo run -p
   lensing-intake -- --config pipeline.toml`. It embeds each record and
   dual-writes to this instance's Postgres (authoritative) and Qdrant
   (derived index). `/bootstrap` runs this interview for you.

   Three things to get right, all of them easy to get wrong:
   - The Qdrant `[[sink]]` takes the **gRPC** port (6334), not the REST port
     (6333) the server and `[corpus] qdrant_url` use. A REST URL here fails
     only once the pipeline is writing — after you have paid for every
     embedding.
   - List **Postgres before Qdrant**: sinks are written in order and the
     authoritative store leads.
   - You don't choose the collection name; it is derived as
     `<provider>_<model>_<Distance>_<identifier>` (e.g.
     `openai_text-embedding-3-small_Cosine_listings`).

   `pipeline.toml` holds plaintext credentials and is gitignored. Re-running
   is safe — both sinks upsert idempotently.

   *Neither fits?* Start an empty local Qdrant with `docker compose --profile
   qdrant up -d` and populate it with your own embedding pipeline.

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

9. **First campaign** — the **experiment-designer** agent designs it with
   you, the **experiment-runner** agent executes the approved design. That
   pair, not an ad-hoc string of runs, is the path: the runner is what writes
   the report and reconciles PROJECT-FACTS.md, and the guard denies
   `POST /api/runs` without a run ticket precisely so a scan cannot happen
   outside it. (`/model-definitions experiment` documents the mechanics the
   runner follows.)

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


## Guard rails

`tools/lensing_guard.py` holds the instantiation invariants; the hooks in
`.claude/settings.json` enforce them in agent sessions. In the framework
checkout every hook is a no-op — this is instance machinery.

| command | what it does |
|---|---|
| `zig build bootstrap-check` | print the verdict (also `python3 tools/lensing_guard.py check`; exit 0 ready / 1 incomplete / 2 invalid) |
| `… bootstrap start` | mark a bootstrap in flight — the Stop hook then refuses to end a session while the invariants are unmet |
| `… bootstrap abort --reason "…"` | record the user's decision to stop, releasing that gate |
| `… ticket issue --kind campaign --design <slug> --report experiments/<date>-<slug>.md --allowance <N>` | authorize N gated API calls for one campaign |
| `… ticket issue --kind oneshot --reason "…"` | authorize exactly one ad-hoc run the user asked for by name |
| `… ticket status` / `… ticket close` | inspect / close; `close` refuses until the report exists and PROJECT-FACTS.md has been touched |

What is gated, and nothing else: `POST /api/runs` and `POST /api/datasets`.
Polling, stopping runs, promotions, preflight and analysis are untouched.
Before bootstrap completes, writes outside bootstrap's own files are denied
too. Every ticket action is appended to `.lensing/tickets.log`.
