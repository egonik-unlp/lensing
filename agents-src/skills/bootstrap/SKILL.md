---
name: bootstrap
description: Initialize this template around a NEW dataset/prediction problem: populate a corpus from a raw source (database/file/HTTP API) via the lensing-intake pipeline if one doesn't exist yet, probe the Qdrant corpus and infer its payload schema, interview the user into a domain.toml (field roles, quality levers, metrics, nouns), render the agent layer, reset the empirical record, decide how manual entries are fetched/presented (ingestion), build the first dataset with tuned quality levers, and set up the starter model definitions for a first evaluation. Use when the user wants to bootstrap, initialize, or adapt this framework to their own data/target variable, or asks "how do I point this at my dataset".
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
  - Bash(cargo run -p lensing-intake *)
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

## Phase −2 — Instantiation gate (do this FIRST, every time)

Two rules are enforced by `tools/lensing_guard.py` and the hooks in
`.claude/settings.json`, not merely documented. Start here:

```sh
python3 tools/lensing_guard.py check      # or: zig build bootstrap-check
```

**Rule: the instance owns its folder.** A new project is a full copy of the
framework living in a directory of its own. The verdict tells you where you
are:

| verdict | meaning | what to do |
|---|---|---|
| `FRAMEWORK` | you are in the lensing framework checkout (`.lensing-mother` present) | **STOP — do not bootstrap here.** Instantiate properly: `zig build package`, then `tar xzf dist/lensing.tar.gz -C <projects-dir>`, rename the folder, and open a session there. Tell the user this and hand them the commands; `bootstrap start` refuses in place. |
| `INVALID` | a partial or improvised tree — no provenance manifest, missing framework files, or nested inside the framework checkout | **STOP.** Report the failing invariants verbatim. A tree that isn't a complete packaged copy cannot be bootstrapped into a sound instance; re-instantiate from a package. |
| `INCOMPLETE` | a real instance, not yet bootstrapped | proceed to Phase −1 |
| `READY` | already bootstrapped | nothing to do — say so, and route the user to the work they actually want |

Never "fix" an INVALID tree by copying files in from another lensing project.
This instance is fully independent: it must never read from or write to
another project's corpus, datasets, server or experiment record.

**Rule: bootstrap always finishes.** Once the user has agreed to bootstrap,
mark it in flight:

```sh
python3 tools/lensing_guard.py bootstrap start
```

From that point the Stop hook refuses to end the session while the invariants
are unmet — you carry the run to a real `domain.toml` and a rendered
`CLAUDE.md`, resuming at whatever phase the repo state implies. If the user
decides to stop partway, that is their call and it is recorded, not silently
dropped:

```sh
python3 tools/lensing_guard.py bootstrap abort --reason "<what the user said>"
```

While the instance is unbootstrapped the guard also denies writes outside
bootstrap's own files (`domain.toml`, `CLAUDE.md`, `PRODUCT.md`,
`pipeline.toml`, `models.toml`, `experiments/`, `docs/`, `branding/`,
`agents-src/`, `.env`) and every API mutation. If you are denied, that is the
signal you have wandered off bootstrap — come back to it rather than working
around the gate.

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

## Phase 0a — Infrastructure: endpoints + where the data lives

Two things must be settled before any probing, because everything below
depends on them and `domain.toml` (Phase 1) has not been written yet.

**1. Where does their data live right now?** Ask before touching Qdrant:

- **Already a vector collection** — an existing Qdrant collection with
  embeddings + JSON payloads. → Phase 0b, existing-collection path.
- **A raw source, not embedded yet** — a database, a flat file, an HTTP
  API. → Phase 0b, intake path. This is the common case: real data starts
  tabular or behind an API, not as vectors.

**2. This instance's own endpoints.** Propose the `[deploy]` defaults and
have the user confirm or change them; two instances must never share a
port (D1). Qdrant publishes **two** host ports and they are not
interchangeable:

| Endpoint | Default | Used by |
|---|---|---|
| Qdrant **REST** | `http://localhost:6333` | your `curl` probing below, the server, `lensing-pipeline`; becomes `[corpus] qdrant_url` |
| Qdrant **gRPC** | `http://localhost:6334` | `lensing-intake`'s `[[sink]]` only |
| Postgres | `postgres://…@localhost:5433/…` | `lensing-intake`'s Postgres `[[sink]]` (intake path only) |

Ask for the Postgres endpoint only on the intake path. Record all confirmed
values — Phase 1 writes them into `domain.toml` (`[corpus] qdrant_url` and
`[deploy]`) rather than asking again or re-defaulting.

> **The two Qdrant ports are the single most common way this phase goes
> wrong.** `lensing-intake` reaches Qdrant through a gRPC client, while
> everything else in the framework speaks REST. A REST URL in the sink is
> accepted by every check and only fails once the pipeline is writing — after
> it has fetched the source and computed (and billed) every embedding —
> showing up as the confusing `Written: [postgres]; not written: qdrant`.
> Never reuse the REST URL for the sink.

## Phase 0b — Probe the corpus

**Intake path only:** populate the corpus first (see "Populate the corpus
with `lensing-intake`" below), then continue here as if the collection had
existed all along — the rest of this phase and all of Phase 1 are identical
for both paths.

Qdrant first (the REST endpoint confirmed in Phase 0a; default
`{{qdrant_url}}`):

```sh
curl -s <qdrant>/collections                                   # what exists
curl -s -X POST <qdrant>/collections/<name>/points/scroll \
  -H content-type:application/json \
  -d '{"limit": 16, "with_payload": true, "with_vector": false}'
```

From the sample, build and SHOW a schema inference table: payload key →
observed JSON type → coverage in sample → proposed role (`categorical` |
`numeric` | `coordinates` | `filter_only` | `timestamp` | `display`) →
proposed options (encode log1p for heavy-tailed sizes,
indicator for sparse numerics, vocab top-N for high-cardinality
categoricals, critical for must-have fields). Do NOT propose a `target`
role for any field — the prediction target is the user's call alone, asked
in Phase 1. The shipped `domain.toml` is a neutral placeholder and the
worked example at `crates/lensing-core/src/example-domain.toml` is
reference material, not a preference: never carry its target (or any
plausible-looking field) over as an assumed target. If several fields could
plausibly be targets, list them neutrally as candidates without ranking.
Also: where the metadata
nests (`metadata_root`), which key holds the embedded text
(`content_field`), the embedding dim (scroll once with
`"with_vector": true, "limit": 1`), and a candidate corpus filter. If they
have no corpus yet, take the intake path below.

### Populate the corpus with `lensing-intake` (intake path)

The framework ships `lensing-intake`, a config-driven pipeline that reads a
source, embeds each record, and dual-writes to this instance's Postgres
(authoritative) and Qdrant (derived index). Drive it — do not ask the user to
hand-build this.

**Interview for the config** (`crates/lensing-intake/src/config.rs` is the
authoritative field list). Propose defaults for everything and have the user
confirm; the embedding choice is target-affecting, so never pick it silently:

1. **Source `kind`** and its fields:
   - `postgres` / `sql-mysql` → `url`, `query`, `identifier`
   - `sql-sqlite` → `path`, `query`, `identifier`
   - `file` → `path`, `identifier`, optional `format` (`csv`|`json`|`jsonl`)
   - `http` → `url`, `identifier`, optional `pointer`, `page_param`
2. **`[embedding]`** — `provider` (`ollama`|`openai`), `model`, `dims`,
   `distance` (`cosine`|`euclid`|`dot`|`manhattan`). Propose a default, show
   it, and ask for confirmation before writing.
3. Leave **`batch_size` at 0** (the default). A non-zero value splits the
   source into `<identifier>_0`, `<identifier>_1`, … — several collections
   instead of one, and nothing for Phase 1 to probe.

**The embedding model is a three-way contract.** Whatever you choose here
must match `LENSING_EMBEDDING_MODEL` (and its dimensionality) wherever the
server runs, or the server will reject the corpus you just built. Say this to
the user and carry the choice into Phase 1's `[corpus] embedding_dim`.

**Write `pipeline.toml`** at the repo root, copying
`crates/lensing-intake/pipeline.example.toml` and filling in the answers:

- `[[sink]]` **Postgres first, then Qdrant** — sinks are written in listed
  order and the authoritative store must lead, which is what makes a partial
  failure legible.
- The Qdrant sink's `url` is the **gRPC** endpoint from Phase 0a. The
  Postgres sink's `url` is this instance's Postgres.
- `pipeline.toml` holds plaintext credentials — it is gitignored; never
  commit it, and never echo the full file back with secrets in it.

**Check the gRPC endpoint before running anything.** Cheap now, expensive
later:

```sh
curl -s -o /dev/null -w '%{http_code}\n' <qdrant-grpc-url>   # expect 000/501, i.e. listening but not HTTP
```

If nothing is listening, stop and fix it here — do not run the pipeline. If
the user gave the REST port by mistake (it answers a normal HTTP `200` on
`/collections`), say so plainly and ask for the gRPC one.

**Predict the collection name.** `lensing-intake` derives it rather than
letting you name it:

```
<provider>_<model>_<Distance>_<identifier>
```

`<provider>` is `ollama`/`openai`, `<model>` has every `:` replaced by `_`,
`<Distance>` is capitalized (`Cosine`/`Euclid`/`Dot`/`Manhattan`), and
`<identifier>` is the source's. E.g. `openai_text-embedding-3-small_Cosine_listings`.
Compute it from the answers above and tell the user what to expect.

**Run it**, then verify:

```sh
cargo run -p lensing-intake -- --config pipeline.toml
curl -s <qdrant-rest-url>/collections        # the predicted name should be here
```

- **Success** (all sinks written) → continue with the probing above against
  the new collection. Phase 1 does not care which path created it.
- **Partial or total failure** (e.g. `Written: [postgres]; not written:
  qdrant`) → **stop.** Show the error verbatim, name the likely cause (a
  REST URL in the sink is the usual one), and offer to re-run once fixed —
  re-running is safe, both sinks upsert idempotently. Never proceed to schema
  inference against an incomplete corpus.

### If `lensing-intake` doesn't fit

For a source or an embedding process its connectors don't cover, the
self-managed route stands: explain the requirement (vectors + JSON payloads),
offer `docker compose --profile qdrant up -d`, and let them populate the
collection with their own pipeline — this skill does not embed documents
itself. Come back to the probing above once a collection exists.

## Phase 1 — Interview → domain.toml

Ask only what you cannot infer; propose defaults for everything else —
with ONE exception: the **target is always asked, never inferred**. It is
the first question of the interview, asked open-endedly ("which field
should the models predict?") with no pre-selected suggestion:

1. **Target**: field (the user's answer — no default), then **transform and
   display format read off the target's *own* distribution from Phase 0, not
   off a default.** There is no default transform: inspect the sampled values.
   `log1p` fits a strictly-positive, heavy-tailed target (and only then) — it
   is one option, never the starting assumption; `none` fits signed, bounded,
   rating-like, count, or already-symmetric targets (anything that can be ≤ 0
   breaks `log1p`). Show the user the observed range/skew and your reasoning,
   and let them confirm. Display format follows the same evidence:
   `style = "number"`, `symbol = ""` is the baseline for any quantity; reach
   for `style = "money"` + a currency symbol **only** when the target is
   literally a monetary amount. Never assume the target is positive, money, or
   heavy-tailed because the framework's worked example happens to be a price.
2. **Nouns + title**: entity noun ("listing"/"vehicle"/"posting"), target
   noun, project name/title.
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
8. **Endpoints — record, don't re-ask.** `[corpus] qdrant_url` takes the
   **REST** endpoint confirmed in Phase 0a, and `[deploy]` takes the ports
   behind it: `qdrant_port` (REST), `qdrant_grpc_port` (gRPC) and `db_port`.
   These were already settled; carrying them over is what keeps the instance
   port-partitioned (D1) and keeps `pipeline.toml` and `domain.toml`
   pointing at the same Qdrant. `scripts/render-deploy-env.py --write`
   derives the container env from `[deploy]`, so a value dropped here
   silently reverts that instance to the shared defaults.
   If the corpus came from the intake path, `[corpus] embedding_dim` must
   match the `[embedding] dims` used to build it.

Then: write `domain.toml`, validate via `cargo test -p lensing-core` (the
embedded-domain tests re-read it) — fix anything it rejects, and run
`rm CLAUDE.md && zig build render-agents` so the skills/agents AND the
repo-root `CLAUDE.md` speak the new domain. (The template ships
`CLAUDE.md` as a bootstrap-pending stub and the renderer preserves the
stub until it is deleted — hence the `rm`. Never leave the stub or
another project's CLAUDE.md in place.) `zig build check` must pass
before moving on.

This is where the instance stops being unbootstrapped. Confirm it
mechanically before continuing — `python3 tools/lensing_guard.py check` must
now read `READY`; if it does not, the reason it prints is a real gap in the
domain config, not noise to route around.

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
| **Scrape from source sites** | Fill in the "Domain notes — FILLED IN BY BOOTSTRAP" section of `agents-src/agents/listing-generator.md` with THEIR source-site lore (portals, embedded-JSON tricks, field-name translations, regional quirks); keep `[agents] ingestion = true`; re-render. |
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

Propose a small starter slate **for the user to confirm** — not a fixed
recipe. The shape of the target and the dataset (decided in Phases 1/4)
drives the picks; explain each in those terms, never as "what the price
project used":

1. `baseline-median` — always. The floor every model must beat ("group
   median"); it is target-agnostic and frames every later result.
2. A robust general-purpose regressor — `xgboost` is the usual first pick:
   it makes no distributional assumption about the target, gives bounded
   predictions, and handles one-hots / indicators / missing values natively.
3. Optionally a third seat chosen by data shape, e.g. a neural predictor
   (`burn-mlp`) when rows are plentiful (≳ 10k), or `svm` / `kernel-ridge`
   on smaller tables (≲ 5k). **Only if the user chose `transform = log1p`**
   does the log-space-MLP guardrail apply — set `clamp_output: true` so an
   un-inverted prediction can't blow up; with `transform = none` it is not
   needed.

These are starting baselines for a first read, not commitments — confirm the
slate with the user before creating definitions (via the
**model-definitions** skill). Launch one run each on the first dataset, watch
them complete, then seed `{{facts_file}}`'s leaderboard with the results.

These baseline launches are gated like every other run — take one campaign
ticket for the slate and close it once the leaderboard is seeded:

```sh
python3 tools/lensing_guard.py ticket issue --kind campaign \
  --design bootstrap-baselines --report {{report_dir}}/<date>-baselines.md \
  --allowance <N runs + the first dataset build>
# ... build the dataset, launch the slate, write the short baseline report ...
python3 tools/lensing_guard.py ticket close
```

Then hand the user off to the **experiment-designer** agent to design the
first real campaign (the **experiment-runner** agent executes the approved
design). From here on, every scan goes through that pair — bootstrap does not
run experiments of its own.

## Ground rules

- **Phase −2 first, always.** Never bootstrap in the framework checkout or in
  a partial tree; never patch a broken instance with another project's files.
  `python3 tools/lensing_guard.py check` must read `INCOMPLETE` before you
  start and `READY` when you finish — if it still reports failing invariants,
  bootstrap is not done, whatever the conversation feels like.
- **Bootstrap always finishes.** `bootstrap start` at the top, and either the
  invariants pass or `bootstrap abort --reason "<why>"` records the user's
  decision to stop. Half-bootstrapped instances are the failure this skill
  exists to prevent — do not leave one behind, and do not start improvising
  domain facts to paper over a phase you could not complete. If you are
  blocked, say what is missing and ask.
- **Do not hand off to the experiment agents until READY.** Phase 5 ends by
  routing to experiment-designer → experiment-runner; the guard denies run
  launches before that anyway.
- Every file write and every destructive step is announced and confirmed
  first. Phase 2 doubly so.
- **No default target, no default transform, no assumed model recipe.** The
  shipped domain is a neutral placeholder and the worked example (prices) is
  reference material, not a bias. Never assume, pre-select, or rank a target
  field — it comes only from the user's explicit answer in Phase 1. The
  transform (`log1p` vs `none`), the display format (`number` vs `money`),
  and the starter-model slate all follow from the *probed target and dataset*
  and are confirmed with the user — never carried over from the price origin.
  In particular: do not reach for `log1p`, a currency symbol, or a log-space
  MLP guardrail unless the target's own distribution calls for it.
- Never write under `data/` by hand; never hand-edit the rendered
  `.claude/.agents/.gemini` outputs (edit `agents-src/` + `domain.toml`,
  then `zig build render-agents`).
- The Rust crates, predictors, `registry.toml` and the UI should NOT need
  changes — if the domain seems to require it (a missing quality-rule kind,
  a new field role), say so explicitly rather than improvising code.
