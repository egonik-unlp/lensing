---
name: model-definitions
description: Define, rename, clone, tag and launch model definitions (named predictor + hyperparameter presets stored in models.toml), rename promoted models, and run experiments (hyperparameter/architecture scans launched as batched runs, written up as a report in experiments/), via the lensing-server API. Use when the user wants to define a model, save a hyperparameter configuration, clone or rename a model/definition, tag datasets on one, launch a run from a saved configuration, run an experiment or scan, or export/share model definitions.
user-invocable: true
argument-hint: "[define|rename|clone|tag|launch|experiment|export] [name]"
allowed-tools:
  - Read
  - Glob
  - Write
  - Bash(curl *)
  - Bash(git *)
  - Bash(sleep *)
---
<!-- GENERATED from agents-src/skills/model-definitions/SKILL.md by agents-src/render.py — edit the template (and domain.toml), not this file; then run `zig build render-agents`. -->

Manage **model definitions**: named, reusable configurations (predictor + concrete
hyperparameters + dataset tags) for this repo's target-prediction experiments.

## Concepts — keep these straight

| Thing | What it is | Lives in | Namespace |
|---|---|---|---|
| **Predictor** | training code + param *schema* | `registry.toml` (hand-edited) | — |
| **Definition** | named preset: predictor + concrete hyperparam *values* + dataset tags | `models.toml` (repo root, server-managed) | definitions |
| **Promoted model** | frozen weights snapshot of a succeeded/stopped run (or any run with a saved checkpoint) | `data/models/<name>/` (gitignored) | models |
| **Best-models group** | server-maintained top-12 by MAE over runs + promoted models; recomputed automatically on run completion and promotion | `GET /api/best-models` (mirrored in `data/best-models.json` + Postgres) | — |

Definitions and promoted models have **separate namespaces**; renaming one never
touches the other. Names must match `^[a-z0-9][a-z0-9-]{0,63}$` — validate before
calling the API.

`models.toml` is git-versionable: **exporting/sharing definitions = committing the
file**. Definitions store hyperparams already merged over the predictor's schema
defaults, so they stay launch-ready.

## Ground rules

- The server must be running (default `http://localhost:8080`; start with
  `zig build serve` if needed). Talk to the API; **do not hand-edit `models.toml`
  while the server runs** — it holds the registry in memory and rewrites the file
  on every mutation, so manual edits get clobbered.
- Never write under `data/` by hand.
- Pretty-print JSON responses for the user (pipe through `python3 -m json.tool`
  or summarize the relevant fields).
- For the predictor contract / dataset format, read README.md — don't guess.

## Workflows

### Define a new model

1. `curl -s localhost:8080/api/predictors` — list predictors and their param
   schemas (types, defaults, min/max). Show the user the schema for their chosen
   predictor and ask which values to change from the defaults (or take them from
   their request).
2. Create — omitted hyperparams keep their defaults:
   ```sh
   curl -s localhost:8080/api/definitions -H content-type:application/json \
     -d '{"name":"<name>","predictor":"<predictor>","hyperparams":{...},"notes":"..."}'
   ```
   `409` = name taken; `400` = unknown predictor/hyperparameter key.

### Clone params into a new definition

- From a definition: `curl -s localhost:8080/api/definitions/<src>/clone -H content-type:application/json -d '{"new_name":"<new>"}'`
  (copies predictor + params + notes; dataset tags are NOT copied — they record usage).
- From a run: `GET /api/runs/<run_id>` → take `.predictor` + `.hyperparams`, then `POST /api/definitions`.
- From a promoted model: `GET /api/models/<name>` → take `.record.predictor` + `.hyperparams`, then `POST /api/definitions`.
- To tweak after cloning: `curl -s -X PATCH localhost:8080/api/definitions/<name> -H content-type:application/json -d '{"hyperparams":{"lr":0.0005}}'` (partial; other keys keep their values).

### Rename

- Definition: `curl -s localhost:8080/api/definitions/<old>/rename -H content-type:application/json -d '{"new_name":"<new>"}'`
- Promoted model: `curl -s localhost:8080/api/models/<old>/rename -H content-type:application/json -d '{"new_name":"<new>"}'`
  (moves `data/models/<old>` → `<new>`; ask which kind the user means if ambiguous — check both `GET /api/definitions` and `GET /api/models`).

### Tag datasets

- Datasets are tagged **automatically** when a run is launched from a definition.
- Manually: read the current tags (`GET /api/definitions/<name>`), then PATCH the
  full list: `-d '{"dataset_tags":["ds-...","ds-..."]}'`. Valid dataset ids come
  from `GET /api/datasets`.

### Launch a run from a definition

```sh
curl -s localhost:8080/api/runs -H content-type:application/json \
  -d '{"definition":"<name>","dataset_id":"<ds-id>"}'
```
Any existing dataset works (`GET /api/datasets` to list). Optional `hyperparams`
overlay the definition's for a one-off tweak (the definition itself is not
modified). The run's meta records `from_definition`; watch progress at
`GET /api/runs/<run_id>/events` (SSE) or in the UI at `/runs/<run_id>`.

### Promote a run / the best-models group

```sh
curl -s localhost:8080/api/models -H content-type:application/json \
  -d '{"name":"<name>","run_id":"<run_id>","notes":"..."}'
```

Promotion (and every run completion) automatically triggers the server's
best-models recompute — the top-12-by-MAE
group at `GET localhost:8080/api/best-models` stays current on its own, and
top runs nobody promoted get auto-promoted as `best-<predictor>-<run-slug>`.
For judgment calls on the group (pin a confirmed champion, exclude a
suspicious fluke, family diversity), hand off to the **best-model-selector**
agent rather than curating by hand; to remove a member, exclude it via
`PUT /api/best-models` — deleting the model alone just gets its run
re-promoted.

### Stop a running run

```sh
curl -s localhost:8080/api/runs/<run_id>/stop -H content-type:application/json -d '{"force":false}'
```

Graceful (`force:false`): the predictor finishes the current epoch, evaluates
and saves the model with the params as they are; the run becomes `stopped`
and is promotable like a succeeded one. `{"force":true}` kills the process —
then the run is promotable only if a periodic checkpoint landed (the
`checkpoint_every` hyperparam). Only predictors with `supports_stop` in
`GET /api/predictors` honor the graceful path; for others use force.

### Run an experiment

A systematic scan over one or two hyperparameter/architecture axes, written up
as a dated report in `experiments/`. Datasets must already exist — if the axis
is dataset-level (PCA dims, quality filters, features), build them first with
the **dataset-design** skill.

1. **Design.** Read `experiments/PROJECT-FACTS.md` for the current
   best-on-record, noise bands and known pitfalls, then the underlying
   `experiments/*.md` reports for detail (the reports are primary; newest
   wins). Agree the scan with the user before launching: the fixed baseline
   (dataset id, reference definition or champion run, seed), the axis/axes
   to probe, and the concrete configs (~5–10).
2. **Launch.** One `POST /api/runs` per config — overlay on a baseline
   definition (`{"definition":"<base>","dataset_id":"<ds>","hyperparams":{<axis deltas>}}`,
   the definition is not modified) or bare `{"predictor":...,"hyperparams":{...}}`.
   Do **not** create a definition per config; only the winner gets saved.
   Launch the whole scan at once and record each run id ↔ config: the server
   trains up to `--max-runs` concurrently (default 2) and queues the rest
   (`queued: waiting for a free training slot` in the run's events).
3. **Collect.** Poll `GET /api/runs/<run_id>` (e.g. `sleep 30` between rounds)
   until no run is `running`. On success `.metrics` holds the task's metric
   keys + `n_test` (regression: `{mae, rmse, r2, mape, medape}`, mape/medape as
   fractions; classification: `{accuracy, logloss, auc, brier|macro_f1}`) —
   render fraction metrics as %. Treat failures as data points: capture
   `.stderr_tail` and any
   blowup pattern (they often become the most interesting finding).
4. **Present & confirm.** Show the results table sorted by MAE plus draft
   findings. Do not persist anything until the user approves.
5. **Persist.** Save the winner as a definition (`POST /api/definitions`, or
   clone the baseline and PATCH) named per the convention
   `<predictor>-<slug>` (e.g. ``); tag the
   dataset (automatic if a confirming run is launched from the definition,
   otherwise PATCH `dataset_tags`). Then write the report — and reconcile
   `experiments/PROJECT-FACTS.md` (leaderboard row, new pitfalls, dataset
   lineage, the "Last updated" header) as part of it.

**Report format** — `experiments/<YYYY-MM-DD>-<slug>.md`, matching the
existing files:

```
# <predictor>: <axis> scan (<date>)

Goal: … (fixed baseline: dataset id, held hyperparams, reference champion
with its metrics and run id)

Outcome in one line: **<key finding> — <MAE / RMSE / R²> —
definition `<name>`**

## Results (sorted by MAE; all on <ds-id>)
| config | MAE | RMSE | R² | run |
(bold the winner row and the best cell per metric; include failed runs)

## Findings
(interpretation: why the winner wins, trade-offs, failure modes, saturation)

## Best on record after this work
| model | MAE | RMSE | R² | run |   ← winner vs. previous champions

## Follow-ups
- next experiments worth running
```

Conventions: MAE in raw target units with thousands
separators; percentage metrics rendered as %; a run id in every table row;
name the dataset id; reference prior experiments by filename when building
on them.


### Export / share

`models.toml` at the repo root is the registry. To share or back up:
```sh
git add models.toml && git commit -m "model definitions: <what changed>"
```
To import someone else's definitions, merge their `[[definitions]]` entries into
`models.toml` **while the server is stopped**, then restart it (it validates on
load). Show the file with `Read` when the user wants to inspect what's stored.

### Delete

`curl -s -X DELETE localhost:8080/api/definitions/<name>` — past runs launched
from it keep their `from_definition` label; nothing else is affected. Confirm
with the user before deleting.
