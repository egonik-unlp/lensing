---
name: information-capture
description: Compare how much target-relevant information competing MLP models capture inside their own hidden activations, using a sparse autoencoder (SAE) trained per hidden layer via the pg-server interpretability API. Use when the user wants to compare MLP models (or blend legs) on information capture — capacity/utilization, interpretable concepts, per-segment representation, concept-vs-decodability by depth, and the target-relevant embedding signal a model drops — beyond leaderboard MAE. Runs `POST /api/interp/model-sae` and distills a side-by-side comparison. For a full campaign write-up, delegate to the information-capture-analyst agent.
user-invocable: true
argument-hint: "[model ...] [compare-embedding]"
allowed-tools:
  - Read
  - Bash(curl *)
  - Bash(sleep *)
  - Bash(python3 *)
---
<!-- GENERATED from agents-src/skills/information-capture/SKILL.md by agents-src/render.py — edit the template (and domain.toml), not this file; then run `zig build render-agents`. -->

Compare **information capture** across MLP models: how much of the target's
structure each trained net actually re-represents inside its own hidden layers,
read with a sparse autoencoder (SAE) — the nonlinear sibling of the layer probe.
This answers a question the leaderboard can't: two MLPs can post the same
MAE while one *captures* far more structure (and drops less of the
embedding's target-relevant signal) than the other.

This is an **agent-layer wrapper** over the existing interpretability API
(`POST /api/interp/model-sae`); it computes nothing itself — it orchestrates the
jobs and distills their JSON into a comparison. The engine, method and prior
results are documented in `docs/interpretability-toolkit-explained.md` (§2) and
`docs/interpretability-queue.md` (P2) — read them before interpreting.

## What "information capture" means here

The toolkit separates three fates of a signal (per
`docs/interpretability-toolkit-explained.md`): a target-relevant signal
can be **absent** from the input, **present but discarded** (compression), or
**present but unused** (modelling). The per-model SAE reads the *model's* side of
that: for each hidden layer it reports

- **capacity** — `utilization` (fraction of atoms that ever fire), `dead_atoms`,
  `rare_atoms`, `l0_mean` (atoms active per item), and `var_explained`
  (how well the sparse code reconstructs the layer). Low utilization / high dead
  count = the layer is using a small slice of its width.
- **n_interpretable_concepts** — atoms whose activation correlates with the
  target above the engine's concept bar; a proxy for how many nameable,
  target-relevant features the layer has formed.
- **segments** — per one-hot category value, whether a dedicated atom *separates*
  that segment (`represented: true/false`). Surfaces rare-segment structure the
  bulk fit buries.
- **concept_vs_decodability** — joins the linear target R² at each depth
  (`linear_r2_target`/`linear_r2_log`) with the interpretable-concept count there,
  so you can see where in the net decodable structure and nameable concepts form.
- **dropped_vs_embedding** (only with `compare_embedding`) — target-relevant
  concepts present in the *embedding* SAE that no atom in this layer tracks: the
  signal the model discarded. Roughly doubles run time (it trains a dataset SAE
  first).

## Ground rules

- The server must be running (default `http://localhost:8080`; `zig build serve`).
  Analysis is **read-only** — it never mutates models, runs or datasets. Never
  restart pg-server (live training runs die on restart); never write under
  `data/`.
- Operates on **promoted models** (`data/models/<name>/`), not runs — promote a
  run first (`model-definitions` skill / `POST /api/models`) if you want to
  analyze it. The server auto-promotes top runs as `best-<predictor>-<slug>`, so
  a campaign's leaders are usually already promotable targets.
- Only predictors that declare `model_sae_args` in `registry.toml` are
  analyzable — the MLP families with a Rust-/Python-reachable activation space.
  `GET /api/interp/models` marks each with `supports_model_sae`; a model without
  it returns `422` and is not an error, just unsupported.
- **Comparisons are only meaningful across models analyzed on the same dataset**
  (same `n_cols`). Pass an explicit `dataset` so every model is read on identical
  rows; otherwise each defaults to its own training dataset and the numbers are
  not comparable.
- Percentage-style reads rendered as %; keep run/dataset ids verbatim.

## Workflow

Analyses are **persisted**: every `model-sae` run is saved to the
`interp_analyses` table (mirrored from `data/interp/<id>/`) and served back by
id, so the first move is always to **reuse, not recompute**. Promoting an MLP
model also **auto-queues** a per-model SAE (embedding-diff read, no atom labels),
so a campaign's leaders usually already have a saved analysis by the time you
look.

### 0. Reuse persisted analyses (don't recompute)

```sh
curl -s 'localhost:8080/api/interp/analyses?tool=model-sae' | python3 -m json.tool
```
Each row carries `id`, `model`, `dataset_id`, `predictor`, `config`, `status`,
`source` (`auto`/`manual`) and `created_at` (no heavy `result`). For every target
model, if a `done` row exists on the shared dataset with an adequate `config`
(e.g. it already has `compare_embedding` when you need the dropped-signal diff),
fetch its stored result instead of launching a job:

```sh
curl -s localhost:8080/api/interp/analyses/<id> | python3 -m json.tool   # `.result` is the full analysis JSON
```
Only launch a new job (step 2) for models with no suitable saved analysis, or
when you need a richer config than the saved one (add `compare_embedding` or
`label_atoms`). Cite the analysis `id` in the report so it can be reopened in the
UI's **Saved analyses** tool.

### 1. Discover analyzable models

```sh
curl -s localhost:8080/api/interp/models | python3 -m json.tool
```
Each entry carries `name`, `predictor`, `dataset_id`, `n_cols`, `hidden`,
`activation`, and `supports_model_sae`. Keep the ones with
`supports_model_sae: true` that share a dataset width; agree the target set and
the shared `dataset` with the user (or, from the analyst agent, take them from
the campaign).

### 2. Launch a per-model SAE (async — only for models with no saved analysis)

```sh
curl -s localhost:8080/api/interp/model-sae -H content-type:application/json -d '{
  "model": "<name>",
  "dataset": "<shared ds-id>",
  "layers": "",
  "n_atoms": 0,
  "compare_embedding": false
}'
```
Body fields: `model` (required), `dataset` (default = model's training set;
**set it explicitly for comparisons**), `layers` (comma-sep 1-based hidden
layers, empty ⇒ all), `n_atoms` (`0` ⇒ 2× layer width), `l1`, `epochs`,
`label_atoms` (LLM auto-interp of top atoms — no-op unless the server has
`OPENAI_API_KEY`), `compare_embedding` (adds the dropped-signal diff). Returns
`{"job_id": "interp-…"}`. SAE jobs are serialized behind the shared build-slot
semaphore, so launch them and poll rather than expecting parallelism. The result
is persisted automatically, so a later look reuses it via step 0.

### 3. Poll each job

```sh
curl -s localhost:8080/api/jobs/<job_id> | python3 -m json.tool
```
`sleep 15` between rounds (these jobs take minutes; longer with
`compare_embedding`). `state` goes `running` (with a live `stage` mirrored from
the engine) → `done` (analysis JSON under `.result`) or `failed` (`.error`).

### 4. Distill the result

The `.result` JSON has top-level `model_dir`, `dataset_id`, `hidden`,
`activation`, `config`, `depth_linear_probe`, `layers[]` (each with `capacity`,
`n_interpretable_concepts`, `probe`, `atoms_by_target_corr`, `segments`,
`dropped_vs_embedding`), `concept_vs_decodability`, and `embedding_diff`. Pull
the per-layer capacity + concept counts and, per model, its best-layer
utilization, total interpretable concepts, segments represented, and (if run)
dropped-signal count.

### 5. Present the comparison

Build a side-by-side table, one row per model, and interpret it against
MAE — not instead of it:

```
| model | predictor | best-layer util | Σ concepts | segments repr. | dropped | peak linear R² | MAE |
```

Read it: does the leaderboard leader also capture the most structure, or is it
winning while discarding target-relevant signal a rival keeps? Where in
depth do concepts form vs. where the target becomes decodable? Which rare
segments does each model represent? Flag any model whose capacity looks
degenerate (near-zero utilization, no interpretable concepts) — a likely
mis-load or a collapsed layer, not a finding.

## Delegate to the information-capture-analyst agent

The workflow above is for a quick, user-driven comparison of a handful of
models. For a campaign write-up — many models, `compare_embedding` diffs, a
ready-to-paste report section and a `experiments/PROJECT-FACTS.md` note — spawn the
**information-capture-analyst** agent with the model set and shared dataset id.
The experiment-runner spawns it automatically when a campaign trains MLP models.
