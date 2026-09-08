<p align="center"><img src="../assets/lattice-data.svg" width="620" alt="A straight row of data points bent into a lensed arc as it passes a mass."></p>

# Reference

The contracts a lensing instance is built out of: the on-disk dataset
artifact, the predictor plugin protocol, the two TOML files the server
manages, and the HTTP API. Everything here is domain-neutral — it holds
whatever `domain.toml` says the domain is.

Start at [README.md](../README.md) for what lensing is and how a session
runs; at [BOOTSTRAP.md](../BOOTSTRAP.md) for standing up a new instance;
at [DEPLOY.md](DEPLOY.md) for containerized and multi-host deployment.

**Contents** — [dataset artifact](#dataset-artifact-format) ·
[predictor contract](#predictor-contract-v2) ·
[registry.toml](#registry-registrytoml) ·
[models.toml](#model-definitions-modelstoml) ·
[HTTP API](#api)

---

## Dataset artifact format

A dataset is a directory `data/datasets/<dataset_id>/` containing raw
little-endian binaries plus a `manifest.json` describing them. Everything is
readable from any language (`numpy.fromfile`, `Float32Array`, …).

| file                 | type | shape                  | notes |
|----------------------|------|------------------------|-------|
| `features.f32`       | f32  | n_rows × n_cols, row-major | ALL rows (train and test) |
| `target.f32`         | f32  | n_rows                 | transformed target (see `manifest.target.transform`) |
| `row_ids.u64`        | u64  | n_rows                 | Qdrant point ids |
| `train_idx.u32`      | u32  | n_train                | sorted row indices |
| `test_idx.u32`       | u32  | n_test                 | sorted row indices |
| `pca_components.f32` | f32  | dims × 1536, row-major | principal components |
| `items.json`         | json | row_id → item          | display payload for drill-down |
| `manifest.json`      | json |                        | schema below |

`manifest.json` fields: `dataset_id`, `created_at`, `source{qdrant_url,
collection, filter}`, `n_rows`, `n_cols`, `columns[]` (per-column
`{name, kind: {type: pca|numeric|onehot, ...}}` in on-disk order), `pca{dims,
mean[1536], components_shape, explained_variance_ratio[]}`, `target{field,
transform: "log1p"|"none"}`, `split{test_ratio, seed, n_train, n_test}`,
`feature_config` (echo of build flags), and optionally `quality`,
`redundancy`, `currency` (reports of build-time processing; absent on
datasets built before each existed).

Notes:

- **Feature standardization is the predictor's job** (fit on train rows
  only). The artifact is a neutral raw feature matrix; tree models won't
  want scaling, neural nets will.
- **The target is already transformed** (`log1p` by default). Metrics and
  predictions must be reported in **target space**: invert with
  `expm1` before computing/writing them.
- The train/test split is reproducible from `(n_rows, seed, test_ratio)`:
  Fisher-Yates over `0..n_rows` driven by a SplitMix64 stream seeded with
  `seed` (`j = next() % (i+1)` with Lemire rejection), first
  `round(n·test_ratio)` shuffled indices are the test set, both lists then
  sorted ascending. See `crates/lensing-pipeline/src/shuffle.rs`.

## Predictor contract (v2)

A predictor is any executable registered in `registry.toml`. It implements
`train` (required) and `predict` (optional; without it, runs are train-only
and cannot be promoted to models).

### train

```sh
<command> train --dataset <dataset_dir> --output <run_dir> --hyperparams <hp.json path>
```

It must:

1. Stream JSON-lines to **stdout** (one object per line):
   - `{"event":"epoch","epoch":N,"total_epochs":T,"train_loss":x,"val_loss":y}`
     once per epoch (omit entirely for non-iterative models),
   - `{"event":"log","msg":"..."}` for anything worth showing in the run log,
   - `{"event":"checkpoint","epoch":N}` after each periodic checkpoint save
     (see below; only for predictors that implement `checkpoint_every`),
   - `{"event":"stopping"}` when it notices a stop request (optional but
     recommended; the UI shows "finishing up…"),
   - `{"event":"done"}` as the final line.
2. Write into `<run_dir>`:
   - `metrics.json`: `{"mae":..,"rmse":..,"r2":..,"mape":..,"medape":..,"n_test":N}`
     computed on the test split, **in target space** (`mape`/`medape` as
     fractions, 0.25 = 25%),
   - `predictions.json`: `[{"row_id":id,"actual":value,"predicted":value}, ...]`
     for every test row, target space,
   - **every file it needs to reload the model later** (checkpoint, scaler,
     …). Promotion copies all non-contract files from the run dir into the
     model dir verbatim (atomic-write temp files containing `.tmp`/`-tmp`
     are skipped),
   - optionally `viz.svg`, a self-contained architecture diagram written
     once at training start (declare `visualization = true` in the
     registry). It is served on `GET /api/runs/{id}/viz`, copied into the
     model dir at promotion and served there too.
3. Exit 0 on success. Anything else marks the run failed; stderr is captured
   and shown in the UI.

Hyperparameters arrive as a flat JSON object (file path in `--hyperparams`).
Each predictor's schema (types, defaults, ranges) lives in `registry.toml`
and drives the auto-generated form in the UI.

#### Stopping a run

`POST /api/runs/{id}/stop` drops an empty **`STOP`** file into the run dir
(and `{"force":true}` additionally SIGKILLs the process). Iterative
predictors that declare `supports_stop = true` in the registry must check
for `<run_dir>/STOP` **between epochs**; on seeing it, they stop training
and run their normal end-of-training path with the params as they are —
evaluate the test split, write `metrics.json` / `predictions.json` / the
final checkpoint, emit `done`, exit 0. The server records such a run as
`stopped` (vs `succeeded`); it is promotable like a succeeded run.

#### Periodic checkpoints (`checkpoint_every`)

Iterative predictors should accept a `checkpoint_every` hyperparameter
(int, 0 = disabled): every N epochs, save the full checkpoint
**atomically** (write to a temp name, then rename — a kill mid-write must
never corrupt the loadable file) and emit `{"event":"checkpoint","epoch":N}`.
After the first such event the server marks the run `has_checkpoint`, which
makes even a force-killed / interrupted run promotable from the last saved
params (without final test metrics).

### predict

```sh
<command> predict --model <model_dir> --input <input_dir> --output <out.json path>
```

- `<model_dir>` is a promoted model directory (see below): the predictor's
  own training outputs plus `hyperparams.json` (copy of the run's hp.json,
  e.g. to rebuild a net's architecture before loading weights) and
  `contract.json` (the frozen featurization contract; carries the target
  transform).
- `<input_dir>` is a **mini-artifact** written by the server per request.
  The features are already in the trained column order; predictors do NOT
  featurize:

  | file           | notes |
  |----------------|-------|
  | `features.f32` | f32, n × n_cols row-major, server-featurized |
  | `row_ids.u64`  | u64, echoes input ids |
  | `manifest.json`| trimmed: `n_rows`, `n_cols`, `columns[]`, `target{field,transform}` |
  | `items.json`   | row_id → raw payload echo (for payload-based predictors; feature-based ones ignore it) |

- It must write `<out.json>` as `[{"row_id":id,"predicted":value}, ...]` in
  **target space** (invert the target transform, exactly as at train time),
  stream optional `{"event":"log",...}` lines plus a final
  `{"event":"done"}`, and exit 0 on success.

### export (optional)

```sh
<command> export --model <model_dir> --output <out_dir>
```

Optional third subcommand (declare `export_args` in the registry; without it
the family simply can't be exported and `GET /api/models/{name}/export`
returns `422`). It converts the predictor's native trained model into a
portable **`model.onnx`** written into `<out_dir>`, with a uniform I/O
signature across every family:

- input `input`: `float32[N, n_cols]` — the **assembled feature vector** (the
  same matrix `predict` receives in `features.f32`),
- output `output`: `float32[N, 1]` — the target in **transformed** space (the
  inverse transform + non-negative clamp live in the export's `featurize.json`,
  applied by the consumer, so the graph stays a uniform "raw predictor").

Any preprocessing the native model applies to the feature vector (a
standardizer, an output clamp) must be **baked into the graph** so the input
stays the raw assembled vector regardless of family. Tree ensembles are
scale-invariant and embed their trees directly. Meta-predictors (blend) instead
write `members/<i>/model.onnx` (by recursively invoking each member's `export`)
plus a `combination.json`. The server wraps whatever the predictor writes with
the portable `featurize.json` spec, the PCA basis, an `input-schema.json` and a
README into the export `.tar.gz` — see the `model-export` skill and
`crates/lensing-onnx` (the dependency-free ONNX writer the Rust predictors use).

### Named models (`data/models/<name>/`)

A run of a predict-capable predictor can be **promoted** to a named model
(`POST /api/models`): `succeeded` and `stopped` runs always; `failed` /
`interrupted` runs only when a periodic checkpoint landed
(`has_checkpoint`) — the model is then built from the params as they were,
without final test metrics. Promotion snapshots everything inference needs,
so the model survives server restarts and dataset/run deletion:

| file | contents |
|------|----------|
| `record.json` | `{name, run_id, predictor, dataset_id, created_at, notes}` |
| `contract.json` | frozen featurization contract: columns, PCA (dims, mean), target transform, input_fields |
| `pca_components.f32` | copied from the training dataset |
| `hyperparams.json` | copy of the run's hp.json |
| *predictor files* | everything the predictor wrote into the run dir (checkpoint, `scaler.json`, `model.json`, `viz.svg`, …) |

At predict time the server featurizes raw inputs (item JSON with a 1536-dim
`embedding`, or Qdrant point ids) under the model's frozen contract: same
PCA projection, same one-hot vocabularies (unknown categories fall into the
trained `__other__` bucket, reported as warnings), same column order. The
training dataset build quantizes its PCA through f32 before projecting, so
inference features are bit-identical to training features.

### Quality filters

Dataset builds run data-quality rules over the corpus before the split /
PCA / vocabularies. Toggles + thresholds arrive in the build request
(`quality{...}`); the applied config and per-rule counts are recorded in the
manifest (`quality`). Built-ins: `nonpositive-price` (nonpositive target, default on),
`price-outlier` (MAD z-score on the log target per outlier group, default
off, threshold `price_outlier_mad_z`), `missing-fields` (default off).
(Rule keys are stable identifiers from the original domain; each domain
rebinds and relabels them in `domain.toml`.)
`POST /api/datasets/preflight` evaluates the rules without building and
returns per-rule counts + sample flagged rows.

### Currency handling

For multi-currency corpora (`[currency]` in domain.toml; the worked example's
corpus mixes ARS- and USD-denominated listings whose derived build collections
dropped `metadata.currency`), builds (and preflight / analyze / export) take a
`currency{...}` config: currency is **reconciled** by point id from a
companion collection (the worked example's `properties`, which still carries
`metadata.currency` + `metadata.createdAt`), then either

- `mode: "filter"` (default) — rows whose currency differs from `keep`
  (default `"USD"`) are excluded via the `foreign-currency` quality rule, or
- `mode: "convert"` — foreign values are rewritten into `keep` using the
  per-date ARS/USD rate (`rate_source: "blue"|"oficial"`, daily series from
  api.argentinadatos.com) keyed on the listing's `createdAt`, *before* the
  target quality rules run, or
- `mode: "off"` — currency is ignored.

Rows with no currency after reconciliation are kept and reported
(`n_missing`). The applied config + counts (+ applied rate range) land in the
manifest under `currency`; collection exports stamp the reconciled currency
(and converted values) into the exported payloads.

## Registry (`registry.toml`)

```toml
[[predictors]]
name = "burn-mlp"                # id used in run metadata
display_name = "MLP (burn)"
command = "target/release/predictor-burn-mlp"
args = ["train", "--dataset", "{dataset}", "--output", "{run_dir}", "--hyperparams", "{hyperparams}"]
# Optional; presence makes runs of this predictor promotable to models.
# predict_command defaults to `command` when omitted.
predict_args = ["predict", "--model", "{model}", "--input", "{input}", "--output", "{output}"]
# Capability flags, both default false:
supports_stop = true             # polls run_dir/STOP between epochs (graceful stop)
visualization = true             # writes viz.svg at training start

[[predictors.params]]
name = "epochs"
type = "int"                     # int | float | bool | ints | enum
default = 50
min = 1
max = 100000
```

`{dataset}`, `{run_dir}`, `{hyperparams}` (train) and `{model}`, `{input}`,
`{output}` (predict) are substituted by the server. Commands run with the
repository root as working directory.

## Model definitions (`models.toml`)

A **definition** is a named preset — predictor + concrete hyperparam values +
dataset tags + notes — stored as `[[definitions]]` entries in `models.toml`
at the repo root. The file is server-managed (rewritten atomically on every
mutation via the `/api/definitions` endpoints or the UI's Definitions pages)
and git-versionable: committing it is how definitions are exported/shared.
Don't hand-edit it while the server runs.

Definitions store hyperparams already merged over the predictor's schema
defaults, validated on create/update (unknown keys rejected). Launching
`POST /api/runs {definition, dataset_id}` uses the definition's predictor and
params (request `hyperparams` overlay for one-off tweaks), records
`from_definition` in the run meta, and auto-appends the dataset to the
definition's `dataset_tags`. Definitions and promoted models are separate
namespaces; both names match `^[a-z0-9][a-z0-9-]{0,63}$`.

The `model-definitions` skill (`.claude/skills/`, mirrored to `.gemini/` and
`.agents/`) drives define / rename / clone / tag / launch / export
conversationally against this API.

## API

Machine-readable spec in [`docs/openapi.yaml`](openapi.yaml), served
by the running server at `/api/openapi.yaml` with an interactive Swagger UI
at `/docs`. The spec ships with packaged instances, so every bootstrapped
instance gets the same browsable API docs out of the box.

```
GET  /api/predictors                predictor registry (params drive the run form)
GET|POST /api/datasets              list / build (async; quality{} toggles filters)
POST /api/datasets/preflight        quality-rule counts + samples, no build
GET  /api/datasets/{id}[/items]     manifest / display payloads
GET  /api/builds/{id}               build status
GET|POST /api/runs                  list / start {dataset_id, predictor, hyperparams}
GET  /api/runs/{id}[/predictions]   meta / test-split predictions
GET  /api/runs/{id}/events          SSE progress (live or replay)
POST /api/runs/{id}/stop            {force?: bool} graceful stop (STOP file) or kill
GET  /api/runs/{id}/viz             architecture SVG (404 if the predictor wrote none)
GET|POST /api/models                list / promote {name, run_id, notes?}
GET|DELETE /api/models/{name}       record + contract summary + hyperparams / remove
GET  /api/models/{name}/viz         architecture SVG copied at promotion (404 if absent)
GET  /api/models/{name}/contract    required input fields, dims, transform
POST /api/models/{name}/predict     {items?: [...], point_ids?: [...]} → predictions
POST /api/models/{name}/rename      {new_name} (moves the model dir)
GET|POST /api/definitions           list / create {name, predictor, hyperparams?, dataset_tags?, notes?}
GET|PATCH|DELETE /api/definitions/{name}  get / partial update / remove
POST /api/definitions/{name}/rename {new_name}
POST /api/definitions/{name}/clone  {new_name} (params + notes; tags reset)
```

`POST /api/runs` also accepts `{definition, dataset_id, hyperparams?}` to
launch from a definition (see above).

A predict item is the payload fields the model consumes (keyed by domain
field name, see `GET /api/domain`) plus the embedding:
`{"<categorical>": "...", "<numeric>": 2, "embedding": [<dim> floats],
"id": optional}`. The response carries target-space predictions plus
warnings for out-of-vocabulary categoricals.
